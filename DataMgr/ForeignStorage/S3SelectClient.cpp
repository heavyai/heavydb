/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/Object.h>
#include <aws/s3/model/SelectObjectContentRequest.h>

#include "DataMgr/ForeignStorage/AbstractFileStorageDataWrapper.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/S3SelectClient.h"
#include "DataMgr/ForeignStorage/S3Utils.h"
#include "Shared/file_type.h"

namespace foreign_storage {

S3SelectClient::S3SelectClient(
    const std::string& obj_key,
    const ForeignServer* foreign_server,
    std::shared_ptr<Aws::Auth::AWSCredentialsProvider> aws_credentials,
    const import_export::CopyParams& copy_params)
    : obj_key_(obj_key), copy_params_(copy_params) {
  CHECK(foreign_server);
  bucket_name_ =
      foreign_server->options.find(AbstractFileStorageDataWrapper::S3_BUCKET_KEY)->second;
  s3_client_ = create_s3_client(foreign_server, aws_credentials);
}

Aws::S3::Model::SelectObjectContentRequest S3SelectClient::createRequest(
    const std::string sql,
    const S3ScanRange range) const {
  // Create s3 select request
  Aws::S3::Model::SelectObjectContentRequest select_object_request;
  select_object_request.SetBucket(bucket_name_);
  select_object_request.SetKey(range.file_name);
  select_object_request.SetExpressionType(Aws::S3::Model::ExpressionType::SQL);
  select_object_request.SetExpression(sql);

  Aws::S3::Model::CSVInput csv_input;
  // The first full line within a range is treated as a header so only ignore header
  // starting at 0
  bool skip_header =
      (copy_params_.has_header != import_export::ImportHeaderRow::kNoHeader) &&
      !(range.byte_range != std::nullopt && range.byte_range.value().first > 0);

  csv_input.SetFileHeaderInfo(skip_header ? Aws::S3::Model::FileHeaderInfo::IGNORE
                                          : Aws::S3::Model::FileHeaderInfo::NONE);

  csv_input.SetRecordDelimiter(std::string({copy_params_.line_delim}));
  csv_input.SetFieldDelimiter(std::string({copy_params_.delimiter}));
  csv_input.SetQuoteCharacter(std::string({copy_params_.quote}));
  csv_input.SetQuoteEscapeCharacter(std::string({copy_params_.escape}));

  Aws::S3::Model::InputSerialization input_serialization;

  input_serialization.SetCSV(csv_input);
  select_object_request.SetInputSerialization(input_serialization);

  Aws::S3::Model::CSVOutput csv_output;
  Aws::S3::Model::OutputSerialization output_serialization;
  output_serialization.SetCSV(csv_output);
  select_object_request.SetOutputSerialization(output_serialization);
  // If range is set add scanrange to request
  if (range.byte_range != std::nullopt) {
    Aws::S3::Model::ScanRange scan_range;
    scan_range.SetStart(range.byte_range.value().first);
    scan_range.SetEnd(range.byte_range.value().second);
    select_object_request.SetScanRange(scan_range);
  }
  return select_object_request;
}

namespace {
std::string get_index_string(int index) {
  // Format is s._N, where is is column index starting at 1
  return "s._" + std::to_string(index + 1);
}

Aws::S3::Model::SelectObjectContentOutcome s3_select_object_content(
    const std::unique_ptr<Aws::S3::S3Client>& s3_client,
    Aws::S3::Model::SelectObjectContentRequest& request) {
  auto timer = DEBUG_TIMER(__func__);
  return s3_client->SelectObjectContent(request);
}
}  // namespace
std::string S3SelectClient::getSelectResult(
    Aws::S3::Model::SelectObjectContentRequest& request) const {
  Aws::S3::Model::SelectObjectContentHandler handler;
  std::string result;
  std::mutex callback_mutex;
  // Callback to save records into string
  handler.SetRecordsEventCallback([&](const Aws::S3::Model::RecordsEvent& records_event) {
    std::lock_guard<std::mutex> callback_lock(callback_mutex);
    auto records_vector = records_event.GetPayload();
    result += std::string(records_vector.begin(), records_vector.end());
  });
  request.SetEventStreamHandler(handler);

  auto outcome = s3_select_object_content(s3_client_, request);
  if (!outcome.IsSuccess()) {
    throw_file_access_error(get_file_path(bucket_name_, obj_key_),
                            get_error_message(outcome.GetError().GetExceptionName(),
                                              outcome.GetError().GetMessage()));
  }
  if (shared::is_compressed_file_extension(obj_key_)) {
    throw_s3_compressed_extension(get_file_path(bucket_name_, obj_key_),
                                  boost::filesystem::path(obj_key_).extension().string());
  }
  return result;
}

std::vector<std::string> S3SelectClient::getSelectResults(
    std::vector<Aws::S3::Model::SelectObjectContentRequest>& requests) const {
  std::vector<std::string> results(requests.size());
  std::vector<std::future<Aws::S3::Model::SelectObjectContentOutcome>> outcomes;
  outcomes.reserve(requests.size());
  std::mutex results_mutex;

  int index = 0;
  for (auto& request : requests) {
    // Callback to save records into string
    Aws::S3::Model::SelectObjectContentHandler handler;
    handler.SetRecordsEventCallback(
        [&, index](const Aws::S3::Model::RecordsEvent& recordsEvent) {
          std::lock_guard<std::mutex> results_lock(results_mutex);
          auto recordsVector = recordsEvent.GetPayload();
          results[index] += std::string(recordsVector.begin(), recordsVector.end());
        });
    index++;
    request.SetEventStreamHandler(handler);
    outcomes.emplace_back(std::async(std::launch::async, [&] {
      return s3_select_object_content(s3_client_, request);
    }));
  }

  for (auto& outcome_callable : outcomes) {
    auto outcome = outcome_callable.get();
    if (!outcome.IsSuccess()) {
      throw_file_access_error(get_file_path(bucket_name_, obj_key_),
                              get_error_message(outcome.GetError().GetExceptionName(),
                                                outcome.GetError().GetMessage()));
    }
    if (shared::is_compressed_file_extension(obj_key_)) {
      throw_s3_compressed_extension(
          get_file_path(bucket_name_, obj_key_),
          boost::filesystem::path(obj_key_).extension().string());
    }
  }
  return results;
}

std::vector<int> S3SelectClient::getNumRows(const std::vector<S3ScanRange> ranges) const {
  std::vector<Aws::S3::Model::SelectObjectContentRequest> requests;
  for (const auto& range : ranges) {
    requests.push_back(createRequest("select COUNT(*) from S3Object s", range));
  }
  auto str_results = getSelectResults(requests);
  std::vector<int> result;
  for (const auto& str : str_results) {
    result.push_back(atoi(str.c_str()));
  }
  return result;
}

// Get columns as comma seperated text
std::string S3SelectClient::getColumnsAsCsv(const std::vector<int>& column_ids,
                                            const S3ScanRange range) const {
  std::string query = "SELECT ";
  // Address columns by index
  for (auto iter = column_ids.begin(); iter != column_ids.end(); ++iter) {
    if (iter != column_ids.begin()) {
      query += ", ";
    }
    query += get_index_string(*iter);
  }
  query += " from S3Object s";
  auto request = createRequest(query, range);
  return getSelectResult(request);
}

// Get columns stats as comma seperated text
std::vector<std::string> S3SelectClient::getStatsAsCsv(
    const std::vector<std::pair<int, std::string>>& column_id_types,
    const std::vector<S3ScanRange> ranges) const {
  std::string query = "SELECT ";
  for (auto iter = column_id_types.begin(); iter != column_id_types.end(); ++iter) {
    if (iter != column_id_types.begin()) {
      query += ", ";
    }
    // Address columns by index
    std::string column = get_index_string((*iter).first);
    // Treat empty strings as null
    std::string nulled_column = "NULLIF(" + column + ",'')";
    // Cast to supplied type
    std::string cast_column = "CAST(" + nulled_column + " as " + (*iter).second + ")";

    // Min/Max
    query += "MIN(" + cast_column + "), ";
    query += "MAX(" + cast_column + "),";
    // Returns 1 if any entries are null
    query += "MAX(case " + column + " when '' then 1 else 0 end)";
  }
  query += " from S3Object s";
  std::vector<Aws::S3::Model::SelectObjectContentRequest> requests;
  for (const auto& range : ranges) {
    requests.push_back(createRequest(query, range));
  };
  return getSelectResults(requests);
}

std::vector<std::string> S3SelectClient::getFirstRowAsCsv(
    const std::vector<S3ScanRange> ranges) {
  std::vector<Aws::S3::Model::SelectObjectContentRequest> requests;
  for (const auto& range : ranges) {
    requests.push_back(createRequest("select * from S3Object s LIMIT 1;", range));
  }
  return getSelectResults(requests);
}

std::vector<S3FileInfo> S3SelectClient::getFileInfos(
    const shared::FilePathOptions& options) const {
  return list_files_s3(s3_client_, obj_key_, bucket_name_, options);
}
}  // namespace foreign_storage
