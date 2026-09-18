/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(HAVE_AWS_S3)
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/Object.h>

#include "Catalog/ForeignServer.h"
#include "DataMgr/ForeignStorage/AbstractFileStorageDataWrapper.h"
#include "DataMgr/ForeignStorage/FileReaderS3.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/S3Utils.h"
#include "Shared/JsonUtils.h"
#include "Shared/file_type.h"
#include "Shared/misc.h"

namespace foreign_storage {

FileReaderS3::FileReaderS3(const std::string& obj_key,
                           size_t file_size,
                           const import_export::CopyParams& copy_params,
                           const ForeignServer* foreign_server,
                           const UserMapping* user_mapping)
    : SingleFileReader(obj_key, copy_params)
    , file_size_(file_size)
    , scan_finished_(false)
    , obj_key_(obj_key)
    , copy_params_(copy_params)
    , current_offset_(0)
    , header_offset_(0) {
  CHECK(foreign_server);
  bucket_name_ =
      foreign_server->options.find(AbstractFileStorageDataWrapper::S3_BUCKET_KEY)->second;
  s3_client_ = create_s3_client(foreign_server, get_credentials(user_mapping));

  skipHeader();
  if (header_offset_ >= file_size_) {
    scan_finished_ = true;
  }
  file_size_ = file_size_ - header_offset_;
}

FileReaderS3::FileReaderS3(const std::string& obj_key,
                           const import_export::CopyParams& copy_params,
                           const ForeignServer* foreign_server,
                           const UserMapping* user_mapping,
                           const rapidjson::Value& value)
    : SingleFileReader(obj_key, copy_params)
    , scan_finished_(false)
    , obj_key_(obj_key)
    , copy_params_(copy_params)
    , current_offset_(0)
    , header_offset_(0) {
  CHECK(foreign_server);
  bucket_name_ =
      foreign_server->options.find(AbstractFileStorageDataWrapper::S3_BUCKET_KEY)->second;
  s3_client_ = create_s3_client(foreign_server, get_credentials(user_mapping));

  scan_finished_ = true;
  json_utils::get_value_from_object(value, header_offset_, "header_offset");
  json_utils::get_value_from_object(value, file_size_, "file_size");
}

void FileReaderS3::serialize(rapidjson::Value& value,
                             rapidjson::Document::AllocatorType& allocator) const {
  CHECK(scan_finished_);
  json_utils::add_value_to_object(value, header_offset_, "header_offset", allocator);
  json_utils::add_value_to_object(value, file_size_, "file_size", allocator);
};

namespace {
void check_s3_file_type(const std::string& mime_type,
                        const std::string& bucket_name,
                        const std::string& obj_key) {
  if (shared::is_compressed_mime_type(mime_type)) {
    throw_s3_compressed_mime_type(get_file_path(bucket_name, obj_key), mime_type);
  } else if (shared::is_compressed_file_extension(obj_key)) {
    throw_s3_compressed_extension(get_file_path(bucket_name, obj_key),
                                  boost::filesystem::path(obj_key).extension().string());
  }
}

Aws::S3::Model::GetObjectOutcome s3_get_object(
    const std::unique_ptr<Aws::S3::S3Client>& s3_client,
    const Aws::S3::Model::GetObjectRequest& object_request) {
  auto timer = DEBUG_TIMER(__func__);
  return s3_client->GetObject(object_request);
}
}  // namespace

size_t FileReaderS3::read(void* buffer, size_t max_size) {
  auto timer = DEBUG_TIMER(__func__);
  size_t byte_start = header_offset_ + current_offset_;
  size_t byte_end = byte_start + max_size;
  auto object_request = create_request(bucket_name_, obj_key_, byte_start, byte_end);
  auto get_object_outcome = s3_get_object(s3_client_, object_request);

  if (!get_object_outcome.IsSuccess()) {
    throw_file_access_error(
        get_file_path(bucket_name_, obj_key_),
        get_error_message(get_object_outcome.GetError().GetExceptionName(),
                          get_object_outcome.GetError().GetMessage()));
  }

  const auto object_mime_type = get_object_outcome.GetResult().GetContentType();
  check_s3_file_type(object_mime_type, bucket_name_, obj_key_);

  get_object_outcome.GetResult().GetBody().read(static_cast<char*>(buffer), max_size);

  size_t read_bytes = get_object_outcome.GetResult().GetBody().gcount();
  current_offset_ += read_bytes;
  if (current_offset_ + header_offset_ >= file_size_) {
    scan_finished_ = true;
  }
  return read_bytes;
}

void FileReaderS3::skipHeader() {
  if (copy_params_.has_header != import_export::ImportHeaderRow::kNoHeader) {
    header_offset_ = getFirstLine().length();
  }
}

std::string FileReaderS3::getFirstLine() const {
  auto timer = DEBUG_TIMER(__func__);
  size_t header_size = DEFAULT_HEADER_READ_SIZE;
  bool header_found = false;
  std::unique_ptr<char[]> header_buff;
  while (!header_found) {
    auto object_request = create_request(bucket_name_, obj_key_, 0, header_size);
    header_buff = std::make_unique<char[]>(header_size);
    auto get_object_outcome = s3_get_object(s3_client_, object_request);
    if (!get_object_outcome.IsSuccess()) {
      throw_file_access_error(
          get_file_path(bucket_name_, obj_key_),
          get_error_message(get_object_outcome.GetError().GetExceptionName(),
                            get_object_outcome.GetError().GetMessage()));
    }

    const auto object_mime_type = get_object_outcome.GetResult().GetContentType();
    check_s3_file_type(object_mime_type, bucket_name_, obj_key_);

    get_object_outcome.GetResult().GetBody().getline(
        (header_buff.get()), header_size, copy_params_.line_delim);
    if (get_object_outcome.GetResult().GetBody().fail()) {
      // We didnt get a full line
      if (header_size == file_size_) {
        // File only contains one header line
        break;
      }
      header_size *= 2;
      if (header_size > file_size_) {
        header_size = file_size_;
      }
    } else {
      header_size = get_object_outcome.GetResult().GetBody().gcount();
      header_found = true;
    }
  }
  return std::string{header_buff.get(), header_size};
}

void FileReaderS3::increaseFileSize(size_t new_size) {
  CHECK(scan_finished_);
  CHECK_GT(new_size, file_size_);
  current_offset_ = file_size_;
  file_size_ = new_size;
  scan_finished_ = false;
}

MultiS3Reader::MultiS3Reader(const std::string& prefix_name,
                             const import_export::CopyParams& copy_params,
                             const ForeignTable* foreign_table,
                             const UserMapping* user_mapping)
    : MultiFileReader(prefix_name, copy_params) {
  CHECK(foreign_table);
  auto foreign_server = foreign_table->foreign_server;
  CHECK(foreign_server);
  bucket_name_ =
      foreign_server->options.find(AbstractFileStorageDataWrapper::S3_BUCKET_KEY)->second;
  s3_client_ = create_s3_client(foreign_server, get_credentials(user_mapping));

  auto file_infos =
      list_files_s3(s3_client_,
                    prefix_name,
                    bucket_name_,
                    AbstractFileStorageDataWrapper::getFilePathOptions(foreign_table));
  for (const auto& file_info : file_infos) {
    files_.emplace_back(std::make_unique<FileReaderS3>(
        file_info.first, file_info.second, copy_params, foreign_server, user_mapping));
    file_locations_.push_back(file_info.first);
    file_sizes_.push_back(file_info.second);
  }
}

std::string FileReaderS3::getCurrentFilePath() const {
  return bucket_name_ + "/" + obj_key_;
}

MultiS3Reader::MultiS3Reader(const std::string& file_path,
                             const import_export::CopyParams& copy_params,
                             const ForeignServer* foreign_server,
                             const UserMapping* user_mapping,
                             const rapidjson::Value& value)
    : MultiFileReader(file_path, copy_params, value) {
  CHECK(foreign_server);
  bucket_name_ =
      foreign_server->options.find(AbstractFileStorageDataWrapper::S3_BUCKET_KEY)->second;
  s3_client_ = create_s3_client(foreign_server, get_credentials(user_mapping));

  // reconstruct files from metadata
  CHECK(value.HasMember("files_metadata"));
  for (size_t index = 0; index < file_locations_.size(); index++) {
    files_.emplace_back(
        std::make_unique<FileReaderS3>(file_locations_[index],
                                       copy_params,
                                       foreign_server,
                                       user_mapping,
                                       value["files_metadata"].GetArray()[index]));
  }
  json_utils::get_value_from_object(value, file_sizes_, "file_sizes");
}

void MultiS3Reader::serialize(rapidjson::Value& value,
                              rapidjson::Document::AllocatorType& allocator) const {
  json_utils::add_value_to_object(value, file_sizes_, "file_sizes", allocator);
  MultiFileReader::serialize(value, allocator);
};

void MultiS3Reader::checkForMoreRows(size_t file_offset,
                                     const shared::FilePathOptions& options,
                                     const ForeignServer* foreign_server,
                                     const UserMapping* user_mapping) {
  CHECK(isScanFinished());
  CHECK(file_offset == current_offset_);
  CHECK(foreign_server != nullptr);

  // Look for new files
  auto file_infos = list_files_s3(s3_client_, file_path_, bucket_name_, options);
  if (!files_.empty()) {
    CHECK_EQ(files_.size(), file_locations_.size());
    CHECK_EQ(files_.size(), file_sizes_.size());

    std::set<std::string> new_file_paths;
    for (const auto& file_info : file_infos) {
      new_file_paths.emplace(file_info.first);
    }

    // Ensure no files are removed when appending. Note that rolled off files are already
    // removed from the file_locations_ vector as this point.
    for (const auto& file_path : file_locations_) {
      if (!shared::contains(new_file_paths, file_path)) {
        throw_removed_file_error(file_path);
      }
    }

    // Check for new rows in the last file
    auto last_file_index = current_index_ - 1;
    CHECK_LT(last_file_index, file_infos.size());
    CHECK_LT(last_file_index, file_sizes_.size());
    if (file_infos[last_file_index].second < file_sizes_[last_file_index]) {
      throw std::runtime_error{
          "Refresh of foreign table created with APPEND update mode failed as remote "
          "file reduced in size: \"" +
          file_locations_[last_file_index] + "\"."};
    } else if (file_infos[last_file_index].second > file_sizes_[last_file_index]) {
      // Go back to the last file, if more rows are found.
      FileReaderS3* s3_reader =
          dynamic_cast<FileReaderS3*>(files_[last_file_index].get());
      CHECK(s3_reader != nullptr);
      s3_reader->increaseFileSize(file_infos[last_file_index].second);
      file_sizes_[last_file_index] = file_infos[last_file_index].second;
      current_index_ = last_file_index;
      cumulative_sizes_.pop_back();
    }
  }

  for (const auto& file_info : file_infos) {
    if (std::find(file_locations_.begin(), file_locations_.end(), file_info.first) ==
        file_locations_.end()) {
      files_.emplace_back(std::make_unique<FileReaderS3>(
          file_info.first, file_info.second, copy_params_, foreign_server, user_mapping));
      file_locations_.push_back(file_info.first);
      file_sizes_.push_back(file_info.second);
    }
  }
}

std::vector<std::string> MultiS3Reader::getAllFilePaths(
    const shared::FilePathOptions& file_path_options) const {
  auto file_infos =
      list_files_s3(s3_client_, file_path_, bucket_name_, file_path_options);
  std::vector<std::string> all_file_paths;
  for (const auto& file_info : file_infos) {
    all_file_paths.emplace_back(file_info.first);
  }
  return all_file_paths;
}

std::set<std::string> MultiS3Reader::checkForRolledOffFiles(
    const shared::FilePathOptions& file_path_options) {
  std::set<std::string> full_file_paths;
  auto rolled_off_files = MultiFileReader::checkForRolledOffFiles(file_path_options);
  if (!rolled_off_files.empty()) {
    file_sizes_.erase(file_sizes_.begin(), file_sizes_.begin() + rolled_off_files.size());
    // Prepend the S3 bucket name in order to get the full path.
    for (const auto& file_path : rolled_off_files) {
      full_file_paths.emplace(bucket_name_ + "/" + file_path);
    }
  }
  return full_file_paths;
}

}  // namespace foreign_storage
#endif  //  defined(HAVE_AWS_S3)
