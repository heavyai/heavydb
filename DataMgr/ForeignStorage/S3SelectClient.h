/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/SelectObjectContentRequest.h>
#include "Catalog/ForeignServer.h"
#include "ImportExport/CopyParams.h"
namespace foreign_storage {
using S3FileInfo = std::pair<std::string, size_t>;

struct S3ScanRange {
  S3ScanRange(std::string file_name) : file_name(file_name), byte_range(std::nullopt){};

  S3ScanRange(std::string file_name, size_t start, size_t end)
      : file_name(file_name), byte_range({start, end}){};
  // Needed to maintain order in set
  bool operator<(const S3ScanRange& other) const {
    return file_name < other.file_name ||
           ((file_name == other.file_name) && byte_range < other.byte_range);
  };

  std::string file_name;
  std::optional<std::pair<size_t, size_t>> byte_range;
};

// Wrapper for S3 CSV select, returning sets of columns or stats in text form
class S3SelectClient {
 public:
  S3SelectClient(const std::string& obj_key,
                 const ForeignServer* server_options,
                 std::shared_ptr<Aws::Auth::AWSCredentialsProvider> aws_credentials,
                 const import_export::CopyParams& copy_params);

  // Get num rows in scan range
  std::vector<int> getNumRows(const std::vector<S3ScanRange> ranges) const;

  // Get columns as comma seperated text
  std::string getColumnsAsCsv(const std::vector<int>& column_ids,
                              S3ScanRange range) const;

  // Get columns stats as comma seperated text
  std::vector<std::string> getStatsAsCsv(
      const std::vector<std::pair<int, std::string>>& column_id_types,
      const std::vector<S3ScanRange> ranges) const;

  // Get total size of file
  std::vector<S3FileInfo> getFileInfos(const shared::FilePathOptions& options) const;

  std::vector<std::string> getFirstRowAsCsv(const std::vector<S3ScanRange> ranges);

 private:
  Aws::S3::Model::SelectObjectContentRequest createRequest(const std::string sql,
                                                           const S3ScanRange range) const;

  std::string getSelectResult(Aws::S3::Model::SelectObjectContentRequest& request) const;
  std::vector<std::string> getSelectResults(
      std::vector<Aws::S3::Model::SelectObjectContentRequest>& requests) const;

  std::unique_ptr<Aws::S3::S3Client> s3_client_;
  std::string obj_key_;
  std::string bucket_name_;
  import_export::CopyParams copy_params_;
};
}  // namespace foreign_storage
