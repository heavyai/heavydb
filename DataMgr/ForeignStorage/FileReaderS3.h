/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(HAVE_AWS_S3)
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include "Catalog/ForeignServer.h"
#include "Catalog/UserMapping.h"
#include "DataMgr/ForeignStorage/FileReader.h"
namespace foreign_storage {

// Single S3  file or directory with multiple files
class MultiS3Reader : public MultiFileReader {
 public:
  MultiS3Reader(const std::string& file_path,
                const import_export::CopyParams& copy_params,
                const ForeignTable* foreign_table,
                const UserMapping* user_mapping);

  MultiS3Reader(const std::string& file_path,
                const import_export::CopyParams& copy_params,
                const ForeignServer* foreign_server,
                const UserMapping* user_mapping,
                const rapidjson::Value& value);
  void checkForMoreRows(size_t file_offset,
                        const shared::FilePathOptions& options,
                        const ForeignServer* foreign_server,
                        const UserMapping* user_mapping) override;
  void serialize(rapidjson::Value& value,
                 rapidjson::Document::AllocatorType& allocator) const override;

  std::set<std::string> checkForRolledOffFiles(
      const shared::FilePathOptions& file_path_options) override;

 private:
  std::vector<std::string> getAllFilePaths(
      const shared::FilePathOptions& file_path_options) const override;

  // We've reached the end of the file
  std::unique_ptr<Aws::S3::S3Client> s3_client_;
  std::vector<size_t> file_sizes_;
  std::string bucket_name_;
};

class FileReaderS3 : public SingleFileReader {
 public:
  FileReaderS3(const std::string& obj_key,
               size_t file_size,
               const import_export::CopyParams& copy_params,
               const ForeignServer* foreign_server,
               const UserMapping* user_mapping);

  FileReaderS3(const std::string& obj_key,
               const import_export::CopyParams& copy_params,
               const ForeignServer* foreign_server,
               const UserMapping* user_mapping,
               const rapidjson::Value& value);

  size_t read(void* buffer, size_t max_size) override;
  size_t readRegion(void* buffer, size_t offset, size_t size) override {
    CHECK(isScanFinished());
    current_offset_ = offset;
    return read(buffer, size);
  }

  bool isScanFinished() const override { return scan_finished_; }

  size_t getRemainingSize() override { return file_size_ - current_offset_; }

  bool isRemainingSizeKnown() override { return true; };

  void serialize(rapidjson::Value& value,
                 rapidjson::Document::AllocatorType& allocator) const override;

  std::string getCurrentFilePath() const override;

  // Increase file size and continue metadata scan
  void increaseFileSize(size_t new_size);

 private:
  void skipHeader() override;
  std::string getFirstLine() const override;

  size_t file_size_;
  // We've reached the end of the file
  bool scan_finished_;
  std::unique_ptr<Aws::S3::S3Client> s3_client_;

  std::string obj_key_;
  std::string bucket_name_;
  import_export::CopyParams copy_params_;

  size_t current_offset_;
  size_t header_offset_;
};

}  // namespace foreign_storage
#endif  //  defined(HAVE_AWS_S3)
