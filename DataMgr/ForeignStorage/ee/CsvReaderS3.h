/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include "Catalog/ForeignServer.h"
#include "DataMgr/ForeignStorage/CsvReader.h"
namespace foreign_storage {

// Single S3  file or directory with multiple files
class MultiS3Reader : public MultiFileReader {
 public:
  MultiS3Reader(const std::string& file_path,
                const import_export::CopyParams& copy_params,
                const ForeignServer* server_options);
};

class CsvReaderS3 : public CsvReader {
 public:
  CsvReaderS3(const std::string& obj_key,
              size_t file_size,
              const import_export::CopyParams& copy_params,
              const ForeignServer* server_options);
  size_t read(void* buffer, size_t max_size) override;
  size_t readRegion(void* buffer, size_t offset, size_t size) override {
    CHECK(isScanFinished());
    current_offset_ = offset;
    return read(buffer, size);
  }

  bool isScanFinished() override { return scan_finished_; }

  bool getSize(size_t& size) override {
    size = file_size_;
    return true;
  }

 private:
  void skipHeader();
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
