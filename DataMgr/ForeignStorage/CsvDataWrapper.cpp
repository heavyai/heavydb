/*
 * Copyright 2021 OmniSci, Inc.
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

#include "CsvDataWrapper.h"

#include "DataMgr/ForeignStorage/FileRegions.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"

namespace foreign_storage {
CsvDataWrapper::CsvDataWrapper() : AbstractTextFileDataWrapper() {}

CsvDataWrapper::CsvDataWrapper(const int db_id, const ForeignTable* foreign_table)
    : AbstractTextFileDataWrapper(db_id, foreign_table) {}

void CsvDataWrapper::validateTableOptions(const ForeignTable* foreign_table) const {
  AbstractTextFileDataWrapper::validateTableOptions(foreign_table);
  csv_file_buffer_parser_.validateAndGetCopyParams(foreign_table);
  validateAndGetIsS3Select(foreign_table);
}

const std::set<std::string_view>& CsvDataWrapper::getSupportedTableOptions() const {
  static const auto supported_table_options = getAllCsvTableOptions();
  return supported_table_options;
}

std::set<std::string_view> CsvDataWrapper::getAllCsvTableOptions() const {
  std::set<std::string_view> supported_table_options(
      AbstractFileStorageDataWrapper::getSupportedTableOptions().begin(),
      AbstractFileStorageDataWrapper::getSupportedTableOptions().end());
  supported_table_options.insert(csv_table_options_.begin(), csv_table_options_.end());
  return supported_table_options;
}

const TextFileBufferParser& CsvDataWrapper::getFileBufferParser() const {
  return csv_file_buffer_parser_;
}

bool CsvDataWrapper::validateAndGetIsS3Select(const ForeignTable* foreign_table) {
  static constexpr const char* S3_DIRECT = "S3_DIRECT";
  static constexpr const char* S3_SELECT = "S3_SELECT";
  static constexpr const char* S3_ACCESS_TYPE = "S3_ACCESS_TYPE";
  auto access_type = foreign_table->options.find(S3_ACCESS_TYPE);

  if (access_type != foreign_table->options.end()) {
    auto& server_options = foreign_table->foreign_server->options;
    if (server_options.find(AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY)->second !=
        AbstractFileStorageDataWrapper::S3_STORAGE_TYPE) {
      throw std::runtime_error{
          "The \"" + std::string{S3_ACCESS_TYPE} +
          "\" option is only valid for foreign tables using servers with \"" +
          AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY + "\" option value of \"" +
          AbstractFileStorageDataWrapper::S3_STORAGE_TYPE + "\"."};
    }
    if (access_type->second != S3_DIRECT && access_type->second != S3_SELECT) {
      throw std::runtime_error{
          "Invalid value provided for the \"" + std::string{S3_ACCESS_TYPE} +
          "\" option. Value must be one of the following: " + S3_DIRECT + ", " +
          S3_SELECT + "."};
    }
    return (access_type->second == S3_SELECT);
  } else {
    return false;
  }
}

const std::set<std::string_view> CsvDataWrapper::csv_table_options_{"ARRAY_DELIMITER",
                                                                    "ARRAY_MARKER",
                                                                    "BUFFER_SIZE",
                                                                    "DELIMITER",
                                                                    "ESCAPE",
                                                                    "HEADER",
                                                                    "LINE_DELIMITER",
                                                                    "LONLAT",
                                                                    "NULLS",
                                                                    "QUOTE",
                                                                    "QUOTED",
                                                                    "S3_ACCESS_TYPE"};

const CsvFileBufferParser CsvDataWrapper::csv_file_buffer_parser_{};
}  // namespace foreign_storage
