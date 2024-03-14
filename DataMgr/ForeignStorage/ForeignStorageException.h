/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include <stdexcept>
#include <string>

namespace foreign_storage {

class ForeignStorageException : public std::runtime_error {
 public:
  ForeignStorageException(const std::string& error_message)
      : std::runtime_error(error_message) {}
};

class MetadataScanInfeasibleFragmentSizeException : public ForeignStorageException {
 public:
  MetadataScanInfeasibleFragmentSizeException(const std::string& error_message)
      : ForeignStorageException(error_message), min_feasible_fragment_size_(-1) {}

  int32_t
      min_feasible_fragment_size_;  // may be set to indicate what the minimum feasible
                                    // fragment size for metadata scan should be
};

struct InvalidIntOptionException : public ForeignStorageException {
  InvalidIntOptionException(const std::string& table,
                            const std::string& option,
                            const std::string& value)
      : ForeignStorageException("Foreign table '" + table + "' with " + option + "='" +
                                value + "': " + option +
                                " must be an integer value greater than zero") {}
};

struct ColumnTypeMismatchException : public ForeignStorageException {
  ColumnTypeMismatchException(std::string msg) : ForeignStorageException(msg) {}
};

struct InvalidOptionException : public ForeignStorageException {
  InvalidOptionException(std::string msg) : ForeignStorageException(msg) {}
};

inline void throw_unexpected_number_of_items(const size_t& num_expected,
                                             const size_t& num_loaded,
                                             const std::string& item_type) {
  throw ForeignStorageException(
      "Unexpected number of " + item_type +
      " while loading from foreign data source: expected " +
      std::to_string(num_expected) + " , obtained " + std::to_string(num_loaded) + " " +
      item_type +
      ". Please use the \"REFRESH FOREIGN TABLES\" command on the foreign table "
      "if data source has been updated.");
}

class RequestedFragmentIdOutOfBoundsException : public ForeignStorageException {
 public:
  RequestedFragmentIdOutOfBoundsException(const std::string& error_message)
      : ForeignStorageException(error_message) {}
};

inline void throw_removed_row_in_result_set_error(const std::string& select_statement) {
  throw ForeignStorageException{
      "Refresh of foreign table created with \"APPEND\" update type failed as result set "
      "of select statement "
      "reduced in size: \"" +
      select_statement + "\""};
}

inline void throw_removed_row_in_file_error(const std::string& file_path) {
  throw ForeignStorageException{
      "Refresh of foreign table created with \"APPEND\" update type failed as file "
      "reduced in size: \"" +
      file_path + "\""};
}

inline void throw_removed_file_error(const std::string& file_path) {
  throw ForeignStorageException{
      "Refresh of foreign table created with \"APPEND\" update type failed as "
      "file \"" +
      file_path + "\" was removed."};
}

inline void throw_number_of_columns_mismatch_error(size_t num_table_cols,
                                                   size_t num_file_cols,
                                                   const std::string& file_path) {
  throw ForeignStorageException{"Mismatched number of logical columns: (expected " +
                                std::to_string(num_table_cols) + " columns, has " +
                                std::to_string(num_file_cols) + "): in file '" +
                                file_path + "'"};
}

inline void throw_file_access_error(const std::string& file_path,
                                    const std::string& message) {
  std::string error_message{"Unable to access file \"" + file_path + "\". " + message};
  throw ForeignStorageException{error_message};
}

inline void throw_file_not_found_error(const std::string& file_path) {
  throw ForeignStorageException{"File or directory \"" + file_path +
                                "\" does not exist."};
}

inline void throw_s3_compressed_mime_type(const std::string& file_path,
                                          const std::string& mime_type) {
  throw ForeignStorageException{
      "File \"" + file_path + "\" has mime type \"" + mime_type +
      "\", compressed file formats are not supported by S3 Foreign Tables."};
}

inline void throw_s3_compressed_extension(const std::string& file_path,
                                          const std::string& ext_type) {
  throw ForeignStorageException{
      "File \"" + file_path + "\" has extension type \"" + ext_type +
      "\", compressed file formats are not supported by S3 Foreign Tables."};
}

}  // namespace foreign_storage
