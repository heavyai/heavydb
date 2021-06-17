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

#include <stdexcept>
#include <string>

namespace foreign_storage {

class ForeignStorageException : public std::runtime_error {
 public:
  ForeignStorageException(const std::string& error_message)
      : std::runtime_error(error_message) {}
};

inline void throw_removed_row_error(const std::string& file_path) {
  throw ForeignStorageException{
      "Refresh of foreign table created with \"APPEND\" update type failed as file "
      "reduced in size: " +
      file_path};
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
}  // namespace foreign_storage
