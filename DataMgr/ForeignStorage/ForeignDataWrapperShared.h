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
inline void throw_removed_row_error(const std::string& file_path) {
  throw std::runtime_error{
      "Refresh of foreign table created with \"APPEND\" update type failed as file "
      "reduced in size: " +
      file_path};
}

inline void throw_removed_file_error(const std::string& file_path) {
  throw std::runtime_error{
      "Refresh of foreign table created with \"APPEND\" update type failed as "
      "file \"" +
      file_path + "\" was removed."};
}
inline void throw_number_of_columns_mismatch_error(size_t num_table_cols,
                                                   size_t num_file_cols,
                                                   const std::string& file_path) {
  throw std::runtime_error{"Mismatched number of logical columns: (expected " +
                           std::to_string(num_table_cols) + " columns, has " +
                           std::to_string(num_file_cols) + "): in file '" + file_path +
                           "'"};
}
}  // namespace foreign_storage
