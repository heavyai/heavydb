/*
 * Copyright 2019 OmniSci, Inc.
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

/*
 * @file DelimitedParserUtils.h
 * @author Mehmet Sariyuce <mehmet.sariyuce@omnisci.com>
 * @brief DelimitedParserUtils class for parsing delimited data
 */

#pragma once

#include <string>
#include <vector>

#include "Import/CopyParams.h"

namespace Importer_NS {
class DelimitedParserUtils {
 public:
  /**
   * @brief Finds the closest possible row beginning in the given buffer.
   *
   * @param buffer               Given buffer which has the rows in csv format. (NOT OWN)
   * @param begin                Start index of buffer to look for the beginning.
   * @param end                  End index of buffer to look for the beginning.
   * @param copy_params          Copy params for the table.
   *
   * @return The position of the closest possible row beginning to the start of the given
   * buffer.
   */
  static size_t find_beginning(const char* buffer,
                               size_t begin,
                               size_t end,
                               const CopyParams& copy_params);

  /**
   * @brief Finds the closest possible row ending to the end of the given buffer.
   *
   * @param buffer               Given buffer which has the rows in csv format. (NOT OWN)
   * @param size                 Size of the buffer.
   * @param copy_params          Copy params for the table.
   * @param num_rows_this_buffer Number of rows until the closest possible row ending.
   *
   * @return The position of the closest possible row ending to the end of the given
   * buffer.
   */
  static size_t find_end(const char* buffer,
                         size_t size,
                         const CopyParams& copy_params,
                         unsigned int& num_rows_this_buffer);

  /**
   * @brief Parses the first row in the given buffer and inserts fields into given vector.
   *
   * @param buf                  Given buffer which has the rows in csv format. (NOT OWN)
   * @param buf_end              End of the sliced buffer for the thread. (NOT OWN)
   * @param entire_buf_end       End of the entire buffer. (NOT OWN)
   * @param copy_params          Copy params for the table.
   * @param is_array             Array of bools which tells if a column is an array type.
   * @param row                  Given vector to be populated with parsed fields.
   * @param try_single_thread    In case of parse errors, this will tell if parsing
   * should continue with single thread.
   *
   * @return Pointer to the next row after the first row is parsed.
   */
  static const char* get_row(const char* buf,
                             const char* buf_end,
                             const char* entire_buf_end,
                             const Importer_NS::CopyParams& copy_params,
                             const bool* is_array,
                             std::vector<std::string>& row,
                             bool& try_single_thread);

  /**
   * @brief Parses given string array and inserts into given vector of strings.
   *
   * @param s                    Given string array
   * @param copy_params          Copy params for the table.
   * @param string_vec           Given vector to be populated with parsed fields.
   */
  static void parseStringArray(const std::string& s,
                               const Importer_NS::CopyParams& copy_params,
                               std::vector<std::string>& string_vec);
};
}  // namespace Importer_NS
