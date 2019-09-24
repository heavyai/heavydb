/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @file CsvParserUtils.h
 * @author Mehmet Sariyuce <mehmet.sariyuce@omnisci.com>
 * @brief CsvParserUtils class for parsing csv
 */
#ifndef _CSVPARSERUTILS_H
#define _CSVPARSERUTILS_H

#include "CopyParams.h"

namespace Importer_NS {
class CsvParserUtils {
 public:
  static size_t find_beginning(const char* buffer,
                               size_t begin,
                               size_t end,
                               const CopyParams& copy_params);

  static size_t find_end(const char* buffer,
                         size_t size,
                         const CopyParams& copy_params,
                         unsigned int& num_rows_this_buffer);

  static const char* get_row(const char* buf,
                             const char* buf_end,
                             const char* entire_buf_end,
                             const Importer_NS::CopyParams& copy_params,
                             const bool* is_array,
                             std::vector<std::string>& row,
                             bool& try_single_thread);

  static void parseStringArray(const std::string& s,
                               const Importer_NS::CopyParams& copy_params,
                               std::vector<std::string>& string_vec);

  static const std::string trim_space(const char* field, const size_t len);
};
}  // namespace Importer_NS

#endif  // _CSVPARSERUTILS_H
