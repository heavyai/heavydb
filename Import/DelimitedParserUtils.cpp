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
 * @file DelimitedParserUtils.cpp
 * @author Mehmet Sariyuce <mehmet.sariyuce@omnisci.com>
 * @brief Implementation of delimited parser utils.
 */

#include "Import/DelimitedParserUtils.h"

#include <string_view>

#include "Shared/Logger.h"
#include "StringDictionary/StringDictionary.h"

namespace {
inline bool is_eol(const char& c, const Importer_NS::CopyParams& copy_params) {
  return c == copy_params.line_delim || c == '\n' || c == '\r';
}

inline void trim_space(const char*& field_begin, const char*& field_end) {
  while (field_begin < field_end && (*field_begin == ' ' || *field_begin == '\r')) {
    ++field_begin;
  }
  while (field_begin < field_end &&
         (*(field_end - 1) == ' ' || *(field_end - 1) == '\r')) {
    --field_end;
  }
}

inline void trim_quotes(const char*& field_begin,
                        const char*& field_end,
                        const Importer_NS::CopyParams& copy_params) {
  if (copy_params.quoted && field_end - field_begin > 0 &&
      *field_begin == copy_params.quote) {
    ++field_begin;
  }
  if (copy_params.quoted && field_end - field_begin > 0 &&
      *(field_end - 1) == copy_params.quote) {
    --field_end;
  }
}
}  // namespace

namespace Importer_NS {
namespace delimited_parser {
size_t find_beginning(const char* buffer,
                      size_t begin,
                      size_t end,
                      const Importer_NS::CopyParams& copy_params) {
  // @TODO(wei) line_delim is in quotes note supported
  if (begin == 0 || (begin > 0 && buffer[begin - 1] == copy_params.line_delim)) {
    return 0;
  }
  size_t i;
  const char* buf = buffer + begin;
  for (i = 0; i < end - begin; i++) {
    if (buf[i] == copy_params.line_delim) {
      return i + 1;
    }
  }
  return i;
}

size_t find_end(const char* buffer,
                size_t size,
                const Importer_NS::CopyParams& copy_params,
                unsigned int& num_rows_this_buffer) {
  size_t last_line_delim_pos = 0;
  if (copy_params.quoted) {
    const char* current = buffer;
    bool in_quote = false;

    while (current < buffer + size) {
      while (!in_quote && current < buffer + size) {
        // We are outside of quotes. We have to find the last possible line delimiter.
        if (*current == copy_params.line_delim) {
          last_line_delim_pos = current - buffer;
          ++num_rows_this_buffer;
        } else if (*current == copy_params.quote) {
          in_quote = true;
        }
        ++current;
      }

      while (in_quote && current < buffer + size) {
        // We are in a quoted field. We have to find the ending quote.
        if ((*current == copy_params.escape) && (current < buffer + size - 1) &&
            (*(current + 1) == copy_params.quote)) {
          ++current;
        } else if (*current == copy_params.quote) {
          in_quote = false;
        }
        ++current;
      }
    }
  } else {
    const char* current = buffer;
    while (current < buffer + size) {
      if (*current == copy_params.line_delim) {
        last_line_delim_pos = current - buffer;
        ++num_rows_this_buffer;
      }
      ++current;
    }
  }

  if (last_line_delim_pos <= 0) {
    size_t slen = size < 50 ? size : 50;
    std::string showMsgStr(buffer, buffer + slen);
    LOG(ERROR) << "No line delimiter in block. Block was of size " << size
               << " bytes, first few characters " << showMsgStr;
    return size;
  }

  return last_line_delim_pos + 1;
}

template <typename T>
const char* get_row(const char* buf,
                    const char* buf_end,
                    const char* entire_buf_end,
                    const Importer_NS::CopyParams& copy_params,
                    const bool* is_array,
                    std::vector<T>& row,
                    std::vector<std::unique_ptr<char[]>>& tmp_buffers,
                    bool& try_single_thread) {
  const char* field = buf;
  const char* p;
  bool in_quote = false;
  bool in_array = false;
  bool has_escape = false;
  bool strip_quotes = false;
  try_single_thread = false;
  for (p = buf; p < entire_buf_end; ++p) {
    if (*p == copy_params.escape && p < entire_buf_end - 1 &&
        *(p + 1) == copy_params.quote) {
      p++;
      has_escape = true;
    } else if (copy_params.quoted && *p == copy_params.quote) {
      in_quote = !in_quote;
      if (in_quote) {
        strip_quotes = true;
      }
    } else if (!in_quote && is_array != nullptr && *p == copy_params.array_begin &&
               is_array[row.size()]) {
      in_array = true;
      while (p < entire_buf_end - 1) {  // Array type will be parsed separately.
        ++p;
        if (*p == copy_params.array_end) {
          in_array = false;
          break;
        }
      }
    } else if (*p == copy_params.delimiter || is_eol(*p, copy_params)) {
      if (!in_quote) {
        if (!has_escape && !strip_quotes) {
          const char* field_end = p;
          trim_space(field, field_end);
          row.emplace_back(field, field_end - field);
        } else {
          tmp_buffers.emplace_back(std::make_unique<char[]>(p - field + 1));
          auto field_buf = tmp_buffers.back().get();
          int j = 0, i = 0;
          for (; i < p - field; i++, j++) {
            if (has_escape && field[i] == copy_params.escape &&
                field[i + 1] == copy_params.quote) {
              field_buf[j] = copy_params.quote;
              i++;
            } else {
              field_buf[j] = field[i];
            }
          }
          const char* field_begin = field_buf;
          const char* field_end = field_buf + j;
          trim_space(field_begin, field_end);
          trim_quotes(field_begin, field_end, copy_params);
          row.emplace_back(field_begin, field_end - field_begin);
        }
        field = p + 1;
        has_escape = false;
        strip_quotes = false;

        if (is_eol(*p, copy_params)) {
          // We are at the end of the row. Skip the line endings now.
          while (p + 1 < buf_end && is_eol(*(p + 1), copy_params)) {
            p++;
          }
          break;
        }
      }
    }
  }
  /*
  @TODO(wei) do error handling
  */
  if (in_quote) {
    LOG(ERROR) << "Unmatched quote.";
    try_single_thread = true;
  }
  if (in_array) {
    LOG(ERROR) << "Unmatched array.";
    try_single_thread = true;
  }
  return p;
}

template const char* get_row(const char* buf,
                             const char* buf_end,
                             const char* entire_buf_end,
                             const Importer_NS::CopyParams& copy_params,
                             const bool* is_array,
                             std::vector<std::string>& row,
                             std::vector<std::unique_ptr<char[]>>& tmp_buffers,
                             bool& try_single_thread);

template const char* get_row(const char* buf,
                             const char* buf_end,
                             const char* entire_buf_end,
                             const Importer_NS::CopyParams& copy_params,
                             const bool* is_array,
                             std::vector<std::string_view>& row,
                             std::vector<std::unique_ptr<char[]>>& tmp_buffers,
                             bool& try_single_thread);

void parse_string_array(const std::string& s,
                        const Importer_NS::CopyParams& copy_params,
                        std::vector<std::string>& string_vec) {
  if (s == copy_params.null_str || s == "NULL" || s.size() < 1 || s.empty()) {
    // TODO: should not convert NULL, empty arrays to {"NULL"},
    //       need to support NULL, empty properly
    string_vec.emplace_back("NULL");
    return;
  }
  if (s[0] != copy_params.array_begin || s[s.size() - 1] != copy_params.array_end) {
    throw std::runtime_error("Malformed Array :" + s);
  }

  std::string row(s.c_str() + 1, s.length() - 2);
  row.push_back('\n');
  bool try_single_thread = false;
  Importer_NS::CopyParams array_params = copy_params;
  array_params.delimiter = copy_params.array_delim;
  std::vector<std::unique_ptr<char[]>> tmp_buffers;
  get_row(row.c_str(),
          row.c_str() + row.length(),
          row.c_str() + row.length(),
          array_params,
          nullptr,
          string_vec,
          tmp_buffers,
          try_single_thread);

  for (size_t i = 0; i < string_vec.size(); ++i) {
    if (string_vec[i].empty()) {  // Disallow empty strings for now
      string_vec.erase(string_vec.begin() + i);
      --i;
    } else if (string_vec[i].size() > StringDictionary::MAX_STRLEN) {
      throw std::runtime_error("Array String too long : " + string_vec[i] + " max is " +
                               std::to_string(StringDictionary::MAX_STRLEN));
    }
  }
}

}  // namespace delimited_parser
}  // namespace Importer_NS
