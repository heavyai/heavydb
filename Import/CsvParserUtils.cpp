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
 * @file CsvParserUtils.cpp
 * @author Mehmet Sariyuce <mehmet.sariyuce@omnisci.com>
 * @brief Implementation of CsvParserUtils class.
 */
#include "CsvParserUtils.h"

#include "Shared/Logger.h"
#include "StringDictionary/StringDictionary.h"

namespace {
static const bool is_eol(const char& p, const std::string& line_delims) {
  for (auto i : line_delims) {
    if (p == i) {
      return true;
    }
  }
  return false;
}
}  // namespace

namespace Importer_NS {
size_t CsvParserUtils::find_beginning(const char* buffer,
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

size_t CsvParserUtils::find_end(const char* buffer,
                                size_t size,
                                const Importer_NS::CopyParams& copy_params,
                                unsigned int& num_rows_this_buffer) {
  size_t last_line_delim_pos = 0;
  if (copy_params.quoted) {
    const char* current = buffer;
    last_line_delim_pos = 0;
    bool in_quote = false;

    while (current < buffer + size) {
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

      // We are outside of quotes. We have to find the last possible line delimiter.
      while (!in_quote && current < buffer + size) {
        if (*current == copy_params.line_delim) {
          last_line_delim_pos = current - buffer;
          ++num_rows_this_buffer;
        } else if (*current == copy_params.quote) {
          in_quote = true;
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

const char* CsvParserUtils::get_row(const char* buf,
                                    const char* buf_end,
                                    const char* entire_buf_end,
                                    const Importer_NS::CopyParams& copy_params,
                                    const bool* is_array,
                                    std::vector<std::string>& row,
                                    bool& try_single_thread) {
  const char* field = buf;
  const char* p;
  bool in_quote = false;
  bool in_array = false;
  bool has_escape = false;
  bool strip_quotes = false;
  try_single_thread = false;
  std::string line_endings({copy_params.line_delim, '\r', '\n'});
  for (p = buf; p < entire_buf_end; p++) {
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
    } else if (!in_quote && is_array != nullptr && *p == copy_params.array_end &&
               is_array[row.size()]) {
      in_array = false;
    } else if (*p == copy_params.delimiter || is_eol(*p, line_endings)) {
      if (!in_quote && !in_array) {
        if (!has_escape && !strip_quotes) {
          std::string s = trim_space(field, p - field);
          row.push_back(s);
        } else {
          auto field_buf = std::make_unique<char[]>(p - field + 1);
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
          std::string s = trim_space(field_buf.get(), j);
          if (copy_params.quoted && s.size() > 0 && s.front() == copy_params.quote) {
            s.erase(0, 1);
          }
          if (copy_params.quoted && s.size() > 0 && s.back() == copy_params.quote) {
            s.pop_back();
          }
          row.push_back(s);
        }
        field = p + 1;
        has_escape = false;
        strip_quotes = false;

        if (is_eol(*p, line_endings)) {
          // We are at the end of the row. Skip the line endings now.
          while (p + 1 < buf_end && is_eol(*(p + 1), line_endings)) {
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

void CsvParserUtils::parseStringArray(const std::string& s,
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
  size_t last = 1;
  for (size_t i = s.find(copy_params.array_delim, 1); i != std::string::npos;
       i = s.find(copy_params.array_delim, last)) {
    if (i > last) {  // if not empty string - disallow empty strings for now
      if (s.substr(last, i - last).length() > StringDictionary::MAX_STRLEN) {
        throw std::runtime_error("Array String too long : " +
                                 std::to_string(s.substr(last, i - last).length()) +
                                 " max is " +
                                 std::to_string(StringDictionary::MAX_STRLEN));
      }

      string_vec.push_back(s.substr(last, i - last));
    }
    last = i + 1;
  }
  if (s.size() - 1 > last) {  // if not empty string - disallow empty strings for now
    if (s.substr(last, s.size() - 1 - last).length() > StringDictionary::MAX_STRLEN) {
      throw std::runtime_error(
          "Array String too long : " +
          std::to_string(s.substr(last, s.size() - 1 - last).length()) + " max is " +
          std::to_string(StringDictionary::MAX_STRLEN));
    }

    string_vec.push_back(s.substr(last, s.size() - 1 - last));
  }
}

const std::string CsvParserUtils::trim_space(const char* field, const size_t len) {
  size_t i = 0;
  size_t j = len;
  while (i < j && (field[i] == ' ' || field[i] == '\r')) {
    i++;
  }
  while (i < j && (field[j - 1] == ' ' || field[j - 1] == '\r')) {
    j--;
  }
  return std::string(field + i, j - i);
}
}  // namespace Importer_NS