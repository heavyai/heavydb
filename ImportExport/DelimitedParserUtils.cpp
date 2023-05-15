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

/*
 * @file DelimitedParserUtils.cpp
 * @brief Implementation of delimited parser utils.
 *
 */

#include "ImportExport/DelimitedParserUtils.h"

#include <string_view>

#include "ImportExport/CopyParams.h"
#include "Logger/Logger.h"
#include "StringDictionary/StringDictionary.h"

namespace {
inline bool is_eol(const char& c, const import_export::CopyParams& copy_params) {
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
                        const import_export::CopyParams& copy_params) {
  auto quote_begin = field_begin, quote_end = field_end;
  if (copy_params.quoted) {
    trim_space(quote_begin, quote_end);
  }
  if (copy_params.quoted && quote_end - quote_begin > 0) {
    if (*quote_begin == copy_params.quote && *(quote_end - 1) == copy_params.quote) {
      field_begin = ++quote_begin;
      field_end = (quote_begin == quote_end) ? quote_end : --quote_end;
    } else {
      throw import_export::delimited_parser::DelimitedParserException(
          "Unable to trim quotes.");
    }
  }
}
}  // namespace

namespace import_export {
namespace delimited_parser {
size_t find_beginning(const char* buffer,
                      size_t begin,
                      size_t end,
                      const import_export::CopyParams& copy_params) {
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
                const import_export::CopyParams& copy_params,
                unsigned int& num_rows_this_buffer,
                size_t buffer_first_row_index,
                bool& in_quote,
                size_t offset) {
  size_t last_line_delim_pos = 0;
  const char* current = buffer + offset;
  if (copy_params.quoted) {
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
    while (current < buffer + size) {
      if (*current == copy_params.line_delim) {
        last_line_delim_pos = current - buffer;
        ++num_rows_this_buffer;
      }
      ++current;
    }
  }

  if (last_line_delim_pos <= 0) {
    size_t excerpt_length = std::min<size_t>(50, size);
    std::string buffer_excerpt{buffer, buffer + excerpt_length};
    if (in_quote) {
      std::string quote(1, copy_params.quote);
      std::string error_message =
          "Unable to find a matching end quote for the quote character '" + quote +
          "' after reading " + std::to_string(size) +
          " characters. Please ensure that all data fields are correctly formatted "
          "or update the \"buffer_size\" option appropriately. Row number: " +
          std::to_string(buffer_first_row_index + 1) +
          ". First few characters in row: " + buffer_excerpt;
      throw InsufficientBufferSizeException{error_message};
    } else {
      std::string error_message =
          "Unable to find an end of line character after reading " +
          std::to_string(size) +
          " characters. Please ensure that the correct \"line_delimiter\" option is "
          "specified or update the \"buffer_size\" option appropriately. Row number: " +
          std::to_string(buffer_first_row_index + 1) +
          ". First few characters in row: " + buffer_excerpt;
      throw InsufficientBufferSizeException{error_message};
    }
  }

  return last_line_delim_pos + 1;
}

static size_t max_buffer_resize = max_import_buffer_resize_byte_size;

size_t get_max_buffer_resize() {
  return max_buffer_resize;
}

void set_max_buffer_resize(const size_t max_buffer_resize_param) {
  max_buffer_resize = max_buffer_resize_param;
}

size_t find_row_end_pos(size_t& alloc_size,
                        std::unique_ptr<char[]>& buffer,
                        size_t& buffer_size,
                        const CopyParams& copy_params,
                        const size_t buffer_first_row_index,
                        unsigned int& num_rows_in_buffer,
                        FILE* file,
                        foreign_storage::FileReader* file_reader) {
  bool found_end_pos{false};
  bool in_quote{false};
  size_t offset{0};
  size_t end_pos;
  CHECK(file != nullptr || file_reader != nullptr);
  const auto max_buffer_resize = get_max_buffer_resize();
  while (!found_end_pos) {
    try {
      end_pos = delimited_parser::find_end(buffer.get(),
                                           buffer_size,
                                           copy_params,
                                           num_rows_in_buffer,
                                           buffer_first_row_index,
                                           in_quote,
                                           offset);
      found_end_pos = true;
    } catch (InsufficientBufferSizeException& e) {
      if (alloc_size >= max_buffer_resize) {
        throw;
      }
      if (file == nullptr && file_reader->isScanFinished()) {
        throw;
      }
      offset = buffer_size;
      extend_buffer(
          buffer, buffer_size, alloc_size, file, file_reader, max_buffer_resize);
    }
  }
  return end_pos;
}

template <typename T>
const char* get_row(const char* buf,
                    const char* buf_end,
                    const char* entire_buf_end,
                    const import_export::CopyParams& copy_params,
                    const bool* is_array,
                    std::vector<T>& row,
                    std::vector<std::unique_ptr<char[]>>& tmp_buffers,
                    bool& try_single_thread,
                    bool filter_empty_lines) {
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
          if (copy_params.trim_spaces) {
            trim_space(field, field_end);
          }
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
          trim_quotes(field_begin, field_end, copy_params);
          if (copy_params.trim_spaces) {
            trim_space(field_begin, field_end);
          }
          row.emplace_back(field_begin, field_end - field_begin);
        }
        field = p + 1;
        has_escape = false;
        strip_quotes = false;

        if (is_eol(*p, copy_params)) {
          // We are at the end of the row. Skip the line endings now.
          if (filter_empty_lines) {
            while (p + 1 < buf_end && is_eol(*(p + 1), copy_params)) {
              p++;
            }
          } else {
            // skip DOS carriage return line feed only
            if (p + 1 < buf_end && *p == '\r' && *(p + 1) == '\n') {
              p++;
            }
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
                             const import_export::CopyParams& copy_params,
                             const bool* is_array,
                             std::vector<std::string>& row,
                             std::vector<std::unique_ptr<char[]>>& tmp_buffers,
                             bool& try_single_thread,
                             bool filter_empty_lines);

template const char* get_row(const char* buf,
                             const char* buf_end,
                             const char* entire_buf_end,
                             const import_export::CopyParams& copy_params,
                             const bool* is_array,
                             std::vector<std::string_view>& row,
                             std::vector<std::unique_ptr<char[]>>& tmp_buffers,
                             bool& try_single_thread,
                             bool filter_empty_lines);

void parse_string_array(const std::string& s,
                        const import_export::CopyParams& copy_params,
                        std::vector<std::string>& string_vec,
                        bool truncate_values) {
  if (s == copy_params.null_str || s == "NULL" || s.size() < 1 || s.empty()) {
    return;
  }
  if (s[0] != copy_params.array_begin || s[s.size() - 1] != copy_params.array_end) {
    throw std::runtime_error("Malformed Array :" + s);
  }

  std::string row(s.c_str() + 1, s.length() - 2);
  if (row.empty()) {  // allow empty arrays
    return;
  }
  row.push_back('\n');

  bool try_single_thread = false;
  import_export::CopyParams array_params = copy_params;
  array_params.delimiter = copy_params.array_delim;
  std::vector<std::unique_ptr<char[]>> tmp_buffers;
  get_row(row.c_str(),
          row.c_str() + row.length(),
          row.c_str() + row.length(),
          array_params,
          nullptr,
          string_vec,
          tmp_buffers,
          try_single_thread,
          true);

  for (size_t i = 0; i < string_vec.size(); ++i) {
    if (string_vec[i].size() > StringDictionary::MAX_STRLEN) {
      if (truncate_values) {
        string_vec[i] = string_vec[i].substr(0, StringDictionary::MAX_STRLEN);
      } else {
        throw std::runtime_error("Array String too long : " + string_vec[i] + " max is " +
                                 std::to_string(StringDictionary::MAX_STRLEN));
      }
    }
  }

  // use empty string to mark nulls
  for (auto& value : string_vec) {
    if (value == copy_params.null_str || value == "NULL" || value.empty()) {
      value.clear();
    }
  }
}

void extend_buffer(std::unique_ptr<char[]>& buffer,
                   size_t& buffer_size,
                   size_t& alloc_size,
                   FILE* file,
                   foreign_storage::FileReader* file_reader,
                   size_t max_buffer_resize) {
  auto old_buffer = std::move(buffer);
  alloc_size = std::min(max_buffer_resize, alloc_size * 2);
  LOG(INFO) << "Setting import thread buffer allocation size to " << alloc_size
            << " bytes";
  buffer = std::make_unique<char[]>(alloc_size);

  memcpy(buffer.get(), old_buffer.get(), buffer_size);
  size_t fread_size;
  CHECK(file != nullptr || file_reader != nullptr);
  if (file != nullptr) {
    fread_size = fread(buffer.get() + buffer_size, 1, alloc_size - buffer_size, file);
  } else {
    fread_size = file_reader->read(buffer.get() + buffer_size, alloc_size - buffer_size);
  }
  buffer_size += fread_size;
}
}  // namespace delimited_parser
}  // namespace import_export
