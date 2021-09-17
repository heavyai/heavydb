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

#include "DataMgr/ForeignStorage/RegexFileBufferParser.h"

#include <boost/regex.hpp>

#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "Shared/StringTransform.h"

namespace foreign_storage {
namespace {
using import_export::delimited_parser::InsufficientBufferSizeException;

size_t find_last_end_of_line(const char* buffer,
                             size_t buffer_size,
                             size_t start,
                             size_t end,
                             char line_delim) {
  int64_t i = end;
  while (i >= static_cast<int64_t>(start)) {
    if (buffer[i] == line_delim) {
      return i;
    } else {
      i--;
    }
  }
  throw InsufficientBufferSizeException{
      "Unable to find an end of line character after reading " +
      std::to_string(buffer_size) + " characters."};
}

bool line_starts_with_regex(const char* buffer,
                            size_t start,
                            size_t end,
                            const boost::regex& line_start_regex) {
  return boost::regex_search(std::string{buffer + start, end - start + 1},
                             line_start_regex,
                             boost::regex_constants::match_continuous);
}

std::optional<std::string> get_line_start_regex(const ForeignTable* foreign_table) {
  if (foreign_table) {
    auto it = foreign_table->options.find(RegexFileBufferParser::LINE_START_REGEX_KEY);
    if (it != foreign_table->options.end()) {
      return it->second;
    }
  }
  return {};
}

std::string get_line_regex(const ForeignTable* foreign_table) {
  if (foreign_table) {
    auto it = foreign_table->options.find(RegexFileBufferParser::LINE_REGEX_KEY);
    CHECK(it != foreign_table->options.end());
    return it->second;
  }
  return {};
}

std::string get_next_row(const char* curr,
                         const char* buffer_end,
                         char line_delim,
                         const std::optional<boost::regex>& line_start_regex) {
  auto row_end = curr;
  bool row_found{false};
  while (!row_found && row_end <= buffer_end) {
    if (*row_end == line_delim) {
      if (row_end == buffer_end) {
        row_found = true;
      } else if (line_start_regex.has_value()) {
        // When a LINE_START_REGEX option is present, concatenate the following lines
        // until a line that starts with the specified regex is found.
        CHECK(line_starts_with_regex(curr, 0, row_end - curr, line_start_regex.value()));
        auto row_str = get_next_row(row_end + 1, buffer_end, line_delim, {});
        while (!line_starts_with_regex(
            row_str.c_str(), 0, row_str.length() - 1, line_start_regex.value())) {
          row_end += row_str.length() + 1;
          if (row_end == buffer_end) {
            break;
          }
          row_str = get_next_row(row_end + 1, buffer_end, line_delim, {});
        }
        row_found = true;
      } else {
        row_found = true;
      }
    }
    row_end++;
  }
  CHECK(row_found);
  return std::string{curr, static_cast<size_t>(row_end - curr - 1)};
}

size_t get_row_count(const char* buffer,
                     size_t start,
                     size_t end,
                     char line_delim,
                     const std::optional<boost::regex>& line_start_regex) {
  size_t row_count{0};
  auto buffer_end = buffer + end;
  auto curr = buffer + start;
  while (curr <= buffer_end) {
    auto row_str = get_next_row(curr, buffer_end, line_delim, line_start_regex);
    curr += row_str.length() + 1;
    row_count++;
  }
  return row_count;
}

bool regex_match_columns(const std::string& row_str,
                         const boost::regex& line_regex,
                         size_t logical_column_count,
                         std::vector<std::string>& parsed_columns_str,
                         std::vector<std::string_view>& parsed_columns_sv,
                         const std::string& file_path) {
  parsed_columns_str.clear();
  parsed_columns_sv.clear();
  boost::smatch match;
  bool set_all_nulls{false};
  if (boost::regex_match(row_str, match, line_regex)) {
    auto matched_column_count = match.size() - 1;
    if (logical_column_count != matched_column_count) {
      throw_number_of_columns_mismatch_error(
          logical_column_count, matched_column_count, file_path);
    }
    CHECK_GT(match.size(), static_cast<size_t>(1));
    for (size_t i = 1; i < match.size(); i++) {
      parsed_columns_str.emplace_back(match[i].str());
      parsed_columns_sv.emplace_back(parsed_columns_str.back());
    }
  } else {
    parsed_columns_sv =
        std::vector<std::string_view>(logical_column_count, std::string_view{});
    set_all_nulls = true;
  }
  return set_all_nulls;
}
}  // namespace

RegexFileBufferParser::RegexFileBufferParser(const ForeignTable* foreign_table)
    : line_regex_(get_line_regex(foreign_table))
    , line_start_regex_(get_line_start_regex(foreign_table)) {}

/**
 * Parses a given file buffer and returns data blocks for each column in the
 * file along with metadata related to rows and row offsets within the buffer.
 */
ParseBufferResult RegexFileBufferParser::parseBuffer(
    ParseBufferRequest& request,
    bool convert_data_blocks,
    bool columns_are_pre_filtered) const {
  CHECK(request.buffer);
  char* buffer_start = request.buffer.get() + request.begin_pos;
  const char* buffer_end = request.buffer.get() + request.end_pos;

  std::vector<size_t> row_offsets;
  row_offsets.emplace_back(request.file_offset + request.begin_pos);

  size_t row_count = 0;
  auto logical_column_count = request.foreign_table_schema->getLogicalColumns().size();
  std::vector<std::string> parsed_columns_str;
  parsed_columns_str.reserve(logical_column_count);
  std::vector<std::string_view> parsed_columns_sv;
  parsed_columns_sv.reserve(logical_column_count);

  std::string row_str;
  size_t remaining_row_count = request.process_row_count;
  auto curr = buffer_start;
  while (curr < buffer_end && remaining_row_count > 0) {
    try {
      row_str = get_next_row(
          curr, buffer_end - 1, request.copy_params.line_delim, line_start_regex_);
      curr += row_str.length() + 1;
      row_count++;
      remaining_row_count--;

      bool skip_all_columns =
          std::all_of(request.import_buffers.begin(),
                      request.import_buffers.end(),
                      [](const auto& import_buffer) { return !import_buffer; });
      if (!skip_all_columns) {
        bool set_all_nulls = regex_match_columns(row_str,
                                                 line_regex_,
                                                 logical_column_count,
                                                 parsed_columns_str,
                                                 parsed_columns_sv,
                                                 request.getFilePath());

        size_t parsed_column_index = 0;
        size_t import_buffer_index = 0;
        auto columns = request.getColumns();
        for (auto cd_it = columns.begin(); cd_it != columns.end(); cd_it++) {
          auto cd = *cd_it;
          const auto& column_type = cd->columnType;
          if (request.import_buffers[import_buffer_index]) {
            bool is_null =
                (set_all_nulls || isNullDatum(parsed_columns_sv[parsed_column_index],
                                              cd,
                                              request.copy_params.null_str));
            if (column_type.is_geometry()) {
              processGeoColumn(request.import_buffers,
                               import_buffer_index,
                               request.copy_params,
                               cd_it,
                               parsed_columns_sv,
                               parsed_column_index,
                               is_null,
                               request.first_row_index,
                               row_count,
                               request.getCatalog());
              // Skip remaining physical columns
              for (int i = 0; i < cd->columnType.get_physical_cols(); ++i) {
                ++cd_it;
              }
            } else {
              request.import_buffers[import_buffer_index]->add_value(
                  cd,
                  parsed_columns_sv[parsed_column_index],
                  is_null,
                  request.copy_params);
              parsed_column_index++;
              import_buffer_index++;
            }
          } else {
            // Skip column
            for (int i = 0; i < column_type.get_physical_cols(); i++) {
              import_buffer_index++;
              cd_it++;
            }
            parsed_column_index++;
            import_buffer_index++;
          }
        }
      }
    } catch (const ForeignStorageException& e) {
      throw;
    } catch (const std::exception& e) {
      throw ForeignStorageException("Parsing failure \"" + std::string(e.what()) +
                                    "\" in row \"" + row_str + "\" in file \"" +
                                    request.getFilePath() + "\"");
    }
  }
  row_offsets.emplace_back(request.file_offset + (curr - request.buffer.get()));

  ParseBufferResult result{};
  result.row_offsets = row_offsets;
  result.row_count = row_count;
  if (convert_data_blocks) {
    result.column_id_to_data_blocks_map =
        convertImportBuffersToDataBlocks(request.import_buffers);
  }
  return result;
}

import_export::CopyParams RegexFileBufferParser::validateAndGetCopyParams(
    const ForeignTable* foreign_table) const {
  import_export::CopyParams copy_params{};
  copy_params.plain_text = true;
  if (skip_first_line_) {
    // This branch should only be executed in tests
    copy_params.has_header = import_export::ImportHeaderRow::HAS_HEADER;
  } else {
    copy_params.has_header = import_export::ImportHeaderRow::NO_HEADER;
  }
  if (auto it = foreign_table->options.find(BUFFER_SIZE_KEY);
      it != foreign_table->options.end()) {
    copy_params.buffer_size = std::stoi(it->second);
  }
  return copy_params;
}

size_t RegexFileBufferParser::findRowEndPosition(
    size_t& alloc_size,
    std::unique_ptr<char[]>& buffer,
    size_t& buffer_size,
    const import_export::CopyParams& copy_params,
    const size_t buffer_first_row_index,
    unsigned int& num_rows_in_buffer,
    foreign_storage::FileReader* file_reader) const {
  CHECK_GT(buffer_size, static_cast<size_t>(0));
  size_t start_pos{0};
  size_t end_pos = buffer_size - 1;
  bool found_end_pos{false};
  while (!found_end_pos) {
    try {
      end_pos = find_last_end_of_line(
          buffer.get(), buffer_size, start_pos, end_pos, copy_params.line_delim);
      if (file_reader->isEndOfLastFile()) {
        CHECK_EQ(end_pos, buffer_size - 1);
        found_end_pos = true;
      } else if (line_start_regex_.has_value()) {
        // When a LINE_START_REGEX option is present and the file reader is not at the end
        // of file, return the position of the end of line before the last line that
        // matches the line start regex, since the last line that matches the line start
        // regex in this buffer may still have to include/concatenate lines beyond this
        // buffer.
        CHECK_GT(end_pos, static_cast<size_t>(0));
        auto old_end_pos = end_pos;
        end_pos = find_last_end_of_line(buffer.get(),
                                        buffer_size,
                                        start_pos,
                                        old_end_pos - 1,
                                        copy_params.line_delim);
        while (!line_starts_with_regex(
            buffer.get(), end_pos + 1, old_end_pos, line_start_regex_.value())) {
          old_end_pos = end_pos;
          end_pos = find_last_end_of_line(buffer.get(),
                                          buffer_size,
                                          start_pos,
                                          old_end_pos - 1,
                                          copy_params.line_delim);
        }
        found_end_pos = true;
      } else {
        found_end_pos = true;
      }
    } catch (InsufficientBufferSizeException& e) {
      auto max_buffer_resize = getMaxBufferResize();
      if (alloc_size >= max_buffer_resize || file_reader->isScanFinished()) {
        throw;
      }
      start_pos = buffer_size;
      import_export::delimited_parser::extend_buffer(
          buffer, buffer_size, alloc_size, nullptr, file_reader, max_buffer_resize);
      end_pos = buffer_size - 1;
    }
  }
  CHECK(found_end_pos);
  num_rows_in_buffer =
      get_row_count(buffer.get(), 0, end_pos, copy_params.line_delim, line_start_regex_);
  return end_pos + 1;
}

void RegexFileBufferParser::validateFiles(const FileReader* file_reader,
                                          const ForeignTable* foreign_table) const {
  if (line_start_regex_.has_value()) {
    // When a LINE_START_REGEX option is specified, at least the first line in each file
    // has to start with the specified regex.
    auto first_line_by_file_path = file_reader->getFirstLineForEachFile();
    for (const auto& [file_path, line] : first_line_by_file_path) {
      if (!line_starts_with_regex(
              line.c_str(), 0, line.length() - 1, line_start_regex_.value())) {
        auto line_start_regex = get_line_start_regex(foreign_table);
        CHECK(line_start_regex.has_value());
        throw ForeignStorageException{"First line in file \"" + file_path +
                                      "\" does not match line start regex \"" +
                                      line_start_regex.value() + "\""};
      }
    }
  }
}

void RegexFileBufferParser::setMaxBufferResize(size_t max_buffer_resize) {
  max_buffer_resize_ = max_buffer_resize;
}

size_t RegexFileBufferParser::getMaxBufferResize() {
  return max_buffer_resize_;
}

void RegexFileBufferParser::setSkipFirstLineForTesting(bool skip) {
  skip_first_line_ = skip;
}
}  // namespace foreign_storage
