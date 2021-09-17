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

#include "DataMgr/ForeignStorage/CsvFileBufferParser.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "ImportExport/Importer.h"
#include "Shared/StringTransform.h"

namespace foreign_storage {

namespace {
inline bool skip_column_import(ParseBufferRequest& request, int column_idx) {
  return request.import_buffers[column_idx] == nullptr;
}

void set_array_flags_and_geo_columns_count(
    std::unique_ptr<bool[]>& array_flags,
    int& phys_cols,
    int& point_cols,
    const std::list<const ColumnDescriptor*>& columns) {
  array_flags = std::unique_ptr<bool[]>(new bool[columns.size()]);
  size_t i = 0;
  for (const auto cd : columns) {
    const auto& col_ti = cd->columnType;
    phys_cols += col_ti.get_physical_cols();
    if (cd->columnType.get_type() == kPOINT) {
      point_cols++;
    }

    if (cd->columnType.get_type() == kARRAY) {
      array_flags.get()[i] = true;
    } else {
      array_flags.get()[i] = false;
    }
    i++;
  }
}

void validate_expected_column_count(std::vector<std::string_view>& row,
                                    size_t num_cols,
                                    int point_cols,
                                    const std::string& file_name) {
  // Each POINT could consume two separate coords instead of a single WKT
  if (row.size() < num_cols || (num_cols + point_cols) < row.size()) {
    throw_number_of_columns_mismatch_error(num_cols, row.size(), file_name);
  }
}

std::string validate_and_get_delimiter(const ForeignTable* foreign_table,
                                       const std::string& option_name) {
  if (auto it = foreign_table->options.find(option_name);
      it != foreign_table->options.end()) {
    if (it->second.length() == 1) {
      return it->second;
    } else {
      if (it->second == std::string("\\n")) {
        return "\n";
      } else if (it->second == std::string("\\t")) {
        return "\t";
      } else {
        throw std::runtime_error{"Invalid value specified for option \"" + option_name +
                                 "\". Expected a single character, \"\\n\" or  \"\\t\"."};
      }
    }
  }
  return "";
}

std::string validate_and_get_string_with_length(const ForeignTable* foreign_table,
                                                const std::string& option_name,
                                                const size_t expected_num_chars) {
  if (auto it = foreign_table->options.find(option_name);
      it != foreign_table->options.end()) {
    if (it->second.length() != expected_num_chars) {
      throw std::runtime_error{"Value of \"" + option_name +
                               "\" foreign table option has the wrong number of "
                               "characters. Expected " +
                               std::to_string(expected_num_chars) + " character(s)."};
    }
    return it->second;
  }
  return "";
}

std::optional<bool> validate_and_get_bool_value(const ForeignTable* foreign_table,
                                                const std::string& option_name) {
  if (auto it = foreign_table->options.find(option_name);
      it != foreign_table->options.end()) {
    if (boost::iequals(it->second, "TRUE")) {
      return true;
    } else if (boost::iequals(it->second, "FALSE")) {
      return false;
    } else {
      throw std::runtime_error{"Invalid boolean value specified for \"" + option_name +
                               "\" foreign table option. "
                               "Value must be either 'true' or 'false'."};
    }
  }
  return std::nullopt;
}
}  // namespace

/**
 * Parses a given CSV file buffer and returns data blocks for each column in the
 * file along with metadata related to rows and row offsets within the buffer.
 */
ParseBufferResult CsvFileBufferParser::parseBuffer(ParseBufferRequest& request,
                                                   bool convert_data_blocks,
                                                   bool columns_are_pre_filtered) const {
  CHECK(request.buffer);
  size_t begin = import_export::delimited_parser::find_beginning(
      request.buffer.get(), request.begin_pos, request.end_pos, request.copy_params);
  const char* thread_buf = request.buffer.get() + request.begin_pos + begin;
  const char* thread_buf_end = request.buffer.get() + request.end_pos;
  const char* buf_end = request.buffer.get() + request.buffer_size;

  std::vector<std::string_view> row;
  size_t row_index_plus_one = 0;
  const char* p = thread_buf;
  bool try_single_thread = false;
  int phys_cols = 0;
  int point_cols = 0;
  std::unique_ptr<bool[]> array_flags;

  set_array_flags_and_geo_columns_count(
      array_flags, phys_cols, point_cols, request.getColumns());
  auto num_cols = request.getColumns().size() - phys_cols;
  if (columns_are_pre_filtered) {
    for (size_t col_idx = 0; col_idx < request.getColumns().size(); ++col_idx) {
      if (skip_column_import(request, col_idx)) {
        --num_cols;
      }
    }
  }

  size_t row_count = 0;
  size_t remaining_row_count = request.process_row_count;
  std::vector<size_t> row_offsets{};
  row_offsets.emplace_back(request.file_offset + (p - request.buffer.get()));

  std::string file_path = request.getFilePath();
  for (; p < thread_buf_end && remaining_row_count > 0; p++, remaining_row_count--) {
    row.clear();
    row_count++;
    std::vector<std::unique_ptr<char[]>>
        tmp_buffers;  // holds string w/ removed escape chars, etc
    const char* line_start = p;
    p = import_export::delimited_parser::get_row(p,
                                                 thread_buf_end,
                                                 buf_end,
                                                 request.copy_params,
                                                 array_flags.get(),
                                                 row,
                                                 tmp_buffers,
                                                 try_single_thread,
                                                 !columns_are_pre_filtered);

    row_index_plus_one++;
    validate_expected_column_count(row, num_cols, point_cols, file_path);

    size_t import_idx = 0;
    size_t col_idx = 0;
    try {
      auto columns = request.getColumns();
      for (auto cd_it = columns.begin(); cd_it != columns.end(); cd_it++) {
        auto cd = *cd_it;
        const auto& col_ti = cd->columnType;
        bool column_is_present =
            !(skip_column_import(request, col_idx) && columns_are_pre_filtered);
        CHECK(row.size() > import_idx || !column_is_present);
        bool is_null =
            column_is_present
                ? isNullDatum(row[import_idx], cd, request.copy_params.null_str)
                : true;

        if (col_ti.is_geometry()) {
          if (!skip_column_import(request, col_idx)) {
            processGeoColumn(request.import_buffers,
                             col_idx,
                             request.copy_params,
                             cd_it,
                             row,
                             import_idx,
                             is_null,
                             request.first_row_index,
                             row_index_plus_one,
                             request.getCatalog());
          } else {
            // update import/col idx according to types
            if (!is_null && cd->columnType == kPOINT &&
                isCoordinateScalar(row[import_idx])) {
              if (!columns_are_pre_filtered)
                ++import_idx;
            }
            if (!columns_are_pre_filtered)
              ++import_idx;
            ++col_idx;
            col_idx += col_ti.get_physical_cols();
          }
          // skip remaining physical columns
          for (int i = 0; i < cd->columnType.get_physical_cols(); ++i) {
            ++cd_it;
          }
        } else {
          if (!skip_column_import(request, col_idx)) {
            request.import_buffers[col_idx]->add_value(
                cd, row[import_idx], is_null, request.copy_params);
          }
          if (column_is_present) {
            ++import_idx;
          }
          ++col_idx;
        }
      }
    } catch (const std::exception& e) {
      throw ForeignStorageException("Parsing failure \"" + std::string(e.what()) +
                                    "\" in row \"" + std::string(line_start, p) +
                                    "\" in file \"" + file_path + "\"");
    }
  }
  row_offsets.emplace_back(request.file_offset + (p - request.buffer.get()));

  ParseBufferResult result{};
  result.row_offsets = row_offsets;
  result.row_count = row_count;
  if (convert_data_blocks) {
    result.column_id_to_data_blocks_map =
        convertImportBuffersToDataBlocks(request.import_buffers);
  }
  return result;
}
/**
 * Takes a single row and verifies number of columns is valid for num_cols and point_cols
 * (number of point columns)
 */
void CsvFileBufferParser::validateExpectedColumnCount(
    const std::string& row,
    const import_export::CopyParams& copy_params,
    size_t num_cols,
    int point_cols,
    const std::string& file_name) const {
  bool is_array = false;
  bool try_single_thread = false;
  std::vector<std::unique_ptr<char[]>> tmp_buffers;
  std::vector<std::string_view> fields;
  // parse columns in row into fields (other return values are intentionally ignored)
  import_export::delimited_parser::get_row(row.c_str(),
                                           row.c_str() + row.size(),
                                           row.c_str() + row.size(),
                                           copy_params,
                                           &is_array,
                                           fields,
                                           tmp_buffers,
                                           try_single_thread,
                                           false  // Don't filter empty lines
  );
  // Check we have right number of columns
  validate_expected_column_count(fields, num_cols, point_cols, file_name);
}

import_export::CopyParams CsvFileBufferParser::validateAndGetCopyParams(
    const ForeignTable* foreign_table) const {
  import_export::CopyParams copy_params{};
  copy_params.plain_text = true;
  if (const auto& value =
          validate_and_get_string_with_length(foreign_table, "ARRAY_DELIMITER", 1);
      !value.empty()) {
    copy_params.array_delim = value[0];
  }
  if (const auto& value =
          validate_and_get_string_with_length(foreign_table, "ARRAY_MARKER", 2);
      !value.empty()) {
    copy_params.array_begin = value[0];
    copy_params.array_end = value[1];
  }
  if (auto it = foreign_table->options.find("BUFFER_SIZE");
      it != foreign_table->options.end()) {
    copy_params.buffer_size = std::stoi(it->second);
  }
  if (const auto& value = validate_and_get_delimiter(foreign_table, "DELIMITER");
      !value.empty()) {
    copy_params.delimiter = value[0];
  }
  if (const auto& value = validate_and_get_string_with_length(foreign_table, "ESCAPE", 1);
      !value.empty()) {
    copy_params.escape = value[0];
  }
  auto has_header = validate_and_get_bool_value(foreign_table, "HEADER");
  if (has_header.has_value()) {
    if (has_header.value()) {
      copy_params.has_header = import_export::ImportHeaderRow::HAS_HEADER;
    } else {
      copy_params.has_header = import_export::ImportHeaderRow::NO_HEADER;
    }
  }
  if (const auto& value = validate_and_get_delimiter(foreign_table, "LINE_DELIMITER");
      !value.empty()) {
    copy_params.line_delim = value[0];
  }
  copy_params.lonlat =
      validate_and_get_bool_value(foreign_table, "LONLAT").value_or(copy_params.lonlat);

  if (auto it = foreign_table->options.find("NULLS");
      it != foreign_table->options.end()) {
    copy_params.null_str = it->second;
  }
  if (const auto& value = validate_and_get_string_with_length(foreign_table, "QUOTE", 1);
      !value.empty()) {
    copy_params.quote = value[0];
  }
  copy_params.quoted =
      validate_and_get_bool_value(foreign_table, "QUOTED").value_or(copy_params.quoted);
  return copy_params;
}

size_t CsvFileBufferParser::findRowEndPosition(
    size_t& alloc_size,
    std::unique_ptr<char[]>& buffer,
    size_t& buffer_size,
    const import_export::CopyParams& copy_params,
    const size_t buffer_first_row_index,
    unsigned int& num_rows_in_buffer,
    foreign_storage::FileReader* file_reader) const {
  return import_export::delimited_parser::find_row_end_pos(alloc_size,
                                                           buffer,
                                                           buffer_size,
                                                           copy_params,
                                                           buffer_first_row_index,
                                                           num_rows_in_buffer,
                                                           nullptr,
                                                           file_reader);
}
}  // namespace foreign_storage
