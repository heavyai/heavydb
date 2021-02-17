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
#include "CsvShared.h"
#include "CsvDataWrapper.h"
#include "DataMgr/ForeignStorage/ForeignTableSchema.h"
#include "ImportExport/CopyParams.h"
#include "Utils/DdlUtils.h"

namespace foreign_storage {
namespace Csv {
namespace {
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

bool validate_and_get_is_s3_select(const ForeignTable* foreign_table) {
  static constexpr const char* S3_DIRECT = "S3_DIRECT";
  static constexpr const char* S3_SELECT = "S3_SELECT";
  static constexpr const char* S3_ACCESS_TYPE = "S3_ACCESS_TYPE";
  auto access_type = foreign_table->options.find(S3_ACCESS_TYPE);

  if (access_type != foreign_table->options.end()) {
    auto& server_options = foreign_table->foreign_server->options;
    if (server_options.find(AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY)->second !=
        AbstractFileStorageDataWrapper::S3_STORAGE_TYPE) {
      throw std::runtime_error{
          "The \"" + std::string{S3_ACCESS_TYPE} +
          "\" option is only valid for foreign tables using servers with \"" +
          AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY + "\" option value of \"" +
          AbstractFileStorageDataWrapper::S3_STORAGE_TYPE + "\"."};
    }
    if (access_type->second != S3_DIRECT && access_type->second != S3_SELECT) {
      throw std::runtime_error{
          "Invalid value provided for the \"" + std::string{S3_ACCESS_TYPE} +
          "\" option. Value must be one of the following: " + S3_DIRECT + ", " +
          S3_SELECT + "."};
    }
    return (access_type->second == S3_SELECT);
  } else {
    return false;
  }
}

void validate_options(const ForeignTable* foreign_table) {
  validate_and_get_copy_params(foreign_table);
  validate_and_get_is_s3_select(foreign_table);
}

import_export::CopyParams validate_and_get_copy_params(
    const ForeignTable* foreign_table) {
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
  if (const auto& value =
          validate_and_get_string_with_length(foreign_table, "DELIMITER", 1);
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
  if (const auto& value =
          validate_and_get_string_with_length(foreign_table, "LINE_DELIMITER", 1);
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

}  // namespace Csv

}  // namespace foreign_storage
