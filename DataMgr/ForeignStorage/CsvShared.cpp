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
#include "FsiJsonUtils.h"
#include "ImportExport/CopyParams.h"
#include "Utils/DdlUtils.h"

namespace foreign_storage {

// Serialization functions for FileRegion
void set_value(rapidjson::Value& json_val,
               const FileRegion& file_region,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetObject();
  json_utils::add_value_to_object(
      json_val, file_region.first_row_file_offset, "first_row_file_offset", allocator);
  json_utils::add_value_to_object(
      json_val, file_region.first_row_index, "first_row_index", allocator);
  json_utils::add_value_to_object(
      json_val, file_region.region_size, "region_size", allocator);
  json_utils::add_value_to_object(
      json_val, file_region.row_count, "row_count", allocator);
}

void get_value(const rapidjson::Value& json_val, FileRegion& file_region) {
  CHECK(json_val.IsObject());
  json_utils::get_value_from_object(
      json_val, file_region.first_row_file_offset, "first_row_file_offset");
  json_utils::get_value_from_object(
      json_val, file_region.first_row_index, "first_row_index");
  json_utils::get_value_from_object(json_val, file_region.region_size, "region_size");
  json_utils::get_value_from_object(json_val, file_region.row_count, "row_count");
}

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

Chunk_NS::Chunk make_chunk_for_column(
    const ChunkKey& chunk_key,
    std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& chunk_metadata_map,
    const std::map<ChunkKey, AbstractBuffer*>& buffers) {
  auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(chunk_key[CHUNK_KEY_DB_IDX]);
  CHECK(catalog);

  ChunkKey data_chunk_key = chunk_key;
  AbstractBuffer* data_buffer = nullptr;
  AbstractBuffer* index_buffer = nullptr;
  const auto column = catalog->getMetadataForColumnUnlocked(
      chunk_key[CHUNK_KEY_TABLE_IDX], chunk_key[CHUNK_KEY_COLUMN_IDX]);

  if (column->columnType.is_varlen_indeed()) {
    data_chunk_key.push_back(1);
    ChunkKey index_chunk_key = chunk_key;
    index_chunk_key.push_back(2);

    CHECK(buffers.find(data_chunk_key) != buffers.end());
    CHECK(buffers.find(index_chunk_key) != buffers.end());

    data_buffer = buffers.find(data_chunk_key)->second;
    index_buffer = buffers.find(index_chunk_key)->second;
    CHECK_EQ(data_buffer->size(), static_cast<size_t>(0));
    CHECK_EQ(index_buffer->size(), static_cast<size_t>(0));

    size_t index_offset_size{0};
    if (column->columnType.is_string() || column->columnType.is_geometry()) {
      index_offset_size = sizeof(StringOffsetT);
    } else if (column->columnType.is_array()) {
      index_offset_size = sizeof(ArrayOffsetT);
    } else {
      UNREACHABLE();
    }
    CHECK(chunk_metadata_map.find(data_chunk_key) != chunk_metadata_map.end());
    index_buffer->reserve(index_offset_size *
                          (chunk_metadata_map.at(data_chunk_key)->numElements + 1));
  } else {
    data_chunk_key = chunk_key;
    CHECK(buffers.find(data_chunk_key) != buffers.end());
    data_buffer = buffers.find(data_chunk_key)->second;
  }
  data_buffer->reserve(chunk_metadata_map.at(data_chunk_key)->numBytes);

  auto retval = Chunk_NS::Chunk{column};
  retval.setBuffer(data_buffer);
  retval.setIndexBuffer(index_buffer);
  retval.initEncoder();
  return retval;
}
}  // namespace Csv

}  // namespace foreign_storage
