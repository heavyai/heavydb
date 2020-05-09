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

#include "CsvDataWrapper.h"

#include <regex>

#include <boost/filesystem.hpp>

#include "Import/Importer.h"

namespace foreign_storage {
CsvDataWrapper::CsvDataWrapper(const int db_id, const ForeignTable* foreign_table)
    : db_id_(db_id), foreign_table_(foreign_table), row_count_(0) {
  initializeChunkBuffers(0);
  fetchChunkBuffers();
}

CsvDataWrapper::CsvDataWrapper(const ForeignTable* foreign_table)
    : db_id_(-1), foreign_table_(foreign_table) {}

void CsvDataWrapper::validateOptions(const ForeignTable* foreign_table) {
  for (const auto& entry : foreign_table->options) {
    const auto& table_options = foreign_table->supported_options;
    if (std::find(table_options.begin(), table_options.end(), entry.first) ==
            table_options.end() &&
        std::find(supported_options_.begin(), supported_options_.end(), entry.first) ==
            supported_options_.end()) {
      throw std::runtime_error{"Invalid foreign table option \"" + entry.first + "\"."};
    }
  }
  CsvDataWrapper{foreign_table}.validateAndGetCopyParams();
}

void CsvDataWrapper::initializeChunkBuffers(const int fragment_index) {
  const auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  const auto& columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);
  for (const auto column : columns) {
    Chunk_NS::Chunk chunk{column};
    if (column->columnType.is_varlen() && !column->columnType.is_fixlen_array()) {
      ChunkKey data_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 1};
      CHECK(chunk_buffer_map_.find(data_chunk_key) == chunk_buffer_map_.end());
      chunk_buffer_map_[data_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      chunk.set_buffer(chunk_buffer_map_[data_chunk_key].get());

      ChunkKey index_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 2};
      CHECK(chunk_buffer_map_.find(index_chunk_key) == chunk_buffer_map_.end());
      chunk_buffer_map_[index_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      chunk.set_index_buf(chunk_buffer_map_[index_chunk_key].get());
    } else {
      ChunkKey data_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index};
      CHECK(chunk_buffer_map_.find(data_chunk_key) == chunk_buffer_map_.end());
      chunk_buffer_map_[data_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      chunk.set_buffer(chunk_buffer_map_[data_chunk_key].get());
    }
    chunk.init_encoder();
  }
}

void CsvDataWrapper::fetchChunkBuffers() {
  auto file_path = getFilePath();
  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  Importer_NS::Importer importer(
      getLoader(*catalog), file_path, validateAndGetCopyParams());
  importer.import();

  if (chunk_buffer_map_.empty()) {
    throw std::runtime_error{
        "An error occurred when attempting to process data from CSV file: " + file_path};
  }
}

std::string CsvDataWrapper::getFilePath() {
  auto& server_options = foreign_table_->foreign_server->options;
  auto base_path_entry = server_options.find("BASE_PATH");
  if (base_path_entry == server_options.end()) {
    throw std::runtime_error{"No base path found in foreign server options."};
  }
  auto file_path_entry = foreign_table_->options.find("FILE_PATH");
  std::string file_path{};
  if (file_path_entry != foreign_table_->options.end()) {
    file_path = file_path_entry->second;
  }
  const std::string separator{boost::filesystem::path::preferred_separator};
  return std::regex_replace(base_path_entry->second + separator + file_path,
                            std::regex{separator + "{2,}"},
                            separator);
}

Importer_NS::CopyParams CsvDataWrapper::validateAndGetCopyParams() {
  Importer_NS::CopyParams copy_params{};
  if (const auto& value = validateAndGetStringWithLength("ARRAY_DELIMITER", 1);
      !value.empty()) {
    copy_params.array_delim = value[0];
  }
  if (const auto& value = validateAndGetStringWithLength("ARRAY_MARKER", 2);
      !value.empty()) {
    copy_params.array_begin = value[0];
    copy_params.array_end = value[1];
  }
  if (auto it = foreign_table_->options.find("BUFFER_SIZE");
      it != foreign_table_->options.end()) {
    copy_params.buffer_size = std::stoi(it->second);
  }
  if (const auto& value = validateAndGetStringWithLength("DELIMITER", 1);
      !value.empty()) {
    copy_params.delimiter = value[0];
  }
  if (const auto& value = validateAndGetStringWithLength("ESCAPE", 1); !value.empty()) {
    copy_params.escape = value[0];
  }
  auto has_header = validateAndGetBoolValue("HEADER");
  if (has_header.has_value()) {
    if (has_header.value()) {
      copy_params.has_header = Importer_NS::ImportHeaderRow::HAS_HEADER;
    } else {
      copy_params.has_header = Importer_NS::ImportHeaderRow::NO_HEADER;
    }
  }
  if (const auto& value = validateAndGetStringWithLength("LINE_DELIMITER", 1);
      !value.empty()) {
    copy_params.line_delim = value[0];
  }
  copy_params.lonlat = validateAndGetBoolValue("LONLAT").value_or(copy_params.lonlat);

  if (auto it = foreign_table_->options.find("NULLS");
      it != foreign_table_->options.end()) {
    copy_params.null_str = it->second;
  }
  if (const auto& value = validateAndGetStringWithLength("QUOTE", 1); !value.empty()) {
    copy_params.quote = value[0];
  }
  copy_params.quoted = validateAndGetBoolValue("QUOTED").value_or(copy_params.quoted);
  return copy_params;
}

std::string CsvDataWrapper::validateAndGetStringWithLength(
    const std::string& option_name,
    const size_t expected_num_chars) {
  if (auto it = foreign_table_->options.find(option_name);
      it != foreign_table_->options.end()) {
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

std::optional<bool> CsvDataWrapper::validateAndGetBoolValue(
    const std::string& option_name) {
  if (auto it = foreign_table_->options.find(option_name);
      it != foreign_table_->options.end()) {
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

Importer_NS::Loader* CsvDataWrapper::getLoader(Catalog_Namespace::Catalog& catalog) {
  auto callback = [this](
                      const std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>&
                          import_buffers,
                      std::vector<DataBlockPtr>& data_blocks,
                      size_t import_row_count) {
    std::lock_guard loader_lock(loader_mutex_);
    size_t processed_import_row_count = 0;
    while (processed_import_row_count < import_row_count) {
      int fragment_index = row_count_ / foreign_table_->maxFragRows;
      size_t row_count_for_fragment;

      if (fragmentIsFull()) {
        row_count_for_fragment = std::min<size_t>(
            foreign_table_->maxFragRows, import_row_count - processed_import_row_count);
        initializeChunkBuffers(fragment_index);
      } else {
        row_count_for_fragment = std::min<size_t>(
            foreign_table_->maxFragRows - (row_count_ % foreign_table_->maxFragRows),
            import_row_count - processed_import_row_count);
      }
      for (size_t i = 0; i < import_buffers.size(); i++) {
        Chunk_NS::Chunk chunk{import_buffers[i]->getColumnDesc()};
        auto column_id = import_buffers[i]->getColumnDesc()->columnId;
        auto& type_info = import_buffers[i]->getTypeInfo();
        if (type_info.is_varlen() && !type_info.is_fixlen_array()) {
          ChunkKey data_chunk_key{
              db_id_, foreign_table_->tableId, column_id, fragment_index, 1};
          CHECK(chunk_buffer_map_.find(data_chunk_key) != chunk_buffer_map_.end());
          chunk.set_buffer(chunk_buffer_map_[data_chunk_key].get());

          ChunkKey index_chunk_key{
              db_id_, foreign_table_->tableId, column_id, fragment_index, 2};
          CHECK(chunk_buffer_map_.find(index_chunk_key) != chunk_buffer_map_.end());
          chunk.set_index_buf(chunk_buffer_map_[index_chunk_key].get());
        } else {
          ChunkKey data_chunk_key{
              db_id_, foreign_table_->tableId, column_id, fragment_index};
          CHECK(chunk_buffer_map_.find(data_chunk_key) != chunk_buffer_map_.end());
          chunk.set_buffer(chunk_buffer_map_[data_chunk_key].get());
        }
        chunk.appendData(
            data_blocks[i], row_count_for_fragment, processed_import_row_count);
        chunk.set_buffer(nullptr);
        chunk.set_index_buf(nullptr);
      }
      row_count_ += row_count_for_fragment;
      processed_import_row_count += row_count_for_fragment;
    }
    return true;
  };

  return new Importer_NS::Loader(catalog, foreign_table_, callback);
}

bool CsvDataWrapper::fragmentIsFull() {
  return row_count_ != 0 && (row_count_ % foreign_table_->maxFragRows) == 0;
}

ForeignStorageBuffer* CsvDataWrapper::getChunkBuffer(const ChunkKey& chunk_key) {
  return getBufferFromMap(chunk_key);
}

void CsvDataWrapper::populateMetadataForChunkKeyPrefix(
    const ChunkKey& chunk_key_prefix,
    ChunkMetadataVector& chunk_metadata_vector) {
  for (auto& [buffer_chunk_key, buffer] : chunk_buffer_map_) {
    if (buffer->has_encoder && prefixMatch(chunk_key_prefix, buffer_chunk_key)) {
      auto chunk_metadata = std::make_shared<ChunkMetadata>();
      buffer->encoder->getMetadata(chunk_metadata);
      chunk_metadata_vector.emplace_back(buffer_chunk_key, chunk_metadata);
    }
  }
}

ForeignStorageBuffer* CsvDataWrapper::getBufferFromMap(const ChunkKey& chunk_key) {
  CHECK(chunk_buffer_map_.find(chunk_key) != chunk_buffer_map_.end());
  return chunk_buffer_map_[chunk_key].get();
}

bool CsvDataWrapper::prefixMatch(const ChunkKey& prefix, const ChunkKey& checked) {
  if (prefix.size() > checked.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.size(); i++) {
    if (prefix[i] != checked[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace foreign_storage
