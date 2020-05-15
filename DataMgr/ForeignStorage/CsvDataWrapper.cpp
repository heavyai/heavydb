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

#include "ImportExport/Importer.h"
#include "Utils/DdlUtils.h"

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
  CsvDataWrapper data_wrapper{foreign_table};
  data_wrapper.validateAndGetCopyParams();
  data_wrapper.validateFilePath();
}

void CsvDataWrapper::initializeChunkBuffers(const int fragment_index) {
  auto timer = DEBUG_TIMER(__func__);
  const auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  const auto& columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);

  // Create an empty vector of file regions used in this fragment
  fragment_fileregion_map_[fragment_index] = {};

  for (const auto column : columns) {
    // Create an empty vector of subchunks corresponding to the file regions in
    // fragment_fileregion_map_
    ChunkKey data_chunk_key{
        db_id_, foreign_table_->tableId, column->columnId, fragment_index};

    Chunk_NS::Chunk chunk{column};
    if (column->columnType.is_varlen() && !column->columnType.is_fixlen_array()) {
      ChunkKey data_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 1};
      CHECK(chunk_buffer_map_.find(data_chunk_key) == chunk_buffer_map_.end());
      chunk_buffer_map_[data_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      chunk.setBuffer(chunk_buffer_map_[data_chunk_key].get());

      ChunkKey index_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 2};
      CHECK(chunk_buffer_map_.find(index_chunk_key) == chunk_buffer_map_.end());
      chunk_buffer_map_[index_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      chunk.setIndexBuffer(chunk_buffer_map_[index_chunk_key].get());
    } else {
      ChunkKey data_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index};
      CHECK(chunk_buffer_map_.find(data_chunk_key) == chunk_buffer_map_.end());
      chunk_buffer_map_[data_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      chunk.setBuffer(chunk_buffer_map_[data_chunk_key].get());
      // Reserve data so we dont need to keep re-allocating and moving
      chunk_buffer_map_[data_chunk_key].get()->reserve(
          column->columnType.get_logical_size() * foreign_table_->maxFragRows);
    }
    chunk.initEncoder();
  }
}

void CsvDataWrapper::discardFragmentBuffers(const int fragment_index) {
  auto timer = DEBUG_TIMER(__func__);
  // Discard data buffers in the chunk buffer map
  const auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  const auto& columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);
  for (const auto column : columns) {
    if (column->columnType.is_varlen() && !column->columnType.is_fixlen_array()) {
      ChunkKey index_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 2};
      CHECK(chunk_buffer_map_.find(index_chunk_key) != chunk_buffer_map_.end());
      chunk_buffer_map_[index_chunk_key].get()->discardBuffer();
      ChunkKey data_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 1};
      CHECK(chunk_buffer_map_.find(data_chunk_key) != chunk_buffer_map_.end());
      chunk_buffer_map_[data_chunk_key].get()->discardBuffer();
    } else {
      ChunkKey data_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index};

      CHECK(chunk_buffer_map_.find(data_chunk_key) != chunk_buffer_map_.end());
      chunk_buffer_map_[data_chunk_key].get()->discardBuffer();
    }
  }
}

void CsvDataWrapper::fetchChunkBuffers() {
  auto timer = DEBUG_TIMER(__func__);
  auto file_path = getFilePath();
  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  CsvLazyLoader loader(
      getMetadataLoader(*catalog), file_path, validateAndGetCopyParams());
  loader.scanMetadata();

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

void CsvDataWrapper::validateFilePath() {
  ddl_utils::validate_allowed_file_path(getFilePath(),
                                        ddl_utils::DataTransferType::IMPORT);
}

import_export::CopyParams CsvDataWrapper::validateAndGetCopyParams() {
  import_export::CopyParams copy_params{};
  copy_params.plain_text = true;
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
      copy_params.has_header = import_export::ImportHeaderRow::HAS_HEADER;
    } else {
      copy_params.has_header = import_export::ImportHeaderRow::NO_HEADER;
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

import_export::Loader* CsvDataWrapper::getMetadataLoader(
    Catalog_Namespace::Catalog& catalog) {
  auto timer = DEBUG_TIMER(__func__);
  auto callback = [this](const std::vector<std::unique_ptr<
                             import_export::TypedImportBuffer>>& import_buffers,
                         std::vector<DataBlockPtr>& data_blocks,
                         size_t import_row_count,
                         const Importer_NS::FileScanMetadata* file_scan_metadata) {
    CHECK(file_scan_metadata);
    auto csv_file_scan_metadata =
        dynamic_cast<const Importer_NS::CsvFileScanMetadata*>(file_scan_metadata);
    CHECK(csv_file_scan_metadata);
    const auto& row_offsets = csv_file_scan_metadata->row_offsets;

    std::lock_guard loader_lock(loader_mutex_);
    size_t processed_import_row_count = 0;
    while (processed_import_row_count < import_row_count) {
      int fragment_index = row_count_ / foreign_table_->maxFragRows;
      size_t row_count_for_fragment;

      if (fragmentIsFull()) {
        row_count_for_fragment = std::min<size_t>(
            foreign_table_->maxFragRows, import_row_count - processed_import_row_count);
        // Discard data. When cache is integrated, try to push into cache instead
        discardFragmentBuffers(fragment_index - 1);
        initializeChunkBuffers(fragment_index);
      } else {
        row_count_for_fragment = std::min<size_t>(
            foreign_table_->maxFragRows - (row_count_ % foreign_table_->maxFragRows),
            import_row_count - processed_import_row_count);
      }

      // Store the file region in the map
      foreign_storage::FileRegion file_region;
      file_region.filename = "";
      file_region.first_row_file_offset = row_offsets[processed_import_row_count];
      file_region.region_size =
          row_offsets[processed_import_row_count + row_count_for_fragment] -
          row_offsets[processed_import_row_count];

      fragment_fileregion_map_[fragment_index].push_back(file_region);

      for (size_t i = 0; i < import_buffers.size(); i++) {
        Chunk_NS::Chunk chunk{import_buffers[i]->getColumnDesc()};
        auto column_id = import_buffers[i]->getColumnDesc()->columnId;
        auto& type_info = import_buffers[i]->getTypeInfo();
        ChunkKey data_chunk_key;
        ChunkKey index_chunk_key;
        size_t index_offset;
        if (type_info.is_varlen() && !type_info.is_fixlen_array()) {
          data_chunk_key = {
              db_id_, foreign_table_->tableId, column_id, fragment_index, 1};
          CHECK(chunk_buffer_map_.find(data_chunk_key) != chunk_buffer_map_.end());
          chunk.setBuffer(chunk_buffer_map_[data_chunk_key].get());

          index_chunk_key = {
              db_id_, foreign_table_->tableId, column_id, fragment_index, 2};
          CHECK(chunk_buffer_map_.find(index_chunk_key) != chunk_buffer_map_.end());
          chunk.setIndexBuffer(chunk_buffer_map_[index_chunk_key].get());

          index_offset = chunk_buffer_map_[index_chunk_key].get()->size();
        } else {
          data_chunk_key = {db_id_, foreign_table_->tableId, column_id, fragment_index};
          CHECK(chunk_buffer_map_.find(data_chunk_key) != chunk_buffer_map_.end());
          chunk.setBuffer(chunk_buffer_map_[data_chunk_key].get());
        }

        size_t init_offset = chunk_buffer_map_[data_chunk_key].get()->size();
        chunk.appendData(
            data_blocks[i], row_count_for_fragment, processed_import_row_count);
        chunk.setBuffer(nullptr);
        chunk.setIndexBuffer(nullptr);

        size_t final_offset = chunk_buffer_map_[data_chunk_key].get()->size();
        size_t size = final_offset - init_offset;

        SubChunkRegion data_chunk_desc;
        data_chunk_desc.buffer_offset = init_offset;
        data_chunk_desc.buffer_size = size;

        // Store this so we can find this subchunk by the chunk key and file offset
        chunk_file_offset_subchunk_map_[{
            data_chunk_key, file_region.first_row_file_offset}] = data_chunk_desc;

        if (type_info.is_varlen() && !type_info.is_fixlen_array()) {
          // Also store info for index chunk
          SubChunkRegion index_chunk_desc;
          index_chunk_desc.buffer_offset = index_offset;
          index_chunk_desc.buffer_size =
              chunk_buffer_map_[data_chunk_key].get()->size() - index_offset;
          chunk_file_offset_subchunk_map_[{
              index_chunk_key, file_region.first_row_file_offset}] = index_chunk_desc;
        }
      }
      row_count_ += row_count_for_fragment;
      processed_import_row_count += row_count_for_fragment;
    }
    return true;
  };

  return new import_export::Loader(catalog, foreign_table_, callback);
}

import_export::Loader* CsvDataWrapper::getLazyLoader(
    Catalog_Namespace::Catalog& catalog,
    const ChunkKey data_chunk_key,
    std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>>* temp_buffer_map_ptr) {
  auto timer = DEBUG_TIMER(__func__);
  auto callback = [this, data_chunk_key, temp_buffer_map_ptr](
                      const std::vector<std::unique_ptr<
                          import_export::TypedImportBuffer>>& import_buffers,
                      std::vector<DataBlockPtr>& data_blocks,
                      size_t import_row_count,
                      const Importer_NS::FileScanMetadata* file_scan_metadata) {
    CHECK(file_scan_metadata);
    auto csv_file_scan_metadata =
        dynamic_cast<const Importer_NS::CsvFileScanMetadata*>(file_scan_metadata);
    CHECK(csv_file_scan_metadata);
    const auto& row_offsets = csv_file_scan_metadata->row_offsets;

    std::lock_guard loader_lock(loader_mutex_);

    // Importer passed us the offset for each row
    // Make sure this data starts at an existing subchunk for this chunk key
    CHECK(row_offsets.size() > 0);
    CHECK(chunk_file_offset_subchunk_map_.find({data_chunk_key, row_offsets[0]}) !=
          chunk_file_offset_subchunk_map_.end());

    SubChunkRegion data_chunk_desc =
        chunk_file_offset_subchunk_map_[{data_chunk_key, row_offsets[0]}];

    // Find import_buffer that contains this requested chunk
    int col_index = -1;
    for (size_t i = 0; i < import_buffers.size(); i++) {
      if (import_buffers[i]->getColumnDesc()->columnId == data_chunk_key[2])
        col_index = i;
    }
    CHECK(col_index > -1);

    Chunk_NS::Chunk chunk{import_buffers[col_index]->getColumnDesc()};

    auto& type_info = import_buffers[col_index]->getTypeInfo();
    if (type_info.is_varlen() && !type_info.is_fixlen_array()) {
      ChunkKey index_chunk_key = {
          data_chunk_key[0], data_chunk_key[1], data_chunk_key[2], data_chunk_key[3], 2};
      chunk.set_buffer((*temp_buffer_map_ptr)[data_chunk_key].get());
      chunk.set_index_buf((*temp_buffer_map_ptr)[index_chunk_key].get());

      size_t data_start = (*temp_buffer_map_ptr)[data_chunk_key].get()->size();
      size_t index_start = (*temp_buffer_map_ptr)[index_chunk_key].get()->size();
      // Append full data_block starting at index 0 into the chunk
      chunk.appendData(data_blocks[col_index], import_row_count, 0);
      chunk.set_buffer(nullptr);
      chunk.set_index_buf(nullptr);
      size_t data_end = (*temp_buffer_map_ptr)[data_chunk_key].get()->size();
      size_t index_end = (*temp_buffer_map_ptr)[index_chunk_key].get()->size();
      CHECK(data_chunk_desc.buffer_size == (data_end - data_start));
      (*temp_buffer_map_ptr)[data_chunk_key].get()->read(
          chunk_buffer_map_[data_chunk_key].get()->getMemoryPtr() +
              data_chunk_desc.buffer_offset,
          data_chunk_desc.buffer_size,
          data_start);

      // Now we need to offset the index buffer entries to account for the copy
      // Make sure this logic is the same for String and Array offsets
      static_assert(sizeof(StringOffsetT) == sizeof(ArrayOffsetT));

      SubChunkRegion index_chunk_desc =
          chunk_file_offset_subchunk_map_[{index_chunk_key, row_offsets[0]}];

      StringOffsetT* tmp_indexes =
          (StringOffsetT*)((*temp_buffer_map_ptr)[index_chunk_key].get()->getMemoryPtr() +
                           index_start);
      StringOffsetT* dst_indexes =
          (StringOffsetT*)(chunk_buffer_map_[index_chunk_key].get()->getMemoryPtr() +
                           index_chunk_desc.buffer_offset);

      size_t num_indexes = (index_end - index_start) / sizeof(StringOffsetT);
      if (index_start == 0) {
        // This is the first subchunk loaded into the temporary buffer
        // Skip the first index to account for extra first row index
        tmp_indexes++;
        num_indexes--;
      }

      if (index_chunk_desc.buffer_offset == 0) {
        // This is the first subchunk in the real chunk
        // Always set first index to 0, and copy the new indexes after it
        dst_indexes[0] = 0;
        dst_indexes++;
      }
      for (size_t i = 0; i < num_indexes; i++) {
        dst_indexes[i] = tmp_indexes[i] + data_chunk_desc.buffer_offset - data_start;
      }

    } else {
      chunk.set_buffer((*temp_buffer_map_ptr)[data_chunk_key].get());
      size_t start = (*temp_buffer_map_ptr)[data_chunk_key].get()->size();
      // Append full data_block starting at data_chunk_keyt index 0 into the chunk
      chunk.appendData(data_blocks[col_index], import_row_count, 0);
      chunk.set_buffer(nullptr);
      (*temp_buffer_map_ptr)[data_chunk_key].get()->read(
          chunk_buffer_map_[data_chunk_key].get()->getMemoryPtr() +
              data_chunk_desc.buffer_offset,
          data_chunk_desc.buffer_size,
          start);
    }
    return true;
  };

  return new import_export::Loader(catalog, foreign_table_, callback);
}

bool CsvDataWrapper::fragmentIsFull() {
  return row_count_ != 0 && (row_count_ % foreign_table_->maxFragRows) == 0;
}

ForeignStorageBuffer* CsvDataWrapper::getChunkBuffer(const ChunkKey& chunk_key) {
  auto timer = DEBUG_TIMER(__func__);
  // Need to populate data before returning it

  if (!chunk_buffer_map_[chunk_key].get()->bufferExists()) {
    auto catalog = Catalog_Namespace::Catalog::get(db_id_);
    CHECK(catalog);
    const auto& column =
        catalog.get()->getMetadataForColumn(foreign_table_->tableId, chunk_key[2]);
    Chunk_NS::Chunk chunk{column};
    ChunkKey load_key = chunk_key;
    std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> temp_buffer_map;
    if (chunk_key.size() > 4) {
      // This is either the data or index buffer of variable length chunk
      ChunkKey data_chunk_key = {
          chunk_key[0], chunk_key[1], chunk_key[2], chunk_key[3], 1};
      // Pass the data chunk key into the loader
      load_key = data_chunk_key;

      temp_buffer_map[data_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      temp_buffer_map[data_chunk_key].get()->reserve(
          chunk_buffer_map_[chunk_key].get()->size());

      ChunkKey index_chunk_key = {
          chunk_key[0], chunk_key[1], chunk_key[2], chunk_key[3], 2};

      CHECK(chunk_buffer_map_.find(index_chunk_key) != chunk_buffer_map_.end());

      temp_buffer_map[index_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      chunk.set_buffer(temp_buffer_map[data_chunk_key].get());
      chunk.set_index_buf(temp_buffer_map[index_chunk_key].get());

      getBufferFromMap(data_chunk_key)->reallocBuffer();
      getBufferFromMap(index_chunk_key)->reallocBuffer();
    } else {
      getBufferFromMap(chunk_key)->reallocBuffer();
      temp_buffer_map[chunk_key] = std::make_unique<ForeignStorageBuffer>();
      temp_buffer_map[chunk_key].get()->reserve(
          chunk_buffer_map_[chunk_key].get()->size());
      chunk.set_buffer(temp_buffer_map[chunk_key].get());
    }
    chunk.init_encoder();
    CsvLazyLoader loader(getLazyLoader(*catalog, load_key, &temp_buffer_map),
                         getFilePath(),
                         validateAndGetCopyParams());

    int fragment_index = chunk_key[3];

    // Make sure the fragment exists and has file regions to load
    CHECK(fragment_fileregion_map_.find(fragment_index) !=
          fragment_fileregion_map_.end());
    CHECK(fragment_fileregion_map_[fragment_index].size() > 0);

    loader.fetchRegions(fragment_fileregion_map_[fragment_index]);

    // Delete temporary buffers
    if (chunk_key.size() > 4) {
      chunk.set_buffer(nullptr);
      chunk.set_index_buf(nullptr);
      ChunkKey data_chunk_key = {
          chunk_key[0], chunk_key[1], chunk_key[2], chunk_key[3], 1};
      temp_buffer_map[data_chunk_key] = nullptr;
      ChunkKey index_chunk_key = {
          chunk_key[0], chunk_key[1], chunk_key[2], chunk_key[3], 2};
      temp_buffer_map[index_chunk_key] = nullptr;
    } else {
      chunk.set_buffer(nullptr);
      temp_buffer_map[chunk_key] = nullptr;
    }

    // At this point, buffer is loaded into memory
    // Once cache is ready, should but put back into cache and returned
    // For now just keep the buffer and return it (won't be deleted again)
  }
  return getBufferFromMap(chunk_key);
}

void CsvDataWrapper::populateMetadataForChunkKeyPrefix(
    const ChunkKey& chunk_key_prefix,
    ChunkMetadataVector& chunk_metadata_vector) {
  auto timer = DEBUG_TIMER(__func__);
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
