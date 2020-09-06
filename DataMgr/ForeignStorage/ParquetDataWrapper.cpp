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

#include "ParquetDataWrapper.h"
#include "LazyParquetChunkLoader.h"
#include "ParquetShared.h"

#include <regex>

#include <boost/filesystem.hpp>

#include "ImportExport/Importer.h"
#include "Utils/DdlUtils.h"

namespace foreign_storage {

namespace {
template <typename T>
std::pair<typename std::map<ChunkKey, T>::iterator,
          typename std::map<ChunkKey, T>::iterator>
prefix_range(std::map<ChunkKey, T>& map, const ChunkKey& chunk_key_prefix) {
  ChunkKey chunk_key_prefix_sentinel = chunk_key_prefix;
  chunk_key_prefix_sentinel.push_back(std::numeric_limits<int>::max());
  auto begin = map.lower_bound(chunk_key_prefix);
  auto end = map.upper_bound(chunk_key_prefix_sentinel);
  return std::make_pair(begin, end);
}
}  // namespace

ParquetDataWrapper::ParquetDataWrapper(const int db_id, const ForeignTable* foreign_table)
    : db_id_(db_id)
    , foreign_table_(foreign_table)
    , last_fragment_index_(0)
    , last_fragment_row_count_(0)
    , last_row_group_(0)
    , schema_(std::make_unique<ForeignTableSchema>(db_id, foreign_table)) {}

ParquetDataWrapper::ParquetDataWrapper(const ForeignTable* foreign_table)
    : db_id_(-1), foreign_table_(foreign_table) {}

void ParquetDataWrapper::validateOptions(const ForeignTable* foreign_table) {
  for (const auto& entry : foreign_table->options) {
    const auto& table_options = foreign_table->supported_options;
    if (std::find(table_options.begin(), table_options.end(), entry.first) ==
            table_options.end() &&
        std::find(supported_options_.begin(), supported_options_.end(), entry.first) ==
            supported_options_.end()) {
      throw std::runtime_error{"Invalid foreign table option \"" + entry.first + "\"."};
    }
  }
  ParquetDataWrapper data_wrapper{foreign_table};
  data_wrapper.validateAndGetCopyParams();
  data_wrapper.validateFilePath();
}

void ParquetDataWrapper::validateFilePath() {
  ddl_utils::validate_allowed_file_path(getFilePath(),
                                        ddl_utils::DataTransferType::IMPORT);
}

void ParquetDataWrapper::resetParquetMetadata() {
  fragment_to_row_group_interval_map_.clear();
  fragment_to_row_group_interval_map_[0] = {0, -1};

  last_row_group_ = 0;
  last_fragment_index_ = 0;
  last_fragment_row_count_ = 0;
}

std::list<const ColumnDescriptor*> ParquetDataWrapper::getColumnsToInitialize(
    const Interval<ColumnType>& column_interval) {
  const auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);
  const auto& columns = schema_->getLogicalAndPhysicalColumns();
  auto column_start = column_interval.start;
  auto column_end = column_interval.end;
  std::list<const ColumnDescriptor*> columns_to_init;
  for (const auto column : columns) {
    auto column_id = column->columnId;
    if (column_id >= column_start && column_id <= column_end) {
      columns_to_init.push_back(column);
    }
  }
  return columns_to_init;
}

void ParquetDataWrapper::initializeChunkBuffers(
    const int fragment_index,
    const Interval<ColumnType>& column_interval,
    std::map<ChunkKey, AbstractBuffer*>& required_buffers,
    const bool reserve_buffers_and_set_stats,
    const size_t physical_byte_size) {
  for (const auto column : getColumnsToInitialize(column_interval)) {
    Chunk_NS::Chunk chunk{column};
    ChunkKey data_chunk_key;
    if (column->columnType.is_varlen_indeed()) {
      data_chunk_key = {
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 1};
      auto data_buffer = required_buffers[data_chunk_key];
      CHECK(data_buffer);
      chunk.setBuffer(data_buffer);

      ChunkKey index_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 2};
      auto index_buffer = required_buffers[index_chunk_key];
      CHECK(index_buffer);
      chunk.setIndexBuffer(index_buffer);
    } else {
      data_chunk_key = {
          db_id_, foreign_table_->tableId, column->columnId, fragment_index};
      auto data_buffer = required_buffers[data_chunk_key];
      CHECK(data_buffer);
      chunk.setBuffer(data_buffer);
    }
    chunk.initEncoder();
    if (reserve_buffers_and_set_stats) {
      const auto metadata_it = chunk_metadata_map_.find(data_chunk_key);
      CHECK(metadata_it != chunk_metadata_map_.end());
      auto buffer = chunk.getBuffer();
      auto& metadata = metadata_it->second;
      auto& encoder = buffer->encoder;
      encoder->resetChunkStats(metadata->chunkStats);
      encoder->setNumElems(metadata->numElements);
      size_t num_bytes_to_reserve = metadata->numElements * column->columnType.get_size();
      buffer->reserve(num_bytes_to_reserve);
    }
  }
}

void ParquetDataWrapper::finalizeFragmentMap() {
  fragment_to_row_group_interval_map_[last_fragment_index_].end_row_group_index =
      last_row_group_;
}

void ParquetDataWrapper::updateFragmentMap(int fragment_index, int row_group) {
  CHECK(fragment_index > 0);
  fragment_to_row_group_interval_map_[fragment_index - 1].end_row_group_index =
      row_group - 1;
  fragment_to_row_group_interval_map_[fragment_index] = {row_group, -1};
}

void ParquetDataWrapper::fetchChunkMetadata() {
  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  resetParquetMetadata();
  LazyParquetImporter::RowGroupMetadataVector metadata_vector;
  LazyParquetImporter importer(getMetadataLoader(*catalog, metadata_vector),
                               getFilePath(),
                               validateAndGetCopyParams(),
                               metadata_vector,
                               *schema_);
  importer.metadataScan();
  finalizeFragmentMap();
}

std::string ParquetDataWrapper::getFilePath() {
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

import_export::CopyParams ParquetDataWrapper::validateAndGetCopyParams() {
  import_export::CopyParams copy_params{};
  if (const auto& value = validateAndGetStringWithLength("ARRAY_DELIMITER", 1);
      !value.empty()) {
    copy_params.array_delim = value[0];
  }
  if (const auto& value = validateAndGetStringWithLength("ARRAY_MARKER", 2);
      !value.empty()) {
    copy_params.array_begin = value[0];
    copy_params.array_end = value[1];
  }
  // The file_type argument is never utilized in the context of FSI,
  // for completeness, set the file_type
  copy_params.file_type = import_export::FileType::PARQUET;
  return copy_params;
}

std::string ParquetDataWrapper::validateAndGetStringWithLength(
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

void ParquetDataWrapper::updateStatsForEncoder(Encoder* encoder,
                                               const SQLTypeInfo type_info,
                                               const DataBlockPtr& data_block,
                                               const size_t import_count) {
  CHECK(encoder);
  if (type_info.is_varlen()) {
    switch (type_info.get_type()) {
      case kARRAY: {
        encoder->updateStats(data_block.arraysPtr, 0, import_count);
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON: {
        encoder->updateStats(data_block.stringsPtr, 0, import_count);
        break;
      }
      default:
        UNREACHABLE();
    }
  } else {
    encoder->updateStats(data_block.numbersPtr, import_count);
  }
  encoder->setNumElems(encoder->getNumElems() + import_count);
}

void ParquetDataWrapper::loadMetadataChunk(const ColumnDescriptor* column,
                                           const ChunkKey& chunk_key,
                                           DataBlockPtr& data_block,
                                           const size_t import_count,
                                           const bool has_nulls,
                                           const bool is_all_nulls) {
  auto type_info = column->columnType;
  ChunkKey data_chunk_key = chunk_key;
  if (type_info.is_varlen_indeed()) {
    data_chunk_key.emplace_back(1);
  }
  ForeignStorageBuffer buffer;
  buffer.initEncoder(type_info);
  auto encoder = buffer.encoder.get();
  if (chunk_metadata_map_.find(data_chunk_key) != chunk_metadata_map_.end()) {
    encoder->resetChunkStats(chunk_metadata_map_[data_chunk_key]->chunkStats);
    encoder->setNumElems(chunk_metadata_map_[data_chunk_key]->numElements);
    buffer.setSize(chunk_metadata_map_[data_chunk_key]->numBytes);
  } else {
    chunk_metadata_map_[data_chunk_key] = std::make_shared<ChunkMetadata>();
  }

  auto logical_type_info =
      schema_->getLogicalColumn(chunk_key[CHUNK_KEY_COLUMN_IDX])->columnType;
  if (is_all_nulls || logical_type_info.is_string() || logical_type_info.is_varlen()) {
    // Do not attempt to load min/max statistics if entire row group is null or
    // if the column is a string or variable length column
    encoder->setNumElems(encoder->getNumElems() + import_count);
  } else {
    // Loads min/max statistics for columns with this information
    updateStatsForEncoder(encoder, type_info, data_block, 2);
    encoder->setNumElems(encoder->getNumElems() + import_count - 2);
  }
  encoder->getMetadata(chunk_metadata_map_[data_chunk_key]);
  chunk_metadata_map_[data_chunk_key]->chunkStats.has_nulls |= has_nulls;
}

void ParquetDataWrapper::loadChunk(
    const ColumnDescriptor* column,
    const ChunkKey& chunk_key,
    DataBlockPtr& data_block,
    const size_t import_count,
    std::map<ChunkKey, AbstractBuffer*>& required_buffers) {
  Chunk_NS::Chunk chunk{column};
  auto column_id = column->columnId;
  CHECK(column_id == chunk_key[CHUNK_KEY_COLUMN_IDX]);
  auto& type_info = column->columnType;
  if (type_info.is_varlen_indeed()) {
    ChunkKey data_chunk_key{chunk_key};
    data_chunk_key.resize(5);
    data_chunk_key[4] = 1;
    CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
    auto& data_buffer = required_buffers[data_chunk_key];
    chunk.setBuffer(data_buffer);

    ChunkKey index_chunk_key{chunk_key};
    index_chunk_key.resize(5);
    index_chunk_key[4] = 2;
    CHECK(required_buffers.find(index_chunk_key) != required_buffers.end());
    auto& index_buffer = required_buffers[index_chunk_key];
    chunk.setIndexBuffer(index_buffer);
  } else {
    CHECK(required_buffers.find(chunk_key) != required_buffers.end());
    auto& buffer = required_buffers[chunk_key];
    chunk.setBuffer(buffer);
  }
  chunk.appendData(data_block, import_count, 0);
  chunk.setBuffer(nullptr);
  chunk.setIndexBuffer(nullptr);
}

import_export::Loader* ParquetDataWrapper::getChunkLoader(
    Catalog_Namespace::Catalog& catalog,
    const Interval<ColumnType>& column_interval,
    const int db_id,
    const int fragment_index,
    std::map<ChunkKey, AbstractBuffer*>& required_buffers) {
  auto callback =
      [this, column_interval, db_id, fragment_index, &required_buffers](
          const std::vector<std::unique_ptr<import_export::TypedImportBuffer>>&
              import_buffers,
          std::vector<DataBlockPtr>& data_blocks,
          size_t import_row_count) {
        for (int column_id = column_interval.start; column_id <= column_interval.end;
             column_id++) {
          // Column ids start at 1, hence the -1 offset
          auto& import_buffer = import_buffers[column_id - 1];
          ChunkKey chunk_key{db_id, foreign_table_->tableId, column_id, fragment_index};
          loadChunk(import_buffer->getColumnDesc(),
                    chunk_key,
                    data_blocks[column_id - 1],
                    import_row_count,
                    required_buffers);
        }
        return true;
      };

  return new import_export::Loader(catalog, foreign_table_, callback, false);
}

import_export::Loader* ParquetDataWrapper::getMetadataLoader(
    Catalog_Namespace::Catalog& catalog,
    const LazyParquetImporter::RowGroupMetadataVector& metadata_vector) {
  auto callback =
      [this, &metadata_vector](
          const std::vector<std::unique_ptr<import_export::TypedImportBuffer>>&
              import_buffers,
          std::vector<DataBlockPtr>& data_blocks,
          size_t import_row_count) {
        int row_group = metadata_vector[0].row_group_index;
        last_row_group_ = row_group;
        if (moveToNextFragment(import_row_count)) {
          last_fragment_index_++;
          last_fragment_row_count_ = 0;
          updateFragmentMap(last_fragment_index_, row_group);
        }

        for (size_t i = 0; i < import_buffers.size(); i++) {
          auto& import_buffer = import_buffers[i];
          const auto column = import_buffer->getColumnDesc();
          auto column_id = column->columnId;
          ChunkKey chunk_key{
              db_id_, foreign_table_->tableId, column_id, last_fragment_index_};
          const auto& metadata = metadata_vector[i];
          CHECK(metadata.row_group_index == row_group);
          CHECK(metadata.metadata_only);
          loadMetadataChunk(column,
                            chunk_key,
                            data_blocks[i],
                            metadata.num_elements,
                            metadata.has_nulls,
                            metadata.is_all_nulls);
        }

        last_fragment_row_count_ += import_row_count;
        return true;
      };

  return new import_export::Loader(catalog, foreign_table_, callback, false);
}

bool ParquetDataWrapper::moveToNextFragment(size_t new_rows_count) {
  return (last_fragment_row_count_ + new_rows_count) >
         static_cast<size_t>(foreign_table_->maxFragRows);
}

void ParquetDataWrapper::populateChunkMetadata(
    ChunkMetadataVector& chunk_metadata_vector) {
  chunk_metadata_map_.clear();
  fetchChunkMetadata();
  for (const auto& [chunk_key, chunk_metadata] : chunk_metadata_map_) {
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
  }
}

void ParquetDataWrapper::loadBuffersUsingLazyParquetChunkLoader(
    const int logical_column_id,
    const int fragment_id,
    const size_t physical_byte_size,
    std::map<ChunkKey, AbstractBuffer*>& required_buffers) {
  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  const ColumnDescriptor* logical_column =
      schema_->getColumnDescriptor(logical_column_id);
  auto parquet_column_index = schema_->getParquetColumnIndex(logical_column_id);

  const Interval<ColumnType> column_interval = {logical_column_id, logical_column_id};
  initializeChunkBuffers(
      fragment_id, column_interval, required_buffers, true, physical_byte_size);

  const auto& row_group_interval = fragment_to_row_group_interval_map_[fragment_id];
  LazyParquetChunkLoader chunk_loader(getFilePath());
  Chunk_NS::Chunk chunk{logical_column};
  ChunkKey chunk_key = {db_id_, foreign_table_->tableId, logical_column_id, fragment_id};
  auto buffer = required_buffers[chunk_key];
  CHECK(buffer);
  chunk.setBuffer(buffer);
  chunk_loader.loadChunk(
      {row_group_interval.start_row_group_index, row_group_interval.end_row_group_index},
      parquet_column_index,
      chunk);
}

void ParquetDataWrapper::loadBuffersUsingLazyParquetImporter(
    const int logical_column_id,
    const int fragment_id,
    std::map<ChunkKey, AbstractBuffer*>& required_buffers) {
  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  const ColumnDescriptor* logical_column =
      schema_->getColumnDescriptor(logical_column_id);
  const Interval<ColumnType> column_interval = {
      logical_column->columnId,
      logical_column->columnId + logical_column->columnType.get_physical_cols()};
  initializeChunkBuffers(fragment_id, column_interval, required_buffers);

  LazyParquetImporter::RowGroupMetadataVector metadata_vector;
  LazyParquetImporter importer(
      getChunkLoader(*catalog, column_interval, db_id_, fragment_id, required_buffers),
      getFilePath(),
      validateAndGetCopyParams(),
      metadata_vector,
      *schema_);
  const auto& row_group_interval = fragment_to_row_group_interval_map_[fragment_id];
  importer.partialImport(
      {row_group_interval.start_row_group_index, row_group_interval.end_row_group_index},
      column_interval);
}

void ParquetDataWrapper::populateChunkBuffers(
    std::map<ChunkKey, AbstractBuffer*>& required_buffers,
    std::map<ChunkKey, AbstractBuffer*>& optional_buffers) {
  std::unique_ptr<parquet::arrow::FileReader> reader;
  open_parquet_table(getFilePath(), reader);

  CHECK(!required_buffers.empty());
  auto fragment_id = required_buffers.begin()->first[CHUNK_KEY_FRAGMENT_IDX];

  std::set<int> logical_column_ids;
  for (const auto& [chunk_key, buffer] : required_buffers) {
    CHECK_EQ(fragment_id, chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
    CHECK_EQ(buffer->size(), static_cast<size_t>(0));
    const auto column_id =
        schema_->getLogicalColumn(chunk_key[CHUNK_KEY_COLUMN_IDX])->columnId;
    logical_column_ids.emplace(column_id);
  }

  for (const auto column_id : logical_column_ids) {
    const ColumnDescriptor* column_descriptor = schema_->getColumnDescriptor(column_id);
    auto parquet_column_index = schema_->getParquetColumnIndex(column_id);
    if (LazyParquetChunkLoader::isColumnMappingSupported(
            column_descriptor, get_column_descriptor(reader, parquet_column_index))) {
      auto physical_byte_size = get_physical_type_byte_size(reader, parquet_column_index);
      loadBuffersUsingLazyParquetChunkLoader(
          column_id, fragment_id, physical_byte_size, required_buffers);
    } else {
      loadBuffersUsingLazyParquetImporter(column_id, fragment_id, required_buffers);
    }
  }
}

}  // namespace foreign_storage
