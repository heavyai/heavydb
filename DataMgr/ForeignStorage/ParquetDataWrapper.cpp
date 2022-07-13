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

#include <queue>

#include <arrow/filesystem/localfs.h>
#include <boost/filesystem.hpp>

#include "Catalog/Catalog.h"
#include "ForeignStorageException.h"
#include "FsiChunkUtils.h"
#include "FsiJsonUtils.h"
#include "ImportExport/RenderGroupAnalyzer.h"
#include "LazyParquetChunkLoader.h"
#include "ParquetShared.h"
#include "Shared/SysDefinitions.h"
#include "Shared/file_path_util.h"
#include "Shared/misc.h"
#include "Utils/DdlUtils.h"

namespace foreign_storage {

namespace {
void reduce_metadata(std::shared_ptr<ChunkMetadata> reduce_to,
                     std::shared_ptr<ChunkMetadata> reduce_from) {
  CHECK(reduce_to->sqlType == reduce_from->sqlType);
  reduce_to->numBytes += reduce_from->numBytes;
  reduce_to->numElements += reduce_from->numElements;
  reduce_to->chunkStats.has_nulls |= reduce_from->chunkStats.has_nulls;

  auto column_type = reduce_to->sqlType;
  column_type = column_type.is_array() ? column_type.get_elem_type() : column_type;

  // metadata reducution is done at metadata scan time, both string & geometry
  // columns have no valid stats to reduce beyond `has_nulls`
  if (column_type.is_string() || column_type.is_geometry()) {
    // Reset to invalid range, as formerly valid metadata
    // needs to be invalidated during an append for these types
    reduce_to->chunkStats.max = reduce_from->chunkStats.max;
    reduce_to->chunkStats.min = reduce_from->chunkStats.min;
    return;
  }

  ForeignStorageBuffer buffer_to;
  buffer_to.initEncoder(column_type);
  auto encoder_to = buffer_to.getEncoder();
  encoder_to->resetChunkStats(reduce_to->chunkStats);

  ForeignStorageBuffer buffer_from;
  buffer_from.initEncoder(column_type);
  auto encoder_from = buffer_from.getEncoder();
  encoder_from->resetChunkStats(reduce_from->chunkStats);

  encoder_to->reduceStats(*encoder_from);
  auto updated_metadata = std::make_shared<ChunkMetadata>();
  encoder_to->getMetadata(updated_metadata);
  reduce_to->chunkStats = updated_metadata->chunkStats;
}
}  // namespace

ParquetDataWrapper::ParquetDataWrapper()
    : do_metadata_stats_validation_(true), db_id_(-1), foreign_table_(nullptr) {}

ParquetDataWrapper::ParquetDataWrapper(const ForeignTable* foreign_table,
                                       std::shared_ptr<arrow::fs::FileSystem> file_system)
    : do_metadata_stats_validation_(false)
    , db_id_(-1)
    , foreign_table_(foreign_table)
    , last_fragment_index_(0)
    , last_fragment_row_count_(0)
    , total_row_count_(0)
    , last_file_row_count_(0)
    , last_row_group_(0)
    , is_restored_(false)
    , file_system_(file_system)
    , file_reader_cache_(std::make_unique<FileReaderMap>()) {}

ParquetDataWrapper::ParquetDataWrapper(const int db_id,
                                       const ForeignTable* foreign_table,
                                       const bool do_metadata_stats_validation)
    : do_metadata_stats_validation_(do_metadata_stats_validation)
    , db_id_(db_id)
    , foreign_table_(foreign_table)
    , last_fragment_index_(0)
    , last_fragment_row_count_(0)
    , total_row_count_(0)
    , last_file_row_count_(0)
    , last_row_group_(0)
    , is_restored_(false)
    , schema_(std::make_unique<ForeignTableSchema>(db_id, foreign_table))
    , file_reader_cache_(std::make_unique<FileReaderMap>()) {
  auto& server_options = foreign_table->foreign_server->options;
  if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
    file_system_ = std::make_shared<arrow::fs::LocalFileSystem>();
  } else {
    UNREACHABLE();
  }
}

void ParquetDataWrapper::resetParquetMetadata() {
  fragment_to_row_group_interval_map_.clear();
  fragment_to_row_group_interval_map_[0] = {};

  last_row_group_ = 0;
  last_fragment_index_ = 0;
  last_fragment_row_count_ = 0;
  total_row_count_ = 0;
  last_file_row_count_ = 0;
  file_reader_cache_->clear();
}

std::list<const ColumnDescriptor*> ParquetDataWrapper::getColumnsToInitialize(
    const Interval<ColumnType>& column_interval) {
  const auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
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
    const ChunkToBufferMap& required_buffers,
    const bool reserve_buffers_and_set_stats) {
  for (const auto column : getColumnsToInitialize(column_interval)) {
    Chunk_NS::Chunk chunk{column, false};
    ChunkKey data_chunk_key;
    if (column->columnType.is_varlen_indeed()) {
      data_chunk_key = {
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 1};
      CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
      auto data_buffer = shared::get_from_map(required_buffers, data_chunk_key);
      chunk.setBuffer(data_buffer);

      ChunkKey index_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 2};
      CHECK(required_buffers.find(index_chunk_key) != required_buffers.end());
      auto index_buffer = shared::get_from_map(required_buffers, index_chunk_key);
      chunk.setIndexBuffer(index_buffer);
    } else {
      data_chunk_key = {
          db_id_, foreign_table_->tableId, column->columnId, fragment_index};
      CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
      auto data_buffer = shared::get_from_map(required_buffers, data_chunk_key);
      chunk.setBuffer(data_buffer);
    }
    chunk.initEncoder();
    if (reserve_buffers_and_set_stats) {
      const auto metadata_it = chunk_metadata_map_.find(data_chunk_key);
      CHECK(metadata_it != chunk_metadata_map_.end());
      auto buffer = chunk.getBuffer();
      auto& metadata = metadata_it->second;
      auto encoder = buffer->getEncoder();
      encoder->resetChunkStats(metadata->chunkStats);
      encoder->setNumElems(metadata->numElements);
      if ((column->columnType.is_string() &&
           column->columnType.get_compression() == kENCODING_NONE) ||
          column->columnType.is_geometry()) {
        // non-dictionary string or geometry WKT string
        auto index_buffer = chunk.getIndexBuf();
        index_buffer->reserve(sizeof(StringOffsetT) * (metadata->numElements + 1));
      } else if (!column->columnType.is_fixlen_array() && column->columnType.is_array()) {
        auto index_buffer = chunk.getIndexBuf();
        index_buffer->reserve(sizeof(ArrayOffsetT) * (metadata->numElements + 1));
      } else {
        size_t num_bytes_to_reserve =
            metadata->numElements * column->columnType.get_size();
        buffer->reserve(num_bytes_to_reserve);
      }
    }
  }
}

void ParquetDataWrapper::finalizeFragmentMap() {
  fragment_to_row_group_interval_map_[last_fragment_index_].back().end_index =
      last_row_group_;
}

void ParquetDataWrapper::addNewFragment(int row_group, const std::string& file_path) {
  const auto last_fragment_entry =
      fragment_to_row_group_interval_map_.find(last_fragment_index_);
  CHECK(last_fragment_entry != fragment_to_row_group_interval_map_.end());

  last_fragment_entry->second.back().end_index = last_row_group_;
  last_fragment_index_++;
  last_fragment_row_count_ = 0;
  fragment_to_row_group_interval_map_[last_fragment_index_].emplace_back(
      RowGroupInterval{file_path, row_group});
  setLastFileRowCount(file_path);
}

bool ParquetDataWrapper::isNewFile(const std::string& file_path) const {
  const auto last_fragment_entry =
      fragment_to_row_group_interval_map_.find(last_fragment_index_);
  CHECK(last_fragment_entry != fragment_to_row_group_interval_map_.end());

  // The entry for the first fragment starts out as an empty vector
  if (last_fragment_entry->second.empty()) {
    // File roll off can result in empty older fragments.
    if (!allowFileRollOff(foreign_table_)) {
      CHECK_EQ(last_fragment_index_, 0);
    }
    return true;
  } else {
    return (last_fragment_entry->second.back().file_path != file_path);
  }
}

void ParquetDataWrapper::addNewFile(const std::string& file_path) {
  const auto last_fragment_entry =
      fragment_to_row_group_interval_map_.find(last_fragment_index_);
  CHECK(last_fragment_entry != fragment_to_row_group_interval_map_.end());

  // The entry for the first fragment starts out as an empty vector
  if (last_fragment_entry->second.empty()) {
    // File roll off can result in empty older fragments.
    if (!allowFileRollOff(foreign_table_)) {
      CHECK_EQ(last_fragment_index_, 0);
    }
  } else {
    last_fragment_entry->second.back().end_index = last_row_group_;
  }
  last_fragment_entry->second.emplace_back(RowGroupInterval{file_path, 0});
  setLastFileRowCount(file_path);
}

void ParquetDataWrapper::setLastFileRowCount(const std::string& file_path) {
  auto reader = file_reader_cache_->getOrInsert(file_path, file_system_);
  last_file_row_count_ = reader->parquet_reader()->metadata()->num_rows();
}

void ParquetDataWrapper::fetchChunkMetadata() {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  std::vector<std::string> new_file_paths;
  auto processed_file_paths = getOrderedProcessedFilePaths();
  if (foreign_table_->isAppendMode() && !processed_file_paths.empty()) {
    auto all_file_paths = getAllFilePaths();
    if (allowFileRollOff(foreign_table_)) {
      const auto rolled_off_files =
          shared::check_for_rolled_off_file_paths(all_file_paths, processed_file_paths);
      updateMetadataForRolledOffFiles(rolled_off_files);
    }

    for (const auto& file_path : processed_file_paths) {
      if (!shared::contains(all_file_paths, file_path)) {
        throw_removed_file_error(file_path);
      }
    }

    // For multi-file appends, reprocess the last file in order to account for appends
    // that may have occurred to this file. For single file appends, reprocess file if new
    // rows have been added.
    if (!processed_file_paths.empty()) {
      // Single file append
      if (all_file_paths.size() == 1) {
        CHECK_EQ(processed_file_paths.size(), size_t(1));
        CHECK_EQ(processed_file_paths[0], all_file_paths[0]);
      }

      const auto& last_file_path = processed_file_paths.back();
      // Since an existing file is being appended to we need to update the cached
      // FileReader as the existing one will be out of date.
      auto reader = file_reader_cache_->insert(last_file_path, file_system_);
      size_t row_count = reader->parquet_reader()->metadata()->num_rows();
      if (row_count < last_file_row_count_) {
        throw_removed_row_in_file_error(last_file_path);
      } else if (row_count > last_file_row_count_) {
        removeMetadataForLastFile(last_file_path);
        new_file_paths.emplace_back(last_file_path);
      }
    }

    for (const auto& file_path : all_file_paths) {
      if (!shared::contains(processed_file_paths, file_path)) {
        new_file_paths.emplace_back(file_path);
      }
    }
  } else {
    CHECK(chunk_metadata_map_.empty());
    new_file_paths = getAllFilePaths();
    resetParquetMetadata();
  }

  if (!new_file_paths.empty()) {
    metadataScanFiles(new_file_paths);
  }
}

void ParquetDataWrapper::updateMetadataForRolledOffFiles(
    const std::set<std::string>& rolled_off_files) {
  if (!rolled_off_files.empty()) {
    std::set<int32_t> deleted_fragment_ids;
    std::optional<int32_t> partially_deleted_fragment_id;
    std::vector<std::string> remaining_files_in_partially_deleted_fragment;
    for (auto& [fragment_id, row_group_interval_vec] :
         fragment_to_row_group_interval_map_) {
      for (auto it = row_group_interval_vec.begin();
           it != row_group_interval_vec.end();) {
        if (shared::contains(rolled_off_files, it->file_path)) {
          it = row_group_interval_vec.erase(it);
        } else {
          remaining_files_in_partially_deleted_fragment.emplace_back(it->file_path);
          it++;
        }
      }
      if (row_group_interval_vec.empty()) {
        deleted_fragment_ids.emplace(fragment_id);
      } else {
        CHECK(!remaining_files_in_partially_deleted_fragment.empty());
        partially_deleted_fragment_id = fragment_id;
        break;
      }
    }

    for (auto it = chunk_metadata_map_.begin(); it != chunk_metadata_map_.end();) {
      const auto& chunk_key = it->first;
      if (shared::contains(deleted_fragment_ids, chunk_key[CHUNK_KEY_FRAGMENT_IDX])) {
        auto& chunk_metadata = it->second;
        chunk_metadata->numElements = 0;
        chunk_metadata->numBytes = 0;
        it++;
      } else if (partially_deleted_fragment_id.has_value() &&
                 chunk_key[CHUNK_KEY_FRAGMENT_IDX] == partially_deleted_fragment_id) {
        // Metadata for the partially deleted fragment will be re-populated.
        it = chunk_metadata_map_.erase(it);
      } else {
        it++;
      }
    }

    if (partially_deleted_fragment_id.has_value()) {
      // Create map of row group to row group metadata for remaining files in the
      // fragment.
      auto row_group_metadata_map =
          getRowGroupMetadataMap(remaining_files_in_partially_deleted_fragment);

      // Re-populate metadata for remaining row groups in partially deleted fragment.
      auto column_interval =
          Interval<ColumnType>{schema_->getLogicalAndPhysicalColumns().front()->columnId,
                               schema_->getLogicalAndPhysicalColumns().back()->columnId};
      auto row_group_intervals = shared::get_from_map(
          fragment_to_row_group_interval_map_, partially_deleted_fragment_id.value());
      for (const auto& row_group_interval : row_group_intervals) {
        for (auto row_group = row_group_interval.start_index;
             row_group <= row_group_interval.end_index;
             row_group++) {
          const auto& row_group_metadata_item = shared::get_from_map(
              row_group_metadata_map, {row_group_interval.file_path, row_group});
          updateChunkMetadataForFragment(column_interval,
                                         row_group_metadata_item.column_chunk_metadata,
                                         partially_deleted_fragment_id.value());
        }
      }
    }
  }
}

std::vector<std::string> ParquetDataWrapper::getOrderedProcessedFilePaths() {
  std::vector<std::string> file_paths;
  for (const auto& entry : fragment_to_row_group_interval_map_) {
    for (const auto& row_group_interval : entry.second) {
      if (file_paths.empty() || file_paths.back() != row_group_interval.file_path) {
        file_paths.emplace_back(row_group_interval.file_path);
      }
    }
  }
  return file_paths;
}

std::vector<std::string> ParquetDataWrapper::getAllFilePaths() {
  auto timer = DEBUG_TIMER(__func__);
  std::vector<std::string> found_file_paths;
  auto file_path = getFullFilePath(foreign_table_);
  const auto file_path_options = getFilePathOptions(foreign_table_);
  auto& server_options = foreign_table_->foreign_server->options;
  if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
    found_file_paths = shared::local_glob_filter_sort_files(file_path, file_path_options);
  } else {
    UNREACHABLE();
  }
  return found_file_paths;
}

void ParquetDataWrapper::metadataScanFiles(const std::vector<std::string>& file_paths) {
  auto row_group_metadata = getRowGroupMetadataForFilePaths(file_paths);
  metadataScanRowGroupMetadata(row_group_metadata);
}

void ParquetDataWrapper::metadataScanRowGroupMetadata(
    const std::list<RowGroupMetadata>& row_group_metadata) {
  auto column_interval =
      Interval<ColumnType>{schema_->getLogicalAndPhysicalColumns().front()->columnId,
                           schema_->getLogicalAndPhysicalColumns().back()->columnId};
  for (const auto& row_group_metadata_item : row_group_metadata) {
    const auto& column_chunk_metadata = row_group_metadata_item.column_chunk_metadata;
    auto column_chunk_metadata_iter = column_chunk_metadata.begin();
    const auto import_row_count = (*column_chunk_metadata_iter)->numElements;
    auto row_group = row_group_metadata_item.row_group_index;
    const auto& file_path = row_group_metadata_item.file_path;
    if (moveToNextFragment(import_row_count)) {
      addNewFragment(row_group, file_path);
    } else if (isNewFile(file_path)) {
      CHECK_EQ(row_group, 0);
      addNewFile(file_path);
    }
    last_row_group_ = row_group;
    updateChunkMetadataForFragment(
        column_interval, column_chunk_metadata, last_fragment_index_);

    last_fragment_row_count_ += import_row_count;
    total_row_count_ += import_row_count;
  }
  finalizeFragmentMap();
}

std::list<RowGroupMetadata> ParquetDataWrapper::getRowGroupMetadataForFilePaths(
    const std::vector<std::string>& file_paths) const {
  LazyParquetChunkLoader chunk_loader(
      file_system_, file_reader_cache_.get(), nullptr, foreign_table_->tableName);
  return chunk_loader.metadataScan(file_paths, *schema_, do_metadata_stats_validation_);
}

void ParquetDataWrapper::updateChunkMetadataForFragment(
    const Interval<ColumnType>& column_interval,
    const std::list<std::shared_ptr<ChunkMetadata>>& column_chunk_metadata,
    int32_t fragment_id) {
  CHECK_EQ(static_cast<int>(column_chunk_metadata.size()),
           schema_->numLogicalAndPhysicalColumns());
  auto column_chunk_metadata_iter = column_chunk_metadata.begin();
  for (auto column_id = column_interval.start; column_id <= column_interval.end;
       column_id++, column_chunk_metadata_iter++) {
    CHECK(column_chunk_metadata_iter != column_chunk_metadata.end());
    const auto column_descriptor = schema_->getColumnDescriptor(column_id);
    const auto& type_info = column_descriptor->columnType;
    ChunkKey const data_chunk_key =
        type_info.is_varlen_indeed()
            ? ChunkKey{db_id_, foreign_table_->tableId, column_id, fragment_id, 1}
            : ChunkKey{db_id_, foreign_table_->tableId, column_id, fragment_id};
    std::shared_ptr<ChunkMetadata> chunk_metadata = *column_chunk_metadata_iter;

    if (chunk_metadata_map_.find(data_chunk_key) == chunk_metadata_map_.end()) {
      chunk_metadata_map_[data_chunk_key] = chunk_metadata;
    } else {
      reduce_metadata(chunk_metadata_map_[data_chunk_key], chunk_metadata);
    }
  }
}

bool ParquetDataWrapper::moveToNextFragment(size_t new_rows_count) const {
  return (last_fragment_row_count_ + new_rows_count) >
         static_cast<size_t>(foreign_table_->maxFragRows);
}

void ParquetDataWrapper::populateChunkMetadata(
    ChunkMetadataVector& chunk_metadata_vector) {
  fetchChunkMetadata();
  for (const auto& [chunk_key, chunk_metadata] : chunk_metadata_map_) {
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
  }
}

void ParquetDataWrapper::loadBuffersUsingLazyParquetChunkLoader(
    const int logical_column_id,
    const int fragment_id,
    const ChunkToBufferMap& required_buffers,
    AbstractBuffer* delete_buffer) {
  const auto& row_group_intervals =
      shared::get_from_map(fragment_to_row_group_interval_map_, fragment_id);
  // File roll off can lead to an empty row group interval vector.
  if (row_group_intervals.empty()) {
    return;
  }

  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  const ColumnDescriptor* logical_column =
      schema_->getColumnDescriptor(logical_column_id);
  auto parquet_column_index = schema_->getParquetColumnIndex(logical_column_id);

  const Interval<ColumnType> column_interval = {
      logical_column_id,
      logical_column_id + logical_column->columnType.get_physical_cols()};
  initializeChunkBuffers(fragment_id, column_interval, required_buffers, true);

  const bool is_dictionary_encoded_string_column =
      logical_column->columnType.is_dict_encoded_string() ||
      (logical_column->columnType.is_array() &&
       logical_column->columnType.get_elem_type().is_dict_encoded_string());

  StringDictionary* string_dictionary = nullptr;
  if (is_dictionary_encoded_string_column) {
    auto dict_descriptor =
        catalog->getMetadataForDict(logical_column->columnType.get_comp_param(), true);
    CHECK(dict_descriptor);
    string_dictionary = dict_descriptor->stringDict.get();
  }

  std::list<Chunk_NS::Chunk> chunks;
  for (int column_id = column_interval.start; column_id <= column_interval.end;
       ++column_id) {
    auto column_descriptor = schema_->getColumnDescriptor(column_id);
    Chunk_NS::Chunk chunk{column_descriptor, false};
    if (column_descriptor->columnType.is_varlen_indeed()) {
      ChunkKey data_chunk_key = {
          db_id_, foreign_table_->tableId, column_id, fragment_id, 1};
      auto buffer = shared::get_from_map(required_buffers, data_chunk_key);
      chunk.setBuffer(buffer);
      ChunkKey index_chunk_key = {
          db_id_, foreign_table_->tableId, column_id, fragment_id, 2};
      auto index_buffer = shared::get_from_map(required_buffers, index_chunk_key);
      chunk.setIndexBuffer(index_buffer);
    } else {
      ChunkKey chunk_key = {db_id_, foreign_table_->tableId, column_id, fragment_id};
      auto buffer = shared::get_from_map(required_buffers, chunk_key);
      chunk.setBuffer(buffer);
    }
    chunks.emplace_back(chunk);
  }

  std::unique_ptr<RejectedRowIndices> rejected_row_indices;
  if (delete_buffer) {
    rejected_row_indices = std::make_unique<RejectedRowIndices>();
  }
  LazyParquetChunkLoader chunk_loader(file_system_,
                                      file_reader_cache_.get(),
                                      &render_group_analyzer_map_,
                                      foreign_table_->tableName);
  auto metadata = chunk_loader.loadChunk(row_group_intervals,
                                         parquet_column_index,
                                         chunks,
                                         string_dictionary,
                                         rejected_row_indices.get());

  if (delete_buffer) {
    // all modifying operations on `delete_buffer` must be synchronized as it is a
    // shared buffer
    std::unique_lock<std::mutex> delete_buffer_lock(delete_buffer_mutex_);

    CHECK(!chunks.empty());
    CHECK(chunks.begin()->getBuffer()->hasEncoder());
    auto num_rows_in_chunk = chunks.begin()->getBuffer()->getEncoder()->getNumElems();

    // ensure delete buffer is sized appropriately
    if (delete_buffer->size() < num_rows_in_chunk) {
      auto remaining_rows = num_rows_in_chunk - delete_buffer->size();
      std::vector<int8_t> data(remaining_rows, false);
      delete_buffer->append(data.data(), remaining_rows);
    }

    // compute a logical OR with current `delete_buffer` contents and this chunks
    // rejected indices
    CHECK(rejected_row_indices);
    auto delete_buffer_data = delete_buffer->getMemoryPtr();
    for (const auto& rejected_index : *rejected_row_indices) {
      CHECK_GT(delete_buffer->size(), static_cast<size_t>(rejected_index));
      delete_buffer_data[rejected_index] = true;
    }
  }

  auto metadata_iter = metadata.begin();
  for (int column_id = column_interval.start; column_id <= column_interval.end;
       ++column_id, ++metadata_iter) {
    auto column = schema_->getColumnDescriptor(column_id);
    ChunkKey data_chunk_key = {db_id_, foreign_table_->tableId, column_id, fragment_id};
    if (column->columnType.is_varlen_indeed()) {
      data_chunk_key.emplace_back(1);
    }
    CHECK(chunk_metadata_map_.find(data_chunk_key) != chunk_metadata_map_.end());

    // Allocate new shared_ptr for metadata so we dont modify old one which may be used
    // by executor
    auto cached_metadata_previous =
        shared::get_from_map(chunk_metadata_map_, data_chunk_key);
    shared::get_from_map(chunk_metadata_map_, data_chunk_key) =
        std::make_shared<ChunkMetadata>();
    auto cached_metadata = shared::get_from_map(chunk_metadata_map_, data_chunk_key);
    *cached_metadata = *cached_metadata_previous;

    CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
    cached_metadata->numBytes =
        shared::get_from_map(required_buffers, data_chunk_key)->size();

    // for certain types, update the metadata statistics
    // should update the cache, and the internal chunk_metadata_map_
    if (is_dictionary_encoded_string_column || logical_column->columnType.is_geometry()) {
      CHECK(metadata_iter != metadata.end());
      cached_metadata->chunkStats = (*metadata_iter)->chunkStats;

      // Update stats on buffer so it is saved in cache
      shared::get_from_map(required_buffers, data_chunk_key)
          ->getEncoder()
          ->resetChunkStats(cached_metadata->chunkStats);
    }
  }
}

void ParquetDataWrapper::populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                                              const ChunkToBufferMap& optional_buffers,
                                              AbstractBuffer* delete_buffer) {
  ChunkToBufferMap buffers_to_load;
  buffers_to_load.insert(required_buffers.begin(), required_buffers.end());
  buffers_to_load.insert(optional_buffers.begin(), optional_buffers.end());

  CHECK(!buffers_to_load.empty());

  std::set<ForeignStorageMgr::ParallelismHint> col_frag_hints;
  for (const auto& [chunk_key, buffer] : buffers_to_load) {
    CHECK_EQ(buffer->size(), static_cast<size_t>(0));
    col_frag_hints.emplace(
        schema_->getLogicalColumn(chunk_key[CHUNK_KEY_COLUMN_IDX])->columnId,
        chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
  }

  std::function<void(const std::set<ForeignStorageMgr::ParallelismHint>&)> lambda =
      [&, this](const std::set<ForeignStorageMgr::ParallelismHint>& hint_set) {
        for (const auto& [col_id, frag_id] : hint_set) {
          loadBuffersUsingLazyParquetChunkLoader(
              col_id, frag_id, buffers_to_load, delete_buffer);
        }
      };

  CHECK(foreign_table_);
  auto num_threads = foreign_storage::get_num_threads(*foreign_table_);
  auto futures = create_futures_for_workers(col_frag_hints, num_threads, lambda);

  // We wait on all futures, then call get because we want all threads to have finished
  // before we propagate a potential exception.
  for (auto& future : futures) {
    future.wait();
  }

  for (auto& future : futures) {
    future.get();
  }
}

void set_value(rapidjson::Value& json_val,
               const RowGroupInterval& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetObject();
  json_utils::add_value_to_object(json_val, value.file_path, "file_path", allocator);
  json_utils::add_value_to_object(json_val, value.start_index, "start_index", allocator);
  json_utils::add_value_to_object(json_val, value.end_index, "end_index", allocator);
}

void get_value(const rapidjson::Value& json_val, RowGroupInterval& value) {
  CHECK(json_val.IsObject());
  json_utils::get_value_from_object(json_val, value.file_path, "file_path");
  json_utils::get_value_from_object(json_val, value.start_index, "start_index");
  json_utils::get_value_from_object(json_val, value.end_index, "end_index");
}

std::string ParquetDataWrapper::getSerializedDataWrapper() const {
  rapidjson::Document d;
  d.SetObject();

  json_utils::add_value_to_object(d,
                                  fragment_to_row_group_interval_map_,
                                  "fragment_to_row_group_interval_map",
                                  d.GetAllocator());
  json_utils::add_value_to_object(d, last_row_group_, "last_row_group", d.GetAllocator());
  json_utils::add_value_to_object(
      d, last_fragment_index_, "last_fragment_index", d.GetAllocator());
  json_utils::add_value_to_object(
      d, last_fragment_row_count_, "last_fragment_row_count", d.GetAllocator());
  json_utils::add_value_to_object(
      d, total_row_count_, "total_row_count", d.GetAllocator());
  json_utils::add_value_to_object(
      d, last_file_row_count_, "last_file_row_count", d.GetAllocator());
  return json_utils::write_to_string(d);
}

void ParquetDataWrapper::restoreDataWrapperInternals(
    const std::string& file_path,
    const ChunkMetadataVector& chunk_metadata_vector) {
  auto d = json_utils::read_from_file(file_path);
  CHECK(d.IsObject());

  json_utils::get_value_from_object(
      d, fragment_to_row_group_interval_map_, "fragment_to_row_group_interval_map");
  json_utils::get_value_from_object(d, last_row_group_, "last_row_group");
  json_utils::get_value_from_object(d, last_fragment_index_, "last_fragment_index");
  json_utils::get_value_from_object(
      d, last_fragment_row_count_, "last_fragment_row_count");
  json_utils::get_value_from_object(d, total_row_count_, "total_row_count");
  if (d.HasMember("last_file_row_count")) {
    json_utils::get_value_from_object(d, last_file_row_count_, "last_file_row_count");
  }

  CHECK(chunk_metadata_map_.empty());
  for (const auto& [chunk_key, chunk_metadata] : chunk_metadata_vector) {
    chunk_metadata_map_[chunk_key] = chunk_metadata;
  }
  is_restored_ = true;
}

bool ParquetDataWrapper::isRestored() const {
  return is_restored_;
}

DataPreview ParquetDataWrapper::getDataPreview(const size_t num_rows) {
  LazyParquetChunkLoader chunk_loader(file_system_,
                                      file_reader_cache_.get(),
                                      &render_group_analyzer_map_,
                                      foreign_table_->tableName);
  auto file_paths = getAllFilePaths();
  if (file_paths.empty()) {
    throw ForeignStorageException{"No file found at \"" +
                                  getFullFilePath(foreign_table_) + "\""};
  }
  return chunk_loader.previewFiles(file_paths, num_rows, *foreign_table_);
}

// declared in three derived classes to avoid
// polluting ForeignDataWrapper virtual base
// @TODO refactor to lower class if needed
void ParquetDataWrapper::createRenderGroupAnalyzers() {
  // must have these
  CHECK_GE(db_id_, 0);
  CHECK(foreign_table_);

  // populate map for all poly columns in this table
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  auto columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);
  for (auto const& column : columns) {
    if (IS_GEO_POLY(column->columnType.get_type())) {
      CHECK(render_group_analyzer_map_
                .try_emplace(column->columnId,
                             std::make_unique<import_export::RenderGroupAnalyzer>())
                .second);
    }
  }
}

void ParquetDataWrapper::removeMetadataForLastFile(const std::string& last_file_path) {
  std::optional<int32_t> first_deleted_fragment_id;
  for (auto it = fragment_to_row_group_interval_map_.begin();
       it != fragment_to_row_group_interval_map_.end();) {
    const auto fragment_id = it->first;
    const auto& row_group_intervals = it->second;
    for (const auto& row_group_interval : row_group_intervals) {
      if (first_deleted_fragment_id.has_value()) {
        // All subsequent fragments should map to the last file.
        CHECK_EQ(last_file_path, row_group_interval.file_path);
      } else if (last_file_path == row_group_interval.file_path) {
        first_deleted_fragment_id = fragment_id;
      }
    }
    if (first_deleted_fragment_id.has_value() &&
        first_deleted_fragment_id.value() < fragment_id) {
      it = fragment_to_row_group_interval_map_.erase(it);
    } else {
      it++;
    }
  }
  CHECK(first_deleted_fragment_id.has_value());

  std::map<int32_t, size_t> remaining_fragments_row_counts;
  for (auto it = chunk_metadata_map_.begin(); it != chunk_metadata_map_.end();) {
    auto fragment_id = it->first[CHUNK_KEY_FRAGMENT_IDX];
    if (fragment_id >= first_deleted_fragment_id.value()) {
      it = chunk_metadata_map_.erase(it);
    } else {
      auto fragment_count_it = remaining_fragments_row_counts.find(fragment_id);
      if (fragment_count_it == remaining_fragments_row_counts.end()) {
        remaining_fragments_row_counts[fragment_id] = it->second->numElements;
      } else {
        CHECK_EQ(remaining_fragments_row_counts[fragment_id], it->second->numElements);
      }
      it++;
    }
  }

  total_row_count_ = 0;
  for (const auto [fragment_id, row_count] : remaining_fragments_row_counts) {
    total_row_count_ += row_count;
  }

  // Re-populate metadata for last fragment with deleted rows, excluding metadata for the
  // last file.
  auto row_group_intervals_to_scan = shared::get_from_map(
      fragment_to_row_group_interval_map_, first_deleted_fragment_id.value());
  auto it = std::find_if(row_group_intervals_to_scan.begin(),
                         row_group_intervals_to_scan.end(),
                         [&last_file_path](const auto& row_group_interval) {
                           return row_group_interval.file_path == last_file_path;
                         });
  CHECK(it != row_group_intervals_to_scan.end());
  row_group_intervals_to_scan.erase(it, row_group_intervals_to_scan.end());

  if (first_deleted_fragment_id.value() > 0) {
    last_fragment_index_ = first_deleted_fragment_id.value() - 1;
    last_fragment_row_count_ =
        shared::get_from_map(remaining_fragments_row_counts, last_fragment_index_);
    const auto& last_row_group_intervals =
        shared::get_from_map(fragment_to_row_group_interval_map_, last_fragment_index_);
    if (last_row_group_intervals.empty()) {
      last_row_group_ = 0;
    } else {
      last_row_group_ = last_row_group_intervals.back().end_index;
    }
    fragment_to_row_group_interval_map_.erase(first_deleted_fragment_id.value());
  } else {
    CHECK_EQ(total_row_count_, size_t(0));
    resetParquetMetadata();
  }

  if (!row_group_intervals_to_scan.empty()) {
    metadataScanRowGroupIntervals(row_group_intervals_to_scan);
  }
}

void ParquetDataWrapper::metadataScanRowGroupIntervals(
    const std::vector<RowGroupInterval>& row_group_intervals) {
  std::vector<std::string> file_paths;
  for (const auto& row_group_interval : row_group_intervals) {
    file_paths.emplace_back(row_group_interval.file_path);
  }
  auto row_group_metadata_map = getRowGroupMetadataMap(file_paths);
  std::list<RowGroupMetadata> row_group_metadata;
  for (const auto& row_group_interval : row_group_intervals) {
    for (auto row_group = row_group_interval.start_index;
         row_group <= row_group_interval.end_index;
         row_group++) {
      row_group_metadata.emplace_back(shared::get_from_map(
          row_group_metadata_map, {row_group_interval.file_path, row_group}));
    }
  }
  metadataScanRowGroupMetadata(row_group_metadata);
}

std::map<FilePathAndRowGroup, RowGroupMetadata>
ParquetDataWrapper::getRowGroupMetadataMap(
    const std::vector<std::string>& file_paths) const {
  auto row_group_metadata = getRowGroupMetadataForFilePaths(file_paths);
  std::map<FilePathAndRowGroup, RowGroupMetadata> row_group_metadata_map;
  for (const auto& row_group_metadata_item : row_group_metadata) {
    row_group_metadata_map[{row_group_metadata_item.file_path,
                            row_group_metadata_item.row_group_index}] =
        row_group_metadata_item;
  }
  return row_group_metadata_map;
}
}  // namespace foreign_storage
