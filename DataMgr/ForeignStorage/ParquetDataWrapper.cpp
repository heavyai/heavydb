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

#include "ForeignStorageException.h"
#include "FsiJsonUtils.h"
#include "LazyParquetChunkLoader.h"
#include "ParquetShared.h"
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

ParquetDataWrapper::ParquetDataWrapper() : db_id_(-1), foreign_table_(nullptr) {}

ParquetDataWrapper::ParquetDataWrapper(const int db_id, const ForeignTable* foreign_table)
    : db_id_(db_id)
    , foreign_table_(foreign_table)
    , last_fragment_index_(0)
    , last_fragment_row_count_(0)
    , total_row_count_(0)
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
    Chunk_NS::Chunk chunk{column};
    ChunkKey data_chunk_key;
    if (column->columnType.is_varlen_indeed()) {
      data_chunk_key = {
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 1};
      CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
      auto data_buffer = required_buffers.at(data_chunk_key);
      chunk.setBuffer(data_buffer);

      ChunkKey index_chunk_key{
          db_id_, foreign_table_->tableId, column->columnId, fragment_index, 2};
      CHECK(required_buffers.find(index_chunk_key) != required_buffers.end());
      auto index_buffer = required_buffers.at(index_chunk_key);
      chunk.setIndexBuffer(index_buffer);
    } else {
      data_chunk_key = {
          db_id_, foreign_table_->tableId, column->columnId, fragment_index};
      CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
      auto data_buffer = required_buffers.at(data_chunk_key);
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
      if (column->columnType.is_string() &&
          column->columnType.get_compression() == kENCODING_NONE) {
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
}

bool ParquetDataWrapper::isNewFile(const std::string& file_path) const {
  const auto last_fragment_entry =
      fragment_to_row_group_interval_map_.find(last_fragment_index_);
  CHECK(last_fragment_entry != fragment_to_row_group_interval_map_.end());

  // The entry for the first fragment starts out as an empty vector
  if (last_fragment_entry->second.empty()) {
    CHECK_EQ(last_fragment_index_, 0);
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
    CHECK_EQ(last_fragment_index_, 0);
  } else {
    last_fragment_entry->second.back().end_index = last_row_group_;
  }
  last_fragment_entry->second.emplace_back(RowGroupInterval{file_path, 0});
}

void ParquetDataWrapper::fetchChunkMetadata() {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  std::set<std::string> new_file_paths;
  auto processed_file_paths = getProcessedFilePaths();
  if (foreign_table_->isAppendMode() && !processed_file_paths.empty()) {
    auto all_file_paths = getAllFilePaths();
    for (const auto& file_path : processed_file_paths) {
      if (all_file_paths.find(file_path) == all_file_paths.end()) {
        throw_removed_file_error(file_path);
      }
    }

    for (const auto& file_path : all_file_paths) {
      if (processed_file_paths.find(file_path) == processed_file_paths.end()) {
        new_file_paths.emplace(file_path);
      }
    }

    // Single file append
    // If an append occurs with multiple files, then we assume any existing files have
    // not been altered.  If an append occurs on a single file, then we check to see if
    // it has changed.
    if (new_file_paths.empty() && all_file_paths.size() == 1) {
      CHECK_EQ(processed_file_paths.size(), static_cast<size_t>(1));
      const auto& file_path = *all_file_paths.begin();
      CHECK_EQ(*processed_file_paths.begin(), file_path);

      // Since an existing file is being appended to we need to update the cached
      // FileReader as the existing one will be out of date.
      auto reader = file_reader_cache_->insert(file_path, file_system_);
      size_t row_count = reader->parquet_reader()->metadata()->num_rows();

      if (row_count < total_row_count_) {
        throw_removed_row_error(file_path);
      } else if (row_count > total_row_count_) {
        new_file_paths = all_file_paths;
        chunk_metadata_map_.clear();
        resetParquetMetadata();
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

std::set<std::string> ParquetDataWrapper::getProcessedFilePaths() {
  std::set<std::string> file_paths;
  for (const auto& entry : fragment_to_row_group_interval_map_) {
    for (const auto& row_group_interval : entry.second) {
      file_paths.emplace(row_group_interval.file_path);
    }
  }
  return file_paths;
}

std::set<std::string> ParquetDataWrapper::getAllFilePaths() {
  auto timer = DEBUG_TIMER(__func__);
  std::set<std::string> file_paths;
  auto file_path = getFullFilePath(foreign_table_);
  auto file_info_result = file_system_->GetFileInfo(file_path);
  if (!file_info_result.ok()) {
    throw_file_access_error(file_path, file_info_result.status().message());
  } else {
    auto& file_info = file_info_result.ValueOrDie();
    if (file_info.type() == arrow::fs::FileType::NotFound) {
      throw_file_not_found_error(file_path);
    } else if (file_info.type() == arrow::fs::FileType::File) {
      file_paths.emplace(file_path);
    } else {
      CHECK_EQ(arrow::fs::FileType::Directory, file_info.type());
      arrow::fs::FileSelector file_selector{};
      file_selector.base_dir = file_path;
      file_selector.recursive = true;
      auto selector_result = file_system_->GetFileInfo(file_selector);
      if (!selector_result.ok()) {
        throw_file_access_error(file_path, selector_result.status().message());
      } else {
        auto& file_info_vector = selector_result.ValueOrDie();
        for (const auto& file_info : file_info_vector) {
          if (file_info.type() == arrow::fs::FileType::File) {
            file_paths.emplace(file_info.path());
          }
        }
      }
    }
  }
  return file_paths;
}

void ParquetDataWrapper::metadataScanFiles(const std::set<std::string>& file_paths) {
  LazyParquetChunkLoader chunk_loader(file_system_, file_reader_cache_.get());
  auto row_group_metadata = chunk_loader.metadataScan(file_paths, *schema_);
  auto column_interval =
      Interval<ColumnType>{schema_->getLogicalAndPhysicalColumns().front()->columnId,
                           schema_->getLogicalAndPhysicalColumns().back()->columnId};

  for (const auto& row_group_metadata_item : row_group_metadata) {
    const auto& column_chunk_metadata = row_group_metadata_item.column_chunk_metadata;
    CHECK(static_cast<int>(column_chunk_metadata.size()) ==
          schema_->numLogicalAndPhysicalColumns());
    auto column_chunk_metadata_iter = column_chunk_metadata.begin();
    const int64_t import_row_count = (*column_chunk_metadata_iter)->numElements;
    int row_group = row_group_metadata_item.row_group_index;
    const auto& file_path = row_group_metadata_item.file_path;
    if (moveToNextFragment(import_row_count)) {
      addNewFragment(row_group, file_path);
    } else if (isNewFile(file_path)) {
      CHECK_EQ(row_group, 0);
      addNewFile(file_path);
    }
    last_row_group_ = row_group;

    for (int column_id = column_interval.start; column_id <= column_interval.end;
         column_id++, column_chunk_metadata_iter++) {
      CHECK(column_chunk_metadata_iter != column_chunk_metadata.end());
      const auto column_descriptor = schema_->getColumnDescriptor(column_id);

      const auto& type_info = column_descriptor->columnType;
      ChunkKey chunk_key{
          db_id_, foreign_table_->tableId, column_id, last_fragment_index_};
      ChunkKey data_chunk_key = chunk_key;
      if (type_info.is_varlen_indeed()) {
        data_chunk_key.emplace_back(1);
      }
      std::shared_ptr<ChunkMetadata> chunk_metadata = *column_chunk_metadata_iter;
      if (chunk_metadata_map_.find(data_chunk_key) == chunk_metadata_map_.end()) {
        chunk_metadata_map_[data_chunk_key] = chunk_metadata;
      } else {
        reduce_metadata(chunk_metadata_map_[data_chunk_key], chunk_metadata);
      }
    }
    last_fragment_row_count_ += import_row_count;
    total_row_count_ += import_row_count;
  }
  finalizeFragmentMap();
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
    const ChunkToBufferMap& required_buffers) {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  const ColumnDescriptor* logical_column =
      schema_->getColumnDescriptor(logical_column_id);
  auto parquet_column_index = schema_->getParquetColumnIndex(logical_column_id);

  const Interval<ColumnType> column_interval = {
      logical_column_id,
      logical_column_id + logical_column->columnType.get_physical_cols()};
  initializeChunkBuffers(fragment_id, column_interval, required_buffers, true);

  const auto& row_group_intervals = fragment_to_row_group_interval_map_.at(fragment_id);

  const bool is_dictionary_encoded_string_column =
      logical_column->columnType.is_dict_encoded_string() ||
      (logical_column->columnType.is_array() &&
       logical_column->columnType.get_elem_type().is_dict_encoded_string());

  StringDictionary* string_dictionary = nullptr;
  if (is_dictionary_encoded_string_column) {
    auto dict_descriptor = catalog->getMetadataForDictUnlocked(
        logical_column->columnType.get_comp_param(), true);
    CHECK(dict_descriptor);
    string_dictionary = dict_descriptor->stringDict.get();
  }

  std::list<Chunk_NS::Chunk> chunks;
  for (int column_id = column_interval.start; column_id <= column_interval.end;
       ++column_id) {
    auto column_descriptor = schema_->getColumnDescriptor(column_id);
    Chunk_NS::Chunk chunk{column_descriptor};
    if (column_descriptor->columnType.is_varlen_indeed()) {
      ChunkKey data_chunk_key = {
          db_id_, foreign_table_->tableId, column_id, fragment_id, 1};
      auto buffer = required_buffers.at(data_chunk_key);
      chunk.setBuffer(buffer);
      ChunkKey index_chunk_key = {
          db_id_, foreign_table_->tableId, column_id, fragment_id, 2};
      auto index_buffer = required_buffers.at(index_chunk_key);
      chunk.setIndexBuffer(index_buffer);
    } else {
      ChunkKey chunk_key = {db_id_, foreign_table_->tableId, column_id, fragment_id};
      auto buffer = required_buffers.at(chunk_key);
      chunk.setBuffer(buffer);
    }
    chunks.emplace_back(chunk);
  }

  LazyParquetChunkLoader chunk_loader(file_system_, file_reader_cache_.get());
  auto metadata = chunk_loader.loadChunk(
      row_group_intervals, parquet_column_index, chunks, string_dictionary);
  auto fragmenter = foreign_table_->fragmenter;

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
    auto cached_metadata_previous = chunk_metadata_map_.at(data_chunk_key);
    chunk_metadata_map_.at(data_chunk_key) = std::make_shared<ChunkMetadata>();
    auto cached_metadata = chunk_metadata_map_.at(data_chunk_key);
    *cached_metadata = *cached_metadata_previous;

    CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
    cached_metadata->numBytes = required_buffers.at(data_chunk_key)->size();

    // for certain types, update the metadata statistics
    // should update the fragmenter, cache, and the internal chunk_metadata_map_
    if (is_dictionary_encoded_string_column || logical_column->columnType.is_geometry()) {
      CHECK(metadata_iter != metadata.end());
      auto& chunk_metadata_ptr = *metadata_iter;
      cached_metadata->chunkStats.max = chunk_metadata_ptr->chunkStats.max;
      cached_metadata->chunkStats.min = chunk_metadata_ptr->chunkStats.min;

      // Update stats on buffer so it is saved in cache
      required_buffers.at(data_chunk_key)
          ->getEncoder()
          ->resetChunkStats(cached_metadata->chunkStats);
    }

    if (fragmenter) {
      fragmenter->updateColumnChunkMetadata(column, fragment_id, cached_metadata);
    }
  }
}

void ParquetDataWrapper::populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                                              const ChunkToBufferMap& optional_buffers) {
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

  auto hints_per_thread = partition_for_threads(col_frag_hints, g_max_import_threads);

  std::vector<std::future<void>> futures;
  for (const auto& hint_set : hints_per_thread) {
    futures.emplace_back(std::async(std::launch::async, [&, hint_set, this] {
      for (const auto& [col_id, frag_id] : hint_set) {
        loadBuffersUsingLazyParquetChunkLoader(col_id, frag_id, buffers_to_load);
      }
    }));
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

  CHECK(chunk_metadata_map_.empty());
  for (const auto& [chunk_key, chunk_metadata] : chunk_metadata_vector) {
    chunk_metadata_map_[chunk_key] = chunk_metadata;
  }
  is_restored_ = true;
}

bool ParquetDataWrapper::isRestored() const {
  return is_restored_;
}

}  // namespace foreign_storage
