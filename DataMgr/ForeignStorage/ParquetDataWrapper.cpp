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

#include <regex>

#include <arrow/filesystem/localfs.h>
#include <boost/filesystem.hpp>

#include "ForeignDataWrapperShared.h"
#include "FsiJsonUtils.h"
#include "ImportExport/Importer.h"
#include "LazyParquetChunkLoader.h"
#include "ParquetShared.h"
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

ParquetDataWrapper::ParquetDataWrapper(const int db_id, const ForeignTable* foreign_table)
    : db_id_(db_id)
    , foreign_table_(foreign_table)
    , last_fragment_index_(0)
    , last_fragment_row_count_(0)
    , total_row_count_(0)
    , last_row_group_(0)
    , is_restored_(false)
    , schema_(std::make_unique<ForeignTableSchema>(db_id, foreign_table)) {
  auto& server_options = foreign_table->foreign_server->options;
  if (server_options.find(ForeignServer::STORAGE_TYPE_KEY)->second ==
      ForeignServer::LOCAL_FILE_STORAGE_TYPE) {
    file_system_ = std::make_shared<arrow::fs::LocalFileSystem>();
  } else {
    UNREACHABLE();
  }
}

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

std::vector<std::string_view> ParquetDataWrapper::getSupportedOptions() {
  return std::vector<std::string_view>{supported_options_.begin(),
                                       supported_options_.end()};
}

void ParquetDataWrapper::validateFilePath() const {
  auto& server_options = foreign_table_->foreign_server->options;
  if (server_options.find(ForeignServer::STORAGE_TYPE_KEY)->second ==
      ForeignServer::LOCAL_FILE_STORAGE_TYPE) {
    ddl_utils::validate_allowed_file_path(foreign_table_->getFullFilePath(),
                                          ddl_utils::DataTransferType::IMPORT);
  }
}

void ParquetDataWrapper::resetParquetMetadata() {
  fragment_to_row_group_interval_map_.clear();
  fragment_to_row_group_interval_map_[0] = {};

  last_row_group_ = 0;
  last_fragment_index_ = 0;
  last_fragment_row_count_ = 0;
  total_row_count_ = 0;
}

std::list<const ColumnDescriptor*> ParquetDataWrapper::getColumnsToInitialize(
    const Interval<ColumnType>& column_interval) {
  const auto catalog =
      Catalog_Namespace::SysCatalog::instance().checkedGetCatalog(db_id_);
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
    const bool reserve_buffers_and_set_stats) {
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
  auto catalog = Catalog_Namespace::SysCatalog::instance().checkedGetCatalog(db_id_);
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
    if (new_file_paths.empty() && all_file_paths.size() == 1) {
      CHECK_EQ(processed_file_paths.size(), static_cast<size_t>(1));
      const auto& file_path = *all_file_paths.begin();
      CHECK_EQ(*processed_file_paths.begin(), file_path);

      std::unique_ptr<parquet::arrow::FileReader> reader;
      open_parquet_table(file_path, reader, file_system_);
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
    new_file_paths = getAllFilePaths();
    chunk_metadata_map_.clear();
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
  arrow::fs::FileSelector file_selector{};
  std::string base_path = foreign_table_->getFullFilePath();
  file_selector.base_dir = base_path;
  file_selector.recursive = true;

  auto file_info_result = file_system_->GetFileInfo(file_selector);
  if (!file_info_result.ok()) {
    // This is expected when `base_path` points to a single file.
    file_paths.emplace(base_path);
  } else {
    auto& file_info_vector = file_info_result.ValueOrDie();
    for (const auto& file_info : file_info_vector) {
      if (file_info.type() == arrow::fs::FileType::File) {
        file_paths.emplace(file_info.path());
      }
    }
    if (file_paths.empty()) {
      throw std::runtime_error{"No file found at given path \"" + base_path + "\"."};
    }
  }
  return file_paths;
}

import_export::CopyParams ParquetDataWrapper::validateAndGetCopyParams() const {
  import_export::CopyParams copy_params{};
  // The file_type argument is never utilized in the context of FSI,
  // for completeness, set the file_type
  copy_params.file_type = import_export::FileType::PARQUET;
  return copy_params;
}

std::string ParquetDataWrapper::validateAndGetStringWithLength(
    const std::string& option_name,
    const size_t expected_num_chars) const {
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

void ParquetDataWrapper::metadataScanFiles(const std::set<std::string>& file_paths) {
  LazyParquetChunkLoader chunk_loader(file_system_);
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
    std::map<ChunkKey, AbstractBuffer*>& required_buffers) {
  auto catalog = Catalog_Namespace::SysCatalog::instance().checkedGetCatalog(db_id_);
  const ColumnDescriptor* logical_column =
      schema_->getColumnDescriptor(logical_column_id);
  auto parquet_column_index = schema_->getParquetColumnIndex(logical_column_id);

  const Interval<ColumnType> column_interval = {
      logical_column_id,
      logical_column_id + logical_column->columnType.get_physical_cols()};
  initializeChunkBuffers(fragment_id, column_interval, required_buffers, true);

  const auto& row_group_intervals = fragment_to_row_group_interval_map_[fragment_id];

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
      auto buffer = required_buffers[data_chunk_key];
      CHECK(buffer);
      chunk.setBuffer(buffer);
      ChunkKey index_chunk_key = {
          db_id_, foreign_table_->tableId, column_id, fragment_id, 2};
      auto index_buffer = required_buffers[index_chunk_key];
      CHECK(index_buffer);
      chunk.setIndexBuffer(index_buffer);
    } else {
      ChunkKey chunk_key = {db_id_, foreign_table_->tableId, column_id, fragment_id};
      auto buffer = required_buffers[chunk_key];
      CHECK(buffer);
      chunk.setBuffer(buffer);
    }
    chunks.emplace_back(chunk);
  }

  LazyParquetChunkLoader chunk_loader(file_system_);
  auto metadata = chunk_loader.loadChunk(
      row_group_intervals, parquet_column_index, chunks, string_dictionary);
  auto fragmenter = foreign_table_->fragmenter;
  if (fragmenter) {
    auto metadata_iter = metadata.begin();
    for (int column_id = column_interval.start; column_id <= column_interval.end;
         ++column_id, ++metadata_iter) {
      auto column = schema_->getColumnDescriptor(column_id);
      ChunkKey data_chunk_key = {db_id_, foreign_table_->tableId, column_id, fragment_id};
      if (column->columnType.is_varlen_indeed()) {
        data_chunk_key.emplace_back(1);
      }
      CHECK(chunk_metadata_map_.find(data_chunk_key) != chunk_metadata_map_.end());
      auto cached_metadata = chunk_metadata_map_[data_chunk_key];
      auto updated_metadata = std::make_shared<ChunkMetadata>();
      *updated_metadata = *cached_metadata;
      // for certain types, update the metadata statistics
      if (is_dictionary_encoded_string_column ||
          logical_column->columnType.is_geometry()) {
        CHECK(metadata_iter != metadata.end());
        auto& chunk_metadata_ptr = *metadata_iter;
        updated_metadata->chunkStats.max = chunk_metadata_ptr->chunkStats.max;
        updated_metadata->chunkStats.min = chunk_metadata_ptr->chunkStats.min;
      }
      CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
      updated_metadata->numBytes = required_buffers[data_chunk_key]->size();
      fragmenter->updateColumnChunkMetadata(column, fragment_id, updated_metadata);
    }
  }
}

void ParquetDataWrapper::populateChunkBuffers(
    std::map<ChunkKey, AbstractBuffer*>& required_buffers,
    std::map<ChunkKey, AbstractBuffer*>& optional_buffers) {
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
    loadBuffersUsingLazyParquetChunkLoader(column_id, fragment_id, required_buffers);
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

void ParquetDataWrapper::serializeDataWrapperInternals(
    const std::string& file_path) const {
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

  json_utils::write_to_file(d, file_path);
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
