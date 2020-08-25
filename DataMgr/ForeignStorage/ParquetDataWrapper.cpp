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

#include <boost/filesystem.hpp>

#include "ImportExport/Importer.h"
#include "Utils/DdlUtils.h"

#define CHUNK_KEY_FRAGMENT_IDX 3
#define CHUNK_KEY_COLUMN_IDX 2
#define CHUNK_KEY_DB_IDX 0

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
    , row_count_(0)
    , max_row_group_(0)
    , current_row_group_(-1)
    , row_group_row_count_(0)
    , foreign_table_column_map_(db_id_, foreign_table_) {}

ParquetDataWrapper::ParquetDataWrapper(const ForeignTable* foreign_table)
    : db_id_(-1)
    , foreign_table_(foreign_table)
    , foreign_table_column_map_(db_id_, foreign_table_) {}

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
  row_count_ = 0;
  fragment_to_row_group_interval_map_.clear();
  fragment_to_row_group_interval_map_[0] = {0, 0, -1};

  max_row_group_ = 0;
  row_group_row_count_ = 0;
  current_row_group_ = -1;
}

std::unique_ptr<ForeignStorageBuffer>& ParquetDataWrapper::initializeChunkBuffer(
    const ChunkKey& chunk_key) {
  auto& buffer = chunk_buffer_map_[chunk_key];
  buffer.reset();
  buffer = std::make_unique<ForeignStorageBuffer>();
  return buffer;
}

std::list<const ColumnDescriptor*> ParquetDataWrapper::getColumnsToInitialize(
    const Interval<ColumnType>& column_interval) {
  const auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);
  const auto& columns = catalog->getAllColumnMetadataForTableUnlocked(
      foreign_table_->tableId, false, false, true);
  auto column_start = column_interval.start;
  auto column_end = column_interval.end;
  column_start = std::max(0, column_start);
  column_end = std::min(columns.back()->columnId - 1, column_end);
  std::list<const ColumnDescriptor*> columns_to_init;
  for (const auto column : columns) {
    auto column_id = column->columnId - 1;
    if (column_id >= column_start && column_id <= column_end) {
      columns_to_init.push_back(column);
    }
  }
  return columns_to_init;
}

void ParquetDataWrapper::initializeChunkBuffers(
    const Interval<FragmentType>& fragment_interval,
    const Interval<ColumnType>& column_interval) {
  for (const auto column : getColumnsToInitialize(column_interval)) {
    for (auto fragment_index = fragment_interval.start;
         fragment_index <= fragment_interval.end;
         ++fragment_index) {
      Chunk_NS::Chunk chunk{column};
      if (column->columnType.is_varlen() && !column->columnType.is_fixlen_array()) {
        ChunkKey data_chunk_key{
            db_id_, foreign_table_->tableId, column->columnId, fragment_index, 1};
        chunk.setBuffer(initializeChunkBuffer(data_chunk_key).get());

        ChunkKey index_chunk_key{
            db_id_, foreign_table_->tableId, column->columnId, fragment_index, 2};
        chunk.setIndexBuffer(initializeChunkBuffer(index_chunk_key).get());
      } else {
        ChunkKey data_chunk_key{
            db_id_, foreign_table_->tableId, column->columnId, fragment_index};
        chunk.setBuffer(initializeChunkBuffer(data_chunk_key).get());
      }
      chunk.initEncoder();
    }
  }
}

void ParquetDataWrapper::initializeChunkBuffers(const int fragment_index) {
  initializeChunkBuffers({fragment_index, fragment_index},
                         {0, std::numeric_limits<int>::max()});
}

void ParquetDataWrapper::finalizeFragmentMap() {
  // Set the last entry in the fragment map for the last processed row
  int fragment_index =
      row_count_ > 0 ? (row_count_ - 1) / foreign_table_->maxFragRows : 0;
  fragment_to_row_group_interval_map_[fragment_index].end_row_group_index =
      max_row_group_;
}

void ParquetDataWrapper::updateFragmentMap(int fragment_index, int row_group) {
  CHECK(fragment_index > 0);
  auto& end_row_group_index =
      fragment_to_row_group_interval_map_[fragment_index - 1].end_row_group_index;
  if (row_group_row_count_ == 0) {
    end_row_group_index = row_group - 1;
  } else {
    end_row_group_index = row_group;
  }
  fragment_to_row_group_interval_map_[fragment_index] = {
      row_group_row_count_, row_group, -1};
}

void ParquetDataWrapper::fetchChunkMetadata() {
  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);

  // reset chunk buffers
  chunk_buffer_map_.clear();
  initializeChunkBuffers(0);

  resetParquetMetadata();
  LazyParquetImporter::RowGroupMetadataVector metadata_vector;
  LazyParquetImporter importer(getMetadataLoader(*catalog, metadata_vector),
                               getFilePath(),
                               validateAndGetCopyParams(),
                               metadata_vector);
  importer.metadataScan();
  finalizeFragmentMap();

  if (chunk_buffer_map_.empty()) {
    throw std::runtime_error{"An error occurred when attempting to process data."};
  }
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

std::optional<bool> ParquetDataWrapper::validateAndGetBoolValue(
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

void ParquetDataWrapper::updateRowGroupMetadata(int row_group) {
  max_row_group_ = std::max(row_group, max_row_group_);
  if (newRowGroup(row_group)) {
    row_group_row_count_ = 0;
  }
  current_row_group_ = row_group;
}

void ParquetDataWrapper::shiftData(DataBlockPtr& data_block,
                                   const size_t import_shift,
                                   const size_t element_size) {
  if (element_size > 0) {
    auto& data_ptr = data_block.numbersPtr;
    data_ptr += element_size * import_shift;
  }
}

void ParquetDataWrapper::updateStatsForBuffer(AbstractBuffer* buffer,
                                              const DataBlockPtr& data_block,
                                              const size_t import_count,
                                              const size_t import_shift) {
  auto& encoder = buffer->encoder;
  CHECK(encoder);
  auto& type_info = buffer->sql_type;
  if (type_info.is_varlen()) {
    switch (type_info.get_type()) {
      case kARRAY: {
        encoder->updateStats(data_block.arraysPtr, import_shift, import_count);
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON: {
        encoder->updateStats(data_block.stringsPtr, import_shift, import_count);
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
  CHECK(!(type_info.is_varlen() && !type_info.is_fixlen_array()));
  CHECK(chunk_buffer_map_.find(chunk_key) != chunk_buffer_map_.end());
  auto buffer = chunk_buffer_map_[chunk_key].get();
  auto& encoder = buffer->encoder;
  std::shared_ptr<ChunkMetadata> chunk_metadata_ptr = std::make_shared<ChunkMetadata>();
  encoder->getMetadata(chunk_metadata_ptr);
  auto& chunk_stats = chunk_metadata_ptr->chunkStats;
  chunk_stats.has_nulls |= has_nulls;
  encoder->resetChunkStats(chunk_stats);
  if (is_all_nulls) {  // do not attempt to load min/max statistics if entire row group is
                       // null
    encoder->setNumElems(encoder->getNumElems() + import_count);
  } else {
    loadChunk(column, chunk_key, data_block, 2, 0, true);
    encoder->setNumElems(encoder->getNumElems() + import_count - 2);
  }
}

void ParquetDataWrapper::loadChunk(const ColumnDescriptor* column,
                                   const ChunkKey& chunk_key,
                                   DataBlockPtr& data_block,
                                   const size_t import_count,
                                   const size_t import_shift,
                                   const bool metadata_only,
                                   const bool first_fragment,
                                   const size_t element_size) {
  Chunk_NS::Chunk chunk{column};
  auto column_id = column->columnId;
  CHECK(column_id == chunk_key[CHUNK_KEY_COLUMN_IDX]);
  auto& type_info = column->columnType;
  if (type_info.is_varlen() && !type_info.is_fixlen_array()) {
    ChunkKey data_chunk_key{chunk_key};
    data_chunk_key.resize(5);
    data_chunk_key[4] = 1;
    CHECK(chunk_buffer_map_.find(data_chunk_key) != chunk_buffer_map_.end());
    auto& data_buffer = chunk_buffer_map_[data_chunk_key];
    chunk.setBuffer(data_buffer.get());

    ChunkKey index_chunk_key{chunk_key};
    index_chunk_key.resize(5);
    index_chunk_key[4] = 2;
    CHECK(chunk_buffer_map_.find(index_chunk_key) != chunk_buffer_map_.end());
    auto& index_buffer = chunk_buffer_map_[index_chunk_key];
    chunk.setIndexBuffer(index_buffer.get());
  } else {
    CHECK(chunk_buffer_map_.find(chunk_key) != chunk_buffer_map_.end());
    auto& buffer = chunk_buffer_map_[chunk_key];
    chunk.setBuffer(buffer.get());
  }
  if (metadata_only) {
    auto buffer = chunk.getBuffer();
    updateStatsForBuffer(buffer, data_block, import_count, import_shift);
  } else {
    if (first_fragment) {
      shiftData(data_block, import_shift, element_size);
    }
    chunk.appendData(data_block, import_count, import_shift);
  }
  chunk.setBuffer(nullptr);
  chunk.setIndexBuffer(nullptr);
}

size_t ParquetDataWrapper::getElementSizeFromImportBuffer(
    const std::unique_ptr<import_export::TypedImportBuffer>& import_buffer) const {
  auto& type_info = import_buffer->getColumnDesc()->columnType;
  switch (type_info.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kFLOAT:
    case kDOUBLE:
    case kDATE:
    case kTIME:
    case kTIMESTAMP:
      break;
    default:
      return 0;
  }
  return import_buffer->getElementSize();
}

import_export::Loader* ParquetDataWrapper::getChunkLoader(
    Catalog_Namespace::Catalog& catalog,
    const Interval<FragmentType>& fragment_interval,
    const Interval<ColumnType>& column_interval,
    const int chunk_key_db) {
  auto callback =
      [this, fragment_interval, column_interval, chunk_key_db](
          const std::vector<std::unique_ptr<import_export::TypedImportBuffer>>&
              import_buffers,
          std::vector<DataBlockPtr>& data_blocks,
          size_t import_row_count) {
        auto first_column = column_interval.start;
        auto last_column = column_interval.end;
        auto first_fragment = fragment_interval.start;
        auto last_fragment = fragment_interval.end;

        size_t processed_import_row_count = 0;
        while (processed_import_row_count < import_row_count) {
          int fragment_index = partial_import_row_count_ / foreign_table_->maxFragRows;
          size_t row_count_for_fragment = std::min<size_t>(
              foreign_table_->maxFragRows -
                  (partial_import_row_count_ % foreign_table_->maxFragRows),
              import_row_count - processed_import_row_count);

          if (fragment_index < first_fragment) {  // skip to the first fragment
            partial_import_row_count_ += row_count_for_fragment;
            processed_import_row_count += row_count_for_fragment;
            continue;
          }
          if (fragment_index > last_fragment) {  // nothing to do after last fragment
            break;
          }

          for (int column_idx = first_column; column_idx <= last_column; ++column_idx) {
            auto& import_buffer = import_buffers[column_idx];
            ChunkKey chunk_key{
                chunk_key_db, foreign_table_->tableId, column_idx + 1, fragment_index};
            loadChunk(import_buffer->getColumnDesc(),
                      chunk_key,
                      data_blocks[column_idx],
                      row_count_for_fragment,
                      processed_import_row_count,
                      false,
                      fragment_index == first_fragment,
                      getElementSizeFromImportBuffer(import_buffer));
          }

          partial_import_row_count_ += row_count_for_fragment;
          processed_import_row_count += row_count_for_fragment;
        }
        return true;
      };

  return new import_export::Loader(catalog, foreign_table_, callback, false);
}

import_export::Loader* ParquetDataWrapper::getMetadataLoader(
    Catalog_Namespace::Catalog& catalog,
    const LazyParquetImporter::RowGroupMetadataVector& metadata_vector) {
  auto callback = [this, &metadata_vector](
                      const std::vector<std::unique_ptr<
                          import_export::TypedImportBuffer>>& import_buffers,
                      std::vector<DataBlockPtr>& data_blocks,
                      size_t import_row_count) {
    size_t processed_import_row_count = 0;
    int row_group = metadata_vector[0].row_group_index;
    updateRowGroupMetadata(row_group);
    while (processed_import_row_count < import_row_count) {
      int fragment_index = row_count_ / foreign_table_->maxFragRows;
      size_t row_count_for_fragment;
      if (fragmentIsFull()) {
        row_count_for_fragment = std::min<size_t>(
            foreign_table_->maxFragRows, import_row_count - processed_import_row_count);
        initializeChunkBuffers(fragment_index);
        updateFragmentMap(fragment_index, row_group);
      } else {
        row_count_for_fragment = std::min<size_t>(
            foreign_table_->maxFragRows - (row_count_ % foreign_table_->maxFragRows),
            import_row_count - processed_import_row_count);
      }
      for (size_t i = 0; i < import_buffers.size(); i++) {
        auto& import_buffer = import_buffers[i];
        const auto column = import_buffer->getColumnDesc();
        auto column_id = column->columnId;
        ChunkKey chunk_key{db_id_, foreign_table_->tableId, column_id, fragment_index};
        const auto& metadata = metadata_vector[i];
        CHECK(metadata.row_group_index == row_group);
        if (!metadata.metadata_only) {
          loadChunk(column,
                    chunk_key,
                    data_blocks[i],
                    row_count_for_fragment,
                    processed_import_row_count,
                    true);
        } else {
          if (row_group_row_count_ == 0) {  // only load metadata once for each row group
            loadMetadataChunk(column,
                              chunk_key,
                              data_blocks[i],
                              metadata.num_elements,
                              metadata.has_nulls,
                              metadata.is_all_nulls);
          }
        }
      }
      row_count_ += row_count_for_fragment;
      processed_import_row_count += row_count_for_fragment;
      row_group_row_count_ += row_count_for_fragment;
    }
    return true;
  };

  return new import_export::Loader(catalog, foreign_table_, callback, false);
}

bool ParquetDataWrapper::newRowGroup(int row_group) {
  return current_row_group_ != row_group;
}

bool ParquetDataWrapper::fragmentIsFull() {
  return row_count_ != 0 && (row_count_ % foreign_table_->maxFragRows) == 0;
}

ForeignStorageBuffer* ParquetDataWrapper::getChunkBuffer(const ChunkKey& chunk_key) {
  return getBufferFromMapOrLoadBufferIntoMap(chunk_key);
}

void ParquetDataWrapper::populateMetadataForChunkKeyPrefix(
    const ChunkKey& chunk_key_prefix,
    ChunkMetadataVector& chunk_metadata_vector) {
  fetchChunkMetadata();
  auto iter_range = prefix_range(chunk_buffer_map_, chunk_key_prefix);
  for (auto it = iter_range.first; it != iter_range.second; ++it) {
    auto& buffer_chunk_key = it->first;
    auto& buffer = it->second;
    if (buffer->has_encoder) {
      auto chunk_metadata = std::make_shared<ChunkMetadata>();
      buffer->encoder->getMetadata(chunk_metadata);
      chunk_metadata_vector.emplace_back(buffer_chunk_key, chunk_metadata);
    }
  }
  chunk_buffer_map_.clear();
}

ParquetDataWrapper::IntervalsToLoad
ParquetDataWrapper::getRowGroupsColumnsAndFragmentsToLoad(const ChunkKey& chunk_key) {
  int fragment_index = chunk_key[CHUNK_KEY_FRAGMENT_IDX];
  auto frag_map_it = fragment_to_row_group_interval_map_.find(fragment_index);
  CHECK(frag_map_it != fragment_to_row_group_interval_map_.end());
  const auto& fragment_to_row_group_interval = frag_map_it->second;
  int start_row_group = fragment_to_row_group_interval.start_row_group_index;
  int end_row_group = fragment_to_row_group_interval.end_row_group_index;
  CHECK(end_row_group >= 0);
  Interval<FragmentType> fragment_interval = {frag_map_it->first, frag_map_it->first};
  auto frag_map_it_up = frag_map_it;
  while (frag_map_it_up != fragment_to_row_group_interval_map_.begin()) {
    --frag_map_it_up;
    if (frag_map_it_up->second.start_row_group_index == start_row_group) {
      fragment_interval.start = frag_map_it_up->first;
    } else {
      break;
    }
  }
  auto frag_map_it_down = frag_map_it;
  while ((++frag_map_it_down) != fragment_to_row_group_interval_map_.end()) {
    if (frag_map_it_down->second.end_row_group_index == end_row_group) {
      fragment_interval.end = frag_map_it_down->first;
    } else {
      break;
    }
  }
  IntervalsToLoad intervals;
  intervals.row_group_interval = {start_row_group, end_row_group};
  intervals.column_interval = foreign_table_column_map_.getPhysicalColumnSpan(
      chunk_key[CHUNK_KEY_COLUMN_IDX] - 1);
  intervals.fragment_interval = fragment_interval;
  return intervals;
}

ForeignStorageBuffer* ParquetDataWrapper::loadBufferIntoMap(const ChunkKey& chunk_key) {
  CHECK(chunk_buffer_map_.find(chunk_key) == chunk_buffer_map_.end());
  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);
  auto intervals = getRowGroupsColumnsAndFragmentsToLoad(chunk_key);

  initializeChunkBuffers(intervals.fragment_interval, intervals.column_interval);
  int first_fragment = intervals.fragment_interval.start;
  CHECK(fragment_to_row_group_interval_map_.find(first_fragment) !=
        fragment_to_row_group_interval_map_.end());
  partial_import_row_count_ =
      foreign_table_->maxFragRows * first_fragment -
      fragment_to_row_group_interval_map_[first_fragment].start_row_group_line;
  ;
  CHECK(partial_import_row_count_ >= 0);

  LazyParquetImporter::RowGroupMetadataVector metadata_vector;
  LazyParquetImporter importer(getChunkLoader(*catalog,
                                              intervals.fragment_interval,
                                              intervals.column_interval,
                                              chunk_key[CHUNK_KEY_DB_IDX]),
                               getFilePath(),
                               validateAndGetCopyParams(),
                               metadata_vector);
  int logical_col_idx =
      foreign_table_column_map_.getLogicalIndex(chunk_key[CHUNK_KEY_COLUMN_IDX] - 1);
  importer.partialImport(intervals.row_group_interval,
                         {logical_col_idx, logical_col_idx});

  return chunk_buffer_map_[chunk_key].get();
}

ForeignStorageBuffer* ParquetDataWrapper::getBufferFromMapOrLoadBufferIntoMap(
    const ChunkKey& chunk_key) {
  auto it = chunk_buffer_map_.find(chunk_key);
  if (it != chunk_buffer_map_.end()) {
    const auto& buffer = chunk_buffer_map_[chunk_key].get();
    return buffer;
  }

  return loadBufferIntoMap(chunk_key);
}

}  // namespace foreign_storage
