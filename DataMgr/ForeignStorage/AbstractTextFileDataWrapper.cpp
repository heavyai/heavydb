/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "AbstractTextFileDataWrapper.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>

#include <rapidjson/document.h>
#include <boost/filesystem.hpp>

#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/ForeignStorage/FileReader.h"
#include "DataMgr/ForeignStorage/ForeignTableSchema.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "ForeignStorageException.h"
#include "FsiJsonUtils.h"
#include "ImportExport/RenderGroupAnalyzer.h"
#include "Shared/misc.h"

namespace foreign_storage {
AbstractTextFileDataWrapper::AbstractTextFileDataWrapper()
    : db_id_(-1)
    , foreign_table_(nullptr)
    , user_mapping_(nullptr)
    , disable_cache_(false)
    , is_first_file_scan_call_(true)
    , is_file_scan_in_progress_(false) {}

AbstractTextFileDataWrapper::AbstractTextFileDataWrapper(
    const int db_id,
    const ForeignTable* foreign_table)
    : db_id_(db_id)
    , foreign_table_(foreign_table)
    , is_restored_(false)
    , user_mapping_(nullptr)
    , disable_cache_(false)
    , is_first_file_scan_call_(true)
    , is_file_scan_in_progress_(false) {}

AbstractTextFileDataWrapper::AbstractTextFileDataWrapper(
    const int db_id,
    const ForeignTable* foreign_table,
    const UserMapping* user_mapping,
    const bool disable_cache)
    : db_id_(db_id)
    , foreign_table_(foreign_table)
    , is_restored_(false)
    , user_mapping_(user_mapping)
    , disable_cache_(disable_cache)
    , is_first_file_scan_call_(true)
    , is_file_scan_in_progress_(false) {}

namespace {

void throw_fragment_id_out_of_bounds_error(const TableDescriptor* table,
                                           const int32_t fragment_id,
                                           const int32_t max_fragment_id) {
  throw RequestedFragmentIdOutOfBoundsException{
      "Attempting to populate fragment id " + std::to_string(fragment_id) +
      " for foreign table " + table->tableName +
      " which is greater than the maximum fragment id of " +
      std::to_string(max_fragment_id) + "."};
}

std::set<const ColumnDescriptor*> get_columns(const ChunkToBufferMap& buffers,
                                              const Catalog_Namespace::Catalog& catalog,
                                              const int32_t table_id,
                                              const int fragment_id) {
  CHECK(!buffers.empty());
  std::set<const ColumnDescriptor*> columns;
  for (const auto& entry : buffers) {
    CHECK_EQ(fragment_id, entry.first[CHUNK_KEY_FRAGMENT_IDX]);
    const auto column_id = entry.first[CHUNK_KEY_COLUMN_IDX];
    const auto column = catalog.getMetadataForColumn(table_id, column_id);
    columns.emplace(column);
  }
  return columns;
}

bool skip_metadata_scan(const ColumnDescriptor* column) {
  return column->columnType.is_dict_encoded_type();
}
}  // namespace

void AbstractTextFileDataWrapper::populateChunkMapForColumns(
    const std::set<const ColumnDescriptor*>& columns,
    const int fragment_id,
    const ChunkToBufferMap& buffers,
    std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map) {
  for (const auto column : columns) {
    ChunkKey data_chunk_key = {
        db_id_, foreign_table_->tableId, column->columnId, fragment_id};
    init_chunk_for_column(data_chunk_key,
                          chunk_metadata_map_,
                          buffers,
                          column_id_to_chunk_map[column->columnId]);
  }
}

void AbstractTextFileDataWrapper::populateChunkBuffers(
    const ChunkToBufferMap& required_buffers,
    const ChunkToBufferMap& optional_buffers,
    AbstractBuffer* delete_buffer) {
  auto timer = DEBUG_TIMER(__func__);
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  CHECK(!required_buffers.empty());

  auto fragment_id = required_buffers.begin()->first[CHUNK_KEY_FRAGMENT_IDX];
  auto required_columns =
      get_columns(required_buffers, *catalog, foreign_table_->tableId, fragment_id);
  std::map<int, Chunk_NS::Chunk> column_id_to_chunk_map;
  populateChunkMapForColumns(
      required_columns, fragment_id, required_buffers, column_id_to_chunk_map);

  if (!optional_buffers.empty()) {
    auto optional_columns =
        get_columns(optional_buffers, *catalog, foreign_table_->tableId, fragment_id);
    populateChunkMapForColumns(
        optional_columns, fragment_id, optional_buffers, column_id_to_chunk_map);
  }
  populateChunks(column_id_to_chunk_map, fragment_id, delete_buffer);
  if (!is_file_scan_in_progress_) {
    updateMetadata(column_id_to_chunk_map, fragment_id);
  }
}

// if column was skipped during scan, update metadata now
void AbstractTextFileDataWrapper::updateMetadata(
    std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
    int fragment_id) {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  for (auto& entry : column_id_to_chunk_map) {
    const auto& column =
        catalog->getMetadataForColumn(foreign_table_->tableId, entry.first);
    if (skip_metadata_scan(column)) {
      ChunkKey data_chunk_key = {
          db_id_, foreign_table_->tableId, column->columnId, fragment_id};
      if (column->columnType.is_varlen_indeed()) {
        data_chunk_key.emplace_back(1);
      }
      CHECK(chunk_metadata_map_.find(data_chunk_key) != chunk_metadata_map_.end());
      // Allocate new shared_ptr for metadata so we dont modify old one which may be
      // used by executor
      auto cached_metadata_previous =
          shared::get_from_map(chunk_metadata_map_, data_chunk_key);
      shared::get_from_map(chunk_metadata_map_, data_chunk_key) =
          std::make_shared<ChunkMetadata>();
      auto cached_metadata = shared::get_from_map(chunk_metadata_map_, data_chunk_key);
      *cached_metadata = *cached_metadata_previous;
      auto chunk_metadata =
          entry.second.getBuffer()->getEncoder()->getMetadata(column->columnType);
      cached_metadata->chunkStats.max = chunk_metadata->chunkStats.max;
      cached_metadata->chunkStats.min = chunk_metadata->chunkStats.min;
      cached_metadata->chunkStats.has_nulls = chunk_metadata->chunkStats.has_nulls;
      cached_metadata->numBytes = entry.second.getBuffer()->size();
    }
  }
}

/**
 * Data structure containing data and metadata gotten from parsing a set of file
 * regions.
 */
struct ParseFileRegionResult {
  size_t file_offset;
  size_t row_count;
  std::map<int, DataBlockPtr> column_id_to_data_blocks_map;
  std::set<size_t> rejected_row_indices;

  bool operator<(const ParseFileRegionResult& other) const {
    return file_offset < other.file_offset;
  }
};

namespace {
void throw_unexpected_number_of_items(const size_t num_expected,
                                      const size_t num_loaded,
                                      const std::string& item_type,
                                      const std::string& foreign_table_name) {
  try {
    foreign_storage::throw_unexpected_number_of_items(
        num_expected, num_loaded, item_type);
  } catch (const foreign_storage::ForeignStorageException& except) {
    throw foreign_storage::ForeignStorageException(
        std::string(except.what()) + " Foreign table: " + foreign_table_name);
  }
}

}  // namespace

/**
 * Parses a set of file regions given a handle to the file and range of indexes
 * for the file regions to be parsed.
 */
ParseFileRegionResult parse_file_regions(
    const FileRegions& file_regions,
    const size_t start_index,
    const size_t end_index,
    FileReader& file_reader,
    ParseBufferRequest& parse_file_request,
    const std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
    const TextFileBufferParser& parser) {
  auto timer = DEBUG_TIMER(__func__);
  ParseFileRegionResult load_file_region_result{};
  load_file_region_result.file_offset = file_regions[start_index].first_row_file_offset;
  load_file_region_result.row_count = 0;

  ParseBufferResult result;
  for (size_t i = start_index; i <= end_index; i++) {
    CHECK(file_regions[i].region_size <= parse_file_request.buffer_size);
    auto read_size = file_reader.readRegion(parse_file_request.buffer.get(),
                                            file_regions[i].first_row_file_offset,
                                            file_regions[i].region_size);
    if (file_regions[i].region_size != read_size) {
      throw_unexpected_number_of_items(file_regions[i].region_size,
                                       read_size,
                                       "bytes",
                                       parse_file_request.getTableName());
    }
    parse_file_request.begin_pos = 0;
    parse_file_request.end_pos = file_regions[i].region_size;
    parse_file_request.first_row_index = file_regions[i].first_row_index;
    parse_file_request.file_offset = file_regions[i].first_row_file_offset;
    parse_file_request.process_row_count = file_regions[i].row_count;

    result = parser.parseBuffer(parse_file_request, i == end_index);
    CHECK_EQ(file_regions[i].row_count, result.row_count);
    for (const auto& rejected_row_index : result.rejected_rows) {
      load_file_region_result.rejected_row_indices.insert(
          load_file_region_result.row_count + rejected_row_index);
    }
    load_file_region_result.row_count += result.row_count;
  }
  load_file_region_result.column_id_to_data_blocks_map =
      result.column_id_to_data_blocks_map;
  return load_file_region_result;
}

namespace {

/**
 * Gets the appropriate buffer size to be used when processing file(s).
 */
size_t get_buffer_size(const import_export::CopyParams& copy_params,
                       const bool size_known,
                       const size_t file_size) {
  size_t buffer_size = copy_params.buffer_size;
  if (size_known && file_size < buffer_size) {
    buffer_size = file_size + 1;  // +1 for end of line character, if missing
  }
  return buffer_size;
}

size_t get_buffer_size(const FileRegions& file_regions) {
  size_t buffer_size = 0;
  for (const auto& file_region : file_regions) {
    buffer_size = std::max(buffer_size, file_region.region_size);
  }
  CHECK(buffer_size);
  return buffer_size;
}

/**
 * Gets the appropriate number of threads to be used for concurrent
 * processing within the data wrapper.
 */
size_t get_thread_count(const import_export::CopyParams& copy_params,
                        const bool size_known,
                        const size_t file_size,
                        const size_t buffer_size) {
  size_t thread_count = copy_params.threads;
  if (thread_count == 0) {
    thread_count = std::thread::hardware_concurrency();
  }
  if (size_known && file_size > 0) {
    size_t num_buffers_in_file = (file_size + buffer_size - 1) / buffer_size;
    if (num_buffers_in_file < thread_count) {
      thread_count = num_buffers_in_file;
    }
  }
  CHECK_GT(thread_count, static_cast<size_t>(0));
  return thread_count;
}

size_t get_thread_count(const import_export::CopyParams& copy_params,
                        const FileRegions& file_regions) {
  size_t thread_count = copy_params.threads;
  if (thread_count == 0) {
    thread_count =
        std::min<size_t>(std::thread::hardware_concurrency(), file_regions.size());
  }
  CHECK_GT(thread_count, static_cast<size_t>(0));
  return thread_count;
}

void resize_delete_buffer(AbstractBuffer* delete_buffer,
                          const size_t chunk_element_count) {
  if (delete_buffer->size() < chunk_element_count) {
    auto remaining_rows = chunk_element_count - delete_buffer->size();
    std::vector<int8_t> data(remaining_rows, false);
    delete_buffer->append(data.data(), remaining_rows);
  }
}

bool no_deferred_requests(MetadataScanMultiThreadingParams& multi_threading_params) {
  std::unique_lock<std::mutex> deferred_requests_lock(
      multi_threading_params.deferred_requests_mutex);
  return multi_threading_params.deferred_requests.empty();
}

bool is_file_scan_finished(const FileReader* file_reader,
                           MetadataScanMultiThreadingParams& multi_threading_params) {
  return file_reader->isScanFinished() && no_deferred_requests(multi_threading_params);
}

}  // namespace

void AbstractTextFileDataWrapper::populateChunks(
    std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
    int fragment_id,
    AbstractBuffer* delete_buffer) {
  const auto copy_params = getFileBufferParser().validateAndGetCopyParams(foreign_table_);

  CHECK(!column_id_to_chunk_map.empty());

  // check to see if a iterative scan step is required
  auto file_regions_it = fragment_id_to_file_regions_map_.find(fragment_id);
  if (file_regions_it == fragment_id_to_file_regions_map_.end() ||
      is_file_scan_in_progress_) {
    // check to see if there is more foreign data to scan
    if (is_first_file_scan_call_ ||
        !is_file_scan_finished(file_reader_.get(), multi_threading_params_)) {
      // NOTE: we can only guarantee the current `fragment_id` is fully done
      // iterative scan if either
      //   1) the scan is finished OR
      //   2) `fragment_id+1` exists in the internal map
      // this is why `fragment_id+1` is checked for below
      auto file_regions_it_one_ahead =
          fragment_id_to_file_regions_map_.find(fragment_id + 1);
      if (is_first_file_scan_call_ ||
          (file_regions_it_one_ahead == fragment_id_to_file_regions_map_.end())) {
        ChunkMetadataVector chunk_metadata_vector;
        IterativeFileScanParameters iterative_params{
            column_id_to_chunk_map, fragment_id, delete_buffer};
        iterativeFileScan(chunk_metadata_vector, iterative_params);
      }
    }

    file_regions_it = fragment_id_to_file_regions_map_.find(fragment_id);
    if (file_regions_it == fragment_id_to_file_regions_map_.end()) {
      CHECK(is_file_scan_finished(file_reader_.get(), multi_threading_params_));
      is_file_scan_in_progress_ = false;  // conclude the iterative scan is finished
      is_first_file_scan_call_ =
          true;  // any subsequent iterative request can assume they will be the first
      throw_fragment_id_out_of_bounds_error(
          foreign_table_, fragment_id, fragment_id_to_file_regions_map_.rbegin()->first);
    } else {
      // iterative scan is required to have loaded all required chunks thus we
      // can exit early
      return;
    }
  }
  CHECK(file_regions_it != fragment_id_to_file_regions_map_.end());

  const auto& file_regions = file_regions_it->second;

  // File roll off can lead to empty file regions.
  if (file_regions.empty()) {
    return;
  }

  const auto buffer_size = get_buffer_size(file_regions);
  const auto thread_count = get_thread_count(copy_params, file_regions);

  const int batch_size = (file_regions.size() + thread_count - 1) / thread_count;

  std::vector<ParseBufferRequest> parse_file_requests{};
  parse_file_requests.reserve(thread_count);
  std::vector<std::future<ParseFileRegionResult>> futures{};
  std::set<int> column_filter_set;
  for (const auto& pair : column_id_to_chunk_map) {
    column_filter_set.insert(pair.first);
  }

  std::vector<std::unique_ptr<FileReader>> file_readers;
  rapidjson::Value reader_metadata(rapidjson::kObjectType);
  rapidjson::Document d;
  auto& server_options = foreign_table_->foreign_server->options;
  file_reader_->serialize(reader_metadata, d.GetAllocator());
  const auto file_path = getFullFilePath(foreign_table_);
  auto& parser = getFileBufferParser();

  for (size_t i = 0; i < file_regions.size(); i += batch_size) {
    parse_file_requests.emplace_back(buffer_size,
                                     copy_params,
                                     db_id_,
                                     foreign_table_,
                                     column_filter_set,
                                     file_path,
                                     &render_group_analyzer_map_,
                                     delete_buffer != nullptr);
    auto start_index = i;
    auto end_index =
        std::min<size_t>(start_index + batch_size - 1, file_regions.size() - 1);

    if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
      file_readers.emplace_back(std::make_unique<LocalMultiFileReader>(
          file_path, copy_params, reader_metadata));
    } else {
      UNREACHABLE();
    }

    futures.emplace_back(std::async(std::launch::async,
                                    parse_file_regions,
                                    std::ref(file_regions),
                                    start_index,
                                    end_index,
                                    std::ref(*(file_readers.back())),
                                    std::ref(parse_file_requests.back()),
                                    std::ref(column_id_to_chunk_map),
                                    std::ref(parser)));
  }

  for (auto& future : futures) {
    future.wait();
  }

  std::vector<ParseFileRegionResult> load_file_region_results{};
  for (auto& future : futures) {
    load_file_region_results.emplace_back(future.get());
  }

  std::set<size_t> chunk_rejected_row_indices;
  size_t chunk_offset = 0;
  for (auto result : load_file_region_results) {
    for (auto& [column_id, chunk] : column_id_to_chunk_map) {
      chunk.appendData(
          result.column_id_to_data_blocks_map[column_id], result.row_count, 0);
    }
    for (const auto& rejected_row_index : result.rejected_row_indices) {
      chunk_rejected_row_indices.insert(rejected_row_index + chunk_offset);
    }
    chunk_offset += result.row_count;
  }

  if (delete_buffer) {
    // ensure delete buffer is sized appropriately
    resize_delete_buffer(delete_buffer, chunk_offset);

    auto delete_buffer_data = delete_buffer->getMemoryPtr();
    for (const auto rejected_row_index : chunk_rejected_row_indices) {
      delete_buffer_data[rejected_row_index] = true;
    }
  }
}

/**
 * Number of rows to process given the current position defined by
 * `start_row_index`, the max fragment size and the number of rows left to
 * process.
 */
size_t num_rows_to_process(const size_t start_row_index,
                           const size_t max_fragment_size,
                           const size_t rows_remaining) {
  size_t start_position_in_fragment = start_row_index % max_fragment_size;
  return std::min<size_t>(rows_remaining, max_fragment_size - start_position_in_fragment);
}

/**
 * Given a start row index, maximum fragment size, and number of rows in
 * a buffer, this function returns a vector indicating how the rows in
 * the buffer should be partitioned in order to fill up available fragment
 * slots while staying within the capacity of fragments.
 */
std::vector<size_t> partition_by_fragment(const size_t start_row_index,
                                          const size_t max_fragment_size,
                                          const size_t buffer_row_count) {
  CHECK(buffer_row_count > 0);
  std::vector<size_t> partitions{};
  size_t remaining_rows_in_last_fragment;
  if (start_row_index % max_fragment_size == 0) {
    remaining_rows_in_last_fragment = 0;
  } else {
    remaining_rows_in_last_fragment =
        max_fragment_size - (start_row_index % max_fragment_size);
  }
  if (buffer_row_count <= remaining_rows_in_last_fragment) {
    partitions.emplace_back(buffer_row_count);
  } else {
    if (remaining_rows_in_last_fragment > 0) {
      partitions.emplace_back(remaining_rows_in_last_fragment);
    }
    size_t remaining_buffer_row_count =
        buffer_row_count - remaining_rows_in_last_fragment;
    while (remaining_buffer_row_count > 0) {
      partitions.emplace_back(
          std::min<size_t>(remaining_buffer_row_count, max_fragment_size));
      remaining_buffer_row_count -= partitions.back();
    }
  }
  return partitions;
}

/**
 * Gets the next metadata scan request object from the pending requests queue.
 * A null optional is returned if there are no further requests to be processed.
 */
std::optional<ParseBufferRequest> get_next_scan_request(
    MetadataScanMultiThreadingParams& multi_threading_params) {
  std::unique_lock<std::mutex> pending_requests_lock(
      multi_threading_params.pending_requests_mutex);
  multi_threading_params.pending_requests_condition.wait(
      pending_requests_lock, [&multi_threading_params] {
        return !multi_threading_params.pending_requests.empty() ||
               !multi_threading_params.continue_processing;
      });
  if (multi_threading_params.pending_requests.empty()) {
    return {};
  }
  auto request = std::move(multi_threading_params.pending_requests.front());
  multi_threading_params.pending_requests.pop();
  pending_requests_lock.unlock();
  multi_threading_params.pending_requests_condition.notify_all();
  return std::move(request);
}

/**
 * Creates a new file region based on metadata from parsed file buffers and
 * adds the new region to the fragment id to file regions map.
 */
void add_file_region(std::map<int, FileRegions>& fragment_id_to_file_regions_map,
                     int fragment_id,
                     size_t first_row_index,
                     const ParseBufferResult& result,
                     const std::string& file_path) {
  fragment_id_to_file_regions_map[fragment_id].emplace_back(
      // file naming is handled by FileReader
      FileRegion(file_path,
                 result.row_offsets.front(),
                 first_row_index,
                 result.row_count,
                 result.row_offsets.back() - result.row_offsets.front()));
}

/**
 * Updates the statistics metadata encapsulated in the encoder
 * given new data in a data block.
 */
void update_stats(Encoder* encoder,
                  const SQLTypeInfo& column_type,
                  DataBlockPtr data_block,
                  const size_t row_count) {
  if (column_type.is_array()) {
    encoder->updateStats(data_block.arraysPtr, 0, row_count);
  } else if (!column_type.is_varlen()) {
    encoder->updateStats(data_block.numbersPtr, row_count);
  } else {
    encoder->updateStats(data_block.stringsPtr, 0, row_count);
  }
}
namespace {
foreign_storage::ForeignStorageCache* get_cache_if_enabled(
    std::shared_ptr<Catalog_Namespace::Catalog>& catalog,
    const bool disable_cache) {
  if (!disable_cache && catalog->getDataMgr()
                            .getPersistentStorageMgr()
                            ->getDiskCacheConfig()
                            .isEnabledForFSI()) {
    return catalog->getDataMgr().getPersistentStorageMgr()->getDiskCache();
  } else {
    return nullptr;
  }
}
}  // namespace

// If cache is enabled, populate cached_chunks buffers with data blocks
void cache_blocks(std::map<ChunkKey, Chunk_NS::Chunk>& cached_chunks,
                  DataBlockPtr data_block,
                  size_t row_count,
                  ChunkKey& chunk_key,
                  const ColumnDescriptor* column,
                  bool is_first_block,
                  bool disable_cache) {
  auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(chunk_key[CHUNK_KEY_DB_IDX]);
  CHECK(catalog);
  auto cache = get_cache_if_enabled(catalog, disable_cache);
  if (cache) {
    // This extra filter needs to be here because this wrapper is the only one that
    // accesses the cache directly and it should not be inserting chunks which are not
    // mapped to the current leaf (in distributed mode).
    if (key_does_not_shard_to_leaf(chunk_key)) {
      return;
    }

    ChunkKey index_key = {chunk_key[CHUNK_KEY_DB_IDX],
                          chunk_key[CHUNK_KEY_TABLE_IDX],
                          chunk_key[CHUNK_KEY_COLUMN_IDX],
                          chunk_key[CHUNK_KEY_FRAGMENT_IDX],
                          2};
    // Create actual data chunks to prepopulate cache
    if (cached_chunks.find(chunk_key) == cached_chunks.end()) {
      cached_chunks[chunk_key] = Chunk_NS::Chunk{column, false};
      cached_chunks[chunk_key].setBuffer(
          cache->getChunkBufferForPrecaching(chunk_key, is_first_block));
      if (column->columnType.is_varlen_indeed()) {
        cached_chunks[chunk_key].setIndexBuffer(
            cache->getChunkBufferForPrecaching(index_key, is_first_block));
      }
      if (is_first_block) {
        cached_chunks[chunk_key].initEncoder();
      }
    }
    cached_chunks[chunk_key].appendData(data_block, row_count, 0);
  }
}

void append_data_block_to_chunk(
    const foreign_storage::IterativeFileScanParameters& file_scan_param,
    DataBlockPtr data_block,
    size_t row_count,
    const int column_id,
    const ColumnDescriptor* column,
    const size_t element_count_required) {
  auto chunk = shared::get_from_map(file_scan_param.column_id_to_chunk_map, column_id);

  auto& conditional_variable = file_scan_param.getChunkConditionalVariable(column_id);
  {
    std::unique_lock<std::mutex> chunk_lock(file_scan_param.getChunkMutex(column_id));
    conditional_variable.wait(chunk_lock, [element_count_required, &chunk]() {
      return chunk.getBuffer()->getEncoder()->getNumElems() == element_count_required;
    });

    chunk.appendData(data_block, row_count, 0);
  }

  conditional_variable
      .notify_all();  // notify any threads waiting on the correct element count
}

/**
 * Partition data blocks such that dictionary encoded columns are disjoint of
 * other columns
 *
 * @return a pair of DataBlockPtr maps with the first being columns that are
 * string dictionary encoded and the second being the complement
 */
std::pair<std::map<int, DataBlockPtr>, std::map<int, DataBlockPtr>> partition_data_blocks(
    const std::map<int, const ColumnDescriptor*>& column_by_id,
    const std::map<int, DataBlockPtr>& data_blocks) {
  std::map<int, DataBlockPtr> dict_encoded_data_blocks;
  std::map<int, DataBlockPtr> none_dict_encoded_data_blocks;
  for (auto& [column_id, data_block] : data_blocks) {
    const auto column = shared::get_from_map(column_by_id, column_id);
    if (column->columnType.is_dict_encoded_string()) {
      dict_encoded_data_blocks[column_id] = data_block;
    } else {
      none_dict_encoded_data_blocks[column_id] = data_block;
    }
  }
  return {dict_encoded_data_blocks, none_dict_encoded_data_blocks};
}

void update_delete_buffer(
    const ParseBufferRequest& request,
    const ParseBufferResult& result,
    const foreign_storage::IterativeFileScanParameters& file_scan_param,
    const size_t start_position_in_fragment) {
  if (file_scan_param.delete_buffer) {
    std::unique_lock delete_buffer_lock(file_scan_param.delete_buffer_mutex);
    auto& delete_buffer = file_scan_param.delete_buffer;
    auto chunk_offset = start_position_in_fragment;
    auto chunk_element_count = chunk_offset + request.processed_row_count;

    // ensure delete buffer is sized appropriately
    resize_delete_buffer(delete_buffer, chunk_element_count);

    auto delete_buffer_data = delete_buffer->getMemoryPtr();
    for (const auto rejected_row_index : result.rejected_rows) {
      CHECK(rejected_row_index + chunk_offset < delete_buffer->size());
      delete_buffer_data[rejected_row_index + chunk_offset] = true;
    }
  }
}

void populate_chunks_using_data_blocks(
    MetadataScanMultiThreadingParams& multi_threading_params,
    int fragment_id,
    const ParseBufferRequest& request,
    ParseBufferResult& result,
    std::map<int, const ColumnDescriptor*>& column_by_id,
    std::map<int, FileRegions>& fragment_id_to_file_regions_map,
    const foreign_storage::IterativeFileScanParameters& file_scan_param,
    const size_t expected_current_element_count) {
  std::unique_lock<std::mutex> lock(multi_threading_params.chunk_encoder_buffers_mutex);
  // File regions should be added in same order as appendData
  add_file_region(fragment_id_to_file_regions_map,
                  fragment_id,
                  request.first_row_index,
                  result,
                  request.getFilePath());
  CHECK_EQ(fragment_id, file_scan_param.fragment_id);

  // start string encoding asynchronously
  std::vector<std::pair<const size_t, std::future<int8_t*>>>
      encoded_data_block_ptrs_futures;

  for (const auto& import_buffer : request.import_buffers) {
    if (import_buffer == nullptr) {
      continue;
    }

    if (import_buffer->getTypeInfo().is_dict_encoded_string()) {
      auto string_payload_ptr = import_buffer->getStringBuffer();
      CHECK_EQ(kENCODING_DICT, import_buffer->getTypeInfo().get_compression());

      auto column_id = import_buffer->getColumnDesc()->columnId;
      encoded_data_block_ptrs_futures.emplace_back(std::make_pair(
          column_id, std::async(std::launch::async, [&import_buffer, string_payload_ptr] {
            import_buffer->addDictEncodedString(*string_payload_ptr);
            return import_buffer->getStringDictBuffer();
          })));
    }
  }

  auto process_subset_of_data_blocks =
      [&](const std::map<int, DataBlockPtr>& data_blocks) {
        for (auto& [column_id, data_block] : data_blocks) {
          const auto column = column_by_id[column_id];
          lock.unlock();  // unlock the fragment based lock in order to achieve better
          // performance
          append_data_block_to_chunk(file_scan_param,
                                     data_block,
                                     result.row_count,
                                     column_id,
                                     column,
                                     expected_current_element_count);
          lock.lock();
        }
      };

  auto [dict_encoded_data_blocks, none_dict_encoded_data_blocks] =
      partition_data_blocks(column_by_id, result.column_id_to_data_blocks_map);

  process_subset_of_data_blocks(
      none_dict_encoded_data_blocks);  // skip dict string columns

  // wait for the async requests we made for string dictionary
  for (auto& encoded_ptr_future : encoded_data_block_ptrs_futures) {
    encoded_ptr_future.second.wait();
  }
  for (auto& encoded_ptr_future : encoded_data_block_ptrs_futures) {
    CHECK_GT(dict_encoded_data_blocks.count(encoded_ptr_future.first), 0UL);
    dict_encoded_data_blocks[encoded_ptr_future.first].numbersPtr =
        encoded_ptr_future.second.get();
  }

  process_subset_of_data_blocks(
      dict_encoded_data_blocks);  // process only dict string columns
}

/**
 * Updates metadata encapsulated in encoders for all table columns given
 * new data blocks gotten from parsing a new set of rows in a file buffer.
 * If cache is available, also append the data_blocks to chunks in the cache
 */
void process_data_blocks(MetadataScanMultiThreadingParams& multi_threading_params,
                         int fragment_id,
                         const ParseBufferRequest& request,
                         ParseBufferResult& result,
                         std::map<int, const ColumnDescriptor*>& column_by_id,
                         std::map<int, FileRegions>& fragment_id_to_file_regions_map) {
  std::lock_guard<std::mutex> lock(multi_threading_params.chunk_encoder_buffers_mutex);
  // File regions should be added in same order as appendData
  add_file_region(fragment_id_to_file_regions_map,
                  fragment_id,
                  request.first_row_index,
                  result,
                  request.getFilePath());

  for (auto& [column_id, data_block] : result.column_id_to_data_blocks_map) {
    ChunkKey chunk_key{request.db_id, request.getTableId(), column_id, fragment_id};
    const auto column = column_by_id[column_id];
    if (column->columnType.is_varlen_indeed()) {
      chunk_key.emplace_back(1);
    }
    if (multi_threading_params.chunk_encoder_buffers.find(chunk_key) ==
        multi_threading_params.chunk_encoder_buffers.end()) {
      multi_threading_params.chunk_encoder_buffers[chunk_key] =
          std::make_unique<ForeignStorageBuffer>();
      multi_threading_params.chunk_encoder_buffers[chunk_key]->initEncoder(
          column->columnType);
    }
    update_stats(multi_threading_params.chunk_encoder_buffers[chunk_key]->getEncoder(),
                 column->columnType,
                 data_block,
                 result.row_count);
    size_t num_elements = multi_threading_params.chunk_encoder_buffers[chunk_key]
                              ->getEncoder()
                              ->getNumElems() +
                          result.row_count;
    multi_threading_params.chunk_encoder_buffers[chunk_key]->getEncoder()->setNumElems(
        num_elements);
    cache_blocks(
        multi_threading_params.cached_chunks,
        data_block,
        result.row_count,
        chunk_key,
        column,
        (num_elements - result.row_count) == 0,  // Is the first block added to this chunk
        multi_threading_params.disable_cache);
  }
}

/**
 * Adds the request object for a processed request back to the request pool
 * for reuse in subsequent requests.
 */
void add_request_to_pool(MetadataScanMultiThreadingParams& multi_threading_params,
                         ParseBufferRequest& request) {
  std::unique_lock<std::mutex> completed_requests_queue_lock(
      multi_threading_params.request_pool_mutex);
  multi_threading_params.request_pool.emplace(std::move(request));
  completed_requests_queue_lock.unlock();
  multi_threading_params.request_pool_condition.notify_all();
}

/**
 * Consumes and processes metadata scan requests from a pending requests queue
 * and updates existing metadata objects based on newly scanned metadata.
 */
void scan_metadata(MetadataScanMultiThreadingParams& multi_threading_params,
                   std::map<int, FileRegions>& fragment_id_to_file_regions_map,
                   const TextFileBufferParser& parser) {
  std::map<int, const ColumnDescriptor*> column_by_id{};
  while (true) {
    auto request_opt = get_next_scan_request(multi_threading_params);
    if (!request_opt.has_value()) {
      break;
    }
    auto& request = request_opt.value();
    try {
      if (column_by_id.empty()) {
        for (const auto column : request.getColumns()) {
          column_by_id[column->columnId] = column;
        }
      }
      auto partitions = partition_by_fragment(
          request.first_row_index, request.getMaxFragRows(), request.buffer_row_count);
      request.begin_pos = 0;
      size_t row_index = request.first_row_index;
      for (const auto partition : partitions) {
        request.process_row_count = partition;
        for (const auto& import_buffer : request.import_buffers) {
          if (import_buffer != nullptr) {
            import_buffer->clear();
          }
        }
        auto result = parser.parseBuffer(request, true);
        int fragment_id = row_index / request.getMaxFragRows();
        process_data_blocks(multi_threading_params,
                            fragment_id,
                            request,
                            result,
                            column_by_id,
                            fragment_id_to_file_regions_map);
        row_index += result.row_count;
        request.begin_pos = result.row_offsets.back() - request.file_offset;
      }
    } catch (...) {
      // Re-add request to pool so we dont block any other threads
      {
        std::lock_guard<std::mutex> pending_requests_lock(
            multi_threading_params.pending_requests_mutex);
        multi_threading_params.continue_processing = false;
      }
      add_request_to_pool(multi_threading_params, request);
      throw;
    }
    add_request_to_pool(multi_threading_params, request);
  }
}

/**
 * Gets a request from the metadata scan request pool.
 */
ParseBufferRequest get_request_from_pool(
    MetadataScanMultiThreadingParams& multi_threading_params) {
  std::unique_lock<std::mutex> request_pool_lock(
      multi_threading_params.request_pool_mutex);
  multi_threading_params.request_pool_condition.wait(
      request_pool_lock,
      [&multi_threading_params] { return !multi_threading_params.request_pool.empty(); });
  auto request = std::move(multi_threading_params.request_pool.front());
  multi_threading_params.request_pool.pop();
  request_pool_lock.unlock();
  CHECK(request.buffer);
  return request;
}

/*
 * Defer processing a request until next iteration. The use case for this is
 * during an iterative file scan, some requests must defer processing until
 * the correct fragment is being processed.
 */
void defer_scan_request(MetadataScanMultiThreadingParams& multi_threading_params,
                        ParseBufferRequest& request) {
  std::unique_lock<std::mutex> deferred_requests_lock(
      multi_threading_params.deferred_requests_mutex);
  multi_threading_params.deferred_requests.emplace(std::move(request));
}

/*
 * Dispatch all requests that are currently deferred onto the pending request queue.
 */
void dispatch_all_deferred_requests(
    MetadataScanMultiThreadingParams& multi_threading_params) {
  std::unique_lock<std::mutex> deferred_requests_lock(
      multi_threading_params.deferred_requests_mutex);
  {
    std::unique_lock<std::mutex> pending_requests_lock(
        multi_threading_params.pending_requests_mutex);

    while (!multi_threading_params.deferred_requests.empty()) {
      auto& request = multi_threading_params.deferred_requests.front();
      multi_threading_params.pending_requests.emplace(std::move(request));
      multi_threading_params.deferred_requests.pop();
    }
    multi_threading_params.pending_requests_condition.notify_all();
  }
}

/**
 * Dispatches a new metadata scan request by adding the request to
 * the pending requests queue to be consumed by a worker thread.
 */
void dispatch_scan_request(MetadataScanMultiThreadingParams& multi_threading_params,
                           ParseBufferRequest& request) {
  {
    std::unique_lock<std::mutex> pending_requests_lock(
        multi_threading_params.pending_requests_mutex);
    multi_threading_params.pending_requests.emplace(std::move(request));
  }
  multi_threading_params.pending_requests_condition.notify_all();
}

/**
 * Consumes and processes scan requests from a pending requests queue
 * and populates chunks during an iterative file scan
 */
void populate_chunks(MetadataScanMultiThreadingParams& multi_threading_params,
                     std::map<int, FileRegions>& fragment_id_to_file_regions_map,
                     const TextFileBufferParser& parser,
                     foreign_storage::IterativeFileScanParameters& file_scan_param) {
  std::map<int, const ColumnDescriptor*> column_by_id{};
  while (true) {
    auto request_opt = get_next_scan_request(multi_threading_params);
    if (!request_opt.has_value()) {
      break;
    }
    ParseBufferRequest& request = request_opt.value();
    try {
      if (column_by_id.empty()) {
        for (const auto column : request.getColumns()) {
          column_by_id[column->columnId] = column;
        }
      }
      CHECK_LE(request.processed_row_count, request.buffer_row_count);
      for (size_t num_rows_left_to_process =
               request.buffer_row_count - request.processed_row_count;
           num_rows_left_to_process > 0;
           num_rows_left_to_process =
               request.buffer_row_count - request.processed_row_count) {
        // NOTE: `request.begin_pos` state is required to be set correctly by this point
        // in execution
        size_t row_index = request.first_row_index + request.processed_row_count;
        int fragment_id = row_index / request.getMaxFragRows();
        if (fragment_id >
            file_scan_param.fragment_id) {  // processing must continue next iteration
          defer_scan_request(multi_threading_params, request);
          return;
        }
        request.process_row_count = num_rows_to_process(
            row_index, request.getMaxFragRows(), num_rows_left_to_process);
        for (const auto& import_buffer : request.import_buffers) {
          if (import_buffer != nullptr) {
            import_buffer->clear();
          }
        }
        auto result = parser.parseBuffer(request, true, true, true);
        size_t start_position_in_fragment = row_index % request.getMaxFragRows();
        populate_chunks_using_data_blocks(multi_threading_params,
                                          fragment_id,
                                          request,
                                          result,
                                          column_by_id,
                                          fragment_id_to_file_regions_map,
                                          file_scan_param,
                                          start_position_in_fragment);

        request.processed_row_count += result.row_count;
        request.begin_pos = result.row_offsets.back() - request.file_offset;

        update_delete_buffer(
            request, result, file_scan_param, start_position_in_fragment);
      }

    } catch (...) {
      // Re-add request to pool so we dont block any other threads
      {
        std::lock_guard<std::mutex> pending_requests_lock(
            multi_threading_params.pending_requests_mutex);
        multi_threading_params.continue_processing = false;
      }
      add_request_to_pool(multi_threading_params, request);
      throw;
    }
    add_request_to_pool(multi_threading_params, request);
  }
}

/**
 * Optionally resizes the given buffer if the buffer size
 * is less than the current buffer allocation size.
 */
void resize_buffer_if_needed(std::unique_ptr<char[]>& buffer,
                             size_t& buffer_size,
                             const size_t alloc_size) {
  CHECK_LE(buffer_size, alloc_size);
  if (buffer_size < alloc_size) {
    buffer = std::make_unique<char[]>(alloc_size);
    buffer_size = alloc_size;
  }
}

void reset_multithreading_params(
    foreign_storage::MetadataScanMultiThreadingParams& multi_threading_params) {
  multi_threading_params.request_pool = {};
  multi_threading_params.cached_chunks = {};
  multi_threading_params.pending_requests = {};
  multi_threading_params.deferred_requests = {};
  multi_threading_params.chunk_encoder_buffers.clear();
}

/**
 * Reads from a text file iteratively and dispatches metadata scan
 * requests that are processed by worker threads.
 */
void dispatch_scan_requests(
    const foreign_storage::ForeignTable* table,
    const size_t& buffer_size,
    const std::string& file_path,
    FileReader& file_reader,
    const import_export::CopyParams& copy_params,
    MetadataScanMultiThreadingParams& multi_threading_params,
    size_t& first_row_index_in_buffer,
    size_t& current_file_offset,
    const TextFileBufferParser& parser,
    const foreign_storage::IterativeFileScanParameters* file_scan_param,
    foreign_storage::AbstractTextFileDataWrapper::ResidualBuffer&
        iterative_residual_buffer,
    const bool is_first_file_scan_call) {
  auto& alloc_size = iterative_residual_buffer.alloc_size;
  auto& residual_buffer = iterative_residual_buffer.residual_data;
  auto& residual_buffer_size = iterative_residual_buffer.residual_buffer_size;
  auto& residual_buffer_alloc_size = iterative_residual_buffer.residual_buffer_alloc_size;

  if (is_first_file_scan_call) {
    alloc_size = buffer_size;
    residual_buffer = std::make_unique<char[]>(alloc_size);
    residual_buffer_size = 0;
    residual_buffer_alloc_size = alloc_size;
  } else if (!no_deferred_requests(multi_threading_params)) {
    dispatch_all_deferred_requests(multi_threading_params);
  }

  while (!file_reader.isScanFinished()) {
    {
      std::lock_guard<std::mutex> pending_requests_lock(
          multi_threading_params.pending_requests_mutex);
      if (!multi_threading_params.continue_processing) {
        break;
      }
    }
    auto request = get_request_from_pool(multi_threading_params);
    request.full_path = file_reader.getCurrentFilePath();
    resize_buffer_if_needed(request.buffer, request.buffer_alloc_size, alloc_size);

    if (residual_buffer_size > 0) {
      memcpy(request.buffer.get(), residual_buffer.get(), residual_buffer_size);
    }
    size_t size = residual_buffer_size;
    size += file_reader.read(request.buffer.get() + residual_buffer_size,
                             alloc_size - residual_buffer_size);

    if (size == 0) {
      // In some cases at the end of a file we will read 0 bytes even when
      // file_reader.isScanFinished() is false. Also add request back to the pool to be
      // picked up again in the next iteration.
      add_request_to_pool(multi_threading_params, request);
      continue;
    } else if (size == 1 && request.buffer[0] == copy_params.line_delim) {
      // In some cases files with newlines at the end will be encoded with a second
      // newline that can end up being the only thing in the buffer. Also add request
      // back to the pool to be picked up again in the next iteration.
      current_file_offset++;
      add_request_to_pool(multi_threading_params, request);
      continue;
    }
    unsigned int num_rows_in_buffer = 0;
    request.end_pos = parser.findRowEndPosition(alloc_size,
                                                request.buffer,
                                                size,
                                                copy_params,
                                                first_row_index_in_buffer,
                                                num_rows_in_buffer,
                                                &file_reader);
    request.buffer_size = size;
    request.buffer_alloc_size = alloc_size;
    request.first_row_index = first_row_index_in_buffer;
    request.file_offset = current_file_offset;
    request.buffer_row_count = num_rows_in_buffer;
    request.processed_row_count = 0;
    request.begin_pos = 0;

    residual_buffer_size = size - request.end_pos;
    if (residual_buffer_size > 0) {
      resize_buffer_if_needed(residual_buffer, residual_buffer_alloc_size, alloc_size);
      memcpy(residual_buffer.get(),
             request.buffer.get() + request.end_pos,
             residual_buffer_size);
    }

    current_file_offset += request.end_pos;
    first_row_index_in_buffer += num_rows_in_buffer;

    if (num_rows_in_buffer > 0) {
      dispatch_scan_request(multi_threading_params, request);
    } else {
      add_request_to_pool(multi_threading_params, request);
    }

    if (file_scan_param) {
      const int32_t last_fragment_index =
          (first_row_index_in_buffer) / table->maxFragRows;
      if (last_fragment_index > file_scan_param->fragment_id) {
        break;
      }
    }
  }

  std::unique_lock<std::mutex> pending_requests_queue_lock(
      multi_threading_params.pending_requests_mutex);
  multi_threading_params.pending_requests_condition.wait(
      pending_requests_queue_lock, [&multi_threading_params] {
        return multi_threading_params.pending_requests.empty() ||
               (multi_threading_params.continue_processing == false);
      });
  multi_threading_params.continue_processing = false;
  pending_requests_queue_lock.unlock();
  multi_threading_params.pending_requests_condition.notify_all();
}

void dispatch_scan_requests_with_exception_handling(
    const foreign_storage::ForeignTable* table,
    const size_t& buffer_size,
    const std::string& file_path,
    FileReader& file_reader,
    const import_export::CopyParams& copy_params,
    MetadataScanMultiThreadingParams& multi_threading_params,
    size_t& first_row_index_in_buffer,
    size_t& current_file_offset,
    const TextFileBufferParser& parser,
    const foreign_storage::IterativeFileScanParameters* file_scan_param,
    foreign_storage::AbstractTextFileDataWrapper::ResidualBuffer&
        iterative_residual_buffer,
    const bool is_first_file_scan_call) {
  try {
    dispatch_scan_requests(table,
                           buffer_size,
                           file_path,
                           file_reader,
                           copy_params,
                           multi_threading_params,
                           first_row_index_in_buffer,
                           current_file_offset,
                           parser,
                           file_scan_param,
                           iterative_residual_buffer,
                           is_first_file_scan_call);
  } catch (...) {
    {
      std::unique_lock<std::mutex> pending_requests_lock(
          multi_threading_params.pending_requests_mutex);
      multi_threading_params.continue_processing = false;
    }
    multi_threading_params.pending_requests_condition.notify_all();
    throw;
  }
}

namespace {
// Create metadata for unscanned columns
// Any fragments with any updated rows between start_row and num_rows will be updated
// Chunks prior to start_row will be restored from  (ie for append
// workflows)
void add_placeholder_metadata(
    const ColumnDescriptor* column,
    const ForeignTable* foreign_table,
    const int db_id,
    const size_t start_row,
    const size_t total_num_rows,
    std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& chunk_metadata_map) {
  ChunkKey chunk_key = {db_id, foreign_table->tableId, column->columnId, 0};
  if (column->columnType.is_varlen_indeed()) {
    chunk_key.emplace_back(1);
  }

  // Create placeholder metadata for every fragment touched by this scan
  int start_fragment = start_row / foreign_table->maxFragRows;
  int end_fragment{0};
  if (total_num_rows > 0) {
    end_fragment = (total_num_rows - 1) / foreign_table->maxFragRows;
  }
  for (int fragment_id = start_fragment; fragment_id <= end_fragment; fragment_id++) {
    size_t num_elements = (static_cast<size_t>(foreign_table->maxFragRows *
                                               (fragment_id + 1)) > total_num_rows)
                              ? total_num_rows % foreign_table->maxFragRows
                              : foreign_table->maxFragRows;

    chunk_key[CHUNK_KEY_FRAGMENT_IDX] = fragment_id;
    chunk_metadata_map[chunk_key] =
        get_placeholder_metadata(column->columnType, num_elements);
  }
}

void initialize_non_append_mode_scan(
    const std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& chunk_metadata_map,
    const std::map<int, FileRegions>& fragment_id_to_file_regions_map,
    const foreign_storage::OptionsMap& server_options,
    std::unique_ptr<FileReader>& file_reader,
    const std::string& file_path,
    const import_export::CopyParams& copy_params,
    const shared::FilePathOptions& file_path_options,
    const std::optional<size_t>& max_file_count,
    const foreign_storage::ForeignTable* foreign_table,
    const foreign_storage::UserMapping* user_mapping,
    const foreign_storage::TextFileBufferParser& parser,
    std::function<std::string()> get_s3_key,
    size_t& num_rows,
    size_t& append_start_offset) {
  // Should only be called once for non-append tables
  CHECK(chunk_metadata_map.empty());
  CHECK(fragment_id_to_file_regions_map.empty());
  if (server_options.find(foreign_storage::AbstractTextFileDataWrapper::STORAGE_TYPE_KEY)
          ->second ==
      foreign_storage::AbstractTextFileDataWrapper::LOCAL_FILE_STORAGE_TYPE) {
    file_reader = std::make_unique<LocalMultiFileReader>(
        file_path, copy_params, file_path_options, max_file_count);
  } else {
    UNREACHABLE();
  }
  parser.validateFiles(file_reader.get(), foreign_table);
  num_rows = 0;
  append_start_offset = 0;
}
}  // namespace

/**
 * Populates provided chunk metadata vector with metadata for table specified in given
 * chunk key. Metadata scan for text file(s) configured for foreign table occurs in
 * parallel whenever appropriate. Parallel processing involves the main thread
 * creating ParseBufferRequest objects, which contain buffers with text content read
 * from file and adding these request objects to a queue that is consumed by a fixed
 * number of threads. After request processing, request objects are put back into a pool
 * for reuse for subsequent requests in order to avoid unnecessary allocation of new
 * buffers.
 *
 * @param chunk_metadata_vector - vector to be populated with chunk metadata
 */
void AbstractTextFileDataWrapper::populateChunkMetadata(
    ChunkMetadataVector& chunk_metadata_vector) {
  auto timer = DEBUG_TIMER(__func__);

  const auto copy_params = getFileBufferParser().validateAndGetCopyParams(foreign_table_);
  const auto file_path = getFullFilePath(foreign_table_);
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  auto& parser = getFileBufferParser();
  const auto file_path_options = getFilePathOptions(foreign_table_);
  auto& server_options = foreign_table_->foreign_server->options;
  std::set<std::string> rolled_off_files;
  if (foreign_table_->isAppendMode() && file_reader_ != nullptr) {
    auto multi_file_reader = dynamic_cast<MultiFileReader*>(file_reader_.get());
    if (allowFileRollOff(foreign_table_) && multi_file_reader) {
      rolled_off_files = multi_file_reader->checkForRolledOffFiles(file_path_options);
    }
    parser.validateFiles(file_reader_.get(), foreign_table_);
    if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
      file_reader_->checkForMoreRows(append_start_offset_, file_path_options);
    } else {
      UNREACHABLE();
    }
  } else {
    initialize_non_append_mode_scan(
        chunk_metadata_map_,
        fragment_id_to_file_regions_map_,
        server_options,
        file_reader_,
        file_path,
        copy_params,
        file_path_options,
        getMaxFileCount(),
        foreign_table_,
        user_mapping_,
        parser,
        [this] { return ""; },
        num_rows_,
        append_start_offset_);
  }

  auto columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);
  std::map<int32_t, const ColumnDescriptor*> column_by_id{};
  for (auto column : columns) {
    column_by_id[column->columnId] = column;
  }
  MetadataScanMultiThreadingParams multi_threading_params;
  multi_threading_params.disable_cache = disable_cache_;

  // Restore previous chunk data
  if (foreign_table_->isAppendMode()) {
    multi_threading_params.chunk_encoder_buffers = std::move(chunk_encoder_buffers_);
  }

  std::set<int> columns_to_scan;
  for (auto column : columns) {
    if (!skip_metadata_scan(column)) {
      columns_to_scan.insert(column->columnId);
    }
  }

  // Track where scan started for appends
  int start_row = num_rows_;
  if (!file_reader_->isScanFinished()) {
    auto buffer_size = get_buffer_size(copy_params,
                                       file_reader_->isRemainingSizeKnown(),
                                       file_reader_->getRemainingSize());
    auto thread_count = get_thread_count(copy_params,
                                         file_reader_->isRemainingSizeKnown(),
                                         file_reader_->getRemainingSize(),
                                         buffer_size);
    multi_threading_params.continue_processing = true;

    std::vector<std::future<void>> futures{};
    for (size_t i = 0; i < thread_count; i++) {
      multi_threading_params.request_pool.emplace(buffer_size,
                                                  copy_params,
                                                  db_id_,
                                                  foreign_table_,
                                                  columns_to_scan,
                                                  getFullFilePath(foreign_table_),
                                                  nullptr,
                                                  disable_cache_);

      futures.emplace_back(std::async(std::launch::async,
                                      scan_metadata,
                                      std::ref(multi_threading_params),
                                      std::ref(fragment_id_to_file_regions_map_),
                                      std::ref(parser)));
    }

    ResidualBuffer residual_buffer;
    dispatch_scan_requests_with_exception_handling(foreign_table_,
                                                   buffer_size,
                                                   file_path,
                                                   (*file_reader_),
                                                   copy_params,
                                                   multi_threading_params,
                                                   num_rows_,
                                                   append_start_offset_,
                                                   getFileBufferParser(),
                                                   nullptr,
                                                   residual_buffer,
                                                   true);

    for (auto& future : futures) {
      // get() instead of wait() because we need to propagate potential exceptions.
      future.get();
    }
  }

  for (auto& [chunk_key, buffer] : multi_threading_params.chunk_encoder_buffers) {
    auto column_entry = column_by_id.find(chunk_key[CHUNK_KEY_COLUMN_IDX]);
    CHECK(column_entry != column_by_id.end());
    const auto& column_type = column_entry->second->columnType;
    auto chunk_metadata = buffer->getEncoder()->getMetadata(column_type);
    chunk_metadata->numElements = buffer->getEncoder()->getNumElems();
    const auto& cached_chunks = multi_threading_params.cached_chunks;
    if (!column_type.is_varlen_indeed()) {
      chunk_metadata->numBytes = column_type.get_size() * chunk_metadata->numElements;
    } else if (auto chunk_entry = cached_chunks.find(chunk_key);
               chunk_entry != cached_chunks.end()) {
      auto buffer = chunk_entry->second.getBuffer();
      CHECK(buffer);
      chunk_metadata->numBytes = buffer->size();
    } else {
      CHECK_EQ(chunk_metadata->numBytes, static_cast<size_t>(0));
    }
    chunk_metadata_map_[chunk_key] = chunk_metadata;
  }

  for (auto column : columns) {
    if (skip_metadata_scan(column)) {
      add_placeholder_metadata(
          column, foreign_table_, db_id_, start_row, num_rows_, chunk_metadata_map_);
    }
  }

  if (!rolled_off_files.empty()) {
    updateRolledOffChunks(rolled_off_files, column_by_id);
  }

  for (auto& [chunk_key, chunk_metadata] : chunk_metadata_map_) {
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
  }

  // Save chunk data
  if (foreign_table_->isAppendMode()) {
    chunk_encoder_buffers_ = std::move(multi_threading_params.chunk_encoder_buffers);
  }
}

void AbstractTextFileDataWrapper::iterativeFileScan(
    ChunkMetadataVector& chunk_metadata_vector,
    IterativeFileScanParameters& file_scan_param) {
  auto timer = DEBUG_TIMER(__func__);

  is_file_scan_in_progress_ = true;

  CHECK(!foreign_table_->isAppendMode())
      << " iterative file scan can not be used with APPEND mode.";

  const auto copy_params = getFileBufferParser().validateAndGetCopyParams(foreign_table_);
  const auto file_path = getFullFilePath(foreign_table_);
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  auto& parser = getFileBufferParser();
  const auto file_path_options = getFilePathOptions(foreign_table_);
  auto& server_options = foreign_table_->foreign_server->options;

  if (is_first_file_scan_call_) {
    initialize_non_append_mode_scan(
        chunk_metadata_map_,
        fragment_id_to_file_regions_map_,
        server_options,
        file_reader_,
        file_path,
        copy_params,
        file_path_options,
        getMaxFileCount(),
        foreign_table_,
        user_mapping_,
        parser,
        [this] { return ""; },
        num_rows_,
        append_start_offset_);
  }

  auto columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);
  std::map<int32_t, const ColumnDescriptor*> column_by_id{};
  for (auto column : columns) {
    column_by_id[column->columnId] = column;
  }

  if (is_first_file_scan_call_) {  // reiniitialize all members that may have state in
                                   // `multi_threading_params_`
    reset_multithreading_params(multi_threading_params_);
  }

  multi_threading_params_.disable_cache = disable_cache_;

  std::set<int> columns_to_scan;
  for (auto column : columns) {
    columns_to_scan.insert(column->columnId);
  }

  if (!is_file_scan_finished(file_reader_.get(), multi_threading_params_)) {
    if (is_first_file_scan_call_) {
      // NOTE: `buffer_size_` and `thread_count_` must not change across an iterative
      // scan
      buffer_size_ = get_buffer_size(copy_params,
                                     file_reader_->isRemainingSizeKnown(),
                                     file_reader_->getRemainingSize());
      thread_count_ = get_thread_count(copy_params,
                                       file_reader_->isRemainingSizeKnown(),
                                       file_reader_->getRemainingSize(),
                                       buffer_size_);
    }
    multi_threading_params_.continue_processing = true;

    std::vector<std::future<void>> futures{};
    for (size_t i = 0; i < thread_count_; i++) {
      if (is_first_file_scan_call_) {
        multi_threading_params_.request_pool.emplace(buffer_size_,
                                                     copy_params,
                                                     db_id_,
                                                     foreign_table_,
                                                     columns_to_scan,
                                                     getFullFilePath(foreign_table_),
                                                     &render_group_analyzer_map_,
                                                     true);
      }
      futures.emplace_back(std::async(std::launch::async,
                                      populate_chunks,
                                      std::ref(multi_threading_params_),
                                      std::ref(fragment_id_to_file_regions_map_),
                                      std::ref(parser),
                                      std::ref(file_scan_param)));
    }

    dispatch_scan_requests_with_exception_handling(foreign_table_,
                                                   buffer_size_,
                                                   file_path,
                                                   (*file_reader_),
                                                   copy_params,
                                                   multi_threading_params_,
                                                   num_rows_,
                                                   append_start_offset_,
                                                   getFileBufferParser(),
                                                   &file_scan_param,
                                                   residual_buffer_,
                                                   is_first_file_scan_call_);

    for (auto& future : futures) {
      // get() instead of wait() because we need to propagate potential exceptions.
      future.get();
    }
  }

  if (is_first_file_scan_call_) {
    is_first_file_scan_call_ = false;
  }

  if (!is_file_scan_in_progress_) {
    reset_multithreading_params(multi_threading_params_);
  }
}

void AbstractTextFileDataWrapper::updateRolledOffChunks(
    const std::set<std::string>& rolled_off_files,
    const std::map<int32_t, const ColumnDescriptor*>& column_by_id) {
  std::set<int32_t> deleted_fragment_ids;
  std::optional<int32_t> partially_deleted_fragment_id;
  std::optional<size_t> partially_deleted_fragment_row_count;
  for (auto& [fragment_id, file_regions] : fragment_id_to_file_regions_map_) {
    bool file_region_deleted{false};
    for (auto it = file_regions.begin(); it != file_regions.end();) {
      if (shared::contains(rolled_off_files, it->file_path)) {
        it = file_regions.erase(it);
        file_region_deleted = true;
      } else {
        it++;
      }
    }
    if (file_regions.empty()) {
      deleted_fragment_ids.emplace(fragment_id);
    } else if (file_region_deleted) {
      partially_deleted_fragment_id = fragment_id;
      partially_deleted_fragment_row_count = 0;
      for (const auto& file_region : file_regions) {
        partially_deleted_fragment_row_count.value() += file_region.row_count;
      }
      break;
    }
  }

  for (auto& [chunk_key, chunk_metadata] : chunk_metadata_map_) {
    if (shared::contains(deleted_fragment_ids, chunk_key[CHUNK_KEY_FRAGMENT_IDX])) {
      chunk_metadata->numElements = 0;
      chunk_metadata->numBytes = 0;
    } else if (chunk_key[CHUNK_KEY_FRAGMENT_IDX] == partially_deleted_fragment_id) {
      CHECK(partially_deleted_fragment_row_count.has_value());
      auto old_chunk_stats = chunk_metadata->chunkStats;
      auto cd = shared::get_from_map(column_by_id, chunk_key[CHUNK_KEY_COLUMN_IDX]);
      chunk_metadata = get_placeholder_metadata(
          cd->columnType, partially_deleted_fragment_row_count.value());
      // Old chunk stats will still be correct (since only row deletion is occurring)
      // and more accurate than that of the placeholder metadata.
      chunk_metadata->chunkStats = old_chunk_stats;
    }
  }
}

std::string AbstractTextFileDataWrapper::getSerializedDataWrapper() const {
  rapidjson::Document d;
  d.SetObject();

  // Save fragment map
  json_utils::add_value_to_object(d,
                                  fragment_id_to_file_regions_map_,
                                  "fragment_id_to_file_regions_map",
                                  d.GetAllocator());

  // Save reader metadata
  rapidjson::Value reader_metadata(rapidjson::kObjectType);
  file_reader_->serialize(reader_metadata, d.GetAllocator());
  d.AddMember("reader_metadata", reader_metadata, d.GetAllocator());

  json_utils::add_value_to_object(d, num_rows_, "num_rows", d.GetAllocator());
  json_utils::add_value_to_object(
      d, append_start_offset_, "append_start_offset", d.GetAllocator());

  return json_utils::write_to_string(d);
}

void AbstractTextFileDataWrapper::restoreDataWrapperInternals(
    const std::string& file_path,
    const ChunkMetadataVector& chunk_metadata) {
  auto d = json_utils::read_from_file(file_path);
  CHECK(d.IsObject());

  // Restore fragment map
  json_utils::get_value_from_object(
      d, fragment_id_to_file_regions_map_, "fragment_id_to_file_regions_map");

  // Construct reader with metadta
  CHECK(d.HasMember("reader_metadata"));
  const auto copy_params = getFileBufferParser().validateAndGetCopyParams(foreign_table_);
  const auto full_file_path = getFullFilePath(foreign_table_);
  auto& server_options = foreign_table_->foreign_server->options;
  if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
    file_reader_ = std::make_unique<LocalMultiFileReader>(
        full_file_path, copy_params, d["reader_metadata"]);
  } else {
    UNREACHABLE();
  }

  json_utils::get_value_from_object(d, num_rows_, "num_rows");
  json_utils::get_value_from_object(d, append_start_offset_, "append_start_offset");

  // Now restore the internal metadata maps
  CHECK(chunk_metadata_map_.empty());
  CHECK(chunk_encoder_buffers_.empty());

  for (auto& pair : chunk_metadata) {
    chunk_metadata_map_[pair.first] = pair.second;

    if (foreign_table_->isAppendMode()) {
      // Restore encoder state for append mode
      chunk_encoder_buffers_[pair.first] = std::make_unique<ForeignStorageBuffer>();
      chunk_encoder_buffers_[pair.first]->initEncoder(pair.second->sqlType);
      chunk_encoder_buffers_[pair.first]->setSize(pair.second->numBytes);
      chunk_encoder_buffers_[pair.first]->getEncoder()->setNumElems(
          pair.second->numElements);
      chunk_encoder_buffers_[pair.first]->getEncoder()->resetChunkStats(
          pair.second->chunkStats);
      chunk_encoder_buffers_[pair.first]->setUpdated();
    }
  }
  is_restored_ = true;
}

bool AbstractTextFileDataWrapper::isRestored() const {
  return is_restored_;
}

// declared in three derived classes to avoid
// polluting ForeignDataWrapper virtual base
// @TODO refactor to lower class if needed
void AbstractTextFileDataWrapper::createRenderGroupAnalyzers() {
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

std::optional<size_t> AbstractTextFileDataWrapper::getMaxFileCount() const {
  return {};
}

}  // namespace foreign_storage
