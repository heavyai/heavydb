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

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>

#include <rapidjson/document.h>
#include <boost/filesystem.hpp>

#include "CsvShared.h"
#include "DataMgr/ForeignStorage/CsvFileBufferParser.h"
#include "DataMgr/ForeignStorage/CsvReader.h"
#include "DataMgr/ForeignStorage/ForeignTableSchema.h"
#include "FsiJsonUtils.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "ImportExport/Importer.h"
#include "Shared/sqltypes.h"
#include "Utils/DdlUtils.h"

namespace foreign_storage {
CsvDataWrapper::CsvDataWrapper() : db_id_(-1), foreign_table_(nullptr) {}

CsvDataWrapper::CsvDataWrapper(const int db_id, const ForeignTable* foreign_table)
    : db_id_(db_id), foreign_table_(foreign_table), is_restored_(false) {}

void CsvDataWrapper::validateTableOptions(const ForeignTable* foreign_table) const {
  AbstractFileStorageDataWrapper::validateTableOptions(foreign_table);
  Csv::validate_options(foreign_table);
}
const std::set<std::string_view>& CsvDataWrapper::getSupportedTableOptions() const {
  static const auto supported_table_options = getAllCsvTableOptions();
  return supported_table_options;
}

std::set<std::string_view> CsvDataWrapper::getAllCsvTableOptions() const {
  std::set<std::string_view> supported_table_options(
      AbstractFileStorageDataWrapper::getSupportedTableOptions().begin(),
      AbstractFileStorageDataWrapper::getSupportedTableOptions().end());
  supported_table_options.insert(csv_table_options_.begin(), csv_table_options_.end());
  return supported_table_options;
}

namespace {
std::set<const ColumnDescriptor*> get_columns(
    const ChunkToBufferMap& buffers,
    std::shared_ptr<Catalog_Namespace::Catalog> catalog,
    const int32_t table_id,
    const int fragment_id) {
  CHECK(!buffers.empty());
  std::set<const ColumnDescriptor*> columns;
  for (const auto& entry : buffers) {
    CHECK_EQ(fragment_id, entry.first[CHUNK_KEY_FRAGMENT_IDX]);
    const auto column_id = entry.first[CHUNK_KEY_COLUMN_IDX];
    const auto column = catalog->getMetadataForColumnUnlocked(table_id, column_id);
    columns.emplace(column);
  }
  return columns;
}
}  // namespace

namespace {
bool skip_metadata_scan(const ColumnDescriptor* column) {
  return column->columnType.is_dict_encoded_type();
}
}  // namespace

void CsvDataWrapper::populateChunkMapForColumns(
    const std::set<const ColumnDescriptor*>& columns,
    const int fragment_id,
    const ChunkToBufferMap& buffers,
    std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map) {
  for (const auto column : columns) {
    ChunkKey data_chunk_key = {
        db_id_, foreign_table_->tableId, column->columnId, fragment_id};
    column_id_to_chunk_map[column->columnId] =
        Csv::make_chunk_for_column(data_chunk_key, chunk_metadata_map_, buffers);
  }
}

void CsvDataWrapper::populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                                          const ChunkToBufferMap& optional_buffers) {
  auto timer = DEBUG_TIMER(__func__);
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  CHECK(!required_buffers.empty());

  auto fragment_id = required_buffers.begin()->first[CHUNK_KEY_FRAGMENT_IDX];
  std::set<const ColumnDescriptor*> required_columns =
      get_columns(required_buffers, catalog, foreign_table_->tableId, fragment_id);
  std::map<int, Chunk_NS::Chunk> column_id_to_chunk_map;
  populateChunkMapForColumns(
      required_columns, fragment_id, required_buffers, column_id_to_chunk_map);

  if (!optional_buffers.empty()) {
    std::set<const ColumnDescriptor*> optional_columns;
    optional_columns =
        get_columns(optional_buffers, catalog, foreign_table_->tableId, fragment_id);
    populateChunkMapForColumns(
        optional_columns, fragment_id, optional_buffers, column_id_to_chunk_map);
  }
  populateChunks(column_id_to_chunk_map, fragment_id);
  updateMetadata(column_id_to_chunk_map, fragment_id);
  for (auto& entry : column_id_to_chunk_map) {
    entry.second.setBuffer(nullptr);
    entry.second.setIndexBuffer(nullptr);
  }
}

// if column was skipped during scan, update metadata now
void CsvDataWrapper::updateMetadata(
    std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
    int fragment_id) {
  auto fragmenter = foreign_table_->fragmenter;
  if (fragmenter) {
    auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
    CHECK(catalog);
    for (auto& entry : column_id_to_chunk_map) {
      const auto& column =
          catalog->getMetadataForColumnUnlocked(foreign_table_->tableId, entry.first);
      if (skip_metadata_scan(column)) {
        ChunkKey data_chunk_key = {
            db_id_, foreign_table_->tableId, column->columnId, fragment_id};
        if (column->columnType.is_varlen_indeed()) {
          data_chunk_key.emplace_back(1);
        }
        CHECK(chunk_metadata_map_.find(data_chunk_key) != chunk_metadata_map_.end());
        auto cached_metadata = chunk_metadata_map_[data_chunk_key];
        auto chunk_metadata =
            entry.second.getBuffer()->getEncoder()->getMetadata(column->columnType);
        cached_metadata->chunkStats.max = chunk_metadata->chunkStats.max;
        cached_metadata->chunkStats.min = chunk_metadata->chunkStats.min;
        cached_metadata->chunkStats.has_nulls = chunk_metadata->chunkStats.has_nulls;
        cached_metadata->numBytes = entry.second.getBuffer()->size();
        fragmenter->updateColumnChunkMetadata(column, fragment_id, cached_metadata);
      }
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

  bool operator<(const ParseFileRegionResult& other) const {
    return file_offset < other.file_offset;
  }
};

/**
 * Parses a set of file regions given a handle to the file and range of indexes
 * for the file regions to be parsed.
 */
ParseFileRegionResult parse_file_regions(
    const FileRegions& file_regions,
    const size_t start_index,
    const size_t end_index,
    CsvReader& csv_reader,
    csv_file_buffer_parser::ParseBufferRequest& parse_file_request,
    const std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map) {
  ParseFileRegionResult load_file_region_result{};
  load_file_region_result.file_offset = file_regions[start_index].first_row_file_offset;
  load_file_region_result.row_count = 0;

  csv_file_buffer_parser::ParseBufferResult result;
  for (size_t i = start_index; i <= end_index; i++) {
    CHECK(file_regions[i].region_size <= parse_file_request.buffer_size);
    size_t read_size;
    {
      read_size = csv_reader.readRegion(parse_file_request.buffer.get(),
                                        file_regions[i].first_row_file_offset,
                                        file_regions[i].region_size);
    }

    CHECK_EQ(file_regions[i].region_size, read_size);
    parse_file_request.begin_pos = 0;
    parse_file_request.end_pos = file_regions[i].region_size;
    parse_file_request.first_row_index = file_regions[i].first_row_index;
    parse_file_request.file_offset = file_regions[i].first_row_file_offset;
    parse_file_request.process_row_count = file_regions[i].row_count;

    result = parse_buffer(parse_file_request, i == end_index);
    CHECK_EQ(file_regions[i].row_count, result.row_count);
    load_file_region_result.row_count += result.row_count;
  }
  load_file_region_result.column_id_to_data_blocks_map =
      result.column_id_to_data_blocks_map;
  return load_file_region_result;
}

/**
 * Gets the appropriate buffer size to be used when processing CSV file(s).
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
  if (size_known) {
    size_t num_buffers_in_file = (file_size + buffer_size - 1) / buffer_size;
    if (num_buffers_in_file < thread_count) {
      thread_count = num_buffers_in_file;
    }
  }
  CHECK(thread_count);
  return thread_count;
}

size_t get_thread_count(const import_export::CopyParams& copy_params,
                        const FileRegions& file_regions) {
  size_t thread_count = copy_params.threads;
  if (thread_count == 0) {
    thread_count =
        std::min<size_t>(std::thread::hardware_concurrency(), file_regions.size());
  }
  CHECK(thread_count);
  return thread_count;
}

void CsvDataWrapper::populateChunks(
    std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
    int fragment_id) {
  const auto copy_params = Csv::validate_and_get_copy_params(foreign_table_);

  CHECK(!column_id_to_chunk_map.empty());
  const auto& file_regions = fragment_id_to_file_regions_map_[fragment_id];
  CHECK(!file_regions.empty());

  const auto buffer_size = get_buffer_size(file_regions);
  const auto thread_count = get_thread_count(copy_params, file_regions);

  const int batch_size = (file_regions.size() + thread_count - 1) / thread_count;

  std::vector<csv_file_buffer_parser::ParseBufferRequest> parse_file_requests{};
  parse_file_requests.reserve(thread_count);
  std::vector<std::future<ParseFileRegionResult>> futures{};
  std::set<int> column_filter_set;
  for (const auto& pair : column_id_to_chunk_map) {
    column_filter_set.insert(pair.first);
  }

  std::vector<std::unique_ptr<CsvReader>> csv_readers;
  rapidjson::Value reader_metadata(rapidjson::kObjectType);
  rapidjson::Document d;
  auto& server_options = foreign_table_->foreign_server->options;
  csv_reader_->serialize(reader_metadata, d.GetAllocator());
  const auto csv_file_path = getFullFilePath(foreign_table_);

  for (size_t i = 0; i < file_regions.size(); i += batch_size) {
    parse_file_requests.emplace_back(buffer_size,
                                     copy_params,
                                     db_id_,
                                     foreign_table_,
                                     column_filter_set,
                                     csv_file_path);
    auto start_index = i;
    auto end_index =
        std::min<size_t>(start_index + batch_size - 1, file_regions.size() - 1);

    if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
      csv_readers.emplace_back(std::make_unique<LocalMultiFileReader>(
          csv_file_path, copy_params, reader_metadata));
    } else {
      UNREACHABLE();
    }

    futures.emplace_back(std::async(std::launch::async,
                                    parse_file_regions,
                                    std::ref(file_regions),
                                    start_index,
                                    end_index,
                                    std::ref(*(csv_readers.back())),
                                    std::ref(parse_file_requests.back()),
                                    std::ref(column_id_to_chunk_map)));
  }

  std::vector<ParseFileRegionResult> load_file_region_results{};
  for (auto& future : futures) {
    future.wait();
    load_file_region_results.emplace_back(future.get());
  }

  for (auto result : load_file_region_results) {
    for (auto& [column_id, chunk] : column_id_to_chunk_map) {
      chunk.appendData(
          result.column_id_to_data_blocks_map[column_id], result.row_count, 0);
    }
  }
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
 * Data structure used to hold shared objects needed for inter-thread
 * synchronization or objects containing data that is updated by
 * multiple threads while scanning CSV files for metadata.
 */
struct MetadataScanMultiThreadingParams {
  std::queue<csv_file_buffer_parser::ParseBufferRequest> pending_requests;
  std::mutex pending_requests_mutex;
  std::condition_variable pending_requests_condition;
  std::queue<csv_file_buffer_parser::ParseBufferRequest> request_pool;
  std::mutex request_pool_mutex;
  std::condition_variable request_pool_condition;
  bool continue_processing;
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_encoder_buffers;
  std::map<ChunkKey, Chunk_NS::Chunk> cached_chunks;
  std::mutex chunk_encoder_buffers_mutex;
  std::map<ChunkKey, size_t> chunk_byte_count;
  std::mutex chunk_byte_count_mutex;
};

/**
 * Gets the next metadata scan request object from the pending requests queue.
 * A null optional is returned if there are no further requests to be processed.
 */
std::optional<csv_file_buffer_parser::ParseBufferRequest> get_next_metadata_scan_request(
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
 * Creates a new file region based on metadata from parsed CSV file buffers and
 * adds the new region to the fragment id to file regions map.
 */
void add_file_region(std::map<int, FileRegions>& fragment_id_to_file_regions_map,
                     int fragment_id,
                     size_t first_row_index,
                     const csv_file_buffer_parser::ParseBufferResult& result,
                     const std::string& file_path) {
  fragment_id_to_file_regions_map[fragment_id].emplace_back(
      // file naming is handled by CsvReader
      FileRegion(result.row_offsets.front(),
                 first_row_index,
                 result.row_count,
                 result.row_offsets.back() - result.row_offsets.front()));
}

/**
 * Get the total number of bytes in the given data block for
 * a variable length column.
 */
size_t get_var_length_data_block_size(DataBlockPtr data_block,
                                      SQLTypeInfo sql_type_info) {
  CHECK(sql_type_info.is_varlen());
  size_t byte_count = 0;
  if (sql_type_info.is_string() || sql_type_info.is_geometry()) {
    for (const auto& str : *data_block.stringsPtr) {
      byte_count += str.length();
    }
  } else if (sql_type_info.is_array()) {
    for (const auto& array : *data_block.arraysPtr) {
      byte_count += array.length;
    }
  } else {
    UNREACHABLE();
  }
  return byte_count;
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
    std::shared_ptr<Catalog_Namespace::Catalog>& catalog) {
  if (catalog->getDataMgr()
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
                  bool is_last_block) {
  auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(chunk_key[CHUNK_KEY_DB_IDX]);
  CHECK(catalog);
  auto cache = get_cache_if_enabled(catalog);
  if (cache) {
    ChunkKey index_key = {chunk_key[CHUNK_KEY_DB_IDX],
                          chunk_key[CHUNK_KEY_TABLE_IDX],
                          chunk_key[CHUNK_KEY_COLUMN_IDX],
                          chunk_key[CHUNK_KEY_FRAGMENT_IDX],
                          2};
    // Create actual data chunks to prepopulate cache
    if (cached_chunks.find(chunk_key) == cached_chunks.end()) {
      cached_chunks[chunk_key] = Chunk_NS::Chunk{column};
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
    if (is_last_block) {
      // cache the chunks now so they are tracked by eviction algorithm
      std::vector<ChunkKey> key_to_cache{chunk_key};
      if (column->columnType.is_varlen_indeed()) {
        key_to_cache.push_back(index_key);
      }
      cache->cacheTableChunks(key_to_cache);
    }
  }
}

/**
 * Updates metadata encapsulated in encoders for all table columns given
 * new data blocks gotten from parsing a new set of rows in a CSV file buffer.
 * If cache is available, also append the data_blocks to chunks in the cache
 */
void process_data_blocks(MetadataScanMultiThreadingParams& multi_threading_params,
                         int fragment_id,
                         const csv_file_buffer_parser::ParseBufferRequest& request,
                         csv_file_buffer_parser::ParseBufferResult& result,
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
    size_t byte_count;
    if (column->columnType.is_varlen_indeed()) {
      chunk_key.emplace_back(1);
      byte_count = get_var_length_data_block_size(data_block, column->columnType);
    } else {
      byte_count = column->columnType.get_size() * result.row_count;
    }

    {
      std::lock_guard<std::mutex> lock(multi_threading_params.chunk_byte_count_mutex);
      multi_threading_params.chunk_byte_count[chunk_key] += byte_count;
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
        num_elements == request.getMaxFragRows()  // Is the last block for this chunk
    );
  }
}

/**
 * Adds the request object for a processed request back to the request pool
 * for reuse in subsequent requests.
 */
void add_request_to_pool(MetadataScanMultiThreadingParams& multi_threading_params,
                         csv_file_buffer_parser::ParseBufferRequest& request) {
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
                   std::map<int, FileRegions>& fragment_id_to_file_regions_map) {
  std::map<int, const ColumnDescriptor*> column_by_id{};
  while (true) {
    auto request_opt = get_next_metadata_scan_request(multi_threading_params);
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
        auto result = parse_buffer(request, true);
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
csv_file_buffer_parser::ParseBufferRequest get_request_from_pool(
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

/**
 * Dispatches a new metadata scan request by adding the request to
 * the pending requests queue to be consumed by a worker thread.
 */
void dispatch_metadata_scan_request(
    MetadataScanMultiThreadingParams& multi_threading_params,
    csv_file_buffer_parser::ParseBufferRequest& request) {
  {
    std::unique_lock<std::mutex> pending_requests_lock(
        multi_threading_params.pending_requests_mutex);
    multi_threading_params.pending_requests.emplace(std::move(request));
  }
  multi_threading_params.pending_requests_condition.notify_all();
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

/**
 * Reads from a CSV file iteratively and dispatches metadata scan
 * requests that are processed by worker threads.
 */
void dispatch_metadata_scan_requests(
    const size_t& buffer_size,
    const std::string& file_path,
    CsvReader& csv_reader,
    const import_export::CopyParams& copy_params,
    MetadataScanMultiThreadingParams& multi_threading_params,
    size_t& first_row_index_in_buffer,
    size_t& current_file_offset) {
  auto alloc_size = buffer_size;
  auto residual_buffer = std::make_unique<char[]>(alloc_size);
  size_t residual_buffer_size = 0;
  size_t residual_buffer_alloc_size = alloc_size;

  while (!csv_reader.isScanFinished()) {
    {
      std::lock_guard<std::mutex> pending_requests_lock(
          multi_threading_params.pending_requests_mutex);
      if (!multi_threading_params.continue_processing) {
        break;
      }
    }
    auto request = get_request_from_pool(multi_threading_params);
    resize_buffer_if_needed(request.buffer, request.buffer_alloc_size, alloc_size);

    if (residual_buffer_size > 0) {
      memcpy(request.buffer.get(), residual_buffer.get(), residual_buffer_size);
    }
    size_t size = residual_buffer_size;
    size += csv_reader.read(request.buffer.get() + residual_buffer_size,
                            alloc_size - residual_buffer_size);

    if (size == 0) {
      // In some cases at the end of a file we will read 0 bytes even when
      // csv_reader.isScanFinished() is false
      continue;
    } else if (size == 1 && request.buffer[0] == copy_params.line_delim) {
      // In some cases files with newlines at the end will be encoded with a second
      // newline that can end up being the only thing in the buffer
      current_file_offset++;
      continue;
    }
    unsigned int num_rows_in_buffer = 0;
    request.end_pos =
        import_export::delimited_parser::find_row_end_pos(alloc_size,
                                                          request.buffer,
                                                          size,
                                                          copy_params,
                                                          first_row_index_in_buffer,
                                                          num_rows_in_buffer,
                                                          nullptr,
                                                          &csv_reader);
    request.buffer_size = size;
    request.buffer_alloc_size = alloc_size;
    request.first_row_index = first_row_index_in_buffer;
    request.file_offset = current_file_offset;
    request.buffer_row_count = num_rows_in_buffer;

    residual_buffer_size = size - request.end_pos;
    if (residual_buffer_size > 0) {
      resize_buffer_if_needed(residual_buffer, residual_buffer_alloc_size, alloc_size);
      memcpy(residual_buffer.get(),
             request.buffer.get() + request.end_pos,
             residual_buffer_size);
    }

    current_file_offset += request.end_pos;
    first_row_index_in_buffer += num_rows_in_buffer;

    dispatch_metadata_scan_request(multi_threading_params, request);
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
  int end_fragment = total_num_rows / foreign_table->maxFragRows;
  for (int fragment_id = start_fragment; fragment_id <= end_fragment; fragment_id++) {
    size_t num_elements = (static_cast<size_t>(foreign_table->maxFragRows *
                                               (fragment_id + 1)) > total_num_rows)
                              ? total_num_rows % foreign_table->maxFragRows
                              : foreign_table->maxFragRows;

    chunk_key[CHUNK_KEY_FRAGMENT_IDX] = fragment_id;
    chunk_metadata_map[chunk_key] = Csv::get_placeholder_metadata(column, num_elements);
  }
}

}  // namespace

/**
 * Populates provided chunk metadata vector with metadata for table specified in given
 * chunk key. Metadata scan for CSV file(s) configured for foreign table occurs in
 * parallel whenever appropriate. Parallel processing involves the main thread
 * creating ParseBufferRequest objects, which contain buffers with CSV content read
 * from file and adding these request objects to a queue that is consumed by a fixed
 * number of threads. After request processing, request objects are put back into a pool
 * for reuse for subsequent requests in order to avoid unnecessary allocation of new
 * buffers.
 *
 * @param chunk_metadata_vector - vector to be populated with chunk metadata
 */
void CsvDataWrapper::populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) {
  auto timer = DEBUG_TIMER(__func__);

  const auto copy_params = Csv::validate_and_get_copy_params(foreign_table_);
  const auto file_path = getFullFilePath(foreign_table_);
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  auto& server_options = foreign_table_->foreign_server->options;
  if (foreign_table_->isAppendMode() && csv_reader_ != nullptr) {
    if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
      csv_reader_->checkForMoreRows(append_start_offset_);
    } else {
      UNREACHABLE();
    }
  } else {
    chunk_metadata_map_.clear();
    fragment_id_to_file_regions_map_.clear();
    if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
      csv_reader_ = std::make_unique<LocalMultiFileReader>(file_path, copy_params);
    } else {
      UNREACHABLE();
    }
    num_rows_ = 0;
    append_start_offset_ = 0;
  }

  auto columns = catalog->getAllColumnMetadataForTableUnlocked(
      foreign_table_->tableId, false, false, true);
  std::map<int32_t, const ColumnDescriptor*> column_by_id{};
  for (auto column : columns) {
    column_by_id[column->columnId] = column;
  }
  MetadataScanMultiThreadingParams multi_threading_params;

  // Restore previous chunk data
  if (foreign_table_->isAppendMode()) {
    multi_threading_params.chunk_byte_count = chunk_byte_count_;
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
  if (!csv_reader_->isScanFinished()) {
    auto buffer_size = get_buffer_size(copy_params,
                                       csv_reader_->isRemainingSizeKnown(),
                                       csv_reader_->getRemainingSize());
    auto thread_count = get_thread_count(copy_params,
                                         csv_reader_->isRemainingSizeKnown(),
                                         csv_reader_->getRemainingSize(),
                                         buffer_size);
    multi_threading_params.continue_processing = true;

    std::vector<std::future<void>> futures{};
    for (size_t i = 0; i < thread_count; i++) {
      multi_threading_params.request_pool.emplace(buffer_size,
                                                  copy_params,
                                                  db_id_,
                                                  foreign_table_,
                                                  columns_to_scan,
                                                  getFullFilePath(foreign_table_));

      futures.emplace_back(std::async(std::launch::async,
                                      scan_metadata,
                                      std::ref(multi_threading_params),
                                      std::ref(fragment_id_to_file_regions_map_)));
    }

    try {
      dispatch_metadata_scan_requests(buffer_size,
                                      file_path,
                                      (*csv_reader_),
                                      copy_params,
                                      multi_threading_params,
                                      num_rows_,
                                      append_start_offset_);
    } catch (...) {
      {
        std::unique_lock<std::mutex> pending_requests_lock(
            multi_threading_params.pending_requests_mutex);
        multi_threading_params.continue_processing = false;
      }
      multi_threading_params.pending_requests_condition.notify_all();
      throw;
    }

    for (auto& future : futures) {
      // get() instead of wait() because we need to propagate potential exceptions.
      future.get();
    }
  }

  for (auto& [chunk_key, buffer] : multi_threading_params.chunk_encoder_buffers) {
    auto chunk_metadata =
        buffer->getEncoder()->getMetadata(column_by_id[chunk_key[2]]->columnType);
    chunk_metadata->numElements = buffer->getEncoder()->getNumElems();
    chunk_metadata->numBytes = multi_threading_params.chunk_byte_count[chunk_key];
    chunk_metadata_map_[chunk_key] = chunk_metadata;
  }

  for (auto column : columns) {
    if (skip_metadata_scan(column)) {
      add_placeholder_metadata(
          column, foreign_table_, db_id_, start_row, num_rows_, chunk_metadata_map_);
    }
  }

  for (auto& [chunk_key, chunk_metadata] : chunk_metadata_map_) {
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
  }

  // Save chunk data
  if (foreign_table_->isAppendMode()) {
    chunk_byte_count_ = multi_threading_params.chunk_byte_count;
    chunk_encoder_buffers_ = std::move(multi_threading_params.chunk_encoder_buffers);
  }

  // Any incomplete chunks should be cached now
  auto cache = get_cache_if_enabled(catalog);
  if (cache) {
    std::vector<ChunkKey> to_cache;
    for (auto& [chunk_key, buffer] : multi_threading_params.cached_chunks) {
      if (buffer.getBuffer()->getEncoder()->getNumElems() !=
          static_cast<size_t>(foreign_table_->maxFragRows)) {
        if (column_by_id[chunk_key[CHUNK_KEY_COLUMN_IDX]]
                ->columnType.is_varlen_indeed()) {
          ChunkKey index_chunk_key = chunk_key;
          index_chunk_key[4] = 2;
          to_cache.push_back(chunk_key);
          to_cache.push_back(index_chunk_key);
        } else {
          to_cache.push_back(chunk_key);
        }
      }
    }
    if (to_cache.size() > 0) {
      cache->cacheTableChunks(to_cache);
    }
  }
}

void CsvDataWrapper::serializeDataWrapperInternals(const std::string& file_path) const {
  rapidjson::Document d;
  d.SetObject();

  // Save fragment map
  json_utils::add_value_to_object(d,
                                  fragment_id_to_file_regions_map_,
                                  "fragment_id_to_file_regions_map",
                                  d.GetAllocator());

  // Save csv_reader metadata
  rapidjson::Value reader_metadata(rapidjson::kObjectType);
  csv_reader_->serialize(reader_metadata, d.GetAllocator());
  d.AddMember("reader_metadata", reader_metadata, d.GetAllocator());

  json_utils::add_value_to_object(d, num_rows_, "num_rows", d.GetAllocator());
  json_utils::add_value_to_object(
      d, append_start_offset_, "append_start_offset", d.GetAllocator());

  json_utils::write_to_file(d, file_path);
}

void CsvDataWrapper::restoreDataWrapperInternals(
    const std::string& file_path,
    const ChunkMetadataVector& chunk_metadata) {
  auto d = json_utils::read_from_file(file_path);
  CHECK(d.IsObject());

  // Restore fragment map
  json_utils::get_value_from_object(
      d, fragment_id_to_file_regions_map_, "fragment_id_to_file_regions_map");

  // Construct csv_reader with metadta
  CHECK(d.HasMember("reader_metadata"));
  const auto copy_params = Csv::validate_and_get_copy_params(foreign_table_);
  const auto csv_file_path = getFullFilePath(foreign_table_);
  auto& server_options = foreign_table_->foreign_server->options;
  if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
    csv_reader_ = std::make_unique<LocalMultiFileReader>(
        csv_file_path, copy_params, d["reader_metadata"]);
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
      chunk_byte_count_[pair.first] = pair.second->numBytes;
    }
  }
  is_restored_ = true;
}

bool CsvDataWrapper::isRestored() const {
  return is_restored_;
}

const std::set<std::string_view> CsvDataWrapper::csv_table_options_{"ARRAY_DELIMITER",
                                                                    "ARRAY_MARKER",
                                                                    "BUFFER_SIZE",
                                                                    "DELIMITER",
                                                                    "ESCAPE",
                                                                    "HEADER",
                                                                    "LINE_DELIMITER",
                                                                    "LONLAT",
                                                                    "NULLS",
                                                                    "QUOTE",
                                                                    "QUOTED",
                                                                    "S3_ACCESS_TYPE"};
}  // namespace foreign_storage
