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

#include "AbstractTextFileDataWrapper.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>

#include <rapidjson/document.h>
#include <boost/filesystem.hpp>

#include "DataMgr/ForeignStorage/FileReader.h"
#include "DataMgr/ForeignStorage/ForeignTableSchema.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "FsiJsonUtils.h"
#include "Shared/misc.h"

namespace foreign_storage {
AbstractTextFileDataWrapper::AbstractTextFileDataWrapper()
    : db_id_(-1), foreign_table_(nullptr) {}

AbstractTextFileDataWrapper::AbstractTextFileDataWrapper(
    const int db_id,
    const ForeignTable* foreign_table)
    : db_id_(db_id), foreign_table_(foreign_table), is_restored_(false) {}

namespace {
std::set<const ColumnDescriptor*> get_columns(const ChunkToBufferMap& buffers,
                                              const Catalog_Namespace::Catalog& catalog,
                                              const int32_t table_id,
                                              const int fragment_id) {
  CHECK(!buffers.empty());
  std::set<const ColumnDescriptor*> columns;
  for (const auto& entry : buffers) {
    CHECK_EQ(fragment_id, entry.first[CHUNK_KEY_FRAGMENT_IDX]);
    const auto column_id = entry.first[CHUNK_KEY_COLUMN_IDX];
    const auto column = catalog.getMetadataForColumnUnlocked(table_id, column_id);
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
    const ChunkToBufferMap& optional_buffers) {
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
  populateChunks(column_id_to_chunk_map, fragment_id);
  updateMetadata(column_id_to_chunk_map, fragment_id);
  for (auto& entry : column_id_to_chunk_map) {
    entry.second.setBuffer(nullptr);
    entry.second.setIndexBuffer(nullptr);
  }
}

// if column was skipped during scan, update metadata now
void AbstractTextFileDataWrapper::updateMetadata(
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
        fragmenter->updateColumnChunkMetadata(column, fragment_id, cached_metadata);
      }
    }
  }
}

/**
 * Data structure containing data and metadata gotten from parsing a set of file regions.
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
    FileReader& file_reader,
    ParseBufferRequest& parse_file_request,
    const std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
    const TextFileBufferParser& parser) {
  ParseFileRegionResult load_file_region_result{};
  load_file_region_result.file_offset = file_regions[start_index].first_row_file_offset;
  load_file_region_result.row_count = 0;

  ParseBufferResult result;
  for (size_t i = start_index; i <= end_index; i++) {
    CHECK(file_regions[i].region_size <= parse_file_request.buffer_size);
    auto read_size = file_reader.readRegion(parse_file_request.buffer.get(),
                                            file_regions[i].first_row_file_offset,
                                            file_regions[i].region_size);
    CHECK_EQ(file_regions[i].region_size, read_size);
    parse_file_request.begin_pos = 0;
    parse_file_request.end_pos = file_regions[i].region_size;
    parse_file_request.first_row_index = file_regions[i].first_row_index;
    parse_file_request.file_offset = file_regions[i].first_row_file_offset;
    parse_file_request.process_row_count = file_regions[i].row_count;

    result = parser.parseBuffer(parse_file_request, i == end_index);
    CHECK_EQ(file_regions[i].row_count, result.row_count);
    load_file_region_result.row_count += result.row_count;
  }
  load_file_region_result.column_id_to_data_blocks_map =
      result.column_id_to_data_blocks_map;
  return load_file_region_result;
}

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
  if (size_known) {
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

void AbstractTextFileDataWrapper::populateChunks(
    std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
    int fragment_id) {
  const auto copy_params = getFileBufferParser().validateAndGetCopyParams(foreign_table_);

  CHECK(!column_id_to_chunk_map.empty());
  const auto& file_regions = fragment_id_to_file_regions_map_[fragment_id];
  CHECK(!file_regions.empty());

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
    parse_file_requests.emplace_back(
        buffer_size, copy_params, db_id_, foreign_table_, column_filter_set, file_path);
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
 * multiple threads while scanning files for metadata.
 */
struct MetadataScanMultiThreadingParams {
  std::queue<ParseBufferRequest> pending_requests;
  std::mutex pending_requests_mutex;
  std::condition_variable pending_requests_condition;
  std::queue<ParseBufferRequest> request_pool;
  std::mutex request_pool_mutex;
  std::condition_variable request_pool_condition;
  bool continue_processing;
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_encoder_buffers;
  std::map<ChunkKey, Chunk_NS::Chunk> cached_chunks;
  std::mutex chunk_encoder_buffers_mutex;
};

/**
 * Gets the next metadata scan request object from the pending requests queue.
 * A null optional is returned if there are no further requests to be processed.
 */
std::optional<ParseBufferRequest> get_next_metadata_scan_request(
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
      FileRegion(result.row_offsets.front(),
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
  }
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
        num_elements == request.getMaxFragRows()  // Is the last block for this chunk
    );
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

/**
 * Dispatches a new metadata scan request by adding the request to
 * the pending requests queue to be consumed by a worker thread.
 */
void dispatch_metadata_scan_request(
    MetadataScanMultiThreadingParams& multi_threading_params,
    ParseBufferRequest& request) {
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
 * Reads from a text file iteratively and dispatches metadata scan
 * requests that are processed by worker threads.
 */
void dispatch_metadata_scan_requests(
    const size_t& buffer_size,
    const std::string& file_path,
    FileReader& file_reader,
    const import_export::CopyParams& copy_params,
    MetadataScanMultiThreadingParams& multi_threading_params,
    size_t& first_row_index_in_buffer,
    size_t& current_file_offset,
    const TextFileBufferParser& parser) {
  auto alloc_size = buffer_size;
  auto residual_buffer = std::make_unique<char[]>(alloc_size);
  size_t residual_buffer_size = 0;
  size_t residual_buffer_alloc_size = alloc_size;

  while (!file_reader.isScanFinished()) {
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
    size += file_reader.read(request.buffer.get() + residual_buffer_size,
                             alloc_size - residual_buffer_size);

    if (size == 0) {
      // In some cases at the end of a file we will read 0 bytes even when
      // file_reader.isScanFinished() is false
      continue;
    } else if (size == 1 && request.buffer[0] == copy_params.line_delim) {
      // In some cases files with newlines at the end will be encoded with a second
      // newline that can end up being the only thing in the buffer
      current_file_offset++;
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
    chunk_metadata_map[chunk_key] = get_placeholder_metadata(column, num_elements);
  }
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
  auto& server_options = foreign_table_->foreign_server->options;
  if (foreign_table_->isAppendMode() && file_reader_ != nullptr) {
    parser.validateFiles(file_reader_.get(), foreign_table_);
    if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
      file_reader_->checkForMoreRows(append_start_offset_);
    } else {
      UNREACHABLE();
    }
  } else {
    // Should only be called once for non-append tables
    CHECK(chunk_metadata_map_.empty());
    CHECK(fragment_id_to_file_regions_map_.empty());
    if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
      file_reader_ = std::make_unique<LocalMultiFileReader>(
          file_path,
          copy_params,
          foreign_table_->getOption(
              AbstractFileStorageDataWrapper::REGEX_PATH_FILTER_KEY),
          foreign_table_->getOption(
              AbstractFileStorageDataWrapper::FILE_SORT_ORDER_BY_KEY),
          foreign_table_->getOption(AbstractFileStorageDataWrapper::FILE_SORT_REGEX_KEY));
    } else {
      UNREACHABLE();
    }
    parser.validateFiles(file_reader_.get(), foreign_table_);
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
                                                  getFullFilePath(foreign_table_));

      futures.emplace_back(std::async(std::launch::async,
                                      scan_metadata,
                                      std::ref(multi_threading_params),
                                      std::ref(fragment_id_to_file_regions_map_),
                                      std::ref(parser)));
    }

    try {
      dispatch_metadata_scan_requests(buffer_size,
                                      file_path,
                                      (*file_reader_),
                                      copy_params,
                                      multi_threading_params,
                                      num_rows_,
                                      append_start_offset_,
                                      getFileBufferParser());
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

  for (auto& [chunk_key, chunk_metadata] : chunk_metadata_map_) {
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
  }

  // Save chunk data
  if (foreign_table_->isAppendMode()) {
    chunk_encoder_buffers_ = std::move(multi_threading_params.chunk_encoder_buffers);
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
}  // namespace foreign_storage
