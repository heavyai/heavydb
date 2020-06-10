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
#include <regex>

#include <boost/filesystem.hpp>

#include "DataMgr/ForeignStorage/CsvFileBufferParser.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "ImportExport/Importer.h"
#include "Utils/DdlUtils.h"

namespace foreign_storage {
CsvDataWrapper::CsvDataWrapper(const int db_id, const ForeignTable* foreign_table)
    : db_id_(db_id), foreign_table_(foreign_table), row_count_(0) {}

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

ForeignStorageBuffer* CsvDataWrapper::getChunkBuffer(const ChunkKey& chunk_key) {
  auto timer = DEBUG_TIMER(__func__);
  if (chunk_buffer_map_.find(chunk_key) == chunk_buffer_map_.end()) {
    auto catalog = Catalog_Namespace::Catalog::get(db_id_);
    CHECK(catalog);
    const auto& column =
        catalog.get()->getMetadataForColumn(foreign_table_->tableId, chunk_key[2]);
    ForeignStorageBuffer* data_buffer = nullptr;
    ForeignStorageBuffer* index_buffer = nullptr;

    if (column->columnType.is_varlen() && !column->columnType.is_fixlen_array()) {
      ChunkKey data_chunk_key = {
          chunk_key[0], chunk_key[1], chunk_key[2], chunk_key[3], 1};
      chunk_buffer_map_[data_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      data_buffer = chunk_buffer_map_[data_chunk_key].get();

      ChunkKey index_chunk_key = {
          chunk_key[0], chunk_key[1], chunk_key[2], chunk_key[3], 2};
      chunk_buffer_map_[index_chunk_key] = std::make_unique<ForeignStorageBuffer>();
      index_buffer = chunk_buffer_map_[index_chunk_key].get();

      size_t index_offset_size;
      if (column->columnType.is_string()) {
        index_offset_size = sizeof(StringOffsetT);
      } else if (column->columnType.is_array()) {
        index_offset_size = sizeof(ArrayOffsetT);
      } else {
        UNREACHABLE();
      }
      index_buffer->reserve(index_offset_size *
                            (chunk_metadata_map_[chunk_key]->numElements + 1));
    } else {
      chunk_buffer_map_[chunk_key] = std::make_unique<ForeignStorageBuffer>();
      data_buffer = chunk_buffer_map_[chunk_key].get();
    }

    data_buffer->reserve(chunk_metadata_map_[chunk_key]->numBytes);
    Chunk_NS::Chunk chunk{column};
    chunk.setBuffer(data_buffer);
    chunk.setIndexBuffer(index_buffer);
    chunk.initEncoder();

    populateChunk(chunk_key, chunk);

    chunk.setBuffer(nullptr);
    chunk.setIndexBuffer(nullptr);
  }
  return getBufferFromMap(chunk_key);
}

/**
 * Adds an end of line character (specified by the line_delim parameter) to provided
 * buffer, if this is the last read buffer and if the buffer does not already end with an
 * end of line character. This allows for appropriate parsing by the
 * csv_file_buffer_parser utility functions, which expect the end of rows to be indicated
 * by end of line characters in the buffer.
 */
void add_end_of_line_if_needed(size_t& read_size,
                               const size_t buffer_size,
                               char* buffer,
                               const char line_delim) {
  if (read_size > 0 && read_size < buffer_size && buffer[read_size - 1] != line_delim) {
    buffer[read_size] = line_delim;
    read_size++;
  }
}

/**
 * Data structure containing data and metadata gotten from parsing a set of file regions.
 */
struct ParseFileRegionResult {
  size_t file_offset;
  size_t row_count;
  DataBlockPtr data_blocks;

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
    FILE* file,
    std::mutex& file_access_mutex,
    csv_file_buffer_parser::ParseBufferRequest& parse_file_request,
    const ChunkKey& chunk_key) {
  ParseFileRegionResult load_file_region_result{};
  load_file_region_result.file_offset = file_regions[start_index].first_row_file_offset;
  load_file_region_result.row_count = 0;

  csv_file_buffer_parser::ParseBufferResult result;
  for (size_t i = start_index; i <= end_index; i++) {
    CHECK(file_regions[i].region_size <= parse_file_request.buffer_size);
    size_t read_size;
    {
      std::lock_guard<std::mutex> lock(file_access_mutex);
      fseek(file, file_regions[i].first_row_file_offset, SEEK_SET);
      read_size =
          fread(parse_file_request.buffer.get(), 1, file_regions[i].region_size, file);
    }
    add_end_of_line_if_needed(read_size,
                              parse_file_request.buffer_size,
                              parse_file_request.buffer.get(),
                              parse_file_request.copy_params.line_delim);

    CHECK_EQ(file_regions[i].region_size, read_size);
    parse_file_request.begin_pos = 0;
    parse_file_request.end_pos = file_regions[i].region_size;
    parse_file_request.first_row_index = file_regions[i].first_row_index;
    parse_file_request.file_offset = file_regions[i].first_row_file_offset;
    parse_file_request.process_row_count = file_regions[i].row_count;

    result = parse_buffer(parse_file_request);
    CHECK(result.data_blocks.find(chunk_key[2]) != result.data_blocks.end());
    CHECK_EQ(file_regions[i].row_count, result.row_count);
    load_file_region_result.row_count += result.row_count;
  }
  load_file_region_result.data_blocks = result.data_blocks.find(chunk_key[2])->second;
  return load_file_region_result;
}

/**
 * Opens a file, at provided file path, as a binary file in read mode.
 * An exception is thrown if attempt to open the file fails.
 */
std::FILE* open_file(const std::string& file_path) {
  auto file = fopen(file_path.c_str(), "rb");
  if (!file) {
    throw std::runtime_error{"An error occurred when attempting to open file \"" +
                             file_path + "\". " + strerror(errno)};
  }
  return file;
}

/**
 * Gets the appropriate buffer size to be used when processing CSV file(s).
 */
size_t get_buffer_size(const import_export::CopyParams& copy_params,
                       const size_t file_size) {
  size_t buffer_size = copy_params.buffer_size;
  if (file_size < buffer_size) {
    buffer_size = file_size + 1;  // +1 for end of line character, if missing
  }
  return buffer_size;
}

/**
 * Gets the appropriate number of threads to be used for concurrent
 * processing within the data wrapper.
 */
size_t get_thread_count(const import_export::CopyParams& copy_params,
                        const size_t file_size,
                        const size_t buffer_size) {
  size_t thread_count = copy_params.threads;
  if (thread_count == 0) {
    thread_count = std::thread::hardware_concurrency();
  }
  size_t num_buffers_in_file = (file_size + buffer_size - 1) / buffer_size;
  if (num_buffers_in_file < thread_count) {
    thread_count = num_buffers_in_file;
  }
  CHECK(thread_count);
  return thread_count;
}

/**
 * Initializes import buffers for each of the provided columns.
 */
void initialize_import_buffers(
    const std::list<const ColumnDescriptor*>& columns,
    std::shared_ptr<Catalog_Namespace::Catalog> catalog,
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers) {
  for (const auto column : columns) {
    StringDictionary* string_dictionary = nullptr;
    if (column->columnType.is_dict_encoded_string() ||
        (column->columnType.is_array() && IS_STRING(column->columnType.get_subtype()) &&
         column->columnType.get_compression() == kENCODING_DICT)) {
      auto dict_descriptor =
          catalog->getMetadataForDict(column->columnType.get_comp_param());
      string_dictionary = dict_descriptor->stringDict.get();
    }
    import_buffers.emplace_back(
        std::make_unique<import_export::TypedImportBuffer>(column, string_dictionary));
  }
}

void CsvDataWrapper::populateChunk(ChunkKey chunk_key, Chunk_NS::Chunk& chunk) {
  const auto copy_params = validateAndGetCopyParams();
  const auto file_path = getFilePath();
  auto file = open_file(file_path);

  fseek(file, 0, SEEK_END);
  const size_t file_size = ftell(file);

  const auto buffer_size = get_buffer_size(copy_params, file_size);
  const auto thread_count = get_thread_count(copy_params, file_size, buffer_size);

  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);
  auto columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);

  const auto& file_regions = fragment_id_to_file_regions_map_[chunk_key[3]];
  const int batch_size = (file_regions.size() + thread_count - 1) / thread_count;

  std::vector<csv_file_buffer_parser::ParseBufferRequest> parse_file_requests{};
  parse_file_requests.reserve(thread_count);
  std::vector<std::future<ParseFileRegionResult>> futures{};

  for (size_t i = 0; i < file_regions.size(); i += batch_size) {
    parse_file_requests.emplace_back();
    csv_file_buffer_parser::ParseBufferRequest& parse_file_request =
        parse_file_requests.back();
    parse_file_request.buffer = std::make_unique<char[]>(buffer_size);
    parse_file_request.buffer_size = buffer_size;
    parse_file_request.copy_params = copy_params;
    parse_file_request.columns = columns;
    parse_file_request.catalog = catalog;
    initialize_import_buffers(columns, catalog, parse_file_request.import_buffers);

    auto start_index = i;
    auto end_index =
        std::min<size_t>(start_index + batch_size - 1, file_regions.size() - 1);
    futures.emplace_back(std::async(std::launch::async,
                                    parse_file_regions,
                                    std::ref(file_regions),
                                    start_index,
                                    end_index,
                                    file,
                                    std::ref(file_access_mutex_),
                                    std::ref(parse_file_request),
                                    std::ref(chunk_key)));
  }

  std::set<ParseFileRegionResult> load_file_region_results{};
  for (auto& future : futures) {
    future.wait();
    load_file_region_results.emplace(future.get());
  }
  fclose(file);

  for (auto result : load_file_region_results) {
    chunk.appendData(result.data_blocks, result.row_count, 0);
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
                     std::mutex& file_region_mutex,
                     int fragment_id,
                     size_t first_row_index,
                     const csv_file_buffer_parser::ParseBufferResult& result) {
  FileRegion file_region;
  file_region.first_row_file_offset = result.row_offsets.front();
  file_region.region_size = result.row_offsets.back() - file_region.first_row_file_offset;
  file_region.first_row_index = first_row_index;
  file_region.row_count = result.row_count;

  {
    std::lock_guard<std::mutex> lock(file_region_mutex);
    fragment_id_to_file_regions_map[fragment_id].emplace_back(file_region);
  }
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

/**
 * Updates metadata encapsulated in encoders for all table columns given
 * new data blocks gotten from parsing a new set of rows in a CSV file buffer.
 */
void update_metadata(MetadataScanMultiThreadingParams& multi_threading_params,
                     int fragment_id,
                     const csv_file_buffer_parser::ParseBufferRequest& request,
                     const csv_file_buffer_parser::ParseBufferResult& result,
                     std::map<int, const ColumnDescriptor*>& column_by_id) {
  for (auto& [column_id, data_block] : result.data_blocks) {
    ChunkKey chunk_key{request.db_id, request.table_id, column_id, fragment_id};
    const auto column = column_by_id[column_id];
    size_t byte_count;
    if (column->columnType.is_varlen() && !column->columnType.is_fixlen_array()) {
      chunk_key.emplace_back(1);
      byte_count = get_var_length_data_block_size(data_block, column->columnType);
    } else {
      byte_count = column->columnType.get_logical_size() * result.row_count;
    }

    {
      std::lock_guard<std::mutex> lock(multi_threading_params.chunk_byte_count_mutex);
      multi_threading_params.chunk_byte_count[chunk_key] += byte_count;
    }

    {
      std::lock_guard<std::mutex> lock(
          multi_threading_params.chunk_encoder_buffers_mutex);
      if (multi_threading_params.chunk_encoder_buffers.find(chunk_key) ==
          multi_threading_params.chunk_encoder_buffers.end()) {
        multi_threading_params.chunk_encoder_buffers[chunk_key] =
            std::make_unique<ForeignStorageBuffer>();
        multi_threading_params.chunk_encoder_buffers[chunk_key]->initEncoder(
            column->columnType);
      }
      update_stats(multi_threading_params.chunk_encoder_buffers[chunk_key]->encoder.get(),
                   column->columnType,
                   data_block,
                   result.row_count);
      size_t num_elements = multi_threading_params.chunk_encoder_buffers[chunk_key]
                                ->encoder->getNumElems() +
                            result.row_count;
      multi_threading_params.chunk_encoder_buffers[chunk_key]->encoder->setNumElems(
          num_elements);
    }
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
                   std::map<int, FileRegions>& fragment_id_to_file_regions_map,
                   std::mutex& file_region_mutex) {
  std::map<int, const ColumnDescriptor*> column_by_id{};
  while (true) {
    auto request_opt = get_next_metadata_scan_request(multi_threading_params);
    if (!request_opt.has_value()) {
      break;
    }
    auto& request = request_opt.value();
    if (column_by_id.empty()) {
      for (const auto column : request.columns) {
        column_by_id[column->columnId] = column;
      }
    }
    auto partitions = partition_by_fragment(
        request.first_row_index, request.max_fragment_rows, request.buffer_row_count);
    request.begin_pos = 0;
    size_t row_index = request.first_row_index;
    for (const auto partition : partitions) {
      request.process_row_count = partition;
      for (const auto& import_buffer : request.import_buffers) {
        import_buffer->clear();
      }
      auto result = parse_buffer(request);
      int fragment_id = row_index / request.max_fragment_rows;
      add_file_region(fragment_id_to_file_regions_map,
                      file_region_mutex,
                      fragment_id,
                      request.first_row_index,
                      result);
      update_metadata(multi_threading_params, fragment_id, request, result, column_by_id);
      row_index += result.row_count;
      request.begin_pos = result.row_offsets.back() - request.file_offset;
    }
    add_request_to_pool(multi_threading_params, request);
  }
}

/**
 * Gets the byte offset in a CSV file after skipping the header (if present).
 */
size_t get_offset_after_header(const std::string& file_path,
                               const import_export::CopyParams& copy_params) {
  size_t file_offset = 0;
  const auto& header_param = copy_params.has_header;
  if (header_param != import_export::ImportHeaderRow::NO_HEADER) {
    std::ifstream file{file_path};
    CHECK(file.good());
    std::string line;
    std::getline(file, line, copy_params.line_delim);
    file.close();
    file_offset = line.size() + 1;
  }
  return file_offset;
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
  std::unique_lock<std::mutex> pending_requests_lock(
      multi_threading_params.pending_requests_mutex);
  multi_threading_params.pending_requests.emplace(std::move(request));
  pending_requests_lock.unlock();
  multi_threading_params.pending_requests_condition.notify_all();
}

/**
 * Reads from a CSV file iteratively and dispatches metadata scan
 * requests that are processed by worker threads.
 */
void dispatch_metadata_scan_requests(
    const size_t buffer_size,
    const std::string& file_path,
    std::FILE* file,
    const import_export::CopyParams& copy_params,
    MetadataScanMultiThreadingParams& multi_threading_params) {
  auto residual_buffer = std::make_unique<char[]>(buffer_size);
  size_t residual_buffer_size = 0;
  size_t current_file_offset = get_offset_after_header(file_path, copy_params);
  size_t first_row_index_in_buffer = 0;
  fseek(file, current_file_offset, SEEK_SET);

  while (!feof(file)) {
    auto request = get_request_from_pool(multi_threading_params);
    if (residual_buffer_size > 0) {
      memcpy(request.buffer.get(), residual_buffer.get(), residual_buffer_size);
    }
    size_t size = residual_buffer_size;
    size += fread(request.buffer.get() + residual_buffer_size,
                  1,
                  buffer_size - residual_buffer_size,
                  file);
    add_end_of_line_if_needed(
        size, buffer_size, request.buffer.get(), copy_params.line_delim);

    unsigned int num_rows_in_buffer = 0;
    request.end_pos = import_export::delimited_parser::find_end(
        request.buffer.get(), size, copy_params, num_rows_in_buffer);
    request.first_row_index = first_row_index_in_buffer;
    request.file_offset = current_file_offset;
    request.buffer_row_count = num_rows_in_buffer;

    residual_buffer_size = size - request.end_pos;
    if (residual_buffer_size > 0) {
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
        return multi_threading_params.pending_requests.empty();
      });
  multi_threading_params.continue_processing = false;
  pending_requests_queue_lock.unlock();
  multi_threading_params.pending_requests_condition.notify_all();
}

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
 * @param chunk_key_prefix - chunk key prefix identifying a table within a database
 * @param chunk_metadata_vector - vector to be populated with chunk metadata
 */
void CsvDataWrapper::populateMetadataForChunkKeyPrefix(
    const ChunkKey& chunk_key_prefix,
    ChunkMetadataVector& chunk_metadata_vector) {
  // TODO: handle multiple files and zip files
  auto timer = DEBUG_TIMER(__func__);
  CHECK_EQ(chunk_key_prefix.size(), static_cast<size_t>(2));
  chunk_metadata_map_.clear();
  fragment_id_to_file_regions_map_.clear();

  const auto copy_params = validateAndGetCopyParams();
  const auto file_path = getFilePath();
  auto file = open_file(file_path);

  fseek(file, 0, SEEK_END);
  size_t file_size = ftell(file);

  auto buffer_size = get_buffer_size(copy_params, file_size);
  auto thread_count = get_thread_count(copy_params, file_size, buffer_size);

  auto catalog = Catalog_Namespace::Catalog::get(db_id_);
  CHECK(catalog);
  auto columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);
  std::map<int32_t, const ColumnDescriptor*> column_by_id{};
  for (auto column : columns) {
    column_by_id[column->columnId] = column;
  }

  MetadataScanMultiThreadingParams multi_threading_params;
  multi_threading_params.continue_processing = true;

  std::vector<std::future<void>> futures{};
  for (size_t i = 0; i < thread_count; i++) {
    futures.emplace_back(std::async(std::launch::async,
                                    scan_metadata,
                                    std::ref(multi_threading_params),
                                    std::ref(fragment_id_to_file_regions_map_),
                                    std::ref(file_regions_mutex_)));

    multi_threading_params.request_pool.emplace();
    csv_file_buffer_parser::ParseBufferRequest& parse_buffer_request =
        multi_threading_params.request_pool.back();
    initialize_import_buffers(columns, catalog, parse_buffer_request.import_buffers);
    parse_buffer_request.copy_params = copy_params;
    parse_buffer_request.columns = columns;
    parse_buffer_request.catalog = catalog;
    parse_buffer_request.db_id = db_id_;
    parse_buffer_request.table_id = foreign_table_->tableId;
    parse_buffer_request.max_fragment_rows = foreign_table_->maxFragRows;
    parse_buffer_request.buffer = std::make_unique<char[]>(buffer_size);
    parse_buffer_request.buffer_size = buffer_size;
  }

  dispatch_metadata_scan_requests(
      buffer_size, file_path, file, copy_params, multi_threading_params);
  for (auto& future : futures) {
    future.wait();
  }
  fclose(file);

  for (auto& [chunk_key, buffer] : multi_threading_params.chunk_encoder_buffers) {
    auto chunk_metadata =
        buffer->encoder->getMetadata(column_by_id[chunk_key[2]]->columnType);
    chunk_metadata->numElements = buffer->encoder->getNumElems();
    chunk_metadata->numBytes = multi_threading_params.chunk_byte_count[chunk_key];
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
    chunk_metadata_map_[chunk_key] = chunk_metadata;
  }

  for (auto& entry : fragment_id_to_file_regions_map_) {
    std::sort(entry.second.begin(), entry.second.end());
  }
}

ForeignStorageBuffer* CsvDataWrapper::getBufferFromMap(const ChunkKey& chunk_key) {
  CHECK(chunk_buffer_map_.find(chunk_key) != chunk_buffer_map_.end());
  return chunk_buffer_map_[chunk_key].get();
}
}  // namespace foreign_storage
