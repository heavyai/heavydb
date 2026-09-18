/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "S3SelectDataWrapper.h"

#include <queue>

#include <boost/filesystem.hpp>

#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ForeignStorage/CsvFileBufferParser.h"
#include "DataMgr/ForeignStorage/FileRegions.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/ForeignTableSchema.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "DataMgr/ForeignStorage/S3Utils.h"
#include "ImportExport/CopyParams.h"
#include "ImportExport/Importer.h"
#include "Shared/JsonUtils.h"

extern size_t g_max_import_threads;

extern bool g_allow_s3_server_privileges;

namespace foreign_storage {
S3SelectDataWrapper::S3SelectDataWrapper()
    : db_id_(-1), foreign_table_(nullptr), user_mapping_(nullptr) {}

S3SelectDataWrapper::S3SelectDataWrapper(const int db_id,
                                         const ForeignTable* foreign_table,
                                         const UserMapping* user_mapping)
    : db_id_(db_id)
    , foreign_table_(foreign_table)
    , is_restored_(false)
    , user_mapping_(user_mapping) {}

void S3SelectDataWrapper::validateTableOptions(const ForeignTable* foreign_table) const {
  CsvDataWrapper{}.validateTableOptions(foreign_table);
  if (allowFileRollOff(foreign_table)) {
    throw ForeignStorageException{
        "File roll off is not currently supported with S3 select access type."};
  }
  if (foreign_table->foreign_server->options.find(
          AbstractFileStorageDataWrapper::S3_ENDPOINT) !=
      foreign_table->foreign_server->options.end()) {
    throw ForeignStorageException{
        "S3_SELECT is not supported when using custom s3 endpoints."};
  }
}

void S3SelectDataWrapper::initializeClient(
    const foreign_storage::UserMapping* user_mapping) {
  const auto copy_params =
      csv_file_buffer_parser_.validateAndGetCopyParams(foreign_table_);

  if (user_mapping == nullptr && !g_allow_s3_server_privileges) {
    throw ForeignStorageException{"S3_SELECT access requires a valid User Mapping."};
  }
  select_client_ = std::make_unique<S3SelectClient>(getS3FileKey(foreign_table_),
                                                    foreign_table_->foreign_server,
                                                    get_credentials(user_mapping),
                                                    copy_params);
}

void S3SelectDataWrapper::validateSchema(const std::list<ColumnDescriptor>& columns,
                                         const ForeignTable*) const {
  bool contains_arrays = false;
  bool contains_geotypes = false;
  for (auto column : columns) {
    contains_arrays |= column.columnType.is_array();
    contains_geotypes |= column.columnType.is_geometry();
  }
  if (contains_arrays || contains_geotypes) {
    throw ForeignStorageException{
        "Geo Types and Array Types not currently supported with S3_ACCESS_TYPE : "
        "S3_SELECT. Please use S3_ACCESS_TYPE : S3_DIRECT"};
  }
}

namespace {
// Fetch requested columns using S3 select and convert to import buffers
foreign_storage::ParseBufferResult fetch_and_import(
    const std::unique_ptr<foreign_storage::S3SelectClient>& s3_client,
    const ForeignTable* foreign_table,
    const foreign_storage::FileRegion& region,
    ParseBufferRequest& request,
    const std::set<int>& columns_to_parse,
    const CsvFileBufferParser& parser) {
  auto range = S3ScanRange(region.file_path,
                           region.first_row_file_offset,
                           region.first_row_file_offset + region.region_size -
                               1);  // ScanRange is inclusive so -1 here

  // TODO - get mapping of columns for geos if they are supported
  std::vector<int> column_ids;
  for (int column_id : columns_to_parse) {
    column_ids.push_back(column_id - 1);
  }
  auto columns = s3_client->getColumnsAsCsv(column_ids, range);

  request.buffer_size = columns.size();
  request.buffer = std::make_unique<char[]>(columns.size() + 1);
  strncpy(request.buffer.get(), columns.c_str(), columns.size() + 1);
  request.begin_pos = 0;
  request.end_pos = region.region_size;
  request.first_row_index = region.first_row_index;
  request.file_offset = region.first_row_file_offset;
  request.process_row_count = region.row_count;
  return parser.parseBuffer(request, true, true);
}

// s3 select will reformat output so change copy params to reflect this
// other parameters such as 'null_str' are still valid
import_export::CopyParams get_copy_params_for_s3_output(
    const ForeignTable* foreign_table,
    const CsvFileBufferParser& parser) {
  auto copy_params = parser.validateAndGetCopyParams(foreign_table);
  copy_params.delimiter = ',';
  copy_params.quote = '"';
  copy_params.line_delim = '\n';
  copy_params.escape = '"';
  return copy_params;
}

}  // namespace

const std::set<std::string_view>& S3SelectDataWrapper::getSupportedTableOptions() const {
  return CsvDataWrapper{}.getSupportedTableOptions();
}

void S3SelectDataWrapper::populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                                               const ChunkToBufferMap& optional_buffers,
                                               AbstractBuffer* delete_buffer) {
  auto fragment_id = required_buffers.begin()->first[CHUNK_KEY_FRAGMENT_IDX];
  std::map<ChunkKey, Chunk_NS::Chunk> chunks;
  std::set<int> columns_to_parse;
  for (auto& buffers : {required_buffers, optional_buffers}) {
    for (const auto& [chunk_key, buffer] : buffers) {
      CHECK(fragment_id == chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
      if (!is_varlen_index_key(chunk_key)) {
        ChunkKey base_key =
            ChunkKey(chunk_key.begin(), chunk_key.begin() + CHUNK_KEY_FRAGMENT_IDX + 1);
        init_chunk_for_column(base_key, chunk_metadata_map_, buffers, chunks[chunk_key]);
        columns_to_parse.emplace(chunk_key[CHUNK_KEY_COLUMN_IDX]);
      }
    }
  }

  // Requests need to exist until data_blocks are added to chunks
  std::vector<ParseBufferRequest> requests;
  std::vector<std::future<ParseBufferResult>> futures{};
  auto num_regions = fragment_id_to_file_regions_map_[fragment_id].size();
  requests.reserve(num_regions);
  futures.reserve(num_regions);

  // Default cap, will be tuned with performance testing
  size_t max_threads = g_max_import_threads;
  std::set<int> outstanding_threads;
  size_t next_import_region = 0;
  auto end = fragment_id_to_file_regions_map_[fragment_id].end();
  // S3 Select will reformat with standard delimiters
  const auto copy_params =
      get_copy_params_for_s3_output(foreign_table_, csv_file_buffer_parser_);

  for (auto iter = fragment_id_to_file_regions_map_[fragment_id].begin(); iter != end;
       iter++) {
    outstanding_threads.insert(requests.size());
    requests.emplace_back(0,  // Buffer size is currently unknown
                          copy_params,
                          required_buffers.begin()->first[CHUNK_KEY_DB_IDX],
                          foreign_table_,
                          columns_to_parse,
                          getFullFilePath(foreign_table_));
    // Launch async to fetch data and convert to import buffers
    futures.emplace_back(std::async(std::launch::async,
                                    fetch_and_import,
                                    std::ref(select_client_),
                                    foreign_table_,
                                    std::ref(*iter),
                                    std::ref(requests.back()),
                                    std::ref(columns_to_parse),
                                    std::ref(csv_file_buffer_parser_)));

    // Cap max outstanding threads
    // Finish all outstanding threads
    if (iter + 1 == end) {
      max_threads = 0;
    }
    std::chrono::milliseconds span(0);
    while (outstanding_threads.size() > max_threads) {
      auto iter = outstanding_threads.begin();
      // Iterate through threads until we find one that is finished
      while (iter != outstanding_threads.end() &&
             outstanding_threads.size() > max_threads) {
        if (!futures[*iter].valid() ||
            futures[*iter].wait_for(span) == std::future_status::ready) {
          iter = outstanding_threads.erase(iter);
        } else {
          iter++;
        }
      }
    }

    // Append import buffers from all completed threads in order
    while ((next_import_region < futures.size()) &&
           futures[next_import_region].wait_for(span) == std::future_status::ready) {
      auto result = futures[next_import_region++].get();
      if (result.row_count > 0) {
        for (auto& buffers : {required_buffers, optional_buffers}) {
          for (const auto& [chunk_key, buffer] : buffers) {
            if (!is_varlen_index_key(chunk_key)) {
              CHECK(chunks.find(chunk_key) != chunks.end());
              chunks[chunk_key].appendData(
                  result.column_id_to_data_blocks_map[chunk_key[CHUNK_KEY_COLUMN_IDX]],
                  result.row_count,
                  0);
            }
          }
        }
      }
    }
  }

  // Update cached metadata
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  for (auto& buffers : {required_buffers, optional_buffers}) {
    for (const auto& [chunk_key, buffer] : buffers) {
      if (!is_varlen_index_key(chunk_key)) {
        CHECK(chunks.find(chunk_key) != chunks.end());
        CHECK_EQ(chunks[chunk_key].getBuffer()->getEncoder()->getNumElems(),
                 chunk_metadata_map_[chunk_key]->numElements);
        chunk_metadata_map_[chunk_key] = std::make_shared<ChunkMetadata>(
            chunks[chunk_key].getBuffer()->getEncoder()->getMetadata());
      }
    }
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
    const std::vector<size_t>& fragment_sizes,
    const int first_fragment_id,
    std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& chunk_metadata_map) {
  ChunkKey chunk_key = {db_id, foreign_table->tableId, column->columnId, 0};
  // Handle varlen column
  if (column->columnType.is_varlen_indeed()) {
    chunk_key.emplace_back(1);
  }
  // Create default metadata for each fragment starting at first_fragment_id
  for (size_t fragment_id = first_fragment_id; fragment_id < fragment_sizes.size();
       fragment_id++) {
    size_t num_elements = fragment_sizes[fragment_id];
    chunk_key[CHUNK_KEY_FRAGMENT_IDX] = fragment_id;
    chunk_metadata_map[chunk_key] = std::make_shared<ChunkMetadata>(
        get_placeholder_metadata(column->columnType, num_elements, {}));
  }
}

// Construct range_size or smaller scan ranges between start and end
void add_scan_ranges(std::string file,
                     size_t start,
                     size_t end,
                     size_t range_size,
                     std::vector<S3ScanRange>& result) {
  size_t partition_start = start;
  while (partition_start <= end) {
    size_t partition_end = partition_start + range_size - 1;
    if (partition_end > end) {
      partition_end = end;
    }
    result.emplace_back(file, partition_start, partition_end);
    partition_start = partition_end + 1;
  }
}

// Split S3ScanRange into num_parts and add to result
void divide_scan_range(const S3ScanRange& range,
                       int num_parts,
                       std::vector<S3ScanRange>& result) {
  // Inclusive range, so add 1 to find total size
  size_t partition_size =
      std::ceil(((range.byte_range->second - range.byte_range->first + 1) / num_parts));
  add_scan_ranges(range.file_name,
                  range.byte_range->first,
                  range.byte_range->second,
                  partition_size,
                  result);
}
}  // namespace

/**
 * Populates provided chunk metadata vector with metadata for table specified in given
 * chunk key.
 *
 * @param chunk_metadata_vector - vector to be populated with chunk metadata
 */
void S3SelectDataWrapper::populateChunkMetadata(
    ChunkMetadataVector& chunk_metadata_vector) {
  auto timer = DEBUG_TIMER(__func__);

  const auto copy_params =
      csv_file_buffer_parser_.validateAndGetCopyParams(foreign_table_);
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);

  initializeClient(user_mapping_);

  // split file into region and count rows to find size of each fragment
  mapFileRegions(copy_params.buffer_size);

  for (auto& [chunk_key, chunk_metadata] : chunk_metadata_map_) {
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
  }
}

/**
 * Fetch first line of each range and confirms the column count is correct
 *
 * @param first_line_ranges - vector of scan range of all files to check
 */
void S3SelectDataWrapper::validateColumnCounts(
    std::vector<S3ScanRange>& first_line_ranges) {
  // Get first row from each file
  const auto sample_rows = select_client_->getFirstRowAsCsv(first_line_ranges);

  // Verify columns count for each row
  // S3 Select will reformat with standard delimiters
  const auto copy_params =
      get_copy_params_for_s3_output(foreign_table_, csv_file_buffer_parser_);
  int num_columns = ForeignTableSchema(db_id_, foreign_table_).numLogicalColumns();
  CHECK(sample_rows.size() == first_line_ranges.size());
  for (size_t i = 0; i < sample_rows.size(); i++) {
    csv_file_buffer_parser_.validateExpectedColumnCount(
        sample_rows[i],
        copy_params,
        num_columns,
        0,  // no point cols as we dont support GEO types
        foreign_table_->foreign_server->options
                .find(AbstractFileStorageDataWrapper::S3_BUCKET_KEY)
                ->second +
            "/" + first_line_ranges[i].file_name);
  }
}

/**
 * Scan the file set, breaking each file into subregions
 * Populate the fragment_id_to_file_regions_map_ with each region
 * Populate the chunk_metadata_map_ with default metadata
 * @param partition_size - default size of regions to partition files into
 */
void S3SelectDataWrapper::mapFileRegions(int partition_size) {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  std::vector<S3ScanRange> ranges;
  // Scan range of each new file to check first line for # of rows
  std::vector<S3ScanRange> first_line_ranges;
  auto new_file_infos = select_client_->getFileInfos(getFilePathOptions(foreign_table_));
  if (foreign_table_->isAppendMode() && !processed_file_infos_.empty()) {
    // Check we are in a legal state
    if (processed_file_infos_.size() == 1 && new_file_infos.size() == 1) {
      // Possible file append
      const std::string old_filename = processed_file_infos_.begin()->first;
      const std::string new_filename = new_file_infos.begin()->first;
      size_t old_size = processed_file_infos_.begin()->second;
      size_t new_size = new_file_infos.begin()->second;
      // Check we are in a legal state
      if (old_filename != new_filename) {
        throw_removed_file_error(old_filename);
      } else if (new_size < old_size) {
        throw_removed_row_in_file_error(old_filename);
      } else if (new_size > old_size) {
        // add remainder of file to ranges
        add_scan_ranges(old_filename, old_size, new_size - 1, partition_size, ranges);
        auto processed_file_infos_it = findProcessedFileInfo(old_filename);
        processed_file_infos_it->second = new_size;
      }  // else file is unchanged, add nothing to ranges

    } else {
      // Possible added files
      // Add any new files to processed_file_infos_ and scan ranges to ranges
      for (const auto& [filename, file_size] : new_file_infos) {
        if (findProcessedFileInfo(filename) == processed_file_infos_.end()) {
          // Make an initial partitioning of the file
          add_scan_ranges(filename, 0, file_size - 1, partition_size, ranges);
          first_line_ranges.emplace_back(filename);
          processed_file_infos_.emplace_back(std::make_pair(filename, file_size));
        }
      }
    }

  } else {
    // ensure no previous data
    CHECK(processed_file_infos_.empty());
    CHECK(fragment_row_counts_.empty());
    CHECK(fragment_id_to_file_regions_map_.empty());
    // Add all files to processed_file_infos_ and scan ranges to ranges
    for (const auto& [filename, file_size] : new_file_infos) {
      // Make an initial partitioning of the file
      add_scan_ranges(filename, 0, file_size - 1, partition_size, ranges);
      first_line_ranges.emplace_back(filename);
      processed_file_infos_.emplace_back(std::pair{filename, file_size});
    }
  }

  if (first_line_ranges.size() > 0) {
    validateColumnCounts(first_line_ranges);
  }

  // Get the row counts for each range
  std::vector<std::pair<S3ScanRange, size_t>> range_row_counts;
  // Some ranges may need so be split if bigger than the fragment size, so multiple passes
  // may be needed
  while (ranges.size()) {
    auto row_counts = select_client_->getNumRows(ranges);
    CHECK(row_counts.size() == ranges.size());
    // Keep track of regions too big to fit in a fragment
    std::vector<S3ScanRange> split_ranges;
    // Iterate over row counts adding them to split_ranges if too big
    for (size_t i = 0; i < ranges.size(); i++) {
      auto row_count = row_counts[i];
      if (row_count > foreign_table_->maxFragRows) {
        // Too many rows to fit into a single frament
        // Split into parts and add to split_ranges
        // Multiply for 1.5 for margin of error to reduce iterations
        int num_parts = std::ceil((1.5 * row_count) / foreign_table_->maxFragRows);
        // Need to split into at least 2 parts
        CHECK(num_parts > 1);
        divide_scan_range(ranges[i], num_parts, split_ranges);
      } else {
        // Add to vector ordered by regex sort key
        range_row_counts.emplace_back(std::pair{ranges[i], row_count});
      }
    }
    // Next iteration will rescan ranges that were too big
    ranges = std::move(split_ranges);
  }

  if (fragment_row_counts_.size() == 0) {
    // Initialize first empty fragment
    fragment_row_counts_.push_back(0);
  }
  // If appending, start at last fragment
  int start_frag_id = fragment_row_counts_.size() - 1;

  // Add ranges to fragment map
  for (const auto& [range, row_count] : range_row_counts) {
    if (static_cast<int>(row_count + fragment_row_counts_.back()) >
        foreign_table_->maxFragRows) {
      fragment_row_counts_.push_back(row_count);
    } else {
      fragment_row_counts_.back() += row_count;
    }
    CHECK(range.byte_range != std::nullopt);
    int frag_id = fragment_row_counts_.size() - 1;
    fragment_id_to_file_regions_map_[frag_id].emplace_back(FileRegion(
        range.file_name,
        range.byte_range->first,                    // start offset
        fragment_row_counts_[frag_id] - row_count,  // first row in the current fragment
        row_count,                                  // num rows
        range.byte_range->second - range.byte_range->first +
            1));  // num bytes from inclusive ScanRange
  }

  // Create placeholder metadata for any new/updated fragments
  auto columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);
  for (auto column : columns) {
    add_placeholder_metadata(column,
                             foreign_table_,
                             db_id_,
                             fragment_row_counts_,
                             start_frag_id,
                             chunk_metadata_map_);
  }
}

std::string S3SelectDataWrapper::getSerializedDataWrapper() const {
  rapidjson::Document d;
  d.SetObject();

  // Save fragment map
  json_utils::add_value_to_object(d,
                                  fragment_id_to_file_regions_map_,
                                  "fragment_id_to_file_regions_map",
                                  d.GetAllocator());

  json_utils::add_value_to_object(
      d, processed_file_infos_, "processed_file_infos", d.GetAllocator());
  json_utils::add_value_to_object(
      d, fragment_row_counts_, "fragment_row_counts", d.GetAllocator());
  return json_utils::write_to_string(d);
}

void S3SelectDataWrapper::restoreDataWrapperInternals(
    const std::string& file_path,
    const ChunkMetadataVector& chunk_metadata) {
  auto d = json_utils::read_from_file(file_path);
  CHECK(d.IsObject());

  // Restore fragment map
  json_utils::get_value_from_object(
      d, fragment_id_to_file_regions_map_, "fragment_id_to_file_regions_map");
  json_utils::get_value_from_object(d, processed_file_infos_, "processed_file_infos");
  json_utils::get_value_from_object(d, fragment_row_counts_, "fragment_row_counts");

  // Now restore the internal metadata maps
  CHECK(chunk_metadata_map_.empty());
  for (auto& [key, metadata] : chunk_metadata) {
    chunk_metadata_map_[key] = metadata;
  }
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  initializeClient(user_mapping_);
  is_restored_ = true;
}

bool S3SelectDataWrapper::isRestored() const {
  return is_restored_;
}

const CsvFileBufferParser S3SelectDataWrapper::csv_file_buffer_parser_{};

std::vector<S3FileInfo>::iterator S3SelectDataWrapper::findProcessedFileInfo(
    const std::string& filename) {
  return std::find_if(processed_file_infos_.begin(),
                      processed_file_infos_.end(),
                      [&filename](const S3FileInfo& file_info) -> bool {
                        return filename == (file_info.first);
                      });
}
}  // namespace foreign_storage
