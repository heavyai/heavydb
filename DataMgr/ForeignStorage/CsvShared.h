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

#pragma once

#include <map>
#include <vector>

#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ForeignDataWrapper.h"
#include "ImportExport/Importer.h"

namespace foreign_storage {
/**
 * Data structure containing details about a CSV file region (subset of rows within a CSV
 * file).
 */
struct FileRegion {
  // Name of file containing region
  std::string filename;
  // Byte offset (within file) for the beginning of file region
  size_t first_row_file_offset;
  // Index of first row in file region relative to the first row/non-header line in the
  // file
  size_t first_row_index;
  // Number of rows in file region
  size_t row_count;
  // Size of file region in bytes
  size_t region_size;

  FileRegion(std::string name,
             size_t first_row_offset,
             size_t first_row_idx,
             size_t row_cnt,
             size_t region_sz)
      : filename(name)
      , first_row_file_offset(first_row_offset)
      , first_row_index(first_row_idx)
      , row_count(row_cnt)
      , region_size(region_sz) {}
  FileRegion() {}
  bool operator<(const FileRegion& other) const {
    return first_row_file_offset < other.first_row_file_offset;
  }
};

using FileRegions = std::vector<FileRegion>;

// Serialization functions for FileRegion
void set_value(rapidjson::Value& json_val,
               const FileRegion& file_region,
               rapidjson::Document::AllocatorType& allocator);

void get_value(const rapidjson::Value& json_val, FileRegion& file_region);

namespace Csv {

// Validate CSV Specific options
void validate_options(const ForeignTable* foreign_table);

import_export::CopyParams validate_and_get_copy_params(const ForeignTable* foreign_table);

// Return true if this used s3 select to access underlying CSV
bool validate_and_get_is_s3_select(const ForeignTable* foreign_table);

Chunk_NS::Chunk make_chunk_for_column(
    const ChunkKey& chunk_key,
    std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& chunk_metadata_map,
    const std::map<ChunkKey, AbstractBuffer*>& buffers);

}  // namespace Csv
}  // namespace foreign_storage
