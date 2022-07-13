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

#pragma once

#include <string>
#include <vector>

#include <rapidjson/document.h>

namespace foreign_storage {
/**
 * Data structure containing details about a CSV file region (subset of rows within a CSV
 * file).
 */
struct FileRegion {
  // Name of file containing region
  std::string file_path;
  // Byte offset (within file) for the beginning of file region
  size_t first_row_file_offset;
  // Index of first row in file region relative to the first row/non-header line in the
  // file
  size_t first_row_index;
  // Number of rows in file region
  size_t row_count;
  // Size of file region in bytes
  size_t region_size;

  FileRegion(std::string path,
             size_t first_row_offset,
             size_t first_row_idx,
             size_t row_cnt,
             size_t region_sz)
      : file_path(path)
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
}  // namespace foreign_storage
