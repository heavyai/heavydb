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
#include "DataMgr/ForeignStorage/CsvReader.h"
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

  bool operator<(const FileRegion& other) const {
    return first_row_file_offset < other.first_row_file_offset;
  }
};

using FileRegions = std::vector<FileRegion>;

class CsvDataWrapper : public ForeignDataWrapper {
 public:
  CsvDataWrapper(const int db_id, const ForeignTable* foreign_table);

  ForeignStorageBuffer* getChunkBuffer(const ChunkKey& chunk_key) override;
  void populateMetadataForChunkKeyPrefix(
      const ChunkKey& chunk_key_prefix,
      ChunkMetadataVector& chunk_metadata_vector) override;
  static void validateOptions(const ForeignTable* foreign_table);

 private:
  CsvDataWrapper(const ForeignTable* foreign_table);

  /**
   * Populates provided chunk with appropriate data by parsing all file regions
   * containing chunk data.
   *
   * @param chunk_key - chunk key for chunk to be populated
   * @param chunk - object containing data and (optional) index buffers to be populated
   */
  void populateChunk(ChunkKey chunk_key, Chunk_NS::Chunk& chunk);

  ForeignStorageBuffer* getBufferFromMap(const ChunkKey& chunk_key);
  std::string getFilePath();
  import_export::CopyParams validateAndGetCopyParams();
  void validateFilePath();

  /**
   * Validates that the value of given table option has the expected number of characters.
   * An exception is thrown if the number of characters do not match.
   *
   * @param option_name - name of table option whose value is validated and returned
   * @param expected_num_chars - expected number of characters for option value
   * @return value of the option if the number of characters match. Returns an
   * empty string if table options do not contain provided option.
   */
  std::string validateAndGetStringWithLength(const std::string& option_name,
                                             const size_t expected_num_chars);

  /**
   * Validates that the string value of given table option is either "true" or "false"
   * (case insensitive). An exception is thrown if option value does not match one of
   * the two possible values.
   *
   * @param option_name - name of table option whose value is validated and returned
   * @return corresponding bool for option value. Returns an empty optional if table
   * options do not contain provided option.
   */
  std::optional<bool> validateAndGetBoolValue(const std::string& option_name);

  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_buffer_map_;
  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> chunk_metadata_map_;
  std::map<int, FileRegions> fragment_id_to_file_regions_map_;

  std::unique_ptr<CsvReader> csv_reader_;

  const int db_id_;
  const ForeignTable* foreign_table_;
  std::mutex file_access_mutex_;
  std::mutex file_regions_mutex_;

  static constexpr std::array<char const*, 13> supported_options_{"BASE_PATH",
                                                                  "FILE_PATH",
                                                                  "ARRAY_DELIMITER",
                                                                  "ARRAY_MARKER",
                                                                  "BUFFER_SIZE",
                                                                  "DELIMITER",
                                                                  "ESCAPE",
                                                                  "HEADER",
                                                                  "LINE_DELIMITER",
                                                                  "LONLAT",
                                                                  "NULLS",
                                                                  "QUOTE",
                                                                  "QUOTED"};
};
}  // namespace foreign_storage
