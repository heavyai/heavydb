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
#include <unordered_set>
#include <vector>

#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ForeignDataWrapper.h"
#include "Import/Importer.h"

namespace foreign_storage {
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

  void initializeChunkBuffers(const int fragment_index);
  void fetchChunkBuffers();
  ForeignStorageBuffer* getBufferFromMap(const ChunkKey& chunk_key);
  bool prefixMatch(const ChunkKey& prefix, const ChunkKey& checked);
  std::string getFilePath();
  Importer_NS::CopyParams validateAndGetCopyParams();

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

  bool fragmentIsFull();
  Importer_NS::Loader* getLoader(Catalog_Namespace::Catalog& catalog);

  std::mutex loader_mutex_;
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_buffer_map_;
  const int db_id_;
  const ForeignTable* foreign_table_;
  size_t row_count_;

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
