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
#include "ImportExport/Importer.h"
#include "Interval.h"
#include "LazyParquetImporter.h"
#include "ParquetForeignTableSchema.h"

namespace foreign_storage {
class ParquetDataWrapper : public ForeignDataWrapper {
 public:
  ParquetDataWrapper(const int db_id, const ForeignTable* foreign_table);

  ForeignStorageBuffer* getChunkBuffer(const ChunkKey& chunk_key) override;
  void populateMetadataForChunkKeyPrefix(
      const ChunkKey& chunk_key_prefix,
      ChunkMetadataVector& chunk_metadata_vector) override;

  static void validateOptions(const ForeignTable* foreign_table);

 private:
  ParquetDataWrapper(const ForeignTable* foreign_table);

  std::unique_ptr<ForeignStorageBuffer>& initializeChunkBuffer(const ChunkKey& chunk_key);

  std::list<const ColumnDescriptor*> getColumnsToInitialize(
      const Interval<ColumnType>& column_interval);
  void initializeChunkBuffers(const int fragment_index,
                              const Interval<ColumnType>& column_interval);
  void initializeChunkBuffers(const int fragment_index);
  void fetchChunkMetadata();
  ForeignStorageBuffer* getBufferFromMapOrLoadBufferIntoMap(const ChunkKey& chunk_key);
  ForeignStorageBuffer* loadBufferIntoMap(const ChunkKey& chunk_key);

  void validateFilePath();
  std::string getFilePath();
  import_export::CopyParams validateAndGetCopyParams();

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

  bool moveToNextFragment(size_t new_rows_count);

  void finalizeFragmentMap();
  void updateFragmentMap(int fragment_index, int row_group);

  void resetParquetMetadata();

  void updateStatsForBuffer(AbstractBuffer* buffer,
                            const DataBlockPtr& data_block,
                            const size_t import_count);

  void loadMetadataChunk(const ColumnDescriptor* column,
                         const ChunkKey& chunk_key,
                         DataBlockPtr& data_block,
                         const size_t import_count,
                         const bool has_nulls,
                         const bool is_all_nulls);

  void loadChunk(const ColumnDescriptor* column,
                 const ChunkKey& chunk_key,
                 DataBlockPtr& data_block,
                 const size_t import_count,
                 const bool metadata_only);

  import_export::Loader* getMetadataLoader(
      Catalog_Namespace::Catalog& catalog,
      const LazyParquetImporter::RowGroupMetadataVector& metadata_vector);

  import_export::Loader* getChunkLoader(Catalog_Namespace::Catalog& catalog,
                                        const Interval<ColumnType>& column_interval,
                                        const int db_id,
                                        const int fragment_index);

  struct FragmentToRowGroupInterval {
    int start_row_group_index;  // Row group where fragment starts
    int end_row_group_index;    // Row group where fragment ends
  };
  std::map<int, FragmentToRowGroupInterval> fragment_to_row_group_interval_map_;
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_buffer_map_;
  const int db_id_;
  const ForeignTable* foreign_table_;
  int last_fragment_index_;
  size_t last_fragment_row_count_;
  int last_row_group_;
  std::unique_ptr<ParquetForeignTableSchema> schema_;

  static constexpr std::array<char const*, 4> supported_options_{"BASE_PATH",
                                                                 "FILE_PATH",
                                                                 "ARRAY_DELIMITER",
                                                                 "ARRAY_MARKER"};
};
}  // namespace foreign_storage
