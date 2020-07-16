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
#include "ForeignTableColumnMap.h"
#include "ImportExport/Importer.h"
#include "Interval.h"
#include "LazyParquetImporter.h"

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
  void initializeChunkBuffers(const Interval<FragmentType>& fragment_interval,
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
  bool newRowGroup(int row_group);

  void updateRowGroupMetadata(int row_group);

  void finalizeFragmentMap();
  void updateFragmentMap(int fragment_index, int row_group);

  void resetParquetMetadata();

  void shiftData(DataBlockPtr& data_block,
                 const size_t import_shift,
                 const size_t element_size);

  void updateStatsForBuffer(AbstractBuffer* buffer,
                            const DataBlockPtr& data_block,
                            const size_t import_count,
                            const size_t import_shift);

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
                 const size_t import_shift,
                 const bool metadata_only = false,
                 const bool first_fragment = false,
                 const size_t element_size = 0);

  import_export::Loader* getMetadataLoader(
      Catalog_Namespace::Catalog& catalog,
      const LazyParquetImporter::RowGroupMetadataVector& metadata_vector);

  import_export::Loader* getChunkLoader(Catalog_Namespace::Catalog& catalog,
                                        const Interval<FragmentType>& fragment_interval,
                                        const Interval<ColumnType>& column_interval,
                                        const int chunk_key_db);

  size_t getElementSizeFromImportBuffer(
      const std::unique_ptr<import_export::TypedImportBuffer>& import_buffer) const;

  struct IntervalsToLoad {
    Interval<RowGroupType> row_group_interval;
    Interval<ColumnType> column_interval;
    Interval<FragmentType> fragment_interval;
  };

  /**
   * Determines intervals of row groups, columns and fragments to load based on a
   * request for a given chunk key.
   *
   * @param chunk_key - chunk key that is to be loaded
   *
   * @return - (inclusive) intervals of row groups, columns, and fragments respectively,
   * to load
   */
  IntervalsToLoad getRowGroupsColumnsAndFragmentsToLoad(const ChunkKey& chunk_key);

  struct FragmentToRowGroupInterval {
    int64_t start_row_group_line;  // Line within row group where fragment starts
    int start_row_group_index;     // Row group where fragment starts
    int end_row_group_index;       // Row group where fragment ends
  };
  std::map<int, FragmentToRowGroupInterval> fragment_to_row_group_interval_map_;
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_buffer_map_;
  const int db_id_;
  const ForeignTable* foreign_table_;
  size_t row_count_;

  int max_row_group_;
  int current_row_group_;
  int64_t row_group_row_count_;
  size_t partial_import_row_count_;

  ForeignTableColumnMap foreign_table_column_map_;

  static constexpr std::array<char const*, 4> supported_options_{"BASE_PATH",
                                                                 "FILE_PATH",
                                                                 "ARRAY_DELIMITER",
                                                                 "ARRAY_MARKER"};
};
}  // namespace foreign_storage
