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
#include "ForeignTableSchema.h"
#include "ImportExport/Importer.h"
#include "Interval.h"
#include "LazyParquetImporter.h"

namespace foreign_storage {
class ParquetDataWrapper : public ForeignDataWrapper {
 public:
  ParquetDataWrapper(const int db_id, const ForeignTable* foreign_table);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(
      std::map<ChunkKey, AbstractBuffer*>& required_buffers,
      std::map<ChunkKey, AbstractBuffer*>& optional_buffers) override;

  static void validateOptions(const ForeignTable* foreign_table);

  static std::vector<std::string_view> getSupportedOptions();

  void serializeDataWrapperInternals(const std::string& file_path) const override;

  void restoreDataWrapperInternals(
      const std::string& file_path,
      const ChunkMetadataVector& chunk_metadata_vector) override;

  bool isRestored() const override;

 private:
  ParquetDataWrapper(const ForeignTable* foreign_table);

  std::list<const ColumnDescriptor*> getColumnsToInitialize(
      const Interval<ColumnType>& column_interval);
  void initializeChunkBuffers(const int fragment_index,
                              const Interval<ColumnType>& column_interval,
                              std::map<ChunkKey, AbstractBuffer*>& required_buffers,
                              const bool reserve_buffers_and_set_stats = false);
  void fetchChunkMetadata();
  void loadBuffersUsingLazyParquetImporter(
      const int logical_column_id,
      const int fragment_id,
      std::map<ChunkKey, AbstractBuffer*>& required_buffers);
  void loadBuffersUsingLazyParquetChunkLoader(
      const int logical_column_id,
      const int fragment_id,
      std::map<ChunkKey, AbstractBuffer*>& required_buffers);

  void validateFilePath() const;
  std::string getConfiguredFilePath() const;
  std::set<std::string> getProcessedFilePaths();
  std::set<std::string> getAllFilePaths();

  import_export::CopyParams validateAndGetCopyParams() const;

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
                                             const size_t expected_num_chars) const;

  bool moveToNextFragment(size_t new_rows_count) const;

  void finalizeFragmentMap();
  void addNewFragment(int row_group, const std::string& file_path);

  bool isNewFile(const std::string& file_path) const;

  void addNewFile(const std::string& file_path);

  void resetParquetMetadata();

  void updateStatsForEncoder(Encoder* encoder,
                             const SQLTypeInfo type_info,
                             const DataBlockPtr& data_block,
                             const size_t import_count);

  void loadMetadataChunk(const ColumnDescriptor* column,
                         const ChunkKey& chunk_key,
                         DataBlockPtr& data_block,
                         const size_t import_count,
                         const bool has_nulls,
                         const bool is_all_nulls,
                         const ArrayMetadataStats& array_stats);

  void loadChunk(const ColumnDescriptor* column,
                 const ChunkKey& chunk_key,
                 DataBlockPtr& data_block,
                 const size_t import_count,
                 std::map<ChunkKey, AbstractBuffer*>& required_buffers);

  import_export::Loader* getMetadataLoader(
      Catalog_Namespace::Catalog& catalog,
      const ParquetLoaderMetadata& parquet_loader_metadata);

  import_export::Loader* getChunkLoader(
      Catalog_Namespace::Catalog& catalog,
      const Interval<ColumnType>& column_interval,
      const int db_id,
      const int fragment_index,
      std::map<ChunkKey, AbstractBuffer*>& required_buffers);

  std::map<int, std::vector<RowGroupInterval>> fragment_to_row_group_interval_map_;
  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> chunk_metadata_map_;
  const int db_id_;
  const ForeignTable* foreign_table_;
  int last_fragment_index_;
  size_t last_fragment_row_count_;
  size_t total_row_count_;
  int last_row_group_;
  bool is_restored_;
  std::unique_ptr<ForeignTableSchema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;

  static constexpr std::array<char const*, 4> supported_options_{"BASE_PATH",
                                                                 "FILE_PATH",
                                                                 "ARRAY_DELIMITER",
                                                                 "ARRAY_MARKER"};
};
}  // namespace foreign_storage
