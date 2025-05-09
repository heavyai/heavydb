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

#include <map>
#include <unordered_set>
#include <vector>

#include "AbstractFileStorageDataWrapper.h"
#include "DataMgr/Chunk/Chunk.h"
#include "DataPreview.h"
#include "ForeignDataWrapper.h"
#include "ForeignTableSchema.h"
#include "ImportExport/Importer.h"
#include "Interval.h"
#include "LazyParquetChunkLoader.h"

namespace arrow {
namespace fs {
class FileSystem;
}
}  // namespace arrow

namespace foreign_storage {

using FilePathAndRowGroup = std::pair<std::string, int32_t>;

class ParquetDataWrapper : public AbstractFileStorageDataWrapper {
 public:
  ParquetDataWrapper();

  ParquetDataWrapper(const int db_id,
                     const ForeignTable* foreign_table,
                     const bool do_metadata_stats_validation = true);

  /**
   * @brief Constructor intended for detect use-case only
   */
  ParquetDataWrapper(const ForeignTable* foreign_table,
                     std::shared_ptr<arrow::fs::FileSystem> file_system);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override;

  bool acceptsPrepopulatedDeleteBuffer() const override { return true; }

  std::string getSerializedDataWrapper() const override;

  void restoreDataWrapperInternals(
      const std::string& file_path,
      const ChunkMetadataVector& chunk_metadata_vector) override;

  bool isRestored() const override;

  ParallelismLevel getCachedParallelismLevel() const override { return INTER_FRAGMENT; }

  ParallelismLevel getNonCachedParallelismLevel() const override {
    return INTRA_FRAGMENT;
  }

  DataPreview getDataPreview(const size_t num_rows);

 private:
  std::list<const ColumnDescriptor*> getColumnsToInitialize(
      const Interval<ColumnType>& column_interval);
  void initializeChunkBuffers(const int fragment_index,
                              const Interval<ColumnType>& column_interval,
                              const ChunkToBufferMap& required_buffers,
                              const bool reserve_buffers_and_set_stats = false);
  void fetchChunkMetadata();
  void loadBuffersUsingLazyParquetChunkLoader(const int logical_column_id,
                                              const int fragment_id,
                                              const ChunkToBufferMap& required_buffers,
                                              AbstractBuffer* delete_buffer);

  std::vector<std::string> getOrderedProcessedFilePaths();
  std::vector<std::string> getAllFilePaths();

  bool moveToNextFragment(size_t new_rows_count) const;

  void finalizeFragmentMap();
  void addNewFragment(int row_group, const std::string& file_path);

  bool isNewFile(const std::string& file_path) const;

  void addNewFile(const std::string& file_path);

  void setLastFileRowCount(const std::string& file_path);

  void resetParquetMetadata();

  void metadataScanFiles(const std::vector<std::string>& file_paths);

  void metadataScanRowGroupMetadata(
      const std::list<RowGroupMetadata>& row_group_metadata);

  std::list<RowGroupMetadata> getRowGroupMetadataForFilePaths(
      const std::vector<std::string>& file_paths) const;

  std::map<FilePathAndRowGroup, RowGroupMetadata> getRowGroupMetadataMap(
      const std::vector<std::string>& file_paths) const;

  void updateChunkMetadataForFragment(
      const Interval<ColumnType>& column_interval,
      const std::list<std::shared_ptr<ChunkMetadata>>& column_chunk_metadata,
      int32_t fragment_id);

  void metadataScanRowGroupIntervals(
      const std::vector<RowGroupInterval>& row_group_intervals);

  void updateMetadataForRolledOffFiles(const std::set<std::string>& rolled_off_files);

  void removeMetadataForLastFile(const std::string& last_file_path);

  const bool do_metadata_stats_validation_;
  std::map<int, std::vector<RowGroupInterval>> fragment_to_row_group_interval_map_;
  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> chunk_metadata_map_;
  const int db_id_;
  const ForeignTable* foreign_table_;
  int last_fragment_index_;
  size_t last_fragment_row_count_;
  size_t total_row_count_;
  size_t last_file_row_count_;
  int last_row_group_;
  bool is_restored_;
  std::unique_ptr<ForeignTableSchema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  std::unique_ptr<FileReaderMap> file_reader_cache_;
  std::unique_ptr<HeavyColumnToParquetColumnMap> column_map_;

  std::mutex delete_buffer_mutex_;
};
}  // namespace foreign_storage
