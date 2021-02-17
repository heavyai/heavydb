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

#include "AbstractFileStorageDataWrapper.h"
#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ForeignDataWrapper.h"
#include "ForeignTableSchema.h"
#include "ImportExport/Importer.h"
#include "Interval.h"
#include "LazyParquetChunkLoader.h"

namespace foreign_storage {
class ParquetDataWrapper : public AbstractFileStorageDataWrapper {
 public:
  ParquetDataWrapper();

  ParquetDataWrapper(const int db_id, const ForeignTable* foreign_table);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(
      std::map<ChunkKey, AbstractBuffer*>& required_buffers,
      std::map<ChunkKey, AbstractBuffer*>& optional_buffers) override;

  void serializeDataWrapperInternals(const std::string& file_path) const override;

  void restoreDataWrapperInternals(
      const std::string& file_path,
      const ChunkMetadataVector& chunk_metadata_vector) override;

  bool isRestored() const override;

 private:
  std::list<const ColumnDescriptor*> getColumnsToInitialize(
      const Interval<ColumnType>& column_interval);
  void initializeChunkBuffers(const int fragment_index,
                              const Interval<ColumnType>& column_interval,
                              std::map<ChunkKey, AbstractBuffer*>& required_buffers,
                              const bool reserve_buffers_and_set_stats = false);
  void fetchChunkMetadata();
  void loadBuffersUsingLazyParquetChunkLoader(
      const int logical_column_id,
      const int fragment_id,
      std::map<ChunkKey, AbstractBuffer*>& required_buffers);

  std::set<std::string> getProcessedFilePaths();
  std::set<std::string> getAllFilePaths();

  bool moveToNextFragment(size_t new_rows_count) const;

  void finalizeFragmentMap();
  void addNewFragment(int row_group, const std::string& file_path);

  bool isNewFile(const std::string& file_path) const;

  void addNewFile(const std::string& file_path);

  void resetParquetMetadata();

  void metadataScanFiles(const std::set<std::string>& file_paths);

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
};
}  // namespace foreign_storage
