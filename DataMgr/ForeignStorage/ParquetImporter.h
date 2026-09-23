/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <unordered_set>
#include <vector>

#include "AbstractFileStorageDataWrapper.h"
#include "Catalog/CatalogFwd.h"
#include "ForeignDataWrapper.h"
#include "ForeignTableSchema.h"
#include "ImportExport/ImportBatchResult.h"
#include "ImportExport/Importer.h"
#include "Interval.h"
#include "LazyParquetChunkLoader.h"

namespace foreign_storage {

class AbstractRowGroupIntervalTracker {
 public:
  virtual ~AbstractRowGroupIntervalTracker() = default;
  virtual std::optional<RowGroupInterval> getNextRowGroupInterval() = 0;
};

class ParquetImporter : public AbstractFileStorageDataWrapper {
 public:
  ParquetImporter();

  ParquetImporter(const int db_id,
                  const ForeignTable* foreign_table,
                  const UserMapping* user_mapping);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override;

  std::string getSerializedDataWrapper() const override;

  void restoreDataWrapperInternals(
      const std::string& file_path,
      const ChunkMetadataVector& chunk_metadata_vector) override;

  bool isRestored() const override;

  ParallelismLevel getCachedParallelismLevel() const override {
    UNREACHABLE();
    return {};
  }

  ParallelismLevel getNonCachedParallelismLevel() const override {
    UNREACHABLE();
    return {};
  }

  /**
   * Produce the next `ImportBatchResult` for import. This is the only
   * functionality of `ParquetImporter` that is required to be implemented.
   *
   * @return a `ImportBatchResult` for import.
   */
  std::unique_ptr<import_export::ImportBatchResult> getNextImportBatch();

  /**
   * Return string dictionaries that are used per column.
   *
   * @return a vector of StringDictionary and ColumnDescriptor pairs
   */
  std::vector<std::pair<const ColumnDescriptor*, StringDictionary*>>
  getStringDictionaries() const;

  /**
   * Get the maximum number of threads that can do useful computation.
   */
  int getMaxNumUsefulThreads() const;

  /**
   * Set the number of threads to use internally when reading batches.
   */
  void setNumThreads(const int num_threads);

 private:
  const int db_id_;
  const ForeignTable* foreign_table_;
  int num_threads_;

  std::set<std::string> getAllFilePaths();

  std::unique_ptr<AbstractRowGroupIntervalTracker> row_group_interval_tracker_;

  std::unique_ptr<ForeignTableSchema> schema_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  std::unique_ptr<FileReaderMap> file_reader_cache_;
  std::unique_ptr<HeavyColumnToParquetColumnMap> column_map_;
  std::vector<std::pair<const ColumnDescriptor*, StringDictionary*>>
      string_dictionaries_per_column_;

  std::shared_mutex row_group_interval_tracker_mutex_;
  std::shared_mutex string_dictionaries_per_column_mutex_;
};
}  // namespace foreign_storage
