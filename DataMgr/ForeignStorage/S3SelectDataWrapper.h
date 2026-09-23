/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <vector>

#include "Catalog/CatalogFwd.h"
#include "DataMgr/ForeignStorage/CsvDataWrapper.h"
#include "DataMgr/ForeignStorage/CsvFileBufferParser.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapper.h"
#include "S3SelectClient.h"

namespace foreign_storage {

class S3SelectDataWrapper : public AbstractFileStorageDataWrapper {
 public:
  S3SelectDataWrapper();

  S3SelectDataWrapper(const int db_id,
                      const ForeignTable* foreign_table,
                      const UserMapping* user_mapping);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override;

  void validateTableOptions(const ForeignTable* foreign_table) const override;

  const std::set<std::string_view>& getSupportedTableOptions() const override;

  std::string getSerializedDataWrapper() const override;

  void restoreDataWrapperInternals(const std::string& file_path,
                                   const ChunkMetadataVector& chunk_metadata) override;
  bool isRestored() const override;

  void validateSchema(const std::list<ColumnDescriptor>& columns,
                      const ForeignTable* foreign_table) const override;

  ParallelismLevel getCachedParallelismLevel() const override { return INTRA_FRAGMENT; }

  ParallelismLevel getNonCachedParallelismLevel() const override {
    return INTRA_FRAGMENT;
  }

 private:
  void mapFileRegions(int partition_size);

  void initializeClient(const foreign_storage::UserMapping* user_mapping);

  void validateColumnCounts(std::vector<S3ScanRange>& first_line_ranges);

  std::vector<S3FileInfo>::iterator findProcessedFileInfo(const std::string& filename);

  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> chunk_metadata_map_;
  std::map<int, FileRegions> fragment_id_to_file_regions_map_;
  const int db_id_;
  const ForeignTable* foreign_table_;

  std::unique_ptr<S3SelectClient> select_client_;

  // Is this datawrapper restored from disk
  bool is_restored_;
  // Files with sizes we have previously scanned
  std::vector<S3FileInfo> processed_file_infos_;
  // Number of rows per fragment
  std::vector<size_t> fragment_row_counts_;

  static const CsvFileBufferParser csv_file_buffer_parser_;

  const UserMapping* user_mapping_;
};
}  // namespace foreign_storage
