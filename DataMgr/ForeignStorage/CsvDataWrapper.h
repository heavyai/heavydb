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

#include "AbstractFileStorageDataWrapper.h"
#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ForeignStorage/CsvReader.h"
#include "DataMgr/ForeignStorage/CsvShared.h"
#include "ForeignDataWrapper.h"
#include "ImportExport/Importer.h"

namespace foreign_storage {

class CsvDataWrapper : public AbstractFileStorageDataWrapper {
 public:
  CsvDataWrapper();

  CsvDataWrapper(const int db_id, const ForeignTable* foreign_table);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers) override;

  void validateTableOptions(const ForeignTable* foreign_table) const override;

  const std::set<std::string_view>& getSupportedTableOptions() const override;

  void serializeDataWrapperInternals(const std::string& file_path) const override;

  void restoreDataWrapperInternals(const std::string& file_path,
                                   const ChunkMetadataVector& chunk_metadata) override;
  bool isRestored() const override;

  ParallelismLevel getCachedParallelismLevel() const override { return INTRA_FRAGMENT; }

  ParallelismLevel getNonCachedParallelismLevel() const override {
    return INTRA_FRAGMENT;
  }

 private:
  CsvDataWrapper(const ForeignTable* foreign_table);

  /**
   * Populates provided chunks with appropriate data by parsing all file regions
   * containing chunk data.
   *
   * @param column_id_to_chunk_map - map of column id to chunks to be populated
   * @param fragment_id - fragment id of given chunks
   */
  void populateChunks(std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
                      int fragment_id);

  void populateChunkMapForColumns(const std::set<const ColumnDescriptor*>& columns,
                                  const int fragment_id,
                                  const ChunkToBufferMap& buffers,
                                  std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map);

  void updateMetadata(std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
                      int fragment_id);

  std::set<std::string_view> getAllCsvTableOptions() const;

  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> chunk_metadata_map_;
  std::map<int, FileRegions> fragment_id_to_file_regions_map_;

  std::unique_ptr<CsvReader> csv_reader_;

  const int db_id_;
  const ForeignTable* foreign_table_;

  // Data needed for append workflow
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_encoder_buffers_;
  std::map<ChunkKey, size_t> chunk_byte_count_;
  // How many rows have been read
  size_t num_rows_;
  // What byte offset we left off at in the csv_reader
  size_t append_start_offset_;
  // Is this datawrapper restored from disk
  bool is_restored_;
  static const std::set<std::string_view> csv_table_options_;
};
}  // namespace foreign_storage
