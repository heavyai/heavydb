/*
 * Copyright 2021 OmniSci, Inc.
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
#include <set>

#include "AbstractFileStorageDataWrapper.h"
#include "Catalog/CatalogFwd.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ForeignStorage/FileReader.h"
#include "DataMgr/ForeignStorage/FileRegions.h"
#include "DataMgr/ForeignStorage/TextFileBufferParser.h"

namespace foreign_storage {

class AbstractTextFileDataWrapper : public AbstractFileStorageDataWrapper {
 public:
  AbstractTextFileDataWrapper();

  AbstractTextFileDataWrapper(const int db_id, const ForeignTable* foreign_table);

  AbstractTextFileDataWrapper(const int db_id,
                              const ForeignTable* foreign_table,
                              const UserMapping* user_mapping,
                              const bool disable_cache);

  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;

  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer) override;

  std::string getSerializedDataWrapper() const override;

  void restoreDataWrapperInternals(const std::string& file_path,
                                   const ChunkMetadataVector& chunk_metadata) override;
  bool isRestored() const override;

  ParallelismLevel getCachedParallelismLevel() const override { return INTRA_FRAGMENT; }

  ParallelismLevel getNonCachedParallelismLevel() const override {
    return INTRA_FRAGMENT;
  }

  void createRenderGroupAnalyzers() override;

 protected:
  virtual const TextFileBufferParser& getFileBufferParser() const = 0;

 private:
  AbstractTextFileDataWrapper(const ForeignTable* foreign_table);

  /**
   * Populates provided chunks with appropriate data by parsing all file regions
   * containing chunk data.
   *
   * @param column_id_to_chunk_map - map of column id to chunks to be populated
   * @param fragment_id - fragment id of given chunks
   * @param delete_buffer - optional buffer to store deleted row indices
   */
  void populateChunks(std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
                      int fragment_id,
                      AbstractBuffer* delete_buffer);

  void populateChunkMapForColumns(const std::set<const ColumnDescriptor*>& columns,
                                  const int fragment_id,
                                  const ChunkToBufferMap& buffers,
                                  std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map);

  void updateMetadata(std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
                      int fragment_id);

  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> chunk_metadata_map_;
  std::map<int, FileRegions> fragment_id_to_file_regions_map_;

  std::unique_ptr<FileReader> file_reader_;

  const int db_id_;
  const ForeignTable* foreign_table_;

  // Data needed for append workflow
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_encoder_buffers_;
  // How many rows have been read
  size_t num_rows_;
  // What byte offset we left off at in the file_reader
  size_t append_start_offset_;
  // Is this datawrapper restored from disk
  bool is_restored_;

  const UserMapping* user_mapping_;

  // Force cache to be disabled
  const bool disable_cache_;

  // declared in three derived classes to avoid
  // polluting ForeignDataWrapper virtual base
  // @TODO refactor to lower class if needed
  RenderGroupAnalyzerMap render_group_analyzer_map_;
};
}  // namespace foreign_storage
