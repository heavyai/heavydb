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
#include <queue>
#include <set>

#include "AbstractFileStorageDataWrapper.h"
#include "Catalog/CatalogFwd.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ForeignStorage/FileReader.h"
#include "DataMgr/ForeignStorage/FileRegions.h"
#include "DataMgr/ForeignStorage/TextFileBufferParser.h"

namespace foreign_storage {

/**
 * Data structure used to hold shared objects needed for inter-thread
 * synchronization or objects containing data that is updated by
 * multiple threads while scanning files for metadata.
 */
struct MetadataScanMultiThreadingParams {
  std::queue<ParseBufferRequest> pending_requests;
  std::queue<ParseBufferRequest>
      deferred_requests;  // holds requests that will be processed in the next iteration
                          // during an iterative file scan
  std::mutex pending_requests_mutex;
  std::mutex deferred_requests_mutex;
  std::condition_variable pending_requests_condition;
  std::queue<ParseBufferRequest> request_pool;
  std::mutex request_pool_mutex;
  std::condition_variable request_pool_condition;
  bool continue_processing;
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_encoder_buffers;
  std::map<ChunkKey, Chunk_NS::Chunk> cached_chunks;
  std::mutex chunk_encoder_buffers_mutex;
  bool disable_cache;
};

struct IterativeFileScanParameters {
  std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map;
  int32_t fragment_id;
  AbstractBuffer* delete_buffer;

  mutable std::map<int, std::unique_ptr<std::mutex>> column_id_to_chunk_mutex;
  mutable std::map<int, std::unique_ptr<std::condition_variable>>
      column_id_to_chunk_conditional_var;
  mutable std::mutex delete_buffer_mutex;

  IterativeFileScanParameters(std::map<int, Chunk_NS::Chunk>& column_id_to_chunk_map,
                              int32_t fragment_id,
                              AbstractBuffer* delete_buffer)
      : column_id_to_chunk_map(column_id_to_chunk_map)
      , fragment_id(fragment_id)
      , delete_buffer(delete_buffer) {
    for (const auto& [key, _] : column_id_to_chunk_map) {
      column_id_to_chunk_mutex[key] = std::make_unique<std::mutex>();
      column_id_to_chunk_conditional_var[key] =
          std::make_unique<std::condition_variable>();
    }
  }

  std::mutex& getChunkMutex(const int col_id) const {
    auto mutex_it = column_id_to_chunk_mutex.find(col_id);
    CHECK(mutex_it != column_id_to_chunk_mutex.end());
    return *mutex_it->second;
  }

  std::condition_variable& getChunkConditionalVariable(const int col_id) const {
    auto var_it = column_id_to_chunk_conditional_var.find(col_id);
    CHECK(var_it != column_id_to_chunk_conditional_var.end());
    return *var_it->second;
  }
};

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

  bool isLazyFragmentFetchingEnabled() const override { return true; }

  struct ResidualBuffer {
    std::unique_ptr<char[]> residual_data;
    size_t alloc_size;
    size_t residual_buffer_size;
    size_t residual_buffer_alloc_size;
  };

 protected:
  virtual const TextFileBufferParser& getFileBufferParser() const = 0;
  virtual std::optional<size_t> getMaxFileCount() const;

 private:
  AbstractTextFileDataWrapper(const ForeignTable* foreign_table);

  /**
   * Implements an iterative file scan that enables populating chunks fragment
   * by fragment.
   */
  void iterativeFileScan(ChunkMetadataVector& chunk_metadata_vector,
                         IterativeFileScanParameters& file_scan_param);

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

  void updateRolledOffChunks(
      const std::set<std::string>& rolled_off_files,
      const std::map<int32_t, const ColumnDescriptor*>& column_by_id);

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

  bool is_first_file_scan_call_;
  bool is_file_scan_in_progress_;

  // Track the fragment for which the last request currently processed belongs to
  int iterative_scan_last_fragment_id_;

  // These parameters may be reused in a iterative file scan
  MetadataScanMultiThreadingParams multi_threading_params_;
  size_t buffer_size_;
  size_t thread_count_;

  ResidualBuffer residual_buffer_;
};
}  // namespace foreign_storage
