
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

#include <shared_mutex>

#include "DataMgr/AbstractBufferMgr.h"
#include "ForeignDataWrapper.h"
#include "Shared/mapd_shared_mutex.h"

using namespace Data_Namespace;

class PostEvictionRefreshException : public std::runtime_error {
 public:
  PostEvictionRefreshException() = delete;
  PostEvictionRefreshException(const PostEvictionRefreshException&) = delete;
  PostEvictionRefreshException& operator=(const PostEvictionRefreshException&) = delete;
  PostEvictionRefreshException(const std::runtime_error& exception)
      : std::runtime_error(""), original_exception_(exception){};

  std::runtime_error getOriginalException() { return original_exception_; }

 private:
  std::runtime_error original_exception_;
};

namespace foreign_storage {

// For testing purposes only
class MockForeignDataWrapper : public ForeignDataWrapper {
 public:
  virtual void setParentWrapper(
      std::shared_ptr<ForeignDataWrapper> parent_data_wrapper) = 0;
};

class ForeignStorageMgr : public AbstractBufferMgr {
 public:
  ForeignStorageMgr();

  virtual ~ForeignStorageMgr() {}

  AbstractBuffer* createBuffer(const ChunkKey& chunk_key,
                               const size_t page_size,
                               const size_t initial_size) override;
  void deleteBuffer(const ChunkKey& chunk_key, const bool purge) override;
  void deleteBuffersWithPrefix(const ChunkKey& chunk_key_prefix,
                               const bool purge) override;
  AbstractBuffer* getBuffer(const ChunkKey& chunk_key, const size_t num_bytes) override;
  void fetchBuffer(const ChunkKey& chunk_key,
                   AbstractBuffer* destination_buffer,
                   const size_t num_bytes) override;
  AbstractBuffer* putBuffer(const ChunkKey& chunk_key,
                            AbstractBuffer* source_buffer,
                            const size_t num_bytes) override;
  /*
    Obtains chunk-metadata relating to a prefix.  Will create and use new
    datawrappers if none are found for the given prefix.
   */
  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata,
                                       const ChunkKey& chunk_key_prefix) override;
  bool isBufferOnDevice(const ChunkKey& chunk_key) override;
  std::string printSlabs() override;
  size_t getMaxSize() override;
  size_t getInUseSize() override;
  size_t getAllocated() override;
  bool isAllocationCapped() override;
  void checkpoint() override;
  void checkpoint(const int db_id, const int tb_id) override;
  AbstractBuffer* alloc(const size_t num_bytes) override;
  void free(AbstractBuffer* buffer) override;
  MgrType getMgrType() override;
  std::string getStringMgrType() override;
  size_t getNumChunks() override;
  void removeTableRelatedDS(const int db_id, const int table_id) override;
  bool hasDataWrapperForChunk(const ChunkKey& chunk_key);
  virtual bool createDataWrapperIfNotExists(const ChunkKey& chunk_key);

  // For testing, is datawrapper state recovered from disk
  bool isDatawrapperRestored(const ChunkKey& chunk_key);
  void setDataWrapper(const ChunkKey& table_key,
                      std::shared_ptr<MockForeignDataWrapper> data_wrapper);

  virtual void refreshTable(const ChunkKey& table_key, const bool evict_cached_entries);

  void prepareTablesForExecution(const ColumnByIdxRefSet& input_cols,
                                 const CompilationOptions& co,
                                 const ExecutionOptions& eo,
                                 ExecutionPhase phase) override;

  const DictDescriptor* getDictMetadata(int db_id,
                                        int dict_id,
                                        bool load_dict = true) override {
    UNREACHABLE();
  }

  using ParallelismHint = std::pair<int, int>;
  void setParallelismHints(
      const std::map<ChunkKey, std::set<ParallelismHint>>& hints_per_table);

 protected:
  void createDataWrapperUnlocked(int32_t db, int32_t tb);
  void clearDataWrapper(const ChunkKey& table_key);
  std::shared_ptr<ForeignDataWrapper> getDataWrapper(const ChunkKey& chunk_key);
  bool fetchBufferIfTempBufferMapEntryExists(const ChunkKey& chunk_key,
                                             AbstractBuffer* destination_buffer,
                                             const size_t num_bytes);
  ChunkToBufferMap allocateTempBuffersForChunks(const std::set<ChunkKey>& chunk_keys);
  void clearTempChunkBufferMapEntriesForTable(const ChunkKey& table_key);
  void clearTempChunkBufferMapEntriesForTableUnlocked(const ChunkKey& table_key);

  void getOptionalChunkKeySet(
      std::set<ChunkKey>& optional_chunk_keys,
      const ChunkKey& chunk_key,
      const std::set<ChunkKey>& required_chunk_keys,
      const ForeignDataWrapper::ParallelismLevel parallelism_level);

  static void checkIfS3NeedsToBeEnabled(const ChunkKey& chunk_key);

  void clear_system_table_cache(const ColumnByIdxRefSet& input_cols,
                                const CompilationOptions& co);
  void set_parallelism_hints(const ColumnByIdxRefSet& input_cols);
  void prepare_string_dictionaries(const ColumnByIdxRefSet& input_cols);

  std::shared_mutex data_wrapper_mutex_;
  std::map<ChunkKey, std::shared_ptr<ForeignDataWrapper>> data_wrapper_map_;

  // TODO: Remove below map, which is used to temporarily hold chunk buffers,
  // when buffer mgr interface is updated to accept multiple buffers in one call
  std::map<ChunkKey, std::unique_ptr<AbstractBuffer>> temp_chunk_buffer_map_;
  std::shared_mutex temp_chunk_buffer_map_mutex_;

  std::shared_mutex parallelism_hints_mutex_;
  std::map<ChunkKey, std::set<ParallelismHint>> parallelism_hints_per_table_;
};

std::vector<ChunkKey> get_keys_vec_from_table(const ChunkKey& destination_chunk_key);
std::set<ChunkKey> get_keys_set_from_table(const ChunkKey& destination_chunk_key);
}  // namespace foreign_storage
