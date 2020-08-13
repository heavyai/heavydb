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
#include "ForeignStorageCache.h"
#include "Shared/mapd_shared_mutex.h"

using namespace Data_Namespace;

extern size_t foreign_cache_entry_limit;

namespace foreign_storage {
class ForeignStorageMgr : public AbstractBufferMgr {
 public:
  ForeignStorageMgr(File_Namespace::GlobalFileMgr* global_file_mgr);

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
    Obtains (and caches) chunk-metadata for all existing data wrappers, but will not
    create new ones.
   */
  void getChunkMetadataVec(ChunkMetadataVector& chunk_metadata) override;
  /*
    Obtains and caches chunk-metadata relating to a prefix.  Will create and use new
    datawrappers if none are found for the given prefix.
   */
  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata,
                                       const ChunkKey& chunk_key_prefix) override;
  bool isBufferOnDevice(const ChunkKey& chunk_key) override;
  std::string printSlabs() override;
  void clearSlabs() override;
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
  ForeignStorageCache* getForeignStorageCache() const;
  void refreshTablesInCache(const std::vector<ChunkKey>& table_keys);
  void evictTablesFromCache(const std::vector<ChunkKey>& table_keys);

 private:
  void createDataWrapperIfNotExists(const ChunkKey& chunk_key);
  std::shared_ptr<ForeignDataWrapper> getDataWrapper(const ChunkKey& chunk_key);

  std::shared_mutex data_wrapper_mutex_;

  std::map<ChunkKey, std::shared_ptr<ForeignDataWrapper>> data_wrapper_map_;
  std::unique_ptr<ForeignStorageCache> foreign_storage_cache_;
};
}  // namespace foreign_storage
