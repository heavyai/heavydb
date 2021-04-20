
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

#include "ForeignStorageCache.h"
#include "ForeignStorageMgr.h"

namespace foreign_storage {

/*
  A version of the ForeignStorageMgr that incorporates disk caching.
 */
class CachingForeignStorageMgr : public ForeignStorageMgr {
 public:
  CachingForeignStorageMgr(ForeignStorageCache* cache);

  void fetchBuffer(const ChunkKey& chunk_key,
                   AbstractBuffer* destination_buffer,
                   const size_t num_bytes) override;
  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata,
                                       const ChunkKey& chunk_key_prefix) override;
  void refreshTable(const ChunkKey& table_key, const bool evict_cached_entries) override;

 private:
  void recoverDataWrapperFromDisk(const ChunkKey& table_key,
                                  const ChunkMetadataVector& chunk_metadata);
  void refreshTableInCache(const ChunkKey& table_key);
  int getHighestCachedFragId(const ChunkKey& table_key);
  void refreshAppendTableInCache(const ChunkKey& table_key,
                                 const std::vector<ChunkKey>& old_chunk_keys);
  void refreshNonAppendTableInCache(const ChunkKey& table_key,
                                    const std::vector<ChunkKey>& old_chunk_keys);
  void refreshChunksInCacheByFragment(const std::vector<ChunkKey>& old_chunk_keys,
                                      int last_frag_id);
  void createOrRecoverDataWrapperIfNotExists(const ChunkKey& chunk_key);
  void populateChunkBuffersSafely(ForeignDataWrapper& data_wrapper,
                                  ChunkToBufferMap& required_buffers,
                                  ChunkToBufferMap& optional_buffers);
  void clearTable(const ChunkKey& table_key);
  ForeignStorageCache* disk_cache_;
};

}  // namespace foreign_storage
