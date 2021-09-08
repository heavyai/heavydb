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

#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "GlobalFileMgr.h"

namespace File_Namespace {
/*
  A GlobalFileMgr with additional functionality for caching mutable tables to disk.
 */
class CachingGlobalFileMgr : public GlobalFileMgr {
 public:
  CachingGlobalFileMgr(int32_t device_id,
                       std::shared_ptr<ForeignStorageInterface> fsi,
                       const std::string& base_path,
                       size_t num_reader_threads,
                       foreign_storage::ForeignStorageCache* disk_cache,
                       size_t defaultPageSize = DEFAULT_PAGE_SIZE);

  AbstractBuffer* createBuffer(const ChunkKey& chunk_key,
                               const size_t page_size,
                               const size_t initial_size) override;

  void deleteBuffer(const ChunkKey& chunk_key, const bool purge) override;

  void deleteBuffersWithPrefix(const ChunkKey& chunk_key_prefix,
                               const bool purge) override;

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata,
                                       const ChunkKey& keyPrefix) override;

  void fetchBuffer(const ChunkKey& chunk_key,
                   AbstractBuffer* destination_buffer,
                   const size_t num_bytes) override;

  AbstractBuffer* putBuffer(const ChunkKey& chunk_key,
                            AbstractBuffer* source_buffer,
                            const size_t num_bytes) override;

  void checkpoint() override;

  void checkpoint(const int db_id, const int tb_id) override;

  void removeTableRelatedDS(const int db_id, const int table_id) override;

 private:
  bool isChunkPrefixCacheable(const ChunkKey& chunk_prefix) const;

  foreign_storage::ForeignStorageCache* disk_cache_;
  std::set<ChunkKey> cached_chunk_keys_;
};
}  // namespace File_Namespace
