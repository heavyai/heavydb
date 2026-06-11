/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

  void removeCachedData(const int db_id, const int table_id);

 private:
  bool isChunkPrefixCacheable(const ChunkKey& chunk_prefix) const;

  foreign_storage::ForeignStorageCache* disk_cache_;
  std::set<ChunkKey> cached_chunk_keys_;

  mutable heavyai::shared_mutex cached_chunk_keys_mutex_;
};
}  // namespace File_Namespace
