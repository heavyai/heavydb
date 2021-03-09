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

#include "MutableCachePersistentStorageMgr.h"

MutableCachePersistentStorageMgr::MutableCachePersistentStorageMgr(
    const std::string& data_dir,
    const size_t num_reader_threads,
    const DiskCacheConfig& disk_cache_config)
    : PersistentStorageMgr(data_dir, num_reader_threads, disk_cache_config) {
  CHECK(disk_cache_);
  CHECK(disk_cache_config_.isEnabledForMutableTables());
}

AbstractBuffer* MutableCachePersistentStorageMgr::createBuffer(
    const ChunkKey& chunk_key,
    const size_t page_size,
    const size_t initial_size) {
  auto buf = PersistentStorageMgr::createBuffer(chunk_key, page_size, initial_size);
  if (isChunkPrefixCacheable(chunk_key)) {
    cached_chunk_keys_.emplace(chunk_key);
  }
  return buf;
}

void MutableCachePersistentStorageMgr::deleteBuffer(const ChunkKey& chunk_key,
                                                    const bool purge) {
  // No need to delete for FSI-only cache as Foreign Tables are immutable and we should
  // not be deleting buffers for them.
  CHECK(!isForeignStorage(chunk_key));
  disk_cache_->deleteBufferIfExists(chunk_key);
  cached_chunk_keys_.erase(chunk_key);
  PersistentStorageMgr::deleteBuffer(chunk_key, purge);
}

void MutableCachePersistentStorageMgr::deleteBuffersWithPrefix(
    const ChunkKey& chunk_key_prefix,
    const bool purge) {
  CHECK(has_table_prefix(chunk_key_prefix));
  disk_cache_->clearForTablePrefix(get_table_key(chunk_key_prefix));

  ChunkKey upper_prefix(chunk_key_prefix);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  auto end_it = cached_chunk_keys_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
  for (auto&& chunk_key_it = cached_chunk_keys_.lower_bound(chunk_key_prefix);
       chunk_key_it != end_it;) {
    chunk_key_it = cached_chunk_keys_.erase(chunk_key_it);
  }
  PersistentStorageMgr::deleteBuffersWithPrefix(chunk_key_prefix, purge);
}

AbstractBuffer* MutableCachePersistentStorageMgr::putBuffer(const ChunkKey& chunk_key,
                                                            AbstractBuffer* source_buffer,
                                                            const size_t num_bytes) {
  auto buf = PersistentStorageMgr::putBuffer(chunk_key, source_buffer, num_bytes);
  disk_cache_->cacheChunk(chunk_key, source_buffer);
  return buf;
}

void MutableCachePersistentStorageMgr::checkpoint() {
  for (auto& key : cached_chunk_keys_) {
    if (global_file_mgr_->getBuffer(key)->isDirty()) {
      foreign_storage::ForeignStorageBuffer temp_buf;
      global_file_mgr_->fetchBuffer(key, &temp_buf, 0);
      disk_cache_->cacheChunk(key, &temp_buf);
    }
  }
  PersistentStorageMgr::global_file_mgr_->checkpoint();
}

void MutableCachePersistentStorageMgr::checkpoint(const int db_id, const int tb_id) {
  ChunkKey chunk_prefix{db_id, tb_id};
  ChunkKey upper_prefix(chunk_prefix);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  auto end_it = cached_chunk_keys_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
  for (auto&& chunk_key_it = cached_chunk_keys_.lower_bound(chunk_prefix);
       chunk_key_it != end_it;
       ++chunk_key_it) {
    if (global_file_mgr_->getBuffer(*chunk_key_it)->isDirty()) {
      foreign_storage::ForeignStorageBuffer temp_buf;
      global_file_mgr_->fetchBuffer(*chunk_key_it, &temp_buf, 0);
      disk_cache_->cacheChunk(*chunk_key_it, &temp_buf);
    }
  }
  PersistentStorageMgr::global_file_mgr_->checkpoint(db_id, tb_id);
}

void MutableCachePersistentStorageMgr::removeTableRelatedDS(const int db_id,
                                                            const int table_id) {
  PersistentStorageMgr::removeTableRelatedDS(db_id, table_id);
  const ChunkKey table_key{db_id, table_id};
  ChunkKey upper_prefix(table_key);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  auto end_it = cached_chunk_keys_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
  for (auto&& chunk_key_it = cached_chunk_keys_.lower_bound(table_key);
       chunk_key_it != end_it;) {
    chunk_key_it = cached_chunk_keys_.erase(chunk_key_it);
  }
}
