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

#include "CachingGlobalFileMgr.h"

#include "DataMgr/ForeignStorage/ForeignStorageBuffer.h"
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"

namespace File_Namespace {
CachingGlobalFileMgr::CachingGlobalFileMgr(
    int32_t device_id,
    std::shared_ptr<ForeignStorageInterface> fsi,
    const std::string& base_path,
    size_t num_reader_threads,
    foreign_storage::ForeignStorageCache* disk_cache,
    size_t default_page_size)
    : GlobalFileMgr(device_id, fsi, base_path, num_reader_threads, default_page_size)
    , disk_cache_(disk_cache) {
  CHECK(disk_cache_);
}

AbstractBuffer* CachingGlobalFileMgr::createBuffer(const ChunkKey& chunk_key,
                                                   const size_t page_size,
                                                   const size_t initial_size) {
  auto buf = GlobalFileMgr::createBuffer(chunk_key, page_size, initial_size);
  if (isChunkPrefixCacheable(chunk_key)) {
    cached_chunk_keys_.emplace(chunk_key);
  }
  return buf;
}

void CachingGlobalFileMgr::deleteBuffer(const ChunkKey& chunk_key, const bool purge) {
  if (isChunkPrefixCacheable(chunk_key)) {
    disk_cache_->deleteBufferIfExists(chunk_key);
    cached_chunk_keys_.erase(chunk_key);
  }
  GlobalFileMgr::deleteBuffer(chunk_key, purge);
}

void CachingGlobalFileMgr::deleteBuffersWithPrefix(const ChunkKey& chunk_key_prefix,
                                                   const bool purge) {
  if (isChunkPrefixCacheable(chunk_key_prefix)) {
    CHECK(has_table_prefix(chunk_key_prefix));
    disk_cache_->clearForTablePrefix(get_table_key(chunk_key_prefix));

    ChunkKey upper_prefix(chunk_key_prefix);
    upper_prefix.push_back(std::numeric_limits<int>::max());
    auto end_it =
        cached_chunk_keys_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
    for (auto&& chunk_key_it = cached_chunk_keys_.lower_bound(chunk_key_prefix);
         chunk_key_it != end_it;) {
      chunk_key_it = cached_chunk_keys_.erase(chunk_key_it);
    }
  }
  GlobalFileMgr::deleteBuffersWithPrefix(chunk_key_prefix, purge);
}

void CachingGlobalFileMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& key_prefix) {
  CHECK(has_table_prefix(key_prefix));
  if (isChunkPrefixCacheable(key_prefix)) {
    // If the disk has any cached metadata for a prefix then it is guaranteed to have all
    // metadata for that table, so we can return a complete set.  If it has no metadata,
    // then it may be that the table has no data, or that it's just not cached, so we need
    // to go to storage to check.
    if (disk_cache_->hasCachedMetadataForKeyPrefix(key_prefix)) {
      disk_cache_->getCachedMetadataVecForKeyPrefix(chunk_metadata, key_prefix);
      return;
    }
  }
  GlobalFileMgr::getChunkMetadataVecForKeyPrefix(chunk_metadata, key_prefix);
  if (isChunkPrefixCacheable(key_prefix)) {
    disk_cache_->cacheMetadataVec(chunk_metadata);
  }
}

void CachingGlobalFileMgr::fetchBuffer(const ChunkKey& chunk_key,
                                       AbstractBuffer* destination_buffer,
                                       const size_t num_bytes) {
  if (isChunkPrefixCacheable(chunk_key)) {
    // If we are recovering after a shutdown, it is possible for there to be cached data
    // without the file_mgr being initialized, so we need to check if the file_mgr exists.
    CHECK(has_table_prefix(chunk_key));
    auto [db, table_id] = get_table_prefix(chunk_key);
    auto file_mgr = GlobalFileMgr::findFileMgr(db, table_id);
    if (file_mgr && file_mgr->getBuffer(chunk_key)->isDirty()) {
      // It is possible for the fragmenter to write data to a FileBuffer and then attempt
      // to fetch that bufer without checkpointing. In that case the cache will not have
      // been updated and the cached buffer will be out of date, so we need to fetch the
      // storage buffer.
      GlobalFileMgr::fetchBuffer(chunk_key, destination_buffer, num_bytes);
    } else {
      AbstractBuffer* buffer = disk_cache_->getCachedChunkIfExists(chunk_key);
      if (buffer) {
        buffer->copyTo(destination_buffer, num_bytes);
      } else {
        GlobalFileMgr::fetchBuffer(chunk_key, destination_buffer, num_bytes);
        disk_cache_->putBuffer(chunk_key, destination_buffer, num_bytes);
      }
    }
  } else {
    GlobalFileMgr::fetchBuffer(chunk_key, destination_buffer, num_bytes);
  }
}

AbstractBuffer* CachingGlobalFileMgr::putBuffer(const ChunkKey& chunk_key,
                                                AbstractBuffer* source_buffer,
                                                const size_t num_bytes) {
  auto buf = GlobalFileMgr::putBuffer(chunk_key, source_buffer, num_bytes);
  if (isChunkPrefixCacheable(chunk_key)) {
    disk_cache_->putBuffer(chunk_key, source_buffer, num_bytes);
  }
  return buf;
}

void CachingGlobalFileMgr::checkpoint() {
  std::set<File_Namespace::TablePair> tables_to_checkpoint;
  for (auto& key : cached_chunk_keys_) {
    if (isChunkPrefixCacheable(key) && GlobalFileMgr::getBuffer(key)->isDirty()) {
      tables_to_checkpoint.emplace(get_table_prefix(key));
      foreign_storage::ForeignStorageBuffer temp_buf;
      GlobalFileMgr::fetchBuffer(key, &temp_buf, 0);
      disk_cache_->putBuffer(key, &temp_buf);
    }
  }
  for (auto [db, tb] : tables_to_checkpoint) {
    disk_cache_->checkpoint(db, tb);
  }
  GlobalFileMgr::checkpoint();
}

void CachingGlobalFileMgr::checkpoint(const int db_id, const int tb_id) {
  if (isChunkPrefixCacheable({db_id, tb_id})) {
    bool need_checkpoint{false};
    ChunkKey chunk_prefix{db_id, tb_id};
    ChunkKey upper_prefix(chunk_prefix);
    upper_prefix.push_back(std::numeric_limits<int>::max());
    auto end_it =
        cached_chunk_keys_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
    for (auto&& chunk_key_it = cached_chunk_keys_.lower_bound(chunk_prefix);
         chunk_key_it != end_it;
         ++chunk_key_it) {
      if (GlobalFileMgr::getBuffer(*chunk_key_it)->isDirty()) {
        need_checkpoint = true;
        foreign_storage::ForeignStorageBuffer temp_buf;
        GlobalFileMgr::fetchBuffer(*chunk_key_it, &temp_buf, 0);
        disk_cache_->putBuffer(*chunk_key_it, &temp_buf);
      }
    }
    if (need_checkpoint) {
      disk_cache_->checkpoint(db_id, tb_id);
    }
  }
  GlobalFileMgr::checkpoint(db_id, tb_id);
}

void CachingGlobalFileMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  if (isChunkPrefixCacheable({db_id, table_id})) {
    const ChunkKey table_key{db_id, table_id};
    disk_cache_->clearForTablePrefix(table_key);
    ChunkKey upper_prefix(table_key);
    upper_prefix.push_back(std::numeric_limits<int>::max());
    auto end_it =
        cached_chunk_keys_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
    for (auto&& chunk_key_it = cached_chunk_keys_.lower_bound(table_key);
         chunk_key_it != end_it;) {
      chunk_key_it = cached_chunk_keys_.erase(chunk_key_it);
    }
  }
  GlobalFileMgr::removeTableRelatedDS(db_id, table_id);
}

bool CachingGlobalFileMgr::isChunkPrefixCacheable(const ChunkKey& chunk_prefix) const {
  CHECK(has_table_prefix(chunk_prefix));
  // If this is an Arrow FSI table then we can't cache it.
  if (fsi_->lookupBufferManager(chunk_prefix[CHUNK_KEY_DB_IDX],
                                chunk_prefix[CHUNK_KEY_TABLE_IDX])) {
    return false;
  }
  return true;
}
}  // namespace File_Namespace
