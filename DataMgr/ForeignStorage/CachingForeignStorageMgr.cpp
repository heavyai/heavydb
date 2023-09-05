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

#include "CachingForeignStorageMgr.h"

#include <boost/filesystem.hpp>

#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "ForeignStorageException.h"
#include "ForeignTableSchema.h"
#ifdef ENABLE_IMPORT_PARQUET
#include "ParquetDataWrapper.h"
#endif
#include "Shared/distributed.h"
#include "Shared/misc.h"

namespace foreign_storage {

namespace {
constexpr int64_t MAX_REFRESH_TIME_IN_SECONDS = 60 * 60;
}  // namespace

CachingForeignStorageMgr::CachingForeignStorageMgr(ForeignStorageCache* cache)
    : ForeignStorageMgr(), disk_cache_(cache) {
  CHECK(disk_cache_);
}

void CachingForeignStorageMgr::populateChunkBuffersSafely(
    ForeignDataWrapper& data_wrapper,
    ChunkToBufferMap& required_buffers,
    ChunkToBufferMap& optional_buffers) {
  CHECK_GT(required_buffers.size(), 0U) << "Must populate at least one buffer";
  try {
    ChunkSizeValidator chunk_size_validator(required_buffers.begin()->first);
    data_wrapper.populateChunkBuffers(required_buffers, optional_buffers);
    chunk_size_validator.validateChunkSizes(required_buffers);
    chunk_size_validator.validateChunkSizes(optional_buffers);
    updateFragmenterMetadata(required_buffers);
    updateFragmenterMetadata(optional_buffers);
  } catch (const std::runtime_error& error) {
    // clear any partially loaded but failed chunks (there may be some
    // fully-loaded chunks as well but they will be cleared conservatively
    // anyways)
    for (const auto& [chunk_key, buffer] : required_buffers) {
      if (auto file_buffer = dynamic_cast<File_Namespace::FileBuffer*>(buffer)) {
        file_buffer->freeChunkPages();
      }
    }
    for (const auto& [chunk_key, buffer] : optional_buffers) {
      if (auto file_buffer = dynamic_cast<File_Namespace::FileBuffer*>(buffer)) {
        file_buffer->freeChunkPages();
      }
    }

    throw ForeignStorageException(error.what());
  }
  // All required buffers should be from the same table.
  auto [db, tb] = get_table_prefix(required_buffers.begin()->first);
  disk_cache_->checkpoint(db, tb);
}

namespace {
bool is_in_memory_system_table_chunk_key(const ChunkKey& chunk_key) {
  return get_foreign_table_for_key(chunk_key).is_in_memory_system_table;
}
}  // namespace

void CachingForeignStorageMgr::fetchBuffer(const ChunkKey& chunk_key,
                                           AbstractBuffer* destination_buffer,
                                           const size_t num_bytes) {
  ChunkSizeValidator chunk_size_validator(chunk_key);
  if (is_in_memory_system_table_chunk_key(chunk_key)) {
    ForeignStorageMgr::fetchBuffer(chunk_key, destination_buffer, num_bytes);
    chunk_size_validator.validateChunkSize(destination_buffer);
    return;
  }
  CHECK(destination_buffer);
  CHECK(!destination_buffer->isDirty());

  AbstractBuffer* buffer = disk_cache_->getCachedChunkIfExists(chunk_key);
  if (buffer) {
    chunk_size_validator.validateChunkSize(buffer);
    buffer->copyTo(destination_buffer, num_bytes);
    return;
  } else {
    auto required_size = getRequiredBuffersSize(chunk_key);
    if (required_size > maxFetchSize(chunk_key[CHUNK_KEY_DB_IDX])) {
      // If we don't have space in the cache then skip the caching.
      ForeignStorageMgr::fetchBuffer(chunk_key, destination_buffer, num_bytes);
      return;
    }

    auto column_keys = get_column_key_set(chunk_key);
    // Avoid caching only a subset of the column key set.
    for (const auto& key : column_keys) {
      disk_cache_->eraseChunk(key);
    }

    // Use hints to prefetch other chunks in fragment into cache
    auto& data_wrapper = *getDataWrapper(chunk_key);
    auto optional_set = getOptionalChunkKeySetAndNormalizeCache(
        chunk_key, column_keys, data_wrapper.getCachedParallelismLevel());

    auto optional_buffers = disk_cache_->getChunkBuffersForCaching(optional_set);
    auto required_buffers = disk_cache_->getChunkBuffersForCaching(column_keys);
    CHECK(required_buffers.find(chunk_key) != required_buffers.end());
    populateChunkBuffersSafely(data_wrapper, required_buffers, optional_buffers);

    AbstractBuffer* buffer = required_buffers.at(chunk_key);
    CHECK(buffer);
    buffer->copyTo(destination_buffer, num_bytes);
  }
}

namespace {
class RestoreDataWrapperException : public std::runtime_error {
 public:
  RestoreDataWrapperException(const std::string& error_message)
      : std::runtime_error(error_message) {}
};
}  // namespace

void CachingForeignStorageMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& key_prefix) {
  if (is_in_memory_system_table_chunk_key(key_prefix)) {
    ForeignStorageMgr::getChunkMetadataVecForKeyPrefix(chunk_metadata, key_prefix);
    return;
  }
  CHECK(has_table_prefix(key_prefix));
  // If the disk has any cached metadata for a prefix then it is guaranteed to have all
  // metadata for that table, so we can return a complete set.  If it has no metadata,
  // then it may be that the table has no data, or that it's just not cached, so we need
  // to go to storage to check.
  if (disk_cache_->hasCachedMetadataForKeyPrefix(key_prefix)) {
    disk_cache_->getCachedMetadataVecForKeyPrefix(chunk_metadata, key_prefix);

    // Assert all metadata in cache is mapped to this leaf node in distributed.
    if (is_shardable_key(key_prefix)) {
      for (auto& [key, meta] : chunk_metadata) {
        CHECK(fragment_maps_to_leaf(key)) << show_chunk(key);
      }
    }

    try {
      // If the data in cache was restored from disk then it is possible that the wrapper
      // does not exist yet.  In this case the wrapper will be restored from disk if
      // possible.
      createDataWrapperIfNotExists(key_prefix);
      return;
    } catch (RestoreDataWrapperException& e) {
      auto [db_id, table_id] = get_table_prefix(key_prefix);
      LOG(ERROR) << "An error occurred while attempting to restore data wrapper using "
                    "disk cached metadata. Clearing cached data for table and proceeding "
                    "with a new data wrapper instance. Database ID: "
                 << db_id << ", table ID: " << table_id << ", error: " << e.what();
      chunk_metadata.clear();
      clearTable({db_id, table_id});
    }
  } else if (dist::is_distributed() &&
             disk_cache_->hasStoredDataWrapperMetadata(key_prefix[CHUNK_KEY_DB_IDX],
                                                       key_prefix[CHUNK_KEY_TABLE_IDX])) {
    // In distributed mode, it is possible to have all the chunk metadata filtered out for
    // this node, after previously getting the chunk metadata from the wrapper and caching
    // the wrapper metadata. In this case, return immediately and avoid doing a redundant
    // metadata scan.
    return;
  }

  // If we have no cached data then either the data was evicted, was never populated, or
  // the data for the table is an empty set (no chunks).  In case we are hitting the first
  // two, we should repopulate the data wrapper so just do it in all cases.
  auto table_key = get_table_key(key_prefix);
  eraseDataWrapper(table_key);
  createDataWrapperIfNotExists(table_key);

  getChunkMetadataVecFromDataWrapper(chunk_metadata, key_prefix);
  disk_cache_->cacheMetadataVec(chunk_metadata);
}

void CachingForeignStorageMgr::getChunkMetadataVecFromDataWrapper(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& chunk_key_prefix) {
  CHECK(has_table_prefix(chunk_key_prefix));
  auto [db_id, tb_id] = get_table_prefix(chunk_key_prefix);
  try {
    ForeignStorageMgr::getChunkMetadataVecForKeyPrefix(chunk_metadata, chunk_key_prefix);
  } catch (...) {
    clearTable({db_id, tb_id});
    throw;
  }
  // If the table was disabled then we will have no wrapper to serialize.
  if (is_table_enabled_on_node(chunk_key_prefix)) {
    auto doc = getDataWrapper(chunk_key_prefix)->getSerializedDataWrapper();
    disk_cache_->storeDataWrapper(doc, db_id, tb_id);

    // If the wrapper populated buffers we want that action to be checkpointed.
    disk_cache_->checkpoint(db_id, tb_id);
  }
}

void CachingForeignStorageMgr::refreshTable(const ChunkKey& table_key,
                                            const bool evict_cached_entries) {
  CHECK(is_table_key(table_key));
  ForeignStorageMgr::checkIfS3NeedsToBeEnabled(table_key);
  clearTempChunkBufferMapEntriesForTable(table_key);
  if (evict_cached_entries) {
    clearTable(table_key);
  } else {
    refreshTableInCache(table_key);
  }
}

bool CachingForeignStorageMgr::hasStoredDataWrapper(int32_t db, int32_t tb) const {
  return disk_cache_->hasStoredDataWrapperMetadata(db, tb);
}

void CachingForeignStorageMgr::refreshTableInCache(const ChunkKey& table_key) {
  CHECK(is_table_key(table_key));

  // Preserve the list of which chunks were cached per table to refresh after clear.
  std::vector<ChunkKey> old_chunk_keys =
      disk_cache_->getCachedChunksForKeyPrefix(table_key);

  // Assert all data in cache is mapped to this leaf node in distributed.
  if (is_shardable_key(table_key)) {
    for (auto& key : old_chunk_keys) {
      CHECK(fragment_maps_to_leaf(key)) << show_chunk(key);
    }
  }

  auto append_mode = is_append_table_chunk_key(table_key);

  append_mode ? refreshAppendTableInCache(table_key, old_chunk_keys)
              : refreshNonAppendTableInCache(table_key, old_chunk_keys);
}

void CachingForeignStorageMgr::eraseDataWrapper(const ChunkKey& key) {
  CHECK(is_table_key(key));
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  auto [db, tb] = get_table_prefix(key);
  // Need to erase serialized version on disk if it exists so we don't accidentally
  // recover it after deleting.  It is possible for a cached wrapper file to exist without
  // a wrapper (in multi-instance-mode for instance, so we make sure to remove the file
  // regardless).
  boost::filesystem::remove_all(disk_cache_->getSerializedWrapperPath(db, tb));
  data_wrapper_map_.erase(key);
}

void CachingForeignStorageMgr::clearTable(const ChunkKey& table_key) {
  disk_cache_->clearForTablePrefix(table_key);
  CHECK(!disk_cache_->hasCachedMetadataForKeyPrefix(table_key));
  ForeignStorageMgr::eraseDataWrapper(table_key);
}

int CachingForeignStorageMgr::getHighestCachedFragId(const ChunkKey& table_key) {
  // Determine last fragment ID
  int last_frag_id = 0;
  if (disk_cache_->hasCachedMetadataForKeyPrefix(table_key)) {
    ChunkMetadataVector cached_metadata;
    disk_cache_->getCachedMetadataVecForKeyPrefix(cached_metadata, table_key);
    for (const auto& [key, metadata] : cached_metadata) {
      last_frag_id = std::max(last_frag_id, key[CHUNK_KEY_FRAGMENT_IDX]);
    }
  }
  return last_frag_id;
}

void CachingForeignStorageMgr::refreshAppendTableInCache(
    const ChunkKey& table_key,
    const std::vector<ChunkKey>& old_chunk_keys) {
  CHECK(is_table_key(table_key));
  int last_frag_id = getHighestCachedFragId(table_key);

  ChunkMetadataVector storage_metadata;
  getChunkMetadataVecFromDataWrapper(storage_metadata, table_key);
  try {
    disk_cache_->cacheMetadataVec(storage_metadata);
    refreshChunksInCacheByFragment(old_chunk_keys, last_frag_id);
  } catch (std::runtime_error& e) {
    throw PostEvictionRefreshException(e);
  }
}

void CachingForeignStorageMgr::refreshNonAppendTableInCache(
    const ChunkKey& table_key,
    const std::vector<ChunkKey>& old_chunk_keys) {
  CHECK(is_table_key(table_key));
  ChunkMetadataVector storage_metadata;
  clearTable(table_key);
  getChunkMetadataVecFromDataWrapper(storage_metadata, table_key);

  try {
    disk_cache_->cacheMetadataVec(storage_metadata);
    refreshChunksInCacheByFragment(old_chunk_keys, 0);
  } catch (std::runtime_error& e) {
    throw PostEvictionRefreshException(e);
  }
}

void CachingForeignStorageMgr::refreshChunksInCacheByFragment(
    const std::vector<ChunkKey>& old_chunk_keys,
    int start_frag_id) {
  int64_t total_time{0};
  auto fragment_refresh_start_time = std::chrono::high_resolution_clock::now();

  if (old_chunk_keys.empty()) {
    return;
  }

  // Iterate through previously cached chunks and re-cache them. Caching is
  // done one fragment at a time, for all applicable chunks in the fragment.
  ChunkToBufferMap optional_buffers;
  std::set<ChunkKey> chunk_keys_to_be_cached;
  auto fragment_id = old_chunk_keys[0][CHUNK_KEY_FRAGMENT_IDX];
  const ChunkKey table_key{get_table_key(old_chunk_keys[0])};
  std::set<ChunkKey> chunk_keys_in_fragment;
  for (const auto& chunk_key : old_chunk_keys) {
    CHECK(chunk_key[CHUNK_KEY_TABLE_IDX] == table_key[CHUNK_KEY_TABLE_IDX]);
    if (chunk_key[CHUNK_KEY_FRAGMENT_IDX] < start_frag_id) {
      continue;
    }
    if (disk_cache_->isMetadataCached(chunk_key)) {
      if (chunk_key[CHUNK_KEY_FRAGMENT_IDX] != fragment_id) {
        if (chunk_keys_in_fragment.size() > 0) {
          auto required_buffers =
              disk_cache_->getChunkBuffersForCaching(chunk_keys_in_fragment);
          populateChunkBuffersSafely(
              *getDataWrapper(table_key), required_buffers, optional_buffers);
          chunk_keys_in_fragment.clear();
        }
        // At this point, cache buffers for refreshable chunks in the last fragment
        // have been populated. Exit if the max refresh time has been exceeded.
        // Otherwise, move to the next fragment.
        auto current_time = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::seconds>(
                          current_time - fragment_refresh_start_time)
                          .count();
        if (total_time >= MAX_REFRESH_TIME_IN_SECONDS) {
          LOG(WARNING) << "Refresh time exceeded for table key: { " << table_key[0]
                       << ", " << table_key[1] << " } after fragment id: " << fragment_id;
          break;
        } else {
          fragment_refresh_start_time = std::chrono::high_resolution_clock::now();
        }
        fragment_id = chunk_key[CHUNK_KEY_FRAGMENT_IDX];
      }
      // Key may have been cached during scan
      if (disk_cache_->getCachedChunkIfExists(chunk_key) == nullptr) {
        auto column_keys = get_column_key_set(chunk_key);
        chunk_keys_in_fragment.insert(column_keys.begin(), column_keys.end());
        chunk_keys_to_be_cached.insert(column_keys.begin(), column_keys.end());
      }
    }
  }
  if (chunk_keys_in_fragment.size() > 0) {
    auto required_buffers =
        disk_cache_->getChunkBuffersForCaching(chunk_keys_in_fragment);
    populateChunkBuffersSafely(
        *getDataWrapper(table_key), required_buffers, optional_buffers);
  }
}

bool CachingForeignStorageMgr::createDataWrapperIfNotExists(const ChunkKey& chunk_key) {
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  ChunkKey table_key = get_table_key(chunk_key);
  auto data_wrapper_it = data_wrapper_map_.find(table_key);
  if (data_wrapper_it != data_wrapper_map_.end()) {
    return false;
  }
  auto [db, tb] = get_table_prefix(chunk_key);
  createDataWrapperUnlocked(db, tb);
  auto wrapper_file = disk_cache_->getSerializedWrapperPath(db, tb);
  if (boost::filesystem::exists(wrapper_file)) {
    ChunkMetadataVector chunk_metadata;
    disk_cache_->getCachedMetadataVecForKeyPrefix(chunk_metadata, table_key);
    try {
      auto data_wrapper = shared::get_from_map(data_wrapper_map_, table_key);
      data_wrapper->restoreDataWrapperInternals(
          disk_cache_->getSerializedWrapperPath(db, tb), chunk_metadata);
    } catch (std::exception& e) {
      throw RestoreDataWrapperException(e.what());
    }
  }
  return true;
}

void CachingForeignStorageMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  disk_cache_->clearForTablePrefix({db_id, table_id});
  eraseDataWrapper({db_id, table_id});
  ForeignStorageMgr::removeTableRelatedDS(db_id, table_id);
}

size_t CachingForeignStorageMgr::maxFetchSize(int32_t db_id) const {
  return disk_cache_->getMaxChunkDataSize();
}

bool CachingForeignStorageMgr::hasMaxFetchSize() const {
  return true;
}

size_t CachingForeignStorageMgr::getRequiredBuffersSize(const ChunkKey& chunk_key) const {
  auto key_set = get_column_key_set(chunk_key);
  size_t total_size = 0U;
  for (const auto& key : key_set) {
    total_size += getBufferSize(key);
  }
  return total_size;
}

std::set<ChunkKey> CachingForeignStorageMgr::getOptionalKeysWithinSizeLimit(
    const ChunkKey& chunk_key,
    const std::set<ChunkKey, decltype(set_comp)*>& same_fragment_keys,
    const std::set<ChunkKey, decltype(set_comp)*>& diff_fragment_keys) const {
  std::set<ChunkKey> optional_keys;
  auto total_chunk_size = getRequiredBuffersSize(chunk_key);
  auto max_size = maxFetchSize(chunk_key[CHUNK_KEY_DB_IDX]);
  // Add keys to the list of optional keys starting with the same fragment.  If we run out
  // of space, then exit early with what we have added so far.
  for (const auto& keys : {same_fragment_keys, diff_fragment_keys}) {
    for (const auto& key : keys) {
      auto column_keys = get_column_key_set(key);
      for (const auto& column_key : column_keys) {
        total_chunk_size += getBufferSize(column_key);
      }
      // Early exist if we exceed the size limit.
      if (total_chunk_size > max_size) {
        return optional_keys;
      }
      for (const auto& column_key : column_keys) {
        optional_keys.emplace(column_key);
      }
    }
  }
  return optional_keys;
}

size_t CachingForeignStorageMgr::getBufferSize(const ChunkKey& key) const {
  size_t num_bytes = 0;
  ChunkMetadataVector meta;
  disk_cache_->getCachedMetadataVecForKeyPrefix(meta, get_fragment_key(key));
  CHECK_EQ(meta.size(), 1U) << show_chunk(key);
  auto metadata = meta.begin()->second;

  if (is_varlen_key(key)) {
    if (is_varlen_data_key(key)) {
      num_bytes = get_max_chunk_size(key);
    } else {
      num_bytes = (metadata->sqlType.is_string())
                      ? sizeof(StringOffsetT) * (metadata->numElements + 1)
                      : sizeof(ArrayOffsetT) * (metadata->numElements + 1);
    }
  } else {
    num_bytes = metadata->numBytes;
  }
  return num_bytes;
}

bool CachingForeignStorageMgr::isChunkCached(const ChunkKey& chunk_key) const {
  return disk_cache_->getCachedChunkIfExists(chunk_key) != nullptr;
}

void CachingForeignStorageMgr::evictChunkFromCache(const ChunkKey& chunk_key) {
  disk_cache_->eraseChunk(chunk_key);
}
}  // namespace foreign_storage
