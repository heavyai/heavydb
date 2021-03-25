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

#include "CachingForeignStorageMgr.h"
#include "Catalog/ForeignTable.h"
#include "CsvDataWrapper.h"
#include "ForeignStorageException.h"
#include "ForeignTableSchema.h"
#include "ParquetDataWrapper.h"

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
  try {
    data_wrapper.populateChunkBuffers(required_buffers, optional_buffers);
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
}

void CachingForeignStorageMgr::fetchBuffer(const ChunkKey& chunk_key,
                                           AbstractBuffer* destination_buffer,
                                           const size_t num_bytes) {
  CHECK(destination_buffer);
  CHECK(!destination_buffer->isDirty());

  createOrRecoverDataWrapperIfNotExists(chunk_key);

  // TODO: Populate optional buffers as part of CSV performance improvement
  std::vector<ChunkKey> chunk_keys = get_keys_vec_from_table(chunk_key);
  std::vector<ChunkKey> optional_keys;
  ChunkToBufferMap optional_buffers;

  // Use hints to prefetch other chunks in fragment into cache
  auto& data_wrapper = *getDataWrapper(chunk_key);
  std::set<ChunkKey> optional_set;
  getOptionalChunkKeySet(optional_set,
                         chunk_key,
                         get_keys_set_from_table(chunk_key),
                         data_wrapper.getCachedParallelismLevel());
  for (const auto& key : optional_set) {
    if (disk_cache_->getCachedChunkIfExists(key) == nullptr) {
      optional_keys.emplace_back(key);
    }
  }

  if (optional_keys.size()) {
    optional_buffers = disk_cache_->getChunkBuffersForCaching(optional_keys);
  }

  ChunkToBufferMap required_buffers = disk_cache_->getChunkBuffersForCaching(chunk_keys);
  CHECK(required_buffers.find(chunk_key) != required_buffers.end());
  populateChunkBuffersSafely(data_wrapper, required_buffers, optional_buffers);
  disk_cache_->cacheTableChunks(chunk_keys);
  if (optional_keys.size()) {
    disk_cache_->cacheTableChunks(optional_keys);
  }

  AbstractBuffer* buffer = required_buffers.at(chunk_key);
  CHECK(buffer);

  buffer->copyTo(destination_buffer, num_bytes);
}

void CachingForeignStorageMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& keyPrefix) {
  auto [db_id, tb_id] = get_table_prefix(keyPrefix);
  ForeignStorageMgr::getChunkMetadataVecForKeyPrefix(chunk_metadata, keyPrefix);
  getDataWrapper(keyPrefix)->serializeDataWrapperInternals(
      disk_cache_->getCacheDirectoryForTable(db_id, tb_id) + wrapper_file_name);
}

void CachingForeignStorageMgr::recoverDataWrapperFromDisk(
    const ChunkKey& table_key,
    const ChunkMetadataVector& chunk_metadata) {
  auto [db_id, tb_id] = get_table_prefix(table_key);
  getDataWrapper(table_key)->restoreDataWrapperInternals(
      disk_cache_->getCacheDirectoryForTable(db_id, tb_id) + wrapper_file_name,
      chunk_metadata);
}

void CachingForeignStorageMgr::refreshTable(const ChunkKey& table_key,
                                            const bool evict_cached_entries) {
  CHECK(is_table_key(table_key));
  ForeignStorageMgr::checkIfS3NeedsToBeEnabled(table_key);
  clearTempChunkBufferMapEntriesForTable(table_key);
  evict_cached_entries ? disk_cache_->clearForTablePrefix(table_key)
                       : refreshTableInCache(table_key);
}

void CachingForeignStorageMgr::refreshTableInCache(const ChunkKey& table_key) {
  CHECK(is_table_key(table_key));

  // Before we can refresh a table we should make sure it has recovered any data
  // if the table has been unused since the last server restart.
  if (!disk_cache_->hasCachedMetadataForKeyPrefix(table_key)) {
    ChunkMetadataVector old_cached_metadata;
    disk_cache_->recoverCacheForTable(old_cached_metadata, table_key);
  }

  // Preserve the list of which chunks were cached per table to refresh after clear.
  std::vector<ChunkKey> old_chunk_keys =
      disk_cache_->getCachedChunksForKeyPrefix(table_key);
  auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(table_key[CHUNK_KEY_DB_IDX]);
  CHECK(catalog);
  bool append_mode =
      catalog->getForeignTableUnlocked(table_key[CHUNK_KEY_TABLE_IDX])->isAppendMode();

  append_mode ? refreshAppendTableInCache(table_key, old_chunk_keys)
              : refreshNonAppendTableInCache(table_key, old_chunk_keys);
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
  createOrRecoverDataWrapperIfNotExists(table_key);
  int last_frag_id = getHighestCachedFragId(table_key);

  ChunkMetadataVector storage_metadata;
  getChunkMetadataVecForKeyPrefix(storage_metadata, table_key);
  try {
    disk_cache_->cacheMetadataWithFragIdGreaterOrEqualTo(storage_metadata, last_frag_id);
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
  disk_cache_->clearForTablePrefix(table_key);
  getChunkMetadataVecForKeyPrefix(storage_metadata, table_key);

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
  std::vector<ChunkKey> chunk_keys_to_be_cached;
  auto fragment_id = old_chunk_keys[0][CHUNK_KEY_FRAGMENT_IDX];
  const ChunkKey table_key{get_table_key(old_chunk_keys[0])};
  std::vector<ChunkKey> chunk_keys_in_fragment;
  for (const auto& chunk_key : old_chunk_keys) {
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
        if (is_varlen_key(chunk_key)) {
          CHECK(is_varlen_data_key(chunk_key));
          ChunkKey index_chunk_key{chunk_key[CHUNK_KEY_DB_IDX],
                                   chunk_key[CHUNK_KEY_TABLE_IDX],
                                   chunk_key[CHUNK_KEY_COLUMN_IDX],
                                   chunk_key[CHUNK_KEY_FRAGMENT_IDX],
                                   2};
          chunk_keys_in_fragment.emplace_back(index_chunk_key);
          chunk_keys_to_be_cached.emplace_back(index_chunk_key);
        }
        chunk_keys_in_fragment.emplace_back(chunk_key);
        chunk_keys_to_be_cached.emplace_back(chunk_key);
      }
    }
  }
  if (chunk_keys_in_fragment.size() > 0) {
    auto required_buffers =
        disk_cache_->getChunkBuffersForCaching(chunk_keys_in_fragment);
    populateChunkBuffersSafely(
        *getDataWrapper(table_key), required_buffers, optional_buffers);
  }
  if (chunk_keys_to_be_cached.size() > 0) {
    disk_cache_->cacheTableChunks(chunk_keys_to_be_cached);
  }
}

void CachingForeignStorageMgr::createOrRecoverDataWrapperIfNotExists(
    const ChunkKey& chunk_key) {
  ChunkKey table_key = get_table_key(chunk_key);
  if (createDataWrapperIfNotExists(table_key)) {
    ChunkMetadataVector chunk_metadata;
    if (disk_cache_->hasCachedMetadataForKeyPrefix(table_key)) {
      disk_cache_->getCachedMetadataVecForKeyPrefix(chunk_metadata, table_key);
      recoverDataWrapperFromDisk(table_key, chunk_metadata);
    } else {
      getDataWrapper(table_key)->populateChunkMetadata(chunk_metadata);
    }
  }
}

}  // namespace foreign_storage
