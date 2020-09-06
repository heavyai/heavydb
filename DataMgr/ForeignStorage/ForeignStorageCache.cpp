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

/*
  TODO(Misiu): A lot of methods here can be made asyncronous.  It may be worth an
  investigation to determine if it's worth adding async versions of them for performance
  reasons.
*/

#include "ForeignStorageCache.h"
#include "Shared/measure.h"

namespace foreign_storage {
using read_lock = mapd_shared_lock<mapd_shared_mutex>;
using write_lock = mapd_unique_lock<mapd_shared_mutex>;

template <typename Func, typename T>
static void iterateOverMatchingPrefix(Func func,
                                      T& chunk_collection,
                                      const ChunkKey& chunk_prefix) {
  ChunkKey upper_prefix(chunk_prefix);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  auto end_it = chunk_collection.upper_bound(static_cast<const ChunkKey>(upper_prefix));
  for (auto chunk_it = chunk_collection.lower_bound(chunk_prefix); chunk_it != end_it;
       ++chunk_it) {
    func(*chunk_it);
  }
}

ForeignStorageCache::ForeignStorageCache(const std::string& cache_dir,
                                         const size_t num_reader_threads,
                                         const size_t limit)
    : entry_limit_(limit) {
  validatePath(cache_dir);
  global_file_mgr_ =
      std::make_unique<File_Namespace::GlobalFileMgr>(0, cache_dir, num_reader_threads);
}

void ForeignStorageCache::cacheTableChunks(const std::vector<ChunkKey>& chunk_keys) {
  auto timer = DEBUG_TIMER(__func__);
  write_lock lock(chunks_mutex_);
  CHECK(!chunk_keys.empty());
  auto db_id = chunk_keys[0][CHUNK_KEY_DB_IDX];
  auto table_id = chunk_keys[0][CHUNK_KEY_TABLE_IDX];
  for (const auto& chunk_key : chunk_keys) {
    CHECK_EQ(db_id, chunk_key[CHUNK_KEY_DB_IDX]);
    CHECK_EQ(table_id, chunk_key[CHUNK_KEY_TABLE_IDX]);
    CHECK(global_file_mgr_->isBufferOnDevice(chunk_key));

    if (isCacheFull()) {
      evictChunkByAlg();
    }
    eviction_alg_->touchChunk(chunk_key);
    cached_chunks_.emplace(chunk_key);
  }
  global_file_mgr_->checkpoint(db_id, table_id);
}

AbstractBuffer* ForeignStorageCache::getCachedChunkIfExists(const ChunkKey& chunk_key) {
  auto timer = DEBUG_TIMER(__func__);
  {
    read_lock lock(chunks_mutex_);
    if (cached_chunks_.find(chunk_key) == cached_chunks_.end()) {
      return nullptr;
    }
  }
  write_lock lock(chunks_mutex_);
  eviction_alg_->touchChunk(chunk_key);
  return global_file_mgr_->getBuffer(chunk_key);
}

bool ForeignStorageCache::isMetadataCached(const ChunkKey& chunk_key) {
  auto timer = DEBUG_TIMER(__func__);
  read_lock lock(metadata_mutex_);
  return (cached_metadata_.find(chunk_key) != cached_metadata_.end());
}

bool ForeignStorageCache::recoverCacheForTable(ChunkMetadataVector& meta_vec,
                                               const ChunkKey& table_key) {
  CHECK(meta_vec.size() == 0);
  CHECK(isTableKey(table_key));
  global_file_mgr_->getChunkMetadataVecForKeyPrefix(meta_vec, table_key);
  for (auto& [chunk_key, metadata] : meta_vec) {
    cached_metadata_.emplace(chunk_key);
    // If a filebuffer has no pages, then it only has cached metadata and no cached chunk.
    if (global_file_mgr_->getBuffer(chunk_key)->pageCount() > 0) {
      cached_chunks_.emplace(chunk_key);
    }
  }
  return (meta_vec.size() > 0);
}

static void setMetadataForBuffer(AbstractBuffer* buffer, ChunkMetadata* meta) {
  buffer->initEncoder(meta->sqlType);
  buffer->setSize(meta->numBytes);
  buffer->encoder->setNumElems(meta->numElements);
  buffer->encoder->resetChunkStats(meta->chunkStats);
  buffer->setUpdated();
}

void ForeignStorageCache::evictThenEraseChunk(const ChunkKey& chunk_key) {
  eviction_alg_->removeChunk(chunk_key);
  eraseChunk(chunk_key);
}

void ForeignStorageCache::cacheMetadataVec(const ChunkMetadataVector& metadata_vec) {
  auto timer = DEBUG_TIMER(__func__);
  write_lock meta_lock(metadata_mutex_);
  write_lock chunk_lock(chunks_mutex_);
  for (auto& [chunk_key, metadata] : metadata_vec) {
    cached_metadata_.emplace(chunk_key);
    AbstractBuffer* buf;
    AbstractBuffer* index_buffer = nullptr;
    ChunkKey index_chunk_key;
    if (isVarLenKey(chunk_key)) {
      // For variable length chunks, metadata is associated with the data chunk
      CHECK(isVarLenDataKey(chunk_key));
      index_chunk_key = {chunk_key[CHUNK_KEY_DB_IDX],
                         chunk_key[CHUNK_KEY_TABLE_IDX],
                         chunk_key[CHUNK_KEY_COLUMN_IDX],
                         chunk_key[CHUNK_KEY_FRAGMENT_IDX],
                         2};
    }

    if (!global_file_mgr_->isBufferOnDevice(chunk_key)) {
      buf = global_file_mgr_->createBuffer(chunk_key);

      if (!index_chunk_key.empty()) {
        CHECK(!global_file_mgr_->isBufferOnDevice(index_chunk_key));
        index_buffer = global_file_mgr_->createBuffer(index_chunk_key);
        CHECK(index_buffer);
      }
    } else {
      buf = global_file_mgr_->getBuffer(chunk_key);

      if (!index_chunk_key.empty()) {
        CHECK(global_file_mgr_->isBufferOnDevice(index_chunk_key));
        index_buffer = global_file_mgr_->getBuffer(chunk_key);
        CHECK(index_buffer);
      }
    }

    setMetadataForBuffer(buf, metadata.get());
    evictThenEraseChunk(chunk_key);

    if (!index_chunk_key.empty()) {
      CHECK(index_buffer);
      index_buffer->isUpdated();
      evictThenEraseChunk(index_chunk_key);
    }
  }
  global_file_mgr_->checkpoint();
}

void ForeignStorageCache::getCachedMetadataVecForKeyPrefix(
    ChunkMetadataVector& metadata_vec,
    const ChunkKey& chunk_prefix) {
  auto timer = DEBUG_TIMER(__func__);
  read_lock r_lock(metadata_mutex_);
  iterateOverMatchingPrefix(
      [&metadata_vec, this](auto chunk) {
        std::shared_ptr<ChunkMetadata> buf_metadata = std::make_shared<ChunkMetadata>();
        global_file_mgr_->getBuffer(chunk)->encoder->getMetadata(buf_metadata);
        metadata_vec.push_back(std::make_pair(chunk, buf_metadata));
      },
      cached_metadata_,
      chunk_prefix);
}

bool ForeignStorageCache::hasCachedMetadataForKeyPrefix(const ChunkKey& chunk_prefix) {
  auto timer = DEBUG_TIMER(__func__);
  read_lock lock(metadata_mutex_);
  // We don't use iterateOvermatchingPrefix() here because we want to exit early if
  // possible.
  ChunkKey upper_prefix(chunk_prefix);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  auto end_it = cached_metadata_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
  for (auto meta_it = cached_metadata_.lower_bound(chunk_prefix); meta_it != end_it;
       ++meta_it) {
    return true;
  }
  return false;
}

void ForeignStorageCache::clearForTablePrefix(const ChunkKey& chunk_prefix) {
  CHECK(isTableKey(chunk_prefix));
  auto timer = DEBUG_TIMER(__func__);
  ChunkKey upper_prefix(chunk_prefix);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  {
    write_lock w_lock(chunks_mutex_);
    // Delete chunks for prefix
    // We won't delete the buffers here because metadata delete will do that for us later.
    auto end_it = cached_chunks_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
    for (auto chunk_it = cached_chunks_.lower_bound(chunk_prefix); chunk_it != end_it;) {
      eviction_alg_->removeChunk(*chunk_it);
      chunk_it = cached_chunks_.erase(chunk_it);
    }
  }
  {
    write_lock w_lock(metadata_mutex_);
    // Delete metadata for prefix
    auto end_it = cached_metadata_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
    for (auto meta_it = cached_metadata_.lower_bound(chunk_prefix); meta_it != end_it;) {
      meta_it = cached_metadata_.erase(meta_it);
    }
  }
  global_file_mgr_->removeTableRelatedDS(chunk_prefix[0], chunk_prefix[1]);
}

void ForeignStorageCache::clear() {
  auto timer = DEBUG_TIMER(__func__);
  std::set<ChunkKey> table_keys;
  {
    write_lock w_lock(chunks_mutex_);
    for (auto chunk_it = cached_chunks_.begin(); chunk_it != cached_chunks_.end();) {
      eviction_alg_->removeChunk(*chunk_it);
      chunk_it = cached_chunks_.erase(chunk_it);
    }
  }
  {
    write_lock w_lock(metadata_mutex_);
    for (auto meta_it = cached_metadata_.begin(); meta_it != cached_metadata_.end();) {
      table_keys.emplace(ChunkKey{(*meta_it)[0], (*meta_it)[1]});
      meta_it = cached_metadata_.erase(meta_it);
    }
  }
  for (const auto& table_key : table_keys) {
    global_file_mgr_->removeTableRelatedDS(table_key[0], table_key[1]);
  }
}

void ForeignStorageCache::setLimit(size_t limit) {
  auto timer = DEBUG_TIMER(__func__);
  write_lock w_lock(chunks_mutex_);
  entry_limit_ = limit;
  // Need to make sure cache doesn't have more entries than the limit
  // in case the limit was lowered.
  while (cached_chunks_.size() > entry_limit_) {
    evictChunkByAlg();
  }
  global_file_mgr_->checkpoint();
}

std::vector<ChunkKey> ForeignStorageCache::getCachedChunksForKeyPrefix(
    const ChunkKey& chunk_prefix) {
  read_lock r_lock(chunks_mutex_);
  std::vector<ChunkKey> ret_vec;
  iterateOverMatchingPrefix(
      [&ret_vec](auto chunk) { ret_vec.push_back(chunk); }, cached_chunks_, chunk_prefix);
  return ret_vec;
}

std::map<ChunkKey, AbstractBuffer*> ForeignStorageCache::getChunkBuffersForCaching(
    const std::vector<ChunkKey>& chunk_keys) {
  auto timer = DEBUG_TIMER(__func__);
  std::map<ChunkKey, AbstractBuffer*> chunk_buffer_map;
  read_lock lock(chunks_mutex_);
  for (const auto& chunk_key : chunk_keys) {
    CHECK(cached_chunks_.find(chunk_key) == cached_chunks_.end());
    CHECK(global_file_mgr_->isBufferOnDevice(chunk_key));
    chunk_buffer_map[chunk_key] = global_file_mgr_->getBuffer(chunk_key);
    CHECK_EQ(chunk_buffer_map[chunk_key]->pageCount(), static_cast<size_t>(0));

    // Clear all buffer metadata
    chunk_buffer_map[chunk_key]->encoder = nullptr;
    chunk_buffer_map[chunk_key]->has_encoder = false;
    chunk_buffer_map[chunk_key]->setSize(0);
  }
  return chunk_buffer_map;
}

// Private functions.  Locks should be acquired in the public interface before calling
// these functions.
// This function assumes the chunk has been erased from the eviction algorithm already.
void ForeignStorageCache::eraseChunk(const ChunkKey& chunk_key) {
  auto timer = DEBUG_TIMER(__func__);
  File_Namespace::FileBuffer* file_buffer =
      static_cast<File_Namespace::FileBuffer*>(global_file_mgr_->getBuffer(chunk_key));
  file_buffer->freeChunkPages();
  cached_chunks_.erase(chunk_key);
}

void ForeignStorageCache::evictChunkByAlg() {
  auto timer = DEBUG_TIMER(__func__);
  eraseChunk(eviction_alg_->evictNextChunk());
}

bool ForeignStorageCache::isCacheFull() const {
  auto timer = DEBUG_TIMER(__func__);
  return (cached_chunks_.size() >= entry_limit_);
}

std::string ForeignStorageCache::dumpCachedChunkEntries() const {
  auto timer = DEBUG_TIMER(__func__);
  std::string ret_string = "Cached chunks:\n";
  for (const auto& chunk_key : cached_chunks_) {
    ret_string += "  " + showChunk(chunk_key) + "\n";
  }
  return ret_string;
}

std::string ForeignStorageCache::dumpCachedMetadataEntries() const {
  auto timer = DEBUG_TIMER(__func__);
  std::string ret_string = "Cached ChunkMetadata:\n";
  for (const auto& meta_key : cached_metadata_) {
    ret_string += "  " + showChunk(meta_key) + "\n";
  }
  return ret_string;
}

std::string ForeignStorageCache::dumpEvictionQueue() const {
  return ((LRUEvictionAlgorithm*)eviction_alg_.get())->dumpEvictionQueue();
}

void ForeignStorageCache::validatePath(const std::string& base_path) {
  // check if base_path already exists, and if not create one
  boost::filesystem::path path(base_path);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path)) {
      throw std::runtime_error{
          "cache path \"" + base_path +
          "\" is not a directory.  Please specify a valid directory "
          "with --disk_cache_path=<path>, or use the default location."};
    }
  } else {  // data directory does not exist
    if (!boost::filesystem::create_directory(path)) {
      throw std::runtime_error{
          "could not create directory at cache path \"" + base_path +
          "\".  Please specify a valid directory location "
          "with --disk_cache_path=<path> or use the default location."};
    }
  }
}

}  // namespace foreign_storage
