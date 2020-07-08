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

void ForeignStorageCache::cacheChunk(const ChunkKey& chunk_key, AbstractBuffer* buffer) {
  auto timer = DEBUG_TIMER(__func__);
  write_lock lock(chunks_mutex_);
  if (isCacheFull()) {
    evictChunkByAlg();
  }
  eviction_alg_->touchChunk(chunk_key);
  cached_chunks_.emplace(chunk_key);
  global_file_mgr_->putBuffer(chunk_key, buffer);
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

void ForeignStorageCache::cacheMetadataVec(ChunkMetadataVector& metadata_vec) {
  auto timer = DEBUG_TIMER(__func__);
  write_lock lock(metadata_mutex_);
  for (auto& [chunk_key, metadata] : metadata_vec) {
    cached_metadata_[chunk_key] = metadata;
  }
}

void ForeignStorageCache::getCachedMetadataVecForKeyPrefix(
    ChunkMetadataVector& metadata_vec,
    const ChunkKey& chunk_prefix) {
  auto timer = DEBUG_TIMER(__func__);
  read_lock r_lock(metadata_mutex_);
  ChunkKey upper_prefix(chunk_prefix);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  auto end_it = cached_metadata_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
  for (auto meta_it = cached_metadata_.lower_bound(chunk_prefix); meta_it != end_it;
       ++meta_it) {
    metadata_vec.push_back(*meta_it);
  }
}

bool ForeignStorageCache::hasCachedMetadataForKeyPrefix(const ChunkKey& chunk_prefix) {
  auto timer = DEBUG_TIMER(__func__);
  read_lock lock(metadata_mutex_);
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
  CHECK(chunk_prefix.size() == 2U);
  auto timer = DEBUG_TIMER(__func__);
  ChunkKey upper_prefix(chunk_prefix);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  {
    write_lock w_lock(chunks_mutex_);
    // Delete chunks for prefix
    auto end_it = cached_chunks_.upper_bound(static_cast<const ChunkKey>(upper_prefix));
    for (auto chunk_it = cached_chunks_.lower_bound(chunk_prefix); chunk_it != end_it;) {
      chunk_it = eraseChunk(chunk_it);
    }
  }
  {
    write_lock w_lock(metadata_mutex_);
    // Delete metadata for prefix
    cached_metadata_.erase(
        cached_metadata_.lower_bound(chunk_prefix),
        cached_metadata_.upper_bound(static_cast<const ChunkKey>(upper_prefix)));
  }
}

void ForeignStorageCache::clear() {
  auto timer = DEBUG_TIMER(__func__);
  {
    write_lock w_lock(chunks_mutex_);
    for (auto chunk_it = cached_chunks_.begin(); chunk_it != cached_chunks_.end();) {
      chunk_it = eraseChunk(chunk_it);
    }
  }
  {
    write_lock w_lock(metadata_mutex_);
    cached_metadata_.clear();
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
}

// Private functions.  Locks should be acquired in the public interface before calling
// these functions.
// This function assumes the chunk has been erased from the eviction algorithm already.
void ForeignStorageCache::eraseChunk(const ChunkKey& chunk_key) {
  auto timer = DEBUG_TIMER(__func__);
  global_file_mgr_->deleteBuffer(chunk_key);
  cached_chunks_.erase(chunk_key);
}

// This function assumes the chunk has not already been erased from the eviction alg.
std::set<ChunkKey>::iterator ForeignStorageCache::eraseChunk(
    const std::set<ChunkKey>::iterator& chunk_it) {
  auto timer = DEBUG_TIMER(__func__);
  global_file_mgr_->deleteBuffer(*chunk_it);
  eviction_alg_->removeChunk(*chunk_it);
  return cached_chunks_.erase(chunk_it);
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
  for (const auto& meta_pair : cached_metadata_) {
    ret_string += "  " + showChunk(meta_pair.first) + "\n";
  }
  return ret_string;
}

std::string ForeignStorageCache::dumpEvictionQueue() const {
  return ((LRUEvictionAlgorithm*)eviction_alg_.get())->dumpEvictionQueue();
}

}  // namespace foreign_storage
