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

/**
 * @file	ForeignStorageCache.h
 * @author	Misiu Godfrey <misiu.godfrey@omnisci.com>
 *
 * This file includes the class specification for the cache used by the Foreign Storage
 * Interface (FSI).  This cache is used by FSI to cache data and metadata locally on disc
 * to avoid repeated loads from foreign storage.
 */

#pragma once

#include <gtest/gtest.h>
#include "../Shared/mapd_shared_mutex.h"
#include "CacheEvictionAlgorithms/CacheEvictionAlgorithm.h"
#include "CacheEvictionAlgorithms/LRUEvictionAlgorithm.h"
#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "ForeignDataWrapper.h"

class CacheTooSmallException : public std::runtime_error {
 public:
  CacheTooSmallException(const std::string& msg) : std::runtime_error(msg) {}
};

struct DiskCacheConfig {
  std::string path;
  bool is_enabled = false;
  uint64_t size_limit = 21474836480;  // 20GB default
  size_t num_reader_threads = 0;
};

using namespace Data_Namespace;

namespace foreign_storage {

struct TableEvictionTracker {
  // We can swap out different eviction algorithms here.
  std::unique_ptr<CacheEvictionAlgorithm> eviction_alg_ =
      std::make_unique<LRUEvictionAlgorithm>();
  size_t num_pages_ = 0;
};

class ForeignStorageCache {
 public:
  ForeignStorageCache(const std::string& cache_dir,
                      const size_t num_reader_threads,
                      const uint64_t limit);

  /**
   * Caches the chunks for the given chunk keys. Chunk buffers
   * for chunks to be cached are expected to have already been
   * populated before calling this method. This method also
   * expects all provided chunk keys to be for the same table.
   *
   * @param chunk_keys - keys of chunks to be cached
   */
  void cacheTableChunks(const std::vector<ChunkKey>& chunk_keys);

  AbstractBuffer* getCachedChunkIfExists(const ChunkKey&);
  bool isMetadataCached(const ChunkKey&) const;
  void cacheMetadataVec(const ChunkMetadataVector&);
  void getCachedMetadataVecForKeyPrefix(ChunkMetadataVector&, const ChunkKey&) const;
  bool hasCachedMetadataForKeyPrefix(const ChunkKey&) const;
  void clearForTablePrefix(const ChunkKey&);
  void clear();
  void setLimit(uint64_t limit);
  std::vector<ChunkKey> getCachedChunksForKeyPrefix(const ChunkKey&) const;
  bool recoverCacheForTable(ChunkMetadataVector&, const ChunkKey&);
  std::map<ChunkKey, AbstractBuffer*> getChunkBuffersForCaching(
      const std::vector<ChunkKey>& chunk_keys) const;

  // Exists for testing purposes.
  inline uint64_t getLimit() const {
    return max_pages_per_table_ * global_file_mgr_->getDefaultPageSize();
  }
  inline size_t getNumCachedChunks() const { return cached_chunks_.size(); }
  inline size_t getNumCachedMetadata() const { return cached_metadata_.size(); }
  size_t getNumChunksAdded() const { return num_chunks_added_; }
  size_t getNumMetadataAdded() const { return num_metadata_added_; }

  // Useful for debugging.
  std::string dumpCachedChunkEntries() const;
  std::string dumpCachedMetadataEntries() const;
  std::string dumpEvictionQueue() const;

  inline File_Namespace::GlobalFileMgr* getGlobalFileMgr() const {
    return global_file_mgr_.get();
  }

  std::string getCacheDirectoryForTablePrefix(const ChunkKey&);

 private:
  // These methods are private and assume locks are already acquired when called.
  std::set<ChunkKey>::iterator eraseChunk(const std::set<ChunkKey>::iterator&);
  void eraseChunk(const ChunkKey&, TableEvictionTracker& tracker);
  std::set<ChunkKey>::iterator eraseChunkByIterator(
      const std::set<ChunkKey>::iterator& chunk_it);
  void evictThenEraseChunk(const ChunkKey&);
  void validatePath(const std::string&) const;
  void cacheChunk(const ChunkKey&);
  void createTrackerMapEntryIfNoneExists(const ChunkKey& chunk_key);

  std::map<const ChunkKey, TableEvictionTracker> eviction_tracker_map_;
  uint64_t max_pages_per_table_;

  // Underlying storage is handled by a GlobalFileMgr unique to the cache.
  std::unique_ptr<File_Namespace::GlobalFileMgr> global_file_mgr_;

  // Keeps tracks of which Chunks/ChunkMetadata are cached.
  std::set<ChunkKey> cached_chunks_;
  std::set<ChunkKey> cached_metadata_;

  // Keeps tracks of how many times we cache chunks or metadata for testing purposes.
  size_t num_chunks_added_;
  size_t num_metadata_added_;

  // Separate mutexes for chunks/metadata.
  mutable mapd_shared_mutex chunks_mutex_;
  mutable mapd_shared_mutex metadata_mutex_;

  // Maximum number of chunk bytes that can be in the cache before eviction.
  uint64_t max_cached_bytes_;
};  // ForeignStorageCache
}  // namespace foreign_storage
