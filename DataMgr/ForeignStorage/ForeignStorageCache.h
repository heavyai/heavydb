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

struct DiskCacheConfig {
  std::string path;
  bool is_enabled = false;
  size_t entry_limit = 1024;
  size_t num_reader_threads = 0;
  DiskCacheConfig() {}
  DiskCacheConfig(std::string p,
                  bool enabled = true,
                  size_t limit = 1024,
                  size_t readers = 0)
      : path(p), is_enabled(enabled), entry_limit(limit), num_reader_threads(readers) {}
};

using namespace Data_Namespace;

namespace foreign_storage {

class ForeignStorageCache {
 public:
  ForeignStorageCache(const std::string& cache_dir,
                      const size_t num_reader_threads,
                      const size_t limit);

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
  bool isMetadataCached(const ChunkKey&);
  void cacheMetadataVec(const ChunkMetadataVector&);
  void getCachedMetadataVecForKeyPrefix(ChunkMetadataVector&, const ChunkKey&);
  bool hasCachedMetadataForKeyPrefix(const ChunkKey&);
  void clearForTablePrefix(const ChunkKey&);
  void clear();
  void setLimit(size_t limit);
  std::vector<ChunkKey> getCachedChunksForKeyPrefix(const ChunkKey&);
  bool recoverCacheForTable(ChunkMetadataVector&, const ChunkKey&);
  std::map<ChunkKey, AbstractBuffer*> getChunkBuffersForCaching(
      const std::vector<ChunkKey>& chunk_keys);

  // Exists for testing purposes.
  size_t getLimit() const { return entry_limit_; }
  size_t getNumCachedChunks() const { return cached_chunks_.size(); }
  size_t getNumCachedMetadata() const { return cached_metadata_.size(); }
  size_t getNumChunksAdded() const { return num_chunks_added_; }
  size_t getNumMetadataAdded() const { return num_metadata_added_; }

  // Useful for debugging.
  std::string dumpCachedChunkEntries() const;
  std::string dumpCachedMetadataEntries() const;
  std::string dumpEvictionQueue() const;

  File_Namespace::GlobalFileMgr* getGlobalFileMgr() { return global_file_mgr_.get(); }

  std::string getCacheDirectoryForTablePrefix(const ChunkKey&);

 private:
  // These methods are private and assume locks are already acquired when called.
  std::set<ChunkKey>::iterator eraseChunk(const std::set<ChunkKey>::iterator&);
  void eraseChunk(const ChunkKey&);
  void evictThenEraseChunk(const ChunkKey&);
  void evictChunkByAlg();
  bool isCacheFull() const;
  void validatePath(const std::string&);

  // We can swap out different eviction algorithms here.
  std::unique_ptr<CacheEvictionAlgorithm> eviction_alg_ =
      std::make_unique<LRUEvictionAlgorithm>();

  // Underlying storage is handled by a GlobalFileMgr unique to the cache.
  std::unique_ptr<File_Namespace::GlobalFileMgr> global_file_mgr_;

  // Keeps tracks of which Chunks/ChunkMetadata are cached.
  std::set<ChunkKey> cached_chunks_;
  std::set<ChunkKey> cached_metadata_;

  // Keeps tracks of how many times we cache chunks or metadata for testing purposes.
  size_t num_chunks_added_;
  size_t num_metadata_added_;

  // Separate mutexes for chunks/metadata.
  mapd_shared_mutex chunks_mutex_;
  mapd_shared_mutex metadata_mutex_;

  // Maximum number of Chunks that can be in the cache before eviction.
  size_t entry_limit_;
};  // ForeignStorageCache
}  // namespace foreign_storage
