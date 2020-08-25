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

using namespace Data_Namespace;

namespace foreign_storage {

class ForeignStorageCache {
 public:
  ForeignStorageCache(File_Namespace::GlobalFileMgr* gfm, size_t limit)
      : global_file_mgr_(gfm), entry_limit_(limit) {}
  void cacheChunk(const ChunkKey&, AbstractBuffer*);
  AbstractBuffer* getCachedChunkIfExists(const ChunkKey&);
  bool isMetadataCached(const ChunkKey&);
  void cacheMetadataVec(ChunkMetadataVector&);
  void getCachedMetadataVecForKeyPrefix(ChunkMetadataVector&, const ChunkKey&);
  bool hasCachedMetadataForKeyPrefix(const ChunkKey&);
  void clearForTablePrefix(const ChunkKey&);
  void clear();
  void setLimit(size_t limit);
  std::vector<ChunkKey> getCachedChunksForKeyPrefix(const ChunkKey&);

  // Exists for testing purposes.
  size_t getLimit() const { return entry_limit_; }
  size_t getNumCachedChunks() const { return cached_chunks_.size(); }
  size_t getNumCachedMetadata() const { return cached_metadata_.size(); }

  // Useful for debugging.
  std::string dumpCachedChunkEntries() const;
  std::string dumpCachedMetadataEntries() const;
  std::string dumpEvictionQueue() const;

 private:
  // These methods are private and assume locks are already acquired when called.
  std::set<ChunkKey>::iterator eraseChunk(const std::set<ChunkKey>::iterator&);
  void eraseChunk(const ChunkKey&);
  void evictChunkByAlg();
  bool isCacheFull() const;

  // We can swap out different eviction algorithms here.
  std::unique_ptr<CacheEvictionAlgorithm> eviction_alg_ =
      std::make_unique<LRUEvictionAlgorithm>();

  // Need pointer to GFM as it is used for storage.
  File_Namespace::GlobalFileMgr* global_file_mgr_;

  // Keeps tracks of which Chunks/ChunkMetadata are cached.
  std::set<ChunkKey> cached_chunks_;
  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> cached_metadata_;

  // Separate mutexes for chunks/metadata.
  mapd_shared_mutex chunks_mutex_;
  mapd_shared_mutex metadata_mutex_;

  // Maximum number of Chunks that can be in the cache before eviction.
  size_t entry_limit_;
};  // ForeignStorageCache
}  // namespace foreign_storage
