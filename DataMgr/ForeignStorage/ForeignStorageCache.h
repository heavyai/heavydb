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

#include "../Shared/mapd_shared_mutex.h"
#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/FileMgr/CachingFileMgr.h"
#include "ForeignDataWrapper.h"

class CacheTooSmallException : public std::runtime_error {
 public:
  CacheTooSmallException(const std::string& msg) : std::runtime_error(msg) {}
};

enum class DiskCacheLevel { none, fsi, non_fsi, all };
struct DiskCacheConfig {
  std::string path;
  DiskCacheLevel enabled_level = DiskCacheLevel::none;
  size_t num_reader_threads = 0;
  size_t page_size = DEFAULT_PAGE_SIZE;
  inline bool isEnabledForMutableTables() const {
    return enabled_level == DiskCacheLevel::non_fsi ||
           enabled_level == DiskCacheLevel::all;
  }
  inline bool isEnabledForFSI() const {
    return enabled_level == DiskCacheLevel::fsi || enabled_level == DiskCacheLevel::all;
  }
  inline bool isEnabled() const { return enabled_level != DiskCacheLevel::none; }
};

using namespace Data_Namespace;

namespace foreign_storage {

const std::string wrapper_file_name = "wrapper_metadata.json";

class ForeignStorageCache {
 public:
  ForeignStorageCache(const DiskCacheConfig& config);

  /**
   * Caches the chunks for the given chunk keys. Chunk buffers
   * for chunks to be cached are expected to have already been
   * populated before calling this method. This method also
   * expects all provided chunk keys to be for the same table.
   *
   * @param chunk_keys - keys of chunks to be cached
   */
  void cacheTableChunks(const std::vector<ChunkKey>& chunk_keys);
  void cacheChunk(const ChunkKey&, AbstractBuffer*);

  File_Namespace::FileBuffer* getCachedChunkIfExists(const ChunkKey&);
  bool isMetadataCached(const ChunkKey&) const;
  void cacheMetadataVec(const ChunkMetadataVector&);
  void getCachedMetadataVecForKeyPrefix(ChunkMetadataVector&, const ChunkKey&) const;
  bool hasCachedMetadataForKeyPrefix(const ChunkKey&) const;
  void clearForTablePrefix(const ChunkKey&);
  void clear();
  std::vector<ChunkKey> getCachedChunksForKeyPrefix(const ChunkKey&) const;
  bool recoverCacheForTable(ChunkMetadataVector&, const ChunkKey&);
  ChunkToBufferMap getChunkBuffersForCaching(
      const std::vector<ChunkKey>& chunk_keys) const;

  // Get a chunk buffer for writing to disk prior to metadata creation/caching
  AbstractBuffer* getChunkBufferForPrecaching(const ChunkKey& chunk_key,
                                              bool is_new_buffer);

  void deleteBufferIfExists(const ChunkKey& chunk_key);

  inline size_t getNumCachedChunks() const { return cached_chunks_.size(); }
  inline size_t getNumCachedMetadata() const { return cached_metadata_.size(); }
  size_t getNumChunksAdded() const { return num_chunks_added_; }
  size_t getNumMetadataAdded() const { return num_metadata_added_; }

  // Useful for debugging.
  std::string dumpCachedChunkEntries() const;
  std::string dumpCachedMetadataEntries() const;

  inline std::string getCacheDirectory() const {
    return caching_file_mgr_->getFileMgrBasePath();
  }

  inline std::string getCacheDirectoryForTable(int db_id, int tb_id) const {
    return caching_file_mgr_->getOrAddTableDir(db_id, tb_id);
  }

  void cacheMetadataWithFragIdGreaterOrEqualTo(const ChunkMetadataVector& metadata_vec,
                                               const int frag_id);
  void evictThenEraseChunk(const ChunkKey&);

  inline uint64_t getSpaceReservedByTable(int db_id, int tb_id) const {
    return caching_file_mgr_->getSpaceReservedByTable(db_id, tb_id);
  }

 private:
  // These methods are private and assume locks are already acquired when called.
  std::set<ChunkKey>::iterator eraseChunk(const std::set<ChunkKey>::iterator&);
  void eraseChunk(const ChunkKey& chunk_key);
  std::set<ChunkKey>::iterator evictChunkByIterator(
      const std::set<ChunkKey>::iterator& chunk_it);
  void evictThenEraseChunkUnlocked(const ChunkKey&);
  void validatePath(const std::string&) const;

  // Underlying storage is handled by a CachingFileMgr unique to the cache.
  std::unique_ptr<File_Namespace::CachingFileMgr> caching_file_mgr_;

  // Keeps tracks of which Chunks/ChunkMetadata are cached.
  std::set<ChunkKey> cached_chunks_;
  std::set<ChunkKey> cached_metadata_;

  // Keeps tracks of how many times we cache chunks or metadata for testing purposes.
  size_t num_chunks_added_;
  size_t num_metadata_added_;

  // Separate mutexes for chunks/metadata.
  mutable mapd_shared_mutex chunks_mutex_;
  mutable mapd_shared_mutex metadata_mutex_;
};  // ForeignStorageCache
}  // namespace foreign_storage
