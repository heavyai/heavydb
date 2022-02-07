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

#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/FileMgr/CachingFileMgr.h"
#include "Shared/mapd_shared_mutex.h"

class CacheTooSmallException : public std::runtime_error {
 public:
  CacheTooSmallException(const std::string& msg) : std::runtime_error(msg) {}
};

using namespace Data_Namespace;

using ChunkToBufferMap = std::map<ChunkKey, AbstractBuffer*>;

namespace foreign_storage {

class ForeignStorageCache {
 public:
  ForeignStorageCache(const File_Namespace::DiskCacheConfig& config);

  void checkpoint(const int32_t db_id, const int32_t tb_id);
  void putBuffer(const ChunkKey&, AbstractBuffer*, const size_t numBytes = 0);
  File_Namespace::FileBuffer* getCachedChunkIfExists(const ChunkKey&);
  bool isMetadataCached(const ChunkKey&) const;
  void cacheMetadataVec(const ChunkMetadataVector&);
  void getCachedMetadataVecForKeyPrefix(ChunkMetadataVector&, const ChunkKey&) const;
  bool hasCachedMetadataForKeyPrefix(const ChunkKey&) const;
  void clearForTablePrefix(const ChunkKey&);
  void clear();
  std::vector<ChunkKey> getCachedChunksForKeyPrefix(const ChunkKey&) const;
  ChunkToBufferMap getChunkBuffersForCaching(
      const std::vector<ChunkKey>& chunk_keys) const;

  // Get a chunk buffer for writing to disk prior to metadata creation/caching
  AbstractBuffer* getChunkBufferForPrecaching(const ChunkKey& chunk_key,
                                              bool is_new_buffer);

  void deleteBufferIfExists(const ChunkKey& chunk_key);

  inline size_t getNumCachedChunks() const {
    return caching_file_mgr_->getNumDataChunks();
  }
  inline size_t getNumCachedMetadata() const {
    return caching_file_mgr_->getNumChunksWithMetadata();
  }

  // Useful for debugging.
  std::string dumpCachedChunkEntries() const;
  std::string dumpCachedMetadataEntries() const;

  inline std::string getCacheDirectory() const {
    return caching_file_mgr_->getFileMgrBasePath();
  }

  inline std::string getCacheDirectoryForTable(int db_id, int tb_id) const {
    return caching_file_mgr_->getTableFileMgrPath(db_id, tb_id);
  }

  inline std::string getSerializedWrapperPath(int32_t db_id, int32_t tb_id) const {
    return getCacheDirectoryForTable(db_id, tb_id) + "/" +
           File_Namespace::CachingFileMgr::WRAPPER_FILE_NAME;
  }

  void cacheMetadataWithFragIdGreaterOrEqualTo(const ChunkMetadataVector& metadata_vec,
                                               const int frag_id);

  inline uint64_t getSpaceReservedByTable(int db_id, int tb_id) const {
    return caching_file_mgr_->getSpaceReservedByTable(db_id, tb_id);
  }

  void storeDataWrapper(const std::string& doc, int32_t db_id, int32_t tb_id);

 private:
  // These methods are private and assume locks are already acquired when called.
  std::set<ChunkKey>::iterator eraseChunk(const std::set<ChunkKey>::iterator&);
  void eraseChunk(const ChunkKey& chunk_key);
  void validatePath(const std::string&) const;

  // Underlying storage is handled by a CachingFileMgr unique to the cache.
  std::unique_ptr<File_Namespace::CachingFileMgr> caching_file_mgr_;

};  // ForeignStorageCache
}  // namespace foreign_storage
