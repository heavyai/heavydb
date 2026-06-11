/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file	ForeignStorageCache.h
 * @brief
 *
 * This file includes the class specification for the cache used by the Foreign Storage
 * Interface (FSI).  This cache is used by FSI to cache data and metadata locally on disc
 * to avoid repeated loads from foreign storage.
 */

#pragma once

#include "../Shared/heavyai_shared_mutex.h"
#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/FileMgr/CachingFileMgr.h"
#include "ForeignDataWrapper.h"

class CacheTooSmallException : public std::runtime_error {
 public:
  CacheTooSmallException(const std::string& msg) : std::runtime_error(msg) {}
};

using namespace Data_Namespace;

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
  size_t getMaxChunkDataSize() const { return caching_file_mgr_->getMaxDataFilesSize(); }
  std::vector<ChunkKey> getCachedChunksForKeyPrefix(const ChunkKey&) const;

  ChunkToBufferMap getChunkBuffersForCaching(const std::set<ChunkKey>& chunk_keys) const;

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
  std::string dumpEvictionQueue() const;
  std::string dump() const { return caching_file_mgr_->dump(); }

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

  inline uint64_t getSpaceReservedByTable(int db_id, int tb_id) const {
    return caching_file_mgr_->getSpaceReservedByTable(db_id, tb_id);
  }

  void storeDataWrapper(const std::string& doc, int32_t db_id, int32_t tb_id);

  bool hasStoredDataWrapperMetadata(int32_t db_id, int32_t table_id) const;

  void eraseChunk(const ChunkKey& chunk_key);

  // Used for unit testing
  inline void setDataSizeLimit(size_t max) const {
    caching_file_mgr_->setDataSizeLimit(max);
  }

 private:
  void validatePath(const std::string&) const;

  // Underlying storage is handled by a CachingFileMgr unique to the cache.
  std::unique_ptr<File_Namespace::CachingFileMgr> caching_file_mgr_;

};  // ForeignStorageCache
}  // namespace foreign_storage
