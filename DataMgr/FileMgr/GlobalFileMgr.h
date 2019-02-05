/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @file        GlobalFileMgr.h
 * @author      Norair Khachiyan <norair@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 *
 * This file includes the class specification for the FILE manager (FileMgr), and related
 * data structures and types.
 */

#ifndef DATAMGR_MEMORY_FILE_GLOBAL_FILEMGR_H
#define DATAMGR_MEMORY_FILE_GLOBAL_FILEMGR_H

#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include "../Shared/mapd_shared_mutex.h"

#include "../AbstractBuffer.h"
#include "../AbstractBufferMgr.h"
#include "FileMgr.h"

using namespace Data_Namespace;

namespace File_Namespace {

/**
 * @class   GlobalFileMgr
 * @brief
 */
class GlobalFileMgr : public AbstractBufferMgr {  // implements

 public:
  /// Constructor
  GlobalFileMgr(const int deviceId,
                std::string basePath = ".",
                const size_t num_reader_threads = 0,
                const size_t defaultPageSize = 2097152);

  /// Destructor
  ~GlobalFileMgr() override;

  /// Creates a chunk with the specified key and page size.
  AbstractBuffer* createBuffer(const ChunkKey& key,
                               size_t pageSize = 0,
                               const size_t numBytes = 0) override {
    return getFileMgr(key)->createBuffer(key, pageSize, numBytes);
  }

  bool isBufferOnDevice(const ChunkKey& key) override {
    return getFileMgr(key)->isBufferOnDevice(key);
  }

  /// Deletes the chunk with the specified key
  // Purge == true means delete the data chunks -
  // can't undelete and revert to previous
  // state - reclaims disk space for chunk
  void deleteBuffer(const ChunkKey& key, const bool purge = true) override {
    return getFileMgr(key)->deleteBuffer(key, purge);
  }

  void deleteBuffersWithPrefix(const ChunkKey& keyPrefix,
                               const bool purge = true) override;

  /// Returns the a pointer to the chunk with the specified key.
  AbstractBuffer* getBuffer(const ChunkKey& key, const size_t numBytes = 0) override {
    return getFileMgr(key)->getBuffer(key, numBytes);
  }

  void fetchBuffer(const ChunkKey& key,
                   AbstractBuffer* destBuffer,
                   const size_t numBytes) override {
    return getFileMgr(key)->fetchBuffer(key, destBuffer, numBytes);
  }

  /**
   * @brief Puts the contents of d into the Chunk with the given key.
   * @param key - Unique identifier for a Chunk.
   * @param d - An object representing the source data for the Chunk.
   * @return AbstractBuffer*
   */
  AbstractBuffer* putBuffer(const ChunkKey& key,
                            AbstractBuffer* d,
                            const size_t numBytes = 0) override {
    return getFileMgr(key)->putBuffer(key, d, numBytes);
  }

  // Buffer API
  AbstractBuffer* alloc(const size_t numBytes) override {
    LOG(FATAL) << "Operation not supported";
  }

  void free(AbstractBuffer* buffer) override { LOG(FATAL) << "Operation not supported"; }

  inline MgrType getMgrType() override { return GLOBAL_FILE_MGR; };
  inline std::string getStringMgrType() override { return ToString(GLOBAL_FILE_MGR); }
  inline std::string printSlabs() override { return "Not Implemented"; }
  inline void clearSlabs() override { /* noop */
  }
  inline size_t getMaxSize() override { return 0; }
  inline size_t getInUseSize() override { return 0; }
  inline size_t getAllocated() override { return 0; }
  inline bool isAllocationCapped() override { return false; }

  void init();

  void getChunkMetadataVec(
      std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec) override;
  void getChunkMetadataVecForKeyPrefix(
      std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
      const ChunkKey& keyPrefix) override {
    return getFileMgr(keyPrefix)->getChunkMetadataVecForKeyPrefix(chunkMetadataVec,
                                                                  keyPrefix);
  }

  /**
   * @brief Fsyncs data files, writes out epoch and
   * fsyncs that
   */
  void checkpoint() override;
  void checkpoint(const int db_id, const int tb_id) override;

  /**
   * @brief Returns number of threads defined by parameter num-reader-threads
   * which should be used during initial load and consequent read of data.
   */
  inline size_t getNumReaderThreads() { return num_reader_threads_; }

  size_t getNumChunks() override;

  FileMgr* findFileMgr(const int db_id,
                       const int tb_id,
                       const bool removeFromMap = false);
  FileMgr* getFileMgr(const int db_id, const int tb_id);
  FileMgr* getFileMgr(const ChunkKey& key) { return getFileMgr(key[0], key[1]); }
  std::string getBasePath() const { return basePath_; }
  size_t getDefaultPageSize() const { return defaultPageSize_; }

  void writeFileMgrData(FileMgr* fileMgr = 0);

  inline int getDBVersion() const { return mapd_db_version_; }
  inline bool getDBConvert() const { return dbConvert_; }
  inline void setDBConvert(bool val) { dbConvert_ = val; }

  void removeTableRelatedDS(const int db_id, const int tb_id);
  void setTableEpoch(const int db_id, const int tb_id, const int start_epoch);
  size_t getTableEpoch(const int db_id, const int tb_id);

 private:
  std::string basePath_;       /// The OS file system path containing the files.
  size_t num_reader_threads_;  /// number of threads used when loading data
  int epoch_; /* the current epoch (time of last checkpoint) will be used for all
               * tables except of the one for which the value of the epoch has been reset
               * using --start-epoch option at start up to rollback this table's updates.
               */
  size_t defaultPageSize_;  /// default page size, used to set FileMgr defaultPageSize_
  // bool isDirty_;               /// true if metadata changed since last writeState()
  int mapd_db_version_;  /// DB version for DataMgr DS and corresponding file buffer
                         /// read/write code
  /* In future mapd_db_version_ may be added to AbstractBufferMgr class.
   * This will allow support of different dbVersions for different tables, so
   * original tables can be generated by different versions of mapd software.
   */
  bool dbConvert_;  /// true if conversion should be done between different
                    /// "mapd_db_version_"
  std::map<std::pair<int, int>, FileMgr*> fileMgrs_;
  mapd_shared_mutex fileMgrs_mutex_;
};

}  // namespace File_Namespace

#endif  // DATAMGR_MEMORY_FILE_GLOBAL_FILEMGR_H
