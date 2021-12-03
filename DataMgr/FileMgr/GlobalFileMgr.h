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

class ForeignStorageInterface;

using namespace Data_Namespace;

namespace File_Namespace {

struct FileMgrParams {
  FileMgrParams() : epoch(-1), max_rollback_epochs(-1) {}
  int32_t epoch;
  int32_t max_rollback_epochs;
};

/**
 * @class   GlobalFileMgr
 * @brief
 */
class GlobalFileMgr : public AbstractBufferMgr {  // implements

 public:
  /// Constructor
  GlobalFileMgr(const int32_t deviceId,
                std::shared_ptr<ForeignStorageInterface> fsi,
                std::string basePath = ".",
                const size_t num_reader_threads = 0,
                const size_t defaultPageSize = DEFAULT_PAGE_SIZE);

  virtual ~GlobalFileMgr() {}

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
    return nullptr;  // satisfy return-type warning
  }

  void free(AbstractBuffer* buffer) override { LOG(FATAL) << "Operation not supported"; }

  inline MgrType getMgrType() override { return GLOBAL_FILE_MGR; };
  inline std::string getStringMgrType() override { return ToString(GLOBAL_FILE_MGR); }
  inline std::string printSlabs() override { return "Not Implemented"; }
  inline size_t getMaxSize() override { return 0; }
  inline size_t getInUseSize() override { return 0; }
  inline size_t getAllocated() override { return 0; }
  inline bool isAllocationCapped() override { return false; }

  void init();

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix) override {
    return getFileMgr(keyPrefix)->getChunkMetadataVecForKeyPrefix(chunkMetadataVec,
                                                                  keyPrefix);
  }

  /**
   * @brief Fsyncs data files, writes out epoch and
   * fsyncs that
   */
  void checkpoint() override;
  void checkpoint(const int32_t db_id, const int32_t tb_id) override;

  /**
   * @brief Returns number of threads defined by parameter num-reader-threads
   * which should be used during initial load and consequent read of data.
   */
  inline size_t getNumReaderThreads() { return num_reader_threads_; }

  size_t getNumChunks() override;

  void compactDataFiles(const int32_t db_id, const int32_t tb_id);

 private:
  AbstractBufferMgr* findFileMgrUnlocked(const int32_t db_id, const int32_t tb_id);
  void deleteFileMgr(const int32_t db_id, const int32_t tb_id);

 public:
  AbstractBufferMgr* findFileMgr(const int32_t db_id, const int32_t tb_id) {
    mapd_shared_lock<mapd_shared_mutex> read_lock(fileMgrs_mutex_);
    return findFileMgrUnlocked(db_id, tb_id);
  }
  void setFileMgrParams(const int32_t db_id,
                        const int32_t tb_id,
                        const FileMgrParams& file_mgr_params);
  AbstractBufferMgr* getFileMgr(const int32_t db_id, const int32_t tb_id);
  AbstractBufferMgr* getFileMgr(const ChunkKey& key) {
    return getFileMgr(key[0], key[1]);
  }

  std::string getBasePath() const { return basePath_; }
  size_t getDefaultPageSize() const { return defaultPageSize_; }

  void writeFileMgrData(FileMgr* fileMgr = 0);

  inline int32_t getDBVersion() const { return omnisci_db_version_; }
  inline bool getDBConvert() const { return dbConvert_; }
  inline void setDBConvert(bool val) { dbConvert_ = val; }

  void removeTableRelatedDS(const int32_t db_id, const int32_t tb_id) override;
  void setTableEpoch(const int32_t db_id, const int32_t tb_id, const int32_t start_epoch);
  size_t getTableEpoch(const int32_t db_id, const int32_t tb_id);
  StorageStats getStorageStats(const int32_t db_id, const int32_t tb_id);

  // For testing purposes only
  std::shared_ptr<FileMgr> getSharedFileMgr(const int db_id, const int table_id);

  // For testing purposes only
  void setFileMgr(const int db_id, const int table_id, std::shared_ptr<FileMgr> file_mgr);
  void closeFileMgr(const int32_t db_id,
                    const int32_t tb_id);  // A locked public wrapper for deleteFileMgr,
                                           // for now for unit testing

  void prepareTablesForExecution(const ColumnByIdxRefSet& input_cols,
                                 const CompilationOptions& co,
                                 const ExecutionOptions& eo,
                                 ExecutionPhase phase) override {}

  const DictDescriptor* getDictMetadata(int db_id,
                                        int dict_id,
                                        bool load_dict = true) override;

 protected:
  std::shared_ptr<ForeignStorageInterface> fsi_;

 private:
  bool existsDiffBetweenFileMgrParamsAndFileMgr(
      FileMgr* file_mgr,
      const FileMgrParams& file_mgr_params) const;
  std::string basePath_;       /// The OS file system path containing the files.
  size_t num_reader_threads_;  /// number of threads used when loading data
  int32_t
      epoch_; /* the current epoch (time of last checkpoint) will be used for all
               * tables except of the one for which the value of the epoch has been reset
               * using --start-epoch option at start up to rollback this table's updates.
               */
  size_t defaultPageSize_;  /// default page size, used to set FileMgr defaultPageSize_
  // bool isDirty_;               /// true if metadata changed since last writeState()

  int32_t omnisci_db_version_;  /// DB version for DataMgr DS and corresponding file
                                /// buffer read/write code
  /* In future omnisci_db_version_ may be added to AbstractBufferMgr class.
   * This will allow support of different dbVersions for different tables, so
   * original tables can be generated by different versions of mapd software.
   */
  bool dbConvert_;  /// true if conversion should be done between different
                    /// "omnisci_db_version_"

  std::map<TablePair, std::shared_ptr<FileMgr>> ownedFileMgrs_;
  std::map<TablePair, AbstractBufferMgr*> allFileMgrs_;
  std::map<TablePair, int32_t> max_rollback_epochs_per_table_;

  mapd_shared_mutex fileMgrs_mutex_;
};

}  // namespace File_Namespace

#endif  // DATAMGR_MEMORY_FILE_GLOBAL_FILEMGR_H
