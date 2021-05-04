/*
 * Copyright 2021 Omnisci, Inc.
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
 * @file	CachingFileMgr.h
 *
 * This file details an extension of the FileMgr that can contain pages from multiple
 * tables (CachingFileMgr).
 * The main differences between a CFM and an FM:
 * - CFM can contain pages from multiple different tables in the same file.
 * - CFM will only retain a single version of each page (no support for rolloff or
 * rollback functionality).
 * - CFM maintains a separate epoch for each table in it's files.
 */

#pragma once

#include "FileMgr.h"
#include "Shared/File.h"

namespace File_Namespace {

inline std::string get_dir_name_for_table(int db_id, int tb_id) {
  std::stringstream file_name;
  file_name << "table_" << db_id << "_" << tb_id << "/";
  return file_name.str();
}

// Struct to group data related to a single epoch.  Manages epoch file pointers.
struct EpochInfo {
  Epoch epoch;
  FILE* epoch_file = nullptr;
  bool is_checkpointed = true;
  EpochInfo(FILE* f) {
    CHECK(f) << "Cannot create EpochInfo from null file descriptor";
    epoch = Epoch();
    epoch_file = f;
    is_checkpointed = true;
  }
  ~EpochInfo() { close(epoch_file); }
  void increment() {
    epoch.increment();
    is_checkpointed = false;
    CHECK(epoch.ceiling() <= Epoch::max_allowable_epoch())
        << "Epoch greater than maximum allowed value (" << epoch.ceiling() << " > "
        << Epoch::max_allowable_epoch() << ").";
  }
};

// Extension of FileBuffer with restricted behaviour.
class CachingFileBuffer : public FileBuffer {
 public:
  using FileBuffer::FileBuffer;
  // The cache can only be appended to, not written, as it lets us maintain a single
  // version of the data.  This override is to make sure we don't accidentally start
  // writing to cache buffers.
  void write(int8_t* src,
             const size_t numBytes,
             const size_t offset = 0,
             const MemoryLevel srcMemoryLevel = CPU_LEVEL,
             const int32_t deviceId = -1) override {
    UNREACHABLE() << "Cache buffers support append(), but not write()";
  }
};

/**
 * @class   CachingFileMgr
 * @brief   A FileMgr capable of limiting it's size and storing data from multiple tables
 * in a shared directory.  For any table that supports DiskCaching, the CachingFileMgr
 * must contain either metadata for all table chunks, or for none (the cache is either has
 * no knowledge of that table, or has complete knowledge of that table).  Any data chunk
 * within a table may or may not be contained within the cache.
 */
class CachingFileMgr : public FileMgr {
 public:
  CachingFileMgr(const std::string& base_path,
                 const size_t num_reader_threads = 0,
                 const size_t default_page_size = DEFAULT_PAGE_SIZE);
  virtual ~CachingFileMgr();

  // Simple getters.
  inline MgrType getMgrType() override { return CACHING_FILE_MGR; };
  inline std::string getStringMgrType() override { return ToString(CACHING_FILE_MGR); }
  inline size_t getDefaultPageSize() { return defaultPageSize_; }

  // TODO(Misiu): These are unimplemented for now, but will become necessary when we want
  // to limit the size.
  inline size_t getMaxSize() override {
    UNREACHABLE() << "Unimplemented";
    return 0;
  }
  inline size_t getInUseSize() override {
    UNREACHABLE() << "Unimplemented";
    return 0;
  }
  inline size_t getAllocated() override {
    UNREACHABLE() << "Unimplemented";
    return 0;
  }
  inline bool isAllocationCapped() override { return false; }

  /**
   * @brief Removes all data related to the given table (pages and subdirectories).
   */
  void clearForTable(int32_t db_id, int32_t tb_id);

  /**
   * @brief Returns (and optionally creates) a subdirectory for table-specific persistent
   * data (e.g. serialized foreign data warppers).
   */
  std::string getOrAddTableDir(int db_id, int tb_id);

  /**
   * @brief Query to determine if the contained pages will have their database and table
   * ids overriden by the filemgr key (FileMgr does this).
   */
  inline bool hasFileMgrKey() const override { return false; }
  /**
   * @brief Closes files and removes the caching directory.
   */
  void closeRemovePhysical() override;

  /**
   * Set of functions to determine how much space is reserved in a table by type.
   */
  uint64_t getChunkSpaceReservedByTable(int db_id, int tb_id);
  uint64_t getMetadataSpaceReservedByTable(int db_id, int tb_id);
  uint64_t getWrapperSpaceReservedByTable(int db_id, int tb_id);
  uint64_t getSpaceReservedByTable(int db_id, int tb_id);

  /**
   * @brief describes this FileMgr for logging purposes.
   */
  std::string describeSelf() override;

  /**
   * @brief writes buffers for the given table, synchronizes files to disk, updates file
   * epoch, and commits free pages.
   */
  void checkpoint(const int32_t db_id, const int32_t tb_id) override;

  /**
   * @brief obtain the epoch version for the given table.
   */
  int32_t epoch(int32_t db_id, int32_t tb_id) const override;

  /**
   * @brief deletes any existing buffer for the given key then copies in a new one.
   */
  FileBuffer* putBuffer(const ChunkKey& key,
                        AbstractBuffer* srcBuffer,
                        const size_t numBytes = 0) override;
  /**
   * @brief allocates a new CachingFileBuffer.
   */
  CachingFileBuffer* allocateBuffer(const size_t page_size,
                                    const ChunkKey& key,
                                    const size_t num_bytes) override;

  /**
   * @brief checks whether a page should be deleted.
   **/
  bool updatePageIfDeleted(FileInfo* file_info,
                           ChunkKey& chunk_key,
                           int32_t contingent,
                           int32_t page_epoch,
                           int32_t page_num) override;

  /**
   * @brief True if a read error should cause a fatal error.
   **/
  inline bool failOnReadError() const override { return false; }

  /**
   * @brief deletes a buffer if it exists in the mgr.  Otherwise do nothing.
   **/
  void deleteBufferIfExists(const ChunkKey& key);

 private:
  void openOrCreateEpochIfNotExists(int32_t db_id, int32_t tb_id);
  void openAndReadEpochFileUnlocked(int32_t db_id, int32_t tb_id);
  void incrementEpoch(int32_t db_id, int32_t tb_id);
  void init(const size_t num_reader_threads);
  void createEpochFileUnlocked(int32_t db_id, int32_t tb_id);
  void writeAndSyncEpochToDisk(int32_t db_id, int32_t tb_id);
  std::string getOrAddTableDirUnlocked(int db_id, int tb_id);
  void readTableDirs();
  void createBufferFromHeaders(const ChunkKey& key,
                               const std::vector<HeaderInfo>::const_iterator& startIt,
                               const std::vector<HeaderInfo>::const_iterator& endIt);
  FileBuffer* createBufferUnlocked(const ChunkKey& key,
                                   size_t pageSize = 0,
                                   const size_t numBytes = 0) override;
  void incrementAllEpochs();
  void removeTableDirectory(int32_t db_id, int32_t tb_id);
  void removeTableBuffers(int32_t db_id, int32_t tb_id);
  void writeDirtyBuffers(int32_t db_id, int32_t tb_id);

  mutable mapd_shared_mutex epochs_mutex_;  // mutex for table_epochs_.
  // each table gets a separate epoch.  Uses pointers for move semantics.
  std::map<TablePair, std::unique_ptr<EpochInfo>> table_epochs_;
};

}  // namespace File_Namespace
