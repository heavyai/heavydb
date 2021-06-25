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

#include "DataMgr/ForeignStorage/CacheEvictionAlgorithms/LRUEvictionAlgorithm.h"
#include "FileMgr.h"
#include "Shared/File.h"

namespace File_Namespace {

enum class DiskCacheLevel { none, fsi, non_fsi, all };
struct DiskCacheConfig {
  static constexpr size_t DEFAULT_MAX_SIZE{1024UL * 1024UL * 1024UL *
                                           100UL};  // 100G default
  std::string path;
  DiskCacheLevel enabled_level = DiskCacheLevel::none;
  size_t num_reader_threads = 0;
  size_t size_limit = DEFAULT_MAX_SIZE;
  size_t page_size = DEFAULT_PAGE_SIZE;
  inline bool isEnabledForMutableTables() const {
    return enabled_level == DiskCacheLevel::non_fsi ||
           enabled_level == DiskCacheLevel::all;
  }
  inline bool isEnabledForFSI() const {
    return enabled_level == DiskCacheLevel::fsi || enabled_level == DiskCacheLevel::all;
  }
  inline bool isEnabled() const { return enabled_level != DiskCacheLevel::none; }
  std::string dump() const {
    std::stringstream ss;
    ss << "DiskCacheConfig(path = " << path << ", level = " << levelAsString()
       << ", threads = " << num_reader_threads << ", size limit = " << size_limit
       << ", page size = " << page_size << ")";
    return ss.str();
  }
  std::string levelAsString() const {
    switch (enabled_level) {
      case DiskCacheLevel::none:
        return "none";
      case DiskCacheLevel::fsi:
        return "fsi";
      case DiskCacheLevel::non_fsi:
        return "non_fsi";
      case DiskCacheLevel::all:
        return "all";
    }
    return "";
  }
  static std::string getDefaultPath(const std::string& base_path) {
    return base_path + "/omnisci_disk_cache";
  }
};

inline std::string get_dir_name_for_table(int db_id, int tb_id) {
  std::stringstream file_name;
  file_name << "table_" << db_id << "_" << tb_id << "/";
  return file_name.str();
}

// Class to control access to the table-specific directories and data created inside a
// CachingFileMgr.
class TableFileMgr {
 public:
  TableFileMgr(const std::string& table_path);
  ~TableFileMgr() { close(epoch_file_); }

  /**
   * @brief increment the epoch for this subdir (not synced to disk).
   */
  void incrementEpoch();

  /**
   * @brief Write and flush the epoch to the epoch file on disk.
   */
  void writeAndSyncEpochToDisk();

  /**
   * @brief Returns the current epoch (locked)
   */
  int32_t getEpoch() const;

  /**
   * @brief Removes all disk data for the subdir.
   */
  void removeDiskContent() const;

  /**
   * @brief Returns the disk space used (in bytes) for the subdir.
   */
  size_t getReservedSpace() const;

  /**
   * @brief Deletes only the wrapper file on disk.
   */
  void deleteWrapperFile() const;

  /**
   * @brief Writes wrapper file to disk.
   */
  void writeWrapperFile(const std::string& doc) const;

 private:
  std::string table_path_;
  std::string epoch_file_path_;
  std::string wrapper_file_path_;
  Epoch epoch_;
  bool is_checkpointed_ = true;
  FILE* epoch_file_ = nullptr;

  mutable mapd_shared_mutex table_mutex_;
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
  static constexpr char WRAPPER_FILE_NAME[] = "wrapper_metadata.json";
  // We currently assign %10 of the cache to data wrapper space arbitrarily.
  // static constexpr size_t WRAPPER_SPACE_RATIO{10};
  // Portion of the CFM space reserved for metadata (metadata files and data wrappers)
  static constexpr float METADATA_SPACE_PERCENTAGE{0.1};
  // Portion of the CFM metadata space reserved for metadata files (subset of above).
  static constexpr float METADATA_FILE_SPACE_PERCENTAGE{0.01};

  static size_t getMinimumSize() {
    // Currently the minimum default size is based on the metadata file size and
    // percentage usage.
    return (METADATA_PAGE_SIZE * DEFAULT_NUM_PAGES_PER_METADATA_FILE) /
           METADATA_FILE_SPACE_PERCENTAGE;
  }

  CachingFileMgr(const DiskCacheConfig& config);

  virtual ~CachingFileMgr();

  // Simple getters.
  inline MgrType getMgrType() override { return CACHING_FILE_MGR; };
  inline std::string getStringMgrType() override { return ToString(CACHING_FILE_MGR); }
  inline size_t getDefaultPageSize() { return defaultPageSize_; }
  inline size_t getMaxSize() override { return max_size_; }
  inline size_t getMaxDataFiles() const { return max_num_data_files_; }
  inline size_t getMaxMetaFiles() const { return max_num_meta_files_; }
  inline size_t getMaxWrapperSize() const { return max_wrapper_space_; }
  inline size_t getDataFileSize() const {
    return defaultPageSize_ * num_pages_per_data_file_;
  }
  inline size_t getMetadataFileSize() const {
    return METADATA_PAGE_SIZE * num_pages_per_metadata_file_;
  }

  size_t getNumDataFiles() const;
  size_t getNumMetaFiles() const;
  inline size_t getAvailableSpace() { return max_size_ - getAllocated(); }
  inline size_t getAvailableWrapperSpace() {
    return max_wrapper_space_ - getTableFileMgrsSize();
  }
  inline size_t getAllocated() override {
    return getFilesSize() + getTableFileMgrsSize();
  }

  /**
   * @brief Free pages for chunk and remove it from the chunk eviction algorithm.
   */
  void removeChunkKeepMetadata(const ChunkKey& key);

  /**
   * @brief Removes all data related to the given table (pages and subdirectories).
   */
  void clearForTable(int32_t db_id, int32_t tb_id);

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
  size_t getChunkSpaceReservedByTable(int32_t db_id, int32_t tb_id) const;
  size_t getMetadataSpaceReservedByTable(int32_t db_id, int32_t tb_id) const;
  size_t getTableFileMgrSpaceReserved(int32_t db_id, int32_t tb_id) const;
  size_t getSpaceReservedByTable(int32_t db_id, int32_t tb_id) const;

  /**
   * @brief describes this FileMgr for logging purposes.
   */
  std::string describeSelf() const override;

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
   * @brief allocates a new CachingFileBuffer and tracks it's use in the eviction
   * algorithms.
   */
  CachingFileBuffer* allocateBuffer(const size_t page_size,
                                    const ChunkKey& key,
                                    const size_t num_bytes = 0) override;
  CachingFileBuffer* allocateBuffer(
      const ChunkKey& key,
      const std::vector<HeaderInfo>::const_iterator& headerStartIt,
      const std::vector<HeaderInfo>::const_iterator& headerEndIt) override;

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

  /**
   * @brief Returns the number of buffers with metadata in the CFM.  Any buffer with an
   * encoder counts.
   **/
  size_t getNumChunksWithMetadata() const;

  /**
   * @brief Returns the number of buffers with chunk data in the CFM.
   **/
  size_t getNumDataChunks() const;

  /**
   * @brief Returns the keys for chunks with chunk data that match the given prefix.
   **/
  std::vector<ChunkKey> getChunkKeysForPrefix(const ChunkKey& prefix) const;

  /**
   * @brief Initializes a new CFM using the initialization values in the current CFM.
   **/
  std::unique_ptr<CachingFileMgr> reconstruct() const;

  /**
   * @brief Deletes the wrapper file from a table subdir.
   **/
  void deleteWrapperFile(int32_t db, int32_t tb);

  /**
   * @brief Writes a wrapper file to a table subdir.
   **/
  void writeWrapperFile(const std::string& doc, int32_t db, int32_t tb);

  std::string getTableFileMgrPath(int32_t db, int32_t tb) const;

  /**
   * @brief Get the total size of page files (data and metadata files).  This includes
   * allocated, but unused space.
   **/
  size_t getFilesSize() const;

  /**
   * @brief Returns the total size of all subdirectory files.  Each table represented in
   * the CFM has a subdirectory for serialized data wrappers and epoch files.
   **/
  size_t getTableFileMgrsSize() const;

  /**
   * @brief an optional version of get buffer if we are not sure a chunk exists.
   **/
  std::optional<FileBuffer*> getBufferIfExists(const ChunkKey& key);

  /**
   * @brief Unlike the FileMgr, the CFM frees pages immediately instead of holding them
   * until the next checkpoint.
   **/
  void free_page(std::pair<FileInfo*, int32_t>&& page) override;

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix) override;

  // Useful for debugging.
  std::string dumpKeysWithMetadata() const;
  std::string dumpKeysWithChunkData() const;
  std::string dumpTableQueue() const { return table_evict_alg_.dumpEvictionQueue(); }

  // Used for unit testing
  void setMaxNumDataFiles(size_t max) { max_num_data_files_ = max; }
  void setMaxNumMetadataFiles(size_t max) { max_num_meta_files_ = max; }
  void setMaxWrapperSpace(size_t max) { max_wrapper_space_ = max; }
  std::set<ChunkKey> getKeysWithMetadata() const;

 private:
  /**
   * @brief Increments epoch for the given table.
   **/
  void incrementEpoch(int32_t db_id, int32_t tb_id);

  /**
   * @brief Initializes a CFM, parsing any existing files and initializing data structures
   * appropriately (currently not thread-safe).
   **/
  void init(const size_t num_reader_threads);

  /**
   * @brief Flushes epoch value to disk for a table.
   **/
  void writeAndSyncEpochToDisk(int32_t db_id, int32_t tb_id);

  /**
   * @brief Checks for any sub-directories containing
   * table-specific data and creates epochs from found files.
   **/
  void readTableFileMgrs();

  /**
   * @brief Creates a buffer and initializes it with info read from files on disk.
   **/
  FileBuffer* createBufferFromHeaders(
      const ChunkKey& key,
      const std::vector<HeaderInfo>::const_iterator& startIt,
      const std::vector<HeaderInfo>::const_iterator& endIt) override;

  /**
   * @brief Creates a buffer.
   **/
  FileBuffer* createBufferUnlocked(const ChunkKey& key,
                                   size_t pageSize = 0,
                                   const size_t numBytes = 0) override;

  /**
   * @brief Create and initialize a subdirectory for a table if none exists.
   **/
  void createTableFileMgrIfNoneExists(const int32_t db_id, const int32_t tb_id);

  /**
   * @brief Increment epochs for each table in the CFM.
   **/
  void incrementAllEpochs();

  /**
   * @brief Removes the subdirectory content for a table.
   **/
  void removeTableFileMgr(int32_t db_id, int32_t tb_id);

  /**
   * @brief Erases and cleans up all buffers for a table.
   **/
  void removeTableBuffers(int32_t db_id, int32_t tb_id);

  /**
   * @brief helper function to flush all dirty buffers to disk.
   **/
  void writeDirtyBuffers(int32_t db_id, int32_t tb_id);

  /**
   * @brief requests a free page similar to FileMgr, but this override will also evict
   * existing pages to make space if there are none available.
   **/
  Page requestFreePage(size_t pagesize, const bool isMetadata) override;

  /**
   * @brief Used to track which tables/chunks were least recently used
   **/
  void touchKey(const ChunkKey& key) const;
  void removeKey(const ChunkKey& key) const;

  /**
   * @brief returns set of keys contained in chunkIndex_ that match the given table
   * prefix.
   **/
  std::vector<ChunkKey> getKeysForTable(int32_t db_id, int32_t tb_id) const;

  /**
   * @brief evicts all metadata pages for the least recently used table.  Returns the
   * first FileInfo that a page was evicted from (guaranteed to now have at least one free
   * page in it).
   **/
  FileInfo* evictMetadataPages();

  /**
   * @brief evicts all data pages for the least recently used Chunk (metadata pages
   * persist).  Returns the first FileInfo that a page was evicted from (guaranteed to now
   * have at least one free page in it).
   **/
  FileInfo* evictPages();

  /**
   * @brief When the cache is read from disk, we don't know which chunks were least
   * recently used.  Rather than try to evict random pages to get down to size we just
   * reset the cache to make sure we have space.
   **/
  void deleteCacheIfTooLarge();

  /**
   * @brief Sets the maximum number of files/space for each type of storage based on the
   * maximum size.
   **/
  void setMaxSizes();

  FileBuffer* getBufferUnlocked(const ChunkKeyToChunkMap::iterator chunk_it,
                                const size_t numBytes = 0) override;
  ChunkKeyToChunkMap::iterator deleteBufferUnlocked(
      const ChunkKeyToChunkMap::iterator chunk_it,
      const bool purge = true) override;

  mutable mapd_shared_mutex table_dirs_mutex_;  // mutex for table_dirs_.
  // each table gest a separate epoch.  Uses pointers for move semantics.
  std::map<TablePair, std::unique_ptr<TableFileMgr>> table_dirs_;

  size_t max_num_data_files_;  // set based on max_size_.
  size_t max_num_meta_files_;  // set based on max_size_.
  size_t max_wrapper_space_;   // set based on max_size_.
  size_t max_size_;
  mutable LRUEvictionAlgorithm chunk_evict_alg_;  // last chunk touched.
  mutable LRUEvictionAlgorithm table_evict_alg_;  // last table touched.
};

}  // namespace File_Namespace
