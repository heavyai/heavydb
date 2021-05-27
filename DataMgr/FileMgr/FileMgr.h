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
 * @file	FileMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 *
 * This file includes the class specification for the FILE manager (FileMgr), and related
 * data structures and types.
 */

#pragma once

#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/FileMgr/Epoch.h"
#include "DataMgr/FileMgr/FileBuffer.h"
#include "DataMgr/FileMgr/FileInfo.h"
#include "DataMgr/FileMgr/Page.h"
#include "Fragmenter/FragmentDefaultValues.h"
#include "Shared/mapd_shared_mutex.h"

using namespace Data_Namespace;

namespace File_Namespace {
class GlobalFileMgr;  // forward declaration
/**
 * @type PageSizeFileMMap
 * @brief Maps logical page sizes to files.
 *
 * The file manager uses this type in order to quickly find files of a certain page size.
 * A multimap is used to associate the key (page size) with values (file identifiers of
 * files having the matching page size).
 */
using PageSizeFileMMap = std::multimap<size_t, int32_t>;

/**
 * @type Chunk
 * @brief A Chunk is the fundamental unit of execution in Map-D.
 *
 * A chunk is composed of logical pages. These pages can exist across multiple files
 * managed by the file manager.
 *
 * The collection of pages is implemented as a FileBuffer object, which is composed of a
 * vector of MultiPage objects, one for each logical page of the file buffer.
 */
using Chunk = FileBuffer;

/**
 * @type ChunkKeyToChunkMap
 * @brief Maps ChunkKeys (unique ids for Chunks) to Chunk objects.
 *
 * The file system can store multiple chunks across multiple files. With that
 * in mind, the challenge is to be able to reconstruct the pages that compose
 * a chunk upon request. A chunk key (ChunkKey) uniquely identifies a chunk,
 * and so ChunkKeyToChunkMap maps chunk keys to Chunk types, which are
 * vectors of MultiPage* pointers (logical pages).
 */
using ChunkKeyToChunkMap = std::map<ChunkKey, FileBuffer*>;
/**
 * @type TablePair
 * @breif Pair detailing the id for a database and table (first two entries in a
 * ChunkKey).
 */
using TablePair = std::pair<const int32_t, const int32_t>;

struct FileMetadata {
  int32_t file_id;
  std::string file_path;
  size_t page_size;
  size_t file_size;
  size_t num_pages;
  bool is_data_file;
};

struct StorageStats {
  int32_t epoch{0};
  int32_t epoch_floor{0};
  uint64_t metadata_file_count{0};
  uint64_t total_metadata_file_size{0};
  uint64_t total_metadata_page_count{0};
  std::optional<uint64_t> total_free_metadata_page_count{};
  uint64_t data_file_count{0};
  uint64_t total_data_file_size{0};
  uint64_t total_data_page_count{0};
  std::optional<uint64_t> total_free_data_page_count{};

  StorageStats() = default;
  StorageStats(const StorageStats& storage_stats) = default;
  virtual ~StorageStats() = default;
};

struct OpenFilesResult {
  std::vector<HeaderInfo> header_infos;
  int32_t max_file_id;
  std::string compaction_status_file_name;
};

// Page header size is serialized/deserialized as an int.
using PageHeaderSizeType = int32_t;

struct PageMapping {
  PageMapping() {}

  PageMapping(int32_t source_file_id,
              size_t source_page_num,
              PageHeaderSizeType source_page_header_size,
              int32_t destination_file_id,
              size_t destination_page_num)
      : source_file_id(source_file_id)
      , source_page_num(source_page_num)
      , source_page_header_size(source_page_header_size)
      , destination_file_id(destination_file_id)
      , destination_page_num(destination_page_num) {}

  int32_t source_file_id;
  size_t source_page_num;
  PageHeaderSizeType source_page_header_size;
  int32_t destination_file_id;
  size_t destination_page_num;
};

/**
 * @class   FileMgr
 * @brief
 */
class FileMgr : public AbstractBufferMgr {  // implements
  friend class GlobalFileMgr;

 public:
  /// Constructor
  FileMgr(const int32_t deviceId,
          GlobalFileMgr* gfm,
          const TablePair fileMgrKey,
          const int32_t max_rollback_epochs = -1,
          const size_t num_reader_threads = 0,
          const int32_t epoch = -1,
          const size_t defaultPageSize = DEFAULT_PAGE_SIZE);

  // used only to initialize enough to drop or to get basic metadata
  FileMgr(const int32_t deviceId,
          GlobalFileMgr* gfm,
          const TablePair fileMgrKey,
          const size_t defaultPageSize,
          const bool runCoreInit);

  FileMgr(GlobalFileMgr* gfm, const size_t defaultPageSize, std::string basePath);

  /// Destructor
  virtual ~FileMgr() override;

  StorageStats getStorageStats();
  /// Creates a chunk with the specified key and page size.
  FileBuffer* createBuffer(const ChunkKey& key,
                           size_t pageSize = 0,
                           const size_t numBytes = 0) override;

  bool isBufferOnDevice(const ChunkKey& key) override;
  /// Deletes the chunk with the specified key
  // Purge == true means delete the data chunks -
  // can't undelete and revert to previous
  // state - reclaims disk space for chunk
  void deleteBuffer(const ChunkKey& key, const bool purge = true) override;
  void deleteBuffersWithPrefix(const ChunkKey& keyPrefix,
                               const bool purge = true) override;

  /// Returns the a pointer to the chunk with the specified key.
  FileBuffer* getBuffer(const ChunkKey& key, const size_t numBytes = 0) override;

  void fetchBuffer(const ChunkKey& key,
                   AbstractBuffer* destBuffer,
                   const size_t numBytes) override;

  /**
   * @brief Puts the contents of d into the Chunk with the given key.
   * @param key - Unique identifier for a Chunk.
   * @param d - An object representing the source data for the Chunk.
   * @return AbstractBuffer*
   */
  FileBuffer* putBuffer(const ChunkKey& key,
                        AbstractBuffer* d,
                        const size_t numBytes = 0) override;

  // Buffer API
  AbstractBuffer* alloc(const size_t numBytes) override;
  void free(AbstractBuffer* buffer) override;
  virtual Page requestFreePage(size_t pagesize, const bool isMetadata);

  inline MgrType getMgrType() override { return FILE_MGR; };
  inline std::string getStringMgrType() override { return ToString(FILE_MGR); }
  inline std::string printSlabs() override { return "Not Implemented"; }
  inline void clearSlabs() override {}  // Noop
  inline size_t getMaxSize() override { return 0; }
  inline size_t getInUseSize() override { return 0; }
  inline size_t getAllocated() override { return 0; }
  inline bool isAllocationCapped() override { return false; }

  inline FileInfo* getFileInfoForFileId(const int32_t fileId) const {
    return files_.at(fileId);
  }

  FileMetadata getMetadataForFile(
      const boost::filesystem::directory_iterator& fileIterator);

  void init(const size_t num_reader_threads, const int32_t epochOverride);
  void init(const std::string& dataPathToConvertFrom, const int32_t epochOverride);

  void copyPage(Page& srcPage,
                FileMgr* destFileMgr,
                Page& destPage,
                const size_t reservedHeaderSize,
                const size_t numBytes,
                const size_t offset);

  /**
   * @brief Obtains free pages -- creates new files if necessary -- of the requested size.
   *
   * Given a page size and number of pages, this method updates the vector "pages"
   * to include free pages of the requested size. These pages are immediately removed
   * from the free list of the affected file(s). If there are not enough pages available
   * among current files, new files are created and their pages are included in the
   * vector.
   *
   * @param npages       The number of free pages requested
   * @param pagesize     The size of each requested page
   * @param pages        A vector containing the free pages obtained by this method
   */
  void requestFreePages(size_t npages,
                        size_t pagesize,
                        std::vector<Page>& pages,
                        const bool isMetadata);

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix) override;

  /**
   * @brief Fsyncs data files, writes out epoch and
   * fsyncs that
   */

  void checkpoint() override;
  void checkpoint(const int32_t db_id, const int32_t tb_id) override {
    LOG(FATAL) << "Operation not supported, api checkpoint() should be used instead";
  }
  /**
   * @brief Returns current value of epoch - should be
   * one greater than recorded at last checkpoint.  Because FileMgr only contains buffers
   * from one table we can just return the FileMgr's epoch instead of finding a
   * table-specific epoch.
   */
  inline virtual int32_t epoch(int32_t db_id, int32_t tb_id) const { return epoch(); }

  inline int32_t epochFloor() const { return static_cast<int32_t>(epoch_.floor()); }

  inline int32_t incrementEpoch() {
    int32_t newEpoch = epoch_.increment();
    epochIsCheckpointed_ = false;
    // We test for error here instead of in Epoch::increment so we can log FileMgr
    // metadata
    if (newEpoch > Epoch::max_allowable_epoch()) {
      LOG(FATAL) << "Epoch for table (" << fileMgrKey_.first << ", " << fileMgrKey_.second
                 << ") greater than maximum allowed value of "
                 << Epoch::max_allowable_epoch() << ".";
    }
    return newEpoch;
  }

  /**
   * @brief Returns value of epoch at last checkpoint
   */
  inline int32_t lastCheckpointedEpoch() {
    return epoch() - (epochIsCheckpointed_ ? 0 : 1);
  }

  /**
   * @brief Returns value max_rollback_epochs
   */
  inline int32_t maxRollbackEpochs() { return maxRollbackEpochs_; }

  /**
   * @brief Returns number of threads defined by parameter num-reader-threads
   * which should be used during initial load and consequent read of data.
   */
  inline size_t getNumReaderThreads() { return num_reader_threads_; }

  /**
   * @brief Returns FILE pointer associated with
   * requested fileId
   *
   * @see FileBuffer
   */

  FILE* getFileForFileId(const int32_t fileId);

  size_t getNumChunks() override;
  size_t getNumUsedMetadataPagesForChunkKey(const ChunkKey& chunkKey) const;

  ChunkKeyToChunkMap chunkIndex_;  /// Index for looking up chunks
                                   // #TM Not sure if we need this below
  int32_t getDBVersion() const;
  bool getDBConvert() const;
  void createTopLevelMetadata();  // create metadata shared by all tables of all DBs
  inline std::string getFileMgrBasePath() const { return fileMgrBasePath_; }
  virtual void closeRemovePhysical();

  void removeTableRelatedDS(const int32_t db_id, const int32_t table_id) override;

  virtual void free_page(std::pair<FileInfo*, int32_t>&& page);
  inline virtual bool hasFileMgrKey() const { return true; }
  const TablePair get_fileMgrKey() const { return fileMgrKey_; }

  inline boost::filesystem::path getFilePath(const std::string& file_name) {
    return boost::filesystem::path(fileMgrBasePath_) / file_name;
  }

  // Visible for use in unit tests.
  void writePageMappingsToStatusFile(const std::vector<PageMapping>& page_mappings);

  // Visible for use in unit tests.
  void renameCompactionStatusFile(const char* const from_status,
                                  const char* const to_status);

  void compactFiles();

  /**
   * @brief deletes or recovers a page based on last checkpointed epoch.
   **/
  virtual bool updatePageIfDeleted(FileInfo* file_info,
                                   ChunkKey& chunk_key,
                                   int32_t contingent,
                                   int32_t page_epoch,
                                   int32_t page_num);

  /**
   * @brief True if a read error should cause a fatal error.
   **/
  inline virtual bool failOnReadError() const { return true; }

  static constexpr size_t DEFAULT_NUM_PAGES_PER_DATA_FILE{256};
  static constexpr size_t DEFAULT_NUM_PAGES_PER_METADATA_FILE{4096};

  // Name of files that indicate the different statuses/phases of data compaction.
  static constexpr char const* COPY_PAGES_STATUS{"pending_data_compaction_0"};
  static constexpr char const* UPDATE_PAGE_VISIBILITY_STATUS{"pending_data_compaction_1"};
  static constexpr char const* DELETE_EMPTY_FILES_STATUS{"pending_data_compaction_2"};

  // Methods that enable override of number of pages per data/metadata file
  // for use in unit tests.
  static void setNumPagesPerDataFile(size_t num_pages);
  static void setNumPagesPerMetadataFile(size_t num_pages);

  static constexpr char LEGACY_EPOCH_FILENAME[] = "epoch";
  static constexpr char EPOCH_FILENAME[] = "epoch_metadata";
  static constexpr char DB_META_FILENAME[] = "dbmeta";
  static constexpr char FILE_MGR_VERSION_FILENAME[] = "filemgr_version";
  static constexpr int32_t INVALID_VERSION = -1;

 protected:
  // Used to initialize CachingFileMgr.
  FileMgr();

  int32_t maxRollbackEpochs_;
  std::string fileMgrBasePath_;  /// The OS file system path containing files related to
                                 /// this FileMgr
  std::map<int32_t, FileInfo*>
      files_;                   /// A map of files accessible via a file identifier.
  PageSizeFileMMap fileIndex_;  /// Maps page sizes to FileInfo objects.
  size_t num_reader_threads_;   /// number of threads used when loading data
  size_t defaultPageSize_;
  unsigned nextFileId_;  /// the index of the next file id
  int32_t db_version_;   /// DB version from dbmeta file, should be compatible with
                         /// GlobalFileMgr::omnisci_db_version_
  int32_t fileMgrVersion_;
  const int32_t latestFileMgrVersion_{1};
  FILE* DBMetaFile_ = nullptr;  /// pointer to DB level metadata
  std::mutex getPageMutex_;
  mutable mapd_shared_mutex chunkIndexMutex_;
  mutable mapd_shared_mutex files_rw_mutex_;

  mutable mapd_shared_mutex mutex_free_page_;
  std::vector<std::pair<FileInfo*, int32_t>> free_pages_;
  bool isFullyInitted_{false};

  static size_t num_pages_per_data_file_;
  static size_t num_pages_per_metadata_file_;

  /**
   * @brief Adds a file to the file manager repository.
   *
   * This method will create a FileInfo object for the file being added, and it will
   * create the corresponding file on physical disk with the indicated number of pages
   * pre-allocated.
   *
   * A pointer to the FileInfo object is returned, which itself has a file pointer (FILE*)
   * and a file identifier (int32_t fileId).
   *
   * @param fileName The name given to the file in physical storage.
   * @param pageSize The logical page size for the pages in the file.
   * @param numPages The number of logical pages to initially allocate for the file.
   * @return FileInfo* A pointer to the FileInfo object of the added file.
   */

  FileInfo* createFile(const size_t pageSize, const size_t numPages);
  FileInfo* openExistingFile(const std::string& path,
                             const int32_t fileId,
                             const size_t pageSize,
                             const size_t numPages,
                             std::vector<HeaderInfo>& headerVec);
  void createEpochFile(const std::string& epochFileName);
  int32_t openAndReadLegacyEpochFile(const std::string& epochFileName);
  void openAndReadEpochFile(const std::string& epochFileName);
  void writeAndSyncEpochToDisk();
  void setEpoch(const int32_t newEpoch);  // resets current value of epoch at startup
  int32_t readVersionFromDisk(const std::string& versionFileName) const;
  void writeAndSyncVersionToDisk(const std::string& versionFileName,
                                 const int32_t version);
  void processFileFutures(std::vector<std::future<std::vector<HeaderInfo>>>& file_futures,
                          std::vector<HeaderInfo>& headerVec);
  virtual FileBuffer* createBufferUnlocked(const ChunkKey& key,
                                           size_t pageSize = 0,
                                           const size_t numBytes = 0);
  virtual FileBuffer* createBufferFromHeaders(
      const ChunkKey& key,
      const std::vector<HeaderInfo>::const_iterator& headerStartIt,
      const std::vector<HeaderInfo>::const_iterator& headerEndIt);

  // Migration functions
  void migrateToLatestFileMgrVersion();
  void migrateEpochFileV0();

  OpenFilesResult openFiles();

  void clearFileInfos();

  // Data compaction methods
  void copySourcePageForCompaction(const Page& source_page,
                                   FileInfo* destination_file_info,
                                   std::vector<PageMapping>& page_mappings,
                                   std::set<Page>& touched_pages);
  int32_t copyPageWithoutHeaderSize(const Page& source_page,
                                    const Page& destination_page);
  void sortAndCopyFilePagesForCompaction(size_t page_size,
                                         std::vector<PageMapping>& page_mappings,
                                         std::set<Page>& touched_pages);
  void updateMappedPagesVisibility(const std::vector<PageMapping>& page_mappings);
  void deleteEmptyFiles();
  void resumeFileCompaction(const std::string& status_file_name);
  std::vector<PageMapping> readPageMappingsFromStatusFile();

  // For testing purposes only
  FileMgr(const int epoch);

  // Used to describe the manager in logging and error messages.
  virtual std::string describeSelf() const;

  void closePhysicalUnlocked();
  void syncFilesToDisk();
  void freePages();
  void initializeNumThreads(size_t num_reader_threads = 0);
  virtual FileBuffer* allocateBuffer(const size_t page_size,
                                     const ChunkKey& key,
                                     const size_t num_bytes = 0);
  virtual FileBuffer* allocateBuffer(
      const ChunkKey& key,
      const std::vector<HeaderInfo>::const_iterator& headerStartIt,
      const std::vector<HeaderInfo>::const_iterator& headerEndIt);
  virtual ChunkKeyToChunkMap::iterator deleteBufferUnlocked(
      const ChunkKeyToChunkMap::iterator chunk_it,
      const bool purge = true);
  virtual FileBuffer* getBufferUnlocked(const ChunkKeyToChunkMap::iterator chunk_it,
                                        const size_t numBytes = 0);

 private:
  void rollOffOldData(const int32_t epochCeiling, const bool shouldCheckpoint);
  void freePagesBeforeEpoch(const int32_t min_epoch);
  void freePagesBeforeEpochUnlocked(const int32_t min_epoch,
                                    const ChunkKeyToChunkMap::iterator lower_bound,
                                    const ChunkKeyToChunkMap::iterator upper_bound);
  FileBuffer* getOrCreateBuffer(const ChunkKey& key);
  /**
   * @brief Determines file path, and if exists, runs file migration and opens and reads
   * epoch file
   * @return a boolean representing whether the directory path existed
   */
  bool coreInit();
  inline int32_t epoch() const { return static_cast<int32_t>(epoch_.ceiling()); }
  void writeDirtyBuffers();

  GlobalFileMgr* gfm_;  /// Global FileMgr
  TablePair fileMgrKey_;

  Epoch epoch_;
  bool epochIsCheckpointed_ = true;
  FILE* epochFile_ = nullptr;
};

}  // namespace File_Namespace
