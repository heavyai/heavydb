/**
 * @file	FileMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 *
 * This file includes the class specification for the FILE manager (FileMgr), and related
 * data structures and types.
 */

#ifndef DATAMGR_MEMORY_FILE_FILEMGR_H
#define DATAMGR_MEMORY_FILE_FILEMGR_H

#include <iostream>
#include <map>
#include <set>
#include <mutex>

#include "Page.h"
#include "FileBuffer.h"
#include "FileInfo.h"
#include "../AbstractBuffer.h"
#include "../AbstractBufferMgr.h"

using namespace Data_Namespace;

namespace File_Namespace {

/**
 * @type PageSizeFileMMap
 * @brief Maps logical page sizes to files.
 *
 * The file manager uses this type in order to quickly find files of a certain page size.
 * A multimap is used to associate the key (page size) with values (file identifiers of files
 * having the matching page size).
 */
typedef std::multimap<size_t, int> PageSizeFileMMap;

/**
 * @type Chunk
 * @brief A Chunk is the fundamental unit of execution in Map-D.
 *
 * A chunk is composed of logical pages. These pages can exist across multiple files managed by
 * the file manager.
 *
 * The collection of pages is implemented as a FileBuffer object, which is composed of a vector of
 * MultiPage objects, one for each logical page of the file buffer.
 */
typedef FileBuffer Chunk;

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
typedef std::map<ChunkKey, FileBuffer*> ChunkKeyToChunkMap;

/**
 * @class   FileMgr
 * @brief
 */
class FileMgr : public AbstractBufferMgr {  // implements

 public:
  /// Constructor
  FileMgr(const int deviceId, std::string basePath = ".", const size_t defaultPageSize = 2097152, const int epoch = -1);

  /// Destructor
  virtual ~FileMgr();

  /// Creates a chunk with the specified key and page size.
  virtual AbstractBuffer* createBuffer(const ChunkKey& key, size_t pageSize = 0, const size_t numBytes = 0);

  virtual bool isBufferOnDevice(const ChunkKey& key);
  /// Deletes the chunk with the specified key
  // Purge == true means delete the data chunks -
  // can't undelete and revert to previous
  // state - reclaims disk space for chunk
  virtual void deleteBuffer(const ChunkKey& key, const bool purge = true);

  virtual void deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool purge = true);

  /// Returns the a pointer to the chunk with the specified key.
  virtual AbstractBuffer* getBuffer(const ChunkKey& key, const size_t numBytes = 0);

  virtual void fetchBuffer(const ChunkKey& key, AbstractBuffer* destBuffer, const size_t numBytes);

  /**
   * @brief Puts the contents of d into the Chunk with the given key.
   * @param key - Unique identifier for a Chunk.
   * @param d - An object representing the source data for the Chunk.
   * @return AbstractBuffer*
   */
  virtual AbstractBuffer* putBuffer(const ChunkKey& key, AbstractBuffer* d, const size_t numBytes = 0);

  // Buffer API
  virtual AbstractBuffer* alloc(const size_t numBytes);
  virtual void free(AbstractBuffer* buffer);
  // virtual AbstractBuffer* putBuffer(AbstractBuffer *d);
  Page requestFreePage(size_t pagesize, const bool isMetadata);

  virtual inline MgrType getMgrType() { return FILE_MGR; };
  virtual inline std::string getStringMgrType() { return ToString(FILE_MGR); }
  virtual inline std::string printSlabs() { return "Not Implemented"; }
  virtual inline size_t getMaxSize() { return 0; }
  virtual inline size_t getInUseSize() { return 0; }

  inline FileInfo* getFileInfoForFileId(const int fileId) { return files_[fileId]; }

  void init();

  /**
   * @brief Obtains free pages -- creates new files if necessary -- of the requested size.
   *
   * Given a page size and number of pages, this method updates the vector "pages"
   * to include free pages of the requested size. These pages are immediately removed
   * from the free list of the affected file(s). If there are not enough pages available
   * among current files, new files are created and their pages are included in the vector.
   *
   * @param npages       The number of free pages requested
   * @param pagesize     The size of each requested page
   * @param pages        A vector containing the free pages obtained by this method
   */
  void requestFreePages(size_t npages, size_t pagesize, std::vector<Page>& pages, const bool isMetadata);

  virtual void getChunkMetadataVec(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec);
  virtual void getChunkMetadataVecForKeyPrefix(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
                                               const ChunkKey& keyPrefix);

  /**
   * @brief Fsyncs data files, writes out epoch and
   * fsyncs that
   */

  void checkpoint();
  /**
   * @brief Returns current value of epoch - should be
   * one greater than recorded at last checkpoint
   */
  inline int epoch() { return epoch_; }

  /**
   * @brief Returns FILE pointer associated with
   * requested fileId
   *
   * @see FileBuffer
   */

  FILE* getFileForFileId(const int fileId);

  inline size_t getNumChunks() {
    // @todo should be locked - but this is more for testing now
    return chunkIndex_.size();
  }
  ChunkKeyToChunkMap chunkIndex_;  /// Index for looking up chunks
                                   // #TM Not sure if we need this below

 private:
  std::string basePath_;          /// The OS file system path containing the files.
  std::vector<FileInfo*> files_;  /// A vector of files accessible via a file identifier.
  PageSizeFileMMap fileIndex_;    /// Maps page sizes to FileInfo objects.
  size_t defaultPageSize_;
  unsigned nextFileId_;  /// the index of the next file id
  int epoch_;            /// the current epoch (time of last checkpoint)
  FILE* epochFile_;
  // bool isDirty_;                      /// true if metadata changed since last writeState()
  std::mutex getPageMutex_;
  std::mutex chunkIndexMutex_;

  /**
   * @brief Adds a file to the file manager repository.
   *
   * This method will create a FileInfo object for the file being added, and it will create
   * the corresponding file on physical disk with the indicated number of pages pre-allocated.
   *
   * A pointer to the FileInfo object is returned, which itself has a file pointer (FILE*) and
   * a file identifier (int fileId).
   *
   * @param fileName The name given to the file in physical storage.
   * @param pageSize The logical page size for the pages in the file.
   * @param numPages The number of logical pages to initially allocate for the file.
   * @return FileInfo* A pointer to the FileInfo object of the added file.
   */

  FileInfo* createFile(const size_t pageSize, const size_t numPages);
  FileInfo* openExistingFile(const std::string& path,
                             const int fileId,
                             const size_t pageSize,
                             const size_t numPages,
                             std::vector<HeaderInfo>& headerVec);
  void createEpochFile(const std::string& epochFileName);
  void openEpochFile(const std::string& epochFileName);
  void writeAndSyncEpochToDisk();
};

}  // File_Namespace

#endif  // DATAMGR_MEMORY_FILE_FILEMGR_H
