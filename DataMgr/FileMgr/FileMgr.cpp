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
 * @file        FileMgr.h
 * @author      Steven Stewart <steve@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 */

#include "DataMgr/FileMgr/FileMgr.h"

#include <fcntl.h>
#include <algorithm>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/system/error_code.hpp>

#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "Shared/File.h"
#include "Shared/checked_alloc.h"
#include "Shared/measure.h"

constexpr char LEGACY_EPOCH_FILENAME[] = "epoch";
constexpr char EPOCH_FILENAME[] = "epoch_metadata";
constexpr char DB_META_FILENAME[] = "dbmeta";
constexpr char FILE_MGR_VERSION_FILENAME[] = "filemgr_version";

constexpr int32_t INVALID_VERSION = -1;

using namespace std;

namespace File_Namespace {

bool headerCompare(const HeaderInfo& firstElem, const HeaderInfo& secondElem) {
  // HeaderInfo.first is a pair of Chunk key with a vector containing
  // pageId and version
  if (firstElem.chunkKey != secondElem.chunkKey) {
    return firstElem.chunkKey < secondElem.chunkKey;
  }
  if (firstElem.pageId != secondElem.pageId) {
    return firstElem.pageId < secondElem.pageId;
  }
  return firstElem.versionEpoch < secondElem.versionEpoch;
}

FileMgr::FileMgr(const int32_t deviceId,
                 GlobalFileMgr* gfm,
                 const std::pair<const int32_t, const int> fileMgrKey,
                 const int32_t maxRollbackEpochs,
                 const size_t num_reader_threads,
                 const int32_t epoch,
                 const size_t defaultPageSize)
    : AbstractBufferMgr(deviceId)
    , gfm_(gfm)
    , fileMgrKey_(fileMgrKey)
    , maxRollbackEpochs_(maxRollbackEpochs)
    , defaultPageSize_(defaultPageSize)
    , nextFileId_(0) {
  init(num_reader_threads, epoch);
}

// used only to initialize enough to drop
FileMgr::FileMgr(const int32_t deviceId,
                 GlobalFileMgr* gfm,
                 const std::pair<const int32_t, const int32_t> fileMgrKey,
                 const size_t defaultPageSize,
                 const bool runCoreInit)
    : AbstractBufferMgr(deviceId)
    , gfm_(gfm)
    , fileMgrKey_(fileMgrKey)
    , maxRollbackEpochs_(-1)
    , defaultPageSize_(defaultPageSize)
    , nextFileId_(0) {
  const std::string fileMgrDirPrefix("table");
  const std::string FileMgrDirDelim("_");
  fileMgrBasePath_ = (gfm_->getBasePath() + fileMgrDirPrefix + FileMgrDirDelim +
                      std::to_string(fileMgrKey_.first) +                     // db_id
                      FileMgrDirDelim + std::to_string(fileMgrKey_.second));  // tb_id
  epochFile_ = nullptr;
  files_.clear();
  if (runCoreInit) {
    coreInit();
  }
}

FileMgr::FileMgr(GlobalFileMgr* gfm, const size_t defaultPageSize, std::string basePath)
    : AbstractBufferMgr(0)
    , gfm_(gfm)
    , fileMgrKey_(0, 0)
    , maxRollbackEpochs_(-1)
    , fileMgrBasePath_(basePath)
    , defaultPageSize_(defaultPageSize)
    , nextFileId_(0) {
  init(basePath, -1);
}

// For testing purposes only
FileMgr::FileMgr(const int epoch) : AbstractBufferMgr(-1) {
  epoch_.ceiling(epoch);
}

FileMgr::~FileMgr() {
  // free memory used by FileInfo objects
  for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
    delete chunkIt->second;
  }
  for (auto file_info : files_) {
    delete file_info;
  }

  if (epochFile_) {
    close(epochFile_);
    epochFile_ = nullptr;
  }

  if (DBMetaFile_) {
    close(DBMetaFile_);
    DBMetaFile_ = nullptr;
  }
}

bool FileMgr::coreInit() {
  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  const std::string fileMgrDirPrefix("table");
  const std::string FileMgrDirDelim("_");
  fileMgrBasePath_ = (gfm_->getBasePath() + fileMgrDirPrefix + FileMgrDirDelim +
                      std::to_string(fileMgrKey_.first) +                     // db_id
                      FileMgrDirDelim + std::to_string(fileMgrKey_.second));  // tb_id
  boost::filesystem::path path(fileMgrBasePath_);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path)) {
      LOG(FATAL) << "Specified path '" << fileMgrBasePath_
                 << "' for table data is not a directory.";
    }
    migrateToLatestFileMgrVersion();
    openAndReadEpochFile(EPOCH_FILENAME);
    return true;
  }
  return false;
}

FileMetadata FileMgr::getMetadataForFile(
    const boost::filesystem::directory_iterator& fileIterator) {
  FileMetadata fileMetadata;
  fileMetadata.is_data_file = false;
  fileMetadata.file_path = fileIterator->path().string();
  if (!boost::filesystem::is_regular_file(fileIterator->status())) {
    return fileMetadata;
  }
  // note that boost::filesystem leaves preceding dot on
  // extension - hence MAPD_FILE_EXT is ".mapd"
  std::string extension(fileIterator->path().extension().string());
  if (extension == MAPD_FILE_EXT) {
    std::string fileStem(fileIterator->path().stem().string());
    // remove trailing dot if any
    if (fileStem.size() > 0 && fileStem.back() == '.') {
      fileStem = fileStem.substr(0, fileStem.size() - 1);
    }
    size_t dotPos = fileStem.find_last_of(".");  // should only be one
    if (dotPos == std::string::npos) {
      LOG(FATAL) << "File `" << fileIterator->path()
                 << "` does not carry page size information in the filename.";
    }
    fileMetadata.is_data_file = true;
    fileMetadata.file_id = boost::lexical_cast<int>(fileStem.substr(0, dotPos));
    fileMetadata.page_size =
        boost::lexical_cast<size_t>(fileStem.substr(dotPos + 1, fileStem.size()));

    fileMetadata.file_size = boost::filesystem::file_size(fileMetadata.file_path);
    CHECK_EQ(fileMetadata.file_size % fileMetadata.page_size,
             size_t(0));  // should be no partial pages
    fileMetadata.num_pages = fileMetadata.file_size / fileMetadata.page_size;
  }
  return fileMetadata;
}

void FileMgr::init(const size_t num_reader_threads, const int32_t epochOverride) {
  // if epochCeiling = -1 this means open from epoch file

  const bool dataExists = coreInit();
  if (dataExists) {
    if (epochOverride != -1) {  // if opening at specified epoch
      setEpoch(epochOverride);
    }
    auto clock_begin = timer_start();

    boost::filesystem::directory_iterator
        endItr;  // default construction yields past-the-end
    int32_t maxFileId = -1;
    int32_t fileCount = 0;
    int32_t threadCount = std::thread::hardware_concurrency();
    std::vector<HeaderInfo> headerVec;
    std::vector<std::future<std::vector<HeaderInfo>>> file_futures;
    boost::filesystem::path path(fileMgrBasePath_);
    for (boost::filesystem::directory_iterator fileIt(path); fileIt != endItr; ++fileIt) {
      FileMetadata fileMetadata = getMetadataForFile(fileIt);
      if (fileMetadata.is_data_file) {
        maxFileId = std::max(maxFileId, fileMetadata.file_id);
        file_futures.emplace_back(std::async(std::launch::async, [fileMetadata, this] {
          std::vector<HeaderInfo> tempHeaderVec;
          openExistingFile(fileMetadata.file_path,
                           fileMetadata.file_id,
                           fileMetadata.page_size,
                           fileMetadata.num_pages,
                           tempHeaderVec);
          return tempHeaderVec;
        }));
        fileCount++;
        if (fileCount % threadCount == 0) {
          processFileFutures(file_futures, headerVec);
        }
      }
    }

    if (file_futures.size() > 0) {
      processFileFutures(file_futures, headerVec);
    }
    int64_t queue_time_ms = timer_stop(clock_begin);

    LOG(INFO) << "Completed Reading table's file metadata, Elapsed time : "
              << queue_time_ms << "ms Epoch: " << epoch_.ceiling()
              << " files read: " << fileCount << " table location: '" << fileMgrBasePath_
              << "'";

    /* Sort headerVec so that all HeaderInfos
     * from a chunk will be grouped together
     * and in order of increasing PageId
     * - Version Epoch */

    std::sort(headerVec.begin(), headerVec.end(), headerCompare);

    /* Goal of next section is to find sequences in the
     * sorted headerVec of the same ChunkId, which we
     * can then initiate a FileBuffer with */

    VLOG(3) << "Number of Headers in Vector: " << headerVec.size();
    if (headerVec.size() > 0) {
      ChunkKey lastChunkKey = headerVec.begin()->chunkKey;
      auto startIt = headerVec.begin();

      for (auto headerIt = headerVec.begin() + 1; headerIt != headerVec.end();
           ++headerIt) {
        if (headerIt->chunkKey != lastChunkKey) {
          chunkIndex_[lastChunkKey] =
              new FileBuffer(this, /*pageSize,*/ lastChunkKey, startIt, headerIt);
          lastChunkKey = headerIt->chunkKey;
          startIt = headerIt;
        }
      }
      // now need to insert last Chunk
      chunkIndex_[lastChunkKey] =
          new FileBuffer(this, /*pageSize,*/ lastChunkKey, startIt, headerVec.end());
    }
    nextFileId_ = maxFileId + 1;
    rollOffOldData(epoch(),
                   true /* shouldCheckpoint - only happens if data is rolled off */);
    incrementEpoch();
    mapd_unique_lock<mapd_shared_mutex> freePagesWriteLock(mutex_free_page_);
    for (auto& free_page : free_pages_) {
      free_page.first->freePageDeferred(free_page.second);
    }
    free_pages_.clear();
  } else {
    boost::filesystem::path path(fileMgrBasePath_);
    if (!boost::filesystem::create_directory(path)) {
      LOG(FATAL) << "Could not create data directory: " << path;
    }
    fileMgrVersion_ = latestFileMgrVersion_;
    if (epochOverride != -1) {
      epoch_.floor(epochOverride);
      epoch_.ceiling(epochOverride);
    } else {
      // These are default constructor values for epoch_, but resetting here for clarity
      epoch_.floor(0);
      epoch_.ceiling(0);
    }

    createEpochFile(EPOCH_FILENAME);
    writeAndSyncEpochToDisk();
    writeAndSyncVersionToDisk(FILE_MGR_VERSION_FILENAME, fileMgrVersion_);
    incrementEpoch();
  }

  /* define number of reader threads to be used */
  size_t num_hardware_based_threads =
      std::thread::hardware_concurrency();  // # of threads is based on # of cores on the
                                            // host
  if (num_reader_threads == 0) {            // # of threads has not been defined by user
    num_reader_threads_ = num_hardware_based_threads;
  } else {
    if (num_reader_threads > num_hardware_based_threads) {
      num_reader_threads_ = num_hardware_based_threads;
    } else {
      num_reader_threads_ = num_reader_threads;
    }
  }
  isFullyInitted_ = true;
}

StorageStats FileMgr::getStorageStats() {
  StorageStats storage_stats;
  mapd_shared_lock<mapd_shared_mutex> read_lock(files_rw_mutex_);
  if (!isFullyInitted_) {
    CHECK(!fileMgrBasePath_.empty());
    boost::filesystem::path path(fileMgrBasePath_);
    if (boost::filesystem::exists(path)) {
      if (!boost::filesystem::is_directory(path)) {
        LOG(FATAL) << "getStorageStats: Specified path '" << fileMgrBasePath_
                   << "' for table data is not a directory.";
      }

      storage_stats.epoch = lastCheckpointedEpoch();
      storage_stats.epoch_floor = epochFloor();
      boost::filesystem::directory_iterator
          endItr;  // default construction yields past-the-end
      for (boost::filesystem::directory_iterator fileIt(path); fileIt != endItr;
           ++fileIt) {
        FileMetadata file_metadata = getMetadataForFile(fileIt);
        if (file_metadata.is_data_file) {
          if (file_metadata.page_size == METADATA_PAGE_SIZE) {
            storage_stats.metadata_file_count++;
            storage_stats.total_metadata_file_size += file_metadata.file_size;
            storage_stats.total_metadata_page_count += file_metadata.num_pages;
          } else if (file_metadata.page_size == defaultPageSize_) {
            storage_stats.data_file_count++;
            storage_stats.total_data_file_size += file_metadata.file_size;
            storage_stats.total_data_page_count += file_metadata.num_pages;
          } else {
            UNREACHABLE() << "Found file with unexpected page size. Page size: "
                          << file_metadata.page_size
                          << ", file path: " << file_metadata.file_path;
          }
        }
      }
    }
  } else {
    storage_stats.epoch = lastCheckpointedEpoch();
    storage_stats.epoch_floor = epochFloor();

    // We already initialized this table so take the faster path of walking through the
    // FileInfo objects and getting metadata from there
    for (const auto& file_info : files_) {
      if (file_info->pageSize == METADATA_PAGE_SIZE) {
        storage_stats.metadata_file_count++;
        storage_stats.total_metadata_file_size +=
            file_info->pageSize * file_info->numPages;
        storage_stats.total_metadata_page_count += file_info->numPages;
        if (storage_stats.total_free_metadata_page_count) {
          storage_stats.total_free_metadata_page_count.value() +=
              file_info->freePages.size();
        } else {
          storage_stats.total_free_metadata_page_count = file_info->freePages.size();
        }
      } else if (file_info->pageSize == defaultPageSize_) {
        storage_stats.data_file_count++;
        storage_stats.total_data_file_size += file_info->pageSize * file_info->numPages;
        storage_stats.total_data_page_count += file_info->numPages;
        if (storage_stats.total_free_data_page_count) {
          storage_stats.total_free_data_page_count.value() += file_info->freePages.size();
        } else {
          storage_stats.total_free_data_page_count = file_info->freePages.size();
        }
      } else {
        UNREACHABLE() << "Found file with unexpected page size. Page size: "
                      << file_info->pageSize;
      }
    }
  }
  return storage_stats;
}

void FileMgr::processFileFutures(
    std::vector<std::future<std::vector<HeaderInfo>>>& file_futures,
    std::vector<HeaderInfo>& headerVec) {
  for (auto& file_future : file_futures) {
    file_future.wait();
  }
  // concatenate the vectors after thread completes
  for (auto& file_future : file_futures) {
    auto tempHeaderVec = file_future.get();
    headerVec.insert(headerVec.end(), tempHeaderVec.begin(), tempHeaderVec.end());
  }
  file_futures.clear();
}

void FileMgr::init(const std::string& dataPathToConvertFrom,
                   const int32_t epochOverride) {
  int32_t converted_data_epoch = 0;
  boost::filesystem::path path(dataPathToConvertFrom);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path)) {
      LOG(FATAL) << "Specified path `" << path << "` is not a directory.";
    }
    openAndReadEpochFile(EPOCH_FILENAME);

    if (epochOverride != -1) {  // if opening at previous epoch
      setEpoch(epochOverride);
    }

    boost::filesystem::directory_iterator
        endItr;  // default construction yields past-the-end
    int32_t maxFileId = -1;
    int32_t fileCount = 0;
    int32_t threadCount = std::thread::hardware_concurrency();
    std::vector<HeaderInfo> headerVec;
    std::vector<std::future<std::vector<HeaderInfo>>> file_futures;
    for (boost::filesystem::directory_iterator fileIt(path); fileIt != endItr; ++fileIt) {
      FileMetadata fileMetadata = getMetadataForFile(fileIt);
      if (fileMetadata.is_data_file) {
        maxFileId = std::max(maxFileId, fileMetadata.file_id);
        file_futures.emplace_back(std::async(std::launch::async, [fileMetadata, this] {
          std::vector<HeaderInfo> tempHeaderVec;
          openExistingFile(fileMetadata.file_path,
                           fileMetadata.file_id,
                           fileMetadata.page_size,
                           fileMetadata.num_pages,
                           tempHeaderVec);
          return tempHeaderVec;
        }));
        fileCount++;
        if (fileCount % threadCount) {
          processFileFutures(file_futures, headerVec);
        }
      }
    }

    if (file_futures.size() > 0) {
      processFileFutures(file_futures, headerVec);
    }

    /* Sort headerVec so that all HeaderInfos
     * from a chunk will be grouped together
     * and in order of increasing PageId
     * - Version Epoch */

    std::sort(headerVec.begin(), headerVec.end(), headerCompare);

    /* Goal of next section is to find sequences in the
     * sorted headerVec of the same ChunkId, which we
     * can then initiate a FileBuffer with */

    if (headerVec.size() > 0) {
      ChunkKey lastChunkKey = headerVec.begin()->chunkKey;
      auto startIt = headerVec.begin();

      for (auto headerIt = headerVec.begin() + 1; headerIt != headerVec.end();
           ++headerIt) {
        if (headerIt->chunkKey != lastChunkKey) {
          FileMgr* c_fm_ =
              dynamic_cast<File_Namespace::FileMgr*>(gfm_->getFileMgr(lastChunkKey));
          CHECK(c_fm_);
          FileBuffer* srcBuf = new FileBuffer(this, lastChunkKey, startIt, headerIt);
          chunkIndex_[lastChunkKey] = srcBuf;
          FileBuffer* destBuf = new FileBuffer(c_fm_, srcBuf->pageSize(), lastChunkKey);
          c_fm_->chunkIndex_[lastChunkKey] = destBuf;
          destBuf->syncEncoder(srcBuf);
          destBuf->setSize(srcBuf->size());
          destBuf->setDirty();  // this needs to be set to force writing out metadata
                                // files from "checkpoint()" call

          size_t totalNumPages = srcBuf->getMultiPage().size();
          for (size_t pageNum = 0; pageNum < totalNumPages; pageNum++) {
            Page srcPage = srcBuf->getMultiPage()[pageNum].current().page;
            Page destPage = c_fm_->requestFreePage(
                srcBuf->pageSize(),
                false);  // may modify and use api "FileBuffer::addNewMultiPage" instead
            MultiPage multiPage(srcBuf->pageSize());
            multiPage.push(destPage, converted_data_epoch);
            destBuf->multiPages_.push_back(multiPage);
            size_t reservedHeaderSize = srcBuf->reservedHeaderSize();
            copyPage(
                srcPage, c_fm_, destPage, reservedHeaderSize, srcBuf->pageDataSize(), 0);
            destBuf->writeHeader(destPage, pageNum, converted_data_epoch, false);
          }
          lastChunkKey = headerIt->chunkKey;
          startIt = headerIt;
        }
      }

      // now need to insert last Chunk
      FileMgr* c_fm_ =
          dynamic_cast<File_Namespace::FileMgr*>(gfm_->getFileMgr(lastChunkKey));
      FileBuffer* srcBuf = new FileBuffer(this, lastChunkKey, startIt, headerVec.end());
      chunkIndex_[lastChunkKey] = srcBuf;
      FileBuffer* destBuf = new FileBuffer(c_fm_, srcBuf->pageSize(), lastChunkKey);
      c_fm_->chunkIndex_[lastChunkKey] = destBuf;
      destBuf->syncEncoder(srcBuf);
      destBuf->setSize(srcBuf->size());
      destBuf->setDirty();  // this needs to be set to write out metadata file from the
                            // "checkpoint()" call

      size_t totalNumPages = srcBuf->getMultiPage().size();
      for (size_t pageNum = 0; pageNum < totalNumPages; pageNum++) {
        Page srcPage = srcBuf->getMultiPage()[pageNum].current().page;
        Page destPage = c_fm_->requestFreePage(
            srcBuf->pageSize(),
            false);  // may modify and use api "FileBuffer::addNewMultiPage" instead
        MultiPage multiPage(srcBuf->pageSize());
        multiPage.push(destPage, converted_data_epoch);
        destBuf->multiPages_.push_back(multiPage);
        size_t reservedHeaderSize = srcBuf->reservedHeaderSize();
        copyPage(srcPage, c_fm_, destPage, reservedHeaderSize, srcBuf->pageDataSize(), 0);
        destBuf->writeHeader(destPage, pageNum, converted_data_epoch, false);
      }
    }
    nextFileId_ = maxFileId + 1;
  } else {
    if (!boost::filesystem::create_directory(path)) {
      LOG(FATAL) << "Specified path does not exist: " << path;
    }
  }
  isFullyInitted_ = true;
}

void FileMgr::closeRemovePhysical() {
  for (auto file_info : files_) {
    if (file_info->f) {
      close(file_info->f);
      file_info->f = nullptr;
    }
  }

  if (DBMetaFile_) {
    close(DBMetaFile_);
    DBMetaFile_ = nullptr;
  }

  if (epochFile_) {
    close(epochFile_);
    epochFile_ = nullptr;
  }

  /* rename for later deletion the directory containing table related data */
  File_Namespace::renameForDelete(getFileMgrBasePath());
}

void FileMgr::copyPage(Page& srcPage,
                       FileMgr* destFileMgr,
                       Page& destPage,
                       const size_t reservedHeaderSize,
                       const size_t numBytes,
                       const size_t offset) {
  CHECK(offset + numBytes <= defaultPageSize_);
  FileInfo* srcFileInfo = getFileInfoForFileId(srcPage.fileId);
  FileInfo* destFileInfo = destFileMgr->getFileInfoForFileId(destPage.fileId);
  int8_t* buffer = reinterpret_cast<int8_t*>(checked_malloc(numBytes));

  size_t bytesRead = srcFileInfo->read(
      srcPage.pageNum * defaultPageSize_ + offset + reservedHeaderSize, numBytes, buffer);
  CHECK(bytesRead == numBytes);
  size_t bytesWritten = destFileInfo->write(
      destPage.pageNum * defaultPageSize_ + offset + reservedHeaderSize,
      numBytes,
      buffer);
  CHECK(bytesWritten == numBytes);
  ::free(buffer);
}

void FileMgr::createEpochFile(const std::string& epochFileName) {
  std::string epochFilePath(fileMgrBasePath_ + "/" + epochFileName);
  if (boost::filesystem::exists(epochFilePath)) {
    LOG(FATAL) << "Epoch file `" << epochFilePath << "` already exists";
  }
  epochFile_ = create(epochFilePath, sizeof(Epoch::byte_size()));
  // Write out current epoch to file - which if this
  // function is being called should be 0
  writeAndSyncEpochToDisk();
}

int32_t FileMgr::openAndReadLegacyEpochFile(const std::string& epochFileName) {
  std::string epochFilePath(fileMgrBasePath_ + "/" + epochFileName);
  if (!boost::filesystem::exists(epochFilePath)) {
    return 0;
  }

  if (!boost::filesystem::is_regular_file(epochFilePath)) {
    LOG(FATAL) << "Epoch file `" << epochFilePath << "` is not a regular file";
  }
  if (boost::filesystem::file_size(epochFilePath) < 4) {
    LOG(FATAL) << "Epoch file `" << epochFilePath
               << "` is not sized properly (current size: "
               << boost::filesystem::file_size(epochFilePath) << ", expected size: 4)";
  }
  FILE* legacyEpochFile = open(epochFilePath);
  int32_t epoch;
  read(legacyEpochFile, 0, sizeof(int32_t), (int8_t*)&epoch);
  return epoch;
}

void FileMgr::openAndReadEpochFile(const std::string& epochFileName) {
  if (!epochFile_) {  // Check to see if already open
    std::string epochFilePath(fileMgrBasePath_ + "/" + epochFileName);
    if (!boost::filesystem::exists(epochFilePath)) {
      LOG(FATAL) << "Epoch file `" << epochFilePath << "` does not exist";
    }
    if (!boost::filesystem::is_regular_file(epochFilePath)) {
      LOG(FATAL) << "Epoch file `" << epochFilePath << "` is not a regular file";
    }
    if (boost::filesystem::file_size(epochFilePath) != 16) {
      LOG(FATAL) << "Epoch file `" << epochFilePath
                 << "` is not sized properly (current size: "
                 << boost::filesystem::file_size(epochFilePath) << ", expected size: 16)";
    }
    epochFile_ = open(epochFilePath);
  }
  read(epochFile_, 0, Epoch::byte_size(), epoch_.storage_ptr());
}

void FileMgr::writeAndSyncEpochToDisk() {
  CHECK(epochFile_);
  write(epochFile_, 0, Epoch::byte_size(), epoch_.storage_ptr());
  int32_t status = fflush(epochFile_);
  if (status != 0) {
    LOG(FATAL) << "Could not flush epoch file to disk";
  }
#ifdef __APPLE__
  status = fcntl(fileno(epochFile_), 51);
#else
  status = omnisci::fsync(fileno(epochFile_));
#endif
  if (status != 0) {
    LOG(FATAL) << "Could not sync epoch file to disk";
  }
  epochIsCheckpointed_ = true;
}

void FileMgr::freePagesBeforeEpoch(const int32_t minRollbackEpoch) {
  for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
    chunkIt->second->freePagesBeforeEpoch(minRollbackEpoch);
  }
}

void FileMgr::rollOffOldData(const int32_t epochCeiling, const bool shouldCheckpoint) {
  if (maxRollbackEpochs_ >= 0) {
    const int32_t minRollbackEpoch =
        std::max(epochCeiling - maxRollbackEpochs_, epoch_.floor());
    if (minRollbackEpoch > epoch_.floor()) {
      freePagesBeforeEpoch(minRollbackEpoch);
      epoch_.floor(minRollbackEpoch);
      if (shouldCheckpoint) {
        checkpoint();
      }
    }
  }
}

void FileMgr::checkpoint() {
  VLOG(2) << "Checkpointing table (" << fileMgrKey_.first << ", " << fileMgrKey_.second
          << " epoch: " << epoch();
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
    if (chunkIt->second->isDirty()) {
      chunkIt->second->writeMetadata(epoch());
      chunkIt->second->clearDirtyBits();
    }
  }
  chunkIndexWriteLock.unlock();

  mapd_shared_lock<mapd_shared_mutex> read_lock(files_rw_mutex_);
  for (auto fileIt = files_.begin(); fileIt != files_.end(); ++fileIt) {
    int32_t status = (*fileIt)->syncToDisk();
    if (status != 0) {
      LOG(FATAL) << "Could not sync file to disk";
    }
  }

  writeAndSyncEpochToDisk();
  incrementEpoch();
  rollOffOldData(lastCheckpointedEpoch(), false /* shouldCheckpoint */);

  mapd_unique_lock<mapd_shared_mutex> freePagesWriteLock(mutex_free_page_);
  for (auto& free_page : free_pages_) {
    free_page.first->freePageDeferred(free_page.second);
  }
  free_pages_.clear();
}

FileBuffer* FileMgr::createBuffer(const ChunkKey& key,
                                  const size_t pageSize,
                                  const size_t numBytes) {
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  return createBufferUnlocked(key, pageSize, numBytes);
}

// The underlying implementation of createBuffer needs to be lockless since
// some of the codepaths that call it will have already obtained a write lock
// and should not release it until they are complete.
FileBuffer* FileMgr::createBufferUnlocked(const ChunkKey& key,
                                          const size_t pageSize,
                                          const size_t numBytes) {
  size_t actualPageSize = pageSize;
  if (actualPageSize == 0) {
    actualPageSize = defaultPageSize_;
  }
  /// @todo Make all accesses to chunkIndex_ thread-safe
  // we will do this lazily and not allocate space for the Chunk (i.e.
  // FileBuffer yet)

  if (chunkIndex_.find(key) != chunkIndex_.end()) {
    LOG(FATAL) << "Chunk already exists for key: " << show_chunk(key);
  }
  chunkIndex_[key] = new FileBuffer(this, actualPageSize, key, numBytes);
  return (chunkIndex_[key]);
}

bool FileMgr::isBufferOnDevice(const ChunkKey& key) {
  mapd_shared_lock<mapd_shared_mutex> chunkIndexReadLock(chunkIndexMutex_);
  return chunkIndex_.find(key) != chunkIndex_.end();
}

void FileMgr::deleteBuffer(const ChunkKey& key, const bool purge) {
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.find(key);
  // ensure the Chunk exists
  if (chunkIt == chunkIndex_.end()) {
    LOG(FATAL) << "Chunk does not exist for key: " << show_chunk(key);
  }
  chunkIndexWriteLock.unlock();
  // chunkIt->second->writeMetadata(-1); // writes -1 as epoch - signifies deleted
  if (purge) {
    chunkIt->second->freePages();
  }
  //@todo need a way to represent delete in non purge case
  delete chunkIt->second;
  chunkIndex_.erase(chunkIt);
}

void FileMgr::deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool purge) {
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.lower_bound(keyPrefix);
  if (chunkIt == chunkIndex_.end()) {
    return;  // should we throw?
  }
  while (chunkIt != chunkIndex_.end() &&
         std::search(chunkIt->first.begin(),
                     chunkIt->first.begin() + keyPrefix.size(),
                     keyPrefix.begin(),
                     keyPrefix.end()) != chunkIt->first.begin() + keyPrefix.size()) {
    if (purge) {
      chunkIt->second->freePages();
    }
    //@todo need a way to represent delete in non purge case
    delete chunkIt->second;
    chunkIndex_.erase(chunkIt++);
  }
}

FileBuffer* FileMgr::getBuffer(const ChunkKey& key, const size_t numBytes) {
  mapd_shared_lock<mapd_shared_mutex> chunkIndexReadLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.find(key);
  CHECK(chunkIt != chunkIndex_.end())
      << "Chunk does not exist for key: " << show_chunk(key);
  return chunkIt->second;
}

void FileMgr::fetchBuffer(const ChunkKey& key,
                          AbstractBuffer* destBuffer,
                          const size_t numBytes) {
  // reads chunk specified by ChunkKey into AbstractBuffer provided by
  // destBuffer
  if (destBuffer->isDirty()) {
    LOG(FATAL)
        << "Aborting attempt to fetch a chunk marked dirty. Chunk inconsistency for key: "
        << show_chunk(key);
  }
  mapd_shared_lock<mapd_shared_mutex> chunkIndexReadLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.find(key);
  if (chunkIt == chunkIndex_.end()) {
    LOG(FATAL) << "Chunk does not exist for key: " << show_chunk(key);
  }
  chunkIndexReadLock.unlock();

  AbstractBuffer* chunk = chunkIt->second;
  // chunk's size is either specified in function call with numBytes or we
  // just look at pageSize * numPages in FileBuffer
  if (numBytes > 0 && numBytes > chunk->size()) {
    LOG(FATAL) << "Chunk retrieved for key `" << show_chunk(key) << "` is smaller ("
               << chunk->size() << ") than number of bytes requested (" << numBytes
               << ")";
  }

  chunk->copyTo(destBuffer, numBytes);
}

FileBuffer* FileMgr::putBuffer(const ChunkKey& key,
                               AbstractBuffer* srcBuffer,
                               const size_t numBytes) {
  // obtain a pointer to the Chunk
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.find(key);
  FileBuffer* chunk;
  if (chunkIt == chunkIndex_.end()) {
    chunk = createBufferUnlocked(key, defaultPageSize_);
  } else {
    chunk = chunkIt->second;
  }
  chunkIndexWriteLock.unlock();
  size_t oldChunkSize = chunk->size();
  // write the buffer's data to the Chunk
  // size_t newChunkSize = numBytes == 0 ? srcBuffer->size() : numBytes;
  size_t newChunkSize = numBytes == 0 ? srcBuffer->size() : numBytes;
  if (chunk->isDirty()) {
    // multiple appends are allowed,
    // but only single update is allowed
    if (srcBuffer->isUpdated() && chunk->isUpdated()) {
      LOG(FATAL) << "Aborting attempt to write a chunk marked dirty. Chunk inconsistency "
                    "for key: "
                 << show_chunk(key);
    }
  }
  if (srcBuffer->isUpdated()) {
    // chunk size is not changed when fixed rows are updated or are marked as deleted.
    // but when rows are vacuumed or varlen rows are updated (new rows are appended),
    // chunk size will change. For vacuum, checkpoint should sync size from cpu to disk.
    // For varlen update, it takes another route via fragmenter using disk-level buffer.
    if (0 == numBytes && !chunk->isDirty()) {
      chunk->setSize(newChunkSize);
    }
    //@todo use dirty flags to only flush pages of chunk that need to
    // be flushed
    chunk->write((int8_t*)srcBuffer->getMemoryPtr(),
                 newChunkSize,
                 0,
                 srcBuffer->getType(),
                 srcBuffer->getDeviceId());
  } else if (srcBuffer->isAppended()) {
    CHECK_LT(oldChunkSize, newChunkSize);
    chunk->append((int8_t*)srcBuffer->getMemoryPtr() + oldChunkSize,
                  newChunkSize - oldChunkSize,
                  srcBuffer->getType(),
                  srcBuffer->getDeviceId());
  } else {
    UNREACHABLE() << "putBuffer() expects a buffer marked is_updated or is_appended";
  }
  // chunk->clearDirtyBits(); // Hack: because write and append will set dirty bits
  //@todo commenting out line above will make sure this metadata is set
  // but will trigger error on fetch chunk
  srcBuffer->clearDirtyBits();
  chunk->syncEncoder(srcBuffer);
  return chunk;
}

AbstractBuffer* FileMgr::alloc(const size_t numBytes = 0) {
  LOG(FATAL) << "Operation not supported";
  return nullptr;  // satisfy return-type warning
}

void FileMgr::free(AbstractBuffer* buffer) {
  LOG(FATAL) << "Operation not supported";
}

Page FileMgr::requestFreePage(size_t pageSize, const bool isMetadata) {
  std::lock_guard<std::mutex> lock(getPageMutex_);

  auto candidateFiles = fileIndex_.equal_range(pageSize);
  int32_t pageNum = -1;
  for (auto fileIt = candidateFiles.first; fileIt != candidateFiles.second; ++fileIt) {
    FileInfo* fileInfo = files_[fileIt->second];
    pageNum = fileInfo->getFreePage();
    if (pageNum != -1) {
      return (Page(fileInfo->fileId, pageNum));
    }
  }
  // if here then we need to add a file
  FileInfo* fileInfo;
  if (isMetadata) {
    fileInfo = createFile(pageSize, MAX_FILE_N_METADATA_PAGES);
  } else {
    fileInfo = createFile(pageSize, MAX_FILE_N_PAGES);
  }
  pageNum = fileInfo->getFreePage();
  CHECK(pageNum != -1);
  return (Page(fileInfo->fileId, pageNum));
}

void FileMgr::requestFreePages(size_t numPagesRequested,
                               size_t pageSize,
                               std::vector<Page>& pages,
                               const bool isMetadata) {
  // not used currently
  // @todo add method to FileInfo to get more than one page
  std::lock_guard<std::mutex> lock(getPageMutex_);
  auto candidateFiles = fileIndex_.equal_range(pageSize);
  size_t numPagesNeeded = numPagesRequested;
  for (auto fileIt = candidateFiles.first; fileIt != candidateFiles.second; ++fileIt) {
    FileInfo* fileInfo = files_[fileIt->second];
    int32_t pageNum;
    do {
      pageNum = fileInfo->getFreePage();
      if (pageNum != -1) {
        pages.emplace_back(fileInfo->fileId, pageNum);
        numPagesNeeded--;
      }
    } while (pageNum != -1 && numPagesNeeded > 0);
    if (numPagesNeeded == 0) {
      break;
    }
  }
  while (numPagesNeeded > 0) {
    FileInfo* fileInfo;
    if (isMetadata) {
      fileInfo = createFile(pageSize, MAX_FILE_N_METADATA_PAGES);
    } else {
      fileInfo = createFile(pageSize, MAX_FILE_N_PAGES);
    }
    int32_t pageNum;
    do {
      pageNum = fileInfo->getFreePage();
      if (pageNum != -1) {
        pages.emplace_back(fileInfo->fileId, pageNum);
        numPagesNeeded--;
      }
    } while (pageNum != -1 && numPagesNeeded > 0);
    if (numPagesNeeded == 0) {
      break;
    }
  }
  CHECK(pages.size() == numPagesRequested);
}

FileInfo* FileMgr::openExistingFile(const std::string& path,
                                    const int fileId,
                                    const size_t pageSize,
                                    const size_t numPages,
                                    std::vector<HeaderInfo>& headerVec) {
  FILE* f = open(path);
  FileInfo* fInfo = new FileInfo(
      this, fileId, f, pageSize, numPages, false);  // false means don't init file

  fInfo->openExistingFile(headerVec, epoch());
  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  if (fileId >= static_cast<int32_t>(files_.size())) {
    files_.resize(fileId + 1);
  }
  files_[fileId] = fInfo;
  fileIndex_.insert(std::pair<size_t, int32_t>(pageSize, fileId));
  return fInfo;
}

FileInfo* FileMgr::createFile(const size_t pageSize, const size_t numPages) {
  // check arguments
  if (pageSize == 0 || numPages == 0) {
    LOG(FATAL) << "File creation failed: pageSize and numPages must be greater than 0.";
  }

  // create the new file
  FILE* f = create(fileMgrBasePath_,
                   nextFileId_,
                   pageSize,
                   numPages);  // TM: not sure if I like naming scheme here - should be in
                               // separate namespace?
  CHECK(f);

  // instantiate a new FileInfo for the newly created file
  int32_t fileId = nextFileId_++;
  FileInfo* fInfo =
      new FileInfo(this, fileId, f, pageSize, numPages, true);  // true means init file
  CHECK(fInfo);

  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  // update file manager data structures
  files_.push_back(fInfo);
  fileIndex_.insert(std::pair<size_t, int32_t>(pageSize, fileId));

  CHECK(files_.back() == fInfo);  // postcondition
  return fInfo;
}

FILE* FileMgr::getFileForFileId(const int32_t fileId) {
  CHECK(fileId >= 0 && static_cast<size_t>(fileId) < files_.size());
  return files_[fileId]->f;
}

void FileMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                              const ChunkKey& keyPrefix) {
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(
      chunkIndexMutex_);  // is this guarding the right structure?  it look slike we oly
                          // read here for chunk
  auto chunkIt = chunkIndex_.lower_bound(keyPrefix);
  if (chunkIt == chunkIndex_.end()) {
    return;  // throw?
  }
  while (chunkIt != chunkIndex_.end() &&
         std::search(chunkIt->first.begin(),
                     chunkIt->first.begin() + keyPrefix.size(),
                     keyPrefix.begin(),
                     keyPrefix.end()) != chunkIt->first.begin() + keyPrefix.size()) {
    if (chunkIt->second->hasEncoder()) {
      auto chunk_metadata = std::make_shared<ChunkMetadata>();
      chunkIt->second->encoder_->getMetadata(chunk_metadata);
      chunkMetadataVec.emplace_back(chunkIt->first, chunk_metadata);
    }
    chunkIt++;
  }
}

size_t FileMgr::getNumUsedPages() const {
  size_t num_used_pages = 0;
  mapd_shared_lock<mapd_shared_mutex> read_lock(files_rw_mutex_);
  for (const auto file : files_) {
    num_used_pages += (file->numPages - file->freePages.size());
  }
  return num_used_pages;
}

size_t FileMgr::getNumUsedMetadataPages() const {
  size_t num_used_metadata_pages = 0;
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  for (const auto& chunkIt : chunkIndex_) {
    num_used_metadata_pages += chunkIt.second->numMetadataPages();
  }
  return num_used_metadata_pages;
}

size_t FileMgr::getNumUsedMetadataPagesForChunkKey(const ChunkKey& chunkKey) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  const auto& chunkIt = chunkIndex_.find(chunkKey);
  if (chunkIt != chunkIndex_.end()) {
    return chunkIt->second->numMetadataPages();
  } else {
    throw std::runtime_error("Chunk was not found.");
  }
}

int32_t FileMgr::getDBVersion() const {
  return gfm_->getDBVersion();
}

bool FileMgr::getDBConvert() const {
  return gfm_->getDBConvert();
}

void FileMgr::createTopLevelMetadata() {
  db_version_ = readVersionFromDisk(DB_META_FILENAME);

  if (db_version_ > getDBVersion()) {
    LOG(FATAL) << "DB forward compatibility is not supported. Version of OmniSci "
                  "software used is older than the version of DB being read: "
               << db_version_;
  }
  if (db_version_ == INVALID_VERSION || db_version_ < getDBVersion()) {
    // new system, or we are moving forward versions
    // system wide migration would go here if required
    writeAndSyncVersionToDisk(DB_META_FILENAME, getDBVersion());
    return;
  }
}

int32_t FileMgr::readVersionFromDisk(const std::string& versionFileName) const {
  const std::string versionFilePath(fileMgrBasePath_ + "/" + versionFileName);
  if (!boost::filesystem::exists(versionFilePath)) {
    return -1;
  }
  if (!boost::filesystem::is_regular_file(versionFilePath)) {
    return -1;
  }
  if (boost::filesystem::file_size(versionFilePath) < 4) {
    return -1;
  }
  FILE* versionFile = open(versionFilePath);
  int32_t version;
  read(versionFile, 0, sizeof(int32_t), (int8_t*)&version);
  close(versionFile);
  return version;
}

void FileMgr::writeAndSyncVersionToDisk(const std::string& versionFileName,
                                        const int32_t version) {
  const std::string versionFilePath(fileMgrBasePath_ + "/" + versionFileName);
  FILE* versionFile;
  if (boost::filesystem::exists(versionFilePath)) {
    int32_t oldVersion = readVersionFromDisk(versionFileName);
    LOG(INFO) << "Storage version file `" << versionFilePath
              << "` already exists, its current version is " << oldVersion;
    versionFile = open(versionFilePath);
  } else {
    versionFile = create(versionFilePath, sizeof(int32_t));
  }
  write(versionFile, 0, sizeof(int32_t), (int8_t*)&version);
  int32_t status = fflush(versionFile);
  if (status != 0) {
    LOG(FATAL) << "Could not flush version file " << versionFilePath << " to disk";
  }
#ifdef __APPLE__
  status = fcntl(fileno(epochFile_), 51);
#else
  status = omnisci::fsync(fileno(versionFile));
#endif
  if (status != 0) {
    LOG(FATAL) << "Could not sync version file " << versionFilePath << " to disk";
  }
  close(versionFile);
}

void FileMgr::migrateEpochFileV0() {
  const std::string versionFilePath(fileMgrBasePath_ + "/" + FILE_MGR_VERSION_FILENAME);
  LOG(INFO) << "Migrating file format version from 0 to 1 for  `" << versionFilePath;
  epoch_.floor(Epoch::min_allowable_epoch());
  epoch_.ceiling(openAndReadLegacyEpochFile(LEGACY_EPOCH_FILENAME));
  createEpochFile(EPOCH_FILENAME);
  writeAndSyncEpochToDisk();
  int32_t migrationCompleteVersion = 1;
  writeAndSyncVersionToDisk(FILE_MGR_VERSION_FILENAME, migrationCompleteVersion);
}

void FileMgr::migrateToLatestFileMgrVersion() {
  fileMgrVersion_ = readVersionFromDisk(FILE_MGR_VERSION_FILENAME);
  if (fileMgrVersion_ == INVALID_VERSION) {
    fileMgrVersion_ = 0;
    writeAndSyncVersionToDisk(FILE_MGR_VERSION_FILENAME, fileMgrVersion_);
  } else if (fileMgrVersion_ > latestFileMgrVersion_) {
    LOG(FATAL)
        << "Table storage forward compatibility is not supported. Version of OmniSci "
           "software used is older than the version of table being read: "
        << fileMgrVersion_;
  }

  if (fileMgrVersion_ < latestFileMgrVersion_) {
    while (fileMgrVersion_ < latestFileMgrVersion_) {
      switch (fileMgrVersion_) {
        case 0: {
          migrateEpochFileV0();
          break;
        }
        default: {
          UNREACHABLE();
        }
      }
      fileMgrVersion_++;
    }
  }
}

/*
 * @brief sets the epoch to a user-specified value
 *
 * With the introduction of optional capped history on files, the possibility of
 * multiple successive rollbacks to earlier epochs means that we cannot rely on
 * the maxRollbackEpochs_ variable alone (initialized from a value stored in Catalog) to
 * guarantee that we can set an epoch at any given value. This function checks the
 * user-specified epoch value to ensure it is both not higher than the last checkpointed
 * epoch AND it is >= the epoch floor, which on an uncapped value will be the epoch the
 * table table was created at (default 0), and for capped tables, the lowest epoch for
 * which we have complete data, now materialized in the epoch metadata itself. */

void FileMgr::setEpoch(const int32_t newEpoch) {
  if (newEpoch < epoch_.floor()) {
    std::stringstream error_message;
    error_message << "Cannot set epoch for table (" << fileMgrKey_.first << ","
                  << fileMgrKey_.second << ") lower than the minimum rollback epoch ("
                  << epoch_.floor() << ").";
    throw std::runtime_error(error_message.str());
  }
  epoch_.ceiling(newEpoch);
  writeAndSyncEpochToDisk();
}

void FileMgr::free_page(std::pair<FileInfo*, int32_t>&& page) {
  std::unique_lock<mapd_shared_mutex> lock(mutex_free_page_);
  free_pages_.push_back(page);
}

void FileMgr::removeTableRelatedDS(const int32_t db_id, const int32_t table_id) {
  UNREACHABLE();
}

uint64_t FileMgr::getTotalFileSize() const {
  uint64_t total_size = 0;
  for (const auto& file : files_) {
    total_size += file->size();
  }
  if (epochFile_) {
    total_size += fileSize(epochFile_);
  }
  if (DBMetaFile_) {
    total_size += fileSize(DBMetaFile_);
  }
  return total_size;
}
}  // namespace File_Namespace
