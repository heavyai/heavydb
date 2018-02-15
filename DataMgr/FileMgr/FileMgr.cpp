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

#include "FileMgr.h"
#include "GlobalFileMgr.h"
#include "File.h"
#include "../Shared/measure.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <string>

#include <algorithm>
#include <fcntl.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include <future>

#define EPOCH_FILENAME "epoch"
#define DB_META_FILENAME "dbmeta"

using namespace std;

namespace File_Namespace {

bool headerCompare(const HeaderInfo& firstElem, const HeaderInfo& secondElem) {
  // HeaderInfo.first is a pair of Chunk key with a vector containing
  // pageId and version
  if (firstElem.chunkKey != secondElem.chunkKey)
    return firstElem.chunkKey < secondElem.chunkKey;
  if (firstElem.pageId != secondElem.pageId)
    return firstElem.pageId < secondElem.pageId;
  return firstElem.versionEpoch < secondElem.versionEpoch;

  /*
  if (firstElem.first.first != secondElem.first.first)
      return firstElem.first.first < secondElem.first.first;
  return firstElem.first.second < secondElem.first.second;
  */
}

FileMgr::FileMgr(const int deviceId,
                 GlobalFileMgr* gfm,
                 const std::pair<const int, const int> fileMgrKey,
                 const size_t num_reader_threads,
                 const int epoch,
                 const size_t defaultPageSize)
    : AbstractBufferMgr(deviceId),
      gfm_(gfm),
      fileMgrKey_(fileMgrKey),
      defaultPageSize_(defaultPageSize),
      nextFileId_(0),
      epoch_(epoch) {
  init(num_reader_threads);
}

FileMgr::FileMgr(GlobalFileMgr* gfm, const size_t defaultPageSize, std::string basePath)
    : AbstractBufferMgr(0),
      gfm_(gfm),
      fileMgrKey_(0, 0),
      fileMgrBasePath_(basePath),
      defaultPageSize_(defaultPageSize),
      nextFileId_(0),
      epoch_(-1) {
  init(basePath);
}

FileMgr::~FileMgr() {
  // checkpoint();
  // free memory used by FileInfo objects
  for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
    delete chunkIt->second;
  }
  for (auto file_info : files_) {
    delete file_info;
  }
}

void FileMgr::init(const size_t num_reader_threads) {
  // if epoch = -1 this means open from epoch file
  const std::string fileMgrDirPrefix("table");
  const std::string FileMgrDirDelim("_");
  fileMgrBasePath_ =
      (gfm_->getBasePath() + fileMgrDirPrefix + FileMgrDirDelim + std::to_string(fileMgrKey_.first) +  // db_id
       FileMgrDirDelim +
       std::to_string(fileMgrKey_.second) + "/");  // tb_id
  boost::filesystem::path path(fileMgrBasePath_);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path))
      LOG(FATAL) << "Specified path '" << fileMgrBasePath_ << "' for table data is not a directory.";
    if (epoch_ != -1) {  // if opening at previous epoch
      int epochCopy = epoch_;
      openEpochFile(EPOCH_FILENAME);
      epoch_ = epochCopy;
    } else {
      openEpochFile(EPOCH_FILENAME);
    }

    auto clock_begin = timer_start();

    boost::filesystem::directory_iterator endItr;  // default construction yields past-the-end
    int maxFileId = -1;
    int fileCount = 0;
    int threadCount = std::thread::hardware_concurrency();
    std::vector<HeaderInfo> headerVec;
    std::vector<std::future<std::vector<HeaderInfo>>> file_futures;
    for (boost::filesystem::directory_iterator fileIt(path); fileIt != endItr; ++fileIt) {
      if (boost::filesystem::is_regular_file(fileIt->status())) {
        // note that boost::filesystem leaves preceding dot on
        // extension - hence MAPD_FILE_EXT is ".mapd"
        std::string extension(fileIt->path().extension().string());

        if (extension == MAPD_FILE_EXT) {
          std::string fileStem(fileIt->path().stem().string());
          // remove trailing dot if any
          if (fileStem.size() > 0 && fileStem.back() == '.') {
            fileStem = fileStem.substr(0, fileStem.size() - 1);
          }
          size_t dotPos = fileStem.find_last_of(".");  // should only be one
          if (dotPos == std::string::npos) {
            LOG(FATAL) << "Filename does not carry page size information.";
          }
          int fileId = boost::lexical_cast<int>(fileStem.substr(0, dotPos));
          if (fileId > maxFileId) {
            maxFileId = fileId;
          }
          size_t pageSize = boost::lexical_cast<size_t>(fileStem.substr(dotPos + 1, fileStem.size()));
          std::string filePath(fileIt->path().string());
          size_t fileSize = boost::filesystem::file_size(filePath);
          assert(fileSize % pageSize == 0);  // should be no partial pages
          size_t numPages = fileSize / pageSize;

          VLOG(1) << "File id: " << fileId << " Page size: " << pageSize << " Num pages: " << numPages;

          file_futures.emplace_back(std::async(std::launch::async, [filePath, fileId, pageSize, numPages, this] {
            std::vector<HeaderInfo> tempHeaderVec;
            openExistingFile(filePath, fileId, pageSize, numPages, tempHeaderVec);
            return tempHeaderVec;
          }));
          fileCount++;
          if (fileCount % threadCount == 0) {
            processFileFutures(file_futures, headerVec);
          }
        }
      }
    }

    if (file_futures.size() > 0) {
      processFileFutures(file_futures, headerVec);
    }
    int64_t queue_time_ms = timer_stop(clock_begin);

    LOG(INFO) << "Completed Reading table's file metadata, Elapsed time : " << queue_time_ms << "ms Epoch: " << epoch_
              << " files read: " << fileCount << " table location: '" << fileMgrBasePath_ << "'";

    /* Sort headerVec so that all HeaderInfos
     * from a chunk will be grouped together
     * and in order of increasing PageId
     * - Version Epoch */

    std::sort(headerVec.begin(), headerVec.end(), headerCompare);

    /* Goal of next section is to find sequences in the
     * sorted headerVec of the same ChunkId, which we
     * can then initiate a FileBuffer with */

    VLOG(1) << "Number of Headers in Vector: " << headerVec.size();
    if (headerVec.size() > 0) {
      ChunkKey lastChunkKey = headerVec.begin()->chunkKey;
      auto startIt = headerVec.begin();

      for (auto headerIt = headerVec.begin() + 1; headerIt != headerVec.end(); ++headerIt) {
        // for (auto chunkIt = headerIt->chunkKey.begin(); chunkIt != headerIt->chunkKey.end(); ++chunkIt) {
        //    std::cout << *chunkIt << " ";
        //}

        if (headerIt->chunkKey != lastChunkKey) {
          chunkIndex_[lastChunkKey] = new FileBuffer(this, /*pageSize,*/ lastChunkKey, startIt, headerIt);
          /*
          if (startIt->versionEpoch != -1) {
              cout << "not skipping bc version != -1" << endl;
              // -1 means that chunk was deleted
              // lets not read it in
              chunkIndex_[lastChunkKey] = new FileBuffer (this,/lastChunkKey,startIt,headerIt);

          }
          else {
              cout << "Skipping bc version == -1" << endl;
          }
          */
          lastChunkKey = headerIt->chunkKey;
          startIt = headerIt;
        }
      }
      // now need to insert last Chunk
      // size_t pageSize = files_[startIt->page.fileId]->pageSize;
      // cout << "Inserting last chunk" << endl;
      // if (startIt->versionEpoch != -1) {
      chunkIndex_[lastChunkKey] = new FileBuffer(this, /*pageSize,*/ lastChunkKey, startIt, headerVec.end());
      //}
    }
    nextFileId_ = maxFileId + 1;
    // std::cout << "next file id: " << nextFileId_ << std::endl;
  } else {  // data directory does not exist
    // std::cout << basePath_ << " does not exist. Creating" << std::endl;
    if (!boost::filesystem::create_directory(path)) {
      LOG(FATAL) << "Could not create data directory";
    }
    // std::cout << basePath_ << " created." << std::endl;
    // now create epoch file
    createEpochFile(EPOCH_FILENAME);
  }

  /* define number of reader threads to be used */
  size_t num_hardware_based_threads =
      std::thread::hardware_concurrency();  // # of threads is based on # of cores on the host
  if (num_reader_threads == 0) {            // # of threads has not been defined by user
    num_reader_threads_ = num_hardware_based_threads;
  } else {
    if (num_reader_threads > num_hardware_based_threads)
      num_reader_threads_ = num_hardware_based_threads;
    else
      num_reader_threads_ = num_reader_threads;
  }
}

void FileMgr::processFileFutures(std::vector<std::future<std::vector<HeaderInfo>>>& file_futures,
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

void FileMgr::init(const std::string dataPathToConvertFrom) {
  int converted_data_epoch = 0;
  boost::filesystem::path path(dataPathToConvertFrom);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path))
      LOG(FATAL) << "Specified path is not a directory.";

    if (epoch_ != -1) {  // if opening at previous epoch
      int epochCopy = epoch_;
      openEpochFile(EPOCH_FILENAME);
      epoch_ = epochCopy;
    } else {
      openEpochFile(EPOCH_FILENAME);
    }

    boost::filesystem::directory_iterator endItr;  // default construction yields past-the-end
    int maxFileId = -1;
    int fileCount = 0;
    int threadCount = std::thread::hardware_concurrency();
    std::vector<HeaderInfo> headerVec;
    std::vector<std::future<std::vector<HeaderInfo>>> file_futures;
    for (boost::filesystem::directory_iterator fileIt(path); fileIt != endItr; ++fileIt) {
      if (boost::filesystem::is_regular_file(fileIt->status())) {
        // note that boost::filesystem leaves preceding dot on
        // extension - hence MAPD_FILE_EXT is ".mapd"
        std::string extension(fileIt->path().extension().string());

        if (extension == MAPD_FILE_EXT) {
          std::string fileStem(fileIt->path().stem().string());
          // remove trailing dot if any
          if (fileStem.size() > 0 && fileStem.back() == '.') {
            fileStem = fileStem.substr(0, fileStem.size() - 1);
          }
          size_t dotPos = fileStem.find_last_of(".");  // should only be one
          if (dotPos == std::string::npos) {
            LOG(FATAL) << "Filename does not carry page size information.";
          }
          int fileId = boost::lexical_cast<int>(fileStem.substr(0, dotPos));
          if (fileId > maxFileId) {
            maxFileId = fileId;
          }
          size_t pageSize = boost::lexical_cast<size_t>(fileStem.substr(dotPos + 1, fileStem.size()));
          std::string filePath(fileIt->path().string());
          size_t fileSize = boost::filesystem::file_size(filePath);
          assert(fileSize % pageSize == 0);  // should be no partial pages
          size_t numPages = fileSize / pageSize;

          file_futures.emplace_back(std::async(std::launch::async, [filePath, fileId, pageSize, numPages, this] {
            std::vector<HeaderInfo> tempHeaderVec;
            openExistingFile(filePath, fileId, pageSize, numPages, tempHeaderVec);
            return tempHeaderVec;
          }));
          fileCount++;
          if (fileCount % threadCount) {
            processFileFutures(file_futures, headerVec);
          }
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

      for (auto headerIt = headerVec.begin() + 1; headerIt != headerVec.end(); ++headerIt) {
        if (headerIt->chunkKey != lastChunkKey) {
          FileMgr* c_fm_ = gfm_->getFileMgr(lastChunkKey);
          FileBuffer* srcBuf = new FileBuffer(this, lastChunkKey, startIt, headerIt);
          chunkIndex_[lastChunkKey] = srcBuf;
          FileBuffer* destBuf = new FileBuffer(c_fm_, srcBuf->pageSize(), lastChunkKey);
          c_fm_->chunkIndex_[lastChunkKey] = destBuf;
          destBuf->syncEncoder(srcBuf);
          destBuf->setSize(srcBuf->size());
          destBuf->setDirty();  // this needs to be set to force writing out metadata files from "checkpoint()" call

          size_t totalNumPages = srcBuf->getMultiPage().size();
          for (size_t pageNum = 0; pageNum < totalNumPages; pageNum++) {
            Page srcPage = srcBuf->getMultiPage()[pageNum].current();
            Page destPage = c_fm_->requestFreePage(
                srcBuf->pageSize(), false);  // may modify and use api "FileBuffer::addNewMultiPage" instead
            MultiPage multiPage(srcBuf->pageSize());
            multiPage.epochs.push_back(converted_data_epoch);
            multiPage.pageVersions.push_back(destPage);
            destBuf->multiPages_.push_back(multiPage);
            size_t reservedHeaderSize = srcBuf->reservedHeaderSize();
            copyPage(srcPage, c_fm_, destPage, reservedHeaderSize, srcBuf->pageDataSize(), 0);
            destBuf->writeHeader(destPage, pageNum, converted_data_epoch, false);
          }
          lastChunkKey = headerIt->chunkKey;
          startIt = headerIt;
        }
      }

      // now need to insert last Chunk
      FileMgr* c_fm_ = gfm_->getFileMgr(lastChunkKey);
      FileBuffer* srcBuf = new FileBuffer(this, lastChunkKey, startIt, headerVec.end());
      chunkIndex_[lastChunkKey] = srcBuf;
      FileBuffer* destBuf = new FileBuffer(c_fm_, srcBuf->pageSize(), lastChunkKey);
      c_fm_->chunkIndex_[lastChunkKey] = destBuf;
      destBuf->syncEncoder(srcBuf);
      destBuf->setSize(srcBuf->size());
      destBuf->setDirty();  // this needs to be set to write out metadata file from the "checkpoint()" call

      size_t totalNumPages = srcBuf->getMultiPage().size();
      for (size_t pageNum = 0; pageNum < totalNumPages; pageNum++) {
        Page srcPage = srcBuf->getMultiPage()[pageNum].current();
        Page destPage = c_fm_->requestFreePage(srcBuf->pageSize(),
                                               false);  // may modify and use api "FileBuffer::addNewMultiPage" instead
        MultiPage multiPage(srcBuf->pageSize());
        multiPage.epochs.push_back(converted_data_epoch);
        multiPage.pageVersions.push_back(destPage);
        destBuf->multiPages_.push_back(multiPage);
        size_t reservedHeaderSize = srcBuf->reservedHeaderSize();
        copyPage(srcPage, c_fm_, destPage, reservedHeaderSize, srcBuf->pageDataSize(), 0);
        destBuf->writeHeader(destPage, pageNum, converted_data_epoch, false);
      }
    }
    nextFileId_ = maxFileId + 1;
  } else {
    if (!boost::filesystem::create_directory(path)) {
      LOG(FATAL) << "Specified path does not exist.";
    }
  }
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
  int8_t* buffer = new int8_t[numBytes];

  size_t bytesRead =
      srcFileInfo->read(srcPage.pageNum * defaultPageSize_ + offset + reservedHeaderSize, numBytes, buffer);
  CHECK(bytesRead == numBytes);
  size_t bytesWritten =
      destFileInfo->write(destPage.pageNum * defaultPageSize_ + offset + reservedHeaderSize, numBytes, buffer);
  CHECK(bytesWritten == numBytes);
  delete[] buffer;
}

void FileMgr::createEpochFile(const std::string& epochFileName) {
  std::string epochFilePath(fileMgrBasePath_ + epochFileName);
  if (boost::filesystem::exists(epochFilePath)) {
    LOG(FATAL) << "Epoch file already exists";
  }
  epochFile_ = create(epochFilePath, sizeof(int));
  // Write out current epoch to file - which if this
  // function is being called should be 0
  write(epochFile_, 0, sizeof(int), (int8_t*)&epoch_);
  epoch_++;
}

void FileMgr::openEpochFile(const std::string& epochFileName) {
  std::string epochFilePath(fileMgrBasePath_ + "/" + epochFileName);
  if (!boost::filesystem::exists(epochFilePath)) {
    LOG(FATAL) << "Epoch file does not exist";
  }
  if (!boost::filesystem::is_regular_file(epochFilePath)) {
    LOG(FATAL) << "Epoch file is not a regular file";
  }
  if (boost::filesystem::file_size(epochFilePath) < 4) {
    LOG(FATAL) << "Epoch file is not sized properly";
  }
  epochFile_ = open(epochFilePath);
  read(epochFile_, 0, sizeof(int), (int8_t*)&epoch_);
  // std::cout << "Epoch after open file: " << epoch_ << std::endl;
  epoch_++;  // we are in new epoch from last checkpoint
}

void FileMgr::writeAndSyncEpochToDisk() {
  write(epochFile_, 0, sizeof(int), (int8_t*)&epoch_);
  int status = fflush(epochFile_);
  // int status = fcntl(fileno(epochFile_),51);
  if (status != 0) {
    LOG(FATAL) << "Could not flush epoch file to disk";
  }
#ifdef __APPLE__
  status = fcntl(fileno(epochFile_), 51);
#else
  status = fsync(fileno(epochFile_));
#endif
  if (status != 0) {
    LOG(FATAL) << "Could not sync epoch file to disk";
  }
  ++epoch_;
}

void FileMgr::createDBMetaFile(const std::string& DBMetaFileName) {
  std::string DBMetaFilePath(fileMgrBasePath_ + DBMetaFileName);
  if (boost::filesystem::exists(DBMetaFilePath)) {
    LOG(FATAL) << "DB metadata file already exists.";
  }
  DBMetaFile_ = create(DBMetaFilePath, sizeof(int));
  int db_ver = getDBVersion();
  write(DBMetaFile_, 0, sizeof(int), (int8_t*)&db_ver);
  // LOG(INFO) << "DB metadata file has been created.";
}

bool FileMgr::openDBMetaFile(const std::string& DBMetaFileName) {
  std::string DBMetaFilePath(fileMgrBasePath_ + DBMetaFileName);

  if (!boost::filesystem::exists(DBMetaFilePath)) {
    // LOG(INFO) << "DB metadata file does not exist, one will be created.";
    return false;
  }
  if (!boost::filesystem::is_regular_file(DBMetaFilePath)) {
    // LOG(INFO) << "DB metadata file is not a regular file, one will be created.";
    return false;
  }
  if (boost::filesystem::file_size(DBMetaFilePath) < 4) {
    // LOG(INFO) << "DB metadata file is not sized properly, one will be created.";
    return false;
  }
  DBMetaFile_ = open(DBMetaFilePath);
  read(DBMetaFile_, 0, sizeof(int), (int8_t*)&db_version_);

  return true;
}

void FileMgr::writeAndSyncDBMetaToDisk() {
  int db_ver = getDBVersion();
  write(DBMetaFile_, 0, sizeof(int), (int8_t*)&db_ver);
  int status = fflush(DBMetaFile_);
  if (status != 0) {
    LOG(FATAL) << "Could not sync DB metadata file to disk";
  }
}

void FileMgr::checkpoint() {
  // std::cout << "Checkpointing " << epoch_ <<  std::endl;
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
    /*
    for (auto vecIt = chunkIt->first.begin(); vecIt != chunkIt->first.end(); ++vecIt) {
        std::cout << *vecIt << ",";
    }
    cout << "Is dirty: " << chunkIt->second->isDirty_ << endl;
    */
    if (chunkIt->second->isDirty_) {
      chunkIt->second->writeMetadata(epoch_);
      chunkIt->second->clearDirtyBits();
    }
  }
  chunkIndexWriteLock.unlock();

  mapd_shared_lock<mapd_shared_mutex> read_lock(files_rw_mutex_);
  for (auto fileIt = files_.begin(); fileIt != files_.end(); ++fileIt) {
    int status = (*fileIt)->syncToDisk();
    if (status != 0)
      LOG(FATAL) << "Could not sync file to disk";
  }

  writeAndSyncEpochToDisk();

  mapd_unique_lock<mapd_shared_mutex> freePagesWriteLock(mutex_free_page);
  for (auto& free_page : free_pages)
    free_page.first->freePageDeferred(free_page.second);
  free_pages.clear();
}

AbstractBuffer* FileMgr::createBuffer(const ChunkKey& key, const size_t pageSize, const size_t numBytes) {
  size_t actualPageSize = pageSize;
  if (actualPageSize == 0) {
    actualPageSize = defaultPageSize_;
  }
  /// @todo Make all accesses to chunkIndex_ thread-safe
  // we will do this lazily and not allocate space for the Chunk (i.e.
  // FileBuffer yet)
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);

  if (chunkIndex_.find(key) != chunkIndex_.end()) {
    LOG(FATAL) << "Chunk already exists.";
  }
  chunkIndex_[key] = new FileBuffer(this, actualPageSize, key, numBytes);
  chunkIndexWriteLock.unlock();
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
    LOG(FATAL) << "Chunk does not exist.";
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
         std::search(
             chunkIt->first.begin(), chunkIt->first.begin() + keyPrefix.size(), keyPrefix.begin(), keyPrefix.end()) !=
             chunkIt->first.begin() + keyPrefix.size()) {
    /*
    cout << "Freeing pages for chunk ";
    for (auto vecIt = chunkIt->first.begin(); vecIt != chunkIt->first.end(); ++vecIt) {
        std::cout << *vecIt << ",";
    }
    cout << endl;
    */
    if (purge) {
      chunkIt->second->freePages();
    }
    //@todo need a way to represent delete in non purge case
    delete chunkIt->second;
    chunkIndex_.erase(chunkIt++);
  }
}

AbstractBuffer* FileMgr::getBuffer(const ChunkKey& key, const size_t numBytes) {
  mapd_shared_lock<mapd_shared_mutex> chunkIndexReadLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.find(key);
  if (chunkIt == chunkIndex_.end()) {
    LOG(ERROR) << "Failed to get chunk " << showChunk(key);
    LOG(FATAL) << "Chunk does not exist." << showChunk(key);
  }
  return chunkIt->second;
}

void FileMgr::fetchBuffer(const ChunkKey& key, AbstractBuffer* destBuffer, const size_t numBytes) {
  // reads chunk specified by ChunkKey into AbstractBuffer provided by
  // destBuffer
  if (destBuffer->isDirty()) {
    LOG(FATAL) << " Chunk inconsitency - fetchChunk";
  }
  mapd_shared_lock<mapd_shared_mutex> chunkIndexReadLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.find(key);
  if (chunkIt == chunkIndex_.end()) {
    LOG(FATAL) << "Chunk does not exist";
  }
  chunkIndexReadLock.unlock();

  AbstractBuffer* chunk = chunkIt->second;
  // ChunkSize is either specified in function call with numBytes or we
  // just look at pageSize * numPages in FileBuffer
  size_t chunkSize = numBytes == 0 ? chunk->size() : numBytes;
  if (numBytes > 0 && numBytes > chunk->size()) {
    LOG(FATAL) << "Chunk is smaller than number of bytes requested";
  }
  destBuffer->reserve(chunkSize);
  // std::cout << "After reserve chunksize: " << chunkSize << std::endl;
  if (chunk->isUpdated()) {
    chunk->read(destBuffer->getMemoryPtr(), chunkSize, 0, destBuffer->getType(), destBuffer->getDeviceId());
  } else {
    chunk->read(destBuffer->getMemoryPtr() + destBuffer->size(),
                chunkSize - destBuffer->size(),
                destBuffer->size(),
                destBuffer->getType(),
                destBuffer->getDeviceId());
  }
  destBuffer->setSize(chunkSize);
  destBuffer->syncEncoder(chunk);
}

AbstractBuffer* FileMgr::putBuffer(const ChunkKey& key, AbstractBuffer* srcBuffer, const size_t numBytes) {
  // obtain a pointer to the Chunk
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.find(key);
  AbstractBuffer* chunk;
  if (chunkIt == chunkIndex_.end()) {
    chunk = createBuffer(key, defaultPageSize_);
  } else {
    chunk = chunkIt->second;
  }
  chunkIndexWriteLock.unlock();
  size_t oldChunkSize = chunk->size();
  // write the buffer's data to the Chunk
  // size_t newChunkSize = numBytes == 0 ? srcBuffer->size() : numBytes;
  size_t newChunkSize = numBytes == 0 ? srcBuffer->size() : numBytes;
  if (chunk->isDirty()) {
    LOG(FATAL) << "Chunk inconsistency";
  }
  if (srcBuffer->isUpdated()) {
    //@todo use dirty flags to only flush pages of chunk that need to
    // be flushed
    chunk->write((int8_t*)srcBuffer->getMemoryPtr(), newChunkSize, 0, srcBuffer->getType(), srcBuffer->getDeviceId());
  } else if (srcBuffer->isAppended()) {
    assert(oldChunkSize < newChunkSize);
    chunk->append((int8_t*)srcBuffer->getMemoryPtr() + oldChunkSize,
                  newChunkSize - oldChunkSize,
                  srcBuffer->getType(),
                  srcBuffer->getDeviceId());
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
}

void FileMgr::free(AbstractBuffer* buffer) {
  LOG(FATAL) << "Operation not supported";
}

// AbstractBuffer* FileMgr::putBuffer(AbstractBuffer *d) {
//    throw std::runtime_error("Operation not supported");
//}

Page FileMgr::requestFreePage(size_t pageSize, const bool isMetadata) {
  std::lock_guard<std::mutex> lock(getPageMutex_);

  auto candidateFiles = fileIndex_.equal_range(pageSize);
  int pageNum = -1;
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
  assert(pageNum != -1);
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
    int pageNum;
    do {
      pageNum = fileInfo->getFreePage();
      if (pageNum != -1) {
        pages.push_back(Page(fileInfo->fileId, pageNum));
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
    int pageNum;
    do {
      pageNum = fileInfo->getFreePage();
      if (pageNum != -1) {
        pages.push_back(Page(fileInfo->fileId, pageNum));
        numPagesNeeded--;
      }
    } while (pageNum != -1 && numPagesNeeded > 0);
    if (numPagesNeeded == 0) {
      break;
    }
  }
  assert(pages.size() == numPagesRequested);
}

FileInfo* FileMgr::openExistingFile(const std::string& path,
                                    const int fileId,
                                    const size_t pageSize,
                                    const size_t numPages,
                                    std::vector<HeaderInfo>& headerVec) {
  FILE* f = open(path);
  FileInfo* fInfo = new FileInfo(this, fileId, f, pageSize, numPages, false);  // false means don't init file

  fInfo->openExistingFile(headerVec, epoch_);
  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  if (fileId >= static_cast<int>(files_.size())) {
    files_.resize(fileId + 1);
  }
  files_[fileId] = fInfo;
  fileIndex_.insert(std::pair<size_t, int>(pageSize, fileId));
  return fInfo;
}

FileInfo* FileMgr::createFile(const size_t pageSize, const size_t numPages) {
  // check arguments
  if (pageSize == 0 || numPages == 0)
    LOG(FATAL) << "File creation failed: pageSize and numPages must be greater than 0.";

  // create the new file
  FILE* f = create(fileMgrBasePath_,
                   nextFileId_,
                   pageSize,
                   numPages);  // TM: not sure if I like naming scheme here - should be in separate namespace?
  if (f == nullptr)
    LOG(FATAL) << "Unable to create the new file.";

  // instantiate a new FileInfo for the newly created file
  int fileId = nextFileId_++;
  FileInfo* fInfo = new FileInfo(this, fileId, f, pageSize, numPages, true);  // true means init file
  assert(fInfo);

  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  // update file manager data structures
  files_.push_back(fInfo);
  fileIndex_.insert(std::pair<size_t, int>(pageSize, fileId));

  assert(files_.back() == fInfo);  // postcondition
  return fInfo;
}

FILE* FileMgr::getFileForFileId(const int fileId) {
  assert(fileId >= 0);
  return files_[fileId]->f;
}
/*
void FileMgr::getAllChunkMetaInfo(std::vector<std::pair<ChunkKey, int64_t> > &metadata) {
    metadata.reserve(chunkIndex_.size());
    for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
        metadata.push_back(std::make_pair(chunkIt->first, chunkIt->second->encoder->numElems));
    }
}
*/
void FileMgr::getChunkMetadataVec(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec) {
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  chunkMetadataVec.reserve(chunkIndex_.size());
  for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
    if (chunkIt->second->hasEncoder) {
      ChunkMetadata chunkMetadata;
      chunkIt->second->encoder->getMetadata(chunkMetadata);
      chunkMetadataVec.push_back(std::make_pair(chunkIt->first, chunkMetadata));
    }
  }
}

void FileMgr::getChunkMetadataVecForKeyPrefix(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
                                              const ChunkKey& keyPrefix) {
  mapd_unique_lock<mapd_shared_mutex> chunkIndexWriteLock(
      chunkIndexMutex_);  // is this guarding the right structure?  it look slike we oly read here for chunk
  auto chunkIt = chunkIndex_.lower_bound(keyPrefix);
  if (chunkIt == chunkIndex_.end()) {
    return;  // throw?
  }
  while (chunkIt != chunkIndex_.end() &&
         std::search(
             chunkIt->first.begin(), chunkIt->first.begin() + keyPrefix.size(), keyPrefix.begin(), keyPrefix.end()) !=
             chunkIt->first.begin() + keyPrefix.size()) {
    /*
    for (auto vecIt = chunkIt->first.begin(); vecIt != chunkIt->first.end(); ++vecIt) {
        std::cout << *vecIt << ",";
    }
    cout << endl;
    */
    if (chunkIt->second->hasEncoder) {
      ChunkMetadata chunkMetadata;
      chunkIt->second->encoder->getMetadata(chunkMetadata);
      chunkMetadataVec.push_back(std::make_pair(chunkIt->first, chunkMetadata));
    }
    chunkIt++;
  }
}

int FileMgr::getDBVersion() const {
  return gfm_->getDBVersion();
}

bool FileMgr::getDBConvert() const {
  return gfm_->getDBConvert();
}

void FileMgr::createTopLevelMetadata() {
  if (openDBMetaFile(DB_META_FILENAME)) {
    if (db_version_ > getDBVersion()) {
      LOG(FATAL) << "DB forward compatibility is not supported. Version of mapd software used is older than the "
                    "version of DB being read: "
                 << db_version_;
    }
  } else {
    createDBMetaFile(DB_META_FILENAME);
  }
}

void FileMgr::setEpoch(int epoch) {
  epoch_ = epoch;
  writeAndSyncEpochToDisk();
}

void FileMgr::free_page(std::pair<FileInfo*, int>&& page) {
  std::unique_lock<mapd_shared_mutex> lock(mutex_free_page);
  free_pages.push_back(page);
}

}  // File_Namespace
