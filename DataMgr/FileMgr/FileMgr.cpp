/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @brief
 *
 */

#include "DataMgr/FileMgr/FileMgr.h"

#include <fcntl.h>
#include <algorithm>
#include <fstream>
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
#include "Shared/scope.h"

using namespace std;

extern bool g_read_only;

namespace File_Namespace {

FileMgr::FileMgr(const int32_t device_id,
                 GlobalFileMgr* gfm,
                 const TablePair file_mgr_key,
                 const int32_t max_rollback_epochs,
                 const size_t num_reader_threads,
                 const int32_t epoch)
    : AbstractBufferMgr(device_id)
    , maxRollbackEpochs_(max_rollback_epochs)
    , nextFileId_(0)
    , gfm_(gfm)
    , fileMgrKey_(file_mgr_key)
    , page_size_(gfm->getPageSize())
    , metadata_page_size_(gfm->getMetadataPageSize()) {
  init(num_reader_threads, epoch);
}

// used only to initialize enough to drop
FileMgr::FileMgr(const int32_t device_id,
                 GlobalFileMgr* gfm,
                 const TablePair file_mgr_key,
                 const bool run_core_init)
    : AbstractBufferMgr(device_id)
    , maxRollbackEpochs_(-1)
    , nextFileId_(0)
    , gfm_(gfm)
    , fileMgrKey_(file_mgr_key)
    , page_size_(gfm->getPageSize())
    , metadata_page_size_(gfm->getMetadataPageSize()) {
  // TODO(Misiu): Standardize prefix and delim.
  const std::string fileMgrDirPrefix("table");
  const std::string fileMgrDirDelim("_");
  fileMgrBasePath_ = (gfm_->getBasePath() + fileMgrDirPrefix + fileMgrDirDelim +
                      std::to_string(fileMgrKey_.first) +                     // db_id
                      fileMgrDirDelim + std::to_string(fileMgrKey_.second));  // tb_id
  epochFile_ = nullptr;
  files_.clear();
  if (run_core_init) {
    coreInit();
  }
}

FileMgr::FileMgr(GlobalFileMgr* gfm, std::string base_path)
    : AbstractBufferMgr(0)
    , maxRollbackEpochs_(-1)
    , fileMgrBasePath_(base_path)
    , nextFileId_(0)
    , gfm_(gfm)
    , fileMgrKey_(0, 0)
    , page_size_(gfm->getPageSize())
    , metadata_page_size_(gfm->getMetadataPageSize()) {
  init(base_path, -1);
}

// For testing purposes only
FileMgr::FileMgr(const int epoch)
    : AbstractBufferMgr(-1)
    , page_size_(DEFAULT_PAGE_SIZE)
    , metadata_page_size_(DEFAULT_METADATA_PAGE_SIZE) {
  epoch_.ceiling(epoch);
}

// Used to initialize CachingFileMgr.
FileMgr::FileMgr(const size_t page_size, const size_t metadata_page_size)
    : AbstractBufferMgr(0)
    , page_size_(page_size)
    , metadata_page_size_(metadata_page_size) {}

FileMgr::~FileMgr() {
  // free memory used by FileInfo objects
  for (auto chunkIt = chunkIndex_.begin(); chunkIt != chunkIndex_.end(); ++chunkIt) {
    delete chunkIt->second;
  }

  files_.clear();

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
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(files_rw_mutex_);
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
    const boost::filesystem::directory_iterator& fileIterator) const {
  FileMetadata fileMetadata;
  fileMetadata.is_data_file = false;
  fileMetadata.file_path = fileIterator->path().string();
  if (!boost::filesystem::is_regular_file(fileIterator->status())) {
    return fileMetadata;
  }
  // note that boost::filesystem leaves preceding dot on
  // extension - hence DATA_FILE_EXT is ".data"
  std::string extension(fileIterator->path().extension().string());
  if (extension == DATA_FILE_EXT) {
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

namespace {
bool is_compaction_status_file(const std::string& file_name) {
  return (file_name == FileMgr::COPY_PAGES_STATUS ||
          file_name == FileMgr::UPDATE_PAGE_VISIBILITY_STATUS ||
          file_name == FileMgr::DELETE_EMPTY_FILES_STATUS);
}
}  // namespace

OpenFilesResult FileMgr::openFiles() {
  auto clock_begin = timer_start();
  boost::filesystem::directory_iterator
      end_itr;  // default construction yields past-the-end
  OpenFilesResult result;
  result.max_file_id = -1;
  int32_t file_count = 0;
  int32_t thread_count = std::thread::hardware_concurrency();
  std::vector<std::future<std::vector<HeaderInfo>>> file_futures;
  boost::filesystem::path path(fileMgrBasePath_);
  for (boost::filesystem::directory_iterator file_it(path); file_it != end_itr;
       ++file_it) {
    FileMetadata file_metadata = getMetadataForFile(file_it);
    if (file_metadata.is_data_file) {
      result.max_file_id = std::max(result.max_file_id, file_metadata.file_id);
      file_futures.emplace_back(std::async(std::launch::async, [file_metadata, this] {
        std::vector<HeaderInfo> temp_header_vec;
        openExistingFile(file_metadata.file_path,
                         file_metadata.file_id,
                         file_metadata.page_size,
                         file_metadata.num_pages,
                         temp_header_vec);
        return temp_header_vec;
      }));
      file_count++;
      if (file_count % thread_count == 0) {
        processFileFutures(file_futures, result.header_infos);
      }
    }

    if (is_compaction_status_file(file_it->path().filename().string())) {
      CHECK(result.compaction_status_file_name.empty());
      result.compaction_status_file_name = file_it->path().filename().string();
    }
  }

  if (file_futures.size() > 0) {
    processFileFutures(file_futures, result.header_infos);
  }

  int64_t queue_time_ms = timer_stop(clock_begin);
  LOG(INFO) << "Completed Reading table's file metadata, Elapsed time : " << queue_time_ms
            << "ms Epoch: " << epoch_.ceiling() << " files read: " << file_count
            << " table location: '" << fileMgrBasePath_ << "'";
  return result;
}

void FileMgr::clearFileInfos() {
  files_.clear();
  fileIndex_.clear();
}

void FileMgr::init(const size_t num_reader_threads, const int32_t epochOverride) {
  // if epochCeiling = -1 this means open from epoch file

  const bool dataExists = coreInit();
  if (dataExists) {
    if (epochOverride != -1) {  // if opening at specified epoch
      setEpoch(epochOverride);
    }

    auto open_files_result = openFiles();
    if (!open_files_result.compaction_status_file_name.empty()) {
      resumeFileCompaction(open_files_result.compaction_status_file_name);
      clearFileInfos();
      open_files_result = openFiles();
      CHECK(open_files_result.compaction_status_file_name.empty());
    }

    /* Sort headerVec so that all HeaderInfos
     * from a chunk will be grouped together
     * and in order of increasing PageId
     * - Version Epoch */
    auto& header_vec = open_files_result.header_infos;
    std::sort(header_vec.begin(), header_vec.end());

    /* Goal of next section is to find sequences in the
     * sorted headerVec of the same ChunkId, which we
     * can then initiate a FileBuffer with */

    VLOG(3) << "Number of Headers in Vector: " << header_vec.size();
    if (header_vec.size() > 0) {
      ChunkKey lastChunkKey = header_vec.begin()->chunkKey;
      auto startIt = header_vec.begin();

      for (auto headerIt = header_vec.begin() + 1; headerIt != header_vec.end();
           ++headerIt) {
        if (headerIt->chunkKey != lastChunkKey) {
          createBufferFromHeaders(lastChunkKey, startIt, headerIt);
          lastChunkKey = headerIt->chunkKey;
          startIt = headerIt;
        }
      }
      // now need to insert last Chunk
      createBufferFromHeaders(lastChunkKey, startIt, header_vec.end());
    }
    nextFileId_ = open_files_result.max_file_id + 1;
    rollOffOldData(epoch(), true /* only checkpoint if data is rolled off */);
    incrementEpoch();
    freePages();
  } else {
    boost::filesystem::path path(fileMgrBasePath_);
    readOnlyCheck("create file", path.string());
    if (!boost::filesystem::create_directory(path)) {
      LOG(FATAL) << "Could not create data directory: " << path;
    }
    fileMgrVersion_ = LATEST_FILE_MGR_VERSION;
    if (epochOverride != -1) {
      epoch_.floor(epochOverride);
      epoch_.ceiling(epochOverride);
    } else {
      // These are default constructor values for epoch_, but resetting here for clarity
      epoch_.floor(0);
      epoch_.ceiling(0);
    }
    createEpochFile(EPOCH_FILENAME);
    writeAndSyncVersionToDisk(FILE_MGR_VERSION_FILENAME, fileMgrVersion_);
    incrementEpoch();
  }

  initializeNumThreads(num_reader_threads);
  isFullyInitted_ = true;
}

namespace {
bool is_metadata_file(size_t file_size,
                      size_t page_size,
                      size_t metadata_page_size,
                      size_t num_pages_per_metadata_file) {
  return (file_size == (metadata_page_size * num_pages_per_metadata_file) &&
          page_size == metadata_page_size);
}
}  // namespace

StorageStats FileMgr::getStorageStats() const {
  StorageStats storage_stats;
  setDataAndMetadataFileStats(storage_stats);
  if (isFullyInitted_) {
    storage_stats.fragment_count = getFragmentCount();
  }
  return storage_stats;
}

void FileMgr::setDataAndMetadataFileStats(StorageStats& storage_stats) const {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(files_rw_mutex_);
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
          if (is_metadata_file(file_metadata.file_size,
                               file_metadata.page_size,
                               metadata_page_size_,
                               num_pages_per_metadata_file_)) {
            storage_stats.metadata_file_count++;
            storage_stats.total_metadata_file_size += file_metadata.file_size;
            storage_stats.total_metadata_page_count += file_metadata.num_pages;
          } else {
            storage_stats.data_file_count++;
            storage_stats.total_data_file_size += file_metadata.file_size;
            storage_stats.total_data_page_count += file_metadata.num_pages;
          }
        }
      }
    }
  } else {
    storage_stats.epoch = lastCheckpointedEpoch();
    storage_stats.epoch_floor = epochFloor();
    storage_stats.total_free_metadata_page_count = 0;
    storage_stats.total_free_data_page_count = 0;

    // We already initialized this table so take the faster path of walking through the
    // FileInfo objects and getting metadata from there
    for (const auto& file_info_entry : files_) {
      const auto file_info = file_info_entry.second.get();
      if (is_metadata_file(file_info->size(),
                           file_info->pageSize,
                           metadata_page_size_,
                           num_pages_per_metadata_file_)) {
        storage_stats.metadata_file_count++;
        storage_stats.total_metadata_file_size +=
            file_info->pageSize * file_info->numPages;
        storage_stats.total_metadata_page_count += file_info->numPages;
        storage_stats.total_free_metadata_page_count.value() +=
            file_info->freePages.size();
      } else {
        storage_stats.data_file_count++;
        storage_stats.total_data_file_size += file_info->pageSize * file_info->numPages;
        storage_stats.total_data_page_count += file_info->numPages;
        storage_stats.total_free_data_page_count.value() += file_info->freePages.size();
      }
    }
  }
}

uint32_t FileMgr::getFragmentCount() const {
  heavyai::shared_lock<heavyai::shared_mutex> chunk_index_read_lock(chunkIndexMutex_);
  std::set<int32_t> fragment_ids;
  for (const auto& [chunk_key, file_buffer] : chunkIndex_) {
    fragment_ids.emplace(chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
  }
  return static_cast<uint32_t>(fragment_ids.size());
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

    std::sort(headerVec.begin(), headerVec.end());

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
          auto srcBuf = createBufferFromHeaders(lastChunkKey, startIt, headerIt);
          auto destBuf = c_fm_->createBuffer(lastChunkKey, srcBuf->pageSize());
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
      auto srcBuf = createBufferFromHeaders(lastChunkKey, startIt, headerVec.end());
      auto destBuf = c_fm_->createBuffer(lastChunkKey, srcBuf->pageSize());
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
    readOnlyCheck("create file", path.string());
    if (!boost::filesystem::create_directory(path)) {
      LOG(FATAL) << "Specified path does not exist: " << path;
    }
  }
  isFullyInitted_ = true;
}

void FileMgr::closePhysicalUnlocked() {
  for (const auto& [idx, file_info] : files_) {
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
}

void FileMgr::closeRemovePhysical() {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(files_rw_mutex_);
  auto path = getFileMgrBasePath();
  readOnlyCheck("rename file", path);

  closePhysicalUnlocked();
  /* rename for later deletion the directory containing table related data */
  File_Namespace::renameForDelete(path);
}

// TODO(Misiu): This function is almost identical to FileInfo::copyPage.  Deduplicate.
void FileMgr::copyPage(Page& srcPage,
                       FileMgr* destFileMgr,
                       Page& destPage,
                       const size_t reservedHeaderSize,
                       const size_t numBytes,
                       const size_t offset) {
  CHECK(offset + numBytes <= page_size_);
  FileInfo* srcFileInfo = getFileInfoForFileId(srcPage.fileId);
  FileInfo* destFileInfo = destFileMgr->getFileInfoForFileId(destPage.fileId);
  int8_t* buffer = reinterpret_cast<int8_t*>(checked_malloc(numBytes));
  ScopeGuard guard = [&buffer] { ::free(buffer); };

  size_t bytesRead = srcFileInfo->read(
      srcPage.pageNum * page_size_ + offset + reservedHeaderSize, numBytes, buffer);
  CHECK(bytesRead == numBytes);
  size_t bytesWritten = destFileInfo->write(
      destPage.pageNum * page_size_ + offset + reservedHeaderSize, numBytes, buffer);
  CHECK(bytesWritten == numBytes);
}

void FileMgr::createEpochFile(const std::string& epochFileName) {
  std::string epochFilePath(fileMgrBasePath_ + "/" + epochFileName);
  readOnlyCheck("create file", epochFilePath);
  if (boost::filesystem::exists(epochFilePath)) {
    LOG(FATAL) << "Epoch file '" << epochFilePath << "' already exists";
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
  read(legacyEpochFile, 0, sizeof(int32_t), (int8_t*)&epoch, epochFilePath);
  close(legacyEpochFile);
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
    if (boost::filesystem::file_size(epochFilePath) != Epoch::byte_size()) {
      LOG(FATAL) << "Epoch file `" << epochFilePath
                 << "` is not sized properly (current size: "
                 << boost::filesystem::file_size(epochFilePath)
                 << ", expected size: " << Epoch::byte_size() << ")";
    }
    epochFile_ = open(epochFilePath);
  }
  read(epochFile_, 0, Epoch::byte_size(), epoch_.storage_ptr(), epochFileName);
}

void FileMgr::writeAndSyncEpochToDisk() {
  CHECK(epochFile_);
  writeFile(epochFile_, 0, Epoch::byte_size(), epoch_.storage_ptr());
  int32_t status = fflush(epochFile_);
  CHECK(status == 0) << "Could not flush epoch file to disk";
#ifdef __APPLE__
  status = fcntl(fileno(epochFile_), 51);
#else
  status = heavyai::fsync(fileno(epochFile_));
#endif
  CHECK(status == 0) << "Could not sync epoch file to disk";
  epochIsCheckpointed_ = true;
}

void FileMgr::freePagesBeforeEpoch(const int32_t min_epoch) {
  heavyai::shared_lock<heavyai::shared_mutex> chunk_index_read_lock(chunkIndexMutex_);
  freePagesBeforeEpochUnlocked(min_epoch, chunkIndex_.begin(), chunkIndex_.end());
}

void FileMgr::freePagesBeforeEpochUnlocked(
    const int32_t min_epoch,
    const ChunkKeyToChunkMap::iterator lower_bound,
    const ChunkKeyToChunkMap::iterator upper_bound) {
  for (auto chunkIt = lower_bound; chunkIt != upper_bound; ++chunkIt) {
    chunkIt->second->freePagesBeforeEpoch(min_epoch);
  }
}

void FileMgr::rollOffOldData(const int32_t epoch_ceiling, const bool should_checkpoint) {
  if (maxRollbackEpochs_ >= 0) {
    auto min_epoch = std::max(epoch_ceiling - maxRollbackEpochs_, epoch_.floor());
    if (min_epoch > epoch_.floor()) {
      freePagesBeforeEpoch(min_epoch);
      epoch_.floor(min_epoch);
      if (should_checkpoint) {
        checkpoint();
      }
    }
  }
}

std::string FileMgr::describeSelf() const {
  stringstream ss;
  ss << "table (" << fileMgrKey_.first << ", " << fileMgrKey_.second << ")";
  return ss.str();
}

void FileMgr::checkpoint() {
  VLOG(2) << "Checkpointing " << describeSelf() << " epoch: " << epoch();
  writeDirtyBuffers();
  rollOffOldData(epoch(), false /* shouldCheckpoint */);
  syncFilesToDisk();
  writeAndSyncEpochToDisk();
  incrementEpoch();
  freePages();
}

FileBuffer* FileMgr::createBuffer(const ChunkKey& key,
                                  const size_t page_size,
                                  const size_t num_bytes) {
  heavyai::unique_lock<heavyai::shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  CHECK(chunkIndex_.find(key) == chunkIndex_.end())
      << "Chunk already exists: " + show_chunk(key);
  return createBufferUnlocked(key, page_size, num_bytes);
}

// Assumes checks for pre-existing key have already occured.
FileBuffer* FileMgr::createBufferUnlocked(const ChunkKey& key,
                                          const size_t page_size,
                                          const size_t num_bytes) {
  size_t actual_page_size = page_size;
  if (actual_page_size == 0) {
    actual_page_size = page_size_;
  }
  chunkIndex_[key] = allocateBuffer(actual_page_size, key, num_bytes);
  return (chunkIndex_[key]);
}

FileBuffer* FileMgr::createBufferFromHeaders(
    const ChunkKey& key,
    const std::vector<HeaderInfo>::const_iterator& headerStartIt,
    const std::vector<HeaderInfo>::const_iterator& headerEndIt) {
  heavyai::unique_lock<heavyai::shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  CHECK(chunkIndex_.find(key) == chunkIndex_.end())
      << "Chunk already exists for key: " << show_chunk(key);
  chunkIndex_[key] = allocateBuffer(key, headerStartIt, headerEndIt);
  return (chunkIndex_[key]);
}

bool FileMgr::isBufferOnDevice(const ChunkKey& key) {
  heavyai::shared_lock<heavyai::shared_mutex> chunkIndexReadLock(chunkIndexMutex_);
  return chunkIndex_.find(key) != chunkIndex_.end();
}

void FileMgr::deleteBuffer(const ChunkKey& key, const bool purge) {
  heavyai::unique_lock<heavyai::shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  auto chunk_it = chunkIndex_.find(key);
  CHECK(chunk_it != chunkIndex_.end()) << "Chunk does not exist: " << show_chunk(key);
  deleteBufferUnlocked(chunk_it, purge);
}

ChunkKeyToChunkMap::iterator FileMgr::deleteBufferUnlocked(
    const ChunkKeyToChunkMap::iterator chunk_it,
    const bool purge) {
  if (purge) {
    chunk_it->second->freePages();
  }
  delete chunk_it->second;
  return chunkIndex_.erase(chunk_it);
}

void FileMgr::deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool purge) {
  heavyai::unique_lock<heavyai::shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  auto chunkIt = chunkIndex_.lower_bound(keyPrefix);
  if (chunkIt == chunkIndex_.end()) {
    return;  // should we throw?
  }
  while (chunkIt != chunkIndex_.end() &&
         std::search(chunkIt->first.begin(),
                     chunkIt->first.begin() + keyPrefix.size(),
                     keyPrefix.begin(),
                     keyPrefix.end()) != chunkIt->first.begin() + keyPrefix.size()) {
    deleteBufferUnlocked(chunkIt++, purge);
  }
}

FileBuffer* FileMgr::getBuffer(const ChunkKey& key, const size_t num_bytes) {
  heavyai::shared_lock<heavyai::shared_mutex> chunk_index_read_lock(chunkIndexMutex_);
  return getBufferUnlocked(key, num_bytes);
}

FileBuffer* FileMgr::getBufferUnlocked(const ChunkKey& key,
                                       const size_t num_bytes) const {
  auto chunk_it = chunkIndex_.find(key);
  CHECK(chunk_it != chunkIndex_.end()) << "Chunk does not exist: " << show_chunk(key);
  return chunk_it->second;
}

void FileMgr::fetchBuffer(const ChunkKey& key,
                          AbstractBuffer* destBuffer,
                          const size_t numBytes) {
  // reads chunk specified by ChunkKey into AbstractBuffer provided by
  // destBuffer
  CHECK(!destBuffer->isDirty())
      << "Aborting attempt to fetch a chunk marked dirty. Chunk inconsistency for key: "
      << show_chunk(key);
  AbstractBuffer* chunk = getBuffer(key);
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
  auto chunk = getOrCreateBuffer(key);
  size_t oldChunkSize = chunk->size();
  // write the buffer's data to the Chunk
  size_t newChunkSize = (numBytes == 0) ? srcBuffer->size() : numBytes;
  if (chunk->isDirty()) {
    // multiple appends are allowed,
    // but only single update is allowed
    if (srcBuffer->isUpdated() && chunk->isUpdated()) {
      LOG(FATAL) << "Aborting attempt to write a chunk marked dirty. Chunk inconsistency "
                    "for key: "
                 << show_chunk(key);
    }
  }
  CHECK(srcBuffer->isDirty()) << "putBuffer expects a dirty buffer";
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
    // If dirty buffer comes in unmarked, it must be empty.
    // Encoder sync is still required to flush the metadata.
    CHECK(numBytes == 0)
        << "Dirty buffer with size > 0 must be marked as isAppended() or isUpdated()";
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
    FileInfo* fileInfo = getFileInfoForFileId(fileIt->second);
    pageNum = fileInfo->getFreePage();
    if (pageNum != -1) {
      return (Page(fileInfo->fileId, pageNum));
    }
  }
  // if here then we need to add a file
  FileInfo* fileInfo;
  if (isMetadata) {
    fileInfo = createFileInfo(pageSize, num_pages_per_metadata_file_);
  } else {
    fileInfo = createFileInfo(pageSize, num_pages_per_data_file_);
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
    FileInfo* fileInfo = getFileInfoForFileId(fileIt->second);
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
      fileInfo = createFileInfo(pageSize, num_pages_per_metadata_file_);
    } else {
      fileInfo = createFileInfo(pageSize, num_pages_per_data_file_);
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
  auto file_info = std::make_unique<FileInfo>(this,
                                              fileId,
                                              f,
                                              pageSize,
                                              numPages,
                                              path,
                                              false);  // false means don't init file

  file_info->openExistingFile(headerVec);
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(files_rw_mutex_);
  CHECK(files_.find(fileId) == files_.end()) << "Attempting to re-open file";
  files_.emplace(fileId, std::move(file_info));
  fileIndex_.insert(std::pair<size_t, int32_t>(pageSize, fileId));
  return getFileInfoForFileId(fileId);
}

FileInfo* FileMgr::createFileInfo(const size_t pageSize, const size_t numPages) {
  readOnlyCheck("create file",
                get_data_file_path(fileMgrBasePath_, nextFileId_, pageSize));
  // check arguments
  if (pageSize == 0 || numPages == 0) {
    LOG(FATAL) << "File creation failed: pageSize and numPages must be greater than 0.";
  }

  // create the new file
  auto [f, file_path] = create(fileMgrBasePath_,
                               nextFileId_,
                               pageSize,
                               numPages);  // TM: not sure if I like naming scheme here -
                                           // should be in separate namespace?
  CHECK(f);

  // instantiate a new FileInfo for the newly created file
  int32_t fileId = nextFileId_++;
  auto fInfo = std::make_unique<FileInfo>(this,
                                          fileId,
                                          f,
                                          pageSize,
                                          numPages,
                                          file_path,
                                          true);  // true means init file
  CHECK(fInfo);

  heavyai::unique_lock<heavyai::shared_mutex> write_lock(files_rw_mutex_);
  // update file manager data structures
  files_[fileId] = std::move(fInfo);
  fileIndex_.insert(std::pair<size_t, int32_t>(pageSize, fileId));

  return getFileInfoForFileId(fileId);
}

FILE* FileMgr::getFileForFileId(const int32_t fileId) {
  CHECK(fileId >= 0);
  CHECK(files_.find(fileId) != files_.end()) << "File does not exist for id: " << fileId;
  return files_.at(fileId)->f;
}

bool FileMgr::hasChunkMetadataForKeyPrefix(const ChunkKey& key_prefix) {
  heavyai::shared_lock<heavyai::shared_mutex> chunk_index_read_lock(chunkIndexMutex_);
  auto chunk_it = chunkIndex_.lower_bound(key_prefix);
  if (chunk_it == chunkIndex_.end()) {
    return false;
  } else {
    auto it_pair =
        std::mismatch(key_prefix.begin(), key_prefix.end(), chunk_it->first.begin());
    return it_pair.first == key_prefix.end();
  }
}

void FileMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                              const ChunkKey& keyPrefix) {
  heavyai::unique_lock<heavyai::shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
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

size_t FileMgr::getNumUsedMetadataPagesForChunkKey(const ChunkKey& chunkKey) const {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(chunkIndexMutex_);
  const auto& chunkIt = chunkIndex_.find(chunkKey);
  if (chunkIt != chunkIndex_.end()) {
    return chunkIt->second->numMetadataPages();
  } else {
    throw std::runtime_error("Chunk was not found.");
  }
}

bool FileMgr::getDBConvert() const {
  return gfm_->getDBConvert();
}

void FileMgr::createOrMigrateTopLevelMetadata() {
  auto file_version = readVersionFromDisk(DB_META_FILENAME);
  auto gfm_version = gfm_->db_version_;

  if (file_version > gfm_version) {
    LOG(FATAL) << "DB forward compatibility is not supported. Version of HeavyDB "
                  "software used is older than the version of DB being read: "
               << file_version;
  }
  if (file_version == INVALID_VERSION || file_version < gfm_version) {
    // new system, or we are moving forward versions
    // system wide migration would go here if required
    writeAndSyncVersionToDisk(DB_META_FILENAME, gfm_version);
    return;
  }
}

int32_t FileMgr::readVersionFromDisk(const std::string& versionFileName) const {
  const std::string versionFilePath(fileMgrBasePath_ + "/" + versionFileName);
  if (!boost::filesystem::exists(versionFilePath) ||
      !boost::filesystem::is_regular_file(versionFilePath) ||
      boost::filesystem::file_size(versionFilePath) < 4) {
    return INVALID_VERSION;
  }
  FILE* versionFile = open(versionFilePath);
  int32_t version;
  read(versionFile, 0, sizeof(int32_t), (int8_t*)&version, versionFilePath);
  close(versionFile);
  return version;
}

void FileMgr::writeAndSyncVersionToDisk(const std::string& versionFileName,
                                        const int32_t version) {
  const std::string versionFilePath(fileMgrBasePath_ + "/" + versionFileName);
  readOnlyCheck("write file", versionFilePath);
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
  status = heavyai::fsync(fileno(versionFile));
#endif
  if (status != 0) {
    LOG(FATAL) << "Could not sync version file " << versionFilePath << " to disk";
  }
  close(versionFile);
}

void FileMgr::migrateEpochFileV0() {
  const std::string versionFilePath(fileMgrBasePath_ + "/" + FILE_MGR_VERSION_FILENAME);
  LOG(INFO) << "Migrating file format version from 0 to 1 for  `" << versionFilePath;
  readOnlyCheck("migrate epoch file", versionFilePath);
  epoch_.floor(Epoch::min_allowable_epoch());
  epoch_.ceiling(openAndReadLegacyEpochFile(LEGACY_EPOCH_FILENAME));
  createEpochFile(EPOCH_FILENAME);
  writeAndSyncEpochToDisk();
  int32_t migrationCompleteVersion = 1;
  writeAndSyncVersionToDisk(FILE_MGR_VERSION_FILENAME, migrationCompleteVersion);
}

void FileMgr::migrateLegacyFilesV1() {
  LOG(INFO) << "Migrating file format version from 1 to 2";
  readOnlyCheck("migrate file format version");
  renameAndSymlinkLegacyFiles(fileMgrBasePath_);
  constexpr int32_t migration_complete_version{2};
  writeAndSyncVersionToDisk(FILE_MGR_VERSION_FILENAME, migration_complete_version);
}

void FileMgr::renameAndSymlinkLegacyFiles(const std::string& table_data_dir) {
  std::map<boost::filesystem::path, boost::filesystem::path> old_to_new_paths;
  for (boost::filesystem::directory_iterator it(table_data_dir), end_it; it != end_it;
       it++) {
    const auto old_path = boost::filesystem::canonical(it->path());
    if (boost::filesystem::is_regular_file(it->status()) &&
        old_path.extension().string() == kLegacyDataFileExtension) {
      auto new_path = old_path;
      new_path.replace_extension(DATA_FILE_EXT);
      old_to_new_paths[old_path] = new_path;
    }
  }
  for (const auto& [old_path, new_path] : old_to_new_paths) {
    boost::filesystem::rename(old_path, new_path);
    LOG(INFO) << "Rebrand migration: Renamed " << old_path << " to " << new_path;
    boost::filesystem::create_symlink(new_path.filename(), old_path);
    LOG(INFO) << "Rebrand migration: Added symlink from " << old_path << " to "
              << new_path.filename();
  }
}

void FileMgr::migrateToLatestFileMgrVersion() {
  fileMgrVersion_ = readVersionFromDisk(FILE_MGR_VERSION_FILENAME);
  if (fileMgrVersion_ == INVALID_VERSION) {
    fileMgrVersion_ = 0;
    writeAndSyncVersionToDisk(FILE_MGR_VERSION_FILENAME, fileMgrVersion_);
  } else if (fileMgrVersion_ > LATEST_FILE_MGR_VERSION) {
    LOG(FATAL)
        << "Table storage forward compatibility is not supported. Version of HeavyDB "
           "software used is older than the version of table being read: "
        << fileMgrVersion_;
  }

  while (fileMgrVersion_ < LATEST_FILE_MGR_VERSION) {
    switch (fileMgrVersion_) {
      case 0: {
        migrateEpochFileV0();
        break;
      }
      case 1: {
        migrateLegacyFilesV1();
        break;
      }
      default: {
        UNREACHABLE();
      }
    }
    fileMgrVersion_++;
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
    error_message << "Cannot set epoch for " << describeSelf()
                  << " lower than the minimum rollback epoch (" << epoch_.floor() << ").";
    throw std::runtime_error(error_message.str());
  }
  epoch_.ceiling(newEpoch);
  writeAndSyncEpochToDisk();
}

void FileMgr::free_page(std::pair<FileInfo*, int32_t>&& page) {
  std::unique_lock<heavyai::shared_mutex> lock(mutex_free_page_);
  free_pages_.push_back(page);
}

void FileMgr::removeTableRelatedDS(const int32_t db_id, const int32_t table_id) {
  UNREACHABLE();
}

/**
 * Resumes an interrupted file compaction process. This method would
 * normally only be called when re-initializing the file manager
 * after a crash occurred in the middle of file compaction.
 */
void FileMgr::resumeFileCompaction(const std::string& status_file_name) {
  readOnlyCheck("resume file compaction", status_file_name);

  if (status_file_name == COPY_PAGES_STATUS) {
    // Delete status file and restart data compaction process
    auto file_path = getFilePath(status_file_name);
    CHECK(boost::filesystem::exists(file_path));
    boost::filesystem::remove(file_path);
    compactFiles();
  } else if (status_file_name == UPDATE_PAGE_VISIBILITY_STATUS) {
    // Execute second and third phases of data compaction
    heavyai::unique_lock<heavyai::shared_mutex> write_lock(files_rw_mutex_);
    auto page_mappings = readPageMappingsFromStatusFile();
    updateMappedPagesVisibility(page_mappings);
    renameCompactionStatusFile(UPDATE_PAGE_VISIBILITY_STATUS, DELETE_EMPTY_FILES_STATUS);
    deleteEmptyFiles();
  } else if (status_file_name == DELETE_EMPTY_FILES_STATUS) {
    // Execute last phase of data compaction
    heavyai::unique_lock<heavyai::shared_mutex> write_lock(files_rw_mutex_);
    deleteEmptyFiles();
  } else {
    UNREACHABLE() << "Unexpected status file name: " << status_file_name;
  }
}

/**
 * Compacts metadata and data file pages and deletes resulting empty
 * files (if any exists). Compaction occurs in 3 idempotent phases
 * in order to enable graceful recovery if a crash/process interruption
 * occurs in the middle data compaction.
 *
 * Phase 1:
 * Create a status file that indicates initiation of this phase. Sort
 * metadata/data files in order of files with the lowest number of free
 * pages to those with the highest number of free pages. Copy over used
 * pages from files at the end of the sorted order (files with the
 * highest number of free pages) to those at the beginning of the
 * sorted order (files with the lowest number of free pages). Keep
 * destination/copied to pages as free while copying. Keep track of
 * copied source to destination page mapping. Write page mapping to
 * the status file (to be used during crash recovery if needed).
 *
 * Phase 2:
 * Rename status file to a file name that indicates initiation of this
 * phase. Go through page mapping and mark source/copied from pages
 * as free while making the destination/copied to pages as used.
 *
 * Phase 3:
 * Rename status file to a file name that indicates initiation of this
 * phase. Delete all empty files (files containing only free pages).
 * Delete status file.
 */
void FileMgr::compactFiles() {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(files_rw_mutex_);

  readOnlyCheck("run file compaction");

  if (files_.empty()) {
    return;
  }

  auto copy_pages_status_file_path = getFilePath(COPY_PAGES_STATUS);
  CHECK(!boost::filesystem::exists(copy_pages_status_file_path));
  std::ofstream status_file(copy_pages_status_file_path.string(),
                            std::ios::out | std::ios::binary);
  status_file.close();

  std::vector<PageMapping> page_mappings;
  std::set<Page> touched_pages;
  std::set<size_t> page_sizes;
  for (const auto& [file_id, file_info] : files_) {
    page_sizes.emplace(file_info->pageSize);
  }
  for (auto page_size : page_sizes) {
    sortAndCopyFilePagesForCompaction(page_size, page_mappings, touched_pages);
  }

  writePageMappingsToStatusFile(page_mappings);
  renameCompactionStatusFile(COPY_PAGES_STATUS, UPDATE_PAGE_VISIBILITY_STATUS);

  updateMappedPagesVisibility(page_mappings);
  renameCompactionStatusFile(UPDATE_PAGE_VISIBILITY_STATUS, DELETE_EMPTY_FILES_STATUS);

  deleteEmptyFiles();
}

/**
 * Sorts all files with the given page size in ascending order of number of
 * free pages. Then copy over pages from files with more free pages to those
 * with less free pages. Leave destination/copied to pages as free when copying.
 * Record copied source and destination pages in page mapping.
 */
void FileMgr::sortAndCopyFilePagesForCompaction(size_t page_size,
                                                std::vector<PageMapping>& page_mappings,
                                                std::set<Page>& touched_pages) {
  std::vector<FileInfo*> sorted_file_infos;
  auto range = fileIndex_.equal_range(page_size);
  for (auto it = range.first; it != range.second; it++) {
    sorted_file_infos.emplace_back(files_.at(it->second).get());
  }
  if (sorted_file_infos.empty()) {
    return;
  }

  // Sort file infos in ascending order of free pages count i.e. from files with
  // the least number of free pages to those with the highest number of free pages.
  std::sort(sorted_file_infos.begin(),
            sorted_file_infos.end(),
            [](const FileInfo* file_1, const FileInfo* file_2) {
              return file_1->freePages.size() < file_2->freePages.size();
            });

  size_t destination_index = 0, source_index = sorted_file_infos.size() - 1;

  // For page copy destinations, skip files without free pages.
  while (destination_index < source_index &&
         sorted_file_infos[destination_index]->freePages.empty()) {
    destination_index++;
  }

  // For page copy sources, skip files with only free pages.
  while (destination_index < source_index &&
         sorted_file_infos[source_index]->freePages.size() ==
             sorted_file_infos[source_index]->numPages) {
    source_index--;
  }

  std::set<size_t> source_used_pages;
  CHECK(destination_index <= source_index);

  // Get the total number of free pages available for compaction
  int64_t total_free_pages{0};
  for (size_t i = destination_index; i <= source_index; i++) {
    total_free_pages += sorted_file_infos[i]->numFreePages();
  }

  while (destination_index < source_index) {
    if (source_used_pages.empty()) {
      // Populate source_used_pages with only used pages in the source file.
      auto source_file_info = sorted_file_infos[source_index];
      auto& free_pages = source_file_info->freePages;
      for (size_t page_num = 0; page_num < source_file_info->numPages; page_num++) {
        if (free_pages.find(page_num) == free_pages.end()) {
          source_used_pages.emplace(page_num);
        }
      }

      // Free pages of current source file will not be copy destinations
      total_free_pages -= source_file_info->numFreePages();
    }

    // Exit early if there are not enough free pages to empty the next file
    if (total_free_pages - static_cast<int64_t>(source_used_pages.size()) < 0) {
      return;
    }

    // Copy pages from source files to destination files
    auto dest_file_info = sorted_file_infos[destination_index];
    while (!source_used_pages.empty() && !dest_file_info->freePages.empty()) {
      // Get next page to copy
      size_t source_page_num = *source_used_pages.begin();
      source_used_pages.erase(source_page_num);

      Page source_page{sorted_file_infos[source_index]->fileId, source_page_num};
      copySourcePageForCompaction(source_page,
                                  sorted_file_infos[destination_index],
                                  page_mappings,
                                  touched_pages);
      total_free_pages--;
    }

    if (source_used_pages.empty()) {
      source_index--;
    }

    if (dest_file_info->freePages.empty()) {
      destination_index++;
    }
  }
}

/**
 * Copies a used page (indicated by the top of the source_used_pages set)
 * from the given source file to a free page in the given destination file.
 * Source and destination pages are recorded in the given page_mappings
 * vector after copying is done.
 */
void FileMgr::copySourcePageForCompaction(const Page& source_page,
                                          FileInfo* destination_file_info,
                                          std::vector<PageMapping>& page_mappings,
                                          std::set<Page>& touched_pages) {
  size_t destination_page_num = destination_file_info->getFreePage();
  CHECK_NE(destination_page_num, static_cast<size_t>(-1));
  Page destination_page{destination_file_info->fileId, destination_page_num};

  // Assert that the same pages are not copied or overridden multiple times
  CHECK(touched_pages.find(source_page) == touched_pages.end());
  touched_pages.emplace(source_page);

  CHECK(touched_pages.find(destination_page) == touched_pages.end());
  touched_pages.emplace(destination_page);

  auto header_size = copyPageWithoutHeaderSize(source_page, destination_page);
  page_mappings.emplace_back(static_cast<size_t>(source_page.fileId),
                             source_page.pageNum,
                             header_size,
                             static_cast<size_t>(destination_page.fileId),
                             destination_page.pageNum);
}

/**
 * Copies content of source_page to destination_page without copying
 * over the source_page header size. The header size is instead
 * returned by the method. Not copying over the header size
 * enables a use case where destination_page has all the content of
 * the source_page but is still marked as a free page.
 */
int32_t FileMgr::copyPageWithoutHeaderSize(const Page& source_page,
                                           const Page& destination_page) {
  FileInfo* source_file_info = getFileInfoForFileId(source_page.fileId);
  CHECK(source_file_info);
  CHECK_EQ(source_file_info->fileId, source_page.fileId);

  FileInfo* destination_file_info = getFileInfoForFileId(destination_page.fileId);
  CHECK(destination_file_info);
  CHECK_EQ(destination_file_info->fileId, destination_page.fileId);
  CHECK_EQ(source_file_info->pageSize, destination_file_info->pageSize);

  auto page_size = source_file_info->pageSize;
  auto buffer = std::make_unique<int8_t[]>(page_size);
  size_t bytes_read =
      source_file_info->read(source_page.pageNum * page_size, page_size, buffer.get());
  CHECK_EQ(page_size, bytes_read);

  auto header_size_offset = sizeof(int32_t);
  size_t bytes_written = destination_file_info->write(
      (destination_page.pageNum * page_size) + header_size_offset,
      page_size - header_size_offset,
      buffer.get() + header_size_offset);
  CHECK_EQ(page_size - header_size_offset, bytes_written);
  return reinterpret_cast<int32_t*>(buffer.get())[0];
}

/**
 * Goes through the given page mapping and marks source/copied from pages as free
 * while marking destination/copied to pages as used (by setting the header size).
 */
void FileMgr::updateMappedPagesVisibility(const std::vector<PageMapping>& page_mappings) {
  for (const auto& page_mapping : page_mappings) {
    auto destination_file = getFileInfoForFileId(page_mapping.destination_file_id);

    // Set destination page header size
    auto header_size = page_mapping.source_page_header_size;
    CHECK_GT(header_size, 0);
    destination_file->write(
        page_mapping.destination_page_num * destination_file->pageSize,
        sizeof(PageHeaderSizeType),
        reinterpret_cast<int8_t*>(&header_size));
    auto source_file = getFileInfoForFileId(page_mapping.source_file_id);

    // Free source page
    PageHeaderSizeType free_page_header_size{0};
    source_file->write(page_mapping.source_page_num * source_file->pageSize,
                       sizeof(PageHeaderSizeType),
                       reinterpret_cast<int8_t*>(&free_page_header_size));
    source_file->freePageDeferred(page_mapping.source_page_num);
  }

  for (const auto& file_info_entry : files_) {
    int32_t status = file_info_entry.second->syncToDisk();
    if (status != 0) {
      LOG(FATAL) << "Could not sync file to disk";
    }
  }
}

/**
 * Deletes files that contain only free pages. Also deletes the compaction
 * status file.
 */
void FileMgr::deleteEmptyFiles() {
  for (const auto& [file_id, file_info] : files_) {
    CHECK_EQ(file_id, file_info->fileId);
    if (file_info->freePages.size() == file_info->numPages) {
      fclose(file_info->f);
      file_info->f = nullptr;
      auto file_path = get_data_file_path(fileMgrBasePath_, file_id, file_info->pageSize);
      readOnlyCheck("delete file", file_path);
      boost::filesystem::remove(get_legacy_data_file_path(file_path));
      boost::filesystem::remove(file_path);
    }
  }

  auto status_file_path = getFilePath(DELETE_EMPTY_FILES_STATUS);
  CHECK(boost::filesystem::exists(status_file_path));
  readOnlyCheck("delete file", status_file_path.string());
  boost::filesystem::remove(status_file_path);
}

/**
 * Serializes a page mapping vector to expected status file. Page
 * mapping vector is serialized in the following format:
 * [{page mapping vector size}, {page mapping vector data bytes ...}]
 */
void FileMgr::writePageMappingsToStatusFile(
    const std::vector<PageMapping>& page_mappings) {
  auto file_path = getFilePath(COPY_PAGES_STATUS);

  readOnlyCheck("write file", file_path.string());

  CHECK(boost::filesystem::exists(file_path));
  CHECK(boost::filesystem::is_empty(file_path));
  std::ofstream status_file{file_path.string(), std::ios::out | std::ios::binary};
  int64_t page_mappings_count = page_mappings.size();
  status_file.write(reinterpret_cast<const char*>(&page_mappings_count), sizeof(int64_t));
  status_file.write(reinterpret_cast<const char*>(page_mappings.data()),
                    page_mappings_count * sizeof(PageMapping));
  status_file.close();
}

/**
 * Deserializes a page mapping vector from expected status file.
 */
std::vector<PageMapping> FileMgr::readPageMappingsFromStatusFile() {
  auto file_path = getFilePath(UPDATE_PAGE_VISIBILITY_STATUS);
  CHECK(boost::filesystem::exists(file_path));
  std::ifstream status_file{file_path.string(),
                            std::ios::in | std::ios::binary | std::ios::ate};
  CHECK(status_file.is_open());
  size_t file_size = status_file.tellg();
  status_file.seekg(0, std::ios::beg);
  CHECK_GE(file_size, sizeof(int64_t));

  int64_t page_mappings_count;
  status_file.read(reinterpret_cast<char*>(&page_mappings_count), sizeof(int64_t));
  auto page_mappings_byte_size = file_size - sizeof(int64_t);
  CHECK_EQ(page_mappings_byte_size % sizeof(PageMapping), static_cast<size_t>(0));
  CHECK_EQ(static_cast<size_t>(page_mappings_count),
           page_mappings_byte_size / sizeof(PageMapping));

  std::vector<PageMapping> page_mappings(page_mappings_count);
  status_file.read(reinterpret_cast<char*>(page_mappings.data()),
                   page_mappings_byte_size);
  status_file.close();
  return page_mappings;
}

/**
 * Renames a given status file name to a new given file name.
 */
void FileMgr::renameCompactionStatusFile(const char* const from_status,
                                         const char* const to_status) {
  auto from_status_file_path = getFilePath(from_status);
  auto to_status_file_path = getFilePath(to_status);

  readOnlyCheck("write file", from_status_file_path.string());

  CHECK(boost::filesystem::exists(from_status_file_path));
  CHECK(!boost::filesystem::exists(to_status_file_path));
  boost::filesystem::rename(from_status_file_path, to_status_file_path);
}

// Methods that enable override of number of pages per data/metadata files
// for use in unit tests.
void FileMgr::setNumPagesPerDataFile(size_t num_pages) {
  num_pages_per_data_file_ = num_pages;
}

void FileMgr::setNumPagesPerMetadataFile(size_t num_pages) {
  num_pages_per_metadata_file_ = num_pages;
}

void FileMgr::syncFilesToDisk() {
  heavyai::shared_lock<heavyai::shared_mutex> files_read_lock(files_rw_mutex_);
  for (const auto& file_info_entry : files_) {
    int32_t status = file_info_entry.second->syncToDisk();
    CHECK(status == 0) << "Could not sync file to disk";
  }
}

void FileMgr::initializeNumThreads(size_t num_reader_threads) {
  // # of threads is based on # of cores on the host
  size_t num_hardware_based_threads = std::thread::hardware_concurrency();
  if (num_reader_threads == 0 || num_reader_threads > num_hardware_based_threads) {
    // # of threads has not been defined by user
    num_reader_threads_ = num_hardware_based_threads;
  } else {
    num_reader_threads_ = num_reader_threads;
  }
}

void FileMgr::freePages() {
  heavyai::unique_lock<heavyai::shared_mutex> free_pages_write_lock(mutex_free_page_);
  for (auto& free_page : free_pages_) {
    free_page.first->freePageDeferred(free_page.second);
  }
  free_pages_.clear();
}

FileBuffer* FileMgr::allocateBuffer(const size_t page_size,
                                    const ChunkKey& key,
                                    const size_t num_bytes) {
  return new FileBuffer(this, page_size, key, num_bytes);
}

FileBuffer* FileMgr::allocateBuffer(
    const ChunkKey& key,
    const std::vector<HeaderInfo>::const_iterator& headerStartIt,
    const std::vector<HeaderInfo>::const_iterator& headerEndIt) {
  return new FileBuffer(this, key, headerStartIt, headerEndIt);
}

// Checks if a page should be deleted or recovered.  Returns true if page was deleted.
bool FileMgr::updatePageIfDeleted(FileInfo* file_info,
                                  ChunkKey& chunk_key,
                                  int32_t contingent,
                                  int32_t page_epoch,
                                  int32_t page_num) {
  // If the parent FileMgr has a fileMgrKey, then all keys are locked to one table and
  // can be set from the manager.
  auto [db_id, tb_id] = get_fileMgrKey();
  chunk_key[CHUNK_KEY_DB_IDX] = db_id;
  chunk_key[CHUNK_KEY_TABLE_IDX] = tb_id;

  auto table_epoch = epoch(db_id, tb_id);

  if (is_page_deleted_with_checkpoint(table_epoch, page_epoch, contingent)) {
    file_info->freePageImmediate(page_num);
    return true;
  }

  // Recover page if it was deleted but not checkpointed.
  if (is_page_deleted_without_checkpoint(table_epoch, page_epoch, contingent)) {
    file_info->recoverPage(chunk_key, page_num);
  }
  return false;
}

FileBuffer* FileMgr::getOrCreateBuffer(const ChunkKey& key) {
  FileBuffer* buf;
  heavyai::unique_lock<heavyai::shared_mutex> chunkIndexWriteLock(chunkIndexMutex_);
  auto chunk_it = chunkIndex_.find(key);
  if (chunk_it == chunkIndex_.end()) {
    buf = createBufferUnlocked(key);
  } else {
    buf = getBufferUnlocked(key);
  }
  return buf;
}

void FileMgr::writeDirtyBuffers() {
  heavyai::unique_lock<heavyai::shared_mutex> chunk_index_write_lock(chunkIndexMutex_);
  for (auto [key, buf] : chunkIndex_) {
    if (buf->isDirty()) {
      buf->writeMetadata(epoch());
      buf->clearDirtyBits();
    }
  }
}

size_t FileMgr::getNumChunks() {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(chunkIndexMutex_);
  return chunkIndex_.size();
}

boost::filesystem::path FileMgr::getFilePath(const std::string& file_name) const {
  return boost::filesystem::path(fileMgrBasePath_) / file_name;
}

FILE* FileMgr::createFile(const std::string& full_path,
                          const size_t requested_file_size) const {
  readOnlyCheck("create file", full_path);
  return create(full_path, requested_file_size);
}

std::pair<FILE*, std::string> FileMgr::createFile(const std::string& base_path,
                                                  const int file_id,
                                                  const size_t page_size,
                                                  const size_t num_pages) const {
  readOnlyCheck("create file", get_data_file_path(base_path, file_id, page_size));
  return create(base_path, file_id, page_size, num_pages);
}

size_t FileMgr::writeFile(FILE* f,
                          const size_t offset,
                          const size_t size,
                          const int8_t* buf) const {
  readOnlyCheck("write file");
  return write(f, offset, size, buf);
}

void FileMgr::readOnlyCheck(const std::string& action,
                            const std::optional<std::string>& file_name) const {
  CHECK(!g_read_only) << "Error trying to " << action
                      << (file_name.has_value() ? (": '" + file_name.value() + "'") : "")
                      << ".  Not allowed in read only mode.";
}

size_t FileMgr::num_pages_per_data_file_{DEFAULT_NUM_PAGES_PER_DATA_FILE};
size_t FileMgr::num_pages_per_metadata_file_{DEFAULT_NUM_PAGES_PER_METADATA_FILE};
}  // namespace File_Namespace
