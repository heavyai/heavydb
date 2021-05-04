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
 * @file        CachingFileMgr.h
 */

#include "DataMgr/FileMgr/CachingFileMgr.h"
#include <boost/filesystem.hpp>
#include "Shared/misc.h"

namespace File_Namespace {
namespace bf = boost::filesystem;

CachingFileMgr::CachingFileMgr(const std::string& base_path,
                               const size_t num_reader_threads,
                               const size_t default_page_size) {
  fileMgrBasePath_ = base_path;
  maxRollbackEpochs_ = 0;
  defaultPageSize_ = default_page_size;
  nextFileId_ = 0;
  init(num_reader_threads);
}

CachingFileMgr::~CachingFileMgr() {}

void CachingFileMgr::init(const size_t num_reader_threads) {
  readTableDirs();
  auto open_files_result = openFiles();
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
    auto startIt = header_vec.begin();
    ChunkKey lastChunkKey = startIt->chunkKey;
    for (auto it = header_vec.begin() + 1; it != header_vec.end(); ++it) {
      if (it->chunkKey != lastChunkKey) {
        createBufferFromHeaders(lastChunkKey, startIt, it);
        lastChunkKey = it->chunkKey;
        startIt = it;
      }
    }
    createBufferFromHeaders(lastChunkKey, startIt, header_vec.end());
  }

  nextFileId_ = open_files_result.max_file_id + 1;
  incrementAllEpochs();
  freePages();
  initializeNumThreads(num_reader_threads);
  isFullyInitted_ = true;
}

/**
 * Assumes a base directory exists.  Checks for any sub-directories containing
 * table-specific data and creates epochs from found files.
 **/
void CachingFileMgr::readTableDirs() {
  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  bf::path path(fileMgrBasePath_);
  CHECK(bf::exists(path)) << "Cache path: " << fileMgrBasePath_ << " does not exit.";
  CHECK(bf::is_directory(path))
      << "Specified path '" << fileMgrBasePath_ << "' for disk cache is not a directory.";

  // Look for directories with table-specific names.
  boost::regex table_filter("table_([0-9]+)_([0-9]+)");
  for (const auto& file : bf::directory_iterator(path)) {
    boost::smatch match;
    auto file_name = file.path().filename().string();
    if (boost::regex_match(file_name, match, table_filter)) {
      int32_t db_id = std::stoi(match[1]);
      int32_t tb_id = std::stoi(match[2]);
      CHECK(table_epochs_.find({db_id, tb_id}) == table_epochs_.end())
          << "Trying to read epoch for existing table";
      openOrCreateEpochIfNotExists(db_id, tb_id);
    }
  }
}

int32_t CachingFileMgr::epoch(int32_t db_id, int32_t tb_id) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(epochs_mutex_);
  auto table_epoch_it = table_epochs_.find({db_id, tb_id});
  CHECK(table_epoch_it != table_epochs_.end());
  auto& [pair, epochInfo] = *table_epoch_it;
  return static_cast<int32_t>(epochInfo->epoch.ceiling());
}

void CachingFileMgr::incrementEpoch(int32_t db_id, int32_t tb_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(epochs_mutex_);
  auto epochs_it = table_epochs_.find({db_id, tb_id});
  CHECK(epochs_it != table_epochs_.end());
  auto& [pair, epochInfo] = *epochs_it;
  epochInfo->increment();
}

void CachingFileMgr::openOrCreateEpochIfNotExists(int32_t db_id, int32_t tb_id) {
  mapd_unique_lock<mapd_shared_mutex> epoch_lock(epochs_mutex_);
  TablePair table_pair{db_id, tb_id};
  if (table_epochs_.find(table_pair) == table_epochs_.end()) {
    openAndReadEpochFileUnlocked(db_id, tb_id);
  }
}

void CachingFileMgr::openAndReadEpochFileUnlocked(int32_t db_id, int32_t tb_id) {
  TablePair table_pair{db_id, tb_id};
  auto table_epoch_it = table_epochs_.find(table_pair);
  if (table_epoch_it == table_epochs_.end()) {
    std::string epoch_file_path(getOrAddTableDirUnlocked(db_id, tb_id) + "/" +
                                EPOCH_FILENAME);
    if (!bf::exists(epoch_file_path)) {
      // Epoch file was missing or malformed.  Create a new one.
      createEpochFileUnlocked(db_id, tb_id);
      return;
    } else {
      CHECK(bf::is_regular_file(epoch_file_path))
          << "Found epoch file '" << epoch_file_path << "' which is not a regular file";
      CHECK(bf::file_size(epoch_file_path) == Epoch::byte_size())
          << "Found epoch file '" << epoch_file_path << "' which is not of expected size";
    }
    table_epochs_.emplace(table_pair, std::make_unique<EpochInfo>(open(epoch_file_path)));
  }
  table_epoch_it = table_epochs_.find(table_pair);
  auto& [epoch, epoch_file, is_checkpointed] = *(table_epoch_it->second);
  read(epoch_file, 0, Epoch::byte_size(), epoch.storage_ptr());
}

void CachingFileMgr::createEpochFileUnlocked(int32_t db_id, int32_t tb_id) {
  std::string epoch_file_path(getOrAddTableDirUnlocked(db_id, tb_id) + "/" +
                              EPOCH_FILENAME);
  CHECK(!bf::exists(epoch_file_path)) << "Can't create epoch file. File already exists";
  TablePair table_pair{db_id, tb_id};
  table_epochs_.emplace(
      table_pair,
      std::make_unique<EpochInfo>(create(epoch_file_path, sizeof(Epoch::byte_size()))));
  writeAndSyncEpochToDisk(db_id, tb_id);
  table_epochs_.at(table_pair)->increment();
}

void CachingFileMgr::writeAndSyncEpochToDisk(int32_t db_id, int32_t tb_id) {
  auto epochs_it = table_epochs_.find({db_id, tb_id});
  CHECK(epochs_it != table_epochs_.end());
  auto& [pair, epoch_info] = *epochs_it;
  auto& [epoch, epoch_file, is_checkpointed] = *epoch_info;
  write(epoch_file, 0, Epoch::byte_size(), epoch.storage_ptr());
  int32_t status = fflush(epoch_file);
  CHECK(status == 0) << "Could not flush epoch file to disk";
#ifdef __APPLE__
  status = fcntl(fileno(epoch_file), 51);
#else
  status = omnisci::fsync(fileno(epoch_file));
#endif
  CHECK(status == 0) << "Could not sync epoch file to disk";
  is_checkpointed = true;
}

void CachingFileMgr::clearForTable(int32_t db_id, int32_t tb_id) {
  removeTableBuffers(db_id, tb_id);
  removeTableDirectory(db_id, tb_id);
  freePages();
}

std::string CachingFileMgr::getOrAddTableDir(int db_id, int tb_id) {
  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  return getOrAddTableDirUnlocked(db_id, tb_id);
}

std::string CachingFileMgr::getOrAddTableDirUnlocked(int db_id, int tb_id) {
  std::string table_dir =
      getFileMgrBasePath() + "/" + get_dir_name_for_table(db_id, tb_id);
  if (!bf::exists(table_dir)) {
    bf::create_directory(table_dir);
  } else {
    if (!bf::is_directory(table_dir)) {
      LOG(FATAL) << "Specified path '" << table_dir
                 << "' for cache table data is not a directory.";
    }
  }
  return table_dir;
}

void CachingFileMgr::closeRemovePhysical() {
  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  closePhysicalUnlocked();
  table_epochs_.clear();
  auto dir_name = getFileMgrBasePath();
  if (bf::exists(dir_name)) {
    bf::remove_all(dir_name);
  }
}

uint64_t CachingFileMgr::getChunkSpaceReservedByTable(int db_id, int tb_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  uint64_t space_used = 0;
  ChunkKey min_table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};
  for (auto it = chunkIndex_.lower_bound(min_table_key);
       it != chunkIndex_.upper_bound(max_table_key);
       ++it) {
    auto& [key, buffer] = *it;
    space_used += (buffer->numChunkPages() * defaultPageSize_);
  }
  return space_used;
}

uint64_t CachingFileMgr::getMetadataSpaceReservedByTable(int db_id, int tb_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  uint64_t space_used = 0;
  ChunkKey min_table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};
  for (auto it = chunkIndex_.lower_bound(min_table_key);
       it != chunkIndex_.upper_bound(max_table_key);
       ++it) {
    auto& [key, buffer] = *it;
    space_used += (buffer->numMetadataPages() * METADATA_PAGE_SIZE);
  }
  return space_used;
}

uint64_t CachingFileMgr::getWrapperSpaceReservedByTable(int db_id, int tb_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(files_rw_mutex_);
  uint64_t space_used = 0;
  std::string table_dir =
      getFileMgrBasePath() + "/" + get_dir_name_for_table(db_id, tb_id);
  if (bf::exists(table_dir)) {
    for (const auto& file : bf::recursive_directory_iterator(table_dir)) {
      if (bf::is_regular_file(file.path())) {
        space_used += bf::file_size(file.path());
      }
    }
  }
  return space_used;
}

uint64_t CachingFileMgr::getSpaceReservedByTable(int db_id, int tb_id) {
  auto chunk_space = getChunkSpaceReservedByTable(db_id, tb_id);
  auto meta_space = getMetadataSpaceReservedByTable(db_id, tb_id);
  auto wrapper_space = getWrapperSpaceReservedByTable(db_id, tb_id);
  return chunk_space + meta_space + wrapper_space;
}

std::string CachingFileMgr::describeSelf() {
  return "cache";
}

// Similar to FileMgr::checkpoint() but only writes a subset of buffers.
void CachingFileMgr::checkpoint(const int32_t db_id, const int32_t tb_id) {
  VLOG(2) << "Checkpointing " << describeSelf() << " (" << db_id << ", " << tb_id
          << ") epoch: " << epoch(db_id, tb_id);
  writeDirtyBuffers(db_id, tb_id);
  syncFilesToDisk();
  writeAndSyncEpochToDisk(db_id, tb_id);
  incrementEpoch(db_id, tb_id);
  freePages();
}

FileBuffer* CachingFileMgr::createBufferUnlocked(const ChunkKey& key,
                                                 size_t page_size,
                                                 const size_t num_bytes) {
  auto [db_id, tb_id] = get_table_prefix(key);
  // We need to have an epoch to correspond to each table for which we have buffers.
  openOrCreateEpochIfNotExists(db_id, tb_id);
  return FileMgr::createBufferUnlocked(key, page_size, num_bytes);
}

void CachingFileMgr::createBufferFromHeaders(
    const ChunkKey& key,
    const std::vector<HeaderInfo>::const_iterator& startIt,
    const std::vector<HeaderInfo>::const_iterator& endIt) {
  if (startIt->pageId == -1) {
    // If the first pageId is not -1 then there is no metadata page for the
    // current key (which means it was never checkpointed), so we should skip.

    // Need to acquire chunkIndexMutex_ lock first to avoid lock order cycles.
    mapd_unique_lock<mapd_shared_mutex> chunk_lock(chunkIndexMutex_);
    auto [db_id, tb_id] = get_table_prefix(key);
    mapd_shared_lock<mapd_shared_mutex> epochs_lock(epochs_mutex_);
    CHECK(table_epochs_.find({db_id, tb_id}) != table_epochs_.end());
    CHECK(chunkIndex_.find(key) == chunkIndex_.end());
    chunkIndex_[key] = new CachingFileBuffer(this, key, startIt, endIt);

    auto buffer = chunkIndex_.at(key);
    if (buffer->isMissingPages()) {
      // Detect the case where a page is missing by comparing the amount of pages read
      // with the metadata size.  If data are missing, discard the chunk.
      buffer->freeChunkPages();
    }
  }
}

/**
 * putBuffer() needs to behave differently than it does in FileMgr. Specifically, it needs
 * to delete the buffer beforehand and then append, rather than overwrite the existing
 * buffer. This way we only store a single version of the buffer rather than accumulating
 * versions that need to be rolled off.
 **/
FileBuffer* CachingFileMgr::putBuffer(const ChunkKey& key,
                                      AbstractBuffer* src_buffer,
                                      const size_t num_bytes) {
  deleteBufferIfExists(key);
  return FileMgr::putBuffer(key, src_buffer, num_bytes);
}

void CachingFileMgr::incrementAllEpochs() {
  mapd_shared_lock<mapd_shared_mutex> read_lock(epochs_mutex_);
  for (auto& [key, epochInfo] : table_epochs_) {
    epochInfo->increment();
  }
}

void CachingFileMgr::removeTableDirectory(int32_t db_id, int32_t tb_id) {
  // Delete table-specific directory (stores table epoch data and serialized data wrapper)
  mapd_unique_lock<mapd_shared_mutex> write_lock(epochs_mutex_);
  table_epochs_.erase({db_id, tb_id});
  auto dir_name = getFileMgrBasePath() + "/" + get_dir_name_for_table(db_id, tb_id);
  bf::remove_all(dir_name);
}

void CachingFileMgr::removeTableBuffers(int32_t db_id, int32_t tb_id) {
  // Free associated FileBuffers and clear buffer entries.
  mapd_unique_lock<mapd_shared_mutex> write_lock(chunkIndexMutex_);
  ChunkKey min_table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};
  for (auto it = chunkIndex_.lower_bound(min_table_key);
       it != chunkIndex_.upper_bound(max_table_key);) {
    auto& [key, buffer] = *it;
    buffer->freePages();
    delete buffer;
    it = chunkIndex_.erase(it);
  }
}

CachingFileBuffer* CachingFileMgr::allocateBuffer(const size_t page_size,
                                                  const ChunkKey& key,
                                                  const size_t num_bytes) {
  return new CachingFileBuffer(this, page_size, key, num_bytes);
}

// Checks if a page should be deleted or recovered.  Returns true if page was deleted.
bool CachingFileMgr::updatePageIfDeleted(FileInfo* file_info,
                                         ChunkKey& chunk_key,
                                         int32_t contingent,
                                         int32_t page_epoch,
                                         int32_t page_num) {
  // These contingents are stored by overwriting the bytes used for chunkKeys.  If
  // we run into a key marked for deletion in a fileMgr with no fileMgrKey (i.e.
  // CachingFileMgr) then we can't know if the epoch is valid because we don't know
  // the key.  At this point our only option is to free the page as though it was
  // checkpointed (which should be fine since we only maintain one version of each
  // page).
  if (contingent == DELETE_CONTINGENT || contingent == ROLLOFF_CONTINGENT) {
    file_info->freePageImmediate(page_num);
    return true;
  }
  return false;
}

void CachingFileMgr::writeDirtyBuffers(int32_t db_id, int32_t tb_id) {
  mapd_unique_lock<mapd_shared_mutex> chunk_index_write_lock(chunkIndexMutex_);
  ChunkKey min_table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};

  for (auto chunkIt = chunkIndex_.lower_bound(min_table_key);
       chunkIt != chunkIndex_.upper_bound(max_table_key);
       ++chunkIt) {
    if (chunkIt->second->isDirty()) {
      // Free previous versions first so we only have one metadata version.
      chunkIt->second->freeMetadataPages();
      chunkIt->second->writeMetadata(epoch(db_id, tb_id));
      chunkIt->second->clearDirtyBits();
    }
  }
}

void CachingFileMgr::deleteBufferIfExists(const ChunkKey& key) {
  mapd_unique_lock<mapd_shared_mutex> chunk_index_write_lock(chunkIndexMutex_);
  auto chunk_it = chunkIndex_.find(key);
  if (chunk_it != chunkIndex_.end()) {
    deleteBufferUnlocked(chunk_it);
  }
}

}  // namespace File_Namespace
