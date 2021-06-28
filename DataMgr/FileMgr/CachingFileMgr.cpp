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

namespace bf = boost::filesystem;

namespace {
size_t size_of_dir(const std::string& dir) {
  size_t space_used = 0;
  if (bf::exists(dir)) {
    for (const auto& file : bf::recursive_directory_iterator(dir)) {
      if (bf::is_regular_file(file.path())) {
        space_used += bf::file_size(file.path());
      }
    }
  }
  return space_used;
}

ChunkKey evict_chunk_or_fail(LRUEvictionAlgorithm& alg) {
  ChunkKey ret;
  try {
    ret = alg.evictNextChunk();
  } catch (const NoEntryFoundException& e) {
    LOG(FATAL) << "Disk cache needs to evict data to make space, but no data found in "
                  "eviction queue.";
  }
  return ret;
}
}  // namespace

namespace File_Namespace {

CachingFileMgr::CachingFileMgr(const DiskCacheConfig& config) {
  fileMgrBasePath_ = config.path;
  maxRollbackEpochs_ = 0;
  defaultPageSize_ = config.page_size;
  nextFileId_ = 0;
  max_size_ = config.size_limit;
  init(config.num_reader_threads);
  setMaxSizes();
}

CachingFileMgr::~CachingFileMgr() {}

void CachingFileMgr::init(const size_t num_reader_threads) {
  deleteCacheIfTooLarge();
  readTableFileMgrs();
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

void CachingFileMgr::readTableFileMgrs() {
  mapd_unique_lock<mapd_shared_mutex> write_lock(table_dirs_mutex_);
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
      TablePair table_pair{db_id, tb_id};
      CHECK(table_dirs_.find(table_pair) == table_dirs_.end())
          << "Trying to read data for existing table";
      table_dirs_.emplace(table_pair,
                          std::make_unique<TableFileMgr>(file.path().string()));
    }
  }
}

int32_t CachingFileMgr::epoch(int32_t db_id, int32_t tb_id) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
  auto tables_it = table_dirs_.find({db_id, tb_id});
  CHECK(tables_it != table_dirs_.end());
  auto& [pair, table_dir] = *tables_it;
  return table_dir->getEpoch();
}

void CachingFileMgr::incrementEpoch(int32_t db_id, int32_t tb_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
  auto tables_it = table_dirs_.find({db_id, tb_id});
  CHECK(tables_it != table_dirs_.end());
  auto& [pair, table_dir] = *tables_it;
  table_dir->incrementEpoch();
}

void CachingFileMgr::writeAndSyncEpochToDisk(int32_t db_id, int32_t tb_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
  auto table_it = table_dirs_.find({db_id, tb_id});
  CHECK(table_it != table_dirs_.end());
  table_it->second->writeAndSyncEpochToDisk();
}

void CachingFileMgr::clearForTable(int32_t db_id, int32_t tb_id) {
  removeTableBuffers(db_id, tb_id);
  removeTableFileMgr(db_id, tb_id);
  freePages();
}

std::string CachingFileMgr::getTableFileMgrPath(int32_t db_id, int32_t tb_id) const {
  return getFileMgrBasePath() + "/" + get_dir_name_for_table(db_id, tb_id);
}

void CachingFileMgr::closeRemovePhysical() {
  {
    mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
    closePhysicalUnlocked();
  }
  {
    mapd_unique_lock<mapd_shared_mutex> tables_lock(table_dirs_mutex_);
    table_dirs_.clear();
  }
  bf::remove_all(getFileMgrBasePath());
}

size_t CachingFileMgr::getChunkSpaceReservedByTable(int32_t db_id, int32_t tb_id) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  size_t space_used = 0;
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

size_t CachingFileMgr::getMetadataSpaceReservedByTable(int32_t db_id,
                                                       int32_t tb_id) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  size_t space_used = 0;
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

size_t CachingFileMgr::getTableFileMgrSpaceReserved(int32_t db_id, int32_t tb_id) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
  size_t space = 0;
  auto table_it = table_dirs_.find({db_id, tb_id});
  if (table_it != table_dirs_.end()) {
    space += table_it->second->getReservedSpace();
  }
  return space;
}

size_t CachingFileMgr::getSpaceReservedByTable(int32_t db_id, int32_t tb_id) const {
  auto chunk_space = getChunkSpaceReservedByTable(db_id, tb_id);
  auto meta_space = getMetadataSpaceReservedByTable(db_id, tb_id);
  auto subdir_space = getTableFileMgrSpaceReserved(db_id, tb_id);
  return chunk_space + meta_space + subdir_space;
}

std::string CachingFileMgr::describeSelf() const {
  return "cache";
}

// Similar to FileMgr::checkpoint() but only writes a subset of buffers.
void CachingFileMgr::checkpoint(const int32_t db_id, const int32_t tb_id) {
  {
    mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
    CHECK(table_dirs_.find({db_id, tb_id}) != table_dirs_.end());
  }
  VLOG(2) << "Checkpointing " << describeSelf() << " (" << db_id << ", " << tb_id
          << ") epoch: " << epoch(db_id, tb_id);
  writeDirtyBuffers(db_id, tb_id);
  syncFilesToDisk();
  writeAndSyncEpochToDisk(db_id, tb_id);
  incrementEpoch(db_id, tb_id);
  freePages();
}

void CachingFileMgr::createTableFileMgrIfNoneExists(const int32_t db_id,
                                                    const int32_t tb_id) {
  mapd_unique_lock<mapd_shared_mutex> write_lock(table_dirs_mutex_);
  TablePair table_pair{db_id, tb_id};
  if (table_dirs_.find(table_pair) == table_dirs_.end()) {
    table_dirs_.emplace(
        table_pair, std::make_unique<TableFileMgr>(getTableFileMgrPath(db_id, tb_id)));
  }
}

FileBuffer* CachingFileMgr::createBufferUnlocked(const ChunkKey& key,
                                                 const size_t page_size,
                                                 const size_t num_bytes) {
  touchKey(key);
  auto [db_id, tb_id] = get_table_prefix(key);
  createTableFileMgrIfNoneExists(db_id, tb_id);
  return FileMgr::createBufferUnlocked(key, page_size, num_bytes);
}

FileBuffer* CachingFileMgr::createBufferFromHeaders(
    const ChunkKey& key,
    const std::vector<HeaderInfo>::const_iterator& startIt,
    const std::vector<HeaderInfo>::const_iterator& endIt) {
  if (startIt->pageId != -1) {
    // If the first pageId is not -1 then there is no metadata page for the
    // current key (which means it was never checkpointed), so we should skip.
    return nullptr;
  }
  touchKey(key);
  auto [db_id, tb_id] = get_table_prefix(key);
  createTableFileMgrIfNoneExists(db_id, tb_id);
  auto buffer = FileMgr::createBufferFromHeaders(key, startIt, endIt);
  if (buffer->isMissingPages()) {
    // Detect the case where a page is missing by comparing the amount of pages read
    // with the metadata size.  If data are missing, discard the chunk.
    buffer->freeChunkPages();
  }
  return buffer;
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
  CHECK(!src_buffer->isDirty()) << "Cannot cache dirty buffers.";
  deleteBufferIfExists(key);
  // Since the buffer is not dirty we mark it as dirty if we are only writing metadata and
  // appended if we are writing chunk data.  We delete + append rather than write to make
  // sure we don't write multiple page versions.
  (src_buffer->size() == 0) ? src_buffer->setDirty() : src_buffer->setAppended();
  return FileMgr::putBuffer(key, src_buffer, num_bytes);
}

void CachingFileMgr::incrementAllEpochs() {
  mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
  for (auto& table_dir : table_dirs_) {
    table_dir.second->incrementEpoch();
  }
}

void CachingFileMgr::removeTableFileMgr(int32_t db_id, int32_t tb_id) {
  // Delete table-specific directory (stores table epoch data and serialized data wrapper)
  mapd_unique_lock<mapd_shared_mutex> write_lock(table_dirs_mutex_);
  auto it = table_dirs_.find({db_id, tb_id});
  if (it != table_dirs_.end()) {
    it->second->removeDiskContent();
    table_dirs_.erase(it);
  }
}

void CachingFileMgr::removeTableBuffers(int32_t db_id, int32_t tb_id) {
  // Free associated FileBuffers and clear buffer entries.
  mapd_unique_lock<mapd_shared_mutex> write_lock(chunkIndexMutex_);
  ChunkKey min_table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};
  for (auto it = chunkIndex_.lower_bound(min_table_key);
       it != chunkIndex_.upper_bound(max_table_key);) {
    it = deleteBufferUnlocked(it);
  }
}

CachingFileBuffer* CachingFileMgr::allocateBuffer(const size_t page_size,
                                                  const ChunkKey& key,
                                                  const size_t num_bytes) {
  return new CachingFileBuffer(this, page_size, key, num_bytes);
}

CachingFileBuffer* CachingFileMgr::allocateBuffer(
    const ChunkKey& key,
    const std::vector<HeaderInfo>::const_iterator& headerStartIt,
    const std::vector<HeaderInfo>::const_iterator& headerEndIt) {
  return new CachingFileBuffer(this, key, headerStartIt, headerEndIt);
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
    file_info->freePage(page_num, false, page_epoch);
    return true;
  }
  return false;
}

void CachingFileMgr::writeDirtyBuffers(int32_t db_id, int32_t tb_id) {
  mapd_unique_lock<mapd_shared_mutex> chunk_index_write_lock(chunkIndexMutex_);
  ChunkKey min_table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};

  for (auto chunk_it = chunkIndex_.lower_bound(min_table_key);
       chunk_it != chunkIndex_.upper_bound(max_table_key);
       ++chunk_it) {
    if (auto [key, buf] = *chunk_it; buf->isDirty()) {
      // Free previous versions first so we only have one metadata version.
      buf->freeMetadataPages();
      buf->writeMetadata(epoch(db_id, tb_id));
      buf->clearDirtyBits();
      touchKey(key);
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

size_t CachingFileMgr::getNumDataChunks() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  size_t num_chunks = 0;
  for (auto [key, buf] : chunkIndex_) {
    if (buf->hasDataPages()) {
      num_chunks++;
    }
  }
  return num_chunks;
}

void CachingFileMgr::deleteCacheIfTooLarge() {
  if (size_of_dir(fileMgrBasePath_) > max_size_) {
    closeRemovePhysical();
    bf::create_directory(fileMgrBasePath_);
    LOG(INFO) << "Cache path over limit.  Existing cache deleted.";
  }
}

Page CachingFileMgr::requestFreePage(size_t pageSize, const bool isMetadata) {
  std::lock_guard<std::mutex> lock(getPageMutex_);
  int32_t pageNum = -1;
  // Splits files into metadata and regular data by size.
  auto candidateFiles = fileIndex_.equal_range(pageSize);
  // Check if there is a free page in an existing file.
  for (auto fileIt = candidateFiles.first; fileIt != candidateFiles.second; ++fileIt) {
    FileInfo* fileInfo = files_.at(fileIt->second);
    pageNum = fileInfo->getFreePage();
    if (pageNum != -1) {
      return (Page(fileInfo->fileId, pageNum));
    }
  }

  // Try to add a new file if there is free space available.
  FileInfo* fileInfo = nullptr;
  if (isMetadata) {
    if (getMaxMetaFiles() > getNumMetaFiles()) {
      fileInfo = createFile(pageSize, num_pages_per_metadata_file_);
    }
  } else {
    if (getMaxDataFiles() > getNumDataFiles()) {
      fileInfo = createFile(pageSize, num_pages_per_data_file_);
    }
  }

  if (!fileInfo) {
    // We were not able to create a new file, so we try to evict space.
    // Eviction will return the first file it evicted a page from (a file now guaranteed
    // to have a free page).
    fileInfo = isMetadata ? evictMetadataPages() : evictPages();
  }
  CHECK(fileInfo);

  pageNum = fileInfo->getFreePage();
  CHECK(pageNum != -1);
  return (Page(fileInfo->fileId, pageNum));
}

std::vector<ChunkKey> CachingFileMgr::getKeysForTable(int32_t db_id,
                                                      int32_t tb_id) const {
  std::vector<ChunkKey> keys;
  ChunkKey min_table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};
  for (auto it = chunkIndex_.lower_bound(min_table_key);
       it != chunkIndex_.upper_bound(max_table_key);
       ++it) {
    keys.emplace_back(it->first);
  }
  return keys;
}

FileInfo* CachingFileMgr::evictMetadataPages() {
  // Locks should already be in place before calling this method.
  FileInfo* file_info{nullptr};
  auto key_to_evict = evict_chunk_or_fail(table_evict_alg_);
  auto [db_id, tb_id] = get_table_prefix(key_to_evict);
  const auto keys = getKeysForTable(db_id, tb_id);
  for (const auto& key : keys) {
    auto chunk_it = chunkIndex_.find(key);
    CHECK(chunk_it != chunkIndex_.end());
    auto& buf = chunk_it->second;
    if (!file_info) {
      // Return the FileInfo for the first file we are freeing a page from so that the
      // caller does not have to search for a FileInfo guaranteed to have at least one
      // free page.
      CHECK(buf->getMetadataPage().pageVersions.size() > 0);
      file_info =
          getFileInfoForFileId(buf->getMetadataPage().pageVersions.front().page.fileId);
    }
    // We erase all pages and entries for the chunk, as without metadata all other
    // entries are useless.
    deleteBufferUnlocked(chunk_it);
  }
  // Serialized datawrappers require metadata to be in the cache.
  deleteWrapperFile(db_id, tb_id);
  CHECK(file_info) << "FileInfo with freed page not found";
  return file_info;
}

FileInfo* CachingFileMgr::evictPages() {
  FileInfo* file_info{nullptr};
  FileBuffer* buf{nullptr};
  while (!file_info) {
    buf = chunkIndex_.at(evict_chunk_or_fail(chunk_evict_alg_));
    CHECK(buf);
    if (!buf->hasDataPages()) {
      // This buffer contains no chunk data (metadata only, uninitialized, size == 0,
      // etc...) so we won't recover any space by evicting it.  In this case it gets
      // removed from the eviction queue (it will get re-added if it gets populated with
      // data) and we look at the next chunk in queue until we find a buffer with page
      // data.
      continue;
    }
    // Return the FileInfo for the first file we are freeing a page from so that the
    // caller does not have to search for a FileInfo guaranteed to have at least one free
    // page.
    CHECK(buf->getMultiPage().front().pageVersions.size() > 0);
    file_info = getFileInfoForFileId(
        buf->getMultiPage().front().pageVersions.front().page.fileId);
  }
  auto pages_freed = buf->freeChunkPages();
  CHECK(pages_freed > 0) << "failed to evict a page";
  CHECK(file_info) << "FileInfo with freed page not found";
  return file_info;
}

void CachingFileMgr::touchKey(const ChunkKey& key) const {
  chunk_evict_alg_.touchChunk(key);
  table_evict_alg_.touchChunk(get_table_key(key));
}

void CachingFileMgr::removeKey(const ChunkKey& key) const {
  // chunkIndex lock should already be acquired.
  chunk_evict_alg_.removeChunk(key);
  auto [db_id, tb_id] = get_table_prefix(key);
  ChunkKey table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};
  for (auto it = chunkIndex_.lower_bound(table_key);
       it != chunkIndex_.upper_bound(max_table_key);
       ++it) {
    if (it->first != key) {
      // If there are any keys in this table other than that one we are removing, then
      // keep the table in the eviction queue.
      return;
    }
  }
  // No other keys exist for this table, so remove it from the queue.
  table_evict_alg_.removeChunk(table_key);
}

size_t CachingFileMgr::getFilesSize() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(files_rw_mutex_);
  size_t sum = 0;
  for (auto [id, file] : files_) {
    sum += file->size();
  }
  return sum;
}

size_t CachingFileMgr::getTableFileMgrsSize() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
  size_t space_used = 0;
  for (const auto& [pair, table_dir] : table_dirs_) {
    space_used += table_dir->getReservedSpace();
  }
  return space_used;
}

size_t CachingFileMgr::getNumDataFiles() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(files_rw_mutex_);
  return fileIndex_.count(defaultPageSize_);
}

size_t CachingFileMgr::getNumMetaFiles() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(files_rw_mutex_);
  return fileIndex_.count(METADATA_PAGE_SIZE);
}

std::vector<ChunkKey> CachingFileMgr::getChunkKeysForPrefix(
    const ChunkKey& prefix) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  std::vector<ChunkKey> chunks;
  for (auto [key, buf] : chunkIndex_) {
    if (in_same_table(key, prefix)) {
      if (buf->hasDataPages()) {
        chunks.emplace_back(key);
        touchKey(key);
      }
    }
  }
  return chunks;
}

void CachingFileMgr::removeChunkKeepMetadata(const ChunkKey& key) {
  if (isBufferOnDevice(key)) {
    auto chunkIt = chunkIndex_.find(key);
    CHECK(chunkIt != chunkIndex_.end());
    auto& buf = chunkIt->second;
    if (buf->hasDataPages()) {
      buf->freeChunkPages();
      chunk_evict_alg_.removeChunk(key);
    }
  }
}

size_t CachingFileMgr::getNumChunksWithMetadata() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  size_t sum = 0;
  for (const auto& [key, buf] : chunkIndex_) {
    if (buf->hasEncoder()) {
      sum++;
    }
  }
  return sum;
}

std::string CachingFileMgr::dumpKeysWithMetadata() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  std::string ret_string = "CFM keys with metadata:\n";
  for (const auto& [key, buf] : chunkIndex_) {
    if (buf->hasEncoder()) {
      ret_string += "  " + show_chunk(key) + "\n";
    }
  }
  return ret_string;
}

std::string CachingFileMgr::dumpKeysWithChunkData() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  std::string ret_string = "CFM keys with chunk data:\n";
  for (const auto& [key, buf] : chunkIndex_) {
    if (buf->hasDataPages()) {
      ret_string += "  " + show_chunk(key) + "\n";
    }
  }
  return ret_string;
}

std::unique_ptr<CachingFileMgr> CachingFileMgr::reconstruct() const {
  DiskCacheConfig config{fileMgrBasePath_,
                         DiskCacheLevel::none,
                         num_reader_threads_,
                         max_size_,
                         defaultPageSize_};
  return std::make_unique<CachingFileMgr>(config);
}

void CachingFileMgr::deleteWrapperFile(int32_t db, int32_t tb) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
  auto it = table_dirs_.find({db, tb});
  CHECK(it != table_dirs_.end());
  it->second->deleteWrapperFile();
}

void CachingFileMgr::writeWrapperFile(const std::string& doc, int32_t db, int32_t tb) {
  createTableFileMgrIfNoneExists(db, tb);
  auto wrapper_size = doc.size();
  CHECK_LE(wrapper_size, getMaxWrapperSize())
      << "Wrapper is too big to fit into the cache";
  while (wrapper_size > getAvailableWrapperSpace()) {
    evictMetadataPages();
  }
  mapd_shared_lock<mapd_shared_mutex> read_lock(table_dirs_mutex_);
  table_dirs_.at({db, tb})->writeWrapperFile(doc);
}

/*
 * While the CFM allows for multiple tables to share the same allocated files for chunk
 * data and metadata, space cannot be reallocated between metadata files and data files
 * (once the space has been reserve for a data file the file won't be deleted unless the
 * cache is deleted).  To prevent a case where we have allocated too many files of one
 * type to the detrement of the other, we have a minimum portion of the cache that is
 * reserved for each type.  This default ratio gives %9 of space to data wrappers, %1 to
 * metadata files, and %90 to data files.
 */
void CachingFileMgr::setMaxSizes() {
  size_t max_meta_space = std::floor(max_size_ * METADATA_SPACE_PERCENTAGE);
  size_t max_meta_file_space = std::floor(max_size_ * METADATA_FILE_SPACE_PERCENTAGE);
  max_wrapper_space_ = max_meta_space - max_meta_file_space;
  auto max_data_space = max_size_ - max_meta_space;
  auto meta_file_size = METADATA_PAGE_SIZE * num_pages_per_metadata_file_;
  auto data_file_size = defaultPageSize_ * num_pages_per_data_file_;
  max_num_data_files_ = max_data_space / data_file_size;
  max_num_meta_files_ = max_meta_file_space / meta_file_size;
  CHECK_GT(max_num_data_files_, 0U) << "Cannot create a cache of size " << max_size_
                                    << ".  Not enough space to create a data file.";
  CHECK_GT(max_num_meta_files_, 0U) << "Cannot create a cache of size " << max_size_
                                    << ".  Not enough space to create a metadata file.";
}

std::optional<FileBuffer*> CachingFileMgr::getBufferIfExists(const ChunkKey& key) {
  mapd_shared_lock<mapd_shared_mutex> chunk_index_read_lock(chunkIndexMutex_);
  auto chunk_it = chunkIndex_.find(key);
  if (chunk_it == chunkIndex_.end()) {
    return {};
  }
  return getBufferUnlocked(chunk_it);
}

ChunkKeyToChunkMap::iterator CachingFileMgr::deleteBufferUnlocked(
    const ChunkKeyToChunkMap::iterator chunk_it,
    const bool purge) {
  removeKey(chunk_it->first);
  return FileMgr::deleteBufferUnlocked(chunk_it, purge);
}

void CachingFileMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunkMetadataVec,
    const ChunkKey& keyPrefix) {
  FileMgr::getChunkMetadataVecForKeyPrefix(chunkMetadataVec, keyPrefix);
  for (const auto& [key, meta] : chunkMetadataVec) {
    touchKey(key);
  }
}

FileBuffer* CachingFileMgr::getBufferUnlocked(const ChunkKeyToChunkMap::iterator chunk_it,
                                              const size_t num_bytes) {
  touchKey(chunk_it->first);
  return FileMgr::getBufferUnlocked(chunk_it, num_bytes);
}

void CachingFileMgr::free_page(std::pair<FileInfo*, int32_t>&& page) {
  page.first->freePageDeferred(page.second);
}

std::set<ChunkKey> CachingFileMgr::getKeysWithMetadata() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  std::set<ChunkKey> ret;
  for (const auto& [key, buf] : chunkIndex_) {
    if (buf->hasEncoder()) {
      ret.emplace(key);
    }
  }
  return ret;
}

TableFileMgr::TableFileMgr(const std::string& table_path)
    : table_path_(table_path)
    , epoch_file_path_(table_path_ + "/" + FileMgr::EPOCH_FILENAME)
    , wrapper_file_path_(table_path_ + "/" + CachingFileMgr::WRAPPER_FILE_NAME)
    , epoch_(Epoch())
    , is_checkpointed_(true) {
  if (!bf::exists(table_path_)) {
    bf::create_directory(table_path_);
  } else {
    CHECK(bf::is_directory(table_path_)) << "Specified path '" << table_path_
                                         << "' for cache table data is not a directory.";
  }
  if (bf::exists(epoch_file_path_)) {
    CHECK(bf::is_regular_file(epoch_file_path_))
        << "Found epoch file '" << epoch_file_path_ << "' which is not a regular file";
    CHECK(bf::file_size(epoch_file_path_) == Epoch::byte_size())
        << "Found epoch file '" << epoch_file_path_ << "' which is not of expected size";
    epoch_file_ = open(epoch_file_path_);
    read(epoch_file_, 0, Epoch::byte_size(), epoch_.storage_ptr());
  } else {
    epoch_file_ = create(epoch_file_path_, sizeof(Epoch::byte_size()));
    writeAndSyncEpochToDisk();
    incrementEpoch();
  }
}

void TableFileMgr::incrementEpoch() {
  mapd_unique_lock<mapd_shared_mutex> w_lock(table_mutex_);
  epoch_.increment();
  is_checkpointed_ = false;
  CHECK(epoch_.ceiling() <= Epoch::max_allowable_epoch())
      << "Epoch greater than maximum allowed value (" << epoch_.ceiling() << " > "
      << Epoch::max_allowable_epoch() << ").";
}

int32_t TableFileMgr::getEpoch() const {
  mapd_shared_lock<mapd_shared_mutex> r_lock(table_mutex_);
  return static_cast<int32_t>(epoch_.ceiling());
}

void TableFileMgr::writeAndSyncEpochToDisk() {
  mapd_unique_lock<mapd_shared_mutex> w_lock(table_mutex_);
  write(epoch_file_, 0, Epoch::byte_size(), epoch_.storage_ptr());
  int32_t status = fflush(epoch_file_);
  CHECK(status == 0) << "Could not flush epoch file to disk";
#ifdef __APPLE__
  status = fcntl(fileno(epoch_file_), 51);
#else
  status = omnisci::fsync(fileno(epoch_file_));
#endif
  CHECK(status == 0) << "Could not sync epoch file to disk";
  is_checkpointed_ = true;
}

void TableFileMgr::removeDiskContent() const {
  mapd_unique_lock<mapd_shared_mutex> w_lock(table_mutex_);
  bf::remove_all(table_path_);
}

size_t TableFileMgr::getReservedSpace() const {
  mapd_shared_lock<mapd_shared_mutex> r_lock(table_mutex_);
  size_t space = 0;
  for (const auto& file : bf::recursive_directory_iterator(table_path_)) {
    if (bf::is_regular_file(file.path())) {
      space += bf::file_size(file.path());
    }
  }
  return space;
}

void TableFileMgr::deleteWrapperFile() const {
  mapd_unique_lock<mapd_shared_mutex> w_lock(table_mutex_);
  bf::remove_all(wrapper_file_path_);
}

void TableFileMgr::writeWrapperFile(const std::string& doc) const {
  mapd_unique_lock<mapd_shared_mutex> w_lock(table_mutex_);
  std::ofstream ofs(wrapper_file_path_);
  if (!ofs) {
    throw std::runtime_error{"Error trying to create file \"" + wrapper_file_path_ +
                             "\". The error was: " + std::strerror(errno)};
  }
  ofs << doc;
}

}  // namespace File_Namespace
