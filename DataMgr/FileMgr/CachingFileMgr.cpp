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
#include "Shared/File.h"

constexpr char EPOCH_FILENAME[] = "epoch_metadata";

namespace File_Namespace {
namespace bf = boost::filesystem;

CachingFileMgr::CachingFileMgr(const std::string& base_path,
                               const size_t num_reader_threads) {
  fileMgrBasePath_ = base_path;
  maxRollbackEpochs_ = 0;
  defaultPageSize_ = DEFAULT_PAGE_SIZE;
  nextFileId_ = 0;
  init(num_reader_threads, -1);
}

bool CachingFileMgr::coreInit() {
  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
  bf::path path(fileMgrBasePath_);
  if (bf::exists(path)) {
    if (!bf::is_directory(path)) {
      LOG(FATAL) << "Specified path '" << fileMgrBasePath_
                 << "' for disk cache is not a directory.";
    }
    migrateToLatestFileMgrVersion();
    openAndReadEpochFile(EPOCH_FILENAME);
    return true;
  }
  LOG(FATAL) << "Cache path: " << fileMgrBasePath_ << "does not exit.";
  return false;
}

void CachingFileMgr::clearForTable(int db_id, int tb_id) {
  {
    mapd_unique_lock<mapd_shared_mutex> write_lock(chunkIndexMutex_);
    for (auto it = chunkIndex_.begin(); it != chunkIndex_.end();) {
      auto& [key, buffer] = *it;
      if (in_same_table(key, {db_id, tb_id})) {
        buffer->freePages();
        delete buffer;
        it = chunkIndex_.erase(it);
      } else {
        ++it;
      }
    }
    auto dir_name = getFileMgrBasePath() + "/" + get_dir_name_for_table(db_id, tb_id);
    if (bf::exists(dir_name)) {
      bf::remove_all(dir_name);
    }
  }
  checkpoint(db_id, tb_id);
  // TODO(Misiu): Implement background file removal.
  // Currently the renameForDelete idiom will only work in the mapd/ directory as the
  // cleanup thread is targetted there.  If we want it to work for arbitrary directories
  // we will need to add a new dir to the thread, or start a second thread.
  // File_Namespace::renameForDelete(get_dir_name_for_table(db_id, tb_id));
}

std::string CachingFileMgr::getOrAddTableDir(int db_id, int tb_id) {
  mapd_unique_lock<mapd_shared_mutex> write_lock(files_rw_mutex_);
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
  auto dir_name = getFileMgrBasePath();
  if (bf::exists(dir_name)) {
    bf::remove_all(dir_name);
  }

  // TODO(Misiu): Implement background file removal.
  // Currently the renameForDelete idiom will only work in the mapd/ directory as the
  // cleanup thread is targetted there.  If we want it to work for arbitrary directories
  // we will need to add a new dir to the thread, or start a second thread.
  // File_Namespace::renameForDelete(getFileMgrBasePath());
}

uint64_t CachingFileMgr::getChunkSpaceReservedByTable(int db_id, int tb_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  uint64_t space_used = 0;
  for (const auto& [key, buffer] : chunkIndex_) {
    if (key[CHUNK_KEY_DB_IDX] == db_id && key[CHUNK_KEY_TABLE_IDX] == tb_id) {
      space_used += buffer->reservedSize();
    }
  }
  return space_used;
}

uint64_t CachingFileMgr::getMetadataSpaceReservedByTable(int db_id, int tb_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(chunkIndexMutex_);
  uint64_t space_used = 0;
  for (const auto& [key, buffer] : chunkIndex_) {
    if (key[CHUNK_KEY_DB_IDX] == db_id && key[CHUNK_KEY_TABLE_IDX] == tb_id) {
      space_used += (buffer->numMetadataPages() * METADATA_PAGE_SIZE);
    }
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
  auto chunkSpace = getChunkSpaceReservedByTable(db_id, tb_id);
  auto metaSpace = getMetadataSpaceReservedByTable(db_id, tb_id);
  auto wrapperSpace = getWrapperSpaceReservedByTable(db_id, tb_id);
  return chunkSpace + metaSpace + wrapperSpace;
}

std::string CachingFileMgr::describeSelf() {
  return "cache";
}

// Similar to FileMgr::checkpoint() but only writes/rolloffs a subset of buffers.
void CachingFileMgr::checkpoint(const int32_t db_id, const int32_t tb_id) {
  VLOG(2) << "Checkpointing " << describeSelf() << " (" << db_id << ", " << tb_id
          << " epoch: " << epoch();
  {
    mapd_unique_lock<mapd_shared_mutex> chunk_index_write_lock(chunkIndexMutex_);
    ChunkKey min_table_key{db_id, tb_id};
    ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};

    for (auto chunkIt = chunkIndex_.lower_bound(min_table_key);
         chunkIt != chunkIndex_.upper_bound(max_table_key);
         ++chunkIt) {
      if (chunkIt->second->isDirty()) {
        chunkIt->second->writeMetadata(epoch());
        chunkIt->second->clearDirtyBits();
      }
    }
  }

  syncFilesToDisk();
  writeAndSyncEpochToDisk();
  incrementEpoch();
  rollOffOldData(db_id, tb_id, lastCheckpointedEpoch());
  freePages();
}

void CachingFileMgr::rollOffOldData(const int32_t db_id,
                                    const int32_t tb_id,
                                    const int32_t epoch_ceiling) {
  if (maxRollbackEpochs_ >= 0) {
    auto min_epoch = std::max(epoch_ceiling - maxRollbackEpochs_, epoch_.floor());
    if (min_epoch > epoch_.floor()) {
      freePagesBeforeEpoch(db_id, tb_id, min_epoch);
      epoch_.floor(min_epoch);
    }
  }
}

void CachingFileMgr::freePagesBeforeEpoch(const int32_t db_id,
                                          const int32_t tb_id,
                                          const int32_t min_rollback_epoch) {
  mapd_shared_lock<mapd_shared_mutex> chunk_index_read_lock(chunkIndexMutex_);
  ChunkKey min_table_key{db_id, tb_id};
  ChunkKey max_table_key{db_id, tb_id, std::numeric_limits<int32_t>::max()};
  freePagesBeforeEpochUnlocked(min_rollback_epoch,
                               chunkIndex_.lower_bound(min_table_key),
                               chunkIndex_.upper_bound(max_table_key));
}

}  // namespace File_Namespace
