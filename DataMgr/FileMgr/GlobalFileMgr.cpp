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
 * @file        GlobalFileMgr.cpp
 * @brief
 *
 */

#include "DataMgr/FileMgr/GlobalFileMgr.h"

#include <fcntl.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"
#include "Shared/File.h"

using namespace std;

namespace File_Namespace {

GlobalFileMgr::GlobalFileMgr(const int32_t device_id,
                             std::shared_ptr<ForeignStorageInterface> fsi,
                             std::string base_path,
                             const size_t num_reader_threads,
                             const size_t page_size,
                             const size_t metadata_page_size)
    : AbstractBufferMgr(device_id)
    , fsi_(fsi)
    , basePath_(base_path)
    , num_reader_threads_(num_reader_threads)
    , epoch_(-1)  // set the default epoch for all tables corresponding to the time of
                  // last checkpoint
    , page_size_(page_size)
    , metadata_page_size_(metadata_page_size) {
  // DS changes also triggered by individual FileMgr per table project (release 2.1.0)
  dbConvert_ = false;
  init();
}

void GlobalFileMgr::init() {
  // check if basePath_ already exists, and if not create one
  boost::filesystem::path path(basePath_);
  if (basePath_.size() > 0 && basePath_[basePath_.size() - 1] != '/') {
    basePath_.push_back('/');
  }
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path)) {
      LOG(FATAL) << "Specified path is not a directory.";
    }
  } else {  // data directory does not exist
    if (!boost::filesystem::create_directory(path)) {
      LOG(FATAL) << "Could not create data directory";
    }
  }
}

void GlobalFileMgr::checkpoint() {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(fileMgrs_mutex_);
  for (auto fileMgrsIt = allFileMgrs_.begin(); fileMgrsIt != allFileMgrs_.end();
       ++fileMgrsIt) {
    fileMgrsIt->second->checkpoint();
  }
}

void GlobalFileMgr::checkpoint(const int32_t db_id, const int32_t tb_id) {
  getFileMgr(db_id, tb_id)->checkpoint();
}

size_t GlobalFileMgr::getNumChunks() {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(fileMgrs_mutex_);
  size_t num_chunks = 0;
  for (auto fileMgrsIt = allFileMgrs_.begin(); fileMgrsIt != allFileMgrs_.end();
       ++fileMgrsIt) {
    num_chunks += fileMgrsIt->second->getNumChunks();
  }

  return num_chunks;
}

void GlobalFileMgr::deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool purge) {
  /* keyPrefix[0] can be -1 only for gpu or cpu buffers but not for FileMgr.
   * There is no assert here, as GlobalFileMgr is being called with -1 value as well in
   * the same loop with other buffers. So the case of -1 will just be ignored, as nothing
   * needs to be done.
   */
  if (keyPrefix[0] != -1) {
    return getFileMgr(keyPrefix)->deleteBuffersWithPrefix(keyPrefix, purge);
  }
}

AbstractBufferMgr* GlobalFileMgr::findFileMgrUnlocked(const int32_t db_id,
                                                      const int32_t tb_id) {
  // NOTE: only call this private function after locking is already in place
  AbstractBufferMgr* fm = nullptr;
  const auto file_mgr_key = std::make_pair(db_id, tb_id);
  if (auto it = allFileMgrs_.find(file_mgr_key); it != allFileMgrs_.end()) {
    fm = it->second;
  }
  return fm;
}

void GlobalFileMgr::deleteFileMgr(const int32_t db_id, const int32_t tb_id) {
  // NOTE: only call this private function after locking is already in place
  const auto file_mgr_key = std::make_pair(db_id, tb_id);
  if (auto it = ownedFileMgrs_.find(file_mgr_key); it != ownedFileMgrs_.end()) {
    ownedFileMgrs_.erase(it);
  }
  if (auto it = allFileMgrs_.find(file_mgr_key); it != allFileMgrs_.end()) {
    allFileMgrs_.erase(it);
  }
}

void GlobalFileMgr::closeFileMgr(const int32_t db_id, const int32_t tb_id) {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(fileMgrs_mutex_);
  deleteFileMgr(db_id, tb_id);
}

bool GlobalFileMgr::existsDiffBetweenFileMgrParamsAndFileMgr(
    FileMgr* file_mgr,
    const FileMgrParams& file_mgr_params) const {
  if (file_mgr_params.epoch != -1 &&
      file_mgr_params.epoch != file_mgr->lastCheckpointedEpoch()) {
    return true;
  }
  if (file_mgr_params.max_rollback_epochs != -1 &&
      file_mgr_params.max_rollback_epochs != file_mgr->maxRollbackEpochs()) {
    return true;
  }
  return false;
}

void GlobalFileMgr::setFileMgrParams(const int32_t db_id,
                                     const int32_t tb_id,
                                     const FileMgrParams& file_mgr_params) {
  auto fm = dynamic_cast<File_Namespace::FileMgr*>(findFileMgr(db_id, tb_id));
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(fileMgrs_mutex_);
  if (fm) {
    deleteFileMgr(db_id, tb_id);
  }
  const auto file_mgr_key = std::make_pair(db_id, tb_id);
  auto max_rollback_epochs =
      (file_mgr_params.max_rollback_epochs >= 0 ? file_mgr_params.max_rollback_epochs
                                                : -1);
  auto s = std::make_shared<FileMgr>(
      0,
      this,
      file_mgr_key,
      max_rollback_epochs,
      num_reader_threads_,
      file_mgr_params.epoch != -1 ? file_mgr_params.epoch : epoch_);
  CHECK(ownedFileMgrs_.insert(std::make_pair(file_mgr_key, s)).second);
  CHECK(allFileMgrs_.insert(std::make_pair(file_mgr_key, s.get())).second);
  max_rollback_epochs_per_table_[file_mgr_key] = max_rollback_epochs;
  lazy_initialized_stats_.erase(file_mgr_key);
  return;
}

AbstractBufferMgr* GlobalFileMgr::getFileMgr(const int32_t db_id, const int32_t tb_id) {
  {  // check if FileMgr already exists for (db_id, tb_id)
    heavyai::shared_lock<heavyai::shared_mutex> read_lock(fileMgrs_mutex_);
    AbstractBufferMgr* fm = findFileMgrUnlocked(db_id, tb_id);
    if (fm) {
      return fm;
    }
  }

  {  // create new FileMgr for (db_id, tb_id)
    heavyai::unique_lock<heavyai::shared_mutex> write_lock(fileMgrs_mutex_);
    AbstractBufferMgr* fm = findFileMgrUnlocked(db_id, tb_id);
    if (fm) {
      return fm;  // mgr was added between the read lock and the write lock
    }
    const auto file_mgr_key = std::make_pair(db_id, tb_id);
    const auto foreign_buffer_manager = fsi_->lookupBufferManager(db_id, tb_id);
    if (foreign_buffer_manager) {
      CHECK(allFileMgrs_.insert(std::make_pair(file_mgr_key, foreign_buffer_manager))
                .second);
      return foreign_buffer_manager;
    } else {
      int32_t max_rollback_epochs{-1};
      if (max_rollback_epochs_per_table_.find(file_mgr_key) !=
          max_rollback_epochs_per_table_.end()) {
        max_rollback_epochs = max_rollback_epochs_per_table_[file_mgr_key];
      }
      auto s = std::make_shared<FileMgr>(
          0, this, file_mgr_key, max_rollback_epochs, num_reader_threads_, epoch_);
      CHECK(ownedFileMgrs_.insert(std::make_pair(file_mgr_key, s)).second);
      CHECK(allFileMgrs_.insert(std::make_pair(file_mgr_key, s.get())).second);
      lazy_initialized_stats_.erase(file_mgr_key);
      return s.get();
    }
  }
}

// For testing purposes only
std::shared_ptr<FileMgr> GlobalFileMgr::getSharedFileMgr(const int db_id,
                                                         const int table_id) {
  const auto table_key = std::make_pair(db_id, table_id);
  if (ownedFileMgrs_.find(table_key) == ownedFileMgrs_.end()) {
    return nullptr;
  }
  return ownedFileMgrs_[table_key];
}

// For testing purposes only
void GlobalFileMgr::setFileMgr(const int db_id,
                               const int table_id,
                               std::shared_ptr<FileMgr> file_mgr) {
  TablePair file_mgr_key{db_id, table_id};
  allFileMgrs_[file_mgr_key] = file_mgr.get();
  ownedFileMgrs_[file_mgr_key] = file_mgr;
  lazy_initialized_stats_.erase(file_mgr_key);
}

void GlobalFileMgr::writeFileMgrData(
    FileMgr* fileMgr) {  // this function is not used, keep it for now for future needs
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(fileMgrs_mutex_);
  for (auto fileMgrIt = allFileMgrs_.begin(); fileMgrIt != allFileMgrs_.end();
       fileMgrIt++) {
    FileMgr* fm = dynamic_cast<FileMgr*>(fileMgrIt->second);
    CHECK(fm);
    if ((fileMgr != 0) && (fileMgr != fm)) {
      continue;
    }
    for (auto chunkIt = fm->chunkIndex_.begin(); chunkIt != fm->chunkIndex_.end();
         chunkIt++) {
      chunkIt->second->write((int8_t*)chunkIt->second, chunkIt->second->size(), 0);
    }
  }
}

void GlobalFileMgr::removeTableRelatedDS(const int32_t db_id, const int32_t tb_id) {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(fileMgrs_mutex_);
  auto abm = findFileMgrUnlocked(db_id, tb_id);
  if (auto fm = dynamic_cast<File_Namespace::FileMgr*>(abm)) {
    fm->closeRemovePhysical();
  } else if (dynamic_cast<ForeignStorageBufferMgr*>(abm)) {
    abm->removeTableRelatedDS(db_id, tb_id);
    fsi_->dropBufferManager(db_id, tb_id);
  } else {
    // fileMgr has not been initialized so there is no need to
    // spend the time initializing
    // initialize just enough to have to rename
    const auto file_mgr_key = std::make_pair(db_id, tb_id);
    auto u = std::make_unique<FileMgr>(0, this, file_mgr_key, true);
    u->closeRemovePhysical();
  }
  // remove table related in-memory DS only if directory was removed successfully

  deleteFileMgr(db_id, tb_id);
  max_rollback_epochs_per_table_.erase({db_id, tb_id});
}

void GlobalFileMgr::setTableEpoch(const int32_t db_id,
                                  const int32_t tb_id,
                                  const int32_t start_epoch) {
  AbstractBufferMgr* opened_fm = findFileMgr(db_id, tb_id);
  if (opened_fm) {
    // Delete this FileMgr to ensure epoch change occurs in constructor with other
    // reads/writes locked out
    deleteFileMgr(db_id, tb_id);
  }
  const auto file_mgr_key = std::make_pair(db_id, tb_id);
  // this is where the real rollback of any data ahead of the currently set epoch is
  // performed
  // Will call set_epoch with start_epoch internally
  auto u = std::make_unique<FileMgr>(
      0, this, file_mgr_key, -1, num_reader_threads_, start_epoch);
  // remove the dummy one we built
  u.reset();
}

size_t GlobalFileMgr::getTableEpoch(const int32_t db_id, const int32_t tb_id) {
  // UX change was made to this function Oct 2020 to return checkpointed epoch. In turn,
  // setTableEpoch was changed to set the epoch at the user's input, instead of input - 1
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(fileMgrs_mutex_);
  AbstractBufferMgr* opened_fm = findFileMgr(db_id, tb_id);
  if (opened_fm) {
    return dynamic_cast<FileMgr*>(opened_fm)->lastCheckpointedEpoch();
  }
  // Do not do full init of table just to get table epoch, just check file instead
  const auto file_mgr_key = std::make_pair(db_id, tb_id);
  auto u = std::make_unique<FileMgr>(0, this, file_mgr_key, true);
  const auto epoch = u->lastCheckpointedEpoch();
  u.reset();
  return epoch;
}

void GlobalFileMgr::resetTableEpochFloor(const int32_t db_id, const int32_t tb_id) {
  AbstractBufferMgr* fm = getFileMgr(db_id, tb_id);
  CHECK(fm);
  dynamic_cast<FileMgr*>(fm)->resetEpochFloor();
}

StorageStats GlobalFileMgr::getStorageStats(const int32_t db_id, const int32_t tb_id) {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(fileMgrs_mutex_);
  AbstractBufferMgr* opened_fm = findFileMgr(db_id, tb_id);
  if (opened_fm) {
    return dynamic_cast<FileMgr*>(opened_fm)->getStorageStats();
  }
  TablePair file_mgr_key{db_id, tb_id};
  auto it = lazy_initialized_stats_.find(file_mgr_key);
  if (it != lazy_initialized_stats_.end()) {
    return it->second;
  } else {
    // Do not do full init of table just to get storage stats, just check file instead
    auto u = std::make_unique<FileMgr>(0, this, file_mgr_key, true);
    lazy_initialized_stats_[file_mgr_key] = u->getStorageStats();
    u.reset();
    return lazy_initialized_stats_[file_mgr_key];
  }
}

void GlobalFileMgr::compactDataFiles(const int32_t db_id, const int32_t tb_id) {
  auto file_mgr = dynamic_cast<File_Namespace::FileMgr*>(findFileMgr(db_id, tb_id));
  {
    heavyai::unique_lock<heavyai::shared_mutex> write_lock(fileMgrs_mutex_);
    if (file_mgr) {
      file_mgr->compactFiles();
      deleteFileMgr(db_id, tb_id);
    }
  }

  // Re-initialize file manager
  getFileMgr(db_id, tb_id);
}
}  // namespace File_Namespace
