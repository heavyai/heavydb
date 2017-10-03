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
 * @file        GlobalFileMgr.cpp
 * @author      Norair Khachiyan <norair@map-d.com>
 * @author      Todd Mostak <todd@map-d.com>
 */

#include "GlobalFileMgr.h"
#include "File.h"
#include <string>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <vector>
#include <utility>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <thread>

using namespace std;

namespace File_Namespace {

GlobalFileMgr::GlobalFileMgr(const int deviceId,
                             std::string basePath,
                             const size_t num_reader_threads,
                             const size_t defaultPageSize)
    : AbstractBufferMgr(deviceId),
      basePath_(basePath),
      num_reader_threads_(num_reader_threads),
      epoch_(-1),  // set the default epoch for all tables corresponding to the time of last checkpoint
      defaultPageSize_(defaultPageSize) {
  mapd_db_version_ = 1;  // DS changes triggered by individual FileMgr per table project (release 2.1.0)
  dbConvert_ = false;
  init();
}

GlobalFileMgr::~GlobalFileMgr() {
  mapd_lock_guard<mapd_shared_mutex> fileMgrsMutex(fileMgrs_mutex_);
  for (auto fileMgrsIt = fileMgrs_.begin(); fileMgrsIt != fileMgrs_.end(); ++fileMgrsIt) {
    delete fileMgrsIt->second;
  }
}

void GlobalFileMgr::init() {
  // check if basePath_ already exists, and if not create one
  boost::filesystem::path path(basePath_);
  if (basePath_.size() > 0 && basePath_[basePath_.size() - 1] != '/')
    basePath_.push_back('/');
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
  mapd_lock_guard<mapd_shared_mutex> fileMgrsMutex(fileMgrs_mutex_);
  for (auto fileMgrsIt = fileMgrs_.begin(); fileMgrsIt != fileMgrs_.end(); ++fileMgrsIt) {
    fileMgrsIt->second->checkpoint();
  }
}

void GlobalFileMgr::checkpoint(const int db_id, const int tb_id) {
  getFileMgr(db_id, tb_id)->checkpoint();
}

size_t GlobalFileMgr::getNumChunks() {
  {
    mapd_shared_lock<mapd_shared_mutex> fileMgrsMutex(fileMgrs_mutex_);
    size_t num_chunks = 0;
    for (auto fileMgrsIt = fileMgrs_.begin(); fileMgrsIt != fileMgrs_.end(); ++fileMgrsIt) {
      num_chunks += fileMgrsIt->second->getNumChunks();
    }

    return num_chunks;
  }
}

void GlobalFileMgr::deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool purge) {
  /* keyPrefix[0] can be -1 only for gpu or cpu buffers but not for FileMgr.
   * There is no assert here, as GlobalFileMgr is being called with -1 value as well in the same
   * loop with other buffers. So the case of -1 will just be ignored, as nothing needs to be done.
   */
  if (keyPrefix[0] != -1) {
    return getFileMgr(keyPrefix)->deleteBuffersWithPrefix(keyPrefix, purge);
  }
}

void GlobalFileMgr::getChunkMetadataVec(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec) {
  mapd_shared_lock<mapd_shared_mutex> fileMgrsMutex(fileMgrs_mutex_);
  std::vector<std::pair<ChunkKey, ChunkMetadata>> chunkMetadataVecForFileMgr;
  for (auto fileMgrsIt = fileMgrs_.begin(); fileMgrsIt != fileMgrs_.end(); ++fileMgrsIt) {
    fileMgrsIt->second->getChunkMetadataVec(chunkMetadataVecForFileMgr);
    while (!chunkMetadataVecForFileMgr.empty()) {
      // norair - order of elements is reversed, consider optimising this later if needed
      chunkMetadataVec.push_back(chunkMetadataVecForFileMgr.back());
      chunkMetadataVecForFileMgr.pop_back();
    }
  }
}

FileMgr* GlobalFileMgr::findFileMgr(const int db_id, const int tb_id, const bool removeFromMap) {
  FileMgr* fm = nullptr;
  const auto file_mgr_key = std::make_pair(db_id, tb_id);
  {
    mapd_lock_guard<mapd_shared_mutex> read_lock(fileMgrs_mutex_);
    auto it = fileMgrs_.find(file_mgr_key);
    if (it != fileMgrs_.end()) {
      fm = it->second;
      if (removeFromMap) {
        fileMgrs_.erase(it);
      }
    }
  }
  return fm;
}

FileMgr* GlobalFileMgr::getFileMgr(const int db_id, const int tb_id) {
  { /* check if FileMgr already exists for (db_id, tb_id) */
    FileMgr* fm = findFileMgr(db_id, tb_id);
    if (fm != nullptr) {
      return fm;
    }
  }

  { /* create new FileMgr for (db_id, tb_id) */
    const auto file_mgr_key = std::make_pair(db_id, tb_id);
    mapd_lock_guard<mapd_shared_mutex> write_lock(fileMgrs_mutex_);
    auto it = fileMgrs_.find(file_mgr_key);
    if (it != fileMgrs_.end()) {
      return it->second;
    }
    FileMgr* fm = new FileMgr(0, this, file_mgr_key, num_reader_threads_, epoch_, defaultPageSize_);
    auto it_ok = fileMgrs_.insert(std::make_pair(file_mgr_key, fm));
    CHECK(it_ok.second);

    return fm;
  }
}

void GlobalFileMgr::writeFileMgrData(FileMgr* fileMgr) {  // this function is not used, keep it for now for future needs
  for (auto fileMgrIt = fileMgrs_.begin(); fileMgrIt != fileMgrs_.end(); fileMgrIt++) {
    FileMgr* fm = fileMgrIt->second;
    if ((fileMgr != 0) && (fileMgr != fm)) {
      continue;
    }
    for (auto chunkIt = fm->chunkIndex_.begin(); chunkIt != fm->chunkIndex_.end(); chunkIt++) {
      chunkIt->second->write((int8_t*)chunkIt->second, chunkIt->second->size(), 0);
      // chunkIt->second->write((int8_t*)chunkIt->second, chunkIt->second->size(), 0, CPU_LEVEL, -1);
    }
  }
}

void GlobalFileMgr::removeTableRelatedDS(const int db_id, const int tb_id) {
  FileMgr* fm = findFileMgr(db_id, tb_id, true);
  if (fm == nullptr) {
    LOG(FATAL) << "Drop table failed. Table " << db_id << " " << tb_id << " does not exist.";
  }

  /* remove directory containing table related data */
  boost::system::error_code ec;
  boost::filesystem::path pathToTableDS(fm->getFileMgrBasePath());
  boost::filesystem::remove_all(pathToTableDS, ec);

  /* remove table related in-memory DS only if directory was removed successfully */
  if (ec == 0) {
    delete fm;
  }
}

void GlobalFileMgr::setTableEpoch(const int db_id, const int tb_id, const int start_epoch) {
  const auto file_mgr_key = std::make_pair(db_id, tb_id);
  // this is where the real rollback of any data ahead of the currently set epoch is performed
  FileMgr* fm = new FileMgr(0, this, file_mgr_key, num_reader_threads_, start_epoch, defaultPageSize_);
  fm->setEpoch(start_epoch - 1);
  // remove the dummy one we built
  delete fm;

  // see if one exists currently, and remove it
  fm = findFileMgr(db_id, tb_id, true);

  if (fm != nullptr) {
    LOG(INFO) << "found and removed fm";
    delete fm;
  }
}

size_t GlobalFileMgr::getTableEpoch(const int db_id, const int tb_id) {
  FileMgr* fm = getFileMgr(db_id, tb_id);

  return fm->epoch_;
}

}  // File_Namespace
