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
 * @file	CachingFileMgr.h
 *
 * This file details an extension of the FileMgr that can contain pages from multiple
 * tables (CachingFileMgr).
 */

#pragma once

#include "FileMgr.h"

namespace File_Namespace {

inline std::string get_dir_name_for_table(int db_id, int tb_id) {
  std::stringstream file_name;
  file_name << "table_" << db_id << "_" << tb_id << "/";
  return file_name.str();
}

/**
 * @class   CachingFileMgr
 * @brief   A FileMgr capable of limiting it's size and storing data from multiple tables
 * in a shared directory.  For any table that supports DiskCaching, the CachingFileMgr
 * must contain either metadata for all table chunks, or for none (the cache is either has
 * no knowledge of that table, or has complete knowledge of that table).  Any data chunk
 * within a table may or may not be contained within the cache.
 */
class CachingFileMgr : public FileMgr {
 public:
  CachingFileMgr(const std::string& base_path, const size_t num_reader_threads = 0);
  ~CachingFileMgr() {}
  /**
   * @brief Determines file path, and if exists, runs file migration and opens and reads
   * epoch file
   * @return a boolean representing whether the directory path existed
   */
  bool coreInit() override;

  // Simple getters.
  inline MgrType getMgrType() override { return CACHING_FILE_MGR; };
  inline std::string getStringMgrType() override { return ToString(CACHING_FILE_MGR); }
  inline size_t getDefaultPageSize() { return defaultPageSize_; }

  // TODO(Misiu): These are unimplemented for now, but will become necessary when we want
  // to limit the size.
  inline size_t getMaxSize() override {
    UNREACHABLE() << "Unimplemented";
    return 0;
  }
  inline size_t getInUseSize() override {
    UNREACHABLE() << "Unimplemented";
    return 0;
  }
  inline size_t getAllocated() override {
    UNREACHABLE() << "Unimplemented";
    return 0;
  }
  inline bool isAllocationCapped() override { return false; }

  /**
   * @brief Removes all data related to the given table (pages and subdirectories).
   */
  void clearForTable(int db_id, int tb_id);

  /**
   * @brief Returns (and optionally creates) a subdirectory for table-specific persistent
   * data (e.g. serialized foreign data warppers).
   */
  std::string getOrAddTableDir(int db_id, int tb_id);

  /**
   * @brief Query to determine if the contained pages will have their database and table
   * ids overriden by the filemgr key (FileMgr does this).
   */
  inline bool hasFileMgrKey() const override { return false; }
  /**
   * @breif Closes files and removes the caching directory.
   */
  void closeRemovePhysical() override;

  /**
   * Set of functions to determine how much space is reserved in a table by type.
   */
  uint64_t getChunkSpaceReservedByTable(int db_id, int tb_id);
  uint64_t getMetadataSpaceReservedByTable(int db_id, int tb_id);
  uint64_t getWrapperSpaceReservedByTable(int db_id, int tb_id);
  uint64_t getSpaceReservedByTable(int db_id, int tb_id);

  std::string describeSelf() override;
};

}  // namespace File_Namespace
