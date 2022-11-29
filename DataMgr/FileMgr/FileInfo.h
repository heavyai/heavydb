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

#pragma once

#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#ifdef __APPLE__
#include <fcntl.h>
#endif

#include "../../Shared/types.h"
#include "Logger/Logger.h"
#include "OSDependent/heavyai_fs.h"
#include "Page.h"
namespace File_Namespace {

struct Page;

/**
 * @type FileInfo
 * @brief A FileInfo type has a file pointer and metadata about a file.
 *
 * A file info structure wraps around a file pointer in order to contain additional
 * information/metadata about the file that is pertinent to the file manager.
 *
 * The free pages (freePages) within a file must be tracked, and this is implemented using
 * a basic STL set. The set ensures that no duplicate pages are included, and that the
 * pages are sorted, faciliating the obtaining of consecutive free pages by a constant
 * time pop operation, which may reduce the cost of DBMS disk accesses.
 *
 * Helper functions are provided: size(), available(), and used().
 */
constexpr int32_t DELETE_CONTINGENT = -1;
constexpr int32_t ROLLOFF_CONTINGENT = -2;

class FileMgr;
struct FileInfo {
  FileMgr* fileMgr;
  int32_t fileId;              /// unique file identifier (i.e., used for a file name)
  FILE* f;                     /// file stream object for the represented file
  size_t pageSize;             /// the fixed size of each page in the file
  size_t numPages;             /// the number of pages in the file
  bool isDirty{false};         // True if writes have occured since last sync
  std::set<size_t> freePages;  /// set of page numbers of free pages
  std::string file_path;
  mutable std::mutex freePagesMutex_;
  mutable std::mutex readWriteMutex_;

  /// Constructor
  FileInfo(FileMgr* fileMgr,
           const int32_t fileId,
           FILE* f,
           const size_t pageSize,
           const size_t numPages,
           const std::string& file_path,
           const bool init = false);

  /// Destructor
  ~FileInfo();

  /// Adds all pages to freePages and zeroes first four bytes of header
  // for each apge
  void initNewFile();

  void freePageDeferred(int32_t pageId);
  void freePage(int32_t pageId, const bool isRolloff, int32_t epoch);
  int32_t getFreePage();
  size_t write(const size_t offset, const size_t size, const int8_t* buf);
  size_t read(const size_t offset, const size_t size, int8_t* buf);

  void openExistingFile(std::vector<HeaderInfo>& headerVec);

  /// Prints a summary of the file to stdout
  std::string print() const;

  /// Returns the number of bytes used by the file
  inline size_t size() const { return pageSize * numPages; }

  /// Syncs file to disk via a buffer flush and then a sync (fflush and fsync on posix
  /// systems)
  int32_t syncToDisk();

  /// Returns the number of free bytes available
  inline size_t available() const { return numFreePages() * pageSize; }

  /// Returns the number of free pages available
  inline size_t numFreePages() const {
    std::lock_guard<std::mutex> lock(freePagesMutex_);
    return freePages.size();
  }

  inline std::set<size_t> getFreePages() const {
    std::lock_guard<std::mutex> lock(freePagesMutex_);
    return freePages;
  }

  /// Returns the amount of used bytes; size() - available()
  inline size_t used() const { return size() - available(); }

  void freePageImmediate(int32_t page_num);
  void recoverPage(const ChunkKey& chunk_key, int32_t page_num);
};

bool is_page_deleted_with_checkpoint(int32_t table_epoch,
                                     int32_t page_epoch,
                                     int32_t contingent);

bool is_page_deleted_without_checkpoint(int32_t table_epoch,
                                        int32_t page_epoch,
                                        int32_t contingent);

}  // namespace File_Namespace
