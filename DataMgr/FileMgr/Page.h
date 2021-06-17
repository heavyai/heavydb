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
 * @file	Page.h
 * @author 	Steven Stewart <steve@map-d.com>
 * This file contains the declaration and definition of a Page type and a MultiPage type.
 */

#ifndef DATAMGR_MEMORY_FILE_PAGE_H
#define DATAMGR_MEMORY_FILE_PAGE_H

#include "Logger/Logger.h"

#include <cassert>
#include <deque>
#include <stdexcept>
#include <vector>
#include "../../Shared/types.h"

namespace File_Namespace {

/**
 * @struct Page
 * @brief A logical page (Page) belongs to a file on disk.
 *
 * A Page struct stores the file id for the file it belongs to, and it
 * stores its page number and number of used bytes within the page.
 *
 * Note: the number of used bytes should not be greater than the page
 * size. The page size is determined by the containing file.
 */
struct Page {
  int32_t fileId;  /// unique identifier of the owning file
  size_t pageNum;  /// page number

  /// Constructor
  Page(int32_t fileId, size_t pageNum) : fileId(fileId), pageNum(pageNum) {}
  Page() : fileId(-1), pageNum(0) {}

  inline bool isValid() { return fileId >= 0; }

  bool operator<(const Page other) const {
    if (fileId != other.fileId) {
      return fileId < other.fileId;
    }
    return pageNum < other.pageNum;
  }
};

/**
 * @struct MultiPage
 * @brief The MultiPage stores versions of the same logical page in a deque.
 *
 * The purpose of MultiPage is to support storing multiple versions of the same
 * page, which may be located in different locations and in different files.
 * Associated with each version of a page is an "epoch" value, which is a temporal
 * reference.
 *
 */
struct EpochedPage {
  Page page;
  int32_t epoch;
};

struct MultiPage {
  size_t pageSize;
  std::deque<EpochedPage> pageVersions;

  /// Constructor
  MultiPage(size_t pageSizeIn) : pageSize(pageSizeIn) {}

  /// Destructor -- purges all pages
  ~MultiPage() {
    while (pageVersions.size() > 0) {
      pop();
    }
  }

  /// Returns a reference to the most recent version of the page
  inline EpochedPage current() const {
    if (pageVersions.size() < 1) {
      LOG(FATAL) << "No current version of the page exists in this MultiPage.";
    }
    return pageVersions.back();
  }

  /// Pushes a new page with epoch value
  inline void push(const Page& page, const int epoch) {
    if (!pageVersions.empty()) {
      CHECK_GT(epoch, pageVersions.back().epoch);
    }
    pageVersions.push_back({page, epoch});
  }

  /// Purges the oldest Page
  inline void pop() {
    if (pageVersions.size() < 1) {
      LOG(FATAL) << "No page to pop.";
    }
    pageVersions.pop_front();
  }

  std::vector<EpochedPage> freePagesBeforeEpoch(const int32_t target_epoch,
                                                const int32_t current_epoch) {
    std::vector<EpochedPage> pagesBeforeEpoch;
    int32_t next_page_epoch = current_epoch + 1;
    for (auto pageIt = pageVersions.rbegin(); pageIt != pageVersions.rend(); ++pageIt) {
      const int32_t epoch_ceiling = next_page_epoch - 1;
      CHECK_LE(pageIt->epoch, epoch_ceiling);
      if (epoch_ceiling < target_epoch) {
        pagesBeforeEpoch.emplace_back(*pageIt);
      }
      next_page_epoch = pageIt->epoch;
    }
    if (!pagesBeforeEpoch.empty()) {
      pageVersions.erase(pageVersions.begin(),
                         pageVersions.begin() + pagesBeforeEpoch.size());
    }
    return pagesBeforeEpoch;
  }
};

/**
 * @type HeaderInfo
 * @brief Stores Pair of ChunkKey and Page id and version, in a pair with
 * a Page struct itself (File id and Page num)
 */

struct HeaderInfo {
  ChunkKey chunkKey;  // a vector of ints
  int32_t pageId;
  int32_t versionEpoch;
  Page page;

  HeaderInfo(const ChunkKey& chunkKey,
             const int32_t pageId,
             const int32_t versionEpoch,
             const Page& page)
      : chunkKey(chunkKey), pageId(pageId), versionEpoch(versionEpoch), page(page) {}

  bool operator<(const HeaderInfo& other) const {
    if (chunkKey != other.chunkKey) {
      return chunkKey < other.chunkKey;
    }
    if (pageId != other.pageId) {
      return pageId < other.pageId;
    }
    return versionEpoch < other.versionEpoch;
  }
};

}  // namespace File_Namespace

#endif  // DATAMGR_MEMORY_FILE_PAGE_H
