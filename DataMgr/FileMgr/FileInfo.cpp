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

#include "FileInfo.h"
#include <iostream>
#include "../../Shared/File.h"
#include "FileMgr.h"
#include "Page.h"

#include <utility>
using namespace std;

namespace File_Namespace {

FileInfo::FileInfo(FileMgr* fileMgr,
                   const int32_t fileId,
                   FILE* f,
                   const size_t pageSize,
                   size_t numPages,
                   bool init)
    : fileMgr(fileMgr), fileId(fileId), f(f), pageSize(pageSize), numPages(numPages) {
  if (init) {
    initNewFile();
  }
}

FileInfo::~FileInfo() {
  // close file, if applicable
  if (f) {
    close(f);
  }
}

void FileInfo::initNewFile() {
  // initialize pages and free page list
  // Also zeroes out first four bytes of every header

  int32_t headerSize = 0;
  int8_t* headerSizePtr = (int8_t*)(&headerSize);
  for (size_t pageId = 0; pageId < numPages; ++pageId) {
    File_Namespace::write(f, pageId * pageSize, sizeof(int32_t), headerSizePtr);
    freePages.insert(pageId);
  }
  isDirty = true;
}

size_t FileInfo::write(const size_t offset, const size_t size, int8_t* buf) {
  std::lock_guard<std::mutex> lock(readWriteMutex_);
  isDirty = true;
  return File_Namespace::write(f, offset, size, buf);
}

size_t FileInfo::read(const size_t offset, const size_t size, int8_t* buf) {
  std::lock_guard<std::mutex> lock(readWriteMutex_);
  return File_Namespace::read(f, offset, size, buf);
}

void FileInfo::openExistingFile(std::vector<HeaderInfo>& headerVec) {
  // HeaderInfo is defined in Page.h

  // Oct 2020: Changing semantics such that fileMgrEpoch should be last checkpointed
  // epoch, not incremented epoch. This changes some of the gt/gte/lt/lte comparison below
  ChunkKey oldChunkKey(4);
  int32_t oldPageId = -99;
  int32_t oldVersionEpoch = -99;
  int32_t skipped = 0;
  for (size_t pageNum = 0; pageNum < numPages; ++pageNum) {
    constexpr size_t MAX_INTS_TO_READ{10};  // currently use 1+6 ints
    int32_t ints[MAX_INTS_TO_READ];
    CHECK_EQ(fseek(f, pageNum * pageSize, SEEK_SET), 0);
    CHECK_EQ(fread(ints, sizeof(int32_t), MAX_INTS_TO_READ, f), MAX_INTS_TO_READ);

    auto headerSize = ints[0];
    if (headerSize == 0) {
      // no header for this page - insert into free list
      freePages.insert(pageNum);
      continue;
    }

    // headerSize doesn't include headerSize itself
    // We're tying ourself to headers of ints here
    size_t numHeaderElems = headerSize / sizeof(int32_t);
    CHECK_GE(numHeaderElems, size_t(2));
    // We don't want to read headerSize in our header - so start
    // reading 4 bytes past it
    ChunkKey chunkKey(&ints[1], &ints[1 + numHeaderElems - 2]);
    if (fileMgr->updatePageIfDeleted(this, chunkKey, ints[1], ints[2], pageNum)) {
      continue;
    }
    // Last two elements of header are always PageId and Version
    // epoch - these are not in the chunk key so seperate them
    int32_t pageId = ints[1 + numHeaderElems - 2];
    int32_t versionEpoch = ints[1 + numHeaderElems - 1];
    if (chunkKey != oldChunkKey || oldPageId != pageId - (1 + skipped)) {
      if (skipped > 0) {
        VLOG(4) << "FId.PSz: " << fileId << "." << pageSize
                << " Chunk key: " << show_chunk(oldChunkKey)
                << " Page id from : " << oldPageId << " to : " << oldPageId + skipped
                << " Epoch: " << oldVersionEpoch;
      } else if (oldPageId != -99) {
        VLOG(4) << "FId.PSz: " << fileId << "." << pageSize
                << " Chunk key: " << show_chunk(oldChunkKey) << " Page id: " << oldPageId
                << " Epoch: " << oldVersionEpoch;
      }
      oldPageId = pageId;
      oldVersionEpoch = versionEpoch;
      oldChunkKey = chunkKey;
      skipped = 0;
    } else {
      skipped++;
    }

    /* Check if version epoch is equal to
     * or greater (note: should never be greater)
     * than FileMgr epoch_ - this means that this
     * page wasn't checkpointed and thus we should
     * not use it
     */
    int32_t fileMgrEpoch =
        fileMgr->epoch(chunkKey[CHUNK_KEY_DB_IDX], chunkKey[CHUNK_KEY_TABLE_IDX]);
    if (versionEpoch > fileMgrEpoch) {
      // First write 0 to first four bytes of
      // header to mark as free
      if (!g_read_only) {
        freePageImmediate(pageNum);
      }
      LOG(WARNING) << "Was not checkpointed: Chunk key: " << show_chunk(chunkKey)
                   << " Page id: " << pageId << " Epoch: " << versionEpoch
                   << " FileMgrEpoch " << fileMgrEpoch << endl;
    } else {  // page was checkpointed properly
      Page page(fileId, pageNum);
      headerVec.emplace_back(chunkKey, pageId, versionEpoch, page);
    }
  }
  // printlast
  if (oldPageId != -99) {
    if (skipped > 0) {
      VLOG(4) << "FId.PSz: " << fileId << "." << pageSize
              << " Chunk key: " << show_chunk(oldChunkKey)
              << " Page id from : " << oldPageId << " to : " << oldPageId + skipped
              << " Epoch: " << oldVersionEpoch;
    } else {
      VLOG(4) << "FId.PSz: " << fileId << "." << pageSize
              << " Chunk key: " << show_chunk(oldChunkKey) << " Page id: " << oldPageId
              << " Epoch: " << oldVersionEpoch;
    }
  }
}

void FileInfo::freePageDeferred(int32_t pageId) {
  std::lock_guard<std::mutex> lock(freePagesMutex_);
  freePages.insert(pageId);
}

#ifdef ENABLE_CRASH_CORRUPTION_TEST
#warning "!!!!! DB corruption crash test is enabled !!!!!"
#include <signal.h>
static bool goto_crash;
static void sighandler(int sig) {
  if (getenv("ENABLE_CRASH_CORRUPTION_TEST"))
    goto_crash = true;
}
#endif

void FileInfo::freePage(int pageId, const bool isRolloff, int32_t epoch) {
  std::lock_guard<std::mutex> lock(readWriteMutex_);
#define RESILIENT_PAGE_HEADER
#ifdef RESILIENT_PAGE_HEADER
  int32_t epoch_freed_page[2] = {DELETE_CONTINGENT, epoch};
  if (isRolloff) {
    epoch_freed_page[0] = ROLLOFF_CONTINGENT;
  }
  File_Namespace::write(f,
                        pageId * pageSize + sizeof(int32_t),
                        sizeof(epoch_freed_page),
                        (int8_t*)epoch_freed_page);
  fileMgr->free_page(std::make_pair(this, pageId));
#else
  freePageImmediate(pageId);
#endif  // RESILIENT_PAGE_HEADER
  isDirty = true;

#ifdef ENABLE_CRASH_CORRUPTION_TEST
  signal(SIGUSR2, sighandler);
  if (goto_crash)
    CHECK(pageId % 8 != 4);
#endif
}

int32_t FileInfo::getFreePage() {
  // returns -1 if there is no free page
  std::lock_guard<std::mutex> lock(freePagesMutex_);
  if (freePages.size() == 0) {
    return -1;
  }
  auto pageIt = freePages.begin();
  int32_t pageNum = *pageIt;
  freePages.erase(pageIt);
  return pageNum;
}

void FileInfo::print(bool pagesummary) {
  std::cout << "File: " << fileId << std::endl;
  std::cout << "Size: " << size() << std::endl;
  std::cout << "Used: " << used() << std::endl;
  std::cout << "Free: " << available() << std::endl;
  if (!pagesummary) {
    return;
  }
}
int32_t FileInfo::syncToDisk() {
  std::lock_guard<std::mutex> lock(readWriteMutex_);
  if (isDirty) {
    if (fflush(f) != 0) {
      LOG(FATAL) << "Error trying to flush changes to disk, the error was: "
                 << std::strerror(errno);
    }
#ifdef __APPLE__
    const int32_t sync_result = fcntl(fileno(f), 51);
#else
    const int32_t sync_result = omnisci::fsync(fileno(f));
#endif
    if (sync_result == 0) {
      isDirty = false;
    }
    return sync_result;
  }
  return 0;  // if file was not dirty and no syncing was needed
}

void FileInfo::freePageImmediate(int32_t page_num) {
  std::lock_guard<std::mutex> lock(freePagesMutex_);
  // we should not get here but putting protection in place
  // as it seems we are no guaranteed to have f/synced so
  // protecting from RO trying to write
  if (!g_read_only) {
    int32_t zero{0};
    File_Namespace::write(
        f, page_num * pageSize, sizeof(int32_t), reinterpret_cast<int8_t*>(&zero));
    freePages.insert(page_num);
  }
}

// Overwrites delete/rollback contingents by re-writing chunk key to page.
void FileInfo::recoverPage(const ChunkKey& chunk_key, int32_t page_num) {
  // we should not get here but putting protection in place
  // as it seems we are no guaranteed to have f/synced so
  // protecting from RO trying to write
  if (!g_read_only) {
    File_Namespace::write(f,
                          page_num * pageSize + sizeof(int32_t),
                          2 * sizeof(int32_t),
                          reinterpret_cast<const int8_t*>(chunk_key.data()));
  }
}
}  // namespace File_Namespace
