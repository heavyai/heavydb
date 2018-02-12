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
#include "FileMgr.h"
#include "File.h"
#include "Page.h"
#include <glog/logging.h>
#include <iostream>

#include <utility>
using namespace std;

namespace File_Namespace {

FileInfo::FileInfo(FileMgr* fileMgr, const int fileId, FILE* f, const size_t pageSize, size_t numPages, bool init)
    : fileMgr(fileMgr), fileId(fileId), f(f), pageSize(pageSize), numPages(numPages) {
  if (init) {
    initNewFile();
  }
}

FileInfo::~FileInfo() {
  // close file, if applicable
  if (f)
    close(f);
}

void FileInfo::initNewFile() {
  // initialize pages and free page list
  // Also zeroes out first four bytes of every header

  int headerSize = 0;
  int8_t* headerSizePtr = (int8_t*)(&headerSize);
  for (size_t pageId = 0; pageId < numPages; ++pageId) {
    File_Namespace::write(f, pageId * pageSize, sizeof(int), headerSizePtr);
    freePages.insert(pageId);
  }
}

size_t FileInfo::write(const size_t offset, const size_t size, int8_t* buf) {
  std::lock_guard<std::mutex> lock(readWriteMutex_);
  return File_Namespace::write(f, offset, size, buf);
}

size_t FileInfo::read(const size_t offset, const size_t size, int8_t* buf) {
  std::lock_guard<std::mutex> lock(readWriteMutex_);
  return File_Namespace::read(f, offset, size, buf);
}

void FileInfo::openExistingFile(std::vector<HeaderInfo>& headerVec, const int fileMgrEpoch) {
  // HeaderInfo is defined in Page.h
  ChunkKey oldChunkKey(4);
  int oldPageId = -99;
  int oldVersionEpoch = -99;
  int skipped = 0;
  for (size_t pageNum = 0; pageNum < numPages; ++pageNum) {
    int headerSize;

#define MAX_INTS_TO_READ 10  // currently use 1+6 ints
    int ints[MAX_INTS_TO_READ];
    fseek(f, pageNum * pageSize, SEEK_SET);
    fread(ints, sizeof(int), MAX_INTS_TO_READ, f);

    headerSize = ints[0];
    if (0 != headerSize)
      if (DELETE_CONTINGENT == ints[1])
        if (fileMgr->epoch() > ints[2]) {
          int zero{0};
          File_Namespace::write(f, pageNum * pageSize, sizeof(int), (int8_t*)&zero);
          headerSize = 0;
        }

    if (headerSize != 0) {
      // headerSize doesn't include headerSize itself
      // We're tying ourself to headers of ints here
      size_t numHeaderElems = headerSize / sizeof(int);
      assert(numHeaderElems >= 2);
      // size_t chunkSize;
      // We don't want to read headerSize in our header - so start
      // reading 4 bytes past it

      // always derive dbid/tbid from FileMgr
      ChunkKey chunkKey(&ints[1], &ints[1 + numHeaderElems - 2]);
      chunkKey[0] = fileMgr->get_fileMgrKey().first;
      chunkKey[1] = fileMgr->get_fileMgrKey().second;
      // recover page in case a crash failed deletion of this page
      if (DELETE_CONTINGENT == ints[1])
        File_Namespace::write(f, pageNum * pageSize + sizeof(int), 2 * sizeof(int), (int8_t*)&chunkKey[0]);

      // cout << "Chunk key: " << showChunk(chunkKey) << endl;
      // Last two elements of header are always PageId and Version
      // epoch - these are not in the chunk key so seperate them
      int pageId = ints[1 + numHeaderElems - 2];
      // cout << "Page id: " << pageId << endl;
      int versionEpoch = ints[1 + numHeaderElems - 1];
      if (chunkKey != oldChunkKey || oldPageId != pageId - (1 + skipped)) {
        if (skipped > 0) {
          VLOG(1) << "FId.PSz: " << fileId << "." << pageSize << " Chunk key: " << showChunk(oldChunkKey)
                  << " Page id from : " << oldPageId << " to : " << oldPageId + skipped
                  << " Epoch: " << oldVersionEpoch;
        } else if (oldPageId != -99) {
          VLOG(1) << "FId.PSz: " << fileId << "." << pageSize << " Chunk key: " << showChunk(oldChunkKey)
                  << " Page id: " << oldPageId << " Epoch: " << oldVersionEpoch;
        }
        oldPageId = pageId;
        oldVersionEpoch = versionEpoch;
        oldChunkKey = chunkKey;
        skipped = 0;
      } else {
        skipped++;
      }
      // read(f,pageNum*pageSize+sizeof(int),headerSize-2*sizeof(int),(int8_t *)(&chunkKey[0]));
      // read(f,pageNum*pageSize+sizeof(int) + headerSize - 2*sizeof(int),sizeof(int),(int8_t *)(&pageId));
      // read(f,pageNum*pageSize+sizeof(int) + headerSize - sizeof(int),sizeof(int),(int8_t *)(&versionEpoch));
      // read(f,pageNum*pageSize+sizeof(int) + headerSize - sizeof(size_t),sizeof(size_t),(int8_t *)(&chunkSize));

      /* Check if version epoch is equal to
       * or greater (note: should never be greater)
       * than FileMgr epoch_ - this means that this
       * page wasn't checkpointed and thus we should
       * not use it
       */
      if (versionEpoch >= fileMgrEpoch) {
        // First write 0 to first four bytes of
        // header to mark as free
        headerSize = 0;
        File_Namespace::write(f, pageNum * pageSize, sizeof(int), (int8_t*)&headerSize);
        // Now add page to free list
        freePages.insert(pageNum);
        LOG(WARNING) << "Was not checkpointed: Chunk key: " << showChunk(chunkKey) << " Page id: " << pageId
                     << " Epoch: " << versionEpoch << " FileMgrEpoch " << fileMgrEpoch << endl;

      } else {  // page was checkpointed properly
        Page page(fileId, pageNum);
        headerVec.push_back(HeaderInfo(chunkKey, pageId, versionEpoch, page));
        // std::cout << "Inserted into headerVec" << std::endl;
      }
    } else {  // no header for this page - insert into free list
      freePages.insert(pageNum);
    }
  }
  // printlast
  if (oldPageId != -99) {
    if (skipped > 0) {
      VLOG(1) << "FId.PSz: " << fileId << "." << pageSize << " Chunk key: " << showChunk(oldChunkKey)
              << " Page id from : " << oldPageId << " to : " << oldPageId + skipped << " Epoch: " << oldVersionEpoch;
    } else {
      VLOG(1) << "FId.PSz: " << fileId << "." << pageSize << " Chunk key: " << showChunk(oldChunkKey)
              << " Page id: " << oldPageId << " Epoch: " << oldVersionEpoch;
    }
  }
}

void FileInfo::freePageDeferred(int pageId) {
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

void FileInfo::freePage(int pageId) {
#define RESILIENT_PAGE_HEADER
#ifdef RESILIENT_PAGE_HEADER
  int epoch_freed_page[2] = {DELETE_CONTINGENT, fileMgr->epoch()};
  File_Namespace::write(f, pageId * pageSize + sizeof(int), sizeof(epoch_freed_page), (int8_t*)epoch_freed_page);
  fileMgr->free_page(std::make_pair(this, pageId));
#else
  int zeroVal = 0;
  int8_t* zeroAddr = reinterpret_cast<int8_t*>(&zeroVal);
  File_Namespace::write(f, pageId * pageSize, sizeof(int), zeroAddr);
  std::lock_guard<std::mutex> lock(freePagesMutex_);
  freePages.insert(pageId);
#endif  // RESILIENT_PAGE_HEADER

#ifdef ENABLE_CRASH_CORRUPTION_TEST
  signal(SIGUSR2, sighandler);
  if (goto_crash)
    CHECK(pageId % 8 != 4);
#endif
}

int FileInfo::getFreePage() {
  // returns -1 if there is no free page
  std::lock_guard<std::mutex> lock(freePagesMutex_);
  if (freePages.size() == 0) {
    return -1;
  }
  auto pageIt = freePages.begin();
  int pageNum = *pageIt;
  freePages.erase(pageIt);
  return pageNum;
}

void FileInfo::print(bool pagesummary) {
  std::cout << "File: " << fileId << std::endl;
  std::cout << "Size: " << size() << std::endl;
  std::cout << "Used: " << used() << std::endl;
  std::cout << "Free: " << available() << std::endl;
  if (!pagesummary)
    return;

  // for (size_t i = 0; i < pages.size(); ++i) {
  //    // @todo page summary
  //}
}
}  // File_Namespace
