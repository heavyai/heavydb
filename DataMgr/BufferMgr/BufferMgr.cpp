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
 * @file    BufferMgr.cpp
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#include "BufferMgr.h"
#include "Buffer.h"
#include "Shared/measure.h"

#include <algorithm>
#include <limits>
#include <iomanip>
#include <glog/logging.h>

using namespace std;

static thread_local std::vector<std::string> oom_trace;

void oom_trace_dump() {
  std::ostringstream oss;
  for (const auto& t : oom_trace)
    oss << t << std::endl;
  LOG(INFO) << "OOM trace:" << std::endl << oss.str();
}

void oom_trace_push(const std::string& trace) {
  oom_trace.push_back(trace);
}

void oom_trace_pop() {
  oom_trace.pop_back();
}

namespace Buffer_Namespace {

std::string BufferMgr::keyToString(const ChunkKey& key) {
  std::ostringstream oss;

  oss << " key: ";
  for (auto subKey : key) {
    oss << subKey << ",";
  }
  return oss.str();
}

/// Allocates memSize bytes for the buffer pool and initializes the free memory map.
BufferMgr::BufferMgr(const int deviceId,
                     const size_t maxBufferSize,
                     const size_t maxSlabSize,
                     const size_t pageSize,
                     AbstractBufferMgr* parentMgr)
    : AbstractBufferMgr(deviceId),
      pageSize_(pageSize),
      maxBufferSize_(maxBufferSize),
      numPagesAllocated_(0),
      maxSlabSize_(maxSlabSize),
      allocationsCapped_(false),
      parentMgr_(parentMgr),
      maxBufferId_(0),
      bufferEpoch_(0) {
  CHECK(maxBufferSize_ > 0 && maxSlabSize_ > 0 && pageSize_ > 0 && maxSlabSize_ % pageSize_ == 0);
  maxNumPages_ = maxBufferSize_ / pageSize_;
  maxNumPagesPerSlab_ = maxSlabSize_ / pageSize_;
  currentMaxSlabPageSize_ =
      maxNumPagesPerSlab_;  // currentMaxSlabPageSize_ will drop as allocations fail - this is the high water mark
}

/// Frees the heap-allocated buffer pool memory
BufferMgr::~BufferMgr() {
  clear();
}

void BufferMgr::clear() {
  std::lock_guard<std::mutex> sizedSegsLock(sizedSegsMutex_);
  std::lock_guard<std::mutex> chunkIndexLock(chunkIndexMutex_);
  std::lock_guard<std::mutex> unsizedSegsLock(unsizedSegsMutex_);
  for (auto bufferIt = chunkIndex_.begin(); bufferIt != chunkIndex_.end(); ++bufferIt) {
    delete bufferIt->second->buffer;
  }
  chunkIndex_.clear();
  slabs_.clear();
  slabSegments_.clear();
  unsizedSegs_.clear();
  bufferEpoch_ = 0;
}

/// Throws a runtime_error if the Chunk already exists
AbstractBuffer* BufferMgr::createBuffer(const ChunkKey& chunkKey,
                                        const size_t chunkPageSize,
                                        const size_t initialSize) {
  // LOG(INFO) << printMap();
  size_t actualChunkPageSize = chunkPageSize;
  if (actualChunkPageSize == 0) {
    actualChunkPageSize = pageSize_;
  }

  // ChunkPageSize here is just for recording dirty pages
  {
    std::lock_guard<std::mutex> lock(chunkIndexMutex_);
    CHECK(chunkIndex_.find(chunkKey) == chunkIndex_.end());
    BufferSeg bufferSeg(BufferSeg(-1, 0, USED));
    bufferSeg.chunkKey = chunkKey;
    std::lock_guard<std::mutex> unsizedSegsLock(unsizedSegsMutex_);
    unsizedSegs_.push_back(bufferSeg);  // race condition?
    chunkIndex_[chunkKey] =
        std::prev(unsizedSegs_.end(),
                  1);  // need to do this before allocating Buffer because doing so could change the segment used
  }
  // following should be safe outside the lock b/c first thing Buffer
  // constructor does is pin (and its still in unsized segs at this point
  // so can't be evicted)
  try {
    allocateBuffer(chunkIndex_[chunkKey], actualChunkPageSize, initialSize);
  } catch (const OutOfMemory&) {
    auto bufferIt = chunkIndex_.find(chunkKey);
    CHECK(bufferIt != chunkIndex_.end());
    bufferIt->second->buffer = 0;  // constructor failed for the buffer object so make sure to mark it zero so
                                   // deleteBuffer doesn't try to delete it
    deleteBuffer(chunkKey);
    throw;
  }
  CHECK(initialSize == 0 || chunkIndex_[chunkKey]->buffer->getMemoryPtr());
  // chunkIndex_[chunkKey]->buffer->pin();
  std::lock_guard<std::mutex> lock(chunkIndexMutex_);
  return chunkIndex_[chunkKey]->buffer;
}

BufferList::iterator BufferMgr::evict(BufferList::iterator& evictStart,
                                      const size_t numPagesRequested,
                                      const int slabNum) {
  // We can assume here that buffer for evictStart either doesn't exist
  // (evictStart is first buffer) or was not free, so don't need ot merge
  // it
  auto evictIt = evictStart;
  size_t numPages = 0;
  size_t startPage = evictStart->startPage;
  while (numPages < numPagesRequested) {
    if (evictIt->memStatus == USED) {
      CHECK(evictIt->buffer->getPinCount() < 1);
    }
    numPages += evictIt->numPages;
    if (evictIt->memStatus == USED && evictIt->chunkKey.size() > 0) {
      chunkIndex_.erase(evictIt->chunkKey);
    }
    evictIt = slabSegments_[slabNum].erase(evictIt);  // erase operations returns next iterator - safe if we ever move
                                                      // to a vector (as opposed to erase(evictIt++)
  }
  BufferSeg dataSeg(startPage, numPagesRequested, USED, bufferEpoch_++);  // until we can
  // dataSeg.pinCount++;
  dataSeg.slabNum = slabNum;
  auto dataSegIt = slabSegments_[slabNum].insert(evictIt, dataSeg);  // Will insert before evictIt
  if (numPagesRequested < numPages) {
    size_t excessPages = numPages - numPagesRequested;
    if (evictIt != slabSegments_[slabNum].end() && evictIt->memStatus == FREE) {  // need to merge with current page
      evictIt->startPage = startPage + numPagesRequested;
      evictIt->numPages += excessPages;
    } else {  // need to insert a free seg before evictIt for excessPages
      BufferSeg freeSeg(startPage + numPagesRequested, excessPages, FREE);
      slabSegments_[slabNum].insert(evictIt, freeSeg);
    }
  }
  return dataSegIt;
}

BufferList::iterator BufferMgr::reserveBuffer(BufferList::iterator& segIt,
                                              const size_t numBytes) {  // assumes buffer is already pinned

  size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
  size_t numPagesExtraNeeded = numPagesRequested - segIt->numPages;

  if (numPagesRequested < segIt->numPages) {  // We already have enough pages in existing segment
    return segIt;
  }
  // First check for freeSeg after segIt
  int slabNum = segIt->slabNum;
  if (slabNum >= 0) {  // not dummy page
    BufferList::iterator nextIt = std::next(segIt);
    if (nextIt != slabSegments_[slabNum].end() && nextIt->memStatus == FREE &&
        nextIt->numPages >= numPagesExtraNeeded) {  // Then we can just use the next BufferSeg which happens to be free
      size_t leftoverPages = nextIt->numPages - numPagesExtraNeeded;
      segIt->numPages = numPagesRequested;
      nextIt->numPages = leftoverPages;
      nextIt->startPage = segIt->startPage + segIt->numPages;
      return segIt;
    }
  }
  /* If we're here then we couldn't keep
   * buffer in existing slot - need to find
   * new segment, copy data over, and then
   * delete old
   */

  auto newSegIt = findFreeBuffer(numBytes);

  /* Below should be in copy constructor for BufferSeg?*/
  newSegIt->buffer = segIt->buffer;
  // newSegIt->buffer->segIt_ = newSegIt;
  newSegIt->chunkKey = segIt->chunkKey;
  int8_t* oldMem = newSegIt->buffer->mem_;
  newSegIt->buffer->mem_ = slabs_[newSegIt->slabNum] + newSegIt->startPage * pageSize_;

  // now need to copy over memory
  // only do this if the old segment is valid (i.e. not new w/
  // unallocated buffer
  if (segIt->startPage >= 0 && segIt->buffer->mem_ != 0) {
    newSegIt->buffer->writeData(oldMem, newSegIt->buffer->size(), 0, newSegIt->buffer->getType(), deviceId_);
  }
  // Deincrement pin count to reverse effect above
  removeSegment(segIt);
  {
    std::lock_guard<std::mutex> lock(chunkIndexMutex_);
    chunkIndex_[newSegIt->chunkKey] = newSegIt;
  }

  return newSegIt;
}

BufferList::iterator BufferMgr::findFreeBufferInSlab(const size_t slabNum, const size_t numPagesRequested) {
  for (auto bufferIt = slabSegments_[slabNum].begin(); bufferIt != slabSegments_[slabNum].end(); ++bufferIt) {
    if (bufferIt->memStatus == FREE && bufferIt->numPages >= numPagesRequested) {
      // startPage doesn't change
      size_t excessPages = bufferIt->numPages - numPagesRequested;
      bufferIt->numPages = numPagesRequested;
      bufferIt->memStatus = USED;
      bufferIt->lastTouched = bufferEpoch_++;
      bufferIt->slabNum = slabNum;
      if (excessPages > 0) {
        BufferSeg freeSeg(bufferIt->startPage + numPagesRequested, excessPages, FREE);
        auto tempIt = bufferIt;  // this should make a copy and not be a reference
        // - as we do not want to increment bufferIt
        tempIt++;
        slabSegments_[slabNum].insert(tempIt, freeSeg);
      }
      return bufferIt;
    }
  }
  // If here then we did not find a free buffer of
  // sufficient size in
  // this slab, return the end iterator
  return slabSegments_[slabNum].end();
}

BufferList::iterator BufferMgr::findFreeBuffer(size_t numBytes) {
  size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
  if (numPagesRequested > maxNumPagesPerSlab_) {
    throw SlabTooBig();  //@todo change to requested allocation too big
  }

  size_t numSlabs = slabSegments_.size();

  for (size_t slabNum = 0; slabNum != numSlabs; ++slabNum) {
    auto segIt = findFreeBufferInSlab(slabNum, numPagesRequested);
    if (segIt != slabSegments_[slabNum].end()) {
      return segIt;
    }
  }

  // If we're here then we didn't find a free segment of sufficient size
  // First we see if we can add another slab
  while (!allocationsCapped_ && numPagesAllocated_ < maxNumPages_) {
    try {
      size_t pagesLeft = maxNumPages_ - numPagesAllocated_;
      if (pagesLeft < currentMaxSlabPageSize_)
        currentMaxSlabPageSize_ = pagesLeft;
      if (numPagesRequested <= currentMaxSlabPageSize_) {  // don't try to allocate if the new slab won't be big enough
        auto alloc_ms = measure<>::execution([&]() { addSlab(currentMaxSlabPageSize_ * pageSize_); });
        LOG(INFO) << "ALLOCATION slab of " << currentMaxSlabPageSize_ << " pages ("
                  << currentMaxSlabPageSize_ * pageSize_ << "B) created in " << alloc_ms << " ms " << getStringMgrType()
                  << ":" << deviceId_;
      } else
        break;
      // if here then addSlab succeeded
      numPagesAllocated_ += currentMaxSlabPageSize_;
      return findFreeBufferInSlab(
          numSlabs,
          numPagesRequested);  // has to succeed since we made sure to request a slab big enough to accomodate request
    } catch (std::runtime_error& error) {  // failed to allocate slab
      LOG(INFO) << "ALLOCATION Attempted slab of " << currentMaxSlabPageSize_ << " pages ("
                << currentMaxSlabPageSize_ * pageSize_ << "B) failed " << getStringMgrType() << ":" << deviceId_;
      // check if there is any point halving currentMaxSlabSize and trying again
      // if the request wont fit in half available then let try once at full size
      // if we have already tries at full size and failed then break as
      // there could still be room enough for other later request but
      // not for his current one
      if (numPagesRequested > currentMaxSlabPageSize_ / 2 && currentMaxSlabPageSize_ != numPagesRequested) {
        currentMaxSlabPageSize_ = numPagesRequested;
      } else {
        currentMaxSlabPageSize_ /= 2;
        if (currentMaxSlabPageSize_ < (maxNumPagesPerSlab_ / 8)) {  // should be a constant
          allocationsCapped_ = true;
          // dump out the slabs and their sizes
          LOG(INFO) << "ALLOCATION Capped " << currentMaxSlabPageSize_
                    << " Minimum size = " << (maxNumPagesPerSlab_ / 8) << " " << getStringMgrType() << ":" << deviceId_;
        }
      }
    }
  }

  if (numPagesAllocated_ == 0 && allocationsCapped_) {
    throw FailedToCreateFirstSlab();
  }

  // If here then we can't add a slab - so we need to evict

  size_t minScore = std::numeric_limits<size_t>::max();
  // We're going for lowest score here, like golf
  // This is because score is the sum of the lastTouched score for all
  // pages evicted. Evicting less pages and older pages will lower the
  // score
  BufferList::iterator bestEvictionStart = slabSegments_[0].end();
  int bestEvictionStartSlab = -1;
  int slabNum = 0;

  for (auto slabIt = slabSegments_.begin(); slabIt != slabSegments_.end(); ++slabIt, ++slabNum) {
    for (auto bufferIt = slabIt->begin(); bufferIt != slabIt->end(); ++bufferIt) {
      /* Note there are some shortcuts we could take here - like we
       * should never consider a USED buffer coming after a free buffer
       * as we would have used the FREE buffer, but we won't worry about
       * this for now
       */

      // We can't evict pinned  buffers - only normal used
      // buffers

      // if (bufferIt->memStatus == FREE || bufferIt->buffer->getPinCount() == 0) {
      size_t pageCount = 0;
      size_t score = 0;
      bool solutionFound = false;
      auto evictIt = bufferIt;
      for (; evictIt != slabSegments_[slabNum].end(); ++evictIt) {
        // pinCount should never go up - only down because we have
        // global lock on buffer pool and pin count only increments
        // on getChunk
        if (evictIt->memStatus == USED && evictIt->buffer->getPinCount() > 0) {
          break;
        }
        pageCount += evictIt->numPages;
        if (evictIt->memStatus == USED) {
          // MAT changed from
          // score += evictIt->lastTouched;
          // Issue was thrashing when going from 8M fragment size chunks back to 64M
          // basically the large chunks were being evicted prior to small as many small chunk
          // score was larger than one large chunk so it always would evict a large chunk
          // so under memory pressure a query would evict its own current chunks and cause reloads
          // rather than evict several smaller unused older chunks.
          score = std::max(score, static_cast<size_t>(evictIt->lastTouched));
        }
        if (pageCount >= numPagesRequested) {
          solutionFound = true;
          break;
        }
      }
      if (solutionFound && score < minScore) {
        minScore = score;
        bestEvictionStart = bufferIt;
        bestEvictionStartSlab = slabNum;
      } else if (evictIt == slabSegments_[slabNum].end()) {
        // this means that every segment after this will fail as
        // well, so our search has proven futile
        // throw std::runtime_error ("Couldn't evict chunks to get free space");
        break;
        // in reality we should try to rearrange the buffer to get
        // more contiguous free space
      }
      // other possibility is ending at PINNED - do nothing in this
      // case
      //}
    }
  }
  if (bestEvictionStart == slabSegments_[0].end()) {
    LOG(ERROR) << "ALLOCATION failed to find " << numBytes << "B throwing out of memory " << getStringMgrType() << ":"
               << deviceId_;
    printSlabs();
    throw OutOfMemory();
  }
  LOG(INFO) << "ALLOCATION failed to find " << numBytes << "B free. Forcing Eviction."
            << " Eviction start " << bestEvictionStart->startPage << " Number pages requested " << numPagesRequested
            << " Best Eviction Start Slab " << bestEvictionStartSlab << " " << getStringMgrType() << ":" << deviceId_;
  bestEvictionStart = evict(bestEvictionStart, numPagesRequested, bestEvictionStartSlab);
  return bestEvictionStart;
}

std::string BufferMgr::printSlab(size_t slabNum) {
  std::ostringstream tss;
  // size_t lastEnd = 0;
  tss << "Slab St.Page   Pages  Touch" << std::endl;
  for (auto segIt = slabSegments_[slabNum].begin(); segIt != slabSegments_[slabNum].end(); ++segIt) {
    tss << setfill(' ') << setw(4) << slabNum;
    // tss << " BSN: " << setfill(' ') << setw(2) << segIt->slabNum;
    tss << setfill(' ') << setw(8) << segIt->startPage;
    tss << setfill(' ') << setw(8) << segIt->numPages;
    // tss << " GAP: " << setfill(' ') << setw(7) << segIt->startPage - lastEnd;
    // lastEnd = segIt->startPage + segIt->numPages;
    tss << setfill(' ') << setw(7) << segIt->lastTouched;
    // tss << " PC: " << setfill(' ') << setw(2) << segIt->buffer->getPinCount();
    if (segIt->memStatus == FREE)
      tss << " FREE"
          << " ";
    else {
      tss << " PC: " << setfill(' ') << setw(2) << segIt->buffer->getPinCount();
      tss << " USED - Chunk: ";

      for (auto vecIt = segIt->chunkKey.begin(); vecIt != segIt->chunkKey.end(); ++vecIt) {
        tss << *vecIt << ",";
      }
    }
    tss << std::endl;
  }
  return tss.str();
}

std::string BufferMgr::printSlabs() {
  std::ostringstream tss;
  tss << std::endl
      << "Slabs Contents: "
      << " " << getStringMgrType() << ":" << deviceId_ << std::endl;
  size_t numSlabs = slabSegments_.size();
  for (size_t slabNum = 0; slabNum != numSlabs; ++slabNum) {
    tss << printSlab(slabNum);
  }
  tss << "--------------------" << std::endl;
  return tss.str();
}

void BufferMgr::clearSlabs() {
  size_t numSlabs = slabSegments_.size();
  for (size_t slabNum = 0; slabNum != numSlabs; ++slabNum) {
    for (auto segIt = slabSegments_[slabNum].begin(); segIt != slabSegments_[slabNum].end(); ++segIt) {
      if (segIt->memStatus == FREE) {
        // no need to free
      } else if (segIt->buffer->getPinCount() < 1) {
        deleteBuffer(segIt->chunkKey, true);
      }
    }
  }
}

// return the maximum size this buffer can be in bytes
size_t BufferMgr::getMaxSize() {
  return pageSize_ * maxNumPages_;
}

// return how large the buffer are currently allocated
size_t BufferMgr::getAllocated() {
  return numPagesAllocated_ * pageSize_;
}

//
bool BufferMgr::isAllocationCapped() {
  return allocationsCapped_;
}

size_t BufferMgr::getPageSize() {
  return pageSize_;
}

// return the size of the chunks in use in bytes
size_t BufferMgr::getInUseSize() {
  size_t inUse = 0;
  size_t numSlabs = slabSegments_.size();
  for (size_t slabNum = 0; slabNum != numSlabs; ++slabNum) {
    for (auto segIt = slabSegments_[slabNum].begin(); segIt != slabSegments_[slabNum].end(); ++segIt) {
      if (segIt->memStatus != FREE) {
        inUse += segIt->numPages * pageSize_;
      }
    }
  }
  return inUse;
}

std::string BufferMgr::printSeg(BufferList::iterator& segIt) {
  std::ostringstream tss;
  tss << "SN: " << setfill(' ') << setw(2) << segIt->slabNum;
  tss << " SP: " << setfill(' ') << setw(7) << segIt->startPage;
  tss << " NP: " << setfill(' ') << setw(7) << segIt->numPages;
  tss << " LT: " << setfill(' ') << setw(7) << segIt->lastTouched;
  tss << " PC: " << setfill(' ') << setw(2) << segIt->buffer->getPinCount();
  if (segIt->memStatus == FREE)
    tss << " FREE"
        << " ";
  else {
    tss << " USED - Chunk: ";
    for (auto vecIt = segIt->chunkKey.begin(); vecIt != segIt->chunkKey.end(); ++vecIt) {
      tss << *vecIt << ",";
    }
    tss << std::endl;
  }
  return tss.str();
}

std::string BufferMgr::printMap() {
  std::ostringstream tss;
  int segNum = 1;
  tss << std::endl
      << "Map Contents: "
      << " " << getStringMgrType() << ":" << deviceId_ << std::endl;
  for (auto segIt = chunkIndex_.begin(); segIt != chunkIndex_.end(); ++segIt, ++segNum) {
    //    tss << "Map Entry " << segNum << ": ";
    //    for (auto vecIt = segIt->first.begin(); vecIt != segIt->first.end(); ++vecIt) {
    //      tss << *vecIt << ",";
    //    }
    //    tss << " " << std::endl;
    tss << printSeg(segIt->second);
  }
  tss << "--------------------" << std::endl;
  return tss.str();
}

void BufferMgr::printSegs() {
  int segNum = 1;
  int slabNum = 1;
  LOG(INFO) << std::endl << " " << getStringMgrType() << ":" << deviceId_;
  for (auto slabIt = slabSegments_.begin(); slabIt != slabSegments_.end(); ++slabIt, ++slabNum) {
    LOG(INFO) << "Slab Num: " << slabNum << " " << getStringMgrType() << ":" << deviceId_;
    for (auto segIt = slabIt->begin(); segIt != slabIt->end(); ++segIt, ++segNum) {
      LOG(INFO) << "Segment: " << segNum << " " << getStringMgrType() << ":" << deviceId_;
      printSeg(segIt);
      LOG(INFO) << " " << getStringMgrType() << ":" << deviceId_;
    }
    LOG(INFO) << "--------------------"
              << " " << getStringMgrType() << ":" << deviceId_;
  }
}

bool BufferMgr::isBufferOnDevice(const ChunkKey& key) {
  std::lock_guard<std::mutex> chunkIndexLock(chunkIndexMutex_);
  if (chunkIndex_.find(key) == chunkIndex_.end()) {
    return false;
  } else {
    return true;
  }
}

/// This method throws a runtime_error when deleting a Chunk that does not exist.
void BufferMgr::deleteBuffer(const ChunkKey& key, const bool purge) {
  std::unique_lock<std::mutex> chunkIndexLock(chunkIndexMutex_);
  // Note: purge is currently unused

  // lookup the buffer for the Chunk in chunkIndex_
  auto bufferIt = chunkIndex_.find(key);
  // Buffer *buffer = bufferIt->second->buffer;
  CHECK(bufferIt != chunkIndex_.end());
  auto segIt = bufferIt->second;
  chunkIndex_.erase(bufferIt);
  chunkIndexLock.unlock();
  std::lock_guard<std::mutex> sizedSegsLock(sizedSegsMutex_);
  if (segIt->buffer) {
    delete segIt->buffer;  // Delete Buffer for segment
    segIt->buffer = 0;
  }
  removeSegment(segIt);
}

void BufferMgr::deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool purge) {
  // Note: purge is unused
  // lookup the buffer for the Chunk in chunkIndex_
  std::lock_guard<std::mutex> sizedSegsLock(sizedSegsMutex_);  // Take this lock early to prevent deadlock with
                                                               // reserveBuffer which needs segsMutex_ and then
                                                               // chunkIndexMutex_
  std::lock_guard<std::mutex> chunkIndexLock(chunkIndexMutex_);
  auto startChunkIt = chunkIndex_.lower_bound(keyPrefix);
  if (startChunkIt == chunkIndex_.end()) {
    return;
  }

  auto bufferIt = startChunkIt;
  while (bufferIt != chunkIndex_.end() &&
         std::search(
             bufferIt->first.begin(), bufferIt->first.begin() + keyPrefix.size(), keyPrefix.begin(), keyPrefix.end()) !=
             bufferIt->first.begin() + keyPrefix.size()) {
    auto segIt = bufferIt->second;
    if (segIt->buffer) {
      delete segIt->buffer;  // Delete Buffer for segment
      segIt->buffer = 0;
    }
    removeSegment(segIt);
    chunkIndex_.erase(bufferIt++);
  }
}

void BufferMgr::removeSegment(BufferList::iterator& segIt) {
  // Note: does not delete buffer as this may be moved somewhere else
  int slabNum = segIt->slabNum;
  // cout << "Slab num: " << slabNum << endl;
  if (slabNum < 0) {
    std::lock_guard<std::mutex> unsizedSegsLock(unsizedSegsMutex_);
    unsizedSegs_.erase(segIt);
  } else {
    if (segIt != slabSegments_[slabNum].begin()) {
      auto prevIt = std::prev(segIt);
      // LOG(INFO) << "PrevIt: " << " " << getStringMgrType() << ":" << deviceId_;
      // printSeg(prevIt);
      if (prevIt->memStatus == FREE) {
        segIt->startPage = prevIt->startPage;
        segIt->numPages += prevIt->numPages;
        slabSegments_[slabNum].erase(prevIt);
      }
    }
    auto nextIt = std::next(segIt);
    if (nextIt != slabSegments_[slabNum].end()) {
      if (nextIt->memStatus == FREE) {
        segIt->numPages += nextIt->numPages;
        slabSegments_[slabNum].erase(nextIt);
      }
    }
    segIt->memStatus = FREE;
    // segIt->pinCount = 0;
    segIt->buffer = 0;
  }
}

void BufferMgr::checkpoint() {
  std::lock_guard<std::mutex> lock(globalMutex_);  // granular lock

  for (auto bufferIt = chunkIndex_.begin(); bufferIt != chunkIndex_.end(); ++bufferIt) {
    if (bufferIt->second->chunkKey[0] != -1 &&
        bufferIt->second->buffer->isDirty_) {  // checks that buffer is actual chunk (not just buffer) and is dirty

      parentMgr_->putBuffer(bufferIt->second->chunkKey, bufferIt->second->buffer);
      bufferIt->second->buffer->clearDirtyBits();
    }
  }
}

void BufferMgr::checkpoint(const int db_id, const int tb_id) {
  std::lock_guard<std::mutex> lock(globalMutex_);  // granular lock
  ChunkKey keyPrefix;
  keyPrefix.push_back(db_id);
  keyPrefix.push_back(tb_id);
  auto startChunkIt = chunkIndex_.lower_bound(keyPrefix);
  if (startChunkIt == chunkIndex_.end()) {
    return;
  }

  auto bufferIt = startChunkIt;
  while (bufferIt != chunkIndex_.end() &&
         std::search(
             bufferIt->first.begin(), bufferIt->first.begin() + keyPrefix.size(), keyPrefix.begin(), keyPrefix.end()) !=
             bufferIt->first.begin() + keyPrefix.size()) {
    if (bufferIt->second->chunkKey[0] != -1 &&
        bufferIt->second->buffer->isDirty_) {  // checks that buffer is actual chunk (not just buffer) and is dirty

      parentMgr_->putBuffer(bufferIt->second->chunkKey, bufferIt->second->buffer);
      bufferIt->second->buffer->clearDirtyBits();
    }
    bufferIt++;
  }
}

/// Returns a pointer to the Buffer holding the chunk, if it exists; otherwise,
/// throws a runtime_error.
AbstractBuffer* BufferMgr::getBuffer(const ChunkKey& key, const size_t numBytes) {
  std::lock_guard<std::mutex> lock(globalMutex_);  // granular lock

  std::unique_lock<std::mutex> sizedSegsLock(sizedSegsMutex_);
  std::unique_lock<std::mutex> chunkIndexLock(chunkIndexMutex_);
  auto bufferIt = chunkIndex_.find(key);
  bool foundBuffer = bufferIt != chunkIndex_.end();
  chunkIndexLock.unlock();
  if (foundBuffer) {
    CHECK(bufferIt->second->buffer);
    bufferIt->second->buffer->pin();
    sizedSegsLock.unlock();
    bufferIt->second->lastTouched = bufferEpoch_++;     // race
    if (bufferIt->second->buffer->size() < numBytes) {  // need to fetch part of buffer we don't have - up to numBytes
      parentMgr_->fetchBuffer(key, bufferIt->second->buffer, numBytes);
    }
    return bufferIt->second->buffer;
  } else {  // If wasn't in pool then we need to fetch it
    sizedSegsLock.unlock();
    AbstractBuffer* buffer = createBuffer(key, pageSize_, numBytes);  // createChunk pins for us
    try {
      parentMgr_->fetchBuffer(key, buffer, numBytes);  // this should put buffer in a BufferSegment
    } catch (std::runtime_error& error) {
      LOG(FATAL) << "Get chunk - Could not find chunk " << keyToString(key)
                 << " in buffer pool or parent buffer pools. Error was " << error.what();
    }
    return buffer;
  }
}

void BufferMgr::fetchBuffer(const ChunkKey& key, AbstractBuffer* destBuffer, const size_t numBytes) {
  std::unique_lock<std::mutex> lock(globalMutex_);  // granular lock
  std::unique_lock<std::mutex> sizedSegsLock(sizedSegsMutex_);
  std::unique_lock<std::mutex> chunkIndexLock(chunkIndexMutex_);

  auto bufferIt = chunkIndex_.find(key);
  bool foundBuffer = bufferIt != chunkIndex_.end();
  chunkIndexLock.unlock();
  AbstractBuffer* buffer;
  if (!foundBuffer) {
    sizedSegsLock.unlock();
    CHECK(parentMgr_ != 0);
    buffer = createBuffer(key, pageSize_, numBytes);  // will pin buffer
    try {
      parentMgr_->fetchBuffer(key, buffer, numBytes);
    } catch (std::runtime_error& error) {
      LOG(FATAL) << "Could not fetch parent buffer " << keyToString(key);
    }
  } else {
    buffer = bufferIt->second->buffer;
    buffer->pin();
    if (numBytes > buffer->size()) {
      try {
        parentMgr_->fetchBuffer(key, buffer, numBytes);
      } catch (std::runtime_error& error) {
        LOG(FATAL) << "Could not fetch parent buffer " << keyToString(key);
      }
    }
    sizedSegsLock.unlock();
  }
  size_t chunkSize = numBytes == 0 ? buffer->size() : numBytes;
  lock.unlock();
  destBuffer->reserve(chunkSize);
  if (buffer->isUpdated()) {
    buffer->read(destBuffer->getMemoryPtr(), chunkSize, 0, destBuffer->getType(), destBuffer->getDeviceId());
  } else {
    buffer->read(destBuffer->getMemoryPtr() + destBuffer->size(),
                 chunkSize - destBuffer->size(),
                 destBuffer->size(),
                 destBuffer->getType(),
                 destBuffer->getDeviceId());
  }
  destBuffer->setSize(chunkSize);
  destBuffer->syncEncoder(buffer);
  buffer->unPin();
}

AbstractBuffer* BufferMgr::putBuffer(const ChunkKey& key, AbstractBuffer* srcBuffer, const size_t numBytes) {
  std::unique_lock<std::mutex> chunkIndexLock(chunkIndexMutex_);
  auto bufferIt = chunkIndex_.find(key);
  bool foundBuffer = bufferIt != chunkIndex_.end();
  chunkIndexLock.unlock();
  AbstractBuffer* buffer;
  if (!foundBuffer) {
    buffer = createBuffer(key, pageSize_);
  } else {
    buffer = bufferIt->second->buffer;
  }
  size_t oldBufferSize = buffer->size();
  size_t newBufferSize = numBytes == 0 ? srcBuffer->size() : numBytes;
  CHECK(!buffer->isDirty());

  if (srcBuffer->isUpdated()) {
    //@todo use dirty flags to only flush pages of chunk that need to
    // be flushed
    buffer->write((int8_t*)srcBuffer->getMemoryPtr(), newBufferSize, 0, srcBuffer->getType(), srcBuffer->getDeviceId());
  } else if (srcBuffer->isAppended()) {
    CHECK(oldBufferSize < newBufferSize);
    buffer->append((int8_t*)srcBuffer->getMemoryPtr() + oldBufferSize,
                   newBufferSize - oldBufferSize,
                   srcBuffer->getType(),
                   srcBuffer->getDeviceId());
  }
  srcBuffer->clearDirtyBits();
  buffer->syncEncoder(srcBuffer);
  return buffer;
}

int BufferMgr::getBufferId() {
  std::lock_guard<std::mutex> lock(bufferIdMutex_);
  return maxBufferId_++;
}

/// client is responsible for deleting memory allocated for b->mem_
AbstractBuffer* BufferMgr::alloc(const size_t numBytes) {
  std::lock_guard<std::mutex> lock(globalMutex_);
  ChunkKey chunkKey = {-1, getBufferId()};
  return createBuffer(chunkKey, pageSize_, numBytes);
}

void BufferMgr::free(AbstractBuffer* buffer) {
  std::lock_guard<std::mutex> lock(globalMutex_);  // hack for now
  Buffer* castedBuffer = dynamic_cast<Buffer*>(buffer);
  if (castedBuffer == 0) {
    LOG(FATAL) << "Wrong buffer type - expects base class pointer to Buffer type.";
  }
  deleteBuffer(castedBuffer->segIt_->chunkKey);
}

size_t BufferMgr::getNumChunks() {
  std::lock_guard<std::mutex> chunkIndexLock(chunkIndexMutex_);
  return chunkIndex_.size();
}

size_t BufferMgr::size() {
  return numPagesAllocated_;
}

size_t BufferMgr::getMaxBufferSize() {
  return maxBufferSize_;
}

size_t BufferMgr::getMaxSlabSize() {
  return maxSlabSize_;
}

void BufferMgr::getChunkMetadataVec(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec) {
  LOG(FATAL) << "getChunkMetadataVec not supported for BufferMgr.";
}

void BufferMgr::getChunkMetadataVecForKeyPrefix(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
                                                const ChunkKey& keyPrefix) {
  LOG(FATAL) << "getChunkMetadataVecForPrefix not supported for BufferMgr.";
}

const std::vector<BufferList>& BufferMgr::getSlabSegments() {
  return slabSegments_;
}
}
