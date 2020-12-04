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
#include "DataMgr/BufferMgr/BufferMgr.h"

#include <algorithm>
#include <iomanip>
#include <limits>

#include "DataMgr/BufferMgr/Buffer.h"
#include "Logger/Logger.h"
#include "Shared/measure.h"

using namespace std;

namespace Buffer_Namespace {

std::string BufferMgr::keyToString(const ChunkKey& key) {
  std::ostringstream oss;

  oss << " key: ";
  for (auto sub_key : key) {
    oss << sub_key << ",";
  }
  return oss.str();
}

/// Allocates memSize bytes for the buffer pool and initializes the free memory map.
BufferMgr::BufferMgr(const int device_id,
                     const size_t max_buffer_pool_size,
                     const size_t min_slab_size,
                     const size_t max_slab_size,
                     const size_t page_size,
                     AbstractBufferMgr* parent_mgr)
    : AbstractBufferMgr(device_id)
    , max_buffer_pool_size_(max_buffer_pool_size)
    , min_slab_size_(min_slab_size)
    , max_slab_size_(max_slab_size)
    , page_size_(page_size)
    , num_pages_allocated_(0)
    , allocations_capped_(false)
    , parent_mgr_(parent_mgr)
    , max_buffer_id_(0)
    , buffer_epoch_(0) {
  CHECK(max_buffer_pool_size_ > 0);
  CHECK(page_size_ > 0);
  // TODO change checks on run-time configurable slab size variables to exceptions
  CHECK(min_slab_size_ > 0);
  CHECK(max_slab_size_ > 0);
  CHECK(min_slab_size_ <= max_slab_size_);
  CHECK(min_slab_size_ % page_size_ == 0);
  CHECK(max_slab_size_ % page_size_ == 0);

  max_buffer_pool_num_pages_ = max_buffer_pool_size_ / page_size_;
  max_num_pages_per_slab_ = max_slab_size_ / page_size_;
  min_num_pages_per_slab_ = min_slab_size_ / page_size_;
  current_max_slab_page_size_ =
      max_num_pages_per_slab_;  // current_max_slab_page_size_ will drop as allocations
                                // fail - this is the high water mark
}

/// Frees the heap-allocated buffer pool memory
BufferMgr::~BufferMgr() {
  clear();
}

void BufferMgr::reinit() {
  num_pages_allocated_ = 0;
  current_max_slab_page_size_ =
      max_num_pages_per_slab_;  // current_max_slab_page_size_ will drop as allocations
                                // fail - this is the high water mark
  allocations_capped_ = false;
}

void BufferMgr::clear() {
  std::lock_guard<std::mutex> sized_segs_lock(sized_segs_mutex_);
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_);
  std::lock_guard<std::mutex> unsized_segs_lock(unsized_segs_mutex_);
  for (auto& buf : chunk_index_) {
    delete buf.second->buffer;
  }

  chunk_index_.clear();
  slabs_.clear();
  slab_segments_.clear();
  unsized_segs_.clear();
  buffer_epoch_ = 0;
}

/// Throws a runtime_error if the Chunk already exists
AbstractBuffer* BufferMgr::createBuffer(const ChunkKey& chunk_key,
                                        const size_t chunk_page_size,
                                        const size_t initial_size) {
  // LOG(INFO) << printMap();
  size_t actual_chunk_page_size = chunk_page_size;
  if (actual_chunk_page_size == 0) {
    actual_chunk_page_size = page_size_;
  }

  // chunk_page_size is just for recording dirty pages
  {
    std::lock_guard<std::mutex> lock(chunk_index_mutex_);
    CHECK(chunk_index_.find(chunk_key) == chunk_index_.end());
    BufferSeg buffer_seg(BufferSeg(-1, 0, USED));
    buffer_seg.chunk_key = chunk_key;
    std::lock_guard<std::mutex> unsizedSegsLock(unsized_segs_mutex_);
    unsized_segs_.push_back(buffer_seg);  // race condition?
    chunk_index_[chunk_key] =
        std::prev(unsized_segs_.end(),
                  1);  // need to do this before allocating Buffer because doing so could
                       // change the segment used
  }
  // following should be safe outside the lock b/c first thing Buffer
  // constructor does is pin (and its still in unsized segs at this point
  // so can't be evicted)
  try {
    allocateBuffer(chunk_index_[chunk_key], actual_chunk_page_size, initial_size);
  } catch (const OutOfMemory&) {
    auto buffer_it = chunk_index_.find(chunk_key);
    CHECK(buffer_it != chunk_index_.end());
    buffer_it->second->buffer =
        nullptr;  // constructor failed for the buffer object so make sure to mark it null
                  // so deleteBuffer doesn't try to delete it
    deleteBuffer(chunk_key);
    throw;
  }
  CHECK(initial_size == 0 || chunk_index_[chunk_key]->buffer->getMemoryPtr());
  // chunk_index_[chunk_key]->buffer->pin();
  std::lock_guard<std::mutex> lock(chunk_index_mutex_);
  return chunk_index_[chunk_key]->buffer;
}

BufferList::iterator BufferMgr::evict(BufferList::iterator& evict_start,
                                      const size_t num_pages_requested,
                                      const int slab_num) {
  // We can assume here that buffer for evictStart either doesn't exist
  // (evictStart is first buffer) or was not free, so don't need ot merge
  // it
  auto evict_it = evict_start;
  size_t num_pages = 0;
  size_t start_page = evict_start->start_page;
  while (num_pages < num_pages_requested) {
    if (evict_it->mem_status == USED) {
      CHECK(evict_it->buffer->getPinCount() < 1);
    }
    num_pages += evict_it->num_pages;
    if (evict_it->mem_status == USED && evict_it->chunk_key.size() > 0) {
      chunk_index_.erase(evict_it->chunk_key);
    }
    evict_it = slab_segments_[slab_num].erase(
        evict_it);  // erase operations returns next iterator - safe if we ever move
                    // to a vector (as opposed to erase(evict_it++)
  }
  BufferSeg data_seg(
      start_page, num_pages_requested, USED, buffer_epoch_++);  // until we can
  // data_seg.pinCount++;
  data_seg.slab_num = slab_num;
  auto data_seg_it =
      slab_segments_[slab_num].insert(evict_it, data_seg);  // Will insert before evict_it
  if (num_pages_requested < num_pages) {
    size_t excess_pages = num_pages - num_pages_requested;
    if (evict_it != slab_segments_[slab_num].end() &&
        evict_it->mem_status == FREE) {  // need to merge with current page
      evict_it->start_page = start_page + num_pages_requested;
      evict_it->num_pages += excess_pages;
    } else {  // need to insert a free seg before evict_it for excess_pages
      BufferSeg free_seg(start_page + num_pages_requested, excess_pages, FREE);
      slab_segments_[slab_num].insert(evict_it, free_seg);
    }
  }
  return data_seg_it;
}

BufferList::iterator BufferMgr::reserveBuffer(
    BufferList::iterator& seg_it,
    const size_t num_bytes) {  // assumes buffer is already pinned

  size_t num_pages_requested = (num_bytes + page_size_ - 1) / page_size_;
  size_t num_pages_extra_needed = num_pages_requested - seg_it->num_pages;

  if (num_pages_requested < seg_it->num_pages) {
    // We already have enough pages in existing segment
    return seg_it;
  }
  // First check for free segment after seg_it
  int slab_num = seg_it->slab_num;
  if (slab_num >= 0) {  // not dummy page
    BufferList::iterator next_it = std::next(seg_it);
    if (next_it != slab_segments_[slab_num].end() && next_it->mem_status == FREE &&
        next_it->num_pages >= num_pages_extra_needed) {
      // Then we can just use the next BufferSeg which happens to be free
      size_t leftover_pages = next_it->num_pages - num_pages_extra_needed;
      seg_it->num_pages = num_pages_requested;
      next_it->num_pages = leftover_pages;
      next_it->start_page = seg_it->start_page + seg_it->num_pages;
      return seg_it;
    }
  }
  // If we're here then we couldn't keep buffer in existing slot
  // need to find new segment, copy data over, and then delete old
  auto new_seg_it = findFreeBuffer(num_bytes);

  // Below should be in copy constructor for BufferSeg?
  new_seg_it->buffer = seg_it->buffer;
  new_seg_it->chunk_key = seg_it->chunk_key;
  int8_t* old_mem = new_seg_it->buffer->mem_;
  new_seg_it->buffer->mem_ =
      slabs_[new_seg_it->slab_num] + new_seg_it->start_page * page_size_;

  // now need to copy over memory
  // only do this if the old segment is valid (i.e. not new w/ unallocated buffer
  if (seg_it->start_page >= 0 && seg_it->buffer->mem_ != 0) {
    new_seg_it->buffer->writeData(old_mem,
                                  new_seg_it->buffer->size(),
                                  0,
                                  new_seg_it->buffer->getType(),
                                  device_id_);
  }
  // Decrement pin count to reverse effect above
  removeSegment(seg_it);
  {
    std::lock_guard<std::mutex> lock(chunk_index_mutex_);
    chunk_index_[new_seg_it->chunk_key] = new_seg_it;
  }

  return new_seg_it;
}

BufferList::iterator BufferMgr::findFreeBufferInSlab(const size_t slab_num,
                                                     const size_t num_pages_requested) {
  for (auto buffer_it = slab_segments_[slab_num].begin();
       buffer_it != slab_segments_[slab_num].end();
       ++buffer_it) {
    if (buffer_it->mem_status == FREE && buffer_it->num_pages >= num_pages_requested) {
      // startPage doesn't change
      size_t excess_pages = buffer_it->num_pages - num_pages_requested;
      buffer_it->num_pages = num_pages_requested;
      buffer_it->mem_status = USED;
      buffer_it->last_touched = buffer_epoch_++;
      buffer_it->slab_num = slab_num;
      if (excess_pages > 0) {
        BufferSeg free_seg(
            buffer_it->start_page + num_pages_requested, excess_pages, FREE);
        auto temp_it = buffer_it;  // this should make a copy and not be a reference
        // - as we do not want to increment buffer_it
        temp_it++;
        slab_segments_[slab_num].insert(temp_it, free_seg);
      }
      return buffer_it;
    }
  }
  // If here then we did not find a free buffer of sufficient size in this slab,
  // return the end iterator
  return slab_segments_[slab_num].end();
}

BufferList::iterator BufferMgr::findFreeBuffer(size_t num_bytes) {
  size_t num_pages_requested = (num_bytes + page_size_ - 1) / page_size_;
  if (num_pages_requested > max_num_pages_per_slab_) {
    throw TooBigForSlab(num_bytes);
  }

  size_t num_slabs = slab_segments_.size();

  for (size_t slab_num = 0; slab_num != num_slabs; ++slab_num) {
    auto seg_it = findFreeBufferInSlab(slab_num, num_pages_requested);
    if (seg_it != slab_segments_[slab_num].end()) {
      return seg_it;
    }
  }

  // If we're here then we didn't find a free segment of sufficient size
  // First we see if we can add another slab
  while (!allocations_capped_ && num_pages_allocated_ < max_buffer_pool_num_pages_) {
    try {
      size_t pagesLeft = max_buffer_pool_num_pages_ - num_pages_allocated_;
      if (pagesLeft < current_max_slab_page_size_) {
        current_max_slab_page_size_ = pagesLeft;
      }
      if (num_pages_requested <=
          current_max_slab_page_size_) {  // don't try to allocate if the
                                          // new slab won't be big enough
        auto alloc_ms = measure<>::execution(
            [&]() { addSlab(current_max_slab_page_size_ * page_size_); });
        LOG(INFO) << "ALLOCATION slab of " << current_max_slab_page_size_ << " pages ("
                  << current_max_slab_page_size_ * page_size_ << "B) created in "
                  << alloc_ms << " ms " << getStringMgrType() << ":" << device_id_;
      } else {
        break;
      }
      // if here then addSlab succeeded
      num_pages_allocated_ += current_max_slab_page_size_;
      return findFreeBufferInSlab(
          num_slabs,
          num_pages_requested);  // has to succeed since we made sure to request a slab
                                 // big enough to accomodate request
    } catch (std::runtime_error& error) {  // failed to allocate slab
      LOG(INFO) << "ALLOCATION Attempted slab of " << current_max_slab_page_size_
                << " pages (" << current_max_slab_page_size_ * page_size_ << "B) failed "
                << getStringMgrType() << ":" << device_id_;
      // check if there is any point halving currentMaxSlabSize and trying again
      // if the request wont fit in half available then let try once at full size
      // if we have already tries at full size and failed then break as
      // there could still be room enough for other later request but
      // not for his current one
      if (num_pages_requested > current_max_slab_page_size_ / 2 &&
          current_max_slab_page_size_ != num_pages_requested) {
        current_max_slab_page_size_ = num_pages_requested;
      } else {
        current_max_slab_page_size_ /= 2;
        if (current_max_slab_page_size_ <
            (min_num_pages_per_slab_)) {  // should be a constant
          allocations_capped_ = true;
          // dump out the slabs and their sizes
          LOG(INFO) << "ALLOCATION Capped " << current_max_slab_page_size_
                    << " Minimum size = " << (min_num_pages_per_slab_) << " "
                    << getStringMgrType() << ":" << device_id_;
        }
      }
    }
  }

  if (num_pages_allocated_ == 0 && allocations_capped_) {
    throw FailedToCreateFirstSlab(num_bytes);
  }

  // If here then we can't add a slab - so we need to evict

  size_t min_score = std::numeric_limits<size_t>::max();
  // We're going for lowest score here, like golf
  // This is because score is the sum of the lastTouched score for all pages evicted.
  // Evicting fewer pages and older pages will lower the score
  BufferList::iterator best_eviction_start = slab_segments_[0].end();
  int best_eviction_start_slab = -1;
  int slab_num = 0;

  for (auto slab_it = slab_segments_.begin(); slab_it != slab_segments_.end();
       ++slab_it, ++slab_num) {
    for (auto buffer_it = slab_it->begin(); buffer_it != slab_it->end(); ++buffer_it) {
      // Note there are some shortcuts we could take here - like we should never consider
      // a USED buffer coming after a free buffer as we would have used the FREE buffer,
      // but we won't worry about this for now

      // We can't evict pinned buffers - only normal usedbuffers

      // if (buffer_it->mem_status == FREE || buffer_it->buffer->getPinCount() == 0) {
      size_t page_count = 0;
      size_t score = 0;
      bool solution_found = false;
      auto evict_it = buffer_it;
      for (; evict_it != slab_segments_[slab_num].end(); ++evict_it) {
        // pinCount should never go up - only down because we have
        // global lock on buffer pool and pin count only increments
        // on getChunk
        if (evict_it->mem_status == USED && evict_it->buffer->getPinCount() > 0) {
          break;
        }
        page_count += evict_it->num_pages;
        if (evict_it->mem_status == USED) {
          // MAT changed from
          // score += evictIt->lastTouched;
          // Issue was thrashing when going from 8M fragment size chunks back to 64M
          // basically the large chunks were being evicted prior to small as many small
          // chunk score was larger than one large chunk so it always would evict a large
          // chunk so under memory pressure a query would evict its own current chunks and
          // cause reloads rather than evict several smaller unused older chunks.
          score = std::max(score, static_cast<size_t>(evict_it->last_touched));
        }
        if (page_count >= num_pages_requested) {
          solution_found = true;
          break;
        }
      }
      if (solution_found && score < min_score) {
        min_score = score;
        best_eviction_start = buffer_it;
        best_eviction_start_slab = slab_num;
      } else if (evict_it == slab_segments_[slab_num].end()) {
        // this means that every segment after this will fail as well, so our search has
        // proven futile
        // throw std::runtime_error ("Couldn't evict chunks to get free space");
        break;
        // in reality we should try to rearrange the buffer to get more contiguous free
        // space
      }
      // other possibility is ending at PINNED - do nothing in this case
      //}
    }
  }
  if (best_eviction_start == slab_segments_[0].end()) {
    LOG(ERROR) << "ALLOCATION failed to find " << num_bytes << "B throwing out of memory "
               << getStringMgrType() << ":" << device_id_;
    VLOG(2) << printSlabs();
    throw OutOfMemory(num_bytes);
  }
  LOG(INFO) << "ALLOCATION failed to find " << num_bytes << "B free. Forcing Eviction."
            << " Eviction start " << best_eviction_start->start_page
            << " Number pages requested " << num_pages_requested
            << " Best Eviction Start Slab " << best_eviction_start_slab << " "
            << getStringMgrType() << ":" << device_id_;
  best_eviction_start =
      evict(best_eviction_start, num_pages_requested, best_eviction_start_slab);
  return best_eviction_start;
}

std::string BufferMgr::printSlab(size_t slab_num) {
  std::ostringstream tss;
  // size_t lastEnd = 0;
  tss << "Slab St.Page   Pages  Touch" << std::endl;
  for (auto segment : slab_segments_[slab_num]) {
    tss << setfill(' ') << setw(4) << slab_num;
    // tss << " BSN: " << setfill(' ') << setw(2) << segment.slab_num;
    tss << setfill(' ') << setw(8) << segment.start_page;
    tss << setfill(' ') << setw(8) << segment.num_pages;
    // tss << " GAP: " << setfill(' ') << setw(7) << segment.start_page - lastEnd;
    // lastEnd = segment.start_page + segment.num_pages;
    tss << setfill(' ') << setw(7) << segment.last_touched;
    // tss << " PC: " << setfill(' ') << setw(2) << segment.buffer->getPinCount();
    if (segment.mem_status == FREE) {
      tss << " FREE"
          << " ";
    } else {
      tss << " PC: " << setfill(' ') << setw(2) << segment.buffer->getPinCount();
      tss << " USED - Chunk: ";

      for (auto&& key_elem : segment.chunk_key) {
        tss << key_elem << ",";
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
      << " " << getStringMgrType() << ":" << device_id_ << std::endl;
  size_t num_slabs = slab_segments_.size();
  for (size_t slab_num = 0; slab_num != num_slabs; ++slab_num) {
    tss << printSlab(slab_num);
  }
  tss << "--------------------" << std::endl;
  return tss.str();
}

void BufferMgr::clearSlabs() {
  bool pinned_exists = false;
  for (auto& segment_list : slab_segments_) {
    for (auto& segment : segment_list) {
      if (segment.mem_status == FREE) {
        // no need to free
      } else if (segment.buffer->getPinCount() < 1) {
        deleteBuffer(segment.chunk_key, true);
      } else {
        pinned_exists = true;
      }
    }
  }
  if (!pinned_exists) {
    // lets actually clear the buffer from memory
    freeAllMem();
    clear();
    reinit();
  }
}

// return the maximum size this buffer can be in bytes
size_t BufferMgr::getMaxSize() {
  return page_size_ * max_buffer_pool_num_pages_;
}

// return how large the buffer are currently allocated
size_t BufferMgr::getAllocated() {
  return num_pages_allocated_ * page_size_;
}

//
bool BufferMgr::isAllocationCapped() {
  return allocations_capped_;
}

size_t BufferMgr::getPageSize() {
  return page_size_;
}

// return the size of the chunks in use in bytes
size_t BufferMgr::getInUseSize() {
  size_t in_use = 0;
  for (auto& segment_list : slab_segments_) {
    for (auto& segment : segment_list) {
      if (segment.mem_status != FREE) {
        in_use += segment.num_pages * page_size_;
      }
    }
  }
  return in_use;
}

std::string BufferMgr::printSeg(BufferList::iterator& seg_it) {
  std::ostringstream tss;
  tss << "SN: " << setfill(' ') << setw(2) << seg_it->slab_num;
  tss << " SP: " << setfill(' ') << setw(7) << seg_it->start_page;
  tss << " NP: " << setfill(' ') << setw(7) << seg_it->num_pages;
  tss << " LT: " << setfill(' ') << setw(7) << seg_it->last_touched;
  tss << " PC: " << setfill(' ') << setw(2) << seg_it->buffer->getPinCount();
  if (seg_it->mem_status == FREE) {
    tss << " FREE"
        << " ";
  } else {
    tss << " USED - Chunk: ";
    for (auto vec_it = seg_it->chunk_key.begin(); vec_it != seg_it->chunk_key.end();
         ++vec_it) {
      tss << *vec_it << ",";
    }
    tss << std::endl;
  }
  return tss.str();
}

std::string BufferMgr::printMap() {
  std::ostringstream tss;
  int seg_num = 1;
  tss << std::endl
      << "Map Contents: "
      << " " << getStringMgrType() << ":" << device_id_ << std::endl;
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_);
  for (auto seg_it = chunk_index_.begin(); seg_it != chunk_index_.end();
       ++seg_it, ++seg_num) {
    //    tss << "Map Entry " << seg_num << ": ";
    //    for (auto vec_it = seg_it->first.begin(); vec_it != seg_it->first.end();
    //    ++vec_it)
    //    {
    //      tss << *vec_it << ",";
    //    }
    //    tss << " " << std::endl;
    tss << printSeg(seg_it->second);
  }
  tss << "--------------------" << std::endl;
  return tss.str();
}

void BufferMgr::printSegs() {
  int seg_num = 1;
  int slab_num = 1;
  LOG(INFO) << std::endl << " " << getStringMgrType() << ":" << device_id_;
  for (auto slab_it = slab_segments_.begin(); slab_it != slab_segments_.end();
       ++slab_it, ++slab_num) {
    LOG(INFO) << "Slab Num: " << slab_num << " " << getStringMgrType() << ":"
              << device_id_;
    for (auto seg_it = slab_it->begin(); seg_it != slab_it->end(); ++seg_it, ++seg_num) {
      LOG(INFO) << "Segment: " << seg_num << " " << getStringMgrType() << ":"
                << device_id_;
      printSeg(seg_it);
      LOG(INFO) << " " << getStringMgrType() << ":" << device_id_;
    }
    LOG(INFO) << "--------------------"
              << " " << getStringMgrType() << ":" << device_id_;
  }
}

bool BufferMgr::isBufferOnDevice(const ChunkKey& key) {
  std::lock_guard<std::mutex> chunkIndexLock(chunk_index_mutex_);
  if (chunk_index_.find(key) == chunk_index_.end()) {
    return false;
  } else {
    return true;
  }
}

/// This method throws a runtime_error when deleting a Chunk that does not exist.
void BufferMgr::deleteBuffer(const ChunkKey& key, const bool) {
  // Note: purge is unused
  std::unique_lock<std::mutex> chunk_index_lock(chunk_index_mutex_);

  // lookup the buffer for the Chunk in chunk_index_
  auto buffer_it = chunk_index_.find(key);
  CHECK(buffer_it != chunk_index_.end());
  auto seg_it = buffer_it->second;
  chunk_index_.erase(buffer_it);
  chunk_index_lock.unlock();
  std::lock_guard<std::mutex> sized_segs_lock(sized_segs_mutex_);
  if (seg_it->buffer) {
    delete seg_it->buffer;  // Delete Buffer for segment
    seg_it->buffer = 0;
  }
  removeSegment(seg_it);
}

void BufferMgr::deleteBuffersWithPrefix(const ChunkKey& key_prefix, const bool) {
  // Note: purge is unused
  // lookup the buffer for the Chunk in chunk_index_
  std::lock_guard<std::mutex> sized_segs_lock(
      sized_segs_mutex_);  // Take this lock early to prevent deadlock with
                           // reserveBuffer which needs segs_mutex_ and then
                           // chunk_index_mutex_
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_);
  auto startChunkIt = chunk_index_.lower_bound(key_prefix);
  if (startChunkIt == chunk_index_.end()) {
    return;
  }

  auto buffer_it = startChunkIt;
  while (buffer_it != chunk_index_.end() &&
         std::search(buffer_it->first.begin(),
                     buffer_it->first.begin() + key_prefix.size(),
                     key_prefix.begin(),
                     key_prefix.end()) != buffer_it->first.begin() + key_prefix.size()) {
    auto seg_it = buffer_it->second;
    if (seg_it->buffer) {
      delete seg_it->buffer;  // Delete Buffer for segment
      seg_it->buffer = 0;
    }
    removeSegment(seg_it);
    chunk_index_.erase(buffer_it++);
  }
}

void BufferMgr::removeSegment(BufferList::iterator& seg_it) {
  // Note: does not delete buffer as this may be moved somewhere else
  int slab_num = seg_it->slab_num;
  // cout << "Slab num: " << slabNum << endl;
  if (slab_num < 0) {
    std::lock_guard<std::mutex> unsized_segs_lock(unsized_segs_mutex_);
    unsized_segs_.erase(seg_it);
  } else {
    if (seg_it != slab_segments_[slab_num].begin()) {
      auto prev_it = std::prev(seg_it);
      // LOG(INFO) << "PrevIt: " << " " << getStringMgrType() << ":" << device_id_;
      // printSeg(prev_it);
      if (prev_it->mem_status == FREE) {
        seg_it->start_page = prev_it->start_page;
        seg_it->num_pages += prev_it->num_pages;
        slab_segments_[slab_num].erase(prev_it);
      }
    }
    auto next_it = std::next(seg_it);
    if (next_it != slab_segments_[slab_num].end()) {
      if (next_it->mem_status == FREE) {
        seg_it->num_pages += next_it->num_pages;
        slab_segments_[slab_num].erase(next_it);
      }
    }
    seg_it->mem_status = FREE;
    // seg_it->pinCount = 0;
    seg_it->buffer = 0;
  }
}

void BufferMgr::checkpoint() {
  std::lock_guard<std::mutex> lock(global_mutex_);  // granular lock
  std::lock_guard<std::mutex> chunkIndexLock(chunk_index_mutex_);

  for (auto& chunk_itr : chunk_index_) {
    // checks that buffer is actual chunk (not just buffer) and is dirty
    auto& buffer_itr = chunk_itr.second;
    if (buffer_itr->chunk_key[0] != -1 && buffer_itr->buffer->isDirty()) {
      parent_mgr_->putBuffer(buffer_itr->chunk_key, buffer_itr->buffer);
      buffer_itr->buffer->clearDirtyBits();
    }
  }
}

void BufferMgr::checkpoint(const int db_id, const int tb_id) {
  std::lock_guard<std::mutex> lock(global_mutex_);  // granular lock
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_);

  ChunkKey key_prefix;
  key_prefix.push_back(db_id);
  key_prefix.push_back(tb_id);
  auto start_chunk_it = chunk_index_.lower_bound(key_prefix);
  if (start_chunk_it == chunk_index_.end()) {
    return;
  }

  auto buffer_it = start_chunk_it;
  while (buffer_it != chunk_index_.end() &&
         std::search(buffer_it->first.begin(),
                     buffer_it->first.begin() + key_prefix.size(),
                     key_prefix.begin(),
                     key_prefix.end()) != buffer_it->first.begin() + key_prefix.size()) {
    if (buffer_it->second->chunk_key[0] != -1 &&
        buffer_it->second->buffer->isDirty()) {  // checks that buffer is actual chunk
                                                 // (not just buffer) and is dirty

      parent_mgr_->putBuffer(buffer_it->second->chunk_key, buffer_it->second->buffer);
      buffer_it->second->buffer->clearDirtyBits();
    }
    buffer_it++;
  }
}

/// Returns a pointer to the Buffer holding the chunk, if it exists; otherwise,
/// throws a runtime_error.
AbstractBuffer* BufferMgr::getBuffer(const ChunkKey& key, const size_t num_bytes) {
  std::lock_guard<std::mutex> lock(global_mutex_);  // granular lock

  std::unique_lock<std::mutex> sized_segs_lock(sized_segs_mutex_);
  std::unique_lock<std::mutex> chunk_index_lock(chunk_index_mutex_);
  auto buffer_it = chunk_index_.find(key);
  bool found_buffer = buffer_it != chunk_index_.end();
  chunk_index_lock.unlock();
  if (found_buffer) {
    CHECK(buffer_it->second->buffer);
    buffer_it->second->buffer->pin();
    sized_segs_lock.unlock();

    buffer_it->second->last_touched = buffer_epoch_++;  // race

    if (buffer_it->second->buffer->size() < num_bytes) {
      // need to fetch part of buffer we don't have - up to numBytes
      parent_mgr_->fetchBuffer(key, buffer_it->second->buffer, num_bytes);
    }
    return buffer_it->second->buffer;
  } else {  // If wasn't in pool then we need to fetch it
    sized_segs_lock.unlock();
    // createChunk pins for us
    AbstractBuffer* buffer = createBuffer(key, page_size_, num_bytes);
    try {
      parent_mgr_->fetchBuffer(
          key, buffer, num_bytes);  // this should put buffer in a BufferSegment
    } catch (const std::exception& error) {
      LOG(FATAL) << "Get chunk - Could not find chunk " << keyToString(key)
                 << " in buffer pool or parent buffer pools. Error was " << error.what();
    }
    return buffer;
  }
}

void BufferMgr::fetchBuffer(const ChunkKey& key,
                            AbstractBuffer* dest_buffer,
                            const size_t num_bytes) {
  std::unique_lock<std::mutex> lock(global_mutex_);  // granular lock
  std::unique_lock<std::mutex> sized_segs_lock(sized_segs_mutex_);
  std::unique_lock<std::mutex> chunk_index_lock(chunk_index_mutex_);

  auto buffer_it = chunk_index_.find(key);
  bool found_buffer = buffer_it != chunk_index_.end();
  chunk_index_lock.unlock();
  AbstractBuffer* buffer;
  if (!found_buffer) {
    sized_segs_lock.unlock();
    CHECK(parent_mgr_ != 0);
    buffer = createBuffer(key, page_size_, num_bytes);  // will pin buffer
    try {
      parent_mgr_->fetchBuffer(key, buffer, num_bytes);
    } catch (std::runtime_error& error) {
      LOG(FATAL) << "Could not fetch parent buffer " << keyToString(key);
    }
  } else {
    buffer = buffer_it->second->buffer;
    buffer->pin();
    if (num_bytes > buffer->size()) {
      try {
        parent_mgr_->fetchBuffer(key, buffer, num_bytes);
      } catch (std::runtime_error& error) {
        LOG(FATAL) << "Could not fetch parent buffer " << keyToString(key);
      }
    }
    sized_segs_lock.unlock();
  }
  lock.unlock();
  buffer->copyTo(dest_buffer, num_bytes);
  buffer->unPin();
}

AbstractBuffer* BufferMgr::putBuffer(const ChunkKey& key,
                                     AbstractBuffer* src_buffer,
                                     const size_t num_bytes) {
  std::unique_lock<std::mutex> chunk_index_lock(chunk_index_mutex_);
  auto buffer_it = chunk_index_.find(key);
  bool found_buffer = buffer_it != chunk_index_.end();
  chunk_index_lock.unlock();
  AbstractBuffer* buffer;
  if (!found_buffer) {
    buffer = createBuffer(key, page_size_);
  } else {
    buffer = buffer_it->second->buffer;
  }
  size_t old_buffer_size = buffer->size();
  size_t new_buffer_size = num_bytes == 0 ? src_buffer->size() : num_bytes;
  CHECK(!buffer->isDirty());

  if (src_buffer->isUpdated()) {
    //@todo use dirty flags to only flush pages of chunk that need to
    // be flushed
    buffer->write((int8_t*)src_buffer->getMemoryPtr(),
                  new_buffer_size,
                  0,
                  src_buffer->getType(),
                  src_buffer->getDeviceId());
  } else if (src_buffer->isAppended()) {
    CHECK(old_buffer_size < new_buffer_size);
    buffer->append((int8_t*)src_buffer->getMemoryPtr() + old_buffer_size,
                   new_buffer_size - old_buffer_size,
                   src_buffer->getType(),
                   src_buffer->getDeviceId());
  } else {
    UNREACHABLE();
  }
  src_buffer->clearDirtyBits();
  buffer->syncEncoder(src_buffer);
  return buffer;
}

int BufferMgr::getBufferId() {
  std::lock_guard<std::mutex> lock(buffer_id_mutex_);
  return max_buffer_id_++;
}

/// client is responsible for deleting memory allocated for b->mem_
AbstractBuffer* BufferMgr::alloc(const size_t num_bytes) {
  std::lock_guard<std::mutex> lock(global_mutex_);
  ChunkKey chunk_key = {-1, getBufferId()};
  return createBuffer(chunk_key, page_size_, num_bytes);
}

void BufferMgr::free(AbstractBuffer* buffer) {
  std::lock_guard<std::mutex> lock(global_mutex_);  // hack for now
  Buffer* casted_buffer = dynamic_cast<Buffer*>(buffer);
  if (casted_buffer == 0) {
    LOG(FATAL) << "Wrong buffer type - expects base class pointer to Buffer type.";
  }
  deleteBuffer(casted_buffer->seg_it_->chunk_key);
}

size_t BufferMgr::getNumChunks() {
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_);
  return chunk_index_.size();
}

size_t BufferMgr::size() {
  return num_pages_allocated_;
}

size_t BufferMgr::getMaxBufferSize() {
  return max_buffer_pool_size_;
}

size_t BufferMgr::getMaxSlabSize() {
  return max_slab_size_;
}

void BufferMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata_vec,
                                                const ChunkKey& key_prefix) {
  LOG(FATAL) << "getChunkMetadataVecForPrefix not supported for BufferMgr.";
}

const std::vector<BufferList>& BufferMgr::getSlabSegments() {
  return slab_segments_;
}

void BufferMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  UNREACHABLE();
}
}  // namespace Buffer_Namespace
