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
 * @file    BufferMgr.cpp
 * @brief
 */
#include "DataMgr/BufferMgr/BufferMgr.h"

#include <algorithm>
#include <iomanip>
#include <limits>

#include "DataMgr/BufferMgr/Buffer.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "Logger/Logger.h"
#include "Shared/measure.h"

using namespace std;

namespace Buffer_Namespace {

namespace {
inline std::string key_to_string(const ChunkKey& key) {
  std::ostringstream oss;

  oss << " key: ";
  for (auto sub_key : key) {
    oss << sub_key << ",";
  }
  return oss.str();
}

inline size_t convert_num_bytes_to_num_pages(size_t num_bytes, size_t page_size) {
  CHECK_GT(page_size, size_t(0));
  CHECK_EQ(num_bytes % page_size, size_t(0));
  return num_bytes / page_size;
}
}  // namespace

/// Allocates memSize bytes for the buffer pool and initializes the free memory map.
BufferMgr::BufferMgr(const int device_id,
                     const size_t max_buffer_pool_size,
                     const size_t min_slab_size,
                     const size_t max_slab_size,
                     const size_t default_slab_size,
                     const size_t page_size,
                     AbstractBufferMgr* parent_mgr)
    : AbstractBufferMgr(device_id)
    , max_buffer_pool_size_(max_buffer_pool_size)
    , min_slab_size_(min_slab_size)
    , max_slab_size_(max_slab_size)
    , default_slab_size_(default_slab_size)
    , page_size_(page_size)
    , max_buffer_pool_num_pages_(
          convert_num_bytes_to_num_pages(max_buffer_pool_size_, page_size_))
    , min_num_pages_per_slab_(convert_num_bytes_to_num_pages(min_slab_size_, page_size_))
    , max_num_pages_per_slab_(convert_num_bytes_to_num_pages(max_slab_size_, page_size_))
    , num_pages_allocated_(0)
    , allocations_capped_(false)
    , parent_mgr_(parent_mgr)
    , max_buffer_id_(0)
    , buffer_epoch_(0) {
  CHECK_GT(max_buffer_pool_size_, size_t(0));
  CHECK_GT(page_size_, size_t(0));
  // TODO change checks on run-time configurable slab size variables to exceptions
  CHECK_GT(min_slab_size_, size_t(0));
  CHECK_GT(max_slab_size_, size_t(0));
  CHECK_GT(default_slab_size_, size_t(0));
  CHECK_LE(min_slab_size_, max_slab_size_);
  CHECK_GE(default_slab_size_, min_slab_size_);
  CHECK_LE(default_slab_size_, max_slab_size_);
  CHECK_EQ(min_slab_size_ % page_size_, size_t(0));
  CHECK_EQ(max_slab_size_ % page_size_, size_t(0));
  CHECK_EQ(default_slab_size_ % page_size_, size_t(0));

  default_num_pages_per_slab_ = default_slab_size_ / page_size_;
  current_max_num_pages_per_slab_ =
      max_num_pages_per_slab_;  // current_max_num_pages_per_slab_ will drop as
                                // allocations fail - this is the high water mark
}

/// Frees the heap-allocated buffer pool memory
BufferMgr::~BufferMgr() {
  std::unique_lock<std::shared_mutex> slab_lock(slab_mutex_);
  clear();
}

void BufferMgr::reinit() {
  num_pages_allocated_ = 0;
  current_max_num_pages_per_slab_ =
      max_num_pages_per_slab_;  // current_max_num_pages_per_slab_ will drop as
                                // allocations fail - this is the high water mark
  allocations_capped_ = false;
}

void BufferMgr::clear() {
  clearChunks();
  clearSlabContainers();
  clearUnsizedSegments();
  clearEpoch();
}

/// Throws a runtime_error if the Chunk already exists
AbstractBuffer* BufferMgr::createBuffer(const ChunkKey& chunk_key,
                                        const size_t chunk_page_size,
                                        const size_t initial_size) {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::lock_guard<std::mutex> chunk_lock(getChunkMutex(chunk_key));
  return createBufferUnlocked(chunk_key, chunk_page_size, initial_size);
}

AbstractBuffer* BufferMgr::createBufferUnlocked(const ChunkKey& chunk_key,
                                                const size_t chunk_page_size,
                                                const size_t initial_size) {
  // LOG(INFO) << printMap();
  size_t actual_chunk_page_size = chunk_page_size;
  if (actual_chunk_page_size == 0) {
    actual_chunk_page_size = page_size_;
  }
  auto buffer_seg_it = addBufferPlaceholder(chunk_key);

  // following should be safe outside the lock b/c first thing Buffer
  // constructor does is pin (and its still in unsized segs at this point
  // so can't be evicted)
  try {
    allocateBuffer(buffer_seg_it, actual_chunk_page_size, initial_size);
  } catch (const OutOfMemory&) {
    auto buffer_it = getChunkSegment(chunk_key);
    CHECK(buffer_it.has_value());
    deleteBufferUnlocked(chunk_key);
    throw;
  }
  auto buffer_it = getChunkSegment(chunk_key);
  CHECK(buffer_it.has_value());
  auto buffer = buffer_it.value()->getBuffer();
  CHECK(initial_size == 0 || buffer->getMemoryPtr());
  return buffer;
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
      CHECK(evict_it->getBuffer()->getPinCount() < 1);
    }
    num_pages += evict_it->num_pages;
    if (evict_it->mem_status == USED && evict_it->chunk_key.size() > 0) {
      eraseChunkSegment(evict_it->chunk_key);
    }
    if (evict_it->getBuffer() != nullptr) {
      // If we don't delete buffers here then we lose reference to them later and cause a
      // memleak.
      delete evict_it->getBuffer();
    }
    evict_it = slab_segments_[slab_num].erase(
        evict_it);  // erase operations returns next iterator - safe if we ever move
                    // to a vector (as opposed to erase(evict_it++)
  }
  BufferSeg data_seg(
      start_page, num_pages_requested, USED, incrementEpoch());  // until we can
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

  BufferList::iterator new_seg_it;
  std::unique_lock<std::shared_mutex> slab_lock(slab_mutex_);
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
  new_seg_it = findFreeBuffer(num_bytes);

  // Below should be in copy constructor for BufferSeg?
  new_seg_it->setBuffer(seg_it->getBuffer());
  new_seg_it->chunk_key = seg_it->chunk_key;
  int8_t* old_mem = new_seg_it->getBuffer()->mem_;
  new_seg_it->getBuffer()->mem_ =
      slabs_[new_seg_it->slab_num] + new_seg_it->start_page * page_size_;

  // now need to copy over memory
  // only do this if the old segment is valid (i.e. not new w/ unallocated buffer
  if (seg_it->start_page >= 0 && seg_it->getBuffer()->mem_ != 0) {
    new_seg_it->getBuffer()->writeData(old_mem,
                                       new_seg_it->getBuffer()->size(),
                                       0,
                                       new_seg_it->getBuffer()->getType(),
                                       device_id_);
  }
  // Decrement pin count to reverse effect above
  removeSegment(seg_it);

  {
    std::unique_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
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
      buffer_it->setLastTouched(incrementEpoch());
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

  // TODO: Move new slab creation to separate method
  // If we're here then we didn't find a free segment of sufficient size
  // First we see if we can add another slab
  while (!allocations_capped_ && num_pages_allocated_ < max_buffer_pool_num_pages_) {
    size_t allocated_num_pages{0};
    try {
      auto pages_left = max_buffer_pool_num_pages_ - num_pages_allocated_;
      if (pages_left < current_max_num_pages_per_slab_) {
        current_max_num_pages_per_slab_ = pages_left;
      }
      if (num_pages_requested <=
          current_max_num_pages_per_slab_) {  // don't try to allocate if the
                                              // new slab won't be big enough
        if (default_num_pages_per_slab_ < current_max_num_pages_per_slab_) {
          allocated_num_pages =
              std::max(default_num_pages_per_slab_, num_pages_requested);
        } else {
          allocated_num_pages = current_max_num_pages_per_slab_;
        }
        const auto slab_in_bytes = allocated_num_pages * page_size_;
        VLOG(1) << "Try to allocate SLAB of " << allocated_num_pages << " pages ("
                << slab_in_bytes << " bytes) on " << getStringMgrType() << ":"
                << device_id_;
        auto alloc_ms = measure<>::execution([&]() { addSlab(slab_in_bytes); });
        LOG(INFO) << "ALLOCATION slab of " << allocated_num_pages << " pages ("
                  << slab_in_bytes << "B) created in " << alloc_ms << " ms "
                  << getStringMgrType() << ":" << device_id_;
      } else {
        break;
      }
      // if here then addSlab succeeded
      CHECK_GT(allocated_num_pages, size_t(0));
      num_pages_allocated_ += allocated_num_pages;
      return findFreeBufferInSlab(
          num_slabs,
          num_pages_requested);  // has to succeed since we made sure to request a slab
                                 // big enough to accomodate request
    } catch (std::runtime_error& error) {  // failed to allocate slab
      LOG(INFO) << "ALLOCATION Attempted slab of " << allocated_num_pages << " pages ("
                << (allocated_num_pages * page_size_) << "B) failed "
                << getStringMgrType() << ":" << device_id_;
      // check if there is any point halving currentMaxSlabSize and trying again
      // if the request wont fit in half available then let try once at full size
      // if we have already tries at full size and failed then break as
      // there could still be room enough for other later request but
      // not for his current one
      if (num_pages_requested > current_max_num_pages_per_slab_ / 2 &&
          current_max_num_pages_per_slab_ != num_pages_requested) {
        current_max_num_pages_per_slab_ = num_pages_requested;
      } else {
        current_max_num_pages_per_slab_ /= 2;
        if (current_max_num_pages_per_slab_ < min_num_pages_per_slab_) {
          allocations_capped_ = true;
          // dump out the slabs and their sizes
          LOG(INFO) << "ALLOCATION Capped " << current_max_num_pages_per_slab_
                    << " Minimum size = " << min_num_pages_per_slab_ << " "
                    << getStringMgrType() << ":" << device_id_;
        }
      }
    }
  }

  if (num_pages_allocated_ == 0 && allocations_capped_) {
    throw FailedToCreateFirstSlab(num_bytes);
  }

  // TODO: Move eviction to separate method
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

      size_t page_count = 0;
      size_t score = 0;
      bool solution_found = false;
      auto evict_it = buffer_it;
      for (; evict_it != slab_segments_[slab_num].end(); ++evict_it) {
        // We can't evict pinned buffers - only normal used buffers.
        // pinCount should never go up - only down because we have
        // global lock on buffer pool and pin count only increments
        // on getChunk
        if (evict_it->mem_status == USED) {
          auto buffer = evict_it->getBuffer();
          if (buffer->getPinCount() > 0) {
            break;
          }
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
          score = std::max(score, static_cast<size_t>(evict_it->getLastTouched()));
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
  for (const auto& segment : slab_segments_[slab_num]) {
    tss << setfill(' ') << setw(4) << slab_num;
    // tss << " BSN: " << setfill(' ') << setw(2) << segment.slab_num;
    tss << setfill(' ') << setw(8) << segment.start_page;
    tss << setfill(' ') << setw(8) << segment.num_pages;
    // tss << " GAP: " << setfill(' ') << setw(7) << segment.start_page - lastEnd;
    // lastEnd = segment.start_page + segment.num_pages;
    tss << setfill(' ') << setw(7) << segment.getLastTouched();
    // tss << " PC: " << setfill(' ') << setw(2) << segment.buffer->getPinCount();
    if (segment.mem_status == FREE) {
      tss << " FREE"
          << " ";
    } else {
      tss << " PC: " << setfill(' ') << setw(2) << segment.getBuffer()->getPinCount();
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
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
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
  LOG(INFO) << "Attempting to acquire clearSlabs global lock for " << getStringMgrType()
            << " buffer manager on device : " << device_id_;
  std::unique_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  LOG(INFO) << "Acquired clearSlabs global lock for " << getStringMgrType()
            << " buffer manager on device : " << device_id_;

  std::vector<ChunkKey> used_chunk_keys;
  {
    std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
    for (auto& segment_list : slab_segments_) {
      for (auto& segment : segment_list) {
        if (segment.mem_status != FREE) {
          used_chunk_keys.emplace_back(segment.chunk_key);
        }
      }
    }
  }

  bool pinned_exists = false;
  for (const auto& chunk_key : used_chunk_keys) {
    std::lock_guard<std::mutex> chunk_lock(getChunkMutex(chunk_key));
    auto buffer_it = getChunkSegment(chunk_key);
    if (buffer_it.has_value()) {
      auto buffer = buffer_it.value()->getBuffer();
      CHECK(buffer) << show_chunk(chunk_key);
      if (buffer->getPinCount() < 1) {
        deleteBufferUnlocked(chunk_key);
      } else {
        pinned_exists = true;
      }
    }
  }

  if (!pinned_exists) {
    // lets actually clear the buffer from memory
    LOG(INFO) << getStringMgrType() << ":" << device_id_ << " clear slab memory";
    std::unique_lock<std::shared_mutex> slab_lock(slab_mutex_);
    freeAllMem();
    clear();
    reinit();
  } else {
    LOG(INFO) << getStringMgrType() << ":" << device_id_ << " keep slab memory (pinned).";
  }
  LOG(INFO) << "Releasing clearSlabs global lock for " << getStringMgrType()
            << " buffer manager on device : " << device_id_;
}

// return the maximum size this buffer can be in bytes
size_t BufferMgr::getMaxSize() const {
  return page_size_ * max_buffer_pool_num_pages_;
}

// return how large the buffer are currently allocated
size_t BufferMgr::getAllocated() const {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  return num_pages_allocated_ * page_size_;
}

//
bool BufferMgr::isAllocationCapped() const {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  return allocations_capped_;
}

size_t BufferMgr::getPageSize() const {
  return page_size_;
}

// return the size of the chunks in use in bytes
size_t BufferMgr::getInUseSize() const {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  size_t in_use = 0;
  for (const auto& segment_list : slab_segments_) {
    for (const auto& segment : segment_list) {
      if (segment.mem_status != FREE) {
        in_use += segment.num_pages * page_size_;
      }
    }
  }
  return in_use;
}

std::string BufferMgr::printSeg(const BufferList::iterator& seg_it) {
  std::ostringstream tss;
  tss << "SN: " << setfill(' ') << setw(2) << seg_it->slab_num;
  tss << " SP: " << setfill(' ') << setw(7) << seg_it->start_page;
  tss << " NP: " << setfill(' ') << setw(7) << seg_it->num_pages;
  tss << " LT: " << setfill(' ') << setw(7) << seg_it->getLastTouched();
  tss << " PC: " << setfill(' ') << setw(2) << seg_it->getBuffer()->getPinCount();
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
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::ostringstream tss;
  int seg_num = 1;
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  tss << std::endl
      << "Map Contents: "
      << " " << getStringMgrType() << ":" << device_id_ << std::endl;
  std::shared_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
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
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  int seg_num = 1;
  int slab_num = 1;
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
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
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  return getChunkSegment(key).has_value();
}

/// This method throws a runtime_error when deleting a Chunk that does not exist.
void BufferMgr::deleteBuffer(const ChunkKey& key, const bool) {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::lock_guard<std::mutex> chunk_lock(getChunkMutex(key));
  deleteBufferUnlocked(key);
}

void BufferMgr::deleteBufferUnlocked(const ChunkKey& key, const bool) {
  // Note: purge is unused

  // lookup the buffer for the Chunk in chunk_index_
  auto buffer_it = getChunkSegment(key);
  CHECK(buffer_it.has_value());
  auto seg_it = buffer_it.value();
  eraseChunkSegment(key);
  {
    std::unique_lock<std::shared_mutex> slab_lock(slab_mutex_);
    if (seg_it->getBuffer()) {
      delete seg_it->getBuffer();  // Delete Buffer for segment
      seg_it->setBuffer(nullptr);
    }
    removeSegment(seg_it);
  }
}

void BufferMgr::deleteBuffersWithPrefix(const ChunkKey& key_prefix, const bool) {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  // Note: purge is unused
  std::vector<ChunkKey> chunk_keys_to_delete;
  {
    std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
    std::shared_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
    auto prefix_upper_bound = key_prefix;
    prefix_upper_bound.emplace_back(std::numeric_limits<ChunkKey::value_type>::max());
    for (auto buffer_it = chunk_index_.lower_bound(key_prefix),
              end_chunk_it = chunk_index_.upper_bound(prefix_upper_bound);
         buffer_it != end_chunk_it;) {
      chunk_keys_to_delete.emplace_back(buffer_it->first);
      buffer_it++;
    }
  }

  for (const auto& chunk_key : chunk_keys_to_delete) {
    std::lock_guard<std::mutex> chunk_lock(getChunkMutex(chunk_key));
    auto buffer_it = getChunkSegment(chunk_key);
    if (buffer_it.has_value()) {
      auto buffer = buffer_it.value()->getBuffer();
      CHECK(buffer) << show_chunk(chunk_key);
      if (buffer->getPinCount() == 0) {
        deleteBufferUnlocked(chunk_key);
      }
    }
  }
}

void BufferMgr::removeSegment(BufferList::iterator& seg_it) {
  // Note: does not delete buffer as this may be moved somewhere else
  int slab_num = seg_it->slab_num;
  // cout << "Slab num: " << slabNum << endl;
  if (slab_num < 0) {
    eraseUnsizedSegment(seg_it);
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
    seg_it->setBuffer(nullptr);
  }
}

void BufferMgr::checkpoint() {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::vector<ChunkKey> chunk_keys_to_checkpoint;
  {
    std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
    std::shared_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
    for (auto& chunk_itr : chunk_index_) {
      // checks that buffer is actual chunk (not just buffer)
      auto& buffer_itr = chunk_itr.second;
      if (buffer_itr->chunk_key[0] != -1) {
        chunk_keys_to_checkpoint.emplace_back(buffer_itr->chunk_key);
      }
    }
  }
  checkpoint(chunk_keys_to_checkpoint);
}

void BufferMgr::checkpoint(const int db_id, const int tb_id) {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::vector<ChunkKey> chunk_keys_to_checkpoint;
  {
    std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
    std::shared_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
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
                       key_prefix.end()) !=
               buffer_it->first.begin() + key_prefix.size()) {
      if (buffer_it->second->chunk_key[0] != -1) {  // checks that buffer is actual chunk
                                                    // (not just buffer)
        chunk_keys_to_checkpoint.emplace_back(buffer_it->first);
      }
      buffer_it++;
    }
  }
  checkpoint(chunk_keys_to_checkpoint);
}

void BufferMgr::checkpoint(const std::vector<ChunkKey>& chunk_keys) {
  for (const auto& chunk_key : chunk_keys) {
    std::lock_guard<std::mutex> chunk_lock(getChunkMutex(chunk_key));
    auto buffer_it = getChunkSegment(chunk_key);
    if (buffer_it.has_value()) {
      auto buffer = buffer_it.value()->getBuffer();
      CHECK(buffer) << show_chunk(chunk_key);
      if (buffer->isDirty()) {
        parent_mgr_->putBuffer(chunk_key, buffer);
        buffer->clearDirtyBits();
      }
    }
  }
}

/// Returns a pointer to the Buffer holding the chunk, if it exists; otherwise,
/// throws a runtime_error.
AbstractBuffer* BufferMgr::getBuffer(const ChunkKey& key, const size_t num_bytes) {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::lock_guard<std::mutex> chunk_lock(getChunkMutex(key));
  // Hold a slab read lock in order to ensure that the buffer, if found, cannot be evicted
  // before it is pinned.
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  auto buffer_it = getChunkSegment(key);
  bool found_buffer = buffer_it.has_value();
  if (found_buffer) {
    auto buffer = buffer_it.value()->getBuffer();
    CHECK(buffer);
    buffer->pin();
    slab_lock.unlock();

    buffer_it.value()->setLastTouched(incrementEpoch());

    auto buffer_size = buffer->size();
    if (buffer_size < num_bytes) {
      // need to fetch part of buffer we don't have - up to numBytes
      VLOG(1) << ToString(getMgrType())
              << ": Fetching buffer from parent manager. Reason: increased buffer size. "
                 "Buffer size: "
              << buffer_size << ", num bytes to fetch: " << num_bytes
              << ", chunk key: " << key_to_string(key);
      parent_mgr_->fetchBuffer(key, buffer, num_bytes);
    }
    return buffer;
  } else {  // If wasn't in pool then we need to fetch it
    slab_lock.unlock();

    // createChunk pins for us
    AbstractBuffer* buffer = createBufferUnlocked(key, page_size_, num_bytes);
    try {
      VLOG(1) << ToString(getMgrType())
              << ": Fetching buffer from parent manager. Reason: cache miss. Num bytes "
                 "to fetch: "
              << num_bytes << ", chunk key: " << key_to_string(key);
      parent_mgr_->fetchBuffer(
          key, buffer, num_bytes);  // this should put buffer in a BufferSegment
    } catch (const foreign_storage::ForeignStorageException& error) {
      deleteBufferUnlocked(key);  // buffer failed to load, ensure it is cleaned up
      LOG(WARNING) << "Get chunk - Could not load chunk " << key_to_string(key)
                   << " from foreign storage. Error was " << error.what();
      throw;
    } catch (const std::exception& error) {
      LOG(FATAL) << "Get chunk - Could not find chunk " << key_to_string(key)
                 << " in buffer pool or parent buffer pools. Error was " << error.what();
    }
    return buffer;
  }
}

void BufferMgr::fetchBuffer(const ChunkKey& key,
                            AbstractBuffer* dest_buffer,
                            const size_t num_bytes) {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::lock_guard<std::mutex> chunk_lock(getChunkMutex(key));
  // Hold a slab read lock in order to ensure that the buffer, if found, cannot be evicted
  // before it is pinned.
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  auto buffer_it = getChunkSegment(key);
  bool found_buffer = buffer_it.has_value();

  AbstractBuffer* buffer;
  if (!found_buffer) {
    slab_lock.unlock();
    CHECK(parent_mgr_ != 0);
    buffer = createBufferUnlocked(key, page_size_, num_bytes);  // will pin buffer
    try {
      VLOG(1) << ToString(getMgrType())
              << ": Fetching buffer from parent manager. Reason: cache miss. Num bytes "
                 "to fetch: "
              << num_bytes << ", chunk key: " << key_to_string(key);
      parent_mgr_->fetchBuffer(key, buffer, num_bytes);
    } catch (const foreign_storage::ForeignStorageException& error) {
      deleteBufferUnlocked(key);  // buffer failed to load, ensure it is cleaned up
      LOG(WARNING) << "Could not fetch parent chunk " << key_to_string(key)
                   << " from foreign storage. Error was " << error.what();
      throw;
    } catch (std::runtime_error& error) {
      LOG(FATAL) << "Could not fetch parent buffer " << key_to_string(key)
                 << " error: " << error.what();
    }
  } else {
    buffer = buffer_it.value()->getBuffer();
    buffer->pin();
    slab_lock.unlock();

    auto buffer_size = buffer->size();
    if (num_bytes > buffer_size) {
      try {
        VLOG(1) << ToString(getMgrType())
                << ": Fetching buffer from parent manager. Reason: increased buffer "
                   "size. Buffer size: "
                << buffer_size << ", num bytes to fetch: " << num_bytes
                << ", chunk key: " << key_to_string(key);
        parent_mgr_->fetchBuffer(key, buffer, num_bytes);
      } catch (const foreign_storage::ForeignStorageException& error) {
        LOG(WARNING) << "Could not fetch parent chunk " << key_to_string(key)
                     << " from foreign storage. Error was " << error.what();
        throw;
      } catch (std::runtime_error& error) {
        LOG(FATAL) << "Could not fetch parent buffer " << key_to_string(key)
                   << " error: " << error.what();
      }
    }
  }
  buffer->copyTo(dest_buffer, num_bytes);
  buffer->unPin();
}

AbstractBuffer* BufferMgr::putBuffer(const ChunkKey& key,
                                     AbstractBuffer* src_buffer,
                                     const size_t num_bytes) {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::lock_guard<std::mutex> chunk_lock(getChunkMutex(key));
  // Hold a slab read lock in order to ensure that the buffer, if found, cannot be evicted
  // before it is pinned.
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  auto buffer_it = getChunkSegment(key);
  bool found_buffer = buffer_it.has_value();

  AbstractBuffer* buffer;
  if (!found_buffer) {
    slab_lock.unlock();
    buffer = createBufferUnlocked(key, page_size_);
  } else {
    buffer = buffer_it.value()->getBuffer();
    buffer->pin();
    slab_lock.unlock();
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
  const ChunkKey chunk_key{-1, getBufferId()};
  return createBuffer(chunk_key, page_size_, num_bytes);
}

void BufferMgr::free(AbstractBuffer* buffer) {
  Buffer* casted_buffer = dynamic_cast<Buffer*>(buffer);
  if (casted_buffer == 0) {
    LOG(FATAL) << "Wrong buffer type - expects base class pointer to Buffer type.";
  }
  deleteBuffer(casted_buffer->seg_it_->chunk_key);
}

size_t BufferMgr::getNumChunks() {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::shared_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
  return chunk_index_.size();
}

size_t BufferMgr::size() const {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  return num_pages_allocated_;
}

size_t BufferMgr::getMaxBufferSize() const {
  return max_buffer_pool_size_;
}

size_t BufferMgr::getMaxSlabSize() const {
  return max_slab_size_;
}

void BufferMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata_vec,
                                                const ChunkKey& key_prefix) {
  LOG(FATAL) << "getChunkMetadataVecForPrefix not supported for BufferMgr.";
}

const std::vector<BufferList>& BufferMgr::getSlabSegments() {
  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  return slab_segments_;
}

uint32_t BufferMgr::incrementEpoch() {
  std::lock_guard<std::mutex> epoch_lock(buffer_epoch_mutex_);
  return buffer_epoch_++;
}

void BufferMgr::clearEpoch() {
  std::lock_guard<std::mutex> epoch_lock(buffer_epoch_mutex_);
  buffer_epoch_ = 0;
}

BufferList::iterator BufferMgr::addBufferPlaceholder(const ChunkKey& chunk_key) {
  auto buffer_seg_it = addUnsizedSegment(chunk_key);

  std::unique_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
  CHECK(chunk_index_.find(chunk_key) == chunk_index_.end());
  chunk_index_[chunk_key] = buffer_seg_it;
  return chunk_index_[chunk_key];
}

BufferList::iterator BufferMgr::addUnsizedSegment(const ChunkKey& chunk_key) {
  std::lock_guard<std::mutex> unsized_segs_lock(unsized_segs_mutex_);
  unsized_segs_.emplace_back(-1, 0, USED);
  unsized_segs_.back().chunk_key = chunk_key;
  return std::prev(unsized_segs_.end(), 1);
}

void BufferMgr::eraseUnsizedSegment(const BufferList::iterator& seg_it) {
  std::lock_guard<std::mutex> unsized_segs_lock(unsized_segs_mutex_);
  unsized_segs_.erase(seg_it);
}

void BufferMgr::clearUnsizedSegments() {
  std::lock_guard<std::mutex> unsized_segs_lock(unsized_segs_mutex_);
  unsized_segs_.clear();
}

std::mutex& BufferMgr::getChunkMutex(const ChunkKey& chunk_key) {
  std::unique_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
  return chunk_mutex_map_[chunk_key];
}

std::optional<BufferList::iterator> BufferMgr::getChunkSegment(
    const ChunkKey& chunk_key) const {
  std::shared_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
  auto it = chunk_index_.find(chunk_key);
  if (it != chunk_index_.end()) {
    return it->second;
  }
  return {};
}

void BufferMgr::eraseChunkSegment(const ChunkKey& chunk_key) {
  std::unique_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
  chunk_index_.erase(chunk_key);
}

void BufferMgr::clearChunks() {
  std::unique_lock<std::shared_mutex> chunk_index_lock(chunk_index_mutex_);
  for (auto& [chunk_key, seg_it] : chunk_index_) {
    delete seg_it->getBuffer();
  }
  chunk_index_.clear();
  chunk_mutex_map_.clear();
}

void BufferMgr::clearSlabContainers() {
  slabs_.clear();
  slab_segments_.clear();
}

void BufferMgr::allocateBuffer(BufferList::iterator seg_it,
                               size_t page_size,
                               size_t num_bytes) {
  auto buffer = createBuffer(seg_it, page_size);  // this line is admittedly a bit weird
                                                  // but the segment iterator passed into
                                                  // buffer takes the address of the new
                                                  // Buffer in its buffer member
  CHECK(buffer);
  seg_it->setBuffer(buffer);
  buffer->reserve(num_bytes);
}

void BufferMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  UNREACHABLE();
}

MemoryInfo BufferMgr::getMemoryInfo() const {
  MemoryInfo memory_info;
  memory_info.page_size = getPageSize();
  memory_info.max_num_pages = getMaxSize() / memory_info.page_size;
  memory_info.is_allocation_capped = isAllocationCapped();
  memory_info.num_page_allocated = getAllocated() / memory_info.page_size;

  std::shared_lock<std::shared_mutex> clear_slabs_global_lock(clear_slabs_global_mutex_);
  std::shared_lock<std::shared_mutex> slab_lock(slab_mutex_);
  for (size_t slab_num = 0; slab_num < slab_segments_.size(); slab_num++) {
    for (const auto& segment : slab_segments_[slab_num]) {
      MemoryData memory_data;
      memory_data.slab_num = slab_num;
      memory_data.start_page = segment.start_page;
      memory_data.num_pages = segment.num_pages;
      memory_data.touch = segment.getLastTouched();
      memory_data.mem_status = segment.mem_status;
      memory_data.chunk_key.insert(memory_data.chunk_key.end(),
                                   segment.chunk_key.begin(),
                                   segment.chunk_key.end());
      memory_info.node_memory_data.push_back(memory_data);
    }
  }
  return memory_info;
}
}  // namespace Buffer_Namespace
