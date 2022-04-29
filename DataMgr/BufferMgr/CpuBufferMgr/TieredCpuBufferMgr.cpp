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

#include "DataMgr/BufferMgr/CpuBufferMgr/TieredCpuBufferMgr.h"
#include "DataMgr/Allocators/ArenaAllocator.h"
#include "DataMgr/BufferMgr/CpuBufferMgr/CpuBuffer.h"
#include "DataMgr/GpuMgr.h"
#include "Shared/misc.h"

#include <iostream>

namespace {
std::string tier_to_string(CpuTier tier) {
  switch (tier) {
    case DRAM:
      return "DRAM";
    case PMEM:
      return "PMEM";
    default:
      return "<UNKNOWN>";
  }
}
}  // namespace

namespace Buffer_Namespace {

TieredCpuBufferMgr::TieredCpuBufferMgr(const int device_id,
                                       const size_t total_size,
                                       GpuMgr* gpu_mgr,
                                       const size_t min_slab_size,
                                       const size_t max_slab_size,
                                       const size_t page_size,
                                       const CpuTierSizeVector& cpu_tier_sizes,
                                       AbstractBufferMgr* parent_mgr)
    : CpuBufferMgr(device_id,
                   total_size,
                   gpu_mgr,
                   min_slab_size,
                   max_slab_size,
                   page_size,
                   parent_mgr) {
  CHECK(cpu_tier_sizes.size() == numCpuTiers);
  for (auto tier_size : cpu_tier_sizes) {
    allocators_.emplace_back(std::unique_ptr<Arena>{}, tier_size);
  }
  initializeMem();
}

Arena* TieredCpuBufferMgr::getAllocatorForSlab(int32_t slab_num) const {
  return shared::get_from_map(slab_to_allocator_map_, slab_num);
}

void TieredCpuBufferMgr::addSlab(const size_t slab_size) {
  CHECK(!allocators_.empty());
  CHECK(allocators_.begin()->first.get() != nullptr);
  slabs_.resize(slabs_.size() + 1);
  auto allocated_slab = false;
  CpuTier last_tier;
  for (auto allocator_type : {CpuTier::DRAM, CpuTier::PMEM}) {
    last_tier = allocator_type;
    auto& [allocator, allocator_limit] = allocators_.at(allocator_type);
    // If there is no space in the current allocator then move to the next one.
    if (allocator_limit >= allocator->bytesUsed() + slab_size) {
      try {
        slabs_.back() = reinterpret_cast<int8_t*>(allocator->allocate(slab_size));
      } catch (std::bad_alloc&) {
        // If anything goes wrong with an allocation, then throw an exception rather than
        // go to the next allocator.
        slabs_.resize(slabs_.size() - 1);
        throw FailedToCreateSlab(slab_size);
      }
      slab_to_allocator_map_[slabs_.size() - 1] = allocator.get();
      allocated_slab = true;
      break;
    }
  }
  if (allocated_slab) {
    // We allocated a new slab, so add segments for it.
    slab_segments_.resize(slab_segments_.size() + 1);
    slab_segments_[slab_segments_.size() - 1].push_back(
        BufferSeg(0, slab_size / page_size_));
    LOG(INFO) << "Allocated slab using " << tier_to_string(last_tier) << ".";
  } else {
    // None of the allocators allocated a slab, so revert to original size and throw.
    slabs_.resize(slabs_.size() - 1);
    throw FailedToCreateSlab(slab_size);
  }
}

void TieredCpuBufferMgr::freeAllMem() {
  CHECK(!allocators_.empty());
  CHECK(allocators_.begin()->first.get() != nullptr);
  initializeMem();
}

void TieredCpuBufferMgr::initializeMem() {
  for (auto& [allocator, allocator_limit] : allocators_) {
    allocator.reset(new Arena(max_slab_size_ + kArenaBlockOverhead));
  }
  slab_to_allocator_map_.clear();
}

std::string TieredCpuBufferMgr::dump() const {
  size_t allocator_num = 0;
  std::stringstream ss;
  ss << "TieredCpuBufferMgr:\n";
  for (auto& [allocator, allocator_limit] : allocators_) {
    ss << "  allocator[" << allocator_num++ << "]\n    limit = " << allocator_limit
       << "\n    used = " << allocator->bytesUsed() << "\n";
  }
  return ss.str();
}

}  // namespace Buffer_Namespace
