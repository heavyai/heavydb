/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Commands/MemoryArena.h"

#include <cerrno>
#include <cstdlib>

#include "Logger/Logger.h"

namespace gfx {

// Alloc blocks aligned to a cache line to ensure initial sub-allocation is aligned
static constexpr size_t cache_line_size = 64;
std::byte* alloc_aligned(size_t size) {
  CHECK_EQ(size % cache_line_size, size_t(0))
      << "Block size must be a multiple of alignment";
  void* ptr = nullptr;
  int result = posix_memalign(&ptr, cache_line_size, size);
  CHECK_NE(result, ENOMEM) << "Failed to allocate aligned memory block of " << size
                           << " bytes";
  CHECK_EQ(result, 0) << "Unknown error allocating aligned memory block";

  return static_cast<std::byte*>(ptr);
}

void free_aligned(std::byte* ptr) {
  free(ptr);
}

MemoryArena::MemoryArena(uint32_t block_size)
    : block_size_{block_size}
    , current_position_{0}
    , max_position_{0}
    , current_block_{alloc_aligned(block_size)} {
  CHECK(current_block_);
}

MemoryArena::~MemoryArena() {
  if (current_block_) {
    free_aligned(current_block_);
  }
  for (auto* block : used_blocks_) {
    free_aligned(block);
  }
  used_blocks_.clear();
  for (auto* block : available_blocks_) {
    free_aligned(block);
  }
  available_blocks_.clear();
}

uint64_t MemoryArena::size() const {
  return getNumBlocks() * block_size_;
}

void* MemoryArena::alloc(uint32_t size, uint32_t alignment) {
  // current_block_ must always be a valid pointer
  CHECK(current_block_);
  // Adjust current position for required alignment
  CHECK_EQ(alignment & (alignment - 1), 0u) << "Alignment must be a power of 2";
  current_position_ = (current_position_ + (alignment - 1)) & ~(alignment - 1);

  // Check if alloc will fit in current block, otherwise get a new block
  if (current_position_ + size > block_size_) {
    // save the current block as used and get a new block
    used_blocks_.push_back(current_block_);
    if (available_blocks_.size() && size <= block_size_) {
      current_block_ = available_blocks_.back();
      available_blocks_.pop_back();
    } else {
      current_block_ = alloc_aligned(std::max(size, block_size_));
    }
    current_position_ = 0;
    max_position_ = 0;
  }
  void* rtn = current_block_ + current_position_;
  current_position_ += size;
  if (current_position_ > max_position_) {
    max_position_ = current_position_;
  }
  return rtn;
}

void MemoryArena::clear() {
  // Reset position in current_block_ and make all used blocks available
  // Note: over time block ordering will change as they are naturally shuffled
  current_position_ = 0;
  while (used_blocks_.size()) {
    available_blocks_.push_back(used_blocks_.back());
    used_blocks_.pop_back();
  }
}

uint32_t MemoryArena::getNumBlocks() const {
  return static_cast<uint32_t>(1 + used_blocks_.size() + available_blocks_.size());
}

uint32_t MemoryArena::getMaxPositionInTailBlock() const {
  return max_position_;
}

}  // namespace gfx
