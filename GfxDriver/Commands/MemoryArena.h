/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

/**
 * MemoryArena
 *
 * Lightweight implementation of a block based memory arena for use with many small
 * allocations.
 *
 * Supports allocations larger than the block size by allocating an extra
 * large block for the one allocation, but is not intended for use with large allocations.
 *
 * Supports recycling of blocks - blocks are not destroyed but moved to an
 * available_blocks vector. Does not track individual block sizes, extra large blocks
 * are not fully reusuable to keep sub-allocations efficient and simple, avoiding a
 * search through blocks.
 *
 * Free backing memory by destroying the MemoryArena.
 * */
class MemoryArena {
 public:
  explicit MemoryArena(uint32_t block_size = 8192);
  ~MemoryArena();

  void* alloc(uint32_t size, uint32_t alignment);

  // Make all blocks available and set the current position cursor to 0
  void clear();

  // High water mark stats
  uint32_t getNumBlocks() const;
  uint32_t getMaxPositionInTailBlock() const;

  // Get total allocated memory for all blocks
  uint64_t size() const;

 private:
  uint32_t block_size_;
  uint32_t current_position_;  // Offset into current_block_
  uint32_t max_position_;      // High water mark for position
  std::byte* current_block_;   // Current memory block for suballocation. Must not be null
  std::vector<std::byte*> used_blocks_;       // Fully occupied blocks
  std::vector<std::byte*> available_blocks_;  // Free blocks
};

}  // namespace gfx
