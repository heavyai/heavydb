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

#include "ArenaAllocator.h"

// An arena allocator that uses direct-access PMem for allocations.
class PMemArena : public Arena {
 public:
  explicit PMemArena(size_t min_block_size = 1ULL << 32, size_t size_limit = 0);

  ~PMemArena() override;

  void* allocate(const size_t num_bytes) override;

  void* allocateAndZero(const size_t num_bytes) override;

  size_t bytesUsed() const override;

  size_t totalBytes() const override;

  MemoryType getMemoryType() const override { return MemoryType::PMEM; }

 private:
  size_t size_limit_;
  size_t size_;
  std::vector<std::pair<void*, size_t>> allocations_;
  struct memkind* pmem_kind_ = NULL;
};
