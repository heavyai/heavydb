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

#include <filesystem>
#include "memkind.h"

#include "PMemAllocator.h"

extern std::string g_pmem_path;
extern size_t g_pmem_size;

PMemArena::PMemArena(size_t min_block_size, size_t size_limit)
    : size_limit_(size_limit), size_(0) {
  std::error_code ec;
  auto pmem_space_info = std::filesystem::space(g_pmem_path.c_str(), ec);
  CHECK(!ec) << "Failed to get pmem space info for path: " << g_pmem_path
             << "error code: " << ec << "\n";
  size_t capacity = pmem_space_info.capacity;
  CHECK_EQ(memkind_check_dax_path(g_pmem_path.c_str()), 0)
      << g_pmem_path << " is not recognized as a direct-access pmem path.";
  CHECK_GE(capacity, g_pmem_size)
      << g_pmem_path << " is not large enough for the requested PMem space";
  CHECK_EQ(memkind_create_pmem(g_pmem_path.c_str(), capacity, &pmem_kind_), 0)
      << "Failed to create PMEM memory.";
  LOG(INFO) << "Created Pmem direct-access allocator at " << g_pmem_path
            << " with capacity " << capacity << "\n";
}

PMemArena::~PMemArena() {
  for (auto [ptr, size] : allocations_) {
    memkind_free(pmem_kind_, ptr);
  }
}

void* PMemArena::allocate(const size_t num_bytes) {
  if (size_limit_ != 0 && size_ + num_bytes > size_limit_) {
    throw OutOfHostMemory(num_bytes);
  }
  auto ret = memkind_malloc(pmem_kind_, num_bytes);
  size_ += num_bytes;
  allocations_.push_back({ret, num_bytes});
  return ret;
}

void* PMemArena::allocateAndZero(const size_t num_bytes) {
  auto ret = allocate(num_bytes);
  std::memset(ret, 0, num_bytes);
  return ret;
}

size_t PMemArena::bytesUsed() const {
  return size_;
}
