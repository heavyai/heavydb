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

#include "DataMgr/DataMgr.h"
#include "Shared/checked_alloc.h"

template <class T>
class SysAllocator {
 public:
  using Self = SysAllocator;
  using value_type = T;

  constexpr SysAllocator() = default;

  constexpr SysAllocator(SysAllocator const&) = default;

  template <class U>
  constexpr SysAllocator(const SysAllocator<U>&) noexcept {}

  [[nodiscard]] T* allocate(size_t count) {
    return reinterpret_cast<T*>(checked_malloc(count));
  }

  void deallocate(T* p, size_t /* count */) { free(p); }

  friend bool operator==(Self const&, Self const&) noexcept { return true; }
  friend bool operator!=(Self const&, Self const&) noexcept { return false; }
};

class Arena {
 public:
  enum class MemoryType { DRAM, PMEM };
  virtual ~Arena() {}
  virtual void* allocate(size_t size) = 0;
  virtual void* allocateAndZero(const size_t size) = 0;
  virtual size_t bytesUsed() const = 0;
  virtual MemoryType getMemoryType() const = 0;
};

#ifdef HAVE_FOLLY

#include <folly/Memory.h>
#include <folly/memory/Arena.h>

using AllocatorType = char;

constexpr size_t kArenaBlockOverhead =
    folly::Arena<::SysAllocator<AllocatorType>>::kBlockOverhead;

/**
 * Arena allocator using checked_malloc with default allocation size 2GB. Note that the
 * allocator only frees memory on destruction.
 */
class DramArena : public Arena, public folly::Arena<::SysAllocator<AllocatorType>> {
 public:
  explicit DramArena(size_t min_block_size = static_cast<size_t>(1ULL << 32) +
                                             kBlockOverhead,
                     size_t size_limit = kNoSizeLimit,
                     size_t max_align = kDefaultMaxAlign)
      : folly::Arena<::SysAllocator<AllocatorType>>({},
                                                    min_block_size,
                                                    size_limit,
                                                    max_align) {}
  ~DramArena() override {}

  void* allocate(size_t size) override {
    return folly::Arena<::SysAllocator<AllocatorType>>::allocate(size);
  }

  void* allocateAndZero(const size_t size) override {
    auto ret = allocate(size);
    std::memset(ret, 0, size);
    return ret;
  }

  size_t bytesUsed() const override {
    return folly::Arena<::SysAllocator<AllocatorType>>::bytesUsed();
  }

  MemoryType getMemoryType() const override { return MemoryType::DRAM; }
};

template <>
struct folly::ArenaAllocatorTraits<::SysAllocator<AllocatorType>> {
  static size_t goodSize(const ::SysAllocator<AllocatorType>& /* alloc */, size_t size) {
    return folly::goodMallocSize(size);
  }
};

#else

constexpr size_t kArenaBlockOverhead = 0;

/**
 * A naive allocator which calls malloc and maintains a list of allocate pointers for
 * freeing. For development and testing only, where folly is not available. Not for
 * production use.
 */
class DramArena : public Arena {
 public:
  explicit DramArena(size_t min_block_size = 1ULL << 32, size_t size_limit = 0)
      : size_limit_(size_limit), size_(0) {}

  ~DramArena() override {
    for (auto [ptr, size] : allocations_) {
      allocator_.deallocate(ptr, 0);
      size_ -= size;
    }
  }

  void* allocate(size_t num_bytes) override {
    if (size_limit_ != 0 && size_ + num_bytes > size_limit_) {
      throw OutOfHostMemory(num_bytes);
    }
    auto ret = allocator_.allocate(num_bytes);
    size_ += num_bytes;
    allocations_.push_back({ret, num_bytes});
    return ret;
  }

  void* allocateAndZero(const size_t size) override {
    auto ret = allocate(size);
    std::memset(ret, 0, size);
    return ret;
  }

  size_t bytesUsed() const override { return size_; }

  MemoryType getMemoryType() const override { return MemoryType::DRAM; }

 private:
  size_t size_limit_;
  size_t size_;
  SysAllocator<void> allocator_;
  std::vector<std::pair<void*, size_t>> allocations_;
};

#endif
