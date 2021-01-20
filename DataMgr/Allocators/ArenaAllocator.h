/*
 * Copyright 2020 OmniSci, Inc.
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
    return static_cast<T*>(checked_malloc(count));
  }

  void deallocate(void* p, size_t /* count */) { free(p); }

  friend bool operator==(Self const&, Self const&) noexcept { return true; }
  friend bool operator!=(Self const&, Self const&) noexcept { return false; }
};

#ifdef HAVE_FOLLY

#include <folly/Memory.h>
#include <folly/memory/Arena.h>

/**
 * Folly arena has some weird code which looks broken and most
 * probably works on some copilers only. On Linux and Windows we
 * use different folly versions with different issues and have
 * to use different workarounds.
 *
 * On Linux we use folly version which passes Arena::Block* to
 * std::allocator_traits<A>::deallocate instead of A::pointer*.
 * This can cause a build failure but works fine when T = void.
 *
 * On Windows we have another version which uses allocator traits
 * rebound to char type and casts Arena::Block* to char*.
 * Unfortunately, it passes the original allocator to this rebound
 * traits and MSVC doesn't allow that. It works if the original
 * traits and the rebound one have the same type and therefore we
 * use char as a base type.
 */
#ifdef _WIN32
using AllocT = char;
#else
using AllocT = void;
#endif  // _WIN32

constexpr size_t kArenaBlockOverhead =
    folly::Arena<::SysAllocator<AllocT>>::kBlockOverhead;

/**
 * Arena allocator using checked_malloc with default allocation size 2GB. Note that the
 * allocator only frees memory on destruction.
 */
class Arena : public folly::Arena<::SysAllocator<AllocT>> {
 public:
  explicit Arena(size_t min_block_size = static_cast<size_t>(1ULL << 32) + kBlockOverhead,
                 size_t size_limit = kNoSizeLimit,
                 size_t max_align = kDefaultMaxAlign)
      : folly::Arena<SysAllocator<AllocT>>({}, min_block_size, size_limit, max_align) {}

  void* allocateAndZero(const size_t size) {
    auto ret = allocate(size);
    std::memset(ret, 0, size);
    return ret;
  }
};

template <>
struct folly::ArenaAllocatorTraits<::SysAllocator<AllocT>> {
  static size_t goodSize(const ::SysAllocator<AllocT>& /* alloc */, size_t size) {
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
class Arena {
 public:
  explicit Arena(size_t min_block_size = 1UL << 32, size_t size_limit = 0) {}

  ~Arena() {
    for (auto ptr : allocations_) {
      allocator_.deallocate(ptr, 0);
    }
  }

  void* allocate(size_t num_bytes) {
    auto ret = allocator_.allocate(num_bytes);
    allocations_.push_back(ret);
    return ret;
  }

  void* allocateAndZero(const size_t size) {
    auto ret = allocate(size);
    std::memset(ret, 0, size);
    return ret;
  }

 private:
  SysAllocator<void> allocator_;
  std::vector<void*> allocations_;
};

#endif
