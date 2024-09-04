/*
 * Copyright 2023 HEAVY.AI, Inc.
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
 * @file    FastAllocator.h
 * @brief   Quickly allocate many memory pieces by reserving them ahead of time.
 *          Calls to allocate() are thread-safe.
 */

#pragma once

#include "Logger/Logger.h"
#include "Shared/SimpleAllocator.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <sstream>

namespace heavyai {
namespace allocator {
namespace detail {

inline std::runtime_error outOfMemoryError(size_t n, size_t remaining, size_t capacity) {
  std::ostringstream oss;
  oss << "allocate(" << n << ") called but only " << remaining << " out of " << capacity
      << " available.";
  return std::runtime_error(oss.str());
}

// FastAllocator accepts a pre-allocated buffer of given capacity and
// allocates sequential chunks, tracking the size_ starting at size_=0.
// If size_ exceeds capacity_ then throw an exception.
// There is no deallocate() function, nor is there a destructor.
template <typename T>
class FastAllocator : public SimpleAllocator {
 public:
  FastAllocator() : buffer_(nullptr), capacity_(0), size_(0) {}
  FastAllocator(T* buffer, size_t capacity)
      : buffer_(buffer), capacity_(capacity), size_(0) {}
  FastAllocator(FastAllocator const&) = delete;
  FastAllocator(FastAllocator&&) = delete;
  FastAllocator& operator=(FastAllocator const&) = delete;
  FastAllocator& operator=(FastAllocator&& rhs) {
    buffer_ = rhs.buffer_;
    capacity_ = rhs.capacity_;
    size_ = rhs.size_;
    rhs.reset();
    return *this;
  }

  // Allocate n>0 elements of type T. Caller responsible for proper data alignment.
  T* allocate(size_t const n) {
    CHECK(n);
    std::lock_guard<std::mutex> lock_guard(mutex_);
    if (n <= available()) {
      T* const ptr = buffer_ + size_;
      size_ += n;
      return ptr;
    }
    throw outOfMemoryError(n, available(), capacity_);
  }

  size_t available() const { return capacity_ - size_; }  // number of available elements
  size_t capacity() const { return capacity_; }           // number of reserved elements

 protected:
  void reset() {
    buffer_ = nullptr;
    capacity_ = 0u;
    size_ = 0u;
  }

  T* buffer_;        // Pointer to reserved buffer.
  size_t capacity_;  // Number of elements of type T reserved.
  size_t size_;      // Number of elements of type T allocated.
  mutable std::mutex mutex_;
};

}  // namespace detail
}  // namespace allocator
}  // namespace heavyai

using heavyai::allocator::detail::FastAllocator;
