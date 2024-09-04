/*
 * Copyright 2024 HEAVY.AI, Inc.
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
 * @file    CpuMgrArenaAllocator.h
 * @brief   Allocate CPU memory using CpuBuffers via DataMgr.
 */

#pragma once

#include "DataMgr/Allocators/ArenaAllocator.h"

namespace Data_Namespace {
class AbstractBuffer;
class DataMgr;
}  // namespace Data_Namespace

class CpuMgrBaseAllocator {
 public:
  CpuMgrBaseAllocator();

  ~CpuMgrBaseAllocator();

  CpuMgrBaseAllocator(const CpuMgrBaseAllocator& other) noexcept {
    data_mgr_ = other.data_mgr_;
    allocated_buffers_ = other.allocated_buffers_;
  }

  CpuMgrBaseAllocator(const CpuMgrBaseAllocator&& other) noexcept {
    data_mgr_ = std::move(other.data_mgr_);
    allocated_buffers_ = std::move(other.allocated_buffers_);
  }

  void* allocate(const size_t num_bytes);

  void deallocate(void* p);

  void deallocateAll();

  bool operator==(const CpuMgrBaseAllocator& other) noexcept {
    return other.data_mgr_ == data_mgr_ && other.allocated_buffers_ == allocated_buffers_;
  }

 private:
  std::list<Data_Namespace::AbstractBuffer*> allocated_buffers_;
  Data_Namespace::DataMgr* data_mgr_;
};

class CpuMgrArenaAllocator : public Arena {
 public:
  CpuMgrArenaAllocator();

  void* allocate(size_t num_bytes) override;

  void* allocateAndZero(const size_t num_bytes) override;

  size_t bytesUsed() const override;

  size_t totalBytes() const override;

  MemoryType getMemoryType() const override;

 private:
  CpuMgrBaseAllocator base_allocator_;
  size_t size_;
};

// TODO: this definition is moved here to support logging. It is likely not
// necessary anymore.
inline std::atomic<size_t> g_import_total_mem{0};
extern bool g_disable_cpu_mem_pool_import_buffers;

template <typename T, bool track_allocations = true>
class CpuMgrStdAllocator : std::allocator<T> {
 public:
  using value_type = T;

  // The following traits inform standard library implementations to propagate
  // allocators with whatever object or container they are used with. This is
  // important to support relevant operations on those objects or containers
  // without issue.
  //
  // See: https://en.cppreference.com/w/cpp/named_req/Allocator
  //      https://en.cppreference.com/w/cpp/named_req/AllocatorAwareContainer
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  template <typename U>
  struct rebind {
    using other = CpuMgrStdAllocator<U>;
  };

  CpuMgrStdAllocator()
      : base_allocator_(g_disable_cpu_mem_pool_import_buffers
                            ? nullptr
                            : std::make_unique<CpuMgrBaseAllocator>()) {}

  [[nodiscard]] T* allocate(std::size_t n) {
    T* p = nullptr;
    if (g_disable_cpu_mem_pool_import_buffers) {
      p = std::allocator<T>::allocate(sizeof(T) * n);
    } else {
      p = static_cast<T*>(base_allocator_->allocate(sizeof(T) * n));
    }
    if constexpr (track_allocations) {  // TODO: remove this code path when
                                        // `g_import_total_mem` is removed
      g_import_total_mem += sizeof(T) * n;
    }
    return p;
  }

  void deallocate(T* p, std::size_t n) noexcept {
    try {
      if (g_disable_cpu_mem_pool_import_buffers) {
        std::allocator<T>::deallocate(p, n);
      } else {
        base_allocator_->deallocate(static_cast<void*>(p));
      }
    } catch (std::exception& e) {
      CHECK(false) << "Encountered exception while freeing DataMgr cpu buffer: "
                   << e.what();
    }
    if constexpr (track_allocations) {  // TODO: remove this code path when
                                        // `g_import_total_mem` is removed
      g_import_total_mem -= sizeof(T) * n;
    }
  }

 private:
  std::unique_ptr<CpuMgrBaseAllocator> base_allocator_;
};

template <typename T, typename U>
bool operator==(const CpuMgrStdAllocator<T>& lhs, const CpuMgrStdAllocator<U>& rhs) {
  // If custom allocator is disabled, always compare true
  return g_disable_cpu_mem_pool_import_buffers ||
         *lhs.base_allocator_ == *rhs.base_allocator_;
}

template <typename T, typename U>
bool operator!=(const CpuMgrStdAllocator<T>& lhs, const CpuMgrStdAllocator<U>& rhs) {
  return !(lhs == rhs);
}

// Some standard library definitions for convenience
namespace managed_memory {
template <typename T>
using vector = std::vector<T, CpuMgrStdAllocator<T>>;
}
