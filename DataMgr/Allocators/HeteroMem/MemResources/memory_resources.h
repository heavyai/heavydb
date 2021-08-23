// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2020 Intel Corporation. */

#pragma once

#include <exception>
#include <memory>
#include <memory_resource>
#include <string>

#include "memkind.h"
#include "memkind_allocator.h"
#include "pmem_allocator.h"

namespace libmemkind {
namespace static_kind {
class memory_resource : public std::pmr::memory_resource {
 public:
  explicit memory_resource(libmemkind::kinds kind) {
    switch (kind) {
      case libmemkind::kinds::DEFAULT:
        _kind = MEMKIND_DEFAULT;
        break;
      case libmemkind::kinds::HUGETLB:
        _kind = MEMKIND_HUGETLB;
        break;
      case libmemkind::kinds::INTERLEAVE:
        _kind = MEMKIND_INTERLEAVE;
        break;
      case libmemkind::kinds::HBW:
        _kind = MEMKIND_HBW;
        break;
      case libmemkind::kinds::HBW_ALL:
        _kind = MEMKIND_HBW_ALL;
        break;
      case libmemkind::kinds::HBW_HUGETLB:
        _kind = MEMKIND_HBW_HUGETLB;
        break;
      case libmemkind::kinds::HBW_ALL_HUGETLB:
        _kind = MEMKIND_HBW_ALL_HUGETLB;
        break;
      case libmemkind::kinds::HBW_PREFERRED:
        _kind = MEMKIND_HBW_PREFERRED;
        break;
      case libmemkind::kinds::HBW_PREFERRED_HUGETLB:
        _kind = MEMKIND_HBW_PREFERRED_HUGETLB;
        break;
      case libmemkind::kinds::HBW_INTERLEAVE:
        _kind = MEMKIND_HBW_INTERLEAVE;
        break;
      case libmemkind::kinds::REGULAR:
        _kind = MEMKIND_REGULAR;
        break;
      case libmemkind::kinds::DAX_KMEM:
        _kind = MEMKIND_DAX_KMEM;
        break;
      case libmemkind::kinds::DAX_KMEM_ALL:
        _kind = MEMKIND_DAX_KMEM_ALL;
        break;
      case libmemkind::kinds::DAX_KMEM_PREFERRED:
        _kind = MEMKIND_DAX_KMEM_PREFERRED;
        break;
      case libmemkind::kinds::DAX_KMEM_INTERLEAVE:
        _kind = MEMKIND_DAX_KMEM_INTERLEAVE;
        break;
      case libmemkind::kinds::HIGHEST_CAPACITY:
        _kind = MEMKIND_HIGHEST_CAPACITY;
        break;
      case libmemkind::kinds::HIGHEST_CAPACITY_PREFERRED:
        _kind = MEMKIND_HIGHEST_CAPACITY_PREFERRED;
        break;
      case libmemkind::kinds::HIGHEST_CAPACITY_LOCAL:
        _kind = MEMKIND_HIGHEST_CAPACITY_LOCAL;
        break;
      case libmemkind::kinds::HIGHEST_CAPACITY_LOCAL_PREFERRED:
        _kind = MEMKIND_HIGHEST_CAPACITY_LOCAL_PREFERRED;
        break;
      case libmemkind::kinds::LOWEST_LATENCY_LOCAL:
        _kind = MEMKIND_LOWEST_LATENCY_LOCAL;
        break;
      case libmemkind::kinds::LOWEST_LATENCY_LOCAL_PREFERRED:
        _kind = MEMKIND_LOWEST_LATENCY_LOCAL_PREFERRED;
        break;
      case libmemkind::kinds::HIGHEST_BANDWIDTH_LOCAL:
        _kind = MEMKIND_HIGHEST_BANDWIDTH_LOCAL;
        break;
      case libmemkind::kinds::HIGHEST_BANDWIDTH_LOCAL_PREFERRED:
        _kind = MEMKIND_HIGHEST_BANDWIDTH_LOCAL_PREFERRED;
        break;
      default:
        throw std::runtime_error("Unknown libmemkind::kinds");
        break;
    }
  }

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    void* res = 0;
#if 0
                //TODO: For some reasons this version throws bad_alloc. Need to investigate
                if(memkind_posix_memalign(_kind, &res, alignment, bytes) != MEMKIND_SUCCESS) {
                    throw std::bad_alloc();
                }
#else
    res = memkind_malloc(_kind, bytes);
    if (!res) {
      throw std::bad_alloc();
    }
#endif

    return res;
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    memkind_free(_kind, p);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    const memory_resource* other_ptr = dynamic_cast<const memory_resource*>(&other);
    return (other_ptr != nullptr && _kind == other_ptr->_kind);
  }

  memkind_t _kind;
};  // class memory_resource
}  // namespace static_kind
namespace pmem {
class memory_resource : public std::pmr::memory_resource {
  using kind_wrapper_type = internal::kind_wrapper_t;
  std::shared_ptr<kind_wrapper_type> kind_wrapper_ptr;

 public:
  explicit memory_resource(const char* dir, size_t max_size)
      : kind_wrapper_ptr(std::make_shared<kind_wrapper_type>(dir, max_size)) {}

  explicit memory_resource(const std::string& dir, size_t max_size)
      : memory_resource(dir.c_str(), max_size) {}

  explicit memory_resource(const char* dir,
                           size_t max_size,
                           libmemkind::allocation_policy alloc_policy)
      : kind_wrapper_ptr(
            std::make_shared<kind_wrapper_type>(dir, max_size, alloc_policy)) {}

  explicit memory_resource(const std::string& dir,
                           size_t max_size,
                           libmemkind::allocation_policy alloc_policy)
      : memory_resource(dir.c_str(), max_size, alloc_policy) {}

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    void* res = 0;
#if 0
                //TODO: For some reasons this version throws bad_alloc. Need to investigate
                if(memkind_posix_memalign(kind_wrapper_ptr->get(), &res, alignment, bytes) != MEMKIND_SUCCESS) {
                    throw std::bad_alloc();
                }
#else
    res = memkind_malloc(kind_wrapper_ptr->get(), bytes);
    if (!res) {
      throw std::bad_alloc();
    }
#endif

    return res;
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    memkind_free(kind_wrapper_ptr->get(), p);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    const memory_resource* other_ptr = dynamic_cast<const memory_resource*>(&other);
    return (other_ptr != nullptr &&
            kind_wrapper_ptr->get() == other_ptr->kind_wrapper_ptr->get());
  }
};  // class memory_resource
}  // namespace pmem
}  // namespace libmemkind
