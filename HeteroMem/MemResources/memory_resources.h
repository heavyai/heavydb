// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2020 Intel Corporation. */

#pragma once

#include <exception>
#include <memory>
#include <memory_resource>
#include <string>

#include "memkind.h"
#include "pmem_allocator.h"

namespace libmemkind {
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
};
}  // namespace pmem

namespace hbm {
// TODO: Implement memory resource for HBM
}  // namespace hbm
}  // namespace libmemkind