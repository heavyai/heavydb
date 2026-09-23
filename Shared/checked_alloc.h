/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHECKED_ALLOC_H
#define CHECKED_ALLOC_H

#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED 1

#include "Shared/boost_stacktrace.hpp"

#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <string>
#include "../Logger/Logger.h"
#include "../Shared/types.h"

class OutOfHostMemory : public std::bad_alloc {
 public:
  OutOfHostMemory(const size_t size)
      : what_str_("Not enough CPU memory available to allocate " + std::to_string(size)) {
    VLOG(1) << "Failed to allocate " << size << " bytes\n"
            << boost::stacktrace::stacktrace();
  }

  const char* what() const noexcept final { return what_str_.c_str(); }

 private:
  const std::string what_str_;
};

inline void* checked_malloc(const size_t size) {
  auto ptr = malloc(size);
  if (!ptr) {
    throw OutOfHostMemory(size);
  }
  return ptr;
}

inline void* checked_calloc(const size_t nmemb, const size_t size) {
  auto ptr = calloc(nmemb, size);
  if (!ptr) {
    throw OutOfHostMemory(nmemb * size);
  }
  return ptr;
}

struct CheckedAllocDeleter {
  void operator()(void* p) { free(p); }
};

#endif  // CHECKED_ALLOC_H
