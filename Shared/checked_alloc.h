/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef CHECKED_ALLOC_H
#define CHECKED_ALLOC_H

#include <cstdlib>
#include <string>
#include <stdexcept>
#include "../Shared/types.h"

class OutOfHostMemory : public std::runtime_error {
 public:
  OutOfHostMemory(const size_t size)
      : std::runtime_error("Failed to allocate " + std::to_string(size) + " bytes of memory") {
    OOM_TRACE_DUMP;
  }
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
