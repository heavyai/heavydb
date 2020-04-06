/*
 * Copyright 2019 OmniSci, Inc.
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
#include "Shared/Logger.h"

class GpuSharedMemoryContext {
 public:
  GpuSharedMemoryContext() : shared_memory_size_(0) {}
  GpuSharedMemoryContext(const size_t shared_mem_size)
      : shared_memory_size_(shared_mem_size) {
    CHECK(shared_mem_size >= 0);
  }

  bool isSharedMemoryUsed() const { return shared_memory_size_ > 0; }
  size_t getSharedMemorySize() const { return shared_memory_size_; }

 private:
  size_t shared_memory_size_;
};
