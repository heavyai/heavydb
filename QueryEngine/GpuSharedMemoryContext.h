/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "Logger/Logger.h"

#include <ostream>

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

inline std::ostream& operator<<(std::ostream& os, GpuSharedMemoryContext const ctx) {
  return os << "shared_memory_size=" << ctx.getSharedMemorySize();
}
