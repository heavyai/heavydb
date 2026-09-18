/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "GfxDriver/Resources/Buffer.h"
#include "GfxDriver/Resources/BufferCreateInfo.h"

namespace gfx {

struct BufferAllocation {
  explicit BufferAllocation(Buffer& buffer,
                            const uint64_t base_ptr,
                            const uint64_t num_bytes,
                            const uint64_t offset_bytes)
      : buffer{buffer}
      , base_ptr{base_ptr}
      , num_bytes{num_bytes}
      , offset_bytes{offset_bytes} {}
  BufferAllocation() = delete;
  ~BufferAllocation() = default;

  Buffer& buffer;
  const uint64_t base_ptr;
  const uint64_t num_bytes;
  const uint64_t offset_bytes;
};

using BufferAllocationUqPtr = std::unique_ptr<BufferAllocation>;

class BufferAllocator {
 public:
  BufferAllocator() = default;
  virtual ~BufferAllocator() = default;

  virtual const DeviceContext& getDeviceContext() const = 0;
  virtual BufferAllocationUqPtr alloc(const uint64_t num_bytes) = 0;
  virtual void free(BufferAllocationUqPtr allocation) = 0;

  virtual void validateCreateInfo(const BufferCreateInfo& create_info) = 0;
};

using BufferAllocatorShPtr = std::shared_ptr<BufferAllocator>;

}  // namespace gfx
