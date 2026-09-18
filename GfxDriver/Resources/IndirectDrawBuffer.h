/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/RenderError.h"
#include "GfxDriver/Resources/BufferAllocator.h"
#include "GfxDriver/Resources/BufferWrapper.h"
#include "Shared/ShapeDrawData.h"

namespace gfx {

class BaseIndirectDrawBuffer : public BufferWrapper {
 public:
  explicit BaseIndirectDrawBuffer(std::string_view resource_tracking_string,
                                  BufferAllocatorShPtr buffer_allocator,
                                  const BufferCreateInfo& create_info,
                                  std::optional<LoggingCallback> oom_logging_cb)
      : BufferWrapper(resource_tracking_string,
                      std::move(buffer_allocator),
                      create_info,
                      oom_logging_cb) {}

  ~BaseIndirectDrawBuffer() override {}

  uint32_t numItems() const { return getNumBytes() / getStructByteSize(); }
  virtual uint32_t getStructByteSize() const = 0;
};

template <typename T>
class BaseIndirectDrawTemplateBuffer : public BaseIndirectDrawBuffer {
 public:
  explicit BaseIndirectDrawTemplateBuffer(std::string_view resource_tracking_string,
                                          BufferAllocatorShPtr buffer_allocator,
                                          const BufferCreateInfo& create_info,
                                          std::optional<LoggingCallback> oom_logging_cb)
      : BaseIndirectDrawBuffer(resource_tracking_string,
                               std::move(buffer_allocator),
                               create_info,
                               oom_logging_cb) {}

  ~BaseIndirectDrawTemplateBuffer() override {}

  uint32_t getStructByteSize() const override { return static_cast<uint32_t>(sizeof(T)); }

  void create(const std::vector<T>& indirect_draw_data,
              std::optional<LoggingCallback> oom_logging_cb = std::nullopt) {
    BufferWrapper::create(indirect_draw_data.data(),
                          static_cast<uint64_t>(indirect_draw_data.size() * sizeof(T)),
                          oom_logging_cb);
  }

  void create(const void* data,
              uint64_t num_bytes,
              std::optional<LoggingCallback> oom_logging_cb = std::nullopt) override {
    RUNTIME_EX_ASSERT(num_bytes % sizeof(T) == 0,
                      "Cannot allocate an indirect draw buffer of type " +
                          to_string(getResourceType()) + " with " +
                          std::to_string(num_bytes) +
                          " bytes. The size of the buffer must be a multiple of " +
                          std::to_string(sizeof(T)));

    BufferWrapper::create(data, num_bytes, oom_logging_cb);
  }
};

class IndirectDrawVertexBuffer
    : public BaseIndirectDrawTemplateBuffer<IndirectDrawVertexData> {
 public:
  explicit IndirectDrawVertexBuffer(std::string_view resource_tracking_string,
                                    BufferAllocatorShPtr buffer_allocator,
                                    const BufferCreateInfo& create_info,
                                    std::optional<LoggingCallback> oom_logging_cb);

  ~IndirectDrawVertexBuffer() override {}
};

class IndirectDrawIndexBuffer
    : public BaseIndirectDrawTemplateBuffer<IndirectDrawIndexData> {
 public:
  explicit IndirectDrawIndexBuffer(std::string_view resource_tracking_string,
                                   BufferAllocatorShPtr buffer_allocator,
                                   const BufferCreateInfo& create_info,
                                   std::optional<LoggingCallback> oom_logging_cb);

  ~IndirectDrawIndexBuffer() override {}
};

}  // namespace gfx
