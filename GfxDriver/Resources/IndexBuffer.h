/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/BufferWrapper.h"

#include <memory>

#include "GfxDriver/Pipeline/PrimitiveAssemblyDependency.h"

namespace gfx {

class IndexBuffer : public BufferWrapper {
 public:
  explicit IndexBuffer(std::string_view resource_tracking_string,
                       BufferAllocatorShPtr buffer_allocator,
                       const BufferCreateInfo& create_info,
                       std::optional<LoggingCallback> oom_logging_cb);

  ~IndexBuffer() override = default;

  uint32_t numItems() const { return num_items_; }

  IndexBufferDataType getIndexDataType() const { return data_type_; }
  uint32_t getIndexDataTypeByteSize() const { return type_size_; }

  void create(const void* data,
              uint64_t num_bytes,
              std::optional<LoggingCallback> oom_logging_cb = std::nullopt) override;
  void rebuild(const void* data,
               uint64_t num_bytes,
               std::optional<LoggingCallback> oom_logging_cb = std::nullopt) override;

  PrimitiveAssemblyDependency* getPrimitiveAssemblyDependency() const {
    return primitive_assembly_dependency_.get();
  }

 private:
  IndexBufferDataType data_type_;
  uint32_t type_size_;
  uint32_t num_items_;
  std::unique_ptr<PrimitiveAssemblyDependency> primitive_assembly_dependency_;
};

inline IndexBuffer* index_buffer_cast(const BufferWrapperUqPtr& ptr) {
  CHECK_EQ(ptr->getType(), BufferType::kIndexBuffer);
  return static_cast<IndexBuffer*>(ptr.get());
}

}  // namespace gfx
