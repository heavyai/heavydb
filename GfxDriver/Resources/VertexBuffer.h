/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/BufferWrapper.h"

#include <memory>

#include "GfxDriver/Pipeline/PrimitiveAssemblyDependency.h"
#include "GfxDriver/Resources/BufferAllocator.h"

namespace gfx {

class VertexBuffer : public BufferWrapper {
 public:
  explicit VertexBuffer(std::string_view resource_tracking_string,
                        BufferAllocatorShPtr buffer_allocator,
                        const BufferCreateInfo& create_info,
                        std::optional<LoggingCallback> oom_logging_cb);

  ~VertexBuffer() override = default;

  void updateSubDataWithLayout(const void* data,
                               const uint64_t num_bytes,
                               const uint64_t offset_bytes,
                               const BufferLayoutShPtr& layout) override;

  PrimitiveAssemblyDependency* getPrimitiveAssemblyDependency() const {
    return primitive_assembly_dependency_.get();
  }

 private:
  std::unique_ptr<PrimitiveAssemblyDependency> primitive_assembly_dependency_;

  void markDirty() override {
    primitive_assembly_dependency_->markPrimitiveAssembliesDirty();
  }
};

}  // namespace gfx
