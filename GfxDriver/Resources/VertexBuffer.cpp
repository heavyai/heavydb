/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/VertexBuffer.h"

namespace gfx {

VertexBuffer::VertexBuffer(std::string_view resource_tracking_string,
                           BufferAllocatorShPtr buffer_allocator,
                           const BufferCreateInfo& create_info,
                           std::optional<LoggingCallback> oom_logging_cb)
    : BufferWrapper(resource_tracking_string,
                    std::move(buffer_allocator),
                    create_info,
                    oom_logging_cb)
    , primitive_assembly_dependency_{std::make_unique<PrimitiveAssemblyDependency>()} {}

void VertexBuffer::updateSubDataWithLayout(const void* data,
                                           const uint64_t num_bytes,
                                           const uint64_t offset_bytes,
                                           const BufferLayoutShPtr& layout) {
  BufferWrapper::updateSubDataWithLayout(data, num_bytes, offset_bytes, layout);
  primitive_assembly_dependency_->markPrimitiveAssembliesDirty();
}

}  // namespace gfx
