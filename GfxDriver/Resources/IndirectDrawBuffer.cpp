/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/IndirectDrawBuffer.h"

namespace gfx {

IndirectDrawVertexBuffer::IndirectDrawVertexBuffer(
    std::string_view resource_tracking_string,
    BufferAllocatorShPtr buffer_allocator,
    const BufferCreateInfo& create_info,
    std::optional<LoggingCallback> oom_logging_cb)
    : BaseIndirectDrawTemplateBuffer<IndirectDrawVertexData>(resource_tracking_string,
                                                             std::move(buffer_allocator),
                                                             create_info,
                                                             oom_logging_cb) {}

IndirectDrawIndexBuffer::IndirectDrawIndexBuffer(
    std::string_view resource_tracking_string,
    BufferAllocatorShPtr buffer_allocator,
    const BufferCreateInfo& create_info,
    std::optional<LoggingCallback> oom_logging_cb)
    : BaseIndirectDrawTemplateBuffer<IndirectDrawIndexData>(resource_tracking_string,
                                                            std::move(buffer_allocator),
                                                            create_info,
                                                            oom_logging_cb) {}

}  // namespace gfx
