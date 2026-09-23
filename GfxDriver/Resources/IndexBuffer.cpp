/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/IndexBuffer.h"

namespace gfx {

IndexBuffer::IndexBuffer(std::string_view resource_tracking_string,
                         BufferAllocatorShPtr buffer_allocator,
                         const BufferCreateInfo& create_info,
                         std::optional<LoggingCallback> oom_logging_cb)
    : BufferWrapper(resource_tracking_string,
                    std::move(buffer_allocator),
                    create_info,
                    oom_logging_cb)
    , data_type_{create_info.index_buffer_data_type}
    , type_size_{data_type_ == IndexBufferDataType::kUnsigned16 ? 2u : 4u}
    , num_items_{static_cast<uint32_t>(create_info.size / type_size_)}
    , primitive_assembly_dependency_{std::make_unique<PrimitiveAssemblyDependency>()} {}

void IndexBuffer::create(const void* data,
                         uint64_t num_bytes,
                         std::optional<LoggingCallback> oom_logging_cb) {
  RUNTIME_EX_ASSERT(num_bytes % type_size_ == 0,
                    "Cannot allocate an index buffer of type " + to_string(data_type_) +
                        " with " + std::to_string(num_bytes) +
                        " bytes. The size of the buffer must be a multiple of " +
                        std::to_string(type_size_));

  primitive_assembly_dependency_->markPrimitiveAssembliesDirty();
  BufferWrapper::create(data, num_bytes, oom_logging_cb);
  num_items_ = num_bytes / type_size_;
}

void IndexBuffer::rebuild(const void* data,
                          uint64_t num_bytes,
                          std::optional<LoggingCallback> oom_logging_cb) {
  RUNTIME_EX_ASSERT(num_bytes % type_size_ == 0,
                    "Cannot rebuild an index buffer of type " + to_string(data_type_) +
                        " with " + std::to_string(num_bytes) +
                        " bytes. The size of the buffer must be a multiple of " +
                        std::to_string(type_size_));

  primitive_assembly_dependency_->markPrimitiveAssembliesDirty();
  BufferWrapper::rebuild(data, num_bytes, oom_logging_cb);
  num_items_ = num_bytes / type_size_;
}

}  // namespace gfx
