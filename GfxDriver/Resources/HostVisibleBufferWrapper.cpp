/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/HostVisibleBufferWrapper.h"

namespace gfx {

HostVisibleBufferWrapper::HostVisibleBufferWrapper(
    BufferWrapperUqPtr source_buffer_wrapper)
    : source_buffer_wrapper_(std::move(source_buffer_wrapper)), is_mapped_(false) {
  CHECK_EQ(source_buffer_wrapper_->getAccessType(), BufferAccessType::kHostVisible);
}

BufferWrapperUqPtr&& HostVisibleBufferWrapper::releaseSourceBuffer() {
  CHECK(!is_mapped_) << "can only release an unmapped buffer";

  // NOTE: source_buffer_wrapper_ should be nulled by stl after the move.
  return std::move(source_buffer_wrapper_);
}

BufferWrapper& HostVisibleBufferWrapper::getSourceBufferWrapper() const {
  CHECK(source_buffer_wrapper_) << "Source buffer is invalid. It was already released.";
  return *source_buffer_wrapper_;
}

Buffer& HostVisibleBufferWrapper::getSourceBuffer() {
  CHECK(source_buffer_wrapper_)
      << "Source buffer wrapper is invalid. It was already released";
  CHECK(source_buffer_wrapper_->buffer_allocation_)
      << "Source buffer is invalid. It was already destroyed";
  return source_buffer_wrapper_->buffer_allocation_->buffer;
}

}  // namespace gfx
