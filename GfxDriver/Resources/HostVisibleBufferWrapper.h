/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/noncopyable.hpp>

#include "GfxDriver/Resources/BufferWrapper.h"

namespace gfx {

struct HostVisibleBufferCreateInfo : public BufferCreateInfo {
  HostVisibleBufferCreateInfo(const BufferType buffer_type,
                              const uint64_t size,
                              const BufferUsageBits usage_bits = BufferUsageBits::kNone)
      : BufferCreateInfo{buffer_type, size, usage_bits, BufferAccessType::kHostVisible} {}
};

class HostVisibleBufferWrapper : boost::noncopyable {
 public:
  explicit HostVisibleBufferWrapper(BufferWrapperUqPtr source_buffer_wrapper);
  HostVisibleBufferWrapper() = delete;
  virtual ~HostVisibleBufferWrapper() = default;

  inline bool isMapped() const { return is_mapped_; }

  // Discarding the returned BufferWrapper would be a resource leak
  [[nodiscard]] BufferWrapperUqPtr&& releaseSourceBuffer();
  BufferWrapper& getSourceBufferWrapper() const;

  virtual void map(void** data) = 0;
  virtual void* map() = 0;
  virtual void unmap() = 0;

 protected:
  BufferWrapperUqPtr source_buffer_wrapper_;
  bool is_mapped_;

  Buffer& getSourceBuffer();

  friend class ResourceManager;
};

}  // namespace gfx
