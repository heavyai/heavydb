/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/BufferAllocator.h"

#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

class LocalBufferAllocator : public BufferAllocator {
 public:
  LocalBufferAllocator(const DeviceContext& device_ctx,
                       std::string_view resource_tracking_string,
                       ResourceManager& resource_mgr,
                       const BufferCreateInfo& create_info,
                       std::optional<LoggingCallback> oom_logging_cb)
      : device_ctx_{device_ctx}
      , resource_tracking_string_{resource_tracking_string}
      , resource_mgr_{resource_mgr}
      , create_info_{create_info}
      , oom_logging_cb_{oom_logging_cb} {}
  ~LocalBufferAllocator() override {
    if (buffer_) {
      resource_mgr_.destroyBaseBuffer(std::move(buffer_));
    }
  }

  BufferAllocationUqPtr alloc(const uint64_t num_bytes) override {
    if (!buffer_) {
      create_info_.size = num_bytes;
      buffer_ = resource_mgr_.createBaseBuffer(
          resource_tracking_string_, create_info_, oom_logging_cb_);
    }
    return std::make_unique<BufferAllocation>(*buffer_, 0, num_bytes, 0);
  }
  void free(BufferAllocationUqPtr allocation) override {
    if (buffer_) {
      resource_mgr_.destroyBaseBuffer(std::move(buffer_));
    }
  }

  void validateCreateInfo(const BufferCreateInfo& create_info) override {}

  const DeviceContext& getDeviceContext() const override { return device_ctx_; }

 private:
  const DeviceContext& device_ctx_;
  std::string_view resource_tracking_string_;
  ResourceManager& resource_mgr_;
  BufferCreateInfo create_info_;
  std::optional<LoggingCallback> oom_logging_cb_;
  resource_ptr<Buffer> buffer_;
};

}  // namespace gfx
