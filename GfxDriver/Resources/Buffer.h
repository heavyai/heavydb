/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include "GfxDriver/Resources/BufferCreateInfo.h"
#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Resource.h"
#include "GfxDriver/Types.h"

namespace gfx {

class Buffer : public Resource {
 public:
  using BufferState = BufferCreateInfo;

  explicit Buffer(const DeviceContext& device_ctx,
                  std::string_view resource_tracking_string,
                  const BufferCreateInfo& create_info);
  Buffer() = delete;
  ~Buffer() override;

  ResourceHandle getResourceHandle() const override { return resource_handle_; }
  BufferType getType() const { return state_.buffer_type; }
  BufferAccessType getAccessType() const { return state_.access_type; }
  BufferUsageBits getUsageBits() const { return state_.usage; }
  inline bool isMappable() const {
    return state_.access_type == BufferAccessType::kHostVisible ||
           state_.access_type == BufferAccessType::kHostVisibleCached;
  }

  uint64_t getNumBytes() const { return state_.size; }

  virtual void create(const void* data,
                      uint64_t num_bytes,
                      std::optional<LoggingCallback> oom_logging_cb = std::nullopt) = 0;
  virtual void rebuild(const void* data,
                       uint64_t num_bytes,
                       std::optional<LoggingCallback> oom_logging_cb = std::nullopt) = 0;

  virtual void updateSubData(const void* data,
                             uint64_t num_bytes,
                             uint64_t byte_offset) = 0;

  virtual void getData(void* data,
                       const uint64_t num_bytes,
                       const uint64_t byte_offset = 0ULL) = 0;

  // returns 0 if invalid
  virtual DeviceAddress getDeviceAddress() const = 0;

 protected:
  ResourceHandle resource_handle_;
  BufferState state_;

  void setSize(const uint64_t new_size) { state_.size = new_size; }
  void setUsage(const BufferUsageBits usage) { state_.usage = usage; }
};

};  // namespace gfx
