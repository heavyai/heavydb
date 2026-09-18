/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Pipeline/BufferLayoutManager.h"
#include "GfxDriver/Resources/Buffer.h"
#include "GfxDriver/Resources/BufferAllocator.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Types.h"

namespace gfx {

class BufferWrapper {
 public:
  explicit BufferWrapper(std::string_view resource_tracking_string,
                         BufferAllocatorShPtr buffer_allocator,
                         const BufferCreateInfo& create_info,
                         std::optional<LoggingCallback> oom_logging_cb);
  BufferWrapper() = delete;
  virtual ~BufferWrapper();

  const DeviceContext& getDeviceContext() const;
  Buffer& getBuffer() const;
  BufferType getType() const;
  ResourceType getResourceType() const;
  ResourceHandle getResourceHandle() const;
  BufferAccessType getAccessType() const;
  BufferUsageBits getUsageBits() const;
  bool isMappable() const;
  const ResourceTrackingData& getTrackingData();
  const std::string& getTrackingDataNameOnly() const;

  uint64_t getNumBytes() const;
  uint64_t getAllocationBasePtr() const;
  uint64_t getAllocationOffsetBytes() const;

  virtual void rebuild(const void* data,
                       uint64_t num_bytes,
                       std::optional<LoggingCallback> oom_logging_cb = std::nullopt);
  void updateSubData(const void* data, uint64_t num_bytes, uint64_t byte_offset);
  void validateUsability(const char* filename, int lineno);
  bool isUsable() const;

  void getData(void* data, const uint64_t num_bytes) const;
  DeviceAddress getDeviceAddress() const;

  virtual void create(const void* data,
                      uint64_t num_bytes,
                      std::optional<LoggingCallback> oom_logging_cb = std::nullopt);
  virtual void markDirty() {}

  //
  // Layout
  //
  bool hasLayout() const;
  BufferLayoutManager* getLayoutManager() const;
  virtual void updateSubDataWithLayout(const void* data,
                                       const uint64_t num_bytes,
                                       const uint64_t offset_bytes,
                                       const BufferLayoutShPtr& layout);

 private:
  std::string resource_tracking_string_;
  BufferAllocatorShPtr buffer_allocator_;
  BufferAllocationUqPtr buffer_allocation_;
  BufferType buffer_type_;
  BufferUsageBits buffer_usage_;
  ResourceTrackingData buffer_tracking_data_;

  void allocate(const uint64_t num_bytes, std::optional<LoggingCallback> oom_logging_cb);
  void free();

  std::unique_ptr<BufferLayoutManager> layout_mgr_;

  friend class ResourceManager;
  friend class HostVisibleBufferWrapper;
  friend struct BufferWrapperInternalsAccessor;
};

}  // namespace gfx
