/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/VulkanMemoryMgr.h"
#include "GfxDriver/Resources/Buffer.h"

namespace gfx {

VkFlags buffer_access_type_to_memory_properties_bits(BufferAccessType access_type);

class VulkanBaseBuffer : public Buffer {
 public:
  explicit VulkanBaseBuffer(const DeviceContext& device_ctx,
                            std::string_view resource_tracking_string,
                            const BufferCreateInfo& create_info,
                            std::optional<LoggingCallback> oom_logging_cb = std::nullopt);
  ~VulkanBaseBuffer() override;

  void create(const void* data,
              uint64_t num_bytes,
              std::optional<LoggingCallback> oom_logging_cb) override;
  void rebuild(const void* data,
               uint64_t num_bytes,
               std::optional<LoggingCallback> oom_logging_cb) override;

  // Dynamic buffers only
  void updateSubData(const void* data, uint64_t num_bytes, uint64_t byte_offset) override;

  void getData(void* data, const uint64_t num_bytes, const uint64_t byte_offset) override;

  uint64_t getGpuAllocationSize() const override;
  DeviceAddress getDeviceAddress() const override;

  // temporary
  VulkanAllocation* getMemoryAllocation() const { return vulkan_allocation_.get(); }

 protected:
  void cleanupResourceBase() override;
  void makeEmpty() override;

 private:
  VulkanMemoryMgr::allocation_ptr vulkan_allocation_;
  VkMemoryRequirements vk_memory_requirements_;

  // Concrete implementation of the virtual `create()`, so we can safely call this from
  // the constructor (which is not true of virtual functions).
  void createInternal(const void* data,
                      uint64_t num_bytes,
                      int32_t import_allocation_fd,
                      std::optional<LoggingCallback> oom_logging_cb);

  friend class VulkanHostVisibleBufferWrapper;
  friend struct VulkanBaseBufferInternalsAccessor;
};

}  // namespace gfx
