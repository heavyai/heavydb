/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/DriverInstance.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDebugUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanPlatform.h"

namespace gfx {

/**
 * Vulkan driver implementation
 * */
class VulkanDriver : public BaseDriver {
 public:
  explicit VulkanDriver(GfxUsage usage,
                        uint32_t command_timeout,
                        const WindowSystemCreateInfo* wsi_ci,
                        bool allow_raytracing_init);
  ~VulkanDriver() override;

  // from BaseDriver
  std::string_view getName() const override { return "Vulkan"; }
  DriverType getDriverType() const override { return DriverType::kVulkan; }

  uint32_t getNumGpus() const override;
  bool queryCapabilities(const DeviceCapabilityBits capabilities) const override;
  const DeviceLimits& getLimits() const override;

  VulkanDebugUtils& getDebugUtils() const;

 private:
  // From BaseDriver
  std::vector<heavyai::UUID> getUUIDs() const override;

  DeviceContextUqPtr createDeviceContextInternal(const heavyai::UUID& uuid,
                                                 const DeviceId gpu_id) override;

  bool destroyDeviceContextInternal(DeviceContextUqPtr device_ctx) override;

  WindowSystemIntegration* getWSI() const override;

  MemoryUsageInfo getPeakMemoryUsage() const override;

  std::unique_ptr<VulkanPlatform> platform_;
};

}  // namespace gfx
