/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanDriver.h"

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"

namespace gfx {

VulkanDriver::VulkanDriver(GfxUsage usage,
                           uint32_t command_timeout,
                           const WindowSystemCreateInfo* wsi_ci,
                           bool allow_raytracing_init)
    : platform_(std::make_unique<VulkanPlatform>(*this,
                                                 usage,
                                                 command_timeout,
                                                 wsi_ci,
                                                 allow_raytracing_init)) {
  CHECK(platform_);
}

VulkanDriver::~VulkanDriver() {
  platform_ = nullptr;
}

uint32_t VulkanDriver::getNumGpus() const {
  return platform_->getNumGpus();
}

std::vector<heavyai::UUID> VulkanDriver::getUUIDs() const {
  return platform_->getUUIDs();
}

bool VulkanDriver::queryCapabilities(const DeviceCapabilityBits capabilities) const {
  return ((platform_->getHomogeneousCapabilityBits() & capabilities) == capabilities);
}

const DeviceLimits& VulkanDriver::getLimits() const {
  return platform_->getHomogenousDeviceLimits();
}

// From BaseDriver
DeviceContextUqPtr VulkanDriver::createDeviceContextInternal(const heavyai::UUID& uuid,
                                                             const DeviceId gpu_id) {
  return platform_->createDeviceContext(uuid, gpu_id);
}

bool VulkanDriver::destroyDeviceContextInternal(DeviceContextUqPtr device_ctx) {
  return platform_->destroyDeviceContext(std::move(device_ctx));
}

WindowSystemIntegration* VulkanDriver::getWSI() const {
  return platform_->getWSI();
}

VulkanDebugUtils& VulkanDriver::getDebugUtils() const {
  return platform_->getDebugUtils();
}

MemoryUsageInfo VulkanDriver::getPeakMemoryUsage() const {
  return platform_->getPeakMemoryUsage();
}

}  // namespace gfx
