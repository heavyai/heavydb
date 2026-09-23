/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/WSI/WindowSystemIntegration.h"

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/WSI/VulkanSwapchain.h"

namespace gfx {

// Middle layer for WindowSystemIntegration implementations
// Captures the presentation device and handles acquiring and presenting the image
// Adds Vulkan specific required interface methods
class VulkanWSI : public WindowSystemIntegration {
 public:
  VulkanWSI();
  ~VulkanWSI() override;

  // Device in use for presentation
  void setPresentDevice(VulkanDeviceContext& device);

  // VkInstance (for imgui)
  const VkInstance getVkInstance() const { return vk_instance_; }

  // from WindowSystemIntegration
  void shutdown() override;
  PixelFormat getWindowPixelFormat() const override;
  void copyAndPresentTexture(const Texture& texture) override;

  // VulkanWSI required by final derived classes
  virtual VkSurfaceKHR createSurface(VkInstance instance) = 0;
  virtual void destroySurface() = 0;
  virtual VkSurfaceKHR getSurface() const = 0;

  // Get required Vulkan Instance extensions
  virtual std::vector<const char*> getRequiredInstanceExtensions() const = 0;

  // Window properties
  virtual std::pair<uint32_t, uint32_t> getWindowSize() const = 0;

 protected:
  VulkanDeviceContext* present_device_;
  VkSemaphore vk_ready_to_present_semaphore_;

 protected:
  VkInstance vk_instance_;
};

}  // namespace gfx
