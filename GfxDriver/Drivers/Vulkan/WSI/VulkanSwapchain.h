/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

namespace gfx {

class VulkanWSI;
class VulkanDeviceContext;

class VulkanSwapchain {
 public:
  explicit VulkanSwapchain(VulkanDeviceContext& device_ctx,
                           VulkanWSI& vulkan_wsi,
                           uint32_t command_timeout_ms);
  ~VulkanSwapchain();

  // Acquire the next available image in the swapchain queue
  // Automatically builds the swapchain the first time called (is_valid_ default is false)
  // returns the index into the swapchain vector and the image handle
  // index may be used for indexing fence or semaphore arrays when rendering directly
  // to swapchain images during continuous rendering
  std::pair<uint32_t, VkImage> acquireNextImage();

  // Invalidate the swapchain
  // Trigger a new swapchain to be built, and retire the current swapchain
  // Call during window events such as resize
  void setInvalid() { is_valid_ = false; }

  // Get the images in the current swapchain
  // Will be empty until acquireNextImage is called the first time
  const std::vector<VkImage>& getImages() const { return vk_swapchain_images_; }

  // Get handle
  // will be VK_NULL_HANDLE until acquireNextImage is called
  VkSwapchainKHR getHandle() const { return vk_swapchain_; }

  // Get color format for the surface
  // can be called prior to VkSwapchain creation
  VkFormat getColorFormat() const { return vk_surface_format_.format; }

 private:
  // Invariants
  VulkanDeviceContext& device_;
  VulkanWSI& vulkan_wsi_;
  VkSurfaceKHR vk_surface_;
  uint64_t command_timeout_;

  VkSurfaceFormatKHR vk_surface_format_;
  VkPresentModeKHR vk_present_mode_;
  VkExtent2D extent_;

  // Only valid after first acquireNextImage call
  VkSwapchainKHR vk_swapchain_;
  std::vector<VkImage> vk_swapchain_images_;
  VkFence vk_fence_;

  bool is_valid_;

  void createSwapchain();
  void destroySwapchain();
  void createFence();
  void destroyFence();
  void checkResultForTimeout(VkResult result, std::string_view message);
  VkSurfaceFormatKHR chooseSwapSurfaceFormat();
  VkPresentModeKHR chooseSwapPresentMode();
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
};

}  // namespace gfx
