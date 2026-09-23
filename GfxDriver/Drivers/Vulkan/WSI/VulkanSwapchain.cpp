/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/WSI/VulkanSwapchain.h"

#include <algorithm>

#include "GfxDriver/Drivers/Vulkan/Resources/ImageLayoutManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Drivers/Vulkan/WSI/VulkanWSI.h"

namespace gfx {

VulkanSwapchain::VulkanSwapchain(VulkanDeviceContext& device_ctx,
                                 VulkanWSI& vulkan_wsi,
                                 uint32_t command_timeout_ms)
    : device_{device_ctx}
    , vulkan_wsi_{vulkan_wsi}
    , vk_surface_{vulkan_wsi.getSurface()}
    , command_timeout_{command_timeout_ms * 1000000ULL}
    , vk_surface_format_{chooseSwapSurfaceFormat()}
    , vk_present_mode_{chooseSwapPresentMode()}
    , extent_{0, 0}
    , vk_swapchain_{VK_NULL_HANDLE}
    , vk_fence_{VK_NULL_HANDLE}
    , is_valid_{false} {}

VulkanSwapchain::~VulkanSwapchain() {
  destroyFence();
  destroySwapchain();
}

void VulkanSwapchain::createSwapchain() {
  // Get current capabilities (image count, extents of current surface, etc)
  VkSurfaceCapabilitiesKHR capabilities;
  CHECK_VKRESULT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                     device_.getPhysicalDeviceHandle(), vk_surface_, &capabilities),
                 "getting surface capabilities");

  auto extent = chooseSwapExtent(capabilities);

  // determine image count for swap chain
  uint32_t image_count = capabilities.minImageCount;
  if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
    image_count = capabilities.maxImageCount;
  }

  // Create the swap chain
  VkSwapchainCreateInfoKHR swapchain_ci = {};
  swapchain_ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapchain_ci.surface = vk_surface_;
  swapchain_ci.minImageCount = image_count;
  swapchain_ci.imageFormat = vk_surface_format_.format;
  swapchain_ci.imageColorSpace = vk_surface_format_.colorSpace;
  swapchain_ci.imageExtent = extent;
  swapchain_ci.imageArrayLayers = 1;  // always 1 for non-stereo renders
  swapchain_ci.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  // Ensure both graphics and presentation are the same queue
  // This can be relaxed once we support queue transfer barriers
  auto const& qf = device_.getPhysicalDevice().getQueueFamilyIndices();
  CHECK_EQ(qf.graphics, qf.present);

  swapchain_ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapchain_ci.queueFamilyIndexCount = 0;      // optional
  swapchain_ci.pQueueFamilyIndices = nullptr;  // optional

  // just use the current transform. rotate or flip here if needed
  swapchain_ci.preTransform = capabilities.currentTransform;

  // ignore alpha bit when window compositing
  swapchain_ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

  swapchain_ci.presentMode = vk_present_mode_;
  swapchain_ci.clipped =
      VK_TRUE;  // clip to other windows (don't use if reading back image)

  // pass current swapchain if we have one
  auto old_swapchain = vk_swapchain_;
  swapchain_ci.oldSwapchain = vk_swapchain_;

  VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
  auto vk_device = device_.getHandle();
  auto result = vkCreateSwapchainKHR(vk_device, &swapchain_ci, nullptr, &new_swapchain);

  // now destroy the old swapchain
  if (old_swapchain != VK_NULL_HANDLE) {
    destroySwapchain();
  }
  vk_swapchain_ = new_swapchain;
  CHECK_VKRESULT(result, "creating Swapchain");

  // retrieve the swapchain images
  vkGetSwapchainImagesKHR(vk_device, vk_swapchain_, &image_count, nullptr);
  vk_swapchain_images_.resize(image_count);
  vkGetSwapchainImagesKHR(
      vk_device, vk_swapchain_, &image_count, vk_swapchain_images_.data());
  auto& image_layout_mgr =
      static_cast<VulkanResourceManager&>(device_.getResourceManager())
          .getImageLayoutManager();

  for (auto const& image : vk_swapchain_images_) {
    image_layout_mgr.addOrSetLayout(image, ImageLayout::kUndefined);
  }

  extent_ = extent;

  if (vk_fence_ == VK_NULL_HANDLE) {
    createFence();
  }
  is_valid_ = true;
}

void VulkanSwapchain::createFence() {
  CHECK(vk_fence_ == VK_NULL_HANDLE);
  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  CHECK_VKRESULT(vkCreateFence(device_.getHandle(), &fence_info, nullptr, &vk_fence_),
                 "creating wsi fence");
}

void VulkanSwapchain::destroyFence() {
  if (vk_fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_.getHandle(), vk_fence_, nullptr);
    vk_fence_ = VK_NULL_HANDLE;
  }
}

void VulkanSwapchain::destroySwapchain() {
  auto& image_layout_mgr =
      static_cast<VulkanResourceManager&>(device_.getResourceManager())
          .getImageLayoutManager();
  for (auto const& image : vk_swapchain_images_) {
    image_layout_mgr.remove(image);
  }
  vk_swapchain_images_.clear();

  auto vk_device = device_.getHandle();
  if (vk_swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(vk_device, vk_swapchain_, nullptr);
    vk_swapchain_ = VK_NULL_HANDLE;
  }
}

void VulkanSwapchain::checkResultForTimeout(VkResult result, std::string_view message) {
  if ((result == VK_ERROR_DEVICE_LOST) || (result == VK_TIMEOUT)) {
    // We're dead. Call to DeviceContext handler which will throw out
    LOG(ERROR) << message << "failed: " << vulkan_result_to_string(result);
    device_.handleDeviceLost(result == VK_TIMEOUT);
  } else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    throw OutOfGpuMemoryError("Out of device memory " + std::string(message));
  }
}

std::pair<uint32_t, VkImage> VulkanSwapchain::acquireNextImage() {
  if (!is_valid_) {
    device_.waitIdle();
    createSwapchain();
  }

  uint32_t image_index = 0;
  auto try_acquire = [this, &image_index]() {
    auto acquire_result = vkAcquireNextImageKHR(device_.getHandle(),
                                                vk_swapchain_,
                                                command_timeout_,
                                                VK_NULL_HANDLE,
                                                vk_fence_,
                                                &image_index);

    checkResultForTimeout(acquire_result, "acquireNextImage timeout");

    // Wait for fence indicating we have the image
    // Can be replaced by a semaphore
    auto fence_result =
        vkWaitForFences(device_.getHandle(), 1, &vk_fence_, VK_TRUE, command_timeout_);
    checkResultForTimeout(fence_result, "Error waiting for swapchain fence");
    CHECK_VKRESULT(fence_result, "Error waiting for swapchain fence");

    // Reset the fence for next time
    CHECK_VKRESULT(vkResetFences(device_.getHandle(), 1, &vk_fence_),
                   "Error resetting swapchain fence");

    // return results from vkAcquireNextImage so we can check for invalid swapchain
    return acquire_result;
  };

  auto result = try_acquire();

  // Check if we need to rebuild the swapchain
  if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
    createSwapchain();
    // Try to acquire again but now treat everything as a fatal error
    CHECK_VKRESULT(try_acquire(), "Error during acquireNextImage retry");
  } else {
    // Check for any other errors
    CHECK_VKRESULT(result, "Error during acquireNextImage")
  }

  CHECK_LT(image_index, vk_swapchain_images_.size());
  return {image_index, vk_swapchain_images_[image_index]};
}

VkSurfaceFormatKHR VulkanSwapchain::chooseSwapSurfaceFormat() {
  VkPhysicalDevice device_handle = device_.getPhysicalDeviceHandle();
  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(
      device_handle, vk_surface_, &format_count, nullptr);
  CHECK(format_count);

  std::vector<VkSurfaceFormatKHR> formats(format_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(
      device_handle, vk_surface_, &format_count, formats.data());

  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
    // surface has no preferred format, choose any format we want
    return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  }

  // check if our desired format is available at all
  for (const auto& format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }

  // TODO: iterate and try to choose something decent. We probably won't have to worry
  // about this meanwhile just pick the first format to get something going
  LOG(WARNING) << "unable to choose desired swap format, using first available";
  return formats[0];
}

VkPresentModeKHR VulkanSwapchain::chooseSwapPresentMode() {
  // Present mode enumeration
  // These are not currently used as we just pick the spec required mode,
  // but keeping them for future use
  VkPhysicalDevice device_handle = device_.getPhysicalDeviceHandle();
  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      device_handle, vk_surface_, &present_mode_count, nullptr);

  std::vector<VkPresentModeKHR> present_modes(present_mode_count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      device_handle, vk_surface_, &present_mode_count, present_modes.data());

  // Use FIFO if mailbox not available (Conformant drivers must support it)
  // TODO: Add support for different present modes if we'd find them useful
  // Since we aren't a hard realtime application FIFO should be fine
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapchain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps) {
  if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    // normally just match the window size
    return caps.currentExtent;
  } else {
    // window manager doesn't require matching window size, so select a resolution that
    // best matches the window within the extent bounds
    auto [width, height] = vulkan_wsi_.getWindowSize();
    VkExtent2D actualExtent = {
        std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height)};

    return actualExtent;
  }
}

}  // namespace gfx
