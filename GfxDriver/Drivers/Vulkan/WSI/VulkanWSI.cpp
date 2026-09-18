/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/WSI/VulkanWSI.h"

#include "GfxDriver/Drivers/Vulkan/Commands/StagingContext.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

VulkanWSI::VulkanWSI()
    : present_device_{nullptr}
    , vk_ready_to_present_semaphore_{VK_NULL_HANDLE}
    , vk_instance_{VK_NULL_HANDLE} {}

VulkanWSI::~VulkanWSI() {
  shutdown();
}

void VulkanWSI::shutdown() {
  if (vk_ready_to_present_semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(
        present_device_->getHandle(), vk_ready_to_present_semaphore_, nullptr);
    vk_ready_to_present_semaphore_ = VK_NULL_HANDLE;
  }
}

void VulkanWSI::setPresentDevice(VulkanDeviceContext& device) {
  CHECK(present_device_ == nullptr);
  CHECK(vk_ready_to_present_semaphore_ == VK_NULL_HANDLE);

  present_device_ = &device;

  VkSemaphoreCreateInfo semaphore_ci = {};
  semaphore_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  CHECK_VKRESULT(vkCreateSemaphore(present_device_->getHandle(),
                                   &semaphore_ci,
                                   nullptr,
                                   &vk_ready_to_present_semaphore_),
                 "Error creating present semaphore");

  present_device_->nameVulkanObject(VK_OBJECT_TYPE_SEMAPHORE,
                                    vk_ready_to_present_semaphore_,
                                    "Semaphore (WSI ready to present)");
}

PixelFormat VulkanWSI::getWindowPixelFormat() const {
  return vk_format_to_pixel_format(present_device_->getSwapchain()->getColorFormat());
}

void VulkanWSI::copyAndPresentTexture(const Texture& texture) {
  CHECK(present_device_);

  auto* swapchain = present_device_->getSwapchain();
  CHECK(swapchain);

  // Get the next available swapchain image
  auto [image_index, vk_swap_image] = swapchain->acquireNextImage();

  auto& vk_texture = static_cast<const VulkanTexture&>(texture);
  auto [width, height] = getWindowSize();
  width = std::min(width, texture.getWidth());
  height = std::min(height, texture.getHeight());

  // Copy source texture into swapchain image
  present_device_->getStagingContext().copyOrResolvePixels(
      vk_texture.getImage(),
      vk_swap_image,
      width,
      height,
      PixelFormat::kRGBA8,
      texture.getNumSamples(),
      false,
      reinterpret_cast<SemaphoreHandle>(vk_ready_to_present_semaphore_),
      ImageLayout::kPresentSrc);

  // Present the frame when ready
  VkPresentInfoKHR present_info = {};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &vk_ready_to_present_semaphore_;

  auto vk_swapchain = swapchain->getHandle();
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &vk_swapchain;
  present_info.pImageIndices = &image_index;
  present_info.pResults = nullptr;

  auto did_present = present_device_->getGraphicsQueue().present(&present_info);
  if (!did_present) {
    // Presentation return OUT_OF_DATE or SUBOPTIMAL
    // Invalidate the swapchain so it will rebuild next time an image is acquired
    swapchain->setInvalid();
  }
}

}  // namespace gfx
