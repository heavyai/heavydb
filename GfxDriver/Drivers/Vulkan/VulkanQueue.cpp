/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanQueue.h"

#include <mutex>

#include "GfxDriver/Drivers/Vulkan/VulkanDebugUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

VulkanQueue::VulkanQueue(VulkanDeviceContext& device_ctx)
    : device_ctx_{device_ctx}
    , vk_queue_{VK_NULL_HANDLE}
    , family_index_{0}
    , is_device_lost_{false} {}

void VulkanQueue::checkForDeviceLost(VkResult result) {
  if (result == VK_ERROR_DEVICE_LOST) {
    is_device_lost_ = true;
    // Pass to device context which will throw
    device_ctx_.handleDeviceLost(false);
  }
}

void VulkanQueue::submit(uint32_t submit_count,
                         const VkSubmitInfo2& submit_info,
                         VkFence fence,
                         const std::string_view name) {
  CHECK(vk_queue_ != VK_NULL_HANDLE);
  std::lock_guard<std::mutex> lock(submit_mutex_);
  if (!is_device_lost_) {
    device_ctx_.getDebugUtils().beginQueueLabel(vk_queue_, name);
    auto result = vkQueueSubmit2(vk_queue_, submit_count, &submit_info, fence);
    device_ctx_.getDebugUtils().endQueueLabel(vk_queue_);
    checkForDeviceLost(result);
    CHECK_VKRESULT(result, "Error submitting command buffers");
  }
}

bool VulkanQueue::present(VkPresentInfoKHR* present_info) {
  std::lock_guard<std::mutex> lock(submit_mutex_);
  if (!is_device_lost_) {
    auto result = vkQueuePresentKHR(vk_queue_, present_info);
    checkForDeviceLost(result);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      return false;
    } else {
      CHECK_VKRESULT(result, "Error presenting image");
    }
  }
  return true;
}

void VulkanQueue::waitIdle() {
  CHECK(vk_queue_);
  std::lock_guard<std::mutex> lock(submit_mutex_);
  if (!is_device_lost_) {
    VkResult result = vkQueueWaitIdle(vk_queue_);
    checkForDeviceLost(result);
    CHECK_VKRESULT(result, "Error waiting for queue to idle");
  }
}

void VulkanQueue::init(VkDevice device,
                       uint32_t family_index,
                       uint32_t queue_index,
                       const std::string& queue_name) {
  vkGetDeviceQueue(device, family_index, queue_index, &vk_queue_);
  family_index_ = family_index;
  device_ctx_.nameVulkanObject(VK_OBJECT_TYPE_QUEUE, vk_queue_, queue_name);
}

}  // namespace gfx
