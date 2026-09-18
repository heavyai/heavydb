/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>
#include <string>

#include <vulkan/vulkan.h>
#include <boost/noncopyable.hpp>

namespace gfx {

class VulkanDeviceContext;

/**
 * Wrapper around VkQueue command queue objects, allowing for submit count tracking,
 * Fence management and synchronization.
 * */
class VulkanQueue : boost::noncopyable {
 public:
  VulkanQueue(VulkanDeviceContext& device_ctx);
  ~VulkanQueue() = default;

  void init(VkDevice device,
            uint32_t family_index,
            uint32_t queue_index,
            const std::string& queue_name);
  void submit(uint32_t submit_count,
              const VkSubmitInfo2& submit_info,
              VkFence fence,
              const std::string_view label);
  // present returns true if successful
  // false if swapchain needs to be rebuilt
  bool present(VkPresentInfoKHR* present_info);
  void waitIdle();

  uint32_t getFamilyIndex() const { return family_index_; }
  VkQueue getHandle() const { return vk_queue_; }

  void setDeviceLostState() { is_device_lost_ = true; }

 private:
  VulkanDeviceContext& device_ctx_;
  VkQueue vk_queue_;
  uint32_t family_index_;
  bool is_device_lost_;
  std::mutex submit_mutex_;

  void checkForDeviceLost(VkResult result);
};

}  // namespace gfx
