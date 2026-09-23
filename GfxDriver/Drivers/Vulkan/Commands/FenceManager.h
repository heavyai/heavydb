/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include <vulkan/vulkan.h>
#include <boost/noncopyable.hpp>

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"

namespace gfx {

class FenceManager;

class Fence {
 public:
  enum Status { kUnsignaled, kSignaled, kDeviceLost };

  explicit Fence(VkDevice vk_device, VkFence vk_fence, uint32_t index);
  Fence() = delete;

  VkFence getHandle() { return vk_fence_; }

  Status wait(uint64_t timeout_in_ms);
  Status getStatus();
  Status refreshStatus();
  void reset();

 private:
  VkDevice vk_device_;
  VkFence vk_fence_;
  Status status_;
  uint32_t index_;

  friend class FenceManager;
};

class FenceManager : boost::noncopyable {
 public:
  explicit FenceManager(VulkanDeviceContext& device_ctx);
  FenceManager() = delete;
  ~FenceManager();

  Fence* acquireFence();
  void releaseFence(Fence* fence);

  void setDeviceLostState();

 private:
  VulkanDeviceContext& device_ctx_;
  std::vector<std::unique_ptr<Fence>> fences_;
  std::vector<uint32_t> free_fences_;

  Fence* addFence();
  void destroyFences();
};

}  // namespace gfx
