/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Commands/FenceManager.h"

#include <limits>

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

namespace {
Fence::Status vk_result_to_fence_status(VkResult result) {
  switch (result) {
    case VK_SUCCESS:
      return Fence::Status::kSignaled;
    case VK_NOT_READY:
    case VK_TIMEOUT:
      return Fence::Status::kUnsignaled;
    case VK_ERROR_DEVICE_LOST:
      return Fence::Status::kDeviceLost;
    default:
      CHECK_VKRESULT(result, "updating Fence status");
  }
  // use DeviceLost as generic error return
  return Fence::Status::kDeviceLost;
}
}  // namespace

//
// Fence
//
Fence::Fence(VkDevice vk_device, VkFence vk_fence, uint32_t index)
    : vk_device_{vk_device}
    , vk_fence_{vk_fence}
    , status_{Status::kUnsignaled}
    , index_{index} {
  CHECK(vk_fence_ != VK_NULL_HANDLE);
}

Fence::Status Fence::wait(uint64_t timeout_in_ms) {
  static constexpr uint64_t max_timeout = std::numeric_limits<uint64_t>::max() / 1000000;
  CHECK_LE(timeout_in_ms, max_timeout);
  if (status_ != kDeviceLost) {
    status_ = vk_result_to_fence_status(
        vkWaitForFences(vk_device_, 1, &vk_fence_, VK_TRUE, timeout_in_ms * 1000000));
  }
  return status_;
}

Fence::Status Fence::getStatus() {
  return status_;
}

Fence::Status Fence::refreshStatus() {
  if (status_ != kDeviceLost) {
    status_ = vk_result_to_fence_status(vkGetFenceStatus(vk_device_, vk_fence_));
  }
  return status_;
}

void Fence::reset() {
  if (status_ != kDeviceLost) {
    CHECK_VKRESULT(vkResetFences(vk_device_, 1, &vk_fence_), "resetting Fence");
    status_ = Status::kUnsignaled;
  }
}

//
// FenceManager
//
FenceManager::FenceManager(VulkanDeviceContext& device_ctx) : device_ctx_{device_ctx} {}

FenceManager::~FenceManager() {
  destroyFences();
}

Fence* FenceManager::acquireFence() {
  if (free_fences_.size()) {
    auto* fence = fences_[free_fences_.back()].get();
    free_fences_.pop_back();
    return fence;
  } else {
    return addFence();
  }
}

void FenceManager::releaseFence(Fence* fence) {
  CHECK(fence);
  fence->reset();
  free_fences_.push_back(fence->index_);
}

Fence* FenceManager::addFence() {
  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  VkFence vk_fence;
  CHECK_VKRESULT(vkCreateFence(device_ctx_.getHandle(), &fence_info, nullptr, &vk_fence),
                 "Error creating fence");

  // don't name fences yet
  // @TODO(scb) name fences along with command annotations
  // device_ctx_.nameVulkanObject(VK_OBJECT_TYPE_FENCE, vk_fence, "Fence");

  uint32_t index = fences_.size();
  fences_.emplace_back(std::make_unique<Fence>(device_ctx_.getHandle(), vk_fence, index));
  return fences_.back().get();
}

void FenceManager::destroyFences() {
  for (auto& fence : fences_) {
    vkDestroyFence(device_ctx_.getHandle(), fence->vk_fence_, nullptr);
  }
  free_fences_.clear();
  fences_.clear();
}

void FenceManager::setDeviceLostState() {
  for (auto& fence : fences_) {
    fence->status_ = Fence::kDeviceLost;
  }
}

}  // namespace gfx
