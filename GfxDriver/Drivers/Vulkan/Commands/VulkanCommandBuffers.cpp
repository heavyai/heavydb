/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"

#include <numeric>

#include "GfxDriver/Drivers/Vulkan/VulkanDebugUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

static std::string_view kEmptyName{""};
//
// Command Buffer
//
VulkanCommandBuffer::VulkanCommandBuffer(const VulkanDeviceContext& device_ctx,
                                         VulkanCommandPool& owning_pool,
                                         VkCommandBuffer handle,
                                         uint32_t index)
    : device_ctx_{device_ctx}
    , owning_pool_{owning_pool}
    , submit_queue_{nullptr}
    , vk_command_buffer_{handle}
    , index_{index}
    , state_{State::kInitial}
    , fence_{device_ctx.getFenceManager().acquireFence()}
    , name_{kEmptyName} {}

VulkanCommandBuffer::~VulkanCommandBuffer() {
  if (fence_) {
    device_ctx_.getFenceManager().releaseFence(fence_);
  }
}

void VulkanCommandBuffer::beginRecording() {
  CHECK(state_ == State::kInitial);

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  // We do not support resubmitting a command buffer, we always reset
  // and re-record them
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  // No secondary command buffers yet, so pInheritanceInfo not used
  begin_info.pInheritanceInfo = nullptr;

  CHECK_VKRESULT(vkBeginCommandBuffer(vk_command_buffer_, &begin_info),
                 "beginning command buffer recording");
  state_ = State::kRecording;
}

void VulkanCommandBuffer::insertLabel(const std::string_view name) {
  device_ctx_.getDebugUtils().insertCmdLabel(
      vk_command_buffer_, name.empty() ? kEmptyName.data() : name.data());
}

void VulkanCommandBuffer::pushLabel(const std::string_view name) {
  device_ctx_.getDebugUtils().beginCmdLabel(vk_command_buffer_, name.data());
}

void VulkanCommandBuffer::popLabel() {
  device_ctx_.getDebugUtils().endCmdLabel(vk_command_buffer_);
}

void VulkanCommandBuffer::endRecordingAndSubmit(const std::string_view name) {
  CHECK(state_ == State::kRecording);

  // Finish recording command buffer and check for errors
  CHECK_VKRESULT(vkEndCommandBuffer(vk_command_buffer_),
                 "ending command buffer recording");
  state_ = State::kExecutable;

  VkCommandBufferSubmitInfo cmd_info = {};
  cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
  cmd_info.commandBuffer = vk_command_buffer_;

  VkSubmitInfo2 submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  submit_info.commandBufferInfoCount = 1;
  submit_info.pCommandBufferInfos = &cmd_info;

  auto vk_fence = fence_->getHandle();
  submit_queue_->submit(1, submit_info, vk_fence, name);

  state_ = State::kPending;
  name_ = name;
}

void VulkanCommandBuffer::endRecordingAndSubmit(
    const std::string_view name,
    uint32_t wait_semaphore_count,
    const VkSemaphore* wait_semaphores,
    const std::vector<VkPipelineStageFlags2>& wait_stages,
    uint32_t signal_semaphore_count,
    const VkSemaphore* signal_semaphores,
    const std::vector<VkPipelineStageFlags2>& signal_stages) {
  CHECK(state_ == State::kRecording);
  // Finish recording command buffer and check for errors
  CHECK_VKRESULT(vkEndCommandBuffer(vk_command_buffer_),
                 "ending command buffer recording");
  state_ = State::kExecutable;

  CHECK_LE(wait_semaphore_count, wait_stages.size());

  VkCommandBufferSubmitInfo cmd_info = {};
  cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
  cmd_info.commandBuffer = vk_command_buffer_;

  std::vector<VkSemaphoreSubmitInfo> wait_infos(wait_semaphore_count);
  for (uint32_t i = 0; i < wait_semaphore_count; ++i) {
    wait_infos[i].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait_infos[i].pNext = nullptr;
    wait_infos[i].semaphore = wait_semaphores[i];
    wait_infos[i].value = 0;  // timeline value
    wait_infos[i].stageMask = wait_stages[i];
    wait_infos[i].deviceIndex = 0;
  }

  std::vector<VkSemaphoreSubmitInfo> signal_infos(signal_semaphore_count);
  for (uint32_t i = 0; i < signal_semaphore_count; ++i) {
    signal_infos[i].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal_infos[i].pNext = nullptr;
    signal_infos[i].semaphore = signal_semaphores[i];
    signal_infos[i].value = 0;  // timeline value
    signal_infos[i].stageMask = signal_stages[i];
    signal_infos[i].deviceIndex = 0;
  }

  VkSubmitInfo2 submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  submit_info.pNext = nullptr;
  submit_info.flags = 0;
  submit_info.commandBufferInfoCount = 1;
  submit_info.pCommandBufferInfos = &cmd_info;
  submit_info.waitSemaphoreInfoCount = wait_semaphore_count;
  submit_info.pWaitSemaphoreInfos = wait_infos.data();
  submit_info.signalSemaphoreInfoCount = signal_semaphore_count;
  submit_info.pSignalSemaphoreInfos = signal_infos.data();

  auto vk_fence = fence_->getHandle();
  submit_queue_->submit(1, submit_info, vk_fence, name);

  state_ = State::kPending;
  name_ = name;
}

VulkanCommandBuffer::State VulkanCommandBuffer::refreshFenceState() {
  if (state_ == State::kPending) {
    if (fence_->refreshStatus() == Fence::Status::kSignaled) {
      state_ = State::kInvalid;
    }
  }
  return state_;
}

void VulkanCommandBuffer::waitForCompletion(uint64_t timeout_in_ms) {
  if ((state_ == State::kPending) &&
      (fence_->wait(timeout_in_ms) == Fence::Status::kSignaled)) {
    // Commands completed, buffer is invalid until reset
    state_ = State::kInvalid;
  }
}

VkResult VulkanCommandBuffer::reset() {
  if (fence_->getStatus() != Fence::Status::kDeviceLost) {
    auto result = vkResetCommandBuffer(vk_command_buffer_,
                                       VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    if (result == VK_SUCCESS) {
      state_ = VulkanCommandBuffer::State::kInitial;
      fence_->reset();
    }
    return result;
  } else {
    return VK_ERROR_DEVICE_LOST;
  }
}

void VulkanCommandBuffer::setSubmitQueue(VulkanQueue* queue) {
  CHECK(queue);
  submit_queue_ = queue;
}

void VulkanCommandBuffer::beginRenderPass(VkRenderPassBeginInfo* begin_info,
                                          VkSubpassContents subpass_contents) {
  CHECK(state_ == State::kRecording);
  vkCmdBeginRenderPass(vk_command_buffer_, begin_info, subpass_contents);
  state_ = VulkanCommandBuffer::State::kInRenderpass;
}

void VulkanCommandBuffer::endRenderPass() {
  CHECK(state_ == State::kInRenderpass);
  vkCmdEndRenderPass(vk_command_buffer_);
  state_ = VulkanCommandBuffer::State::kRecording;
}

//
// Command Pool
//
bool VulkanCommandPool::is_device_lost_ = false;

VulkanCommandPool::VulkanCommandPool(VulkanDeviceContext& device_context,
                                     VulkanQueue& queue,
                                     VulkanDeviceContext::CommandPoolSelector selector,
                                     uint32_t command_timeout_ms)
    : device_ctx_{device_context}
    , vk_command_pool_{VK_NULL_HANDLE}
    , queue_{queue}
    , command_timeout_ms_{command_timeout_ms} {
  is_device_lost_ = false;
  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue.getFamilyIndex();
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  VkResult result = vkCreateCommandPool(
      device_ctx_.getHandle(), &pool_info, nullptr, &vk_command_pool_);
  CHECK_VKRESULT(result, "creating CommandPool");

  device_ctx_.nameVulkanObject(VK_OBJECT_TYPE_COMMAND_POOL,
                               vk_command_pool_,
                               "Command Pool (" + to_string(selector) + ")");
}

VulkanCommandPool::~VulkanCommandPool() {
  waitForPendingFences();
  command_buffers_.clear();
  if (vk_command_pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_ctx_.getHandle(), vk_command_pool_, nullptr);
    vk_command_pool_ = VK_NULL_HANDLE;
  }
}

VulkanCommandBuffer* VulkanCommandPool::acquireBuffer() {
  VulkanCommandBuffer* cmd_buffer = nullptr;
  if (free_buffer_ids_.size()) {
    cmd_buffer = command_buffers_[free_buffer_ids_.back()].get();
    free_buffer_ids_.pop_back();
  } else {
    cmd_buffer = addBuffer();
  }
  CHECK(cmd_buffer);
  cmd_buffer->setSubmitQueue(&queue_);
  cmd_buffer->beginRecording();
  return cmd_buffer;
}

void VulkanCommandPool::submitBuffer(VulkanCommandBuffer* cmd_buffer,
                                     const std::string_view name) {
  CHECK(cmd_buffer);
  CHECK_EQ(&cmd_buffer->getOwningPool(), this);
  cmd_buffer->endRecordingAndSubmit(name);
  pending_buffer_ids_.push_back(cmd_buffer->index_);
}

void VulkanCommandPool::submitBuffer(
    VulkanCommandBuffer* cmd_buffer,
    const std::string_view name,
    uint32_t wait_semaphore_count,
    const VkSemaphore* wait_semaphores,
    const std::vector<VkPipelineStageFlags2>& wait_stages,
    uint32_t signal_semaphore_count,
    const VkSemaphore* signal_semaphores,
    const std::vector<VkPipelineStageFlags2>& signal_stages) {
  CHECK(cmd_buffer);
  CHECK_EQ(&cmd_buffer->getOwningPool(), this);
  cmd_buffer->endRecordingAndSubmit(name,
                                    wait_semaphore_count,
                                    wait_semaphores,
                                    wait_stages,
                                    signal_semaphore_count,
                                    signal_semaphores,
                                    signal_stages);
  pending_buffer_ids_.push_back(cmd_buffer->index_);
}

VkResult VulkanCommandPool::waitForPendingFences() {
  if (pending_buffer_ids_.empty() || is_device_lost_) {
    return VK_SUCCESS;
  }
  // Gather fences for pending buffers
  std::vector<VkFence> pending_fences;
  pending_fences.reserve(pending_buffer_ids_.size());
  for (auto const buffer_id : pending_buffer_ids_) {
    pending_fences.push_back(command_buffers_[buffer_id]->fence_->getHandle());
  }
  // wait on all pending fences
  return vkWaitForFences(device_ctx_.getHandle(),
                         pending_fences.size(),
                         pending_fences.data(),
                         VK_TRUE,
                         command_timeout_ms_ * 1000000ULL);
}

bool VulkanCommandPool::waitForPendingBuffers(WaitCallback callback) {
  if (pending_buffer_ids_.empty() || is_device_lost_) {
    return true;
  }

  auto result = waitForPendingFences();

  if (result == VK_SUCCESS) {
    // Reset all the command buffers and add to free list
    for (auto const buffer_id : pending_buffer_ids_) {
      command_buffers_[buffer_id]->reset();
    }
    free_buffer_ids_.insert(
        free_buffer_ids_.end(), pending_buffer_ids_.begin(), pending_buffer_ids_.end());
    pending_buffer_ids_.clear();
  } else if ((result == VK_ERROR_DEVICE_LOST) || (result == VK_TIMEOUT)) {
    is_device_lost_ = true;
    // We're dead. Call to DeviceContext handler which will throw out
    LOG(ERROR) << "Command buffer failed to complete: "
               << vulkan_result_to_string(result);
    if (callback) {
      // Assume callback handles pending buffer name logging
      callback(result);
    } else {
      std::stringstream ss;
      logPendingBufferNames(ss);
      LOG(ERROR) << ss.str();
    }
    device_ctx_.handleDeviceLost(result == VK_TIMEOUT);
  } else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    if (callback) {
      callback(result);
    }
    throw OutOfGpuMemoryError("Out of device memory waiting for VkFences");
  } else {
    if (callback) {
      callback(result);
    }
    // Not all fences completed, refresh the state of any pending buffers that did
    // complete and let the caller handle the timeout
    // refreshFencesAndBufferIds();
    LOG(FATAL) << "Vulkan error waiting for fences: " << vulkan_result_to_string(result);
  }
  return (result == VK_SUCCESS);
}

void VulkanCommandPool::refreshFencesAndBufferIds() {
  std::vector<uint32_t> new_free_ids;
  std::vector<uint32_t> still_pending_ids;
  // Iterate all pending buffers, refreshing fence status
  // Reset any buffers that have completed and build id lists for
  // still pending and newly free ids
  for (auto const buffer_id : pending_buffer_ids_) {
    auto& cmd_buffer = command_buffers_[buffer_id];
    if (cmd_buffer->refreshFenceState() == VulkanCommandBuffer::State::kInvalid) {
      command_buffers_[buffer_id]->reset();
      new_free_ids.push_back(buffer_id);
    } else {
      still_pending_ids.push_back(buffer_id);
    }
  }

  // Update pending and free id caches
  pending_buffer_ids_.swap(still_pending_ids);
  free_buffer_ids_.insert(
      free_buffer_ids_.begin(), new_free_ids.begin(), new_free_ids.end());
}

void VulkanCommandPool::resetPool() {
  CHECK_VKRESULT(waitForPendingFences(), "waiting for pending fences");
  pending_buffer_ids_.clear();
  std::iota(free_buffer_ids_.begin(), free_buffer_ids_.end(), 0);
  for (auto& cmd_buffer : command_buffers_) {
    cmd_buffer->reset();
  }
  auto result = vkResetCommandPool(device_ctx_.getHandle(),
                                   vk_command_pool_,
                                   VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
  if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    throw OutOfGpuMemoryError("Out of device memory resetting vkCommandPool");
  }
  CHECK_VKRESULT(result, "resetting command pool");
}

VulkanCommandBuffer* VulkanCommandPool::addBuffer() {
  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = vk_command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer buffer_handle;
  CHECK_VKRESULT(
      vkAllocateCommandBuffers(device_ctx_.getHandle(), &alloc_info, &buffer_handle),
      "allocating command buffers");
  uint32_t index = command_buffers_.size();
  command_buffers_.emplace_back(
      std::make_unique<VulkanCommandBuffer>(device_ctx_, *this, buffer_handle, index));

  return command_buffers_.back().get();
}

void VulkanCommandPool::logPendingBufferNames(std::ostream& os) const {
  os << "Pending command buffers: ";
  int pending_count = 0;
  for (auto const& buffer : command_buffers_) {
    if (buffer->getState() == VulkanCommandBuffer::State::kPending) {
      os << "\n  " << buffer->getName();
      pending_count++;
    }
  }
  if (pending_count == 0) {
    os << "None";
  }
  os << "\n";
}

}  // namespace gfx
