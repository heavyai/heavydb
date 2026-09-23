/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>
#include <string_view>

#include <vulkan/vulkan.h>
#include <boost/noncopyable.hpp>

#include "GfxDriver/Drivers/Vulkan/Commands/FenceManager.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanQueue.h"

namespace gfx {

class VulkanCommandPool;

//
// VulkanCommandBuffer
//
class VulkanCommandBuffer : boost::noncopyable {
 public:
  enum State { kInitial, kRecording, kInRenderpass, kExecutable, kPending, kInvalid };

  explicit VulkanCommandBuffer(const VulkanDeviceContext& device_ctx,
                               VulkanCommandPool& owning_pool,
                               VkCommandBuffer handle,
                               uint32_t index);
  VulkanCommandBuffer() = delete;
  ~VulkanCommandBuffer();

  // RenderPasses
  void beginRenderPass(VkRenderPassBeginInfo* begin_info,
                       VkSubpassContents subpass_contents);
  void endRenderPass();

  // Debug labels
  void insertLabel(const std::string_view name);
  void pushLabel(const std::string_view name);
  void popLabel();

  // Internal state
  const State getState() const { return state_; }
  const VkCommandBuffer getHandle() const { return vk_command_buffer_; }
  const VulkanCommandPool& getOwningPool() const { return owning_pool_; }
  const VulkanQueue& getSubmitQueue() const { return *submit_queue_; }
  std::string_view getName() const { return name_; }

 private:
  const VulkanDeviceContext& device_ctx_;
  VulkanCommandPool& owning_pool_;
  VulkanQueue* submit_queue_;
  VkCommandBuffer vk_command_buffer_;
  uint32_t index_;
  State state_;
  Fence* fence_;
  std::string_view name_;

  VkResult reset();
  void setSubmitQueue(VulkanQueue* submit_queue);

  void beginRecording();
  void endRecordingAndSubmit(const std::string_view name);
  void endRecordingAndSubmit(const std::string_view name,
                             uint32_t wait_semaphore_count,
                             const VkSemaphore* wait_semaphores,
                             const std::vector<VkPipelineStageFlags2>& wait_stages,
                             uint32_t signal_semaphore_count,
                             const VkSemaphore* signal_semaphores,
                             const std::vector<VkPipelineStageFlags2>& signal_stages);

  State refreshFenceState();
  void waitForCompletion(uint64_t timeout_in_ms);

  friend class VulkanCommandPool;
};

//
// VulkanCommandPool
//
class VulkanCommandPool : boost::noncopyable {
 public:
  explicit VulkanCommandPool(VulkanDeviceContext& device_context,
                             VulkanQueue& queue,
                             VulkanDeviceContext::CommandPoolSelector selector,
                             uint32_t command_timeout_ms);
  VulkanCommandPool() = delete;
  ~VulkanCommandPool();

  // resetPool:
  // resets all buffers to kInitial, resets the pool itself, clearing storage on the
  // device, and resets pending and free lists. This must be called regularly to prevent
  // gpu memory accumulation. It is very cheap.
  void resetPool();

  // Acquire a command buffer for recording, either from the free list or
  // by allocating a new buffer
  // submit_queue family index must match the one used by createPool (e.g. graphics)
  // returned command buffer is set to kRecording state
  VulkanCommandBuffer* acquireBuffer();

  // Submit buffer to the queue specified by acquireBuffer. This will set the state to
  // kPending, making the buffer unusable until it completes and is reset
  void submitBuffer(VulkanCommandBuffer* cmd_buffer, const std::string_view name);

  // Submit buffer including optional wait and signal semaphores
  // Commands will hold until all wait semaphores are completed before execution starts
  // All signal semaphores will be set to the signalled state after commands complete
  // Raw pointers are used for semaphore vectors to allow casting a pointer from
  // gfx::SemaphoreHandle to VkSemaphore
  void submitBuffer(
      VulkanCommandBuffer* cmd_buffer,
      const std::string_view name,
      uint32_t wait_semaphore_count,
      const VkSemaphore* wait_semaphores,  // array of length wait_semaphore_count
      const std::vector<VkPipelineStageFlags2>& wait_stages,
      uint32_t signal_semaphore_count,
      const VkSemaphore* signal_semaphores,  // array of length signal_semaphore_count
      const std::vector<VkPipelineStageFlags2>& signal_stages);

  // Wait for any pending buffers for timeout milliseconds
  // If a callback function is provided, it will be called if there were any
  //  pending buffers, proving a hook for extended logging in the event
  //  of a VK_TIMEOUT or VK_DEVICE_LOST
  // Returns true if all pending buffers completed
  using WaitCallback = std::function<void(VkResult result)>;
  bool waitForPendingBuffers(WaitCallback callback = nullptr);

  VkCommandPool getHandle() const { return vk_command_pool_; }

  // Iterate command_buffers_ logging any that are pending
  // Order is not guaranteed to match submission order due to freelist usage
  void logPendingBufferNames(std::ostream& os) const;

 private:
  VulkanDeviceContext& device_ctx_;
  VkCommandPool vk_command_pool_;
  VulkanQueue& queue_;
  uint32_t command_timeout_ms_;

  std::vector<std::unique_ptr<VulkanCommandBuffer>> command_buffers_;
  std::vector<uint32_t> pending_buffer_ids_;
  std::vector<uint32_t> free_buffer_ids_;

  static bool is_device_lost_;

  VulkanCommandBuffer* addBuffer();
  void refreshFencesAndBufferIds();
  VkResult waitForPendingFences();
};

using VulkanCommandPoolUqPtr = std::unique_ptr<VulkanCommandPool>;

}  // namespace gfx
