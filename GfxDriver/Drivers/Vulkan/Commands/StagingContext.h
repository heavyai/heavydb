/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include <vulkan/vulkan.h>
#include <boost/noncopyable.hpp>

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

class StagingContext : boost::noncopyable {
 public:
  explicit StagingContext(VulkanDeviceContext& device_context);
  StagingContext() = delete;
  ~StagingContext();

  void getPixels(VkImage src_image,
                 const uint32_t width,
                 const uint32_t height,
                 const uint32_t layer_count,
                 const PixelFormat pixel_format,
                 std::byte* pixel_data,
                 const uint64_t pixel_data_size,
                 const bool invert_y = false);

  // Destination image must not be multi-sampled
  void copyOrResolvePixels(
      const VkImage src_image,
      const VkImage dst_image,
      const uint32_t width,
      const uint32_t height,
      const PixelFormat pixel_format,
      const uint32_t src_num_samples,
      const bool wait_for_completion,
      const std::optional<SemaphoreHandle>& signal_semaphore = std::nullopt,
      const std::optional<ImageLayout>& dst_final_layout = std::nullopt);

  void copyTextureToPixelBufferStart(const VulkanTexture& texture,
                                     PixelBuffer2d& pbo,
                                     const uint32_t width,
                                     const uint32_t height);

  void copyTextureToPixelBufferFinish(PixelBuffer2d& pbo);

  void getBufferData(const Buffer& src_buffer, void* dst_data, const uint64_t num_bytes);

  struct LockedStagingBuffer {
    void* buffer;
    std::unique_lock<std::mutex> lock;
  };

  // Get a host visible staging buffer that can be mapped
  // The returned object contains a lock which ensures that only
  // one buffer can be acquired from this context at a time
  [[nodiscard]] LockedStagingBuffer acquireStagingBuffer(uint64_t size);

  // Release the staging buffer, automatically copying the data to
  // the destination buffer/image, which must be device local
  // pass nullptr to skip copy
  void releaseStagingBuffer(LockedStagingBuffer&& locked_staging_buffer,
                            Buffer* dest_buffer,
                            uint64_t dst_offset);
  void releaseStagingBuffer(LockedStagingBuffer&& locked_staging_buffer,
                            Texture* dest_texture);

  void waitForCompletion();

 private:
  enum class SubmitQueue { kGraphics, kTransfer };
  enum class StagingBuffer { kSmall, kDynamic };

  VulkanDeviceContext& device_context_;
  VulkanCommandPool& graphics_command_pool_;
  VulkanCommandPool& transfer_command_pool_;
  HostVisibleBufferWrapperUqPtr staging_buffer_dynamic_;
  HostVisibleBufferWrapperUqPtr staging_buffer_small_;
  void* staging_buffer_small_mapped_ptr_;
  StagingBuffer active_buffer_;
  uint64_t acquire_size_;
  std::mutex staging_buffer_mutex_;

  VulkanCommandBuffer* beginCommandRecording(SubmitQueue submit_queue);
  void submitCommandSequence(
      VulkanCommandBuffer* cmd_buffer,
      SubmitQueue submit_queue,
      std::string_view submit_name,
      bool do_wait_complete,
      std::optional<SemaphoreHandle> signal_semaphore = std::nullopt);
  void waitForCommandSequence(SubmitQueue submit_queue);

  // Caller must perform range checking to ensure source and dest can accomodate size
  void copyBuffer(VkBuffer source,
                  VkBuffer dest,
                  VkDeviceSize size,
                  VkDeviceSize dst_offset);

  void releaseStagingBuffer(VkImage image,
                            const uint32_t width,
                            const uint32_t height,
                            const uint32_t layer_count,
                            ImageLayout final_layout);

  void releaseActiveBuffer(LockedStagingBuffer&& locked_staging_buffer);
  HostVisibleBufferWrapper* getActiveBuffer() const;
};

}  // namespace gfx
