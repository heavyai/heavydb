/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Commands/StagingContext.h"

#include "GfxDriver/Drivers/Vulkan/Resources/ImageLayoutManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanBaseBuffer.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanPixelBuffer2d.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanMemoryMgr.h"
#include "Shared/scope.h"

namespace gfx {

static constexpr uint64_t kSmallStagingBufferSize = 256000;

StagingContext::StagingContext(VulkanDeviceContext& device_context)
    : device_context_{device_context}
    , graphics_command_pool_{device_context_.getCommandPool(
          VulkanDeviceContext::CommandPoolSelector::kStagingGraphics)}
    , transfer_command_pool_{device_context_.getCommandPool(
          VulkanDeviceContext::CommandPoolSelector::kStagingTransfer)}
    , staging_buffer_small_mapped_ptr_{nullptr}
    , active_buffer_{StagingBuffer::kSmall}
    , acquire_size_{0} {
  staging_buffer_small_ = device_context_.getResourceManager().createHostVisibleBuffer(
      "Small staging buffer", {BufferType::kUnspecified, kSmallStagingBufferSize});
  staging_buffer_small_mapped_ptr_ = staging_buffer_small_->map();
}

StagingContext::~StagingContext() {
  // Cleanup an dangling command buffers, generally from async ID readback in
  // VulkanPixelBuffer
  graphics_command_pool_.waitForPendingBuffers();
  transfer_command_pool_.waitForPendingBuffers();
  auto& resource_mgr = device_context_.getResourceManager();
  if (staging_buffer_dynamic_) {
    resource_mgr.destroyHostVisibleBuffer(std::move(staging_buffer_dynamic_));
  }
  if (staging_buffer_small_) {
    if (staging_buffer_small_->isMapped()) {
      staging_buffer_small_->unmap();
    }
    resource_mgr.destroyHostVisibleBuffer(std::move(staging_buffer_small_));
  }
}

void StagingContext::getPixels(VkImage src_image,
                               const uint32_t width,
                               const uint32_t height,
                               const uint32_t layer_count,
                               const PixelFormat pixel_format,
                               std::byte* pixel_data,
                               const uint64_t pixel_data_size,
                               const bool invert_y) {
  auto& resource_mgr = device_context_.getResourceManager();
  auto& image_layout_mgr =
      static_cast<VulkanResourceManager*>(&resource_mgr)->getImageLayoutManager();

  // create temp buffer
  auto temp_buffer =
      resource_mgr.createBaseBuffer("Staging Context getPixels Temp",
                                    {BufferType::kUnspecified,
                                     pixel_data_size,
                                     BufferUsageBits::kNone,
                                     BufferAccessType::kHostVisibleCached});

  // destroy buffer on exit
  ScopeGuard destroy_temp_buffer = [&]() {
    resource_mgr.destroyBaseBuffer(std::move(temp_buffer));
  };

  // start recording
  auto* cmd_buffer = beginCommandRecording(SubmitQueue::kTransfer);

  auto image_aspect = pixel_format_to_vk_image_aspect_flags(pixel_format);
  VkImageSubresourceRange subresource_range{image_aspect, 0, 1, 0, layer_count};

  // if the image is in General, leave it alone
  // if not, transition it to TransferSrc
  // @TODO(se/scb) ensure access safety if we skip the barrier here
  auto const src_layout = image_layout_mgr.getCurrentLayout(src_image);
  if (src_layout != ImageLayout::kGeneral && src_layout != ImageLayout::kTransferSrc) {
    image_layout_mgr.transitionToLayout(
        src_image,
        ImageLayout::kTransferSrc,
        *cmd_buffer,
        get_stage_mask_for_layout(src_layout, is_color_pixel_format(pixel_format)),
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        subresource_range);
  }

  // the region to copy
  VkBufferImageCopy copy_region{};
  // If the image is depth+stencil, we can only copy one aspect at a time. Currently only
  // depth is supported so mask off the stencil bit
  copy_region.imageSubresource.aspectMask = image_aspect & (~VK_IMAGE_ASPECT_STENCIL_BIT);
  copy_region.imageSubresource.layerCount = layer_count;
  copy_region.imageExtent.width = width;
  copy_region.imageExtent.height = height;
  copy_region.imageExtent.depth = 1;

  // copy to buffer
  auto vk_temp_buffer = reinterpret_cast<VkBuffer>(temp_buffer->getResourceHandle());
  auto const layout_for_copy = image_layout_to_vk_image_layout(
      image_layout_mgr.getCurrentLayout(src_image), false);
  vkCmdCopyImageToBuffer(cmd_buffer->getHandle(),
                         src_image,
                         layout_for_copy,
                         vk_temp_buffer,
                         1,
                         &copy_region);

  // end recording, submit and wait
  submitCommandSequence(cmd_buffer, SubmitQueue::kTransfer, "Staging get pixels", true);

  if (invert_y) {
    // extract to a temp CPU buffer
    std::vector<std::byte> temp_pixel_data(pixel_data_size);
    temp_buffer->getData(temp_pixel_data.data(), pixel_data_size);

    // copy to the output, inverting y
    const uint32_t row_data_size = width * pixelFormatDataSize(pixel_format);
    const uint32_t layer_data_size = row_data_size * height;
    for (uint32_t l = 0; l < layer_count; l++) {
      std::byte* src_layer = temp_pixel_data.data() + (layer_data_size * l);
      std::byte* dst_layer = pixel_data + (layer_data_size * l);
      for (uint32_t y = 0, inv_y = height - 1; y < height; y++, inv_y--) {
        std::byte* src_row = src_layer + (row_data_size * y);
        std::byte* dst_row = dst_layer + (row_data_size * inv_y);
        std::memcpy(dst_row, src_row, row_data_size);
      }
    }
  } else {
    // extract the image data directly from the temp buffer
    temp_buffer->getData(pixel_data, pixel_data_size);
  }
}

void StagingContext::copyOrResolvePixels(
    const VkImage src_image,
    const VkImage dst_image,
    const uint32_t width,
    const uint32_t height,
    const PixelFormat pixel_format,
    const uint32_t src_num_samples,
    const bool wait_for_completion,
    const std::optional<SemaphoreHandle>& signal_semaphore,
    const std::optional<ImageLayout>& dst_final_layout) {
  static constexpr uint32_t layer_count = 1;

  VkImageSubresourceRange subresource_range{
      pixel_format_to_vk_image_aspect_flags(pixel_format), 0, 1, 0, layer_count};

  auto* cmd_buffer = beginCommandRecording(SubmitQueue::kGraphics);

  auto& image_layout_mgr =
      static_cast<VulkanResourceManager*>(&device_context_.getResourceManager())
          ->getImageLayoutManager();

  // transition source image to "transfer-source"
  image_layout_mgr.transitionToLayout(src_image,
                                      ImageLayout::kTransferSrc,
                                      *cmd_buffer,
                                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      subresource_range);

  // Transition destination image to "transfer-destination"
  auto const dst_current_layout = image_layout_mgr.getCurrentLayout(dst_image);
  auto is_color_format = is_color_pixel_format(pixel_format);
  auto dst_stage_mask = get_stage_mask_for_layout(dst_current_layout, is_color_format);

  image_layout_mgr.transitionToLayout(dst_image,
                                      ImageLayout::kTransferDst,
                                      *cmd_buffer,
                                      dst_stage_mask,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      subresource_range);

  if (src_num_samples > 1) {
    VkImageResolve resolve_region{};
    resolve_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    resolve_region.srcSubresource.layerCount = layer_count;
    resolve_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    resolve_region.dstSubresource.layerCount = layer_count;
    resolve_region.extent.width = width;
    resolve_region.extent.height = height;
    resolve_region.extent.depth = 1;

    vkCmdResolveImage(cmd_buffer->getHandle(),
                      src_image,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      dst_image,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      1,
                      &resolve_region);
  } else {
    VkImageCopy copy_region{};
    copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.srcSubresource.layerCount = layer_count;
    copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.dstSubresource.layerCount = layer_count;
    copy_region.extent.width = width;
    copy_region.extent.height = height;
    copy_region.extent.depth = 1;

    vkCmdCopyImage(cmd_buffer->getHandle(),
                   src_image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst_image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1,
                   &copy_region);
  }

  if (dst_final_layout) {
    image_layout_mgr.transitionToLayout(
        dst_image,
        *dst_final_layout,
        *cmd_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        get_stage_mask_for_layout(*dst_final_layout, is_color_format),
        subresource_range);
  }

  // Make it so!
  submitCommandSequence(cmd_buffer,
                        SubmitQueue::kGraphics,
                        "Staging copy or resolve",
                        wait_for_completion,
                        signal_semaphore);
}

void StagingContext::copyTextureToPixelBufferStart(const VulkanTexture& texture,
                                                   PixelBuffer2d& pbo,
                                                   const uint32_t width,
                                                   const uint32_t height) {
  auto image = texture.getImage();
  auto buffer = reinterpret_cast<VkBuffer>(pbo.getResourceHandle());

  static constexpr uint32_t layer_count = 1;

  VkImageSubresourceRange subresource_range{
      pixel_format_to_vk_image_aspect_flags(texture.getPixelFormat()),
      0,
      1,
      0,
      layer_count};

  auto* cmd_buffer = beginCommandRecording(SubmitQueue::kTransfer);

  auto& image_layout_mgr =
      static_cast<VulkanResourceManager*>(&device_context_.getResourceManager())
          ->getImageLayoutManager();

  // transition source image to "transfer-source"
  image_layout_mgr.transitionToLayout(image,
                                      ImageLayout::kTransferSrc,
                                      *cmd_buffer,
                                      std::nullopt,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      subresource_range);

  VkBufferImageCopy copy_region{};
  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.layerCount = layer_count;
  copy_region.imageExtent.width = width;
  copy_region.imageExtent.height = height;
  copy_region.imageExtent.depth = 1;

  vkCmdCopyImageToBuffer(cmd_buffer->getHandle(),
                         image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         buffer,
                         1,
                         &copy_region);

  // submit but do not wait
  submitCommandSequence(cmd_buffer, SubmitQueue::kTransfer, "Staging copy to pbo", true);

  // store the command buffer handle in the PBO
  auto& vk_pbo = static_cast<VulkanPixelBuffer2d&>(pbo);
  vk_pbo.setCommandBuffer(cmd_buffer);
}

void StagingContext::copyTextureToPixelBufferFinish(PixelBuffer2d& pbo) {
  // get the command buffer handle from the PBO
  auto& vk_pbo = static_cast<VulkanPixelBuffer2d&>(pbo);
  auto* cmd_buffer = vk_pbo.releaseCommandBuffer();
  if (cmd_buffer) {
    // ensure the command sequence has completed
    waitForCommandSequence(SubmitQueue::kTransfer);
  }
}

void StagingContext::getBufferData(const Buffer& src_buffer,
                                   void* dst_data,
                                   const uint64_t num_bytes) {
  auto locked_staging_buffer = acquireStagingBuffer(num_bytes);
  auto* buffer_data = locked_staging_buffer.buffer;
  CHECK(buffer_data);

  ScopeGuard release_staging_buffer = [&] {
    releaseActiveBuffer(std::move(locked_staging_buffer));
  };

  copyBuffer(reinterpret_cast<VkBuffer>(src_buffer.getResourceHandle()),
             reinterpret_cast<VkBuffer>(
                 getActiveBuffer()->getSourceBufferWrapper().getResourceHandle()),
             num_bytes,
             0);

  // copy data
  std::memcpy(dst_data, buffer_data, num_bytes);
}

StagingContext::LockedStagingBuffer StagingContext::acquireStagingBuffer(uint64_t size) {
  // take the lock immediately
  // other threads will be blocked here
  std::unique_lock<std::mutex> lock(staging_buffer_mutex_);

  CHECK_EQ(acquire_size_, 0u)
      << "Attempted to acquire staging buffer when already in use";
  CHECK_GT(size, 0u);
  acquire_size_ = size;
  if (size < kSmallStagingBufferSize) {
    active_buffer_ = StagingBuffer::kSmall;
    return {staging_buffer_small_mapped_ptr_, std::move(lock)};
  }

  active_buffer_ = StagingBuffer::kDynamic;

  CHECK(staging_buffer_dynamic_ == nullptr);
  auto& resource_mgr = device_context_.getResourceManager();
  // we require a host visible coherent buffer for staging purposes
  staging_buffer_dynamic_ = resource_mgr.createHostVisibleBuffer(
      "Dynamic staging buffer", {BufferType::kUnspecified, size});
  return {staging_buffer_dynamic_->map(), std::move(lock)};
}

void StagingContext::releaseActiveBuffer(LockedStagingBuffer&& locked_staging_buffer) {
  if (active_buffer_ == StagingBuffer::kDynamic && staging_buffer_dynamic_) {
    if (staging_buffer_dynamic_->isMapped()) {
      staging_buffer_dynamic_->unmap();
    }
    device_context_.getResourceManager().destroyHostVisibleBuffer(
        std::move(staging_buffer_dynamic_));
  }
  acquire_size_ = 0;
  // lock will be released automatically
}

HostVisibleBufferWrapper* StagingContext::getActiveBuffer() const {
  CHECK_NE(acquire_size_, 0u) << "No active staging buffer";
  auto* rtn = active_buffer_ == StagingBuffer::kDynamic ? staging_buffer_dynamic_.get()
                                                        : staging_buffer_small_.get();
  CHECK(rtn);
  return rtn;
}

void StagingContext::releaseStagingBuffer(LockedStagingBuffer&& locked_staging_buffer,
                                          Buffer* dest_buffer,
                                          uint64_t dst_offset) {
  auto* staging_buffer = getActiveBuffer();
  ScopeGuard release_staging_buffer = [&] {
    releaseActiveBuffer(std::move(locked_staging_buffer));
  };
  if (dest_buffer != nullptr) {
    CHECK(buffer_access_type_to_memory_properties_bits(dest_buffer->getAccessType()) &
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto src_vk_buffer = reinterpret_cast<VkBuffer>(
        staging_buffer->getSourceBufferWrapper().getResourceHandle());
    auto dest_vk_buffer = reinterpret_cast<VkBuffer>(dest_buffer->getResourceHandle());
    // Ensure target buffer is large enough
    RUNTIME_EX_ASSERT(dest_buffer->getNumBytes() >= acquire_size_,
                      "Unable to copy staging buffer of " +
                          std::to_string(acquire_size_) +
                          " bytes to device buffer, target buffer is only " +
                          std::to_string(dest_buffer->getNumBytes()) + " bytes");
    copyBuffer(src_vk_buffer, dest_vk_buffer, acquire_size_, dst_offset);
  }
}

void StagingContext::releaseStagingBuffer(LockedStagingBuffer&& locked_staging_buffer,
                                          Texture* dest_texture) {
  ScopeGuard release_staging_buffer = [&] {
    releaseActiveBuffer(std::move(locked_staging_buffer));
  };

  if (dest_texture != nullptr) {
    // Ensure image and buffer sizes match
    size_t dest_size = dest_texture->getWidth() * dest_texture->getHeight() *
                       dest_texture->getDepth() *
                       pixelFormatDataSize(dest_texture->getPixelFormat());
    // Ensure target buffer is large enough
    RUNTIME_EX_ASSERT(dest_size >= acquire_size_,
                      "Unable to copy staging buffer of " +
                          std::to_string(acquire_size_) +
                          " bytes to device buffer, target buffer is only " +
                          std::to_string(dest_size) + " bytes");

    auto const* vk_dest_texture = static_cast<VulkanTexture*>(dest_texture);

    auto const final_layout =
        image_usage_bits_to_final_layout(vk_dest_texture->getUsageBits());

    releaseStagingBuffer(vk_dest_texture->getImage(),
                         dest_texture->getWidth(),
                         dest_texture->getHeight(),
                         dest_texture->getDepth(),
                         final_layout);
  }
}

void StagingContext::releaseStagingBuffer(VkImage image,
                                          const uint32_t width,
                                          const uint32_t height,
                                          const uint32_t layer_count,
                                          ImageLayout final_layout) {
  auto* cmd_buffer = beginCommandRecording(SubmitQueue::kTransfer);

  VkImageSubresourceRange subresource_range{
      VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layer_count};

  auto& image_layout_mgr =
      static_cast<VulkanResourceManager*>(&device_context_.getResourceManager())
          ->getImageLayoutManager();

  // transition image to transfer-destination
  image_layout_mgr.transitionToLayout(image,
                                      ImageLayout::kTransferDst,
                                      *cmd_buffer,
                                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      subresource_range);

  VkBufferImageCopy copy_region = {};

  copy_region.bufferOffset = 0;
  copy_region.bufferRowLength = 0;
  copy_region.bufferImageHeight = 0;

  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.mipLevel = 0;
  copy_region.imageSubresource.baseArrayLayer = 0;
  copy_region.imageSubresource.layerCount = layer_count;

  copy_region.imageOffset = {0, 0, 0};
  copy_region.imageExtent = {width, height, 1};

  auto src_vk_buffer = reinterpret_cast<VkBuffer>(
      getActiveBuffer()->getSourceBufferWrapper().getResourceHandle());
  vkCmdCopyBufferToImage(cmd_buffer->getHandle(),
                         src_vk_buffer,
                         image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         1,
                         &copy_region);

  // transition to requested final layout
  image_layout_mgr.transitionToLayout(image,
                                      final_layout,
                                      *cmd_buffer,
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                      subresource_range);

  // submit and wait
  submitCommandSequence(
      cmd_buffer, SubmitQueue::kTransfer, "Staging release buffer", true);
}

VulkanCommandBuffer* StagingContext::beginCommandRecording(
    const SubmitQueue submit_queue) {
  gfx::VulkanCommandBuffer* cmd_buffer = nullptr;
  switch (submit_queue) {
    case SubmitQueue::kGraphics:
      cmd_buffer = graphics_command_pool_.acquireBuffer();
      break;
    case SubmitQueue::kTransfer:
      cmd_buffer = transfer_command_pool_.acquireBuffer();
      break;
  }
  return cmd_buffer;
}

void StagingContext::submitCommandSequence(
    VulkanCommandBuffer* cmd_buffer,
    SubmitQueue submit_queue,
    std::string_view submit_name,
    bool do_wait_complete,
    std::optional<SemaphoreHandle> signal_semaphore) {
  VulkanCommandPool& pool_to_use = submit_queue == SubmitQueue::kGraphics
                                       ? graphics_command_pool_
                                       : transfer_command_pool_;
  if (signal_semaphore) {
    auto vk_semaphore = reinterpret_cast<VkSemaphore>(*signal_semaphore);
    pool_to_use.submitBuffer(cmd_buffer,
                             submit_name,
                             0,
                             nullptr,
                             {},
                             1,
                             &vk_semaphore,
                             {VK_PIPELINE_STAGE_2_TRANSFER_BIT});
  } else {
    pool_to_use.submitBuffer(cmd_buffer, submit_name);
  }
  if (do_wait_complete) {
    bool result = pool_to_use.waitForPendingBuffers();
    RUNTIME_EX_ASSERT(result, "Staging context command buffer timeout");
  }
}

void StagingContext::waitForCommandSequence(SubmitQueue submit_queue) {
  VulkanCommandPool& pool_to_use = submit_queue == SubmitQueue::kGraphics
                                       ? graphics_command_pool_
                                       : transfer_command_pool_;
  bool result = pool_to_use.waitForPendingBuffers();
  RUNTIME_EX_ASSERT(result, "Staging context command buffer timeout");
}

void StagingContext::waitForCompletion() {
  bool result = graphics_command_pool_.waitForPendingBuffers();
  result = result && transfer_command_pool_.waitForPendingBuffers();
  RUNTIME_EX_ASSERT(result, "Staging context command buffer timeout");
}

void StagingContext::copyBuffer(VkBuffer src_buffer,
                                VkBuffer dest_buffer,
                                VkDeviceSize size,
                                VkDeviceSize dst_offset) {
  auto* cmd_buffer = beginCommandRecording(SubmitQueue::kTransfer);

  VkBufferCopy copy_region = {};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = dst_offset;
  copy_region.size = size;

  vkCmdCopyBuffer(cmd_buffer->getHandle(), src_buffer, dest_buffer, 1, &copy_region);

  // TODO(scb): memory barrier or semaphore instead of Fence?
  // TODO(scb): configurable wait option?

  submitCommandSequence(cmd_buffer, SubmitQueue::kTransfer, "Staging copy buffer", true);
}

}  // namespace gfx
