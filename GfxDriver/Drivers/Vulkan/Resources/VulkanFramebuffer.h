/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/Framebuffer.h"

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/Resources/ImageLayoutManager.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Resources/PixelBuffer2d.h"

namespace gfx {
class VulkanFramebuffer : public Framebuffer {
 public:
  explicit VulkanFramebuffer(const VulkanDeviceContext& device_ctx,
                             std::string_view resource_tracking_string,
                             const RenderPass& render_pass,
                             AttachmentManager& attachment_mgr,
                             uint32_t width,
                             uint32_t height,
                             uint32_t num_samples);
  ~VulkanFramebuffer() override;
  VulkanFramebuffer() = delete;

  ResourceHandle getResourceHandle() const override {
    return reinterpret_cast<ResourceHandle>(vk_framebuffer_);
  }

  void resize(const uint32_t width, const uint32_t height) override;

  void activateEnabledAttachmentsForDrawing() override {}

  void transitionToImageLayout(VulkanCommandBuffer& cmd_buffer, const ImageLayout layout);

  void readPixels(const Attachment attachment,
                  const uint32_t start_x,
                  const uint32_t start_y,
                  const uint32_t width,
                  const uint32_t height,
                  const PixelFormat pixel_format,
                  void* data) override;

  void copyToFramebuffer(Framebuffer& dst_fbo,
                         const Attachment src_attachment,
                         const uint32_t src_x,
                         const uint32_t src_y,
                         const uint32_t src_width,
                         const uint32_t src_height,
                         const Attachment dst_attachment,
                         const uint32_t dst_x,
                         const uint32_t dst_y,
                         const uint32_t dst_width,
                         const uint32_t dst_height,
                         const SamplerFilterMode filter) override;

  void copyToFramebuffer(Framebuffer& dst_fbo,
                         const std::vector<Attachment>& attachments,
                         const uint32_t src_x,
                         const uint32_t src_y,
                         const uint32_t src_width,
                         const uint32_t src_height,
                         const uint32_t dst_x,
                         const uint32_t dst_y,
                         const uint32_t dst_width,
                         const uint32_t dst_height,
                         const bool do_async_copy) override;

  void copyToPixelBuffer(PixelBuffer2d& dst_pbo,
                         const Attachment attachment,
                         const uint32_t start_x,
                         const uint32_t start_y,
                         const uint32_t width,
                         const uint32_t height,
                         const uint64_t offset_bytes,
                         const PixelFormat pixel_format) override;

 private:
  VulkanDeviceContext& vk_device_;
  const RenderPass& render_pass_;

  VkFramebuffer vk_framebuffer_;
  std::vector<VkImage> vk_color_images_;
  std::optional<ImageLayoutManager::ImageAndFormat> depth_image_and_format_;

  void initResource() override;
  void destroyResource();
  void cleanupResourceBase() override;
  void makeEmpty() override;
};

}  // namespace gfx
