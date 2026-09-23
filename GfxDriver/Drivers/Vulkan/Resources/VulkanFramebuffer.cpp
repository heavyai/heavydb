/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanFramebuffer.h"

#include "GfxDriver/Drivers/Vulkan/Commands/StagingContext.h"
#include "GfxDriver/Drivers/Vulkan/Resources/ImageLayoutManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanPixelBuffer2d.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Resources/AttachmentManager.h"

namespace gfx {

VulkanFramebuffer::VulkanFramebuffer(const VulkanDeviceContext& device_ctx,
                                     std::string_view resource_tracking_string,
                                     const RenderPass& render_pass,
                                     AttachmentManager& attachment_mgr,
                                     uint32_t width,
                                     uint32_t height,
                                     uint32_t num_samples)
    : Framebuffer{device_ctx,
                  resource_tracking_string,
                  render_pass,
                  attachment_mgr,
                  width,
                  height,
                  num_samples}
    , vk_device_{*const_cast<VulkanDeviceContext*>(&device_ctx)}
    , render_pass_{render_pass}
    , vk_framebuffer_{VK_NULL_HANDLE}
    , depth_image_and_format_{std::nullopt} {
  initResource();
}

VulkanFramebuffer::~VulkanFramebuffer() {
  destroyResource();
  cleanupResource();
}

void VulkanFramebuffer::initResource() {
  std::vector<VkImageView> image_views;
  auto const& all_attachments = attachment_mgr_.getAllAttachments();
  image_views.reserve(all_attachments.size());
  vk_color_images_.clear();
  for (auto const& attachment : all_attachments) {
    auto* vk_texture = static_cast<const VulkanTexture*>(
        attachment_mgr_.getAttachmentTexture(attachment));
    image_views.push_back(vk_texture->getImageView());
    // Stash VkImage handles for batch transitions
    if (AttachmentManager::isColorAttachment(attachment)) {
      vk_color_images_.push_back(vk_texture->getImage());
    } else {
      depth_image_and_format_ = {vk_texture->getImage(), vk_texture->getPixelFormat()};
    }
  }

  VkFramebufferCreateInfo framebuffer_ci = {};
  framebuffer_ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_ci.renderPass =
      reinterpret_cast<VkRenderPass>(render_pass_.getResourceHandle());
  framebuffer_ci.attachmentCount = static_cast<uint32_t>(image_views.size());
  framebuffer_ci.pAttachments = image_views.data();
  framebuffer_ci.width = width_;
  framebuffer_ci.height = height_;
  framebuffer_ci.layers = 1;

  auto result = vkCreateFramebuffer(
      vk_device_.getHandle(), &framebuffer_ci, nullptr, &vk_framebuffer_);
  CHECK_VKRESULT(result, "creating Framebuffer");

  // name it
  vk_device_.nameVulkanObject(
      VK_OBJECT_TYPE_FRAMEBUFFER, vk_framebuffer_, getTrackingData().origin);
}

void VulkanFramebuffer::destroyResource() {
  vk_color_images_.clear();
  depth_image_and_format_ = std::nullopt;
  if (vk_framebuffer_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(vk_device_.getHandle(), vk_framebuffer_, nullptr);
    vk_framebuffer_ = VK_NULL_HANDLE;
  }
}

void VulkanFramebuffer::resize(uint32_t width, uint32_t height) {
  // Ensure no device activity is pending
  vk_device_.waitIdle();

  // need to recreate the framebuffer since resize will invalidate all the imageviews
  destroyResource();
  width_ = width;
  height_ = height;
  auto const& all_attachments = attachment_mgr_.getAllAttachments();
  for (auto const& attachment : all_attachments) {
    attachment_mgr_.getAttachmentTexture(attachment)->resize(width, height, 1);
  }
  initResource();
}

void VulkanFramebuffer::cleanupResourceBase() {
  destroyResource();
  makeEmpty();
}

void VulkanFramebuffer::makeEmpty() {
  vk_framebuffer_ = VK_NULL_HANDLE;
}

void VulkanFramebuffer::transitionToImageLayout(VulkanCommandBuffer& cmd_buffer,
                                                const ImageLayout layout) {
  // Get src and dst pipeline stage flags
  // There are three cases we are currently concerned about for full Framebuffer
  // transitions:
  // 1/ moving from some previous step to Attachment layout for rendering
  // 2/ moving from either Attachment or TransferSrc to ShaderReadOnly (for
  // SeparateMultiSamplePass, both single-node and distributed)
  // 3/ moving from Attachment to General for MultiGpuCompositing ID images
  // Optimize stage flags for these cases with a fallback to ALL_COMMANDS which is
  // safe but a full stall, and a violation of best-practices validation
  VkPipelineStageFlags src_stage_flags{};
  VkPipelineStageFlags dst_stage_flags{};

  switch (layout) {
    case ImageLayout::kAttachment:
      src_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_TRANSFER_BIT |
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      if (depth_image_and_format_) {
        dst_stage_flags |= (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
      }
      break;
    case ImageLayout::kShaderReadOnly:
      src_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_TRANSFER_BIT |
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      break;
    case ImageLayout::kGeneral:
      src_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    default:
      src_stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  }

  static_cast<VulkanResourceManager*>(&vk_device_.getResourceManager())
      ->getImageLayoutManager()
      .transitionToLayout(vk_color_images_,
                          1,
                          depth_image_and_format_,
                          layout,
                          cmd_buffer,
                          src_stage_flags,
                          dst_stage_flags);
}

void VulkanFramebuffer::readPixels(const Attachment attachment,
                                   const uint32_t start_x,
                                   const uint32_t start_y,
                                   const uint32_t width,
                                   const uint32_t height,
                                   const PixelFormat pixel_format,
                                   void* data) {
  uint64_t pixel_data_size = width * height * pixelFormatDataSize(pixel_format);
  auto& staging = vk_device_.getStagingContext();
  auto* texture = attachment_mgr_.getAttachmentTexture(attachment);

  CHECK_EQ(start_x, 0U);
  CHECK_EQ(start_y, 0U);
  CHECK_LE(width, texture->getWidth());
  CHECK_LE(height, texture->getHeight());

  auto vk_image = static_cast<VulkanTexture*>(texture)->getImage();
  staging.getPixels(vk_image,
                    width,
                    height,
                    1,
                    pixel_format,
                    static_cast<std::byte*>(data),
                    pixel_data_size);
}

void VulkanFramebuffer::copyToFramebuffer(Framebuffer& dst_fbo,
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
                                          const SamplerFilterMode filter) {
  CHECK(AttachmentManager::isColorAttachment(src_attachment));
  CHECK(AttachmentManager::isColorAttachment(dst_attachment));

  auto* src_texture = attachment_mgr_.getAttachmentTexture(src_attachment);
  auto* dst_texture = dst_fbo.getAttachmentManager().getAttachmentTexture(dst_attachment);
  CHECK(src_texture) << "Invalid src texture";
  CHECK(src_texture) << "Invalid dst texture";

  auto src_pixel_format = src_texture->getPixelFormat();
  auto dst_pixel_format = dst_texture->getPixelFormat();
  CHECK_EQ(src_pixel_format, dst_pixel_format) << "Mismatched pixel formats";

  CHECK_EQ(src_x, 0U);
  CHECK_EQ(src_y, 0U);
  CHECK_EQ(dst_x, 0U);
  CHECK_EQ(dst_y, 0U);
  CHECK_EQ(src_width, dst_width);
  CHECK_EQ(src_height, dst_height);

  CHECK_EQ(dst_texture->getNumSamples(), 1U);

  CHECK_EQ(filter, SamplerFilterMode::kNearest);

  auto& staging = vk_device_.getStagingContext();
  staging.copyOrResolvePixels(static_cast<VulkanTexture*>(src_texture)->getImage(),
                              static_cast<VulkanTexture*>(dst_texture)->getImage(),
                              src_width,
                              src_height,
                              src_pixel_format,
                              src_texture->getNumSamples(),
                              true);
}

void VulkanFramebuffer::copyToFramebuffer(Framebuffer& dst_fbo,
                                          const std::vector<Attachment>& attachments,
                                          const uint32_t src_x,
                                          const uint32_t src_y,
                                          const uint32_t src_width,
                                          const uint32_t src_height,
                                          const uint32_t dst_x,
                                          const uint32_t dst_y,
                                          const uint32_t dst_width,
                                          const uint32_t dst_height,
                                          const bool do_async_copy) {
  CHECK_EQ(src_x, 0U);
  CHECK_EQ(src_y, 0U);
  CHECK_EQ(dst_x, 0U);
  CHECK_EQ(dst_y, 0U);
  CHECK_EQ(src_width, dst_width);
  CHECK_EQ(src_height, dst_height);

  auto& staging = vk_device_.getStagingContext();

  for (auto const& attachment : attachments) {
    CHECK(AttachmentManager::isColorAttachment(attachment));

    auto* src_texture = attachment_mgr_.getAttachmentTexture(attachment);
    auto* dst_texture = dst_fbo.getAttachmentManager().getAttachmentTexture(attachment);
    CHECK(src_texture) << "Invalid src texture";
    CHECK(dst_texture) << "Invalid dst texture";

    auto src_pixel_format = src_texture->getPixelFormat();
    auto dst_pixel_format = dst_texture->getPixelFormat();
    CHECK_EQ(src_pixel_format, dst_pixel_format) << "Mismatched pixel formats";

    CHECK_EQ(dst_texture->getNumSamples(), 1U);

    staging.copyOrResolvePixels(static_cast<VulkanTexture*>(src_texture)->getImage(),
                                static_cast<VulkanTexture*>(dst_texture)->getImage(),
                                src_width,
                                src_height,
                                src_pixel_format,
                                src_texture->getNumSamples(),
                                false);
  }

  if (!do_async_copy) {
    staging.waitForCompletion();
  }
}

void VulkanFramebuffer::copyToPixelBuffer(PixelBuffer2d& dst_pbo,
                                          const Attachment attachment,
                                          const uint32_t start_x,
                                          const uint32_t start_y,
                                          const uint32_t width,
                                          const uint32_t height,
                                          const uint64_t offset_bytes,
                                          const PixelFormat pixel_format) {
  const uint64_t buffer_num_bytes = dst_pbo.getNumBytes();
  const uint64_t image_num_bytes = width * height * pixelFormatDataSize(pixel_format);

  CHECK_EQ(start_x, 0U);
  CHECK_EQ(start_y, 0U);
  CHECK_EQ(offset_bytes, 0ULL);
  CHECK_EQ(buffer_num_bytes, image_num_bytes);

  CHECK(AttachmentManager::isColorAttachment(attachment));
  auto* texture = attachment_mgr_.getAttachmentTexture(attachment);

  // start the asynchronous image-to-buffer copy
  auto& staging = vk_device_.getStagingContext();
  staging.copyTextureToPixelBufferStart(
      *static_cast<VulkanTexture*>(texture), dst_pbo, width, height);
}

}  // namespace gfx
