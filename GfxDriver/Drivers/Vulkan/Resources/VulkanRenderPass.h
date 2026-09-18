/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/RenderPass.h"

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"

namespace gfx {

class VulkanRenderPass : public RenderPass {
 public:
  explicit VulkanRenderPass(const DeviceContext& device_ctx,
                            std::string_view resource_tracking_string,
                            const Framebuffer::Layout& framebuffer_layout,
                            ClearBits clear_bits,
                            ImageLayout initial_layout,
                            ImageLayout final_layout,
                            const std::vector<SubpassDescriptor>& subpass_descriptors,
                            const AttachmentToImageLayoutMap& unused_attachment_layouts);

  ~VulkanRenderPass() override;

  void updateImageLayouts(Framebuffer& framebuffer) const override;

  const std::vector<VkClearValue>& getClearValues() const { return vk_clear_values_; }

  uint32_t getNumSubpasses() const override { return num_subpasses_; }
  uint32_t getSubpassColorAttachmentCount(uint32_t subpass_index) const;
  bool subpassColorAttachmentIsBlendable(uint32_t subpass_index,
                                         uint32_t color_attachment_index) const;

  void clearAttachment(Framebuffer& framebuffer,
                       Framebuffer::Attachment attachment,
                       VulkanCommandBuffer& cmd_buffer,
                       const VkRect2D& render_area,
                       uint32_t subpass_index = 0);

  ResourceHandle getResourceHandle() const override {
    return reinterpret_cast<ResourceHandle>(vk_render_pass_);
  }

 private:
  VkRenderPass vk_render_pass_;
  ImageLayout final_layout_;
  uint32_t num_subpasses_;
  std::vector<uint32_t> subpass_color_attachment_counts_;
  std::vector<std::vector<bool>> subpass_color_attachments_blendable_;
  std::vector<VkClearValue> vk_clear_values_;

  void makeEmpty() override;
  void cleanupResourceBase() override;
};

}  // namespace gfx
