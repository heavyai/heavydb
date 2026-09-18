/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/Utils/AttachmentUtils.h"

#include <sstream>

RenderTargetReturn build_attachments(gfx::ResourceManager& resource_mgr,
                                     const uint32_t width,
                                     const uint32_t height,
                                     const std::vector<RenderTargetDesc>& target_descs,
                                     const std::string& texture_name_base,
                                     const uint32_t num_samples,
                                     const gfx::ImageUsageBits extra_color_usage_bits) {
  gfx::AttachmentManager attachment_mgr;
  std::vector<gfx::resource_ptr<gfx::Texture>> textures;

  std::string name_base = texture_name_base.empty() ? "Framebuffer" : texture_name_base;
  for (auto const& [pixel_format, attachment] : target_descs) {
    gfx::ImageUsageBits extra_usage_bits =
        gfx::AttachmentManager::isColorAttachment(attachment)
            ? gfx::ImageUsageBits::kColorAttachmentBit | extra_color_usage_bits
            : gfx::ImageUsageBits::kDepthStencilAttachmentBit;
    std::stringstream ss;
    ss << name_base << " " << attachment;
    textures.emplace_back(resource_mgr.createTexture(
        ss.str(),
        width,
        height,
        1,
        pixel_format,
        num_samples,
        false,
        extra_usage_bits,
        gfx::get_default_sampler_state_for_format(pixel_format)));
    CHECK(textures.back());
    attachment_mgr.setAttachment(attachment, textures.back().get());
  }
  attachment_mgr.freezeLayout();
  return {attachment_mgr, std::move(textures)};
}
