/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/ResourceManager.h"

using RenderTargetDesc = std::pair<gfx::PixelFormat, gfx::Framebuffer::Attachment>;
struct RenderTargetReturn {
  gfx::AttachmentManager attachment_mgr;
  std::vector<gfx::resource_ptr<gfx::Texture>> textures;
};

/**
 * Utility to build an attachment manager with associated textures.
 * The returned texture 2d resources need to be destroyed by the caller.
 * extra_color_usage_bits is useful for adding bits that are not supported
 * for depth attachments, such as kSampled
 */
RenderTargetReturn build_attachments(
    gfx::ResourceManager& resource_mgr,
    const uint32_t width,
    const uint32_t height,
    const std::vector<RenderTargetDesc>& target_descs,
    const std::string& texture_name_base = "",
    const uint32_t num_samples = 1,
    const gfx::ImageUsageBits extra_color_usage_bits = gfx::ImageUsageBits::kNone);
