/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/Resource.h"

#include <map>

#include "GfxDriver/Resources/Framebuffer.h"
#include "GfxDriver/Types.h"
#include "Shared/EnumBitmaskOps.h"

namespace gfx {

enum class SubpassDependencyBits {
  kNone = 0,
  kFragmentShaderRead = 1 << 0,
  kFragmentShaderWrite = 1 << 1
};

struct SubpassDescriptor {
  Framebuffer::AttachmentSet attachments;
  Framebuffer::AttachmentSet input_attachments;
  SubpassDependencyBits dependencies = SubpassDependencyBits::kNone;
};

using AttachmentToImageLayoutMap = std::map<Framebuffer::Attachment, ImageLayout>;

class RenderPass : public Resource {
 public:
  enum class ClearBits {
    kNone = 0,
    kColor = 1 << 0,
    kDepth = 1 << 1,
    kStencil = 1 << 2,
    kDepthStencil = kDepth | kStencil,
    kAll = 0xffff
  };

  explicit RenderPass(const DeviceContext& device_ctx,
                      std::string_view resource_tracking_string);
  ~RenderPass() override = default;

  virtual uint32_t getNumSubpasses() const = 0;

  virtual void updateImageLayouts(Framebuffer& framebuffer) const {}
};

}  // namespace gfx

ENABLE_BITMASK_OPS(::gfx::RenderPass::ClearBits);
ENABLE_BITMASK_OPS(::gfx::SubpassDependencyBits);
