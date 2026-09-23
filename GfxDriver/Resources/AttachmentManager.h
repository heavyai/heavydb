/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>

#include "GfxDriver/Resources/Framebuffer.h"

namespace gfx {

/**
 * AttachmentManager
 *  Class used to manage attachments of a framebuffer
 */
class AttachmentManager {
 public:
  // @TODO(se) enforce minimum MAX_COLOR_ATTACHMENTS in DriverInstance
  AttachmentManager();
  ~AttachmentManager() = default;

  bool hasAttachment(const Framebuffer::Attachment attachment) const;
  void setAttachment(const Framebuffer::Attachment attachment, Texture* texturePtr);

  static bool isColorAttachment(const Framebuffer::Attachment attachment);

  void validateAttachments(uint32_t width, uint32_t height, uint32_t num_samples);

  void setEnabledAttachments(const Framebuffer::AttachmentVector& attachments);

  Framebuffer::AttachmentVector getAllAttachments() const;
  Framebuffer::AttachmentSet getAllAttachmentsSet() const;

  Framebuffer::AttachmentVector getEnabledAttachments() const;
  Framebuffer::AttachmentVector getEnabledColorAttachments() const;

  Texture* getAttachmentTexture(const Framebuffer::Attachment attachment) const;

  const Framebuffer::Layout& getLayout();
  void freezeLayout();

  void clear();
  bool is_empty() const;

 private:
  struct AttachmentData {
    AttachmentData(Texture* texture, bool active) : texture{texture}, active{active} {}
    Texture* texture = nullptr;
    bool active = false;
  };

  // ordered map so that "get" iterations are in attachment enum order
  std::map<Framebuffer::Attachment, AttachmentData> attachment_map_;

  bool is_layout_frozen_;
  Framebuffer::Layout layout_;
};

}  // namespace gfx
