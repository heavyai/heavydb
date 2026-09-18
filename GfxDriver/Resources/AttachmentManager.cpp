/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/AttachmentManager.h"

#include "GfxDriver/Resources/Texture.h"

namespace gfx {

/* static */ bool AttachmentManager::isColorAttachment(
    const Framebuffer::Attachment attachment) {
  return attachment == Framebuffer::Attachment::kColor0 ||
         attachment == Framebuffer::Attachment::kColor1 ||
         attachment == Framebuffer::Attachment::kColor2 ||
         attachment == Framebuffer::Attachment::kColor3;
}

AttachmentManager::AttachmentManager() : is_layout_frozen_{false} {}

bool AttachmentManager::hasAttachment(const Framebuffer::Attachment attachment) const {
  return (attachment_map_.find(attachment) != attachment_map_.end());
}

void AttachmentManager::setAttachment(const Framebuffer::Attachment attachment,
                                      Texture* texture) {
  CHECK(texture) << "Must be passed a texture!";
  if (is_layout_frozen_) {
    // find in the map
    auto map_itr = attachment_map_.find(attachment);
    auto layout_desc = layout_.getAttachmentDesc(attachment);
    CHECK(map_itr != attachment_map_.end()) << "Attachment not found!";
    // ensure new texture has the same format as the Layout's
    CHECK(layout_desc.format == texture->getPixelFormat())
        << "New texture format does not match Layout!";
    CHECK_EQ(layout_desc.num_samples, texture->getNumSamples())
        << "New texture has different samples from Layout!";
    // update texture
    map_itr->second.texture = texture;
  } else {
    // insert in map (must not already be present)
    CHECK(attachment_map_.try_emplace(attachment, texture, true).second)
        << "Attachment already exists!";
  }
}

void AttachmentManager::validateAttachments(uint32_t width,
                                            uint32_t height,
                                            uint32_t num_samples) {
  CHECK(is_layout_frozen_);
  for (auto const& elem : attachment_map_) {
    auto* texture = elem.second.texture;
    CHECK(texture);
    CHECK_EQ(texture->getWidth(), width);
    CHECK_EQ(texture->getHeight(), height);
    CHECK_EQ(texture->getNumSamples(), num_samples);
  }
}

void AttachmentManager::setEnabledAttachments(
    const Framebuffer::AttachmentVector& attachments) {
  CHECK(is_layout_frozen_);
  for (auto& elem : attachment_map_) {
    elem.second.active = false;
  }
  for (auto const& attachment : attachments) {
    auto itr = attachment_map_.find(attachment);
    CHECK(itr != attachment_map_.end());
    itr->second.active = true;
  }
}

Framebuffer::AttachmentVector AttachmentManager::getAllAttachments() const {
  CHECK(is_layout_frozen_);
  return layout_.getAttachmentBindingVector();
}

Framebuffer::AttachmentSet AttachmentManager::getAllAttachmentsSet() const {
  CHECK(is_layout_frozen_);
  return layout_.getAttachmentBindingSet();
}

Framebuffer::AttachmentVector AttachmentManager::getEnabledAttachments() const {
  Framebuffer::AttachmentVector rtn;
  CHECK(is_layout_frozen_);
  for (auto const& elem : attachment_map_) {
    if (elem.second.active) {
      rtn.push_back(elem.first);
    }
  }
  return rtn;
}

Framebuffer::AttachmentVector AttachmentManager::getEnabledColorAttachments() const {
  Framebuffer::AttachmentVector rtn;
  CHECK(is_layout_frozen_);
  for (auto const& elem : attachment_map_) {
    if (isColorAttachment(elem.first) && elem.second.active) {
      rtn.push_back(elem.first);
    }
  }
  return rtn;
}

Texture* AttachmentManager::getAttachmentTexture(
    const Framebuffer::Attachment attachment) const {
  CHECK(is_layout_frozen_);
  auto itr = attachment_map_.find(attachment);
  CHECK(itr != attachment_map_.end());
  return itr->second.texture;
}

void AttachmentManager::freezeLayout() {
  if (!is_layout_frozen_) {
    // Allow empty attachment_map_ for CommandTest mocking
    if (!attachment_map_.empty()) {
      // Grab num samples from the first texture as they all must match
      Framebuffer::Layout::Builder builder(
          attachment_map_.begin()->second.texture->getNumSamples());
      for (auto const& elem : attachment_map_) {
        builder.addAttachment(elem.first, elem.second.texture->getPixelFormat());
      }
      layout_.build(std::move(builder));
    }
    is_layout_frozen_ = true;
  }
}

const Framebuffer::Layout& AttachmentManager::getLayout() {
  freezeLayout();
  return layout_;
}

void AttachmentManager::clear() {
  attachment_map_.clear();
  layout_.clear();
  is_layout_frozen_ = false;
}

bool AttachmentManager::is_empty() const {
  return attachment_map_.empty();
}

}  // namespace gfx
