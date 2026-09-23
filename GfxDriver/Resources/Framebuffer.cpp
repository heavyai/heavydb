/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/Framebuffer.h"

#include "GfxDriver/Resources/AttachmentManager.h"

namespace gfx {

Framebuffer::Layout::Builder::Builder(uint32_t num_samples) : num_samples_{num_samples} {}
Framebuffer::Layout::Builder& Framebuffer::Layout::Builder::addAttachment(
    Attachment bind_point,
    PixelFormat pixel_format) {
  auto const& itr = attachment_map_.find(bind_point);
  CHECK(itr == attachment_map_.end());
  attachment_map_.try_emplace(bind_point, pixel_format);
  return *this;
}

void Framebuffer::Layout::Builder::buildLayout(Layout& layout) {
  uint32_t index{0};
  for (auto [binding, pixel_format] : attachment_map_) {
    layout.attachments_.push_back({binding, pixel_format, num_samples_});
    layout.indices_.try_emplace(binding, index++);
    layout.binding_vector_.push_back(binding);
    layout.binding_set_.insert(binding);
  }

  auto map_size = attachment_map_.size();
  CHECK_EQ(layout.attachments_.size(), map_size);
  CHECK_EQ(layout.indices_.size(), map_size);
  CHECK_EQ(layout.binding_vector_.size(), map_size);
  CHECK_EQ(layout.binding_set_.size(), map_size)
      << "Duplicate Framebuffer Layout bindings are not allowed";
}

void Framebuffer::Layout::build(Framebuffer::Layout::Builder&& builder) {
  CHECK(attachments_.empty()) << "Framebuffer::Layout must be cleared before rebuilding";
  builder.buildLayout(*this);
}

const std::vector<Framebuffer::Layout::AttachmentDesc>&
Framebuffer::Layout::getAttachmentDescs() const {
  return attachments_;
}

const Framebuffer::Layout::AttachmentDesc& Framebuffer::Layout::getAttachmentDesc(
    const Framebuffer::Attachment attachment) const {
  auto index = getIndex(attachment);
  CHECK_LT(index, attachments_.size());
  return attachments_[index];
}

const uint32_t Framebuffer::Layout::getIndex(
    const Framebuffer::Attachment attachment) const {
  auto const& itr = indices_.find(attachment);
  CHECK(itr != indices_.end());
  return itr->second;
}

Framebuffer::AttachmentVector Framebuffer::Layout::getAttachmentBindingVector() const {
  return binding_vector_;
}

Framebuffer::AttachmentSet Framebuffer::Layout::getAttachmentBindingSet() const {
  return binding_set_;
}

bool Framebuffer::Layout::is_empty() const {
  return attachments_.empty();
}

void Framebuffer::Layout::clear() {
  attachments_.clear();
}

Framebuffer::Framebuffer(const DeviceContext& device_context,
                         std::string_view resource_tracking_string,
                         const RenderPass& render_pass,
                         AttachmentManager& attachment_mgr,
                         uint32_t width,
                         uint32_t height,
                         uint32_t num_samples)
    : Resource(device_context, resource_tracking_string, ResourceType::kFramebuffer)
    , attachment_mgr_{attachment_mgr}
    , width_{width}
    , height_{height}
    , num_samples_{num_samples} {
  attachment_mgr_.freezeLayout();
  attachment_mgr_.validateAttachments(width_, height_, num_samples_);
}

Framebuffer::~Framebuffer() {}

std::string to_string(const Framebuffer::Attachment value) {
  switch (value) {
    case Framebuffer::Attachment::kDepth:
      return "Depth";
    case Framebuffer::Attachment::kStencil:
      return "Stencil";
    case Framebuffer::Attachment::kDepthStencil:
      return "Depth/Stencil";
    case Framebuffer::Attachment::kColor0:
      return "Color[0]";
    case Framebuffer::Attachment::kColor1:
      return "Color[1]";
    case Framebuffer::Attachment::kColor2:
      return "Color[2]";
    case Framebuffer::Attachment::kColor3:
      return "Color[3]";
    case Framebuffer::Attachment::kCOUNT:
      CHECK(false);
    default:
      CHECK(false);
  }
  return "";
}

}  // namespace gfx

std::ostream& operator<<(std::ostream& os, const gfx::Framebuffer::Attachment value) {
  os << gfx::to_string(value);
  return os;
}
