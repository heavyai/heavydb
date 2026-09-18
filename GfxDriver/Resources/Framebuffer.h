/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <set>
#include <vector>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/PixelBuffer2d.h"
#include "GfxDriver/Resources/Resource.h"
#include "GfxDriver/Types.h"

namespace gfx {

class AttachmentManager;

class Framebuffer : public Resource {
 public:
  enum class Attachment : uint8_t {
    kDepth,
    kStencil,
    kDepthStencil,
    kColor0,
    kColor1,
    kColor2,
    kColor3,
    kCOUNT
  };

  using AttachmentVector = std::vector<Attachment>;
  using AttachmentSet = std::set<Attachment>;

  //
  // Framebuffer::Layout
  //
  class Layout {
   public:
    struct AttachmentDesc {
      Framebuffer::Attachment bind_point;
      PixelFormat format;
      uint32_t num_samples;
    };

    //
    // Framebuffer::Layout::Builder
    //
    class Builder {
     public:
      explicit Builder(uint32_t num_samples);
      Builder& addAttachment(Framebuffer::Attachment bind_point,
                             PixelFormat pixel_format);

     private:
      uint32_t num_samples_;
      std::map<Framebuffer::Attachment, PixelFormat> attachment_map_;
      void buildLayout(Layout& layout);
      friend class Layout;
    };

    // Consume a builder, populating the internal Layout data
    void build(Builder&& builder);

    const std::vector<AttachmentDesc>& getAttachmentDescs() const;
    const AttachmentDesc& getAttachmentDesc(
        const Framebuffer::Attachment attachment) const;
    const uint32_t getIndex(const Framebuffer::Attachment attachment) const;
    AttachmentVector getAttachmentBindingVector() const;
    AttachmentSet getAttachmentBindingSet() const;

    bool is_empty() const;
    void clear();

   private:
    std::vector<AttachmentDesc> attachments_;
    std::map<Framebuffer::Attachment, uint32_t> indices_;
    AttachmentVector binding_vector_;
    AttachmentSet binding_set_;
  };

  //
  // Framebuffer
  //
  explicit Framebuffer(const DeviceContext& device_ctx,
                       std::string_view resource_tracking_string,
                       const RenderPass& render_pass,
                       AttachmentManager& attachment_mgr,
                       uint32_t width,
                       uint32_t height,
                       uint32_t num_samples);
  ~Framebuffer() override;
  Framebuffer() = delete;

  uint32_t getWidth() const { return width_; }
  uint32_t getHeight() const { return height_; }
  uint32_t getNumSamples() const { return num_samples_; }

  AttachmentManager& getAttachmentManager() const { return attachment_mgr_; }

  virtual void readPixels(const Attachment attachment,
                          const uint32_t start_x,
                          const uint32_t start_y,
                          const uint32_t width,
                          const uint32_t height,
                          const PixelFormat pixel_format,
                          void* data) = 0;

  virtual void copyToFramebuffer(Framebuffer& dst_fbo,
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
                                 const SamplerFilterMode filter) = 0;

  virtual void copyToFramebuffer(Framebuffer& dst_fbo,
                                 const std::vector<Attachment>& attachments,
                                 const uint32_t src_x,
                                 const uint32_t src_y,
                                 const uint32_t src_width,
                                 const uint32_t src_height,
                                 const uint32_t dst_x,
                                 const uint32_t dst_y,
                                 const uint32_t dst_width,
                                 const uint32_t dst_height,
                                 const bool do_async_copy) = 0;

  virtual void copyToPixelBuffer(PixelBuffer2d& dst_pbo,
                                 const Attachment attachment,
                                 const uint32_t start_x,
                                 const uint32_t start_y,
                                 const uint32_t width,
                                 const uint32_t height,
                                 const uint64_t offset_bytes,
                                 const PixelFormat pixel_format) = 0;

  virtual void resize(const uint32_t width, const uint32_t height) = 0;

  virtual void activateEnabledAttachmentsForDrawing() = 0;

 protected:
  AttachmentManager& attachment_mgr_;
  uint32_t width_;
  uint32_t height_;
  uint32_t num_samples_;

  virtual void initResource() = 0;
};

std::string to_string(const Framebuffer::Attachment value);

};  // namespace gfx

std::ostream& operator<<(std::ostream& os, const gfx::Framebuffer::Attachment value);
