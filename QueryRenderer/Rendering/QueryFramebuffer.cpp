/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/QueryFramebuffer.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/Texture.h"
#include "QueryRenderer/Rendering/QueryIdMapPixelBuffer.h"
#include "QueryRenderer/ResourceTracking.h"
#include "Shared/scope.h"

namespace QueryRenderer {

using ::gfx::DeviceContext;
using ::gfx::Framebuffer;
using ::gfx::ImageUsageBits;
using ::gfx::PixelFormat;
using ::gfx::ResourceManager;
using ::gfx::SamplerFilterMode;
using ::gfx::SamplerWrapMode;
using ::gfx::Texture;

static bool clampWidthAndHeightToSrc(uint32_t start_x,
                                     uint32_t start_y,
                                     int32_t width,
                                     int32_t height,
                                     uint32_t src_width,
                                     uint32_t src_height,
                                     uint32_t& clamped_width,
                                     uint32_t& clamped_height) {
  bool rtn = false;

  clamped_width = (width < 0 ? src_width - start_x : width);
  clamped_height = (height < 0 ? src_height - start_y : height);

  bool width_overflow = (start_x + clamped_width > src_width);
  bool height_overflow = (start_y + clamped_height > src_height);

  if (width_overflow || height_overflow) {
    rtn = true;

    if (width_overflow) {
      clamped_width = src_width - start_x;
    }

    if (height_overflow) {
      clamped_height = src_height - start_y;
    }
  }

  return rtn;
}

QueryFramebuffer::QueryFramebuffer(const DeviceContext& device_ctx,
                                   std::string_view name,
                                   const gfx::RasterSampleCount raster_sample_count,
                                   const CreateFlags flags)
    : device_ctx_{device_ctx}
    , name_{name}
    , raster_sample_count_{raster_sample_count}
    , num_samples_{gfx::raster_sample_count_enum_to_value(raster_sample_count)}
    , supports_hit_test_{any_bits_set(flags & CreateFlags::kSupportHitTest)}
    , supports_depth_test_{any_bits_set(flags & CreateFlags::kSupportDepthTest)}
    , supports_stencil_{any_bits_set(flags & CreateFlags::kSupportStencil)}
    , enable_api_export_{any_bits_set(flags & CreateFlags::kEnableApiExport)}
    , create_r32ui_view_{any_bits_set(flags & CreateFlags::kCreateR32UIView)}
    , initialized_{false}
    , rgba_texture_{nullptr}
    , id1A_texture_{nullptr}
    , id1B_texture_{nullptr}
    , id2_texture_{nullptr}
    , depth_texture_{nullptr}
    , fbo_{nullptr} {
  // Build our Framebuffer::Layout and create internal renderpass
  Framebuffer::Layout::Builder layout_builder(num_samples_);
  layout_builder.addAttachment(Framebuffer::Attachment::kColor0, PixelFormat::kRGBA8);
  if (supports_hit_test_) {
    layout_builder.addAttachment(Framebuffer::Attachment::kColor1, PixelFormat::kR32UI)
        .addAttachment(Framebuffer::Attachment::kColor2, PixelFormat::kR32UI)
        .addAttachment(Framebuffer::Attachment::kColor3, PixelFormat::kR32UI);
  }
  if (supports_depth_test_) {
    if (supports_stencil_) {
      layout_builder.addAttachment(Framebuffer::Attachment::kDepthStencil,
                                   PixelFormat::kDepthStencil);
    } else {
      layout_builder.addAttachment(Framebuffer::Attachment::kDepth, PixelFormat::kDepth);
    }
  }
  framebuffer_layout_.build(std::move(layout_builder));

  render_pass_ =
      device_ctx_.getResourceManager().createRenderPass("QFB " + name_ + " RenderPass",
                                                        framebuffer_layout_,
                                                        gfx::RenderPass::ClearBits::kAll,
                                                        gfx::ImageLayout::kUndefined,
                                                        gfx::ImageLayout::kAttachment);
  CHECK(render_pass_);
}

QueryFramebuffer::~QueryFramebuffer() {
  destroy();
}

void QueryFramebuffer::init(const uint32_t width, const uint32_t height) {
  CHECK(!initialized_);

  ScopeGuard exit_init = [this]() {
    if (!initialized_) {
      LOG(WARNING) << "QueryFramebuffer failed to initialize completely";
      destroy();
    }
  };

  auto& resource_mgr = device_ctx_.getResourceManager();

  ImageUsageBits extra_usage_bits = ImageUsageBits::kStorageBit;
  if (enable_api_export_) {
    extra_usage_bits |= ImageUsageBits::kExternalApiBit;
  }
  if (create_r32ui_view_) {
    extra_usage_bits |= ImageUsageBits::kMutableViewBit;
  }

  rgba_texture_ = createFboTexture(ResourceTrackingString("QFB " + name_ + " RGBA"),
                                   resource_mgr,
                                   FboAttachment::Color,
                                   width,
                                   height,
                                   num_samples_,
                                   extra_usage_bits);
  attachment_mgr_.setAttachment(Framebuffer::Attachment::kColor0, rgba_texture_.get());

  if (create_r32ui_view_) {
    rgba_texture_->createView(kR32UIViewId, PixelFormat::kR32UI);
  }

  if (supports_hit_test_) {
    id1A_texture_ = createFboTexture(ResourceTrackingString("QFB " + name_ + " ID1A"),
                                     resource_mgr,
                                     FboAttachment::ID1A,
                                     width,
                                     height,
                                     num_samples_,
                                     extra_usage_bits);
    attachment_mgr_.setAttachment(Framebuffer::Attachment::kColor1, id1A_texture_.get());

    id1B_texture_ = createFboTexture(ResourceTrackingString("QFB " + name_ + " ID1B"),
                                     resource_mgr,
                                     FboAttachment::ID1B,
                                     width,
                                     height,
                                     num_samples_,
                                     extra_usage_bits);
    attachment_mgr_.setAttachment(Framebuffer::Attachment::kColor2, id1B_texture_.get());

    id2_texture_ = createFboTexture(ResourceTrackingString("QFB " + name_ + " ID2"),
                                    resource_mgr,
                                    FboAttachment::ID2,
                                    width,
                                    height,
                                    num_samples_,
                                    extra_usage_bits);
    attachment_mgr_.setAttachment(Framebuffer::Attachment::kColor3, id2_texture_.get());
  }

  if (supports_depth_test_ && supports_stencil_) {
    depth_texture_ =
        createFboTexture(ResourceTrackingString("QFB " + name_ + " Depth/Stencil"),
                         resource_mgr,
                         FboAttachment::Depth_And_Stencil,
                         width,
                         height,
                         num_samples_);
    attachment_mgr_.setAttachment(Framebuffer::Attachment::kDepthStencil,
                                  depth_texture_.get());
  } else if (supports_depth_test_) {
    depth_texture_ = createFboTexture(ResourceTrackingString("QFB " + name_ + " Depth"),
                                      resource_mgr,
                                      FboAttachment::Depth,
                                      width,
                                      height,
                                      num_samples_);
    attachment_mgr_.setAttachment(Framebuffer::Attachment::kDepth, depth_texture_.get());
  }

  fbo_ = resource_mgr.createFramebuffer("QFB" + name_ + " FBO",
                                        *render_pass_,
                                        attachment_mgr_,
                                        width,
                                        height,
                                        num_samples_);
  CHECK(fbo_);

  initialized_ = true;
}

gfx::Framebuffer* QueryFramebuffer::getOrCreateFramebufferForRenderPass(
    const gfx::RenderPass& render_pass) {
  CHECK(initialized_);
  gfx::Framebuffer* rtn{nullptr};
  auto itr = custom_framebuffer_map_.find(&render_pass);
  if (itr == custom_framebuffer_map_.end()) {
    auto& resource_mgr = device_ctx_.getResourceManager();
    auto fb = resource_mgr.createFramebuffer("QFB " + name_ + " Custom FBO",
                                             render_pass,
                                             attachment_mgr_,
                                             fbo_->getWidth(),
                                             fbo_->getHeight(),
                                             num_samples_);
    auto [item, result] =
        custom_framebuffer_map_.try_emplace(&render_pass, std::move(fb));
    if (!result) {
      resource_mgr.destroyFramebuffer(std::move(fb));
      THROW_RUNTIME_EX("Failed to insert custom framebuffer in map");
    }
    itr = item;
  }
  rtn = itr->second.get();

  CHECK(rtn);
  return rtn;
}

void QueryFramebuffer::destroy() {
  ScopeGuard exit_destroy = [this]() {
    if (initialized_) {
      LOG(WARNING) << "QueryFramebuffer failed to destroy";
    }
    initialized_ = false;
  };

  auto& resource_mgr = device_ctx_.getResourceManager();

  // destroy RenderPass
  if (render_pass_) {
    resource_mgr.destroyRenderPass(std::move(render_pass_));
  }

  // destroy primary FBO
  if (fbo_) {
    resource_mgr.destroyFramebuffer(std::move(fbo_));
  }

  // destroy custom RenderPass FBOs
  for (auto& map_item : custom_framebuffer_map_) {
    if (map_item.second) {
      resource_mgr.destroyFramebuffer(std::move(map_item.second));
    }
  }

  // clear AttachmentManager
  attachment_mgr_.clear();

  // destroy mandatory FBO textures
  if (rgba_texture_) {
    resource_mgr.destroyTexture(std::move(rgba_texture_));
  }

  // destroy optional FBO textures
  if (id1A_texture_) {
    resource_mgr.destroyTexture(std::move(id1A_texture_));
  }
  if (id1B_texture_) {
    resource_mgr.destroyTexture(std::move(id1B_texture_));
  }
  if (id2_texture_) {
    resource_mgr.destroyTexture(std::move(id2_texture_));
  }
  if (depth_texture_) {
    resource_mgr.destroyTexture(std::move(depth_texture_));
  }

  initialized_ = false;
}

void QueryFramebuffer::resize(int32_t width, int32_t height) {
  if (!initialized_) {
    init(width, height);
    CHECK(initialized_);
  } else {
    CHECK(fbo_);
    fbo_->resize(width, height);
    // resize custom RenderPass FBOs
    // This just updates the internal sizes since the textures
    // will have already been sized
    for (auto& map_item : custom_framebuffer_map_) {
      CHECK(map_item.second);
      map_item.second->resize(width, height);
    }
  }
}

uint32_t QueryFramebuffer::getWidth() const {
  if (!initialized_) {
    return 1;
  }
  return fbo_->getWidth();
}

uint32_t QueryFramebuffer::getHeight() const {
  if (!initialized_) {
    return 1;
  }
  return fbo_->getHeight();
}

gfx::RasterSampleCount QueryFramebuffer::getRasterSampleCount() const {
  return raster_sample_count_;
}

uint32_t QueryFramebuffer::getNumSamples() const {
  return num_samples_;
}

const DeviceContext& QueryFramebuffer::getDeviceContext() const {
  return device_ctx_;
}

Texture* QueryFramebuffer::getTexture(FboAttachment attachment) const {
  CHECK(initialized_);
  switch (attachment) {
    case FboAttachment::Color:
      return rgba_texture_.get();
    case FboAttachment::ID1A:
      return id1A_texture_.get();
    case FboAttachment::ID1B:
      return id1B_texture_.get();
    case FboAttachment::ID2:
      return id2_texture_.get();
    case FboAttachment::Depth:
    case FboAttachment::Depth_And_Stencil:
      return depth_texture_.get();
    default:
      CHECK(false);
  }
  return nullptr;
}

std::vector<std::byte> QueryFramebuffer::readColorBuffer(uint32_t start_x,
                                                         uint32_t start_y,
                                                         int32_t width,
                                                         int32_t height) {
  RENDER_LOG_SCOPE();
  CHECK(initialized_);
  uint32_t my_width = getWidth();
  uint32_t my_height = getHeight();

  if (width < 0) {
    width = static_cast<int32_t>(my_width);
  }

  if (height < 0) {
    height = static_cast<int32_t>(my_height);
  }

  uint32_t width_to_use, height_to_use;
  if (clampWidthAndHeightToSrc(start_x,
                               start_y,
                               width,
                               height,
                               my_width,
                               my_height,
                               width_to_use,
                               height_to_use)) {
    LOG(WARNING) << "QueryFramebuffer: bounds of the pixels to read ((x, y) = ("
                 << start_x << ", " << start_y << "), width = " << width
                 << ", height = " << height
                 << ") extend beyond the bounds of the framebuffer (width = " << my_width
                 << ", height = " << my_height
                 << "). Only pixels within the bounds will be read. The rest will be "
                    "initialized to (0,0,0,0).";
  }

  std::vector<std::byte> pixels(width * height * 4, std::byte(0));
  uint8_t* raw_pixels = reinterpret_cast<uint8_t*>(pixels.data());

  if (width_to_use > 0 && height_to_use > 0) {
    fbo_->readPixels(Framebuffer::Attachment::kColor0,
                     start_x,
                     start_y,
                     width_to_use,
                     height_to_use,
                     PixelFormat::kRGBA8,
                     raw_pixels);
  }

  return pixels;
}

RowIdVector QueryFramebuffer::readRowIdBuffer(const bool least_significant_bits,
                                              uint32_t start_x,
                                              uint32_t start_y,
                                              int32_t width,
                                              int32_t height) {
  RENDER_LOG_SCOPE();
  CHECK(initialized_);
  RUNTIME_EX_ASSERT(supports_hit_test_,
                    "QueryFramebuffer: The framebuffer was not setup for hit-testing. "
                    "Cannot read row IDs.");

  uint32_t my_width = getWidth();
  uint32_t my_height = getHeight();

  if (width < 0) {
    width = static_cast<int32_t>(my_width);
  }

  if (height < 0) {
    height = static_cast<int32_t>(my_height);
  }

  RowIdVector pixels;
  if ((least_significant_bits || id1B_texture_)) {
    uint32_t width_to_use, height_to_use;
    if (clampWidthAndHeightToSrc(start_x,
                                 start_y,
                                 width,
                                 height,
                                 my_width,
                                 my_height,
                                 width_to_use,
                                 height_to_use)) {
      LOG(WARNING) << "QueryFramebuffer: bounds of the row ids to read ((x, y) = ("
                   << start_x << ", " << start_y << "), width = " << width
                   << ", height = " << height
                   << ") extend beyond the bounds of the framebuffer (width = "
                   << my_width << ", height = " << my_height
                   << "). Only pixels within the bounds will be read. The rest will be "
                      "initialized to 0.";
    }

    pixels.resize(width * height, 0U);
    auto raw_pixels = pixels.data();
    if (width_to_use > 0 && height_to_use > 0) {
      fbo_->readPixels((least_significant_bits ? Framebuffer::Attachment::kColor1
                                               : Framebuffer::Attachment::kColor2),
                       start_x,
                       start_y,
                       width_to_use,
                       height_to_use,
                       PixelFormat::kR32UI,
                       raw_pixels);
    }
  }

  return pixels;
}

TableIdVector QueryFramebuffer::readTableIdBuffer(uint32_t start_x,
                                                  uint32_t start_y,
                                                  int32_t width,
                                                  int32_t height) {
  RENDER_LOG_SCOPE();
  CHECK(initialized_);
  RUNTIME_EX_ASSERT(supports_hit_test_,
                    "QueryFramebuffer: The framebuffer was not setup for hit-testing. "
                    "Cannot read table IDs.");

  uint32_t my_width = getWidth();
  uint32_t my_height = getHeight();

  if (width < 0) {
    width = static_cast<int32_t>(my_width);
  }

  if (height < 0) {
    height = static_cast<int32_t>(my_height);
  }

  uint32_t width_to_use, height_to_use;
  if (clampWidthAndHeightToSrc(start_x,
                               start_y,
                               width,
                               height,
                               my_width,
                               my_height,
                               width_to_use,
                               height_to_use)) {
    LOG(WARNING) << "QueryFramebuffer: bounds of the table ids to read ((x, y) = ("
                 << start_x << ", " << start_y << "), width = " << width
                 << ", height = " << height
                 << ") extend beyond the bounds of the framebuffer (width = " << my_width
                 << ", height = " << my_height
                 << "). Only pixels within the bounds will be read. The rest will be "
                    "initialized to 0.";
  }

  TableIdVector pixels(width * height, 0);
  auto raw_pixels = pixels.data();
  if (width_to_use > 0 && height_to_use > 0) {
    fbo_->readPixels(Framebuffer::Attachment::kColor3,
                     start_x,
                     start_y,
                     width_to_use,
                     height_to_use,
                     PixelFormat::kR32UI,
                     raw_pixels);
  }

  return pixels;
}

IdBufferTuple QueryFramebuffer::readIdBuffers(int32_t width, int32_t height) {
  RENDER_LOG_SCOPE();
  // TODO(scb): lots of redundant checks going through these 3 functions, but that's
  // cleanup for later. This at least pulls the 3 calls into this class instead of having
  // them scattered in several places in QueryRenderer.cpp
  CHECK(initialized_);
  return {readRowIdBuffer(true, 0, 0, width, height),
          readRowIdBuffer(false, 0, 0, width, height),
          readTableIdBuffer(0, 0, width, height)};
}

std::vector<float> QueryFramebuffer::readDepthBuffer(uint32_t start_x,
                                                     uint32_t start_y,
                                                     int32_t width,
                                                     int32_t height) {
  RENDER_LOG_SCOPE();
  CHECK(initialized_);
  RUNTIME_EX_ASSERT(supports_depth_test_,
                    "QueryFramebuffer: The framebuffer was not setup for depth testing. "
                    "Cannot read the depth buffer.");

  uint32_t my_width = getWidth();
  uint32_t my_height = getHeight();

  if (width < 0) {
    width = static_cast<int32_t>(my_width);
  }

  if (height < 0) {
    height = static_cast<int32_t>(my_height);
  }

  uint32_t width_to_use, height_to_use;
  if (clampWidthAndHeightToSrc(start_x,
                               start_y,
                               width,
                               height,
                               my_width,
                               my_height,
                               width_to_use,
                               height_to_use)) {
    LOG(WARNING) << "QueryFramebuffer: bounds of the depth buffer to read ((x, y) = ("
                 << start_x << ", " << start_y << "), width = " << width
                 << ", height = " << height
                 << ") extend beyond the bounds of the framebuffer (width = " << my_width
                 << ", height = " << my_height
                 << "). Only pixels within the bounds will be read. The rest will be "
                    "initialized to 0.";
  }

  std::vector<float> pixels(width * height, 0.0f);
  auto raw_pixels = pixels.data();
  if (width_to_use > 0 && height_to_use > 0) {
    fbo_->readPixels(Framebuffer::Attachment::kDepth,
                     start_x,
                     start_y,
                     width_to_use,
                     height_to_use,
                     PixelFormat::kDepth,
                     raw_pixels);
  }

  return pixels;
}

void QueryFramebuffer::copyToFramebuffer(QueryFramebuffer& dst_fbo,
                                         const uint32_t start_x,
                                         const uint32_t start_y,
                                         const uint32_t width,
                                         const uint32_t height,
                                         const bool do_color,
                                         const bool do_hit,
                                         const bool do_depth,
                                         const bool do_async_copy) {
  RENDER_LOG_SCOPE();
  CHECK(initialized_);
  CHECK(dst_fbo.getDeviceContext().getGpuId() == device_ctx_.getGpuId());
  uint32_t my_width = getWidth();
  uint32_t my_height = getHeight();

  CHECK(my_width == dst_fbo.getWidth() && my_height == dst_fbo.getHeight());

  std::vector<Framebuffer::Attachment> attachments;
  if (do_color) {
    attachments.push_back(Framebuffer::Attachment::kColor0);
  }
  if (do_hit && supports_hit_test_ && dst_fbo.supportsHitTest()) {
    attachments.push_back(Framebuffer::Attachment::kColor1);
    attachments.push_back(Framebuffer::Attachment::kColor2);
    attachments.push_back(Framebuffer::Attachment::kColor3);
  }
  if (do_depth && supports_depth_test_ && dst_fbo.supportsDepthTest()) {
    attachments.push_back(Framebuffer::Attachment::kDepth);
  }
  if (!attachments.empty()) {
    fbo_->copyToFramebuffer(*(dst_fbo.getFramebuffer()),
                            attachments,
                            start_x,
                            start_y,
                            width,
                            height,
                            start_x,
                            start_y,
                            width,
                            height,
                            do_async_copy);
  }
}

void QueryFramebuffer::copyRowIdBufferToPbo(QueryIdMapPixelBufferShPtr& pbo,
                                            const bool least_significant_bits) {
  RENDER_LOG_SCOPE();
  CHECK(initialized_);
  RUNTIME_EX_ASSERT(pbo != nullptr,
                    "Pbo is empty. Cannot copy pixels to an undefined pbo.");

  RUNTIME_EX_ASSERT((least_significant_bits && id1A_texture_ != nullptr) ||
                        (!least_significant_bits && id1B_texture_),
                    "QueryFramebuffer: The framebuffer was not setup for an ID map. "
                    "Cannot copy ID map to pixel buffer object.");

  uint32_t my_width = getWidth();
  uint32_t my_height = getHeight();

  uint32_t pbo_width = pbo->getWidth();
  uint32_t pbo_height = pbo->getHeight();

  RUNTIME_EX_ASSERT(pbo_width <= my_width && pbo_height <= my_height,
                    "The pbo for the idmap is too big for the framebuffer. It is " +
                        std::to_string(pbo_width) + "x" + std::to_string(pbo_height) +
                        " and the fbo is " + std::to_string(my_width) + "x" +
                        std::to_string(my_height) +
                        ". The pbo size needs to be <= the fbo size.");

  // TODO(croot): should we initialize the buffer to something? Like all 0s beforehand?
  CHECK(my_width > 0 && my_height > 0);

  fbo_->copyToPixelBuffer(pbo->getPixelBuffer2d(),
                          (least_significant_bits ? Framebuffer::Attachment::kColor1
                                                  : Framebuffer::Attachment::kColor2),
                          0,
                          0,
                          pbo_width,
                          pbo_height,
                          0,
                          PixelFormat::kR32UI);
}

void QueryFramebuffer::copyResultCacheIdBufferToPbo(QueryIdMapPixelBufferShPtr& pbo) {
  RENDER_LOG_SCOPE();
  CHECK(initialized_);
  RUNTIME_EX_ASSERT(pbo != nullptr,
                    "Pbo is empty. Cannot copy pixels to an undefined pbo.");

  RUNTIME_EX_ASSERT(id2_texture_ != nullptr,
                    "QueryFramebuffer: The framebuffer was not setup for an ID map. "
                    "Cannot copy ID map to pixel buffer object.");

  uint32_t my_width = getWidth();
  uint32_t my_height = getHeight();

  uint32_t pbo_width = pbo->getWidth();
  uint32_t pbo_height = pbo->getHeight();

  RUNTIME_EX_ASSERT(pbo_width <= my_width && pbo_height <= my_height,
                    "The pbo for the idmap is too big for the framebuffer. It is " +
                        std::to_string(pbo_width) + "x" + std::to_string(pbo_height) +
                        " and the fbo is " + std::to_string(my_width) + "x" +
                        std::to_string(my_height) +
                        ". The pbo size needs to be <= the fbo size.");

  // TODO(croot): should we initialize the buffer to something? Like all 0s beforehand?
  CHECK(my_width > 0 && my_height > 0);

  fbo_->copyToPixelBuffer(pbo->getPixelBuffer2d(),
                          Framebuffer::Attachment::kColor3,
                          0,
                          0,
                          pbo_width,
                          pbo_height,
                          0,
                          PixelFormat::kR32UI);
}

gfx::resource_ptr<Texture> QueryFramebuffer::createFboTexture(
    const std::string& resource_tracking_string,
    ResourceManager& resource_mgr,
    FboAttachment texture_type,
    uint32_t width,
    uint32_t height,
    uint32_t num_samples,
    ImageUsageBits extra_usage_bits) {
  PixelFormat pixel_format = PixelFormat::kCOUNT;
  ImageUsageBits attachment_usage_bit = ImageUsageBits::kNone;
  switch (texture_type) {
    case FboAttachment::Color:
      pixel_format = PixelFormat::kRGBA8;
      attachment_usage_bit = ImageUsageBits::kColorAttachmentBit;
      break;
    case FboAttachment::ID1A:
    case FboAttachment::ID1B:
    case FboAttachment::ID2:
      // The ID2_BUFFER is used to store the table-id of the id stored in ID_BUFFER.
      // We need the table id hit-testing in multi-layer rendering.
      pixel_format = PixelFormat::kR32UI;
      attachment_usage_bit = ImageUsageBits::kColorAttachmentBit;
      break;
    case FboAttachment::Depth:
      pixel_format = PixelFormat::kDepth;
      attachment_usage_bit = ImageUsageBits::kDepthStencilAttachmentBit;
      break;
    case FboAttachment::Depth_And_Stencil:
      pixel_format = PixelFormat::kDepthStencil;
      attachment_usage_bit = ImageUsageBits::kDepthStencilAttachmentBit;
      break;
  }

  CHECK(pixel_format != PixelFormat::kCOUNT);
  CHECK(attachment_usage_bit != gfx::ImageUsageBits::kNone);

  return resource_mgr.createTexture(
      resource_tracking_string,
      width,
      height,
      1,
      pixel_format,
      num_samples,
      false,
      attachment_usage_bit | extra_usage_bits,
      gfx::get_default_sampler_state_for_format(pixel_format));
}

}  // namespace QueryRenderer
