/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/Framebuffer.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "QueryRenderer/Interface/RawBufferTypes.h"
#include "QueryRenderer/Rendering/Types.h"
#include "Shared/EnumBitmaskOps.h"

namespace QueryRenderer {

enum class FboAttachment : uint8_t {
  Color,
  ID1A,
  ID1B,
  ID2,
  Depth,
  Depth_And_Stencil,
  Max_Attachments = Depth_And_Stencil
};

///////////////////////////////////////////////////////////////////////
/**
 * QueryFramebuffer
 *  Class used for managing framebuffers for backend rendering of
 *  database queries.
 */

class QueryFramebuffer {
 public:
  enum class CreateFlags {
    kNone = 0,
    kSupportHitTest = 1 << 0,
    kSupportDepthTest = 1 << 1,
    kSupportStencil = 1 << 2,
    kEnableApiExport = 1 << 3,
    kCreateR32UIView = 1 << 4
  };

  static constexpr uint32_t kR32UIViewId = 1u;

  QueryFramebuffer(const gfx::DeviceContext& device_ctx,
                   std::string_view name,
                   const gfx::RasterSampleCount raster_sample_count,
                   const CreateFlags flags);

  ~QueryFramebuffer();

  void resize(int32_t width, int32_t height);

  std::vector<std::byte> readColorBuffer(uint32_t start_x = 0,
                                         uint32_t start_y = 0,
                                         int32_t width = -1,
                                         int32_t height = -1);
  RowIdVector readRowIdBuffer(const bool least_significant_bits = true,
                              uint32_t start_x = 0,
                              uint32_t start_y = 0,
                              int32_t width = -1,
                              int32_t height = -1);
  TableIdVector readTableIdBuffer(uint32_t start_x = 0,
                                  uint32_t start_y = 0,
                                  int32_t width = -1,
                                  int32_t height = -1);
  IdBufferTuple readIdBuffers(int32_t width, int32_t height);

  std::vector<float> readDepthBuffer(uint32_t start_x = 0,
                                     uint32_t start_y = 0,
                                     int32_t width = -1,
                                     int32_t height = -1);

  void copyToFramebuffer(QueryFramebuffer& dst_fbo,
                         const uint32_t start_x,
                         const uint32_t start_y,
                         const uint32_t width,
                         const uint32_t height,
                         const bool do_color = true,
                         const bool do_hit = true,
                         const bool do_depth = true,
                         const bool do_async_copy = false);

  void copyRowIdBufferToPbo(QueryIdMapPixelBufferShPtr& pbo,
                            const bool least_significant_bits = true);
  void copyResultCacheIdBufferToPbo(QueryIdMapPixelBufferShPtr& pbo);

  uint32_t getWidth() const;
  uint32_t getHeight() const;
  uint32_t getNumSamples() const;
  gfx::RasterSampleCount getRasterSampleCount() const;

  bool supportsHitTest() const { return supports_hit_test_; }
  bool supportsDepthTest() const { return supports_depth_test_; }

  const gfx::DeviceContext& getDeviceContext() const;

  gfx::Framebuffer* getFramebuffer() const { return fbo_.get(); }
  gfx::Texture* getTexture(FboAttachment attachment) const;
  gfx::AttachmentManager& getAttachmentManager() { return attachment_mgr_; }
  const gfx::Framebuffer::Layout& getFramebufferLayout() const {
    return framebuffer_layout_;
  }

  gfx::Framebuffer* getOrCreateFramebufferForRenderPass(
      const gfx::RenderPass& render_pass);

  // kColorAttachment, and kDepthStencilAttachment usage bits will be handled
  // automatically. Other bits like kTransferDest and kSampled should be passed in
  // extra_usage_bits if required
  static gfx::resource_ptr<gfx::Texture> createFboTexture(
      const std::string& resource_tracking_string,
      gfx::ResourceManager& resource_mgr,
      FboAttachment texture_type,
      uint32_t width,
      uint32_t height,
      uint32_t num_samples = 1u,
      gfx::ImageUsageBits extra_usage_bits = gfx::ImageUsageBits::kNone);

 private:
  const gfx::DeviceContext& device_ctx_;
  const std::string name_;
  gfx::RasterSampleCount raster_sample_count_;
  uint32_t num_samples_;
  bool supports_hit_test_;
  bool supports_depth_test_;
  bool supports_stencil_;
  bool enable_api_export_;
  bool create_r32ui_view_;
  bool initialized_;

  gfx::resource_ptr<gfx::Texture> rgba_texture_;
  gfx::resource_ptr<gfx::Texture> id1A_texture_;
  gfx::resource_ptr<gfx::Texture> id1B_texture_;
  gfx::resource_ptr<gfx::Texture> id2_texture_;
  gfx::resource_ptr<gfx::Texture> depth_texture_;  // depth or depth+stencil
  gfx::resource_ptr<gfx::Framebuffer> fbo_;
  gfx::resource_ptr<gfx::RenderPass> render_pass_;
  gfx::Framebuffer::Layout framebuffer_layout_;
  gfx::AttachmentManager attachment_mgr_;

  using FramebufferMap =
      std::unordered_map<const gfx::RenderPass*, gfx::resource_ptr<gfx::Framebuffer>>;
  FramebufferMap custom_framebuffer_map_;

  void init(uint32_t width, uint32_t height);
  void destroy();
};

using QueryFramebufferUqPtr = std::unique_ptr<QueryFramebuffer>;
using QueryFramebufferShPtr = std::shared_ptr<QueryFramebuffer>;

}  // namespace QueryRenderer

ENABLE_BITMASK_OPS(QueryRenderer::QueryFramebuffer::CreateFlags);
