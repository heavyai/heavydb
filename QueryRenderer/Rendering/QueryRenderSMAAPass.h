/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>
#include <vector>

#include <boost/noncopyable.hpp>

#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Rendering/SeparateMultiSamplesPass.h"

namespace QueryRenderer {

class QueryRenderSMAAPass : boost::noncopyable {
 public:
  enum class SMAAQualityPreset { kLow, kMedium, kHigh, kUltra };
  enum class SMAAEdgeDetectionType { kLuma, kColor, kDepth };

  QueryRenderSMAAPass(
      const GlobalRenderContext& global_ctx,
      SMAAQualityPreset quality_preset = SMAAQualityPreset::kHigh,
      SMAAEdgeDetectionType edge_detect_type = SMAAEdgeDetectionType::kColor);
  ~QueryRenderSMAAPass();

  void prepareRenderTargets(uint32_t width, uint32_t height);
  void postPrepareRenderTargets();

  void runPass(uint32_t viewport_width,
               uint32_t viewport_height,
               const gfx::DeviceContext& device_ctx,
               SeparateMultiSamplesPass::SourceFramebuffer source);

 private:
  class GpuData {
   public:
    explicit GpuData(const gfx::DeviceContext& device_ctx);
    ~GpuData();

    void prepareRenderTargets(uint32_t width,
                              uint32_t height,
                              uint32_t num_samples,
                              gfx::AttachmentManager& aa_fb_attachment_mgr);
    void destroyResources();

   private:
    const gfx::DeviceContext& device_ctx_;

    gfx::resource_ptr<gfx::RenderPass> edge_detect_render_pass_;
    gfx::resource_ptr<gfx::RenderPass> blending_weight_render_pass_;
    gfx::resource_ptr<gfx::RenderPass> neighborhood_blend_first_render_pass_;
    gfx::resource_ptr<gfx::RenderPass> neighborhood_blend_other_render_pass_;

    gfx::resource_ptr<gfx::Framebuffer> edge_detect_fbo_;
    gfx::resource_ptr<gfx::Framebuffer> blending_weight_fbo_;
    gfx::resource_ptr<gfx::Framebuffer> neighborhood_blend_fbo_;
    gfx::AttachmentManager edge_detect_attachment_mgr_;
    gfx::AttachmentManager blending_weight_attachment_mgr_;
    gfx::AttachmentManager neighborhood_blend_attachment_mgr_;

    gfx::resource_ptr<gfx::Texture> area_texture_;
    gfx::resource_ptr<gfx::Texture> search_texture_;
    gfx::resource_ptr<gfx::Texture> edge_detection_texture_;
    gfx::resource_ptr<gfx::Texture> blending_weight_texture_;

    gfx::MaterialUqPtr edge_detection_material_;
    gfx::MaterialUqPtr blending_weight_material_;
    gfx::MaterialUqPtr neighborhood_blend_material_;

    gfx::resource_ptr<gfx::GraphicsPipeline> edge_detection_pipeline_;
    gfx::resource_ptr<gfx::GraphicsPipeline> blending_weight_pipeline_;
    gfx::resource_ptr<gfx::GraphicsPipeline> neighborhood_blend_first_pipeline_;
    gfx::resource_ptr<gfx::GraphicsPipeline> neighborhood_blend_other_pipeline_;

    friend class QueryRenderSMAAPass;
  };

  const GlobalRenderContext& global_ctx_;
  std::unordered_map<GpuId, GpuData> gpu_data_map_;

  SMAAQualityPreset quality_preset_;
  SMAAEdgeDetectionType edge_detect_type_;

  bool use_predication_;   // TODO(croot)
  bool use_reprojection_;  // TODO(croot)
  uint32_t num_samples_;

  bool initialized_;

  gfx::PipelineDescriptorUqPtr edge_detection_pipeline_descriptor_;
  gfx::PipelineDescriptorUqPtr blending_weight_pipeline_descriptor_;
  gfx::PipelineDescriptorUqPtr neighborhood_blend_first_pipeline_descriptor_;
  gfx::PipelineDescriptorUqPtr neighborhood_blend_other_pipeline_descriptor_;

  // Initialize the gpu data map and build the material resources
  void initBaseResources();
  void destroyResources();

  void updateViewportUniforms(uint32_t viewport_width,
                              uint32_t viewport_height,
                              GpuData& gpu_data);

  // Separate multi-sample support
  std::vector<std::array<float, 4>> subsample_indices_;
};

}  // namespace QueryRenderer
