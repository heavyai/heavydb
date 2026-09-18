/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/QueryRenderSMAAPass.h"

#include <string>
#include <string_view>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Rendering/QueryFramebuffer.h"
#include "QueryRenderer/Rendering/SeparateMultiSamplesPass.h"
#include "QueryRenderer/Rendering/textures/AreaTex.h"
#include "QueryRenderer/Rendering/textures/SearchTex.h"
#include "QueryRenderer/ResourceTracking.h"
#include "Shared/scope.h"

using ::gfx::DeviceContext;
using ::gfx::ImageUsageBits;
using ::gfx::PixelFormat;
using ::gfx::ResourceManager;
using ::gfx::ResourceType;
using ::gfx::SamplerFilterMode;
using ::gfx::SamplerWrapMode;
using ::gfx::TextureSamplerState;

namespace QueryRenderer {

QueryRenderSMAAPass::GpuData::GpuData(const gfx::DeviceContext& device_ctx)
    : device_ctx_{device_ctx} {}

QueryRenderSMAAPass::GpuData::~GpuData() {
  destroyResources();
}

void QueryRenderSMAAPass::GpuData::prepareRenderTargets(
    uint32_t width,
    uint32_t height,
    uint32_t num_samples,
    gfx::AttachmentManager& aa_fb_attachment_mgr) {
  auto& resource_mgr = device_ctx_.getResourceManager();

  // Edge and weight RenderPass and Framebuffer
  if (edge_detect_fbo_ == nullptr) {
    edge_detection_texture_ =
        resource_mgr.createTexture(ResourceTrackingString("SMAA Edge Detection"),
                                   width,
                                   height,
                                   1,
                                   PixelFormat::kRGBA8,
                                   1,
                                   false,
                                   ImageUsageBits::kColorAttachmentBit,
                                   TextureSamplerState(SamplerFilterMode::kLinear,
                                                       SamplerFilterMode::kLinear,
                                                       SamplerWrapMode::kClampEdge,
                                                       SamplerWrapMode::kClampEdge));
    edge_detect_attachment_mgr_.setAttachment(gfx::Framebuffer::Attachment::kColor0,
                                              edge_detection_texture_.get());

    if (edge_detect_render_pass_) {
      resource_mgr.destroyRenderPass(std::move(edge_detect_render_pass_));
    }

    edge_detect_render_pass_ =
        resource_mgr.createRenderPass("SMAA Edge Detection",
                                      edge_detect_attachment_mgr_.getLayout(),
                                      gfx::RenderPass::ClearBits::kAll,
                                      gfx::ImageLayout::kUndefined,
                                      gfx::ImageLayout::kShaderReadOnly);

    edge_detect_fbo_ = resource_mgr.createFramebuffer("SMAA Edge and Weight",
                                                      *edge_detect_render_pass_,
                                                      edge_detect_attachment_mgr_,
                                                      width,
                                                      height,
                                                      1);
    edge_detection_pipeline_->create(*edge_detect_render_pass_);
  } else {
    edge_detect_fbo_->resize(width, height);
  }

  if (blending_weight_fbo_ == nullptr) {
    blending_weight_texture_ =
        resource_mgr.createTexture(ResourceTrackingString("SMAA Blending Weight"),
                                   width,
                                   height,
                                   1,
                                   PixelFormat::kRGBA8,
                                   1,
                                   false,
                                   ImageUsageBits::kColorAttachmentBit,
                                   TextureSamplerState(SamplerFilterMode::kLinear,
                                                       SamplerFilterMode::kLinear,
                                                       SamplerWrapMode::kClampEdge,
                                                       SamplerWrapMode::kClampEdge));
    blending_weight_attachment_mgr_.setAttachment(gfx::Framebuffer::Attachment::kColor0,
                                                  blending_weight_texture_.get());

    if (blending_weight_render_pass_) {
      resource_mgr.destroyRenderPass(std::move(blending_weight_render_pass_));
    }

    blending_weight_render_pass_ =
        resource_mgr.createRenderPass("SMAA Blending Weight",
                                      blending_weight_attachment_mgr_.getLayout(),
                                      gfx::RenderPass::ClearBits::kAll,
                                      gfx::ImageLayout::kUndefined,
                                      gfx::ImageLayout::kShaderReadOnly);

    blending_weight_fbo_ = resource_mgr.createFramebuffer("SMAA Blending Weight",
                                                          *blending_weight_render_pass_,
                                                          blending_weight_attachment_mgr_,
                                                          width,
                                                          height,
                                                          1);
    blending_weight_pipeline_->create(*blending_weight_render_pass_);
  } else {
    blending_weight_fbo_->resize(width, height);
  }

  auto* texture =
      aa_fb_attachment_mgr.getAttachmentTexture(gfx::Framebuffer::Attachment::kColor0);
  CHECK(texture);
  neighborhood_blend_attachment_mgr_.setAttachment(gfx::Framebuffer::Attachment::kColor0,
                                                   texture);

  // Final blend FBO and renderpasses
  if (neighborhood_blend_first_render_pass_ == nullptr) {
    neighborhood_blend_first_render_pass_ =
        resource_mgr.createRenderPass("SMAA Blend First",
                                      neighborhood_blend_attachment_mgr_.getLayout(),
                                      gfx::RenderPass::ClearBits::kAll,
                                      gfx::ImageLayout::kUndefined,
                                      gfx::ImageLayout::kAttachment,
                                      {{{gfx::Framebuffer::Attachment::kColor0}}});
    neighborhood_blend_first_pipeline_->create(*neighborhood_blend_first_render_pass_);
  }

  if (neighborhood_blend_other_render_pass_ == nullptr) {
    neighborhood_blend_other_render_pass_ =
        resource_mgr.createRenderPass("SMAA Blend Other",
                                      neighborhood_blend_attachment_mgr_.getLayout(),
                                      gfx::RenderPass::ClearBits::kNone,
                                      gfx::ImageLayout::kAttachment,
                                      gfx::ImageLayout::kAttachment,
                                      {{{gfx::Framebuffer::Attachment::kColor0}}});
    neighborhood_blend_other_pipeline_->create(*neighborhood_blend_other_render_pass_);
  }

  if (neighborhood_blend_fbo_ == nullptr) {
    neighborhood_blend_fbo_ =
        resource_mgr.createFramebuffer("SMAA Blend",
                                       *neighborhood_blend_first_render_pass_,
                                       neighborhood_blend_attachment_mgr_,
                                       width,
                                       height,
                                       1);
  } else {
    neighborhood_blend_fbo_->resize(width, height);
  }
}

void QueryRenderSMAAPass::GpuData::destroyResources() {
  auto& resource_mgr = device_ctx_.getResourceManager();

  // destroy renderpasses
  if (edge_detect_render_pass_) {
    resource_mgr.destroyRenderPass(std::move(edge_detect_render_pass_));
  }
  if (blending_weight_render_pass_) {
    resource_mgr.destroyRenderPass(std::move(blending_weight_render_pass_));
  }
  if (neighborhood_blend_first_render_pass_) {
    resource_mgr.destroyRenderPass(std::move(neighborhood_blend_first_render_pass_));
  }
  if (neighborhood_blend_other_render_pass_) {
    resource_mgr.destroyRenderPass(std::move(neighborhood_blend_other_render_pass_));
  }

  // destroy standalone textures
  if (area_texture_) {
    resource_mgr.destroyTexture(std::move(area_texture_));
  }
  if (search_texture_) {
    resource_mgr.destroyTexture(std::move(search_texture_));
  }

  // destroy FBOs
  if (edge_detect_fbo_) {
    resource_mgr.destroyFramebuffer(std::move(edge_detect_fbo_));
  }
  if (blending_weight_fbo_) {
    resource_mgr.destroyFramebuffer(std::move(blending_weight_fbo_));
  }
  if (neighborhood_blend_fbo_) {
    resource_mgr.destroyFramebuffer(std::move(neighborhood_blend_fbo_));
  }

  // clear AttachmentManagers
  edge_detect_attachment_mgr_.clear();
  blending_weight_attachment_mgr_.clear();
  neighborhood_blend_attachment_mgr_.clear();

  // destroy FBO textures
  if (edge_detection_texture_) {
    resource_mgr.destroyTexture(std::move(edge_detection_texture_));
  }
  if (blending_weight_texture_) {
    resource_mgr.destroyTexture(std::move(blending_weight_texture_));
  }

  // destroy materials
  edge_detection_material_ = nullptr;
  blending_weight_material_ = nullptr;
  neighborhood_blend_material_ = nullptr;

  // destroy pipelines
  if (edge_detection_pipeline_) {
    resource_mgr.destroyPipeline(std::move(edge_detection_pipeline_));
  }
  if (blending_weight_pipeline_) {
    resource_mgr.destroyPipeline(std::move(blending_weight_pipeline_));
  }
  if (neighborhood_blend_first_pipeline_) {
    resource_mgr.destroyPipeline(std::move(neighborhood_blend_first_pipeline_));
  }
  if (neighborhood_blend_other_pipeline_) {
    resource_mgr.destroyPipeline(std::move(neighborhood_blend_other_pipeline_));
  }
}

QueryRenderSMAAPass::QueryRenderSMAAPass(const GlobalRenderContext& global_ctx,
                                         SMAAQualityPreset quality_preset,
                                         SMAAEdgeDetectionType edge_detect_type)
    : global_ctx_{global_ctx}
    , quality_preset_{quality_preset}
    , edge_detect_type_{edge_detect_type}
    , use_predication_{false}
    , use_reprojection_{false}
    , num_samples_{global_ctx.getNumSamples()}
    , initialized_{false} {
  CHECK(num_samples_ == 1 || num_samples_ == 2 || num_samples_ == 4)
      << "SMAA Anti-aliasing is currently only supported for 1, 2, or 4 samples "
         "per-pixel, not "
      << num_samples_ << " samples.";

  edge_detection_pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
  edge_detection_pipeline_descriptor_->setEnableBlend(false);

  blending_weight_pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
  blending_weight_pipeline_descriptor_->setEnableBlend(false);
  blending_weight_pipeline_descriptor_->setPushConstantRanges(
      {gfx::PushConstantRange{gfx::ShaderStageBits::kFragment, 0, sizeof(uint32_t)}});

  neighborhood_blend_first_pipeline_descriptor_ =
      std::make_unique<gfx::PipelineDescriptor>();
  neighborhood_blend_first_pipeline_descriptor_->setEnableBlend(false);

  neighborhood_blend_other_pipeline_descriptor_ =
      std::make_unique<gfx::PipelineDescriptor>();
  neighborhood_blend_other_pipeline_descriptor_->setEnableBlend(true);
  neighborhood_blend_other_pipeline_descriptor_->setBlendFunc(gfx::BlendFunc::kOne,
                                                              gfx::BlendFunc::kOne);
}

QueryRenderSMAAPass::~QueryRenderSMAAPass() {
  destroyResources();
}

void QueryRenderSMAAPass::initBaseResources() {
  // create shader builders
  std::string quality_preset_str = "\n#define SMAA_PRESET_";
  switch (quality_preset_) {
    case SMAAQualityPreset::kLow:
      quality_preset_str += "LOW";
      break;
    case SMAAQualityPreset::kMedium:
      quality_preset_str += "MEDIUM";
      break;
    case SMAAQualityPreset::kHigh:
      quality_preset_str += "HIGH";
      break;
    case SMAAQualityPreset::kUltra:
      quality_preset_str += "ULTRA";
      break;
  }

  auto const& shader_mgr = global_ctx_.getGfxContext().getShaderManager();

  constexpr static std::array<std::string_view, 3> edge_detect_type_to_func_name = {
      "SMAALumaEdgeDetection", "SMAAColorEdgeDetection", "SMAADepthEdgeDetection"};
  CHECK_LE(static_cast<int>(edge_detect_type_), 3);

  auto edge_detect_builders = shader_mgr.createBuilderVector(
      {{"Rendering/SMAAPassThru.vert"},
       {"Rendering/SMAAEdgeDetection.frag",
        gfx::ShaderManager::Builder::Requirements::kNothing,
        std::string(
            edge_detect_type_to_func_name[static_cast<int>(edge_detect_type_)])}});

  edge_detect_builders[1]->addPreambleString(quality_preset_str);
  edge_detect_builders[1]->replaceFirstTag("numSamples", std::to_string(num_samples_));
  if (use_predication_) {
    edge_detect_builders[1]->addPreambleString("#define SMAA_PREDICATION");
  }
  auto edge_detect_caches = shader_mgr.createCacheVector(std::move(edge_detect_builders));

  // build the blending weight calculation shader
  auto blend_weight_builders =
      shader_mgr.createBuilderVector({{"Rendering/SMAAPassThru.vert"},
                                      {"Rendering/SMAABlendingWeightCalculation.frag"}});

  blend_weight_builders[1]->addPreambleString(quality_preset_str);
  blend_weight_builders[1]->replaceFirstTag("numSamples", std::to_string(num_samples_));
  auto blend_weight_caches =
      shader_mgr.createCacheVector(std::move(blend_weight_builders));

  // build the neighborhood blending shader
  auto blend_builders = shader_mgr.createBuilderVector(
      {{"Rendering/SMAAPassThru.vert"}, {"Rendering/SMAANeighborhoodBlending.frag"}});

  blend_builders[1]->addPreambleString(quality_preset_str);
  if (use_reprojection_) {
    blend_builders[1]->addPreambleString("#define SMAA_REPROJECTION");
  }
  blend_builders[1]->replaceFirstTag("numSamples", std::to_string(num_samples_));
  auto blend_caches = shader_mgr.createCacheVector(std::move(blend_builders));

  //
  // initialize GpuData map, creating the GpuData classes and materials
  //
  auto& root_gpu_data_map = global_ctx_.getRootPerGpuData();
  for (auto& root_gpu_data_itr : root_gpu_data_map) {
    auto const& device_ctx = root_gpu_data_itr->getDeviceContext();
    auto [gpu_data_itr, result] =
        gpu_data_map_.try_emplace(device_ctx.getGpuId(), device_ctx);
    CHECK(result) << "Failed to create GpuData for SMAA pass";
    auto& gpu_data = gpu_data_itr->second;

    auto& resource_mgr = device_ctx.getResourceManager();

    gpu_data.edge_detection_material_ =
        resource_mgr.createMaterial("SMAA Edge Detection", edge_detect_caches);
    gpu_data.blending_weight_material_ =
        resource_mgr.createMaterial("SMAA Blending Weight", blend_weight_caches);
    gpu_data.neighborhood_blend_material_ =
        resource_mgr.createMaterial("SMAA Neighborhood Blend", blend_caches);

    gpu_data.area_texture_ =
        resource_mgr.createTexture(ResourceTrackingString("SMAA Area"),
                                   AREATEX_WIDTH,
                                   AREATEX_HEIGHT,
                                   1,
                                   PixelFormat::kRG8,
                                   1,
                                   false,
                                   ImageUsageBits::kNone,
                                   TextureSamplerState(SamplerFilterMode::kLinear,
                                                       SamplerFilterMode::kLinear,
                                                       SamplerWrapMode::kClampEdge,
                                                       SamplerWrapMode::kClampEdge),
                                   areaTexBytes);

    gpu_data.search_texture_ =
        resource_mgr.createTexture(ResourceTrackingString("SMAA Search"),
                                   SEARCHTEX_WIDTH,
                                   SEARCHTEX_HEIGHT,
                                   1,
                                   PixelFormat::kR8,
                                   1,
                                   false,
                                   ImageUsageBits::kNone,
                                   TextureSamplerState(SamplerFilterMode::kLinear,
                                                       SamplerFilterMode::kLinear,
                                                       SamplerWrapMode::kClampEdge,
                                                       SamplerWrapMode::kClampEdge),
                                   searchTexBytes);

    gpu_data.edge_detection_pipeline_ =
        resource_mgr.createGraphicsPipeline("SMAA Edge Detection",
                                            *gpu_data.edge_detection_material_,
                                            *edge_detection_pipeline_descriptor_);

    gpu_data.blending_weight_pipeline_ =
        resource_mgr.createGraphicsPipeline("SMAA Blending Weight",
                                            *gpu_data.blending_weight_material_,
                                            *blending_weight_pipeline_descriptor_);

    gpu_data.neighborhood_blend_first_pipeline_ = resource_mgr.createGraphicsPipeline(
        "SMAA Blending Weight (first pass)",
        *gpu_data.neighborhood_blend_material_,
        *neighborhood_blend_first_pipeline_descriptor_);

    gpu_data.neighborhood_blend_other_pipeline_ = resource_mgr.createGraphicsPipeline(
        "SMAA Blending Weight (other passes)",
        *gpu_data.neighborhood_blend_material_,
        *neighborhood_blend_other_pipeline_descriptor_);
  }

  initialized_ = true;
}

void QueryRenderSMAAPass::destroyResources() {
  ScopeGuard exit_destroy = [this]() {
    if (initialized_) {
      LOG(WARNING) << "QueryRenderSMAAPass failed to destroy";
    }
    initialized_ = false;
  };

  for (auto& gpu_data_itr : gpu_data_map_) {
    gpu_data_itr.second.destroyResources();
  }
  gpu_data_map_.clear();

  edge_detection_pipeline_descriptor_ = nullptr;
  blending_weight_pipeline_descriptor_ = nullptr;
  neighborhood_blend_first_pipeline_descriptor_ = nullptr;
  neighborhood_blend_other_pipeline_descriptor_ = nullptr;

  initialized_ = false;
}

namespace smaa_attr_names {
static constexpr std::string_view kEdgeTex{"edgeTex"};
static constexpr std::string_view kAreaTex{"areaTex"};
static constexpr std::string_view kSearchTex{"searchTex"};
static constexpr std::string_view kColorTex{"colorTex"};
static constexpr std::string_view kBlendTex{"blendTex"};
}  // namespace smaa_attr_names

void QueryRenderSMAAPass::prepareRenderTargets(uint32_t width, uint32_t height) {
  if (!initialized_) {
    initBaseResources();
    CHECK(initialized_);
  }
  for (auto& gpu_data_itr : gpu_data_map_) {
    auto& gpu_data = gpu_data_itr.second;
    auto* aa_framebuffer = global_ctx_.getGpuData(gpu_data.device_ctx_.getGpuId())
                               .getAntiAliasingFramebuffer();
    gpu_data.prepareRenderTargets(
        width, height, num_samples_, aa_framebuffer->getAttachmentManager());

    // Material binding caches
    gpu_data.blending_weight_material_->setSamplerAttribute(
        smaa_attr_names::kEdgeTex, *gpu_data.edge_detection_texture_);
    gpu_data.blending_weight_material_->setSamplerAttribute(smaa_attr_names::kAreaTex,
                                                            *gpu_data.area_texture_);
    gpu_data.blending_weight_material_->setSamplerAttribute(smaa_attr_names::kSearchTex,
                                                            *gpu_data.search_texture_);

    gpu_data.blending_weight_material_->updateDescriptorSets();
  }
}

void QueryRenderSMAAPass::postPrepareRenderTargets() {
  // Disallow SMAA if not multi-sampling for now
  // Single-sampling support will return once everything flows through the compositor
  auto* separate_multisample_pass = global_ctx_.getSeparateMultiSamplesPass();
  CHECK(separate_multisample_pass);

  // Update descriptors
  for (auto& gpu_data_itr : gpu_data_map_) {
    auto& gpu_data = gpu_data_itr.second;

    auto const& input_texture =
        separate_multisample_pass->getTexture(gpu_data.device_ctx_.getGpuId());

    gpu_data.edge_detection_material_->setSamplerAttribute(smaa_attr_names::kColorTex,
                                                           input_texture);
    gpu_data.edge_detection_material_->updateDescriptorSets();

    gpu_data.neighborhood_blend_material_->setSamplerAttribute(smaa_attr_names::kColorTex,
                                                               input_texture);
    gpu_data.neighborhood_blend_material_->setSamplerAttribute(
        smaa_attr_names::kBlendTex, *gpu_data.blending_weight_texture_);
    gpu_data.neighborhood_blend_material_->updateDescriptorSets();

    gpu_data.neighborhood_blend_material_->setUniformAttribute("sampleWeight",
                                                               1.0f / num_samples_);
  }
}

void QueryRenderSMAAPass::updateViewportUniforms(uint32_t viewport_width,
                                                 uint32_t viewport_height,
                                                 GpuData& gpu_data) {
  auto fwidth = static_cast<float>(viewport_width);
  auto fheight = static_cast<float>(viewport_height);

  auto full_width = static_cast<float>(gpu_data.edge_detection_texture_->getWidth());
  auto full_height = static_cast<float>(gpu_data.edge_detection_texture_->getHeight());

  std::array<float, 4> viewport_metrics({1.0f / fwidth, 1.0f / fheight, fwidth, fheight});
  std::array<float, 4> full_viewport_metrics(
      {1.0f / full_width, 1.0f / full_height, full_width, full_height});

  std::array<float, 2> viewport_scaling(
      {full_width / viewport_width, full_height / viewport_height});

  auto set_viewport_uniforms = [&](gfx::Material& material) {
    material.setUniformAttribute<std::array<float, 4>>("SMAA_RT_METRICS",
                                                       viewport_metrics);
    material.setUniformAttribute<std::array<float, 4>>("FULL_SMAA_RT_METRICS",
                                                       full_viewport_metrics);
    material.setUniformAttribute<std::array<float, 2>>("VIEWPORT_SCALING",
                                                       viewport_scaling);
  };

  set_viewport_uniforms(*gpu_data.edge_detection_material_);
  set_viewport_uniforms(*gpu_data.blending_weight_material_);
  set_viewport_uniforms(*gpu_data.neighborhood_blend_material_);
}

void QueryRenderSMAAPass::runPass(uint32_t viewport_width,
                                  uint32_t viewport_height,
                                  const gfx::DeviceContext& device_ctx,
                                  SeparateMultiSamplesPass::SourceFramebuffer source) {
  CHECK(initialized_);

  auto gpu_id = device_ctx.getGpuId();
  auto& gpu_data = gpu_data_map_.at(gpu_id);
  auto& cmd_list = device_ctx.getCommandList();

  RENDER_LOG_SCOPE_P(gpu_id);

  // Disallow SMAA if not multi-sampling for now
  // Single-sampling support will return once everything flows through the compositor
  auto* separate_multisample_pass = global_ctx_.getSeparateMultiSamplesPass();
  CHECK(separate_multisample_pass);

  // Viewport
  updateViewportUniforms(viewport_width, viewport_height, gpu_data);

  for (uint32_t i = 0; i < num_samples_; ++i) {
    // Extract sample
    separate_multisample_pass->runPass(gpu_id,
                                       cmd_list,
                                       source,
                                       SeparateMultiSamplesPass::OutputUsage::kShaderRead,
                                       i,
                                       {},
                                       {});

    // Edge detection
    cmd_list.pushLabel("SMAA edge")
        .beginRenderPass(*gpu_data.edge_detect_render_pass_, *gpu_data.edge_detect_fbo_)
        .drawFullscreen(*gpu_data.edge_detection_pipeline_)
        .endRenderPass()
        .popLabel();

    // Blending weight calc
    cmd_list.pushLabel("SMAA weights")
        .setPushConstantUInt32(*gpu_data.blending_weight_pipeline_,
                               "subsampleIndicesIndex",
                               gfx::ShaderStageBits::kFragment,
                               i)
        .beginRenderPass(*gpu_data.blending_weight_render_pass_,
                         *gpu_data.blending_weight_fbo_)
        .drawFullscreen(*gpu_data.blending_weight_pipeline_)
        .endRenderPass()
        .popLabel();

    // Blend
    cmd_list.pushLabel("SMAA blend");
    if (i == 0) {
      cmd_list
          .beginRenderPass(*gpu_data.neighborhood_blend_first_render_pass_,
                           *gpu_data.neighborhood_blend_fbo_)
          .drawFullscreen(*gpu_data.neighborhood_blend_first_pipeline_);
    } else {
      cmd_list
          .beginRenderPass(*gpu_data.neighborhood_blend_other_render_pass_,
                           *gpu_data.neighborhood_blend_fbo_)
          .drawFullscreen(*gpu_data.neighborhood_blend_other_pipeline_);
    }
    cmd_list.endRenderPass().popLabel();

    // Submit on each loop iteration for order guarantee
    // Moving to a single submit will require either semaphores or more explicit subpass
    // barriers
    cmd_list.flush("SMAA pass", gfx::CommandList::SubmitType::kImmediateReturn);
  }

  gpu_data.device_ctx_.getCommandExecutor().waitForCompletion(false);
}

}  // namespace QueryRenderer
