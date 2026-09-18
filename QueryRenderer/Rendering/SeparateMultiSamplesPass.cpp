/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/SeparateMultiSamplesPass.h"

#include <string_view>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Rendering/MultiGpuCompositor.h"
#include "QueryRenderer/Rendering/QueryFramebuffer.h"
#include "QueryRenderer/ResourceTracking.h"
#include "Shared/scope.h"

using ::gfx::DeviceContext;
using ::gfx::ImageUsageBits;
using ::gfx::SamplerFilterMode;
using ::gfx::SamplerWrapMode;

namespace QueryRenderer {

static constexpr std::string_view resource_tracking_name = "Separate MS Pass";
static constexpr std::string_view color_tex_attr_name = "colorTex";

SeparateMultiSamplesPass::GpuData::GpuData(const gfx::DeviceContext& device_ctx)
    : device_ctx_{device_ctx} {}

SeparateMultiSamplesPass::GpuData::~GpuData() {
  destroyResources();
}

void SeparateMultiSamplesPass::GpuData::prepareRenderTargets(uint32_t width,
                                                             uint32_t height,
                                                             bool enable_api_export) {
  auto& resource_mgr = device_ctx_.getResourceManager();

  // Create or resize Framebuffer and RenderPass
  if (fbo_ == nullptr) {
    // Populate attachments
    attachment_mgr_.clear();
    ImageUsageBits extra_usage_bits = ImageUsageBits::kColorAttachmentBit;
    if (enable_api_export) {
      extra_usage_bits |= ImageUsageBits::kExternalApiBit;
    }
    texture_ =
        resource_mgr.createTexture(ResourceTrackingString(resource_tracking_name),
                                   width,
                                   height,
                                   1,
                                   gfx::PixelFormat::kRGBA8,
                                   1,
                                   false,
                                   extra_usage_bits,
                                   gfx::TextureSamplerState(SamplerFilterMode::kLinear,
                                                            SamplerFilterMode::kLinear,
                                                            SamplerWrapMode::kClampEdge,
                                                            SamplerWrapMode::kClampEdge));
    attachment_mgr_.setAttachment(gfx::Framebuffer::Attachment::kColor0, texture_.get());
    if (!render_pass_shader_read_) {
      render_pass_shader_read_ =
          resource_mgr.createRenderPass(resource_tracking_name,
                                        attachment_mgr_.getLayout(),
                                        gfx::RenderPass::ClearBits::kNone,
                                        gfx::ImageLayout::kUndefined,
                                        gfx::ImageLayout::kShaderReadOnly);
    }
    if (!render_pass_transfer_src_) {
      render_pass_transfer_src_ =
          resource_mgr.createRenderPass(resource_tracking_name,
                                        attachment_mgr_.getLayout(),
                                        gfx::RenderPass::ClearBits::kNone,
                                        gfx::ImageLayout::kUndefined,
                                        gfx::ImageLayout::kTransferSrc);
    }

    fbo_ = resource_mgr.createFramebuffer(resource_tracking_name,
                                          *render_pass_shader_read_,
                                          attachment_mgr_,
                                          width,
                                          height,
                                          1);

    resources_rndr_.pipeline->create(*render_pass_shader_read_);
    resources_comp_.pipeline->create(*render_pass_shader_read_);
  } else {
    fbo_->resize(width, height);
  }
}

void SeparateMultiSamplesPass::GpuData::postPrepareRenderTargets(
    const GlobalRenderContext& global_ctx,
    const GpuId gpu_id) {
  // this should always exist fully
  resources_rndr_.framebuffer =
      global_ctx.getGpuData(gpu_id).getRenderFramebuffer()->getFramebuffer();
  CHECK(resources_rndr_.framebuffer);
  auto* texture =
      resources_rndr_.framebuffer->getAttachmentManager().getAttachmentTexture(
          gfx::Framebuffer::Attachment::kColor0);
  CHECK(texture);
  resources_rndr_.material->setSamplerAttribute(color_tex_attr_name, *texture);
  resources_rndr_.material->updateDescriptorSets();

  // this will only exist in multi-GPU and only on one GPU
  // tolerate various levels of lazy initialization
  resources_comp_.framebuffer = nullptr;
  auto* compositor = global_ctx.getMultiGpuCompositor();
  if (compositor) {
    if (gpu_id == global_ctx.getCompositorGpuId()) {
      resources_comp_.framebuffer = compositor->getFramebuffer().getFramebuffer();
      if (resources_comp_.framebuffer) {
        auto* texture =
            resources_comp_.framebuffer->getAttachmentManager().getAttachmentTexture(
                gfx::Framebuffer::Attachment::kColor0);
        CHECK(texture);
        resources_comp_.material->setSamplerAttribute(color_tex_attr_name, *texture);
        resources_comp_.material->updateDescriptorSets();
      }
    }
  }
}

void SeparateMultiSamplesPass::GpuData::destroyResources() {
  auto& resource_mgr = device_ctx_.getResourceManager();
  if (render_pass_shader_read_) {
    resource_mgr.destroyRenderPass(std::move(render_pass_shader_read_));
  }
  if (render_pass_transfer_src_) {
    resource_mgr.destroyRenderPass(std::move(render_pass_transfer_src_));
  }

  if (fbo_) {
    resource_mgr.destroyFramebuffer(std::move(fbo_));
  }

  attachment_mgr_.clear();

  if (texture_) {
    resource_mgr.destroyTexture(std::move(texture_));
  }

  if (resources_rndr_.pipeline) {
    resource_mgr.destroyPipeline(std::move(resources_rndr_.pipeline));
  }
  resources_rndr_.framebuffer = nullptr;
  resources_rndr_.material = nullptr;

  if (resources_comp_.pipeline) {
    resource_mgr.destroyPipeline(std::move(resources_comp_.pipeline));
  }
  resources_comp_.framebuffer = nullptr;
  resources_comp_.material = nullptr;
}

SeparateMultiSamplesPass::SeparateMultiSamplesPass(const GlobalRenderContext& global_ctx,
                                                   bool enable_api_export)
    : global_ctx_{global_ctx}
    , enable_api_export_{enable_api_export}
    , initialized_{false} {
  pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
  pipeline_descriptor_->setEnableBlend(false);
  pipeline_descriptor_->setPushConstantRanges(
      {gfx::PushConstantRange{gfx::ShaderStageBits::kFragment, 0, sizeof(uint32_t)}});
}

SeparateMultiSamplesPass::~SeparateMultiSamplesPass() {
  destroyResources();
}

void SeparateMultiSamplesPass::initBaseResources() {
  // create shader builders
  auto const& shader_mgr = global_ctx_.getGfxContext().getShaderManager();
  auto shader_caches = shader_mgr.createCacheVectorFromTemplate(
      {{"Rendering/fullScreenTriangle.vert"}, {"Rendering/SeparateMultiSample.frag"}},
      false);

  //
  // initialize GpuData map
  //
  auto& root_gpu_data_map = global_ctx_.getRootPerGpuData();
  for (auto& root_gpu_data_itr : root_gpu_data_map) {
    auto const& device_ctx = root_gpu_data_itr->getDeviceContext();
    auto [gpu_data_itr, result] =
        gpu_data_map_.try_emplace(device_ctx.getGpuId(), device_ctx);
    CHECK(result) << "Failed to create GpuData for SeparateMultiSamplesPass";
    auto& gpu_data = gpu_data_itr->second;

    auto& resource_mgr = device_ctx.getResourceManager();

    gpu_data.resources_rndr_.material = resource_mgr.createMaterial(
        std::string(resource_tracking_name) + " Rndr", shader_caches);
    gpu_data.resources_rndr_.pipeline =
        resource_mgr.createGraphicsPipeline(std::string(resource_tracking_name) + " Rndr",
                                            *gpu_data.resources_rndr_.material,
                                            *pipeline_descriptor_);

    gpu_data.resources_comp_.material = resource_mgr.createMaterial(
        std::string(resource_tracking_name) + " Comp", shader_caches);
    gpu_data.resources_comp_.pipeline =
        resource_mgr.createGraphicsPipeline(std::string(resource_tracking_name) + " Comp",
                                            *gpu_data.resources_comp_.material,
                                            *pipeline_descriptor_);
  }

  initialized_ = true;
}

void SeparateMultiSamplesPass::destroyResources() {
  ScopeGuard exit_destroy = [this]() {
    if (initialized_) {
      LOG(WARNING) << "SeparateMultiSamplesPass failed to destroy";
    }
    initialized_ = false;
  };

  for (auto& gpu_data_itr : gpu_data_map_) {
    gpu_data_itr.second.destroyResources();
  }
  gpu_data_map_.clear();

  pipeline_descriptor_ = nullptr;

  initialized_ = false;
}

void SeparateMultiSamplesPass::prepareRenderTargets(uint32_t width, uint32_t height) {
  if (!initialized_) {
    initBaseResources();
    CHECK(initialized_);
  }
  for (auto& gpu_data_itr : gpu_data_map_) {
    gpu_data_itr.second.prepareRenderTargets(width, height, enable_api_export_);
  }
}

void SeparateMultiSamplesPass::postPrepareRenderTargets() {
  for (auto& gpu_data_itr : gpu_data_map_) {
    gpu_data_itr.second.postPrepareRenderTargets(global_ctx_, gpu_data_itr.first);
  }
}

const gfx::Texture& SeparateMultiSamplesPass::getTexture(GpuId gpu_id) const {
  return *gpu_data_map_.at(gpu_id).texture_;
}

void SeparateMultiSamplesPass::runPass(
    GpuId gpu_id,
    gfx::CommandList& cmd_list,
    SourceFramebuffer source_fbo,
    OutputUsage usage,
    uint32_t sample_index,
    const std::vector<gfx::SemaphoreHandle>& wait_semaphores,
    const std::vector<gfx::SemaphoreHandle>& signal_semaphores) {
  CHECK(initialized_);
  RENDER_LOG_SCOPE_P(gpu_id);
  auto& gpu_data = gpu_data_map_.at(gpu_id);

  auto& resources = source_fbo == SourceFramebuffer::kRender ? gpu_data.resources_rndr_
                                                             : gpu_data.resources_comp_;
  cmd_list.pushLabel("SepMS pass");
  if (sample_index == 0) {
    cmd_list.transitionFramebufferLayout(*resources.framebuffer,
                                         gfx::ImageLayout::kShaderReadOnly);
  }
  cmd_list
      .setPushConstantUInt32(*resources.pipeline,
                             "sampleIndex",
                             gfx::ShaderStageBits::kFragment,
                             sample_index)
      .beginRenderPass(usage == OutputUsage::kShaderRead
                           ? *gpu_data.render_pass_shader_read_
                           : *gpu_data.render_pass_transfer_src_,
                       *gpu_data.fbo_)
      .drawFullscreen(*resources.pipeline)
      .endRenderPass()
      .popLabel()
      .flush("SeparateMS pass",
             gfx::CommandList::SubmitType::kImmediateReturn,
             wait_semaphores,
             signal_semaphores);
}

}  // namespace QueryRenderer
