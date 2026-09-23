/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/MultiGpuCompositor.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "Logger/Logger.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Renderer.h"
#include "QueryRenderer/Rendering/AccumRenderer.h"
#include "QueryRenderer/Rendering/QueryFramebuffer.h"
#include "QueryRenderer/ResourceTracking.h"
#include "QueryRenderer/Scales/ScaleAccumRenderState.h"
#include "Shared/scope.h"

#define PROFILE_GPU_CALLBACK false
#define PROFILE_RENDER false

#if PROFILE_GPU_CALLBACK || PROFILE_RENDER
#include <iostream>
#include "Shared/measure.h"
#endif

using ::gfx::ImageUsageBits;
using ::gfx::PixelFormat;
using ::gfx::SamplerFilterMode;
using ::gfx::SamplerWrapMode;

namespace QueryRenderer {

MultiGpuCompositor::MultiGpuCompositor(const GlobalRenderContext& global_context,
                                       const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const bool use_last_gpu)
    : global_context_{global_context}
    , cuda_mgr_{cuda_mgr}
    , comp_gpu_data_{global_context.getGpuData(
          use_last_gpu ? global_context.getLastGpuId() : global_context.getStartGpuId())}
    , comp_device_ctx_{comp_gpu_data_.getDeviceContext()}
    , start_gpu_id_{global_context.getStartGpuId()}
    , raster_sample_count_{global_context_.getRasterSampleCount()}
    , num_samples_{gfx::raster_sample_count_enum_to_value(raster_sample_count_)}
    , are_resources_complete_{false}
    , render_target_width_{0}
    , render_target_height_{0} {
  CHECK(cuda_mgr) << "cuda is currently required for multi-gpu compositing";
  LOG(INFO) << "Multi-GPU compositor using GPU " << comp_device_ctx_.getGpuId();
  RENDER_LOG() << "Using gpu " << comp_device_ctx_.getGpuId()
               << " for multi-gpu compositing";

  // Create the internal compositing framebuffer
  // Depth is enabled to allow it to work interchangably with the msFramebuffer
  // when calling AccumRenderer::render. Once all renders flow through the compositor
  // it should be possible to disable it
  framebuffer_ = std::make_unique<QueryFramebuffer>(
      comp_device_ctx_,
      "MultiGpuCompositor",
      raster_sample_count_,
      QueryFramebuffer::CreateFlags::kSupportHitTest |
          QueryFramebuffer::CreateFlags::kSupportDepthTest |
          QueryFramebuffer::CreateFlags::kSupportStencil);

  initRenderPasses();
  initMaterialResources();
  initPipelines();
}

MultiGpuCompositor::~MultiGpuCompositor() {
  destroyPipelines();
  destroyRenderPasses();
  purgeResourceCache();
}

const gfx::DeviceContext& MultiGpuCompositor::getDeviceContext() const {
  return comp_device_ctx_;
}

QueryFramebuffer& MultiGpuCompositor::getFramebuffer() {
  CHECK(framebuffer_);
  return *framebuffer_;
}

void MultiGpuCompositor::destroyBaseTextures() {
  auto& resource_mgr = comp_device_ctx_.getResourceManager();

  for (auto& tex : rgba_textures_) {
    if (tex) {
      resource_mgr.destroyTexture(std::move(tex));
    }
  }
  rgba_textures_.resize(0);

  for (auto& tex : id_textures_) {
    if (tex) {
      resource_mgr.destroyTexture(std::move(tex));
    }
  }
}

void MultiGpuCompositor::destroyAccumTextureArray() {
  auto& resource_mgr = comp_device_ctx_.getResourceManager();

  if (accumulation_copy_texture_) {
    resource_mgr.destroyTexture(std::move(accumulation_copy_texture_));
  }
}

void MultiGpuCompositor::initRenderPasses() {
  auto& resource_mgr = comp_device_ctx_.getResourceManager();
  auto const& layout = framebuffer_->getFramebufferLayout();
  render_pass_clear_ = resource_mgr.createRenderPass("MultiGpuCompositor_All_Clear",
                                                     layout,
                                                     gfx::RenderPass::ClearBits::kAll,
                                                     gfx::ImageLayout::kUndefined,
                                                     gfx::ImageLayout::kGeneral);
}

void MultiGpuCompositor::destroyRenderPasses() {
  auto& resource_mgr = comp_device_ctx_.getResourceManager();
  if (render_pass_clear_) {
    resource_mgr.destroyRenderPass(std::move(render_pass_clear_));
  }
}

void MultiGpuCompositor::prepareRenderTargets(uint32_t width,
                                              uint32_t height,
                                              const QueryRendererContext& render_context,
                                              const std::set<GpuId>& used_gpus) {
  RENDER_LOG_SCOPE_P(used_gpus);

  // If caches are incomplete, call purge to ensure there is no
  // partial state remaining from an exception
  if (!are_resources_complete_) {
    purgeResourceCache();
  } else {
    are_resources_complete_ = false;
  }

  // check if all used_gpus have been registered
  bool did_used_gpus_change = false;
  for (auto id : used_gpus) {
    if (registered_gpus_.find(id) == registered_gpus_.end()) {
      did_used_gpus_change = true;
      break;
    }
  }

  // Check for render target size change
  bool did_render_target_size_change =
      (render_target_width_ != width) || (render_target_height_ != height);

  // Purge everything for major structural changes
  if (did_used_gpus_change || did_render_target_size_change) {
    RENDER_LOG() << "rebuilding render targets";
    purgeResourceCache();

    registered_gpus_ = used_gpus;
    auto num_gpus = registered_gpus_.size();

    render_target_width_ = width;
    render_target_height_ = height;

    if (num_gpus > 1) {
      createBaseTextures();
      framebuffer_->resize(width, height);
    }
  }

  // Check if the required number accumulation textures has changed
  auto required_num_accum_textures = render_context.getNumRequiredAccumulatorTextures();

  // Prepare for accumulation
  auto num_gpus = registered_gpus_.size();
  if (!accumulation_copy_texture_ && required_num_accum_textures) {
    createAccumTexture();
  }

  // init transfer context and fill bitmap
  if (num_gpus > 1) {
    initTransferContext(render_context);
    // tiling is Vulkan only
    if (transfer_ctx_) {
      // Adjust tiling to match the current render size to better optimize
      // tiles given the current tile buffer size
      transfer_ctx_->updateTileQueue(render_context.getWidth(),
                                     render_context.getHeight());
    }

    initMaterialResourceBindingCaches();
  }

  are_resources_complete_ = true;
}

void MultiGpuCompositor::purgeResourceCache() {
  registered_gpus_.clear();
  // destroy transfer contexts prior to destroying texture arrays
  // so cuda resources are disconnected
  transfer_ctx_ = nullptr;

  destroyBaseTextures();
  destroyAccumTextureArray();
}

static constexpr std::array<FboAttachment, 3> id_attachment_types = {FboAttachment::ID1A,
                                                                     FboAttachment::ID1B,
                                                                     FboAttachment::ID2};
void MultiGpuCompositor::initTransferContext(const QueryRendererContext& render_context) {
  if (!transfer_ctx_) {
    transfer_ctx_ = TextureTransferContext::create(
        global_context_, cuda_mgr_, comp_device_ctx_, "Vulkan");
    std::vector<const gfx::Texture*> src_textures(id_attachment_types.size());
    for (auto gpu_id : registered_gpus_) {
      auto const& gpu_data = global_context_.getGpuData(gpu_id);
      auto* src_framebuffer = gpu_data.getAntiAliasingFramebuffer();
      for (size_t i = 0; i < id_attachment_types.size(); ++i) {
        src_textures[i] = src_framebuffer->getTexture(id_attachment_types[i]);
      }

      transfer_ctx_->registerSourceDevice(gpu_data, src_textures);
    }
    transfer_ctx_->buildResources(render_target_width_, render_target_height_);
  }
}

void MultiGpuCompositor::createBaseTextures() {
  auto create_texture = [this](auto name, auto format, int index = -1) {
    return comp_device_ctx_.getResourceManager().createTexture(
        ResourceTrackingString(name, index),
        render_target_width_,
        render_target_height_,
        1,
        format,
        1,
        false,
        ImageUsageBits::kExternalApiBit | ImageUsageBits::kStorageBit,
        gfx::get_default_sampler_state_for_format(format));
  };

  rgba_textures_.resize(num_samples_);
  for (uint32_t i = 0; i < num_samples_; ++i) {
    rgba_textures_[i] = create_texture("MultiGpuCompositor RGBA", PixelFormat::kRGBA8, i);
  }

  if (framebuffer_->supportsHitTest()) {
    id_textures_.resize(IDIndex::kCount);

    id_textures_[IDIndex::k1A] =
        create_texture("MultiGpuCompositor ID1A", PixelFormat::kR32UI);
    id_textures_[IDIndex::k1B] =
        create_texture("MultiGpuCompositor ID1B", PixelFormat::kR32UI);
    id_textures_[IDIndex::k2] =
        create_texture("MultiGpuCompositor ID2", PixelFormat::kR32UI);

    for (auto& tex : id_textures_) {
      tex->clearPixels();
    }
  }
}

void MultiGpuCompositor::createAccumTexture() {
  auto& resource_mgr = comp_device_ctx_.getResourceManager();

  accumulation_copy_texture_ = resource_mgr.createTexture(
      ResourceTrackingString("MultiGpuCompositor Accum Cp"),
      render_target_width_,
      render_target_height_,
      1,
      PixelFormat::kR32UI,
      1,
      false,
      ImageUsageBits::kStorageBit | ImageUsageBits::kExternalApiBit,
      gfx::get_default_sampler_state_for_format(PixelFormat::kR32UI));

  // Call clearPixels on the copy array which will transition it to General layout
  // accumulation_texture_array_ is marked as dirty during in the ctor and will be
  // cleared later
  accumulation_copy_texture_->clearPixels();
}

void MultiGpuCompositor::initMaterialResources() {
  auto& shader_mgr = global_context_.getGfxContext().getShaderManager();

  // Create RGBA and ID composite compute shader materials
  auto num_samples_str = std::to_string(num_samples_);
  auto do_multisample_str = std::to_string(num_samples_ > 1);
  auto subgroup_size_str = std::to_string(comp_device_ctx_.getLimits().subgroup_size);

  auto create_base_material = [&](auto shader_template, auto material_name) {
    auto builders = shader_mgr.createBuilderVector({{shader_template}});
    builders[0]->replaceFirstTag("numSamples", num_samples_str);
    builders[0]->replaceFirstTag("doMultiSample", do_multisample_str);
    builders[0]->replaceFirstTag("workgroupSize", subgroup_size_str);
    auto caches = shader_mgr.createCacheVector(std::move(builders));

    return comp_device_ctx_.getResourceManager().createMaterial(material_name, caches);
  };

  compositor_material_ =
      create_base_material("Rendering/MSMultiGpuComposite.comp", "Multi GPU Comp");
  if (framebuffer_->supportsHitTest()) {
    compositor_id_material_ =
        create_base_material("Rendering/multiGpuCompositeIDs.comp", "Multi GPU ID Comp");
  }

  // Create builders and material for peer accumulation compositing pass
  auto accum_caches = shader_mgr.createCacheVectorFromTemplate(
      {{"Rendering/fullScreenTriangle.vert"},
       {"Rendering/accumulationCompositePeer.frag"}});

  accumulator_peer_material_ = comp_device_ctx_.getResourceManager().createMaterial(
      "Multi GPU Comp Peer Accum", accum_caches);
}

void MultiGpuCompositor::initMaterialResourceBindingCaches() {
  // Compositor materials
  CHECK(compositor_material_);
  CHECK(compositor_id_material_);
  CHECK_GT(rgba_textures_.size(), 0u);

  // Clear writers to handle cases where ID enable/disable state changes
  // which can result in attempting to set a descriptor to VK_NULL_HANDLE
  compositor_material_->setSamplerArrayAttribute("srcRGBA", rgba_textures_);
  compositor_material_->setImageLoadStoreAttribute(
      "outputColor", *framebuffer_->getTexture(FboAttachment::Color));

  // TODO: move these to a different descriptor set in vulkan so we can skip
  // them if a render doesn't require them
  CHECK(id_textures_[IDIndex::k1A]);
  CHECK(id_textures_[IDIndex::k1B]);
  CHECK(id_textures_[IDIndex::k2]);

  auto set_id_resources = [this](auto& material) {
    material.setSamplerAttribute("srcIDA", *id_textures_[IDIndex::k1A]);
    material.setSamplerAttribute("srcIDB", *id_textures_[IDIndex::k1B]);
    material.setSamplerAttribute("srcResultCacheId", *id_textures_[IDIndex::k2]);
    material.setImageLoadStoreAttribute("outputIDA",
                                        *framebuffer_->getTexture(FboAttachment::ID1A));
    material.setImageLoadStoreAttribute("outputIDB",
                                        *framebuffer_->getTexture(FboAttachment::ID1B));
    material.setImageLoadStoreAttribute("outputResultCacheId",
                                        *framebuffer_->getTexture(FboAttachment::ID2));
  };

  set_id_resources(*compositor_material_);
  set_id_resources(*compositor_id_material_);

  if (framebuffer_->supportsHitTest()) {
    compositor_id_material_->updateDescriptorSets();
  }

  compositor_material_->updateDescriptorSets();
}

void MultiGpuCompositor::initPipelines() {
  //
  // pipeline descriptors
  //
  if (!accumulator_pipeline_descriptor_) {
    CHECK(!accumulator_pipeline_descriptor_);
    accumulator_pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
    CHECK(accumulator_pipeline_descriptor_);
  }

  accumulator_pipeline_descriptor_->setPushConstantRanges(
      {gfx::PushConstantRange{gfx::ShaderStageBits::kFragment, 0, sizeof(int32_t)}});

  //
  // pipelines
  //
  compositor_pipeline_ = comp_device_ctx_.getResourceManager().createComputePipeline(
      "Multi-comp composite pass", *compositor_material_);

  compositor_id_pipeline_ = comp_device_ctx_.getResourceManager().createComputePipeline(
      "Multi-comp composite pass (ID)", *compositor_id_material_);

  accumulator_peer_pipeline_ =
      comp_device_ctx_.getResourceManager().createGraphicsPipeline(
          "Multi-comp peer accumulation pass",
          *accumulator_peer_material_,
          *accumulator_pipeline_descriptor_);

  compositor_pipeline_->create();
  compositor_id_pipeline_->create();

  auto const& empty_renderpass = comp_gpu_data_.getEmptyRenderPass();
  accumulator_peer_pipeline_->create(empty_renderpass);
}

void MultiGpuCompositor::destroyPipelines() {
  if (compositor_pipeline_) {
    comp_device_ctx_.getResourceManager().destroyPipeline(
        std::move(compositor_pipeline_));
  }
  if (compositor_id_pipeline_) {
    comp_device_ctx_.getResourceManager().destroyPipeline(
        std::move(compositor_id_pipeline_));
  }
  if (accumulator_peer_pipeline_) {
    comp_device_ctx_.getResourceManager().destroyPipeline(
        std::move(accumulator_peer_pipeline_));
  }
  accumulator_pipeline_descriptor_ = nullptr;
}

void MultiGpuCompositor::compAccumLayerCallback(
    uint32_t layer_index,
    const std::vector<gfx::SemaphoreHandle>& signal_semaphores) {
  VLOG(1) << "Begin accumulation layer composite: " << layer_index;
  auto [empty_renderpass, empty_fbo] = comp_gpu_data_.getEmptyRenderPassAndFramebuffer();
  comp_device_ctx_.getCommandList()
      .beginRenderPass(empty_renderpass, empty_fbo)
      .setPushConstantUInt32(*accumulator_peer_pipeline_,
                             "layerIndex",
                             gfx::ShaderStageBits::kFragment,
                             layer_index)
      .drawFullscreen(*accumulator_peer_pipeline_)
      .endRenderPass()
      .flush("MultiGpuComp accum pass",
             gfx::CommandList::SubmitType::kImmediateReturn,
             {},
             signal_semaphores);
}

void MultiGpuCompositor::compColorAndIDs(uint32_t width,
                                         uint32_t height,
                                         bool should_clear_dst) {
  VLOG(1) << "Begin compositing color and IDs\n";
  auto& cmd_list = comp_device_ctx_.getCommandList();
  if (should_clear_dst) {
    cmd_list.beginRenderPass(*render_pass_clear_, *framebuffer_->getFramebuffer())
        .endRenderPass()
        .flush("MultiGpuComp clear fbo", gfx::CommandList::SubmitType::kImmediateReturn);
  } else {
    cmd_list.transitionFramebufferLayout(*framebuffer_->getFramebuffer(),
                                         gfx::ImageLayout::kGeneral);
  }
  auto subgroup_size = comp_device_ctx_.getLimits().subgroup_size;

  cmd_list
      .dispatchCompute(*compositor_pipeline_,
                       0u,
                       (width + subgroup_size - 1) / subgroup_size,
                       (height + subgroup_size - 1) / subgroup_size,
                       1u)
      .transitionFramebufferLayout(*framebuffer_->getFramebuffer(),
                                   gfx::ImageLayout::kAttachment)
      .flush("MultiGpuComp rgba pass", gfx::CommandList::SubmitType::kImmediateReturn);
}

void MultiGpuCompositor::compIDs(uint32_t width, uint32_t height, bool should_clear_dst) {
  auto& cmd_list = comp_device_ctx_.getCommandList();

  // Transition framebuffer attachments to general layou for use as storage images
  // Clear if necessary
  if (should_clear_dst) {
    cmd_list
        .clearTexture(*framebuffer_->getTexture(FboAttachment::ID1A),
                      gfx::ImageLayout::kGeneral)
        .clearTexture(*framebuffer_->getTexture(FboAttachment::ID1B),
                      gfx::ImageLayout::kGeneral)
        .clearTexture(*framebuffer_->getTexture(FboAttachment::ID2),
                      gfx::ImageLayout::kGeneral);
  } else {
    cmd_list
        .imageMemoryBarrier(*framebuffer_->getTexture(FboAttachment::ID1A),
                            gfx::ImageMemoryBarrierType::kFragmentShaderToCompute,
                            gfx::ImageLayout::kGeneral)
        .imageMemoryBarrier(*framebuffer_->getTexture(FboAttachment::ID1B),
                            gfx::ImageMemoryBarrierType::kFragmentShaderToCompute,
                            gfx::ImageLayout::kGeneral)
        .imageMemoryBarrier(*framebuffer_->getTexture(FboAttachment::ID2),
                            gfx::ImageMemoryBarrierType::kFragmentShaderToCompute,
                            gfx::ImageLayout::kGeneral);
  }
  auto subgroup_size = comp_device_ctx_.getLimits().subgroup_size;

  cmd_list.dispatchCompute(*compositor_id_pipeline_,
                           0u,
                           (width + subgroup_size - 1) / subgroup_size,
                           (height + subgroup_size - 1) / subgroup_size,
                           1u);

  // Transition attachments back to attachment optimal layout which is the
  // layout expected by the rest of the renderer
  cmd_list
      .imageMemoryBarrier(*framebuffer_->getTexture(FboAttachment::ID1A),
                          gfx::ImageMemoryBarrierType::kComputeToFragmentShader,
                          gfx::ImageLayout::kAttachment)
      .imageMemoryBarrier(*framebuffer_->getTexture(FboAttachment::ID1B),
                          gfx::ImageMemoryBarrierType::kComputeToFragmentShader,
                          gfx::ImageLayout::kAttachment)
      .imageMemoryBarrier(*framebuffer_->getTexture(FboAttachment::ID2),
                          gfx::ImageMemoryBarrierType::kComputeToFragmentShader,
                          gfx::ImageLayout::kAttachment);

  cmd_list.flush("MultiGpuComp id pass", gfx::CommandList::SubmitType::kImmediateReturn);
}

void MultiGpuCompositor::postPassPerGpuCB(const gfx::DeviceContext& src_device_ctx,
                                          QueryFramebuffer& src_framebuffer,
                                          const QueryRendererContext& render_context,
                                          const bool should_clear_dst,
                                          const bool should_comp,
                                          ScaleAccumRenderState* scale_accum_render_state,
                                          const int accumulator_index) {
  auto src_gpu_id = src_device_ctx.getGpuId();
  RENDER_LOG_SCOPE_P(src_gpu_id);
  std::string src_gpu_id_str("[gpu " + std::to_string(src_gpu_id) + "]");
  VLOG(1) << "MultiGpuCompositor post render pass " << src_gpu_id_str;

  auto& src_gpu_data = global_context_.getGpuData(src_gpu_id);

  auto viewport_width = render_context.getWidth();
  auto viewport_height = render_context.getHeight();

#if PROFILE_GPU_CALLBACK
  comp_device_ctx_.getCommandExecutor().waitForCompletion(true);
  auto clock_begin = timer_start();
  ScopeGuard profile_guard = [&]() {
    auto const wall_time =
        timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
            clock_begin);
    std::cout << "GPU[" << src_gpu_id << "] callback time: " << wall_time << std::endl;
  };
#endif

  // Ensure source gpu is done drawing
  // TODO(scb): This is not strictly necessary, as command submission provides some
  // ordering guarantees. It appears to work correctly without this call, however it would
  // be safer if there were draw semaphores that transfer context operations could wait
  // on, so I'm leaving it for now out of paranoia. Cost is minimal
  src_device_ctx.getCommandExecutor().waitForCompletion(false);
  VLOG(1) << "Pending render commands completed " << src_gpu_id_str;

  if (scale_accum_render_state) {
    RENDER_LOG() << "compositing accumulation " << src_gpu_id_str;
    auto const is_peer_gpu = src_gpu_id != comp_device_ctx_.getGpuId();
    auto const* src_texture_array = src_gpu_data.getAccumTextureArray();

    // Update descriptors to use either the local accumulation texture array or the copy
    // from the transfer context
    // TODO: use two materials, one for peer and one for local and move
    // these descriptor updates out of here
    if (is_peer_gpu) {
      accumulator_peer_material_->setImageLoadStoreAttribute("srcAccumTx",
                                                             *accumulation_copy_texture_);
      accumulator_peer_material_->setImageLoadStoreAttribute(
          "inTxArrayPixelCounter", *comp_gpu_data_.getAccumTextureArray());
      accumulator_peer_material_->updateDescriptorSets();
    }

    auto layer_cb = [this](auto&&... args) {
      compAccumLayerCallback(std::forward<decltype(args)>(args)...);
    };

    // Copy the source TextureArray into the local copy TextureArray
    if (is_peer_gpu) {
      VLOG(1) << "Begin accumulation texture transfer " << src_gpu_id_str;
      // Copy accumulation texture layers from the source device. As each layer completes
      // the transfer context will call compAccumLayerCB to add the layer to the
      // compositor's texture array
      transfer_ctx_->copyTextureArrayFromDevice(
          src_gpu_id,
          *src_texture_array,
          *accumulation_copy_texture_,
          scale_accum_render_state->getNumTextures(),
          layer_cb);
    }

    if (render_context.doHitTest()) {
      // Accumulation writes Ids to single-sample (AA) fbo so no resolve needed
      VLOG(1) << "Begin accumulation ID texture transfer " << src_gpu_id_str;
      transfer_ctx_->copyTexturesFromDevice(
          src_gpu_id, TextureTransferContext::SyncType::kID, id_textures_);
      compIDs(viewport_width, viewport_height, should_clear_dst);
    }
  } else {
    compositor_material_->setUniformAttribute("shouldComp", should_comp ? 1 : 0);
    RENDER_LOG() << "Compositing rgba texture " << src_gpu_id_str;
    if (raster_sample_count_ != gfx::RasterSampleCount::k1) {
      VLOG(1) << "Begin RGBA texture transfer " << src_gpu_id_str;

      transfer_ctx_->copyTexturesFromDevice(
          src_gpu_id, TextureTransferContext::SyncType::kRGBA, rgba_textures_);

      if (render_context.doHitTest()) {
        // multisample resolve src Ids into src single sample framebuffer
        auto ss_fbo = src_gpu_data.getAntiAliasingFramebuffer();
        src_framebuffer.copyToFramebuffer(
            *(ss_fbo), 0, 0, viewport_width, viewport_height, false, true, false);
        VLOG(1) << "Begin ID texture transfer " << src_gpu_id_str;
        transfer_ctx_->copyTexturesFromDevice(
            src_gpu_id, TextureTransferContext::SyncType::kID, id_textures_);
      }
      compColorAndIDs(viewport_width, viewport_height, should_clear_dst);
    } else {
      // TODO(scb): Fix rendering without multi-sampling
      CHECK(false) << "Disabling multi-sampling is not supported";
    }
  }

  // Ensure all pending vulkan command buffers have completed
  if (transfer_ctx_) {
    transfer_ctx_->waitComplete();
    VLOG(1) << "Texture transfer operations complete " << src_gpu_id_str;
  }

  // Ensure any command buffers on the source gpu are cleaned up
  src_gpu_data.getDeviceContext().getCommandExecutor().waitForCompletion(
      PROFILE_GPU_CALLBACK);
  VLOG(1) << "All source GPU operations complete " << src_gpu_id_str;

  // Finally wait for the compositor commands
  comp_device_ctx_.getCommandExecutor().waitForCompletion(PROFILE_GPU_CALLBACK);
  VLOG(1) << "Composite complete " << src_gpu_id_str;
}

void MultiGpuCompositor::compositePass(const std::set<GpuId>& render_used_gpus,
                                       const std::set<GpuId>& pass_used_gpus,
                                       const QueryRendererContext& render_context,
                                       const int pass_index,
                                       ScaleAccumRenderState* scale_accum_render_state) {
  RENDER_LOG_SCOPE_P(pass_used_gpus) << " pass_index: " << pass_index;

  auto viewport_width = render_context.getWidth();
  auto viewport_height = render_context.getHeight();

  // TODO(scb): these checks are now superfluous
  CHECK_LE(viewport_width, render_target_width_);
  CHECK_LE(viewport_height, render_target_height_);

  if (pass_used_gpus.size() > 0) {
    if (scale_accum_render_state) {
      // The color attachment will still be in general layout, so transition it back
      // to attachment optimal which is required for the AccumRenderer color pass
      // and distributed compositor (the later being pass_complete_override_cb)
      // NOTE (scb): the flush should not be required, but we get a VE without it
      // (layers bug?)
      comp_device_ctx_.getCommandList()
          .imageMemoryBarrier(*framebuffer_->getTexture(FboAttachment::Color),
                              gfx::ImageMemoryBarrierType::kImageLayout,
                              gfx::ImageLayout::kAttachment)
          .flush("Comp accum layout", gfx::CommandList::SubmitType::kImmediateReturn);
      if (!pass_complete_override_cb_) {
        RENDER_LOG() << "Run accumulation 2nd pass targetting compositor framebuffer";
        global_context_.getAccumRenderer().render(comp_device_ctx_,
                                                  *scale_accum_render_state,
                                                  *comp_gpu_data_.getAccumTextureArray(),
                                                  *framebuffer_,
                                                  false);
      }
    }
  }
  if (pass_complete_override_cb_) {
    RENDER_LOG() << "calling pass complete override callback";
    pass_complete_override_cb_(render_used_gpus,
                               pass_used_gpus,
                               render_context,
                               pass_index,
                               scale_accum_render_state);
  }
}

void MultiGpuCompositor::render(const QueryRendererContext& ctx,
                                const std::set<GpuId>& used_gpus,
                                PassCompleteCBFunc pass_complete_override_cb) {
  ScopeGuard compositeStateHolder = [this] { pass_complete_override_cb_ = nullptr; };

  RENDER_LOG_SCOPE_P(used_gpus);
  // store the overide callback
  pass_complete_override_cb_ = pass_complete_override_cb;

#if PROFILE_RENDER
  auto start_time = timer_start();

  ScopeGuard profile_guard = [&]() {
    comp_device_ctx_.getCommandExecutor().waitForCompletion(true);
    auto time =
        timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
            start_time);
    std::cout << "time: " << time << std::endl;
  };
#endif

  compositor_material_->setUniformAttribute("imageWidth", ctx.getWidth());
  compositor_material_->setUniformAttribute("imageHeight", ctx.getHeight());

  comp_device_ctx_.getCommandExecutor().setDefaultViewportAndRenderArea(
      0, 0, ctx.getWidth(), ctx.getHeight());

  // Clear the compositor's framebuffer and put everything in general layout
  comp_device_ctx_.getCommandList()
      .beginRenderPass(*render_pass_clear_, *framebuffer_->getFramebuffer())
      .endRenderPass()
      .flush("clear", gfx::CommandList::SubmitType::kImmediateReturn);

  if (ctx.getNumRequiredAccumulatorTextures()) {
    comp_gpu_data_.getAccumTextureArray()->clearPixels();
    compositor_id_material_->setUniformAttribute("imageWidth", ctx.getWidth());
    compositor_id_material_->setUniformAttribute("imageHeight", ctx.getHeight());
  }

  auto per_pass_gpu_cb = [this](auto&&... args) {
    postPassPerGpuCB(std::forward<decltype(args)>(args)...);
  };

  auto pass_complete_cb = [this](auto&&... args) {
    compositePass(std::forward<decltype(args)>(args)...);
  };

  global_context_.getRenderer().renderPasses(
      ctx, used_gpus, per_pass_gpu_cb, pass_complete_cb);
}

}  // namespace QueryRenderer
