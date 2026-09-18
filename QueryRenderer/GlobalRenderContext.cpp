/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GlobalRenderContext.h"

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/DataMgr.h"
#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/DriverInstance.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Render/GeoCountResources.h"
#include "GfxDriver/Render/PPLLConstants.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "QueryRenderer/Cache/ResultCache.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Renderer.h"
#include "QueryRenderer/Rendering/AccumRenderer.h"
#include "QueryRenderer/Rendering/MultiGpuCompositor.h"
#include "QueryRenderer/Rendering/QueryFramebuffer.h"
#include "QueryRenderer/Rendering/QueryIdMapPboPool.h"
#include "QueryRenderer/Rendering/QueryRenderSMAAPass.h"
#include "QueryRenderer/Rendering/SeparateMultiSamplesPass.h"
#include "Shared/DeviceGroup.h"
#include "Shared/scope.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA

namespace QueryRenderer {

using ::gfx::DeviceContext;

std::pair<uint32_t, uint32_t> get_render_target_size_from_render_size(
    uint32_t render_width,
    uint32_t render_height) {
  return {render_width + 1, render_height};
}

GlobalRenderContext::GlobalRenderContext(const gfx::GfxContext& gfx_context,
                                         Data_Namespace::DataMgr* const data_mgr,
                                         CudaMgr_Namespace::CudaMgr* const cuda_mgr,
                                         const size_t render_mem_bytes,
                                         const bool use_last_gpu_for_compositor,
                                         const gfx::RasterSampleCount raster_sample_count,
                                         const bool renderer_enable_slab_allocation)
    : gfx_context_{gfx_context}
    , data_mgr_{data_mgr}
    , cuda_mgr_{cuda_mgr}
    , query_result_cache_{std::make_unique<QueryResultCache>()}
    , render_mem_bytes_{render_mem_bytes}
    , use_last_gpu_for_compositor_{use_last_gpu_for_compositor}
    , raster_sample_count_{raster_sample_count}
    , num_samples_{gfx::raster_sample_count_enum_to_value(raster_sample_count)}
#ifdef HAVE_CUDA
    , renderer_enable_slab_allocation_{renderer_enable_slab_allocation}
#endif
    , render_target_width_{0}
    , render_target_height_{0}
    , accum_tx_array_depth_{0} {
  CHECK(query_result_cache_);
}

GlobalRenderContext::~GlobalRenderContext() {
  clearGpuMemory();
}

void GlobalRenderContext::init() {
  if (areCachesAndResourcesComplete(false)) {
    return;
  }

#ifdef HAVE_CUDA
  auto const& device_group = gfx_context_.getDeviceGroup();
  for (auto const& device_info : device_group) {
    auto const& device_ctx = gfx_context_.getDeviceContext(device_info);
    gpu_data_map_.insert(std::make_unique<RootPerGpuData>(
        device_ctx, data_mgr_, renderer_enable_slab_allocation_));
  }
#else
  // for non-cuda builds, only single gpu rendering is currently supported due to the
  // compositor requiring cuda. So, we'll just use the first available gpu.

  // NOTE: device_group in this context is a vector copy because we're going to modify it.
  auto const& global_device_group = gfx_context_.getDeviceGroup();
  CHECK_GT(global_device_group.size(), size_t(0));
  heavyai::DeviceGroup device_group = {global_device_group[0]};

  auto const& primary_device_info = device_group[0];
  LOG(INFO) << "CUDA-disabled servers are forced to use a single gpu for rendering. "
               "Using uuid: "
            << primary_device_info.uuid;

  // scoping so the context guard is cleared after the device context is created
  {
    auto const& device_ctx = gfx_context_.getDeviceContext(primary_device_info);
    const auto inserted =
        gpu_data_map_.insert(std::make_unique<RootPerGpuData>(device_ctx, nullptr, false))
            .second;
    CHECK(inserted) << primary_device_info.uuid;
  }
#endif  // HAVE_CUDA

  ScopeGuard completeness_check = [this] {
    if (!areCachesAndResourcesComplete(true)) {
      // Log error and clean up partial initialization. Repeated requests will still fail
      // if memory is already exhausted.
      // TODO(scb) - many things, this is a stop gap
      LOG(ERROR) << "Render resources and caches incomplete, unable to render";
      clearGpuMemory();
    }
  };
  initCachesAndResources(device_group);
}

void GlobalRenderContext::initCachesAndResources(
    const heavyai::DeviceGroup& device_group) {
  RENDER_LOG_SCOPE();
  auto const& shader_mgr = gfx_context_.getShaderManager();

  // Subgroup size for compute shaders
  auto const& driver = gfx_context_.getPrimaryDriver();
  auto subgroup_size = driver.getLimits().subgroup_size;
  CHECK_GT(subgroup_size, 0u);
  auto subgroup_size_str = std::to_string(subgroup_size);

  // Extents and StdDev compute shader caches
  static constexpr gfx::DeviceCapabilityBits extents_requires =
      gfx::DeviceCapabilityBits::kSubgroupVote |
      gfx::DeviceCapabilityBits::kSubgroupBallot |
      gfx::DeviceCapabilityBits::kSubgroupArithmetic;
  static const constexpr gfx::DeviceCapabilityBits std_dev_requires =
      gfx::DeviceCapabilityBits::kSubgroupVote |
      gfx::DeviceCapabilityBits::kSubgroupArithmetic |
      gfx::DeviceCapabilityBits::kSubgroupExtendedTypes;

  auto accum_extents_builder =
      shader_mgr.createBuilderVector({{"Rendering/accumulatorTx_findExtents.comp"}});
  auto accum_stddev_builder =
      shader_mgr.createBuilderVector({{"Rendering/accumulatorTx_findStdDev.comp"}});

  accum_extents_builder[0]->replaceFirstTag("workgroupSize", subgroup_size_str);
  accum_stddev_builder[0]->replaceFirstTag("workgroupSize", subgroup_size_str);

  bool extents_use_subgroups = driver.queryCapabilities(extents_requires);
  bool stddev_use_subgroups = driver.queryCapabilities(std_dev_requires);

  accum_extents_builder[0]->replaceFirstTag("useSubgroups",
                                            std::to_string(extents_use_subgroups));
  accum_stddev_builder[0]->replaceFirstTag("useSubgroups",
                                           std::to_string(stddev_use_subgroups));

  auto accum_extents_cache =
      shader_mgr.createCacheVector(std::move(accum_extents_builder));
  auto accum_stddev_cache = shader_mgr.createCacheVector(std::move(accum_stddev_builder));
  CHECK(!accum_extents_cache.empty());
  CHECK(!accum_stddev_cache.empty());

  auto accum_id_pass_caches = shader_mgr.createCacheVectorFromTemplate(
      {{"Rendering/fullScreenTriangle.vert"}, {"Rendering/accumulatorIDPass.frag"}});

  auto clear_query_output_buffer_cache = shader_mgr.createCacheVectorFromTemplate(
      {{"Rendering/clearQueryOutputBuffer.comp"}});

  //
  // PPLL shader caches
  //
  auto create_ppll_builders = [&](const std::string& template_name,
                                  std::string_view ubo_name) {
    auto builders = shader_mgr.createBuilderVector({template_name});
    builders[0]->replaceFirstTag("workgroupSize", subgroup_size_str);
    builders[0]->setExternalUniformBuffers({ubo_name});
    return builders;
  };

  // Stats
  auto shader_builders = create_ppll_builders("PPLL/ppllStats.comp", "IMAGE_INFO_UBO");
  auto ppll_stats_cache = shader_mgr.createCacheVector(std::move(shader_builders));

  // Stats tiled
  shader_builders = create_ppll_builders("PPLL/ppllStatsTiled.comp", "IMAGE_TILES_UBO");
  shader_builders[0]->replaceFirstTag("numTiles", std::to_string(gfx::kPPLLTilesUBOSize));
  auto ppll_tile_stats_cache = shader_mgr.createCacheVector(std::move(shader_builders));

  // Stats tiled batches - stage 1
  shader_builders =
      create_ppll_builders("PPLL/ppllStatsTiledBatches_Stage1.comp", "IMAGE_TILES_UBO");
  shader_builders[0]->replaceFirstTag("numTiles", std::to_string(gfx::kPPLLTilesUBOSize));
  shader_builders[0]->replaceFirstTag("maxNumBatches",
                                      std::to_string(gfx::kMaxNumPPLLPrimitiveBatches));
  auto ppll_stats_tiled_batches_cache =
      shader_mgr.createCacheVector(std::move(shader_builders));

  // Stats tiled batches - stage 2
  shader_builders =
      create_ppll_builders("PPLL/ppllStatsTiledBatches_Stage2.comp", "IMAGE_TILES_UBO");
  shader_builders[0]->replaceFirstTag("numTiles", std::to_string(gfx::kPPLLTilesUBOSize));
  shader_builders[0]->replaceFirstTag("maxNumBatches",
                                      std::to_string(gfx::kMaxNumPPLLPrimitiveBatches));
  auto ppll_stats_max_fragments_per_pixel_cache =
      shader_mgr.createCacheVector(std::move(shader_builders));

  // Debug visualizer
  shader_builders =
      create_ppll_builders("PPLL/ppllDebugVisualizer.comp", "IMAGE_INFO_UBO");
  shader_builders[0]->replaceFirstTag("numSamples", std::to_string(num_samples_));
  auto ppll_debug_viz_cache = shader_mgr.createCacheVector(std::move(shader_builders));

  //
  // Build global gpu resources
  //
  // The multi-gpu compositor requires cuda / vulkan interop when rendering with
  // vulkan on multiple gpus. The ms_framebuffer attachments must be backed by
  // exportable memory
  bool require_api_export_for_compositing = (device_group.size() > 1) && (cuda_mgr_);

  for (auto const& device_info : device_group) {
    if (cuda_mgr_) {
      // need to set a cuda context before creating interop buffers
      cuda_mgr_->setContext(device_info.index);
    }

    auto itr = gpu_data_map_.find(device_info.gpu_id);
    CHECK(itr != gpu_data_map_.end());
    auto* gpu_data = itr->get();

    auto const& device_context = gpu_data->getDeviceContext();
    auto& rsrc_mgr = gpu_data->getResourceManager();
    auto& query_buffer_mgr = gpu_data->getQueryBufferManager();

    // create a multi-sampled framebuffer
    QueryFramebuffer::CreateFlags framebuffer_flags =
        QueryFramebuffer::CreateFlags::kSupportHitTest |
        QueryFramebuffer::CreateFlags::kSupportDepthTest |
        QueryFramebuffer::CreateFlags::kSupportStencil;

    gpu_data->ms_framebuffer_ = std::make_unique<QueryFramebuffer>(
        device_context, "Render MS", raster_sample_count_, framebuffer_flags);

    // create a single-sampled framebuffer
    // this is used for accumulation mark passes (needs depth/stencil)
    // it is also used for anti-aliasing
    // If multi-sampling is enabled, this is used to blit the multi-sampled framebuffers
    // into, otherwise it's used to store the output of an anti-aliasing post-processing
    // pass
    // NOTE: The multi-gpu compositor uses these resources to blit/sample multi-sampled
    // textures
    if (require_api_export_for_compositing) {
      framebuffer_flags |= QueryFramebuffer::CreateFlags::kEnableApiExport;
    }
    framebuffer_flags |= QueryFramebuffer::CreateFlags::kCreateR32UIView;
    gpu_data->aa_framebuffer_ = std::make_unique<QueryFramebuffer>(
        device_context, "Render SS", gfx::RasterSampleCount::k1, framebuffer_flags);

    // create common RenderPasses
    createCommonRenderPasses(*gpu_data);

    // TODO(croot): can make the query result buffer a unique ptr when the embedded data
    // is decoupled from the BaseQueryDataTableVBO class in QueryDataLayout.cpp and the
    // PerGpuData.vbo in that class is made into a pointer or reference.
    gpu_data->query_result_buffer_ =
        (render_mem_bytes_
             ? std::make_shared<QueryVertexBuffer>(query_buffer_mgr, render_mem_bytes_)
             : nullptr);

    gpu_data->id_pbo_pool_ = std::make_unique<QueryIdMapPboPool>(rsrc_mgr);

    gpu_data->vbo_buffer_pool_ =
        std::make_unique<QueryBufferPool<QueryVertexBuffer>>(query_buffer_mgr);
    gpu_data->ibo_buffer_pool_ =
        std::make_unique<QueryBufferPool<QueryIndexBuffer>>(query_buffer_mgr);
    gpu_data->ssbo_buffer_pool_ =
        std::make_unique<QueryBufferPool<QueryShaderStorageBuffer>>(query_buffer_mgr);
    gpu_data->indibo_buffer_pool_ =
        std::make_unique<QueryBufferPool<QueryIndirectIbo>>(query_buffer_mgr);
    gpu_data->indvbo_buffer_pool_ =
        std::make_unique<QueryBufferPool<QueryIndirectVbo>>(query_buffer_mgr);

    // Empty RenderPass and Framebuffer

    // Use a static empty AttachmentManager since it contains no per gpu resources.
    // TODO: This *could* be retrieved from the Framebuffer and modified, which
    // would break a bunch of stuff. Once active attachment juggling is moved into
    // GLRenderPass we should be able to remove the Framebuffer getter, making
    // it safer
    static gfx::AttachmentManager empty_attachment_mgr;
    gpu_data->empty_renderpass_ =
        rsrc_mgr.createRenderPass("Empty", empty_attachment_mgr.getLayout());
    // Create empty Framebuffer as 1x1 (0 size is not legal). It will be resized in
    // prepareRenderTargets
    gpu_data->empty_framebuffer_ = rsrc_mgr.createFramebuffer(
        "Empty", *gpu_data->empty_renderpass_, empty_attachment_mgr, 1, 1, 1);

    // Build resources for accumulation ID pass
    gpu_data->accum_id_pass_resources_ = std::make_unique<AccumIdPassResources>();
    auto& id_resources = *gpu_data->accum_id_pass_resources_;
    gfx::SubpassDescriptor subpass_desc;
    subpass_desc.attachments = {gfx::Framebuffer::Attachment::kColor1,
                                gfx::Framebuffer::Attachment::kColor2,
                                gfx::Framebuffer::Attachment::kColor3};
    id_resources.renderpass =
        rsrc_mgr.createRenderPass("MultiGpuCompositor_ID_Only",
                                  gpu_data->ms_framebuffer_->getFramebufferLayout(),
                                  gfx::RenderPass::ClearBits::kNone,
                                  gfx::ImageLayout::kAttachment,
                                  gfx::ImageLayout::kAttachment,
                                  {subpass_desc});
    id_resources.material =
        rsrc_mgr.createMaterial("Accum ID Pass", accum_id_pass_caches);
    id_resources.pipeline_desc = std::make_unique<gfx::PipelineDescriptor>();
    id_resources.pipeline_desc->setRasterSampleCount(raster_sample_count_);
    id_resources.pipeline = rsrc_mgr.createGraphicsPipeline(
        "Accum ID Pass", *id_resources.material, *id_resources.pipeline_desc);
    id_resources.pipeline->create(*id_resources.renderpass);

    // accum extents
    gpu_data->accum_extents_pipelines_ = std::make_unique<AccumExtentsPipelines>();
    {
      auto& pipelines = *gpu_data->accum_extents_pipelines_;

      // create materials
      pipelines.extents_material =
          rsrc_mgr.createMaterial("Accum Extents", accum_extents_cache);
      pipelines.std_dev_material =
          rsrc_mgr.createMaterial("Accum StdDev", accum_stddev_cache);

      // create pipelines
      pipelines.extents_pipeline =
          rsrc_mgr.createComputePipeline("Accum Extents", *pipelines.extents_material);
      pipelines.extents_pipeline->create();

      pipelines.std_dev_pipeline =
          rsrc_mgr.createComputePipeline("Accum StdDev", *pipelines.std_dev_material);
      pipelines.std_dev_pipeline->create();
    }

    // geo count (only if required functionality is available)
    if (canUseSlabAddressTable() && canUseMeshShaders()) {
      gpu_data->geo_count_resources_ =
          std::make_unique<gfx::GeoCountResources>(gfx_context_, rsrc_mgr);
    }

    // create accum extents buffer
    gpu_data->accum_extents_buffer_ =
        rsrc_mgr.createBuffer("Accum Extents",
                              {gfx::BufferType::kUnspecified,
                               sizeof(AccumRenderer::Extents),
                               gfx::BufferUsageBits::kStorageBufferBit});

    // clear query output buffer
    gpu_data->clear_query_output_buffer_resources_ =
        std::make_unique<ClearQueryOutputBufferResources>();
    gpu_data->clear_query_output_buffer_resources_->material = rsrc_mgr.createMaterial(
        "Clear Query Output Buffer", clear_query_output_buffer_cache);
    gpu_data->clear_query_output_buffer_resources_->pipeline =
        rsrc_mgr.createComputePipeline(
            "Clear Query Output Buffer",
            *gpu_data->clear_query_output_buffer_resources_->material);
    gpu_data->clear_query_output_buffer_resources_->pipeline->create();

    //
    // PPLL material and pipeline resources
    //
    auto create_ppll_resources = [&](auto name,
                                     auto shader_caches,
                                     int num_push_constants) {
      gfx::PPLLResources::PipelineResources resources;
      resources.material = rsrc_mgr.createMaterial(name, shader_caches);
      gfx::PushConstantRanges ranges{gfx::PushConstantRange(
          gfx::ShaderStageBits::kCompute, 0, sizeof(uint32_t) * num_push_constants)};
      if (num_push_constants) {
        resources.pipeline =
            rsrc_mgr.createComputePipeline(name, *resources.material, {}, ranges);
      } else {
        resources.pipeline = rsrc_mgr.createComputePipeline(name, *resources.material);
      }
      resources.pipeline->create();
      return resources;
    };

    // Stats
    auto stats_resources =
        create_ppll_resources("PPLL Fragment Stats", ppll_stats_cache, 0);

    // Stats tiled
    auto stats_tiled_resources =
        create_ppll_resources("PPLL Fragment Tile Stats", ppll_tile_stats_cache, 2);

    // Stats tiled and batched - stage 1
    auto stats_tiled_and_batched_stage_1_resources = create_ppll_resources(
        "PPLL Fragment Tiled Batched Stats - Stage 1", ppll_stats_tiled_batches_cache, 2);

    // Stats tiled and batched - stage 2
    auto stats_tiled_and_batched_stage_2_resources =
        create_ppll_resources("PPLL Fragment Tiled Batched Stats - Stage 2",
                              ppll_stats_max_fragments_per_pixel_cache,
                              1);

    // Debug visualization
    auto debug_viz_resources =
        create_ppll_resources("PPLL Debug Viz", ppll_debug_viz_cache, 0);

    gpu_data->ppll_resources_ = std::make_unique<gfx::PPLLResources>(
        device_context,
        std::move(stats_resources),
        std::move(stats_tiled_resources),
        std::move(stats_tiled_and_batched_stage_1_resources),
        std::move(stats_tiled_and_batched_stage_2_resources),
        std::move(debug_viz_resources));
  }

  //
  // Renderer and support components
  //
  renderer_ = std::make_unique<Renderer>(*this);
  smaa_pass_ = std::make_unique<QueryRenderSMAAPass>(
      *this,
      QueryRenderSMAAPass::SMAAQualityPreset::kHigh,
      QueryRenderSMAAPass::SMAAEdgeDetectionType::kLuma);

  if (raster_sample_count_ != gfx::RasterSampleCount::k1) {
    separate_multisamples_pass_ = std::make_unique<SeparateMultiSamplesPass>(
        *this, require_api_export_for_compositing);
  }

  // Initialize the multi-gpu compositor
  if (cuda_mgr_ && device_group.size() > 1u) {
    multi_gpu_compositor_ = std::make_unique<MultiGpuCompositor>(
        *this, cuda_mgr_, use_last_gpu_for_compositor_);
  }

  // make an accum renderer
  accum_renderer_ = std::make_unique<AccumRenderer>(*this);
}

bool GlobalRenderContext::areCachesAndResourcesComplete(const bool log_incomplete) const {
  // TODO(scb): Consider pushing this down into RootPerGpuData as an isComplete()
  // function. Leaving here for now since the GlobalRenderContext is doing the creation
  // of required bits (it has the knowledge of what completeness means)

  if (gpu_data_map_.empty()) {
    return false;
  } else {
    for (auto const& data : gpu_data_map_) {
      auto const gpu_id = data->getGpuId();
      auto complete = [gpu_id, log_incomplete](const auto& ptr,
                                               const std::string& name) -> bool {
        auto const is_complete = (ptr != nullptr);
        LOG_IF(ERROR, log_incomplete && !is_complete)
            << "GPU " << gpu_id << " has incomplete " << name;
        return is_complete;
      };
      bool res = true;
      // check all components of RootPerGpuData except for the following
      // which are null until first accum render
      //   accum_tx_array_
      //   accum_id_pass_resources_->framebuffer
      res = res && complete(data->ms_framebuffer_, "MS Framebuffer");
      res = res && complete(data->aa_framebuffer_, "AA Framebuffer");
      res = res && complete(data->id_pbo_pool_, "ID PBO Pool");
      res = res && complete(data->empty_renderpass_, "Empty RenderPass");
      res = res && complete(data->empty_framebuffer_, "Empty Framebuffer");
      for (auto const& crp : data->common_render_passes_) {
        auto crp_name = crp ? crp->getTrackingData().origin : "";
        res = res && complete(crp, "Common Render Pass " + crp_name);
      }
      res = res && complete(data->vbo_buffer_pool_, "VBO Pool");
      res = res && complete(data->ibo_buffer_pool_, "IBO Pool");
      res = res && complete(data->ssbo_buffer_pool_, "SSBO Pool");
      res = res && complete(data->indibo_buffer_pool_, "IndIBO Pool");
      res = res && complete(data->indvbo_buffer_pool_, "IndVBO Pool");
      res = res && complete(data->accum_extents_buffer_, "Accum Ext Buffer");
      res = res && complete(data->accum_extents_pipelines_, "Accum Ext Pipelines");
      res = res && complete(data->accum_extents_pipelines_->extents_material,
                            "Accum Ext Pipelines Extents Material");
      res = res && complete(data->accum_extents_pipelines_->std_dev_material,
                            "Accum Ext Pipelines Std Dev Material");
      res = res && complete(data->accum_extents_pipelines_->extents_pipeline,
                            "Accum Ext Pipelines Extents Pipeline");
      res = res && complete(data->accum_extents_pipelines_->std_dev_pipeline,
                            "Accum Ext Pipelines Std Dev Pipeline");
      res = res && complete(data->accum_id_pass_resources_, "Accum ID Pass Resources");
      res = res &&
            complete(data->accum_id_pass_resources_->material, "Accum ID Pass Material");
      res = res && complete(data->accum_id_pass_resources_->pipeline_desc,
                            "Accum ID Pass PipelineDesc");
      res = res &&
            complete(data->accum_id_pass_resources_->pipeline, "Accum ID Pass Pipeline");
      res = res && complete(data->accum_id_pass_resources_->renderpass,
                            "Accum ID Pass RenderPass");
      if (!res) {
        return false;
      }
    }
  }
  return true;
}

void GlobalRenderContext::prepareRenderTargets(const QueryRendererContext& render_context,
                                               const std::set<GpuId>& used_gpus) {
  // Fix for [BE-3904] Right column of pixels incorrect with tight fitting framebuffers
  // This guarantees we have a 1 pixel pad on the right edge. Bottom edges don't seem
  // to be impacted.

  // TODO: profile this so we understand the time versus memory tradeoff when going from
  // 1 to many to 1 gpu configurations.
  // TODO: maybe destroy any render targets we don't need anymore base on above
  // information
  RENDER_LOG_SCOPE_P(used_gpus);

  auto [required_width, required_height] = get_render_target_size_from_render_size(
      render_context.getWidth(), render_context.getHeight());
  auto width_to_use = std::max(render_target_width_, required_width);
  auto height_to_use = std::max(render_target_height_, required_height);
  if ((width_to_use != render_target_width_) ||
      (height_to_use != render_target_height_)) {
    RENDER_LOG() << "render target size [" << width_to_use << "x" << height_to_use << "]";
    // Ensure any resource caches are cleared before resizing render target resources
    if (multi_gpu_compositor_) {
      multi_gpu_compositor_->purgeResourceCache();
    }

    // Prepare framebuffer resources for all gpus
    for (auto& gpu_data : gpu_data_map_) {
      gpu_data->prepareRenderTargets(width_to_use, height_to_use);
    }

    // Update antialiasing passes
    smaa_pass_->prepareRenderTargets(width_to_use, height_to_use);
    if (separate_multisamples_pass_ != nullptr) {
      separate_multisamples_pass_->prepareRenderTargets(width_to_use, height_to_use);
    }
  }

  // prepare PPLL resources
  if (render_context.usesPerPixelLinkedLists()) {
    for (auto& gpu_data : gpu_data_map_) {
      gpu_data->ppll_resources_->setExternalCountsTexture(
          *gpu_data->aa_framebuffer_->getTexture(QueryRenderer::FboAttachment::Color),
          QueryFramebuffer::kR32UIViewId);
      gpu_data->ppll_resources_->setExternalBatchCountsTexture(
          *gpu_data->aa_framebuffer_->getTexture(QueryRenderer::FboAttachment::ID1A), 0);

      // PPLL targets must be sized to the true render size, not the padded size
      // required for multi-sampled framebuffer attachments
      gpu_data->ppll_resources_->prepareRenderTargets(render_context.getWidth(),
                                                      render_context.getHeight());
    }
  } else {
    for (auto& gpu_data : gpu_data_map_) {
      gpu_data->ppll_resources_->destroyDynamicBuffersAndImages();
    }
  }

  // prepare/destroy accum texture array
  auto required_accum_depth = render_context.getNumRequiredAccumulatorTextures();
  auto accum_tx_array_depth_to_use =
      std::max(accum_tx_array_depth_, required_accum_depth);
  if (accum_tx_array_depth_to_use > 0u) {
    if ((width_to_use > render_target_width_) ||
        (height_to_use > render_target_height_) ||
        (accum_tx_array_depth_to_use > accum_tx_array_depth_)) {
      RENDER_LOG() << "preparing accum tx array [" << width_to_use << "x" << height_to_use
                   << "x" << accum_tx_array_depth_to_use << "]";
      for (auto& gpu_data : gpu_data_map_) {
        gpu_data->prepareAccumTextureArray(
            width_to_use, height_to_use, accum_tx_array_depth_to_use);
        gpu_data->updateAccumIDPassResources();
      }
    }
  } else {
    for (auto& gpu_data : gpu_data_map_) {
      gpu_data->destroyAccumTextureArray();
    }
  }

  // the new sizes, whether we resized anything or not
  render_target_width_ = width_to_use;
  render_target_height_ = height_to_use;
  accum_tx_array_depth_ = accum_tx_array_depth_to_use;

  // Update the multi-gpu compositor
  if (multi_gpu_compositor_) {
    multi_gpu_compositor_->prepareRenderTargets(
        render_target_width_, render_target_height_, render_context, used_gpus);
  }

  // update descriptors / bindings
  if (separate_multisamples_pass_ != nullptr) {
    separate_multisamples_pass_->postPrepareRenderTargets();
  }
  smaa_pass_->postPrepareRenderTargets();
}

void GlobalRenderContext::createCommonRenderPasses(RootPerGpuData& gpu_data) {
  auto& rsrc_mgr = gpu_data.getDeviceContext().getResourceManager();
  auto const& ms_layout = gpu_data.ms_framebuffer_->getFramebufferLayout();
  auto const& ss_layout = gpu_data.aa_framebuffer_->getFramebufferLayout();

  static constexpr int kAllAttachmentsClear =
      static_cast<int>(CommonRenderPassType::kAllAttachmentsClear);
  static constexpr int kAllAttachments =
      static_cast<int>(CommonRenderPassType::kAllAttachments);
  static constexpr int kDepthStencilThenAll =
      static_cast<int>(CommonRenderPassType::kDepthStencilThenAll);
  static constexpr int kCount = static_cast<int>(CommonRenderPassType::kCount);

  // Build common renderpasses
  // the triples are identical apart from AllAttachmentsClear where we set
  // the final layout for the SS variant to be ShaderReadOnly
  auto create_common_render_passes = [&](const gfx::Framebuffer::Layout& layout,
                                         const gfx::ImageLayout all_clear_final_layout,
                                         const std::string& suffix,
                                         int offset) {
    gpu_data.common_render_passes_[kAllAttachmentsClear + offset] =
        rsrc_mgr.createRenderPass("Common all attachments (cleared) " + suffix,
                                  layout,
                                  gfx::RenderPass::ClearBits::kAll,
                                  gfx::ImageLayout::kUndefined,
                                  all_clear_final_layout);

    gpu_data.common_render_passes_[kAllAttachments + offset] =
        rsrc_mgr.createRenderPass("Common all attachments " + suffix, layout);

    std::vector<gfx::SubpassDescriptor> subpasses(2);
    subpasses[0] = {{gfx::Framebuffer::Attachment::kDepthStencil}, {}};
    subpasses[1] = {{ms_layout.getAttachmentBindingSet()}, {}};
    gpu_data.common_render_passes_[kDepthStencilThenAll + offset] =
        rsrc_mgr.createRenderPass("Common depthstencil then all " + suffix,
                                  layout,
                                  gfx::RenderPass::ClearBits::kStencil,
                                  gfx::ImageLayout::kAttachment,
                                  gfx::ImageLayout::kAttachment,
                                  subpasses);
  };

  create_common_render_passes(ms_layout, gfx::ImageLayout::kAttachment, "(MS)", 0);
  create_common_render_passes(
      ss_layout, gfx::ImageLayout::kShaderReadOnly, "(SS)", kCount);

  // Build accum renderpasses
  // 0 = "2ndPass" only, no extents
  std::vector<gfx::SubpassDescriptor> subpasses(1);
  subpasses[0].attachments = {gfx::Framebuffer::Attachment::kColor0};
  subpasses[0].dependencies = gfx::SubpassDependencyBits::kFragmentShaderRead;

  gpu_data.accum_render_pass_ =
      rsrc_mgr.createRenderPass("Accum",
                                ms_layout,
                                gfx::RenderPass::ClearBits::kNone,
                                gfx::ImageLayout::kAttachment,
                                gfx::ImageLayout::kAttachment,
                                subpasses);
}

void GlobalRenderContext::resetRenderTargetSize() {
  RENDER_LOG_SCOPE();
  render_target_width_ = 0u;
  render_target_height_ = 0u;
  accum_tx_array_depth_ = 0u;
}

void GlobalRenderContext::clearGpuMemory() {
  RENDER_LOG_SCOPE();
  for (auto& gpu_data : gpu_data_map_) {
    gpu_data->clearResources();
  }
  multi_gpu_compositor_ = nullptr;
  renderer_ = nullptr;
  accum_renderer_ = nullptr;
  smaa_pass_ = nullptr;
  separate_multisamples_pass_ = nullptr;
  gpu_data_map_.clear();
  render_target_width_ = 0u;
  render_target_height_ = 0u;
  accum_tx_array_depth_ = 0u;

  // Clear hit-test cache
  query_result_cache_->clear();
}

void GlobalRenderContext::clearCpuMemory() {
  query_result_cache_->clear();
}

GpuId GlobalRenderContext::getStartGpuId() const {
  const RootPerGpuDataMap_in_order& in_order = gpu_data_map_.get<inorder>();
  CHECK(in_order.size());
  return in_order[0]->getGpuId();
}

GpuId GlobalRenderContext::getLastGpuId() const {
  const RootPerGpuDataMap_in_order& in_order = gpu_data_map_.get<inorder>();
  CHECK(in_order.size());
  return in_order.back()->getGpuId();
}

GpuId GlobalRenderContext::getLeastSubscribedGpuId() const {
  uint64_t max_free = 0;
  GpuId best_gpu = 0;  // default return
  for (auto const& gpu_data : gpu_data_map_) {
    auto mem_info = gpu_data->getDeviceContext().getMemoryBudget();
    if (mem_info.available > max_free) {
      best_gpu = gpu_data->getGpuId() - getStartGpuId();
      max_free = mem_info.available;
    }
  }
  return best_gpu;
}

RootPerGpuData& GlobalRenderContext::getGpuData(GpuId gpu_id) const {
  auto itr = gpu_data_map_.find(gpu_id);
  CHECK(itr != gpu_data_map_.end());
  return *(itr->get());
}

GpuId GlobalRenderContext::getGpuId(size_t gpu_index) const {
  const auto& in_order = gpu_data_map_.get<inorder>();
  RUNTIME_EX_ASSERT(gpu_index < in_order.size(),
                    "GlobalRenderContext::getGpuId(): Invalid gpu index " +
                        std::to_string(gpu_index) + ". There are only " +
                        std::to_string(in_order.size()) + " gpus available.");
  return in_order[gpu_index]->getGpuId();
}

RootPerGpuData& GlobalRenderContext::getGpuDataFromIndex(size_t gpu_index) const {
  const auto& in_order = gpu_data_map_.get<inorder>();
  RUNTIME_EX_ASSERT(gpu_index < in_order.size(),
                    "GlobalRenderContext::getGpuId(): Invalid gpu index " +
                        std::to_string(gpu_index) + ". There are only " +
                        std::to_string(in_order.size()) + " gpus available.");
  return *in_order[gpu_index];
}

GpuId GlobalRenderContext::getCompositorGpuId() const {
  RUNTIME_EX_ASSERT(multi_gpu_compositor_ != nullptr,
                    "Cannot get the compositor gpu id. The compositor is uninitialized.");

  return multi_gpu_compositor_->getDeviceContext().getGpuId();
}

const gfx::GfxContext& GlobalRenderContext::getGfxContext() const {
  return gfx_context_;
}

const Renderer& GlobalRenderContext::getRenderer() const {
  return *renderer_;
}

Data_Namespace::DataMgr* GlobalRenderContext::getDataMgr() {
  return data_mgr_;
}

const CudaMgr_Namespace::CudaMgr* GlobalRenderContext::getCudaMgr() const {
  return cuda_mgr_;
}

gfx::DriverType GlobalRenderContext::getDriverType() const {
  return gfx_context_.getPrimaryDriver().getType();
}

void GlobalRenderContext::logMemorySummary(std::ostream& os) const {
  os << "========================\n";
  os << "Renderer memory summary\n\n";
  for (auto const& gpu_data : gpu_data_map_) {
    if (gpu_data.get() != gpu_data_map_.begin()->get()) {
      os << "\n------\n\n";
    }
    gpu_data->device_ctx_.getResourceManager().logMemorySummary(os);
  }
  os << "========================\n";
}

bool GlobalRenderContext::canUseMeshShaders() const {
  return getGfxContext().queryDeviceCapabilities(gfx::DeviceCapabilityBits::kMeshShaders |
                                                 gfx::DeviceCapabilityBits::kTaskShaders);
}

bool GlobalRenderContext::canUseSlabAddressTable() const {
#ifdef HAVE_CUDA
  return getGfxContext().queryDeviceCapabilities(
      gfx::DeviceCapabilityBits::kBufferDeviceAddress);
#else
  return false;
#endif
}

void GlobalRenderContext::updateSlabAddressTableAndBuffers(const int gpu_id) {
#ifdef HAVE_CUDA
  // get GPU Data for this ID
  auto itr = gpu_data_map_.find(gpu_id);
  CHECK(itr != gpu_data_map_.end());
  auto* gpu_data = itr->get();
  CHECK(gpu_data);

  // tell QBM to update
  CHECK(gpu_data->query_buffer_mgr_);
  gpu_data->query_buffer_mgr_->updateSlabAddressTableAndBuffers();
#endif
}

}  // namespace QueryRenderer
