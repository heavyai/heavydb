/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <set>

#include "CudaMgr/CudaMgr.h"
#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Types.h"
#include "QueryRenderer/Rendering/TextureTransferContext.h"
#include "QueryRenderer/Rendering/Types.h"

namespace QueryRenderer {

/**
 *
 * Multi-Gpu Compositor
 *
 * The compositor is responsible for combining rendered results from multiple GPUs into a
 * single framebuffer, which is then treated as if rendering completed on a single gpu
 *
 * Data transfer is handled by a TextureTransferContext instance, using Cuda to perform
 * fast peer-to-peer copying as this is not supported natively in Vulkan
 *
 * - Multisampled images cannot be copied peer-to-peer. In order to handle multisampling
 *   with RGBA, extract each sample as a separate image and transfer each image to the
 *   compositor's GPU. The compositor then performs a standard pre-multiplied alpha
 *   composite of each sample into its multisampled framebuffer using a compute shader
 *
 * - ID buffers are handled if hit-testing is enabled in QueryRendererContext. In the case
 *   of multi-sampling, a single ID is extracted by first "blitting" from the source
 *   framebuffer into the dedicated single-sample anti-aliasing framebuffer on the source
 *   GPU. All 3 R32UI images are then transferred to the compositor GPU. During the RGBA
 *   composite, if an RGBA sample has alpha > 0, the ID values for that GPU are also
 *   copied to the sample
 *
 * - Accumulation is handled by copying one layer at a time from each peer gpu, and
 *   running a shader pass to accumulate the values into the RootPerGpuData texture array
 *   on the compositor's gpu. A special ID only compositing compute shader then writes
 *   any non-zero ID values for that pixel into all the samples of the compositor's
 *   framebuffer
 */
class MultiGpuCompositor {
 public:
  explicit MultiGpuCompositor(const GlobalRenderContext& global_context,
                              const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                              const bool use_last_gpu);
  MultiGpuCompositor() = delete;

  ~MultiGpuCompositor();

  const gfx::DeviceContext& getDeviceContext() const;
  QueryFramebuffer& getFramebuffer();

  // Ensure the required texture arrays and transfer contexts are initialized and
  // properly sized for the set of used gpus for the current render context.
  void prepareRenderTargets(uint32_t render_target_width,
                            uint32_t render_target_height,
                            const QueryRendererContext& render_context,
                            const std::set<GpuId>& used_gpus);

  // Destroy any resources used by the compositor, including transfer context
  // resources. Ensures Cuda graphics resources are destroyed prior to resizing or
  // recreating render target textures
  void purgeResourceCache();

  // Primary render function called by QueryRenderer. Sets up the per gpu post
  // render callback and the per pass post render callback
  void render(const QueryRendererContext& ctx,
              const std::set<GpuId>& used_gpus,
              PassCompleteCBFunc pass_complete_callback);

 private:
  const GlobalRenderContext& global_context_;
  const CudaMgr_Namespace::CudaMgr* cuda_mgr_;
  const RootPerGpuData& comp_gpu_data_;
  const gfx::DeviceContext& comp_device_ctx_;
  GpuId start_gpu_id_;
  const gfx::RasterSampleCount raster_sample_count_;
  const uint32_t num_samples_;

  std::set<GpuId> registered_gpus_;

  bool are_resources_complete_;

  // render target dimensions
  uint32_t render_target_width_;
  uint32_t render_target_height_;

  // Composited framebuffer (manager)
  std::unique_ptr<QueryFramebuffer> framebuffer_;

  // Composite pass RenderPasses
  gfx::resource_ptr<gfx::RenderPass> render_pass_clear_;

  // Transfer context
  std::unique_ptr<TextureTransferContext> transfer_ctx_;

  //
  // Compositing texture arrays
  // These are the targets for the transfer contexts
  //
  using TextureResource = gfx::resource_ptr<gfx::Texture>;

  // RGBA vector index = sample index
  std::vector<TextureResource> rgba_textures_;

  // ID textures for hit testing
  enum IDIndex { k1A, k1B, k2, kCount };
  std::vector<TextureResource> id_textures_;

  // Accumulation
  TextureResource accumulation_copy_texture_;  // copy of texture layer from src

  //
  // Compositing materials for fullscreen passes
  //
  gfx::MaterialUqPtr compositor_material_;        // composite RGBA and ID texture arrays
  gfx::MaterialUqPtr compositor_id_material_;     // composite ID texture arrays only
  gfx::MaterialUqPtr accumulator_peer_material_;  // add copy texture to texture array

  // Pipeline Descriptors and Pipelines
  gfx::PipelineDescriptorUqPtr accumulator_pipeline_descriptor_;
  gfx::resource_ptr<gfx::ComputePipeline> compositor_pipeline_;
  gfx::resource_ptr<gfx::ComputePipeline> compositor_id_pipeline_;
  gfx::resource_ptr<gfx::GraphicsPipeline> accumulator_peer_pipeline_;

  void createBaseTextures();
  void createAccumTexture();

  void destroyBaseTextures();
  void destroyAccumTextureArray();

  void initRenderPasses();
  void destroyRenderPasses();

  void initMaterialResources();
  void initTransferContext(const QueryRendererContext& render_context);
  void initMaterialResourceBindingCaches();

  void initPipelines();
  void destroyPipelines();

  void compAccumLayerCallback(uint32_t layer_index,
                              const std::vector<gfx::SemaphoreHandle>& signal_semaphore);

  void compColorAndIDs(uint32_t width, uint32_t height, bool should_clear_dst);
  void compIDs(uint32_t width, uint32_t height, bool should_clear_dst);

  void postPassPerGpuCB(const gfx::DeviceContext& src_device_ctx,
                        QueryFramebuffer& framebuffer,
                        const QueryRendererContext& render_context,
                        const bool should_clear_dst,
                        const bool should_comp,
                        ScaleAccumRenderState* scale_accum_render_state,
                        const int accumulator_index);

  void compositePass(const std::set<GpuId>& render_used_gpus,
                     const std::set<GpuId>& pass_used_gpus,
                     const QueryRendererContext& render_context,
                     const int pass_index,
                     ScaleAccumRenderState* scale_accum_render_state);

  PassCompleteCBFunc pass_complete_override_cb_;
};

}  // namespace QueryRenderer
