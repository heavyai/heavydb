/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Renderer.h"

#include "GfxDriver/RenderDoc/RenderDoc.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/Texture.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Interface/RenderRequestInfo.h"
#include "QueryRenderer/Marks/BaseMark.h"
#include "QueryRenderer/PngData.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Rendering/AccumRenderer.h"
#include "QueryRenderer/Rendering/HitTestBuffers.h"
#include "QueryRenderer/Rendering/MultiGpuCompositor.h"
#include "QueryRenderer/Rendering/QueryFramebuffer.h"
#include "QueryRenderer/Rendering/QueryRenderSMAAPass.h"
#include "QueryRenderer/Rendering/SeparateMultiSamplesPass.h"
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/VegaElements.h"
#include "QueryRenderer/VegaParser.h"

#define USE_SSE_UNPREMULT 0
#if USE_SSE_UNPREMULT
#include <nmmintrin.h>
#endif

namespace QueryRenderer {

using ::gfx::DeviceContext;
using ::gfx::PixelFormat;

Renderer::Renderer(GlobalRenderContext& global_context) noexcept
    : global_context_{global_context} {}

void write_debug_png_file(uint32_t width,
                          uint32_t height,
                          QueryFramebuffer* source_fbo,
                          QueryFramebuffer* output_fbo,
                          int index,
                          GpuId gpu_id) {
  // CROOT test/debug code
  source_fbo->copyToFramebuffer(*output_fbo, 0, 0, width, height, true, false, false);
  auto pixels = output_fbo->readColorBuffer(0, 0, width, height);
  PngData png_data(width, height, pixels);
  png_data.writeToFile("render_" + std::to_string(index) + "_" + std::to_string(gpu_id) +
                       ".png");
}

void Renderer::renderPasses(const QueryRendererContext& ctx,
                            const std::set<GpuId>& used_gpus,
                            PerPassGpuCBFunc per_pass_gpu_callback,
                            PassCompleteCBFunc pass_complete_callback) const {
  ScaleAccumRenderState* active_accum_render_state{nullptr};
  int accumulator_index = 0;
  int pass_index = 0;
  std::set<GpuId> pass_gpus;

  RENDER_LOG_SCOPE_P(used_gpus);

  auto const& mark_vector = ctx.vega_elements_->getMarkVector();
  for (size_t i = 0; i < mark_vector.size(); ++i) {
    auto curr_accum_scale = mark_vector[i]->getAccumulatorScale();
    ScaleAccumRenderState* next_accum_render_state =
        curr_accum_scale ? curr_accum_scale->getAccumRenderState() : nullptr;
    if (active_accum_render_state) {
      if (active_accum_render_state != next_accum_render_state) {
        if (pass_complete_callback) {
          pass_complete_callback(
              used_gpus, pass_gpus, ctx, pass_index++, active_accum_render_state);
        }
        pass_gpus.clear();
        accumulator_index = 0;
        if (global_context_.getMultiGpuCompositor()) {
          auto& comp_gpu_data =
              global_context_.getGpuData(global_context_.getCompositorGpuId());
          comp_gpu_data.getAccumTextureArray()->clearPixels();
        }
      } else {
        accumulator_index++;
      }
    } else if (!next_accum_render_state) {
      accumulator_index = 0;
    }
    active_accum_render_state = next_accum_render_state;

    RENDER_LOG() << "Drawing mark " << i
                 << (active_accum_render_state ? " with accum" : "");

    // use AA FBO if this mark is accumulating
    // otherwise regular MS FBO
    // active_accum_render_state is set or cleared per mark
    auto const use_ss_framebuffer = active_accum_render_state ? true : false;
    // First call draw on all gpus
    for (auto gpu_id : used_gpus) {
      auto& gpu_data = global_context_.getGpuData(gpu_id);
      auto const& device_context = gpu_data.getDeviceContext();

      if (accumulator_index == 0 && active_accum_render_state) {
        gpu_data.getAccumTextureArray()->clearPixels();
      }

      auto* ms_framebuffer = gpu_data.getRenderFramebuffer();
      auto* framebuffer =
          use_ss_framebuffer ? gpu_data.getAntiAliasingFramebuffer() : ms_framebuffer;

      RUNTIME_EX_ASSERT(framebuffer != nullptr,
                        "QueryRenderer " + std::string(ctx.getRenderSessionKey()) +
                            ": The framebuffer is not initialized for gpu " +
                            std::to_string(gpu_id) + ". Cannot render.");

      auto& cmd_list = device_context.getCommandList();

      if (i == 0) {
        device_context.getCommandExecutor().setDefaultViewportAndRenderArea(
            0, 0, ctx.getWidth(), ctx.getHeight());
      }

      if (i == 0 || used_gpus.size() > 1) {
        // clear framebuffer on first pass thru or on every pass if there are more than 1
        // gpus
        // Use an "empty" renderpass for this, which automatically handles
        // transitioning the images to attachment layout from any other layout

        // we don't need to clear the SS buffer, as all mark accum draws will auto-clear
        // we must ALWAYS just clear the MS buffer, as that's where the final image goes
        cmd_list.pushLabel("renderPasses clear");
        cmd_list
            .beginRenderPass(gpu_data.getCommonRenderPass(
                                 CommonRenderPassType::kAllAttachmentsClear, true),
                             *ms_framebuffer->getFramebuffer())
            .endRenderPass();
        cmd_list.popLabel();
      }

      // TODO(croot): only draw geom on this gpu if the geom is configured to use
      // this gpu
      if (ctx.drawMark(
              i, device_context, *framebuffer->getFramebuffer(), accumulator_index)) {
        pass_gpus.insert(gpu_id);
      }
      // Ensure all commands have been flushed
      if (cmd_list.hasCommands()) {
        cmd_list.flush("renderPasses Mark draw",
                       gfx::CommandList::SubmitType::kImmediateReturn);
      }
    }

    // Loop over all used gpus
    // If we have a per_pass_gpu_callback from the compositor let it finalize things
    // otherwise call the pass (mark) complete callback if we have one
    if (per_pass_gpu_callback) {
      bool should_clear = false;

      // The compositor can skip alpha compositing into the framebuffer for the first
      // gpu of the first pass, saving texture fetches
      bool should_comp = pass_index > 0;

      for (auto const& gpu_id : pass_gpus) {
        auto& gpu_data = global_context_.getGpuData(gpu_id);
        auto const& device_context = gpu_data.getDeviceContext();
        per_pass_gpu_callback(device_context,
                              use_ss_framebuffer ? *gpu_data.getAntiAliasingFramebuffer()
                                                 : *gpu_data.getRenderFramebuffer(),
                              ctx,
                              should_clear,
                              should_comp,
                              active_accum_render_state,
                              accumulator_index);
        // Ensure all commands have been flushed
        should_comp = true;
        auto& cmd_list = device_context.getCommandList();
        if (cmd_list.hasCommands()) {
          cmd_list.flush("QueryRendererContext gpu callback",
                         gfx::CommandList::SubmitType::kImmediateReturn);
        }
        should_clear = false;
      }
    } else {
      // Ensure all draw calls are complete
      // TODO(scb): this can be replaced with a memory barrier or semaphore
      for (auto gpu_id : pass_gpus) {
        auto& gpu_data = global_context_.getGpuData(gpu_id);
        auto& executor = gpu_data.getDeviceContext().getCommandExecutor();
        executor.waitForCompletion(false);  // wait for vkFences
      }
    }

    if (!active_accum_render_state) {
      if (pass_complete_callback) {
        pass_complete_callback(used_gpus, pass_gpus, ctx, pass_index++, nullptr);
      }
      pass_gpus.clear();
    }
  }

  if (active_accum_render_state) {
    // finish the accumulation by running the blending pass
    if (pass_complete_callback) {
      pass_complete_callback(
          used_gpus, pass_gpus, ctx, pass_index++, active_accum_render_state);
    }
    pass_gpus.clear();
  }

  for (auto gpu_id : used_gpus) {
    global_context_.getGpuData(gpu_id).getDeviceContext().resetCommandPools();
  }
}

// private method called by renderToPixels()
void Renderer::render(QueryRendererContext& ctx, const std::set<GpuId>& used_gpus) const {
  RENDER_LOG_SCOPE_P(used_gpus);
  try {
    // Must be called first to ensure RenderPasses are built before Pipelines
    global_context_.prepareRenderTargets(ctx, used_gpus);

    // update slab wrappers if CUDA allocations changed
    for (auto const& gpu_id : used_gpus) {
      global_context_.updateSlabAddressTableAndBuffers(gpu_id);
    }

    // update everything marked dirty before rendering
    ctx.updateMarksAndBuildShaders();

    int num_gpus_to_render = used_gpus.size();
    auto is_multisampled =
        global_context_.getRasterSampleCount() != gfx::RasterSampleCount::k1;

    if (num_gpus_to_render) {
      if (num_gpus_to_render == 1) {
        // Single gpu, no compositor
        auto const gpu_id = *used_gpus.begin();
        auto& gpu_data = global_context_.getGpuData(gpu_id);
        auto const& device_context = gpu_data.getDeviceContext();
        auto* render_fbo = gpu_data.getRenderFramebuffer();
        if (ctx.doHitTest()) {
          ctx.hit_test_buffers_->createPbo(gpu_id);
        }
        CHECK(render_fbo);

        // Apply accumulation after the mark is finished drawing
        auto pass_complete_callback =
            [&device_context, render_fbo, &ctx, &gpu_data](
                const std::set<GpuId>&,
                const std::set<GpuId>&,
                const QueryRendererContext&,
                int,
                ScaleAccumRenderState* scale_accum_render_state) {
              if (scale_accum_render_state) {
                ctx.getGlobalContext().getAccumRenderer().render(
                    device_context,
                    *scale_accum_render_state,
                    *gpu_data.getAccumTextureArray(),
                    *render_fbo,
                    true);  // do run ID pass
              }
            };

        // Render all the passes
        renderPasses(ctx, {gpu_id}, nullptr, pass_complete_callback);

        // Antialias an copy hit buffers if needed
        auto* aa_fbo =
            runAntialiasingPass(ctx,
                                gpu_data,
                                *render_fbo,
                                SeparateMultiSamplesPass::SourceFramebuffer::kRender);
        if (ctx.doHitTest()) {
          ctx.hit_test_buffers_->updateFromFramebuffer(is_multisampled ? *aa_fbo
                                                                       : *render_fbo);
        }
      } else {
        auto const compositor_gpu_id = global_context_.getCompositorGpuId();
        auto& comp_gpu_data = global_context_.getGpuData(compositor_gpu_id);
        if (ctx.doHitTest()) {
          ctx.hit_test_buffers_->createPbo(compositor_gpu_id);
        }
        auto* compositor = global_context_.getMultiGpuCompositor();
        compositor->render(ctx, used_gpus, nullptr);
        auto& comp_fbo = compositor->getFramebuffer();
        auto* aa_fbo =
            runAntialiasingPass(ctx,
                                comp_gpu_data,
                                comp_fbo,
                                SeparateMultiSamplesPass::SourceFramebuffer::kComp);

        if (ctx.doHitTest()) {
          ctx.hit_test_buffers_->updateFromFramebuffer(is_multisampled ? *aa_fbo
                                                                       : comp_fbo);
        }
      }

      ctx.hit_test_buffers_->setCpuCacheDirty();
    }
  } catch (...) {
    // TODO(scb): Clearing the context seems heavy handed but this is how it was
    // previously handled. Is this required?
    ctx.clear();
    std::rethrow_exception(std::current_exception());
  }
}

QueryFramebuffer* Renderer::runAntialiasingPass(
    const QueryRendererContext& ctx,
    RootPerGpuData& gpu_data,
    QueryFramebuffer& source_fbo,
    SeparateMultiSamplesPass::SourceFramebuffer source) const {
  RENDER_LOG_SCOPE_P(gpu_data.getGpuId());

  auto* output_fbo = gpu_data.getAntiAliasingFramebuffer();
  CHECK(output_fbo);

  // TODO(scb): just pass width/height into here?
  auto const width = ctx.getWidth();
  auto const height = ctx.getHeight();
  global_context_.getSMAAPass().runPass(
      width, height, gpu_data.getDeviceContext(), source);

  // If multi-sampling (color), blit the single sample ID buffers into the target
  if (source_fbo.getRasterSampleCount() != gfx::RasterSampleCount::k1) {
    CHECK_EQ(output_fbo->getRasterSampleCount(), gfx::RasterSampleCount::k1);
    source_fbo.copyToFramebuffer(*output_fbo, 0, 0, width, height, false, true, false);
  }
  return output_fbo;
}

namespace {
#if USE_SSE_UNPREMULT
// TODO(scb) function dispatch based on cpuinfo
void unpremultiply_pixels(uint32_t width, uint32_t height, uint8_t* p) {
  uint32_t num_elems = width * height;

  for (uint32_t i = 0; i < num_elems; ++i) {
    uint8_t a = p[3];
    if (a) {
      // unpack bytes
      __m128i xmm_i = _mm_cvtepu8_epi32(*(__m128i*)p);

      // convert to float
      __m128 xmm_fl = _mm_cvtepi32_ps(xmm_i);

      const __m128 scale = _mm_set_ps1(255.0f / (float)a);
      xmm_fl = _mm_mul_ps(xmm_fl, scale);

      // back to int
      xmm_i = _mm_cvtps_epi32(xmm_fl);

      // pack to 16
      xmm_i = _mm_packus_epi32(xmm_i, xmm_i);

      // pack to 8 and extract low 32-bits
      xmm_i = _mm_packus_epi16(xmm_i, xmm_i);

      *(int*)p = _mm_cvtsi128_si32(xmm_i);
      p[3] = a;
    }
    p += 4;
  }
}
#else
void unpremultiply_pixels(uint32_t width, uint32_t height, unsigned char* p) {
  uint32_t num_elems = width * height;
  for (uint32_t i = 0; i < num_elems; ++i) {
    if (p[3]) {
      float mu = (255.0f / (float)p[3]);
      p[0] = static_cast<uint8_t>((float)p[0] * mu);
      p[1] = static_cast<uint8_t>((float)p[1] * mu);
      p[2] = static_cast<uint8_t>((float)p[2] * mu);
    }
    p += 4;
  }
}
#endif
}  // namespace

RenderPixels Renderer::renderToPixels(QueryRendererContext& ctx) const {
#if ENABLE_RENDERDOC
  renderdoc::begin_frame_capture();
  ScopeGuard end_capture = []() { renderdoc::end_frame_capture(); };
#endif
  auto used_gpus = ctx.getUsedGpus();
  RENDER_LOG_SCOPE_P(used_gpus);
  render(ctx, used_gpus);

  try {
    auto width = ctx.getWidth();
    auto height = ctx.getHeight();

    std::vector<std::byte> pixels;

    // TODO(scb): should we ever be able to even get here if numGpus < 1?
    int num_gpus = used_gpus.size();
    if (num_gpus) {
      if (num_gpus > 1) {
        auto* compositor = global_context_.getMultiGpuCompositor();
        CHECK(compositor);
        auto const& device_context = compositor->getDeviceContext();
        auto& comp_gpu_data = global_context_.getGpuData(device_context.getGpuId());
        RENDER_LOG() << "Reading color buffer from compositor's gpu "
                     << device_context.getGpuId();

        auto* antialiasing_fbo = comp_gpu_data.getAntiAliasingFramebuffer();
        if (antialiasing_fbo) {
          RENDER_LOG() << "using global antialiasing framebuffer";
          pixels = antialiasing_fbo->readColorBuffer(0, 0, width, height);
        } else {
          RENDER_LOG() << "reading directly from compositor's framebuffer";
          pixels = compositor->getFramebuffer().readColorBuffer(0, 0, width, height);
        }
      } else {
        auto& gpu_data = global_context_.getGpuData(*used_gpus.begin());
        auto* antialiasing_fbo = gpu_data.getAntiAliasingFramebuffer();
        if (!antialiasing_fbo) {
          antialiasing_fbo = gpu_data.getRenderFramebuffer();
        }

        pixels = antialiasing_fbo->readColorBuffer(0, 0, width, height);
      }
      // @TODO(se) leave unpremultiply here as it may become a GPU pass later
      if (!ctx.getViewRenderOptions().premultiplied_alpha) {
        unpremultiply_pixels(
            width, height, reinterpret_cast<unsigned char*>(pixels.data()));
      }
    } else if (width > 0 && height > 0) {
      // empty image
      pixels.resize(width * height * 4, std::byte(0));

      if (ctx.doHitTest()) {
        ctx.hit_test_buffers_->resetCpuCache();
      }
    }
    return RenderPixels(std::move(pixels), width, height);
  } catch (...) {
    // TODO(scb): Clearing the context seems heavy handed but this is how it was
    // previously handled. Is this required?
    ctx.clear();
    std::rethrow_exception(std::current_exception());
  }
}

}  // namespace QueryRenderer
