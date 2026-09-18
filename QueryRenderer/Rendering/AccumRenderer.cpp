/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/AccumRenderer.h"

#include <limits>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/Texture.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/Rendering/QueryFramebuffer.h"
#include "QueryRenderer/Rendering/RenderAccumStats.h"
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/Scales/ScaleAccumRenderState.h"
#include "QueryRenderer/Scales/ScaleAccumState.h"

#define PROFILE_ACCUMULATION 0
#define LOG_EXTENTS_VALUES 0

#if LOG_EXTENTS_VALUES || PROFILE_ACCUMULATION
#include <iostream>
#endif
#if PROFILE_ACCUMULATION
#include "Shared/measure.h"
#endif

namespace QueryRenderer {

using ::gfx::BufferWrapper;
using ::gfx::DeviceContext;
using ::gfx::Framebuffer;
using ::gfx::ShaderManager;
using ::gfx::ShaderStage;
using ::gfx::Texture;
using ::gfx::TypeGLSLShPtr;

std::pair<BufferWrapper*, bool> AccumRenderer::updateMaterials(
    ScaleAccumRenderState& scale_accum_render_state,
    const gfx::Texture& accum_texture_array,
    const RootPerGpuData& gpu_data) {
  bool do_extents = scale_accum_render_state.getDoFindExtents();
  bool do_find_std_dev = scale_accum_render_state.getDoFindStdDev();

  BufferWrapper* extents_buffer = nullptr;

  // Update 2nd pass material
  auto* accum_material = scale_accum_render_state.get2ndPassMaterial(gpu_data.getGpuId());
  CHECK(accum_material);

  accum_material->setImageLoadStoreAttribute("inTxArrayPixelCounter",
                                             accum_texture_array);

  bool use_domain, use_range;
  switch (scale_accum_render_state.getAccumulatorType()) {
    case AccumulatorType::kMin:
    case AccumulatorType::kMax:
    case AccumulatorType::kBlend:
      use_domain = false;
      use_range = false;
      break;
    case AccumulatorType::kDensity:
    case AccumulatorType::kPct:
      use_domain = true;
      use_range = true;
      break;
    default:
      THROW_RUNTIME_EX("Accumulator type " +
                       to_string(scale_accum_render_state.getAccumulatorType()) +
                       " is currently unsupported for rendering");
  }

  scale_accum_render_state.getParentScale().bindUniforms(
      *accum_material,
      "_ACCUMULATION",  // TODO(croot): expose as global
      use_domain,
      use_range,
      false,
      nullptr);

  accum_material->setUniformAttribute<uint32_t>("minDensity",
                                                scale_accum_render_state.getMinDensity());
  accum_material->setUniformAttribute<uint32_t>("maxDensity",
                                                scale_accum_render_state.getMaxDensity());
  accum_material->setUniformAttribute<float>(
      "minStdDev", static_cast<float>(scale_accum_render_state.getNumMinStdDev()));
  accum_material->setUniformAttribute<float>(
      "maxStdDev", static_cast<float>(scale_accum_render_state.getNumMaxStdDev()));

  scale_accum_render_state.getParentScale().bindAccumulatorColors(*accum_material,
                                                                  "inColors");

  if (do_extents) {
    extents_buffer = gpu_data.getAccumExtentsBuffer();
    CHECK(extents_buffer);

    // Initialize image stats
    Extents extents{.totalNonZeroCount = 0ull,
                    .totalSqrDiff = 0ull,
                    .minCount = std::numeric_limits<uint32_t>::max(),
                    .maxCount = std::numeric_limits<uint32_t>::min(),
                    .numNonZeroCount = 0u};
    extents_buffer->updateSubData(&extents, sizeof(Extents), 0ULL);

    accum_material->bindShaderStorageBufferToBlock("EXTENTS_SSBO", *extents_buffer);

    auto& extent_pipelines = gpu_data.getAccumExtentsPipelines();
    auto& extents_material = extent_pipelines.extents_material;
    extents_material->setImageLoadStoreAttribute("inTxArrayPixelCounter",
                                                 accum_texture_array);
    extents_material->bindShaderStorageBufferToBlock("EXTENTS_SSBO", *extents_buffer);
    extents_material->setUniformAttribute("imageWidth", accum_texture_array.getWidth());
    extents_material->updateDescriptorSets();

    if (do_find_std_dev) {
      auto& std_dev_material = extent_pipelines.std_dev_material;
      std_dev_material->bindShaderStorageBufferToBlock("EXTENTS_SSBO", *extents_buffer);
      std_dev_material->setImageLoadStoreAttribute("inTxArrayPixelCounter",
                                                   accum_texture_array);
      std_dev_material->setUniformAttribute("imageWidth", accum_texture_array.getWidth());
      std_dev_material->updateDescriptorSets();
    }
  }

  accum_material->updateDescriptorSets();

  return {extents_buffer, do_find_std_dev};
}

void AccumRenderer::render(const gfx::DeviceContext& device_context,
                           ScaleAccumRenderState& scale_accum_render_state,
                           Texture& accum_texture_array,
                           QueryFramebuffer& framebuffer,
                           const bool do_id_pass) {
  RENDER_LOG_SCOPE();

  auto gpu_id = device_context.getGpuId();
  auto& gpu_data = global_ctx_.getGpuData(gpu_id);
  auto& cmd_list = device_context.getCommandList();

  auto [extents_buffer, do_find_std_dev] =
      updateMaterials(scale_accum_render_state, accum_texture_array, gpu_data);

  auto& render_pass = gpu_data.getAccumRenderPass();

  // Get FBO for the RenderPass from the QueryFramebuffer
  // TODO(scb): Ideally these would be cached in advance so we aren't creating any
  // resources here, but that gets messy given the prepareRenderTargets ordering.
  // Will save it for a dedicated set of changes
  auto* fbo = framebuffer.getOrCreateFramebufferForRenderPass(render_pass);
  CHECK(fbo != nullptr);

#if PROFILE_ACCUMULATION
  // Final image will be incorrect, this is just for timing
  const int kProfileLoopCount = 10;
  int64_t total_time{0};
  // Ensure all previous commands are complete
  device_context.getCommandExecutor().waitForCompletion(true);
  for (int i = 0; i < kProfileLoopCount; i++) {
    auto const clock_begin = timer_start();
#endif

    cmd_list.pushLabel("Accum render");

    // Do extents and stddev passes if needed
    if (extents_buffer) {
      uint32_t subgroup_size = device_context.getLimits().subgroup_size;
      uint32_t width = accum_texture_array.getWidth();
      uint32_t height = accum_texture_array.getHeight();
      uint32_t x_workgroup_size = (width + subgroup_size) / subgroup_size;
      auto& pipelines = gpu_data.getAccumExtentsPipelines();
      cmd_list.pushLabel("extents")
          .dispatchCompute(*pipelines.extents_pipeline, 0u, x_workgroup_size, height, 1)
          .popLabel();
      if (do_find_std_dev) {
        cmd_list.pushLabel("std dev")
            .dispatchCompute(*pipelines.std_dev_pipeline, 0u, x_workgroup_size, height, 1)
            .popLabel();
      }
    }

    // Do the final color pass
    cmd_list.pushLabel("color")
        .beginRenderPass(render_pass, *fbo)
        .drawFullscreen(*scale_accum_render_state.getPipeline(gpu_id))
        .endRenderPass()
        .popLabel();

    // Optional ID pass
    if (do_id_pass) {
      auto& id_pass = gpu_data.getAccumIDPassResources();
      cmd_list.pushLabel("id")
          .beginRenderPass(*id_pass.renderpass, *id_pass.framebuffer)
          .drawFullscreen(*id_pass.pipeline)
          .endRenderPass()
          .popLabel();
    }

    // do it!
    cmd_list.popLabel().flush("AccumRenderer 2nd pass");

#if PROFILE_ACCUMULATION
    // Wait for draw commands to complete
    device_context.getCommandExecutor().waitForCompletion(true);
    auto const this_time =
        timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
            clock_begin);
    total_time += this_time;
  }
  auto const average_time = total_time / kProfileLoopCount;
  std::cout << "AccumRenderer wall time (average): " << average_time << std::endl;
#endif

  // update accumulation stats
  if (extents_buffer) {
    // fetch final values from buffer
    Extents extents = {};
    extents_buffer->getData(&extents, sizeof(Extents));

#if LOG_EXTENTS_VALUES
    std::cout << "DEBUG: Extents:" << std::endl;
    std::cout << "DEBUG:   minCount          " << extents.minCount << std::endl;
    std::cout << "DEBUG:   maxCount          " << extents.maxCount << std::endl;
    std::cout << "DEBUG:   totalNonZeroCount " << extents.totalNonZeroCount << std::endl;
    std::cout << "DEBUG:   numNonZeroCount   " << extents.numNonZeroCount << std::endl;
    std::cout << "DEBUG:   totalSqrDiff      " << extents.totalSqrDiff << std::endl;
#endif

    // send to state
    scale_accum_render_state.setAccumStats(
        std::make_unique<RenderAccumStats>(extents.minCount,
                                           extents.maxCount,
                                           extents.totalNonZeroCount,
                                           extents.numNonZeroCount,
                                           extents.totalSqrDiff));
  }
}

}  // namespace QueryRenderer
