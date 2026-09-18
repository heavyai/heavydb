/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/Types.h"

#include "QueryRenderer/Rendering/SeparateMultiSamplesPass.h"

namespace QueryRenderer {

struct RenderPixels;
class PngData;

class Renderer {
 public:
  explicit Renderer(GlobalRenderContext& global_context) noexcept;
  ~Renderer() = default;

  // Primary rendering loop
  void renderPasses(const QueryRendererContext& ctx,
                    const std::set<GpuId>& used_gpus,
                    PerPassGpuCBFunc per_pass_gpu_callback,
                    PassCompleteCBFunc pass_complete_callback) const;

  // Local rendering entry point (single or multi-gpu)
  RenderPixels renderToPixels(QueryRendererContext& ctx) const;

 private:
  GlobalRenderContext& global_context_;

  void render(QueryRendererContext& ctx, const std::set<GpuId>& used_gpus) const;
  QueryFramebuffer* runAntialiasingPass(
      const QueryRendererContext& ctx,
      RootPerGpuData& gpu_data,
      QueryFramebuffer& render_fbo,
      SeparateMultiSamplesPass::SourceFramebuffer source) const;
};

};  // namespace QueryRenderer
