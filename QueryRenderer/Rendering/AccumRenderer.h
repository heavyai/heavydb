/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/noncopyable.hpp>

#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

class AccumRenderer : boost::noncopyable {
 public:
  explicit AccumRenderer(GlobalRenderContext& global_ctx) : global_ctx_{global_ctx} {}
  AccumRenderer() = delete;
  ~AccumRenderer() = default;

  void render(const gfx::DeviceContext& device_context,
              ScaleAccumRenderState& scale_accum_render_state,
              gfx::Texture& accum_texture_array,
              QueryFramebuffer& framebuffer,
              const bool do_id_pass);

  struct Extents {
    uint64_t totalNonZeroCount;
    uint64_t totalSqrDiff;
    uint32_t minCount;
    uint32_t maxCount;
    uint32_t numNonZeroCount;
  };

 private:
  // extents buffer (may be null), do_std_dev
  std::pair<gfx::BufferWrapper*, bool> updateMaterials(
      ScaleAccumRenderState& scale_accum_render_state,
      const gfx::Texture& accum_texture_array,
      const RootPerGpuData& gpu_data);

  GlobalRenderContext& global_ctx_;
};

}  // namespace QueryRenderer
