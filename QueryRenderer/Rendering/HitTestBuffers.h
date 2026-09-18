/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Objects/Array2d.h"
#include "QueryRenderer/Interface/ResultCacheTypes.h"
#include "QueryRenderer/Rendering/Types.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

class QueryRendererContext;
class QueryFramebuffer;

struct HitInfo {
  ResultCacheId result_cache_id = 0u;
  uint64_t row_id = 0u;
  uint8_t vega_data_id = 0u;
  int16_t node_index = 0;  // encoded in hit-test cache id; always 0 for single-node
};

class HitTestBuffers {
 public:
  explicit HitTestBuffers(const QueryRendererContext& ctx);
  ~HitTestBuffers();

  void updateFromFramebuffer(QueryFramebuffer& fbo);
  HitInfo getIdAt(uint32_t x, uint32_t y, uint32_t pixel_radius);

  // PBO management
  void createPbo(const GpuId final_render_gpu_id, int width = -1, int height = -1);
  void releasePbo();

  // Cpu buffers
  inline void setCpuCacheDirty() { id_pixels_dirty_ = true; }
  void resetCpuCache();

  void resize(uint32_t width, uint32_t height);

  HitTestBuffers() = delete;
  HitTestBuffers(const HitTestBuffers&) = delete;
  HitTestBuffers& operator=(const HitTestBuffers&) = delete;

 private:
  bool updateCpuCache();

  // parent QueryRendererContext
  const QueryRendererContext& ctx_;

  // Id buffers for hit-testing for this render session
  GpuId pbo_gpu_;
  // The row id is a 64-bit int, so its packed into 2 32-bit textures
  QueryIdMapPixelBufferShPtr pbo_1a_;
  QueryIdMapPixelBufferWkPtr pbo_1a_wk_;
  QueryIdMapPixelBufferShPtr pbo_1b_;
  QueryIdMapPixelBufferWkPtr pbo_1b_wk_;
  // ResultCacheId buffers
  QueryIdMapPixelBufferShPtr pbo_2_;
  QueryIdMapPixelBufferWkPtr pbo_2_wk_;

  using Array2dui = ::gfx::Objects::Array2d<uint32_t>;

  bool id_pixels_dirty_;

  std::unique_ptr<Array2dui> id_1a_pixels_;
  std::unique_ptr<Array2dui> id_1b_pixels_;
  std::unique_ptr<Array2dui> id_2_pixels_;
};

using HitTestBuffersUqPtr = std::unique_ptr<HitTestBuffers>;

}  // namespace QueryRenderer
