/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/noncopyable.hpp>

#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/Rendering/RenderAccumStats.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/AnyDataType.h"

namespace QueryRenderer {

class QueryFramebuffer;

class ScaleAccumRenderState : boost::noncopyable {
 public:
  static const uint32_t max_textures;

  explicit ScaleAccumRenderState(ScaleAccumState& sar);
  ScaleAccumRenderState() = delete;
  ~ScaleAccumRenderState();

  void bindPercentUniforms(gfx::Material& material,
                           const std::string& extra_suffix,
                           const AnyDataType* pct_cat_val,
                           const AnyDataType* pct_margin_val);
  void bindUniforms(gfx::Material& material,
                    const std::string& extra_suffix,
                    const AnyDataType* pct_cat_val,
                    const AnyDataType* pct_margin_val);

  void initGpuResources(QueryRendererContext& render_context, bool is_initializing);

  gfx::Material* get2ndPassMaterial(const GpuId gpu_id);
  gfx::Pipeline* getPipeline(const GpuId gpu_id);

  // facades to ScaleAccumState as needed by AccumRenderer
  AccumulatorType getAccumulatorType() const;
  uint32_t getNumTextures() const;
  bool getDoFindMinDensity() const;
  bool getDoFindMaxDensity() const;
  bool getDoFindExtents() const;
  bool getDoFindStdDev() const;
  uint32_t getMinDensity() const;
  uint32_t getMaxDensity() const;
  uint8_t getNumMinStdDev() const;
  uint8_t getNumMaxStdDev() const;
  void setAccumStats(RenderAccumStatsUqPtr&& accum_stats) const;
  BaseScale& getParentScale() const;

 private:
  ScaleAccumState& scale_accum_state_;

  class PerGpuData;
  using PerGpuDataMap = std::map<GpuId, PerGpuData>;

  PerGpuDataMap per_gpu_data_map_;

  gfx::PipelineDescriptorUqPtr pipeline_descriptor_;

  void clearResources();
  void destroyPipelines();

  PerGpuData& getGpuData(GpuId gpu_id);
};

}  // namespace QueryRenderer
