/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseMark.h"

#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderProperty.h"

namespace QueryRenderer {

class WindBarbMark : public BaseMark {
 public:
  WindBarbMark(const JSONLocation& obj_loc, QueryRendererContext& ctx);
  ~WindBarbMark() override;

  bool draw(const gfx::DeviceContext& device_ctx,
            const MarkPerGpuData& mark_gpu_data,
            gfx::Framebuffer& framebuffer,
            const int accumulator_index) final;

  operator std::string() const final;

 private:
  RenderProperty<float> size_;
  RenderProperty<float> speed_;
  RenderProperty<float> direction_;
  RenderProperty<float> anchor_scale_;
  BoolRenderProperty do_quantize_direction_;

  BaseRenderPropertySet used_props_;
  BaseRenderPropertyConstSet used_props_const_;

  std::unique_ptr<MarkProjectionShaderPolicy> projection_policy_;

  gfx::PipelineDescriptorUqPtr pipeline_descriptor_;

  BaseRenderPropertyConstSet getUsedProps() const final;
  void initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                 const bool data_changed,
                                 const bool init) final;

  void updateShader() final;
  void setUniformAttributes(MarkPerGpuData& mark_gpu_data) final;

  void buildPipelineDescriptors() final;
  void buildPipelines(MarkPerGpuData& gpu_data) final;
  void buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) final;
  void updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                        const std::vector<GpuId>& remove_gpus) final;
};

}  // namespace QueryRenderer
