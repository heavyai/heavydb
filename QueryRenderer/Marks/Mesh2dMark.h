/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseMark.h"

#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"

namespace QueryRenderer {

class Mesh2dMark : public BaseMark {
 public:
  Mesh2dMark(const JSONLocation& obj_loc, QueryRendererContext& ctx);
  ~Mesh2dMark() override;

  bool draw(const gfx::DeviceContext& device_ctx,
            const MarkPerGpuData& mark_gpu_data,
            gfx::Framebuffer& framebuffer,
            const int accumulator_index) final;

  operator std::string() const final;

 private:
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
