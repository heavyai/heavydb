/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseMark.h"

#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderProperty.h"

namespace QueryRenderer {

class LineMark : public BaseMark {
 public:
  LineMark(const JSONLocation& obj_loc, QueryRendererContext& ctx);
  ~LineMark() override;

  bool draw(const gfx::DeviceContext& device_ctx,
            const MarkPerGpuData& mark_gpu_data,
            gfx::Framebuffer& framebuffer,
            const int accumulator_index) final;

  operator std::string() const final;

 private:
  BoolRenderProperty fill_below_line_;

  BaseRenderPropertySet used_props_;
  BaseRenderPropertyConstSet used_props_const_;

  std::unique_ptr<MarkProjectionShaderPolicy> projection_policy_;

  gfx::PipelineDescriptorUqPtr pipeline_descriptor_;

  void dataRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj) final;
  BaseRenderPropertyConstSet getUsedProps() const final;
  void initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                 const bool data_changed,
                                 const bool init) final;
  CoordPackingTypeBits getSupportedCoordPackingTypes() const final {
    return CoordPackingTypeBits::kCompressedGeo;
  }
  void buildShaders(ShaderBuilderVector& builders,
                    const BaseRenderPropertyConstSet& props,
                    const std::string& ssbo_name,
                    const std::string& ssbo_instance_name);

  void updateShader() final;
  void setUniformAttributes(MarkPerGpuData& gpu_data) final;

  void buildPipelineDescriptors() final;
  void buildPipelines(MarkPerGpuData& gpu_data) final;
  void buildStrokePrimitiveAssemblies(MarkPerGpuData& gpu_data) final;

  void buildSubroutineBindings(ShaderBuilder& builder);

  void updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                        const std::vector<GpuId>& remove_gpus) final;

  static void buildPrimitiveAssemblyData(const GpuId& gpu_id,
                                         const BaseRenderPropertyConstSet& vbo_props,
                                         const BaseDataTableShPtr& data,
                                         gfx::Material& active_material,
                                         gfx::PrimitiveAssemblyAttrInfo& attr_info,
                                         const gfx::IndexBuffer*& ibo);
};

};  // namespace QueryRenderer
