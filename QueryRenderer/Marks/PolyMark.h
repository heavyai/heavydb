/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseMark.h"

#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"

namespace QueryRenderer {

class PolyMark : public BaseMark {
 public:
  PolyMark(const JSONLocation& obj_loc, QueryRendererContext& ctx);
  ~PolyMark() override;

  bool usesPerPixelLinkedLists() const final { return true; }

  bool draw(const gfx::DeviceContext& device_ctx,
            const MarkPerGpuData& mark_gpu_data,
            gfx::Framebuffer& framebuffer,
            const int accumulator_index) final;

  operator std::string() const final;

 private:
  BaseRenderPropertySet used_fill_props_;
  BaseRenderPropertySet used_stroke_props_;
  BaseRenderPropertyConstSet used_fill_props_const_;
  BaseRenderPropertyConstSet used_stroke_props_const_;
  BaseRenderPropertyConstSet used_projection_props_const_;

  std::unique_ptr<MarkProjectionShaderPolicy> projection_policy_;

  enum Materials {
    kCountFragmentsMaterial,
    kCaptureFragmentsMaterial,
    kCompositeFragmentsMaterial,
    kOutlineMaterial,
    kNumMaterials
  };

  enum GraphicsPipelines {
    kCountFragmentsPipeline,
    kCaptureFragmentsPipeline,
    kOutlinePipeline,
    kNumGraphicsPipelines
  };

  enum ComputePipelines { kCompositeFragmentsPipeline, kNumComputePipelines };

  enum Specializations { kDefaultComposite, kOverflowComposite };

  static const uint32_t kTileIndexPushConstantOffset{4};
  static gfx::PushConstantRanges capture_and_comp_push_constants;
  gfx::PipelineDescriptorUqPtr pipeline_descriptors_[4];

  void dataRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj) final;
  BaseRenderPropertyConstSet getUsedProps() const final;
  void initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                 const bool data_changed,
                                 const bool init) final;

  void initPPLLPerGpuData(MarkPerGpuData& gpu_data) final;

  CoordPackingTypeBits getSupportedCoordPackingTypes() const final {
    return CoordPackingTypeBits::kCompressedGeo;
  }
  void buildShaders(ShaderBuilderVector& builders,
                    const BaseRenderPropertyConstSet& props,
                    const std::string& ssbo_name,
                    const std::string& ssbo_instance_name,
                    const bool auto_inject_main,
                    const std::string& vertex_shader_inputs,
                    const std::string& ubo_block_string);

  void updateShader() final;
  void setUniformAttributes(MarkPerGpuData& gpu_data) final;

  void buildPipelineDescriptors() final;
  void buildPipelines(MarkPerGpuData& gpu_data) final;
  void buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) final;
  void buildStrokePrimitiveAssemblies(MarkPerGpuData& gpu_data) final;

  // Only the vertex shader uses subroutines
  void buildSubroutineBindings(ShaderBuilder& builder,
                               const BaseRenderPropertyConstSet& props);

  void setUniformAttributes(gfx::Material& active_material,
                            const BaseRenderPropertyConstSet& props,
                            const bool is_stencil_pass,
                            const bool requires_projection);

  void updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                        const std::vector<GpuId>& remove_gpus) final;

  CommonRenderPassTypeBits getRequiredCommonRenderPassTypes() const final;

  void drawFill(const gfx::DeviceContext& device_ctx,
                MarkPerGpuData& mark_gpu_data,
                gfx::Framebuffer& framebuffer,
                const int accumulator_index);

  static void buildAttrMap(const GpuId& gpu_id,
                           const BaseRenderPropertyConstSet& vbo_props,
                           gfx::Material& active_material,
                           gfx::PrimitiveAssemblyAttrInfo& attr_info);
};

}  // namespace QueryRenderer
