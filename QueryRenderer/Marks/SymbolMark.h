/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseMark.h"

#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderProperty.h"

namespace QueryRenderer {

class SymbolMark : public BaseMark {
 public:
  SymbolMark(const JSONLocation& obj_loc, QueryRendererContext& ctx);
  ~SymbolMark() override;

  bool draw(const gfx::DeviceContext& device_ctx,
            const MarkPerGpuData& mark_gpu_data,
            gfx::Framebuffer& framebuffer,
            const int accumulator_index) final;
  std::vector<CoordAttrInfo2d> getCoordPropAttrInfos() const final;

  operator std::string() const final;

  enum class CoordinateType { kPrimary = 0, kSecondary, kCenter };
  enum class DimensionType { kValue = 0, kCoords };
  struct CoordDimensionTypes {
    CoordinateType x_type;
    CoordinateType y_type;
    DimensionType width_type;
    DimensionType height_type;

    bool operator==(const CoordDimensionTypes& a) const {
      return x_type == a.x_type && y_type == a.y_type && width_type == a.width_type &&
             height_type == a.height_type;
    }

    bool operator!=(const CoordDimensionTypes& a) const { return !operator==(a); }
  };

 private:
  EnumRenderProperty shape_;
  RenderProperty<float> width_;
  RenderProperty<float> height_;
  RenderProperty<float> angle_;
  EnumRenderProperty angle_unit_;

  CoordDimensionTypes pos_dim_types_;

  BaseRenderPropertySet used_fill_props_;
  BaseRenderPropertySet used_stroke_props_;
  BaseRenderPropertyConstSet used_fill_props_const_;
  BaseRenderPropertyConstSet used_stroke_props_const_;

  std::unique_ptr<MarkProjectionShaderPolicy> projection_policy_;

  static std::vector<gfx::IndirectDrawIndexData> geom_fill_data;
  static std::vector<gfx::IndirectDrawVertexData> geom_stroke_data;

  enum DrawStage { kFill0, kFill1, kStroke0, kStroke1 };
  gfx::PipelineDescriptorUqPtr pipeline_descriptor_;

  BaseRenderPropertyConstSet getUsedProps() const final;
  void initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                 const bool data_changed,
                                 const bool init) final;
  CoordPackingTypeBits getSupportedCoordPackingTypes() const final {
    return CoordPackingTypeBits::kPackedPixel;
  }

  void buildShaders(ShaderBuilderVector& builders,
                    const BaseRenderPropertyConstSet& props);

  // Only the vertex shader uses subroutines
  void buildSubroutineBindings(ShaderBuilder& builder,
                               const BaseRenderPropertyConstSet& props);

  void updateShader() final;
  void setUniformAttributes(MarkPerGpuData& mark_gpu_data) final;
  void setUniformAttributes(const BaseRenderPropertyConstSet& props,
                            gfx::Material& active_material);

  void buildPipelineDescriptors() final;
  void buildPipelines(MarkPerGpuData& gpu_data) final;
  void buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) final;
  void buildStrokePrimitiveAssemblies(MarkPerGpuData& gpu_data) final;

  void updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                        const std::vector<GpuId>& remove_gpus) final;
};

}  // namespace QueryRenderer
