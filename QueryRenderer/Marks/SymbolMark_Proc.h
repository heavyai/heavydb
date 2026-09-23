/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseMark.h"

#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderProperty.h"

namespace QueryRenderer {

class SymbolMark_Proc : public BaseMark {
 public:
  SymbolMark_Proc(const JSONLocation& obj_loc, QueryRendererContext& ctx);
  ~SymbolMark_Proc() override;

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

  BaseRenderPropertySet used_props_;
  BaseRenderPropertyConstSet used_props_const_;

  std::unique_ptr<MarkProjectionShaderPolicy> projection_policy_;

  gfx::PipelineDescriptorUqPtr pipeline_descriptor_;

  bool is_binned_heatmap_;

  BaseRenderPropertyConstSet getUsedProps() const final;
  void initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                 const bool data_changed,
                                 const bool init) final;
  CoordPackingTypeBits getSupportedCoordPackingTypes() const final {
    return CoordPackingTypeBits::kPackedPixel;
  }
  bool doAngle();

  void updateIsBinnedHeatmap();

  void updateShader() final;
  void setUniformAttributes(MarkPerGpuData& mark_gpu_data) final;

  void buildPipelineDescriptors() final;
  void buildPipelines(MarkPerGpuData& gpu_data) final;
  void buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) final;
  void updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                        const std::vector<GpuId>& remove_gpus) final;

  void drawWithMeshShader(const gfx::DeviceContext& device_ctx,
                          const MarkPerGpuData& mark_gpu_data,
                          gfx::Framebuffer& framebuffer,
                          gfx::RenderPass& render_pass);
  void drawWithVertexShader(const gfx::DeviceContext& device_ctx,
                            const MarkPerGpuData& mark_gpu_data,
                            gfx::Framebuffer& framebuffer,
                            gfx::RenderPass& render_pass);
};

}  // namespace QueryRenderer
