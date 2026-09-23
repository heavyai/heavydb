/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidjson/document.h>
#include <boost/noncopyable.hpp>

#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Rendering/RenderAccumStats.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/AnyDataType.h"

namespace QueryRenderer {

class QueryFramebuffer;

class ScaleAccumState : boost::noncopyable {
 public:
  static const uint32_t max_textures;

  explicit ScaleAccumState(BaseScale& parent_scale, QueryRendererContext& render_context);
  ScaleAccumState() = delete;
  ~ScaleAccumState() = default;

  AccumulatorType getType() const { return accumulator_type_; }
  bool supportsDomainCoercion() const;
  bool supportsCustomNullValue() const;

  static uint32_t convertNumTexturesToNumVals(const uint32_t num_textures,
                                              const AccumulatorType accum_type);

  //
  // parsing
  //
  // return true to indicate the accumulator changed (will trigger shader rebuild)
  bool updateFromJSONObj(const JSONLocation& json_loc);

  void postScaleUpdateFromJSONObj(const bool scale_props_changed);

  // validity
  bool hasTypeChanged() const { return type_changed_; }
  bool hasNumTexturesChanged() const { return num_textures_changed_; }
  bool hasPctAccumChanged() const { return pct_accum_changed_; }
  void resetChangedFlags();

  // Serialization
  void toJSON(rapidjson::Value& obj, rapidjson::Document::AllocatorType& allocator) const;

  //
  // Shaders
  //
  // Override the scale's shader template (needed for pct accumulation)
  // returns either an empty string or the name of an alternate template to use
  std::string getScaleShaderTemplateOverride() const;
  ScaleShaderUpdateFlags updateScaleShaderBuilder(
      const BaseScaleRef* scale_ref,
      gfx::ShaderManager::Builder& builder) const;
  gfx::ShaderManager::BuilderUqPtr get1stPassFragSubBuilder();

  void bindScaleSubroutines(gfx::ShaderManager::Builder& builder);

  // textures
  uint32_t getNumTextures();

  //
  // Percent Accumulation
  //
  gfx::TypeGLSLShPtr getPercentTypeGLSL() const;
  const AnyDataType* getPercentCategoryVal() const;
  const AnyDataType* getPercentMargin() const;

  // Logging
  // FIXME(scb) - verify output is useful
  std::string toString() const;

  void setAccumStats(RenderAccumStatsUqPtr&& accum_stats);

  bool getDoFindExtents() const;

 private:
  BaseScale& parent_scale_;
  AccumulatorType accumulator_type_;
  QueryRendererContext& render_context_;

  bool type_changed_;
  bool scale_props_changed_;

  uint32_t num_values_;
  uint32_t num_textures_;
  bool num_values_changed_;
  bool num_textures_changed_;

  // dirty flag for the accum 2nd pass shader
  bool is_shader_dirty_;

  // changed flags that affect the accum 2nd pass shader
  void setTypeChanged() {
    is_shader_dirty_ = true;
    type_changed_ = true;
  }
  void setScalePropsChanged() {
    is_shader_dirty_ = true;
    scale_props_changed_ = true;
  }
  void setNumValsChanged() {
    is_shader_dirty_ = true;
    num_values_changed_ = true;
  }
  void setFindDensityExtentsChanged() { is_shader_dirty_ = true; }

  uint8_t num_min_std_dev_;
  uint32_t min_density_;
  bool do_find_min_density_;

  uint8_t num_max_std_dev_;
  uint32_t max_density_;
  bool do_find_max_density_;
  bool do_find_std_dev_;
  RenderAccumStatsUqPtr accum_stats_;

  std::unique_ptr<AnyDataType> pct_cat_val_;
  std::unique_ptr<AnyDataType> pct_margin_val_;
  bool pct_accum_changed_;

  std::string getSubroutineName() const;

  // Build the full-screen shader that will gather the results from the
  // first accumulation pass and generate a final color mapped image
  // num_textures is per-gpu, tex-arrays are for multi-gpu or distributed compositors
  gfx::ShaderManager::BuilderUqPtrVector get2ndPassShaderBuilders(uint32_t num_textures);
  void buildSubroutineBindings(gfx::ShaderManager::Builder& builder);

  std::string getPercentCategoryUniformName() const;
  std::string getPercentMarginUniformName() const;

  void setNumVals(const uint32_t num_vals);

  friend class ScaleAccumRenderState;
};

}  // namespace QueryRenderer
