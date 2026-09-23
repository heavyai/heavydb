/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Projections/ProjectionShader.h"
#include "QueryRenderer/Projections/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

class ProjectionShaderPolicy {
 public:
  virtual ~ProjectionShaderPolicy() {}

  virtual void updateShader(
      gfx::ShaderManager::Builder& builder,
      const ProjectionShaderShPtr& x_projection_shader_src,
      const ProjectionShaderShPtr& y_projection_shader_src) const = 0;
};

class Projection : public JSONRefObject {
 public:
  Projection(const JSONLocation& json_loc,
             QueryRendererContext& ctx,
             const std::string& name = "",
             const ProjectionType type = ProjectionType::kUndefined)
      : JSONRefObject(ctx, RefType::kProjection, name, json_loc.getPathRef())
      , type_{type} {}

  ~Projection() override {}

  virtual bool updateFromJSONObj(const JSONLocation& json_loc) = 0;

  virtual void setUniformAttributes(gfx::Material& active_material) = 0;
  virtual ProjectionShaderShPtrPair getShaderSrc() = 0;

  void updateShader(gfx::ShaderManager::Builder& builder,
                    const ProjectionShaderPolicy& mark_policy) {
    auto projection_shader_src = getShaderSrc();
    mark_policy.updateShader(builder, projection_shader_src.x, projection_shader_src.y);
  }

  ProjectionType type() const { return type_; }

  static void updateShaderFunction(gfx::ShaderManager::Builder& builder,
                                   const std::string& func_name,
                                   const std::string& func_type,
                                   const ProjectionShaderShPtr& projection_shader_src);

 protected:
  ProjectionType type_;

  void toJSONInternal(rapidjson::Value& obj,
                      rapidjson::Document::AllocatorType& allocator) const final {
    THROW_RUNTIME_EX("Projection(" + name_ +
                     "): Evaluate projection to JSON is not supported.");
  }

  friend class QueryRendererContext;
};

/**
 * The Mercator projection implements a transformation from WGS84 spherical longitude /
 * latitude coordinates (EPSG 4326) to WebMercator cartesian coordinates (EPSG 3857 /
 * 900913). The projection accepts a bounds argument, also in latlon coordinates, that
 * determines both the translation and scaling of the resulting cartesian coordinates.
 */
class MercatorProjection : public Projection {
 public:
  MercatorProjection(const JSONLocation& json_loc,
                     QueryRendererContext& ctx,
                     const std::string& name,
                     const ProjectionType type)
      : Projection(json_loc, ctx, name, type) {
    updateFromJSONObj(json_loc);
    initShaderSource();
  }

  bool updateFromJSONObj(const JSONLocation& json_loc) override;

  ProjectionShaderShPtrPair getShaderSrc() final;

  void setUniformAttributes(gfx::Material& active_material) final;

 protected:
  ProjectionShaderShPtr x_projection_shader_;
  ProjectionShaderShPtr y_projection_shader_;

  std::array<double, 2> x_bounds_ = {{-180., 180.}};
  std::array<double, 2> y_bounds_ = {{-85.06, 85.06}};

  void initShaderSource();

  static double transformX(const double x);
  static double transformY(const double y);
};

}  // namespace QueryRenderer
