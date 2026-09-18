/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"

namespace QueryRenderer {

static constexpr const char* kDeclFuncNameStr = "projectx";

void MarkProjectionShaderPolicy::updateShader(
    gfx::ShaderManager::Builder& builder,
    const ProjectionShaderShPtr& x_projection_shader_src,
    const ProjectionShaderShPtr& y_projection_shader_src) const {
  // Add the uniform buffer declaration. Since there's currently just a single
  // UBO with a matrix, xDeclStr contains the actual body, and yDeclStr will be
  // be empty (or just newline).
  // TODO(scb) Builder should handle UBO constructions
  // auto const insert_pos = declFuncRange.begin() - markShaderSrc.begin();
  builder.insertBeforeFunction(kDeclFuncNameStr, x_projection_shader_src->getDecl());
  builder.insertBeforeFunction(kDeclFuncNameStr, y_projection_shader_src->getDecl());

  // TODO(scb): Fragile. Properties should have explicit flags for whether or not they
  // should be transformed by projections, and which axis to use
  for (auto const& prop : proj_props_) {
    auto const prop_name = prop->getName();
    auto const prop_type_str =
        (prop->isDecimal() ? prop->getDecimalTypeGLSL()->declString() : "double");
    auto const project_func = "project" + prop_name;
    switch (prop_name[0]) {
      case 'x':
        Projection::updateShaderFunction(
            builder, project_func, prop_type_str, x_projection_shader_src);
        break;
      case 'y':
        Projection::updateShaderFunction(
            builder, project_func, prop_type_str, y_projection_shader_src);
        break;
      default:
        CHECK(false) << "not a valid position property name: " << prop_name
                     << ". It must start with an x or y.";
    }
  }
}

}  // namespace QueryRenderer
