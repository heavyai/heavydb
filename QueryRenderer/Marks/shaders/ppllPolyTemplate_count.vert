/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<extensions>
#extension GL_ARB_shader_draw_parameters : require
//</extensions>
//<includes>
#include "Marks/typeDefines.glsl"
#include "Marks/coordConvertSubroutines.glsl"
//</includes>
// VERTEX SHADER
// Input RenderProperties
<RenderPropertyTypeInfos>

// Vertex attribute inputs
<VertexProperties>

// viewport data
struct Viewport {
  int x;
  int y;
  int width;
  int height;
};

layout(std430, binding = 0) uniform SHARED_VIEWPORT_UBO {
  Viewport viewport;
};

layout(std430) uniform POLY_COUNT_VERT_UBO_TYPE {
  mat3x2 uViewProjMatrix;
};

// get* functions for properties
<PropertyGetters>

// project* functions for coord inputs
double projectx(in double x) {
  return x;
}
double projecty(in double y) {
  return y;
}

void main() {
  // For counting we can ignore z
  gl_Position =
      vec4(float(getx(projectx(x))) * uViewProjMatrix[0][0] + uViewProjMatrix[2][0],
           float(gety(projecty(y))) * uViewProjMatrix[1][1] + uViewProjMatrix[2][1],
           0.5,
           1.0);
}
