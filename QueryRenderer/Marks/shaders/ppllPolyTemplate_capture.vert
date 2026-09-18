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

layout(std430) uniform POLY_CAPTURE_VERT_UBO_TYPE {
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

layout(location = 0) flat out uint32_t fPolygonID;

layout(binding = 5, std430) buffer POLY_CAPTURE_POLYGON_ID_SSBO {
  uint32_t polygonIDs[];
};

// push constants
layout(push_constant) uniform POLY_CAPTURE_VERT_PUSH_CONSTANTS {
  uint32_t batchOffset;
} pushConstants;


void main() {
  if (gl_DrawIDARB < polygonIDs.length()) {
    fPolygonID = polygonIDs[gl_DrawIDARB + pushConstants.batchOffset];
  } else {
    fPolygonID = 0u;
  }

  gl_Position =
      vec4(float(getx(projectx(x))) * uViewProjMatrix[0][0] + uViewProjMatrix[2][0],
           float(gety(projecty(y))) * uViewProjMatrix[1][1] + uViewProjMatrix[2][1],
           0.5,
           1.0);
}
