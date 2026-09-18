/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<extensions>
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_shader_draw_parameters : require
//</extensions>
//<includes>
#include "Marks/typeDefines.glsl"
#include "Utils/colorConvertSubroutines.glsl"
#include "Marks/coordConvertSubroutines.glsl"
#include "Marks/slabAddressTable.glsl"
//</includes>
// VERTEX SHADER

// Input RenderProperties
<RenderPropertyTypeInfos>

// Vertex attribute inputs
<VertexProperties>

// Outputs

// uint64_t gRowId
// int accumIdx (accumulation renders only)
// float gOpacity
// vec4 gColor
// float gStrokeWidth
// float gStrokeOpacity
// int gLineJoin
// float gMiterLimit
out <GeometryShaderInputs>;

// Uniform RenderProperties
layout(std430) uniform <UniformProperties>;

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

// set to 1 to handle accumulation rendering in a geometry shader
#define supportsGeoAccum 1
#define useSSBO <useSSBO>

// SSBO properties
#if useSSBO == 1
<lineData>
#endif

// Push constants
layout(push_constant) uniform LINE_VERT_PUSH_CONSTANTS {
  uint totalNumItems;
} pushConstants;

// Color conversion stubs
vec4 transformstrokeColorToRGB(in vec4 c) { return c; }
vec4 unpackstrokeColor(in uint c) { return vec4(0); }

// get* functions for properties
<PropertyGetters>

// project* functions for coord inputs
double projectx(in double x) {
  return x;
}
double projecty(in double y) {
  return y;
}

// General #defines
#define numid <numid>

vec2 NDCtoScreen(in float x, in float y) {
  return vec2((x + 1) * (float(viewport.width) / 2.0) + viewport.x,
              (y + 1) * (float(viewport.height) / 2.0) + viewport.y);
}
void main() {
#if inTxEnum == UNSIGNED_INT64_ARB
  float px = float(projectx(getx_ptr(x, propCompressionBits, PROP_COMPRESSION_BIT_X, 0)));
#else
  float px = float(getx(projectx(x)));
#endif
#if inTyEnum == UNSIGNED_INT64_ARB
  float py = float(projecty(gety_ptr(y, propCompressionBits, PROP_COMPRESSION_BIT_Y, 0)));
#else
  float py = float(gety(projecty(y)));
#endif
#if useSSBO == 1
  int iSSBOIndex = gl_DrawIDARB + uSSBOIndexBase;
  gl_Position =
      vec4(NDCtoScreen(
               px * uViewProjMatrix[0][0] + uViewProjMatrix[2][0],
               py * uViewProjMatrix[1][1] + uViewProjMatrix[2][1]),
           clamp(float(iSSBOIndex + 1) / float(pushConstants.totalNumItems), 0.0, 1.0),
           1.0);
#else
  gl_Position =
      vec4(NDCtoScreen(
               px * uViewProjMatrix[0][0] + uViewProjMatrix[2][0],
               py * uViewProjMatrix[1][1] + uViewProjMatrix[2][1]),
           0.5,
           1.0);
#endif

#if useSSBO == 1 && useUid == 0
#if numid >= 1
  gRowId = lineData[iSSBOIndex].<id> + 1;
#endif
#if numid >= 2
  int total_bitshift = id_offsets[0];
  gRowId = ((lineData[iSSBOIndex].<id1> + 1) << total_bitshift) | gRowId;
#endif
#if numid >= 3
  total_bitshift += id_offsets[1];
  gRowId = ((lineData[iSSBOIndex].<id2> + 1) << total_bitshift) | gRowId;
#endif
#else
#if numid >= 1
  gRowId = uint64_t(id + 1);
#endif
#if numid >= 2
  int total_bitshift = id_offsets[0];
  gRowId = (uint64_t(id1 + 1) << total_bitshift) | gRowId;
#endif
#if numid >= 3
  total_bitshift += id_offsets[1];
  gRowId = (uint64_t(id2 + 1) << total_bitshift) | gRowId;
#endif
#endif

#if useSSBO == 1 && useUstrokeColor == 0
  gColor = transformstrokeColorToRGB(getstrokeColor(lineData[iSSBOIndex].<strokeColor>));
#else
  gColor = transformstrokeColorToRGB(getstrokeColor(strokeColor));
#endif

#if useSSBO == 1 && useUopacity == 0
  gOpacity = float(getopacity(lineData[iSSBOIndex].<opacity>));
#else
  gOpacity = float(getopacity(opacity));
#endif

#if useSSBO == 1 && useUstrokeWidth == 0
  gStrokeWidth = float(getstrokeWidth(lineData[iSSBOIndex].<strokeWidth>));
#else
  gStrokeWidth = float(getstrokeWidth(strokeWidth));
#endif

#if useSSBO == 1 && useUstrokeOpacity == 0
  gStrokeOpacity = float(getstrokeOpacity(lineData[iSSBOIndex].<strokeOpacity>));
#else
  gStrokeOpacity = float(getstrokeOpacity(strokeOpacity));
#endif

#if useSSBO == 1 && useUlineJoin == 0
  gLineJoin = lineData[iSSBOIndex].<lineJoin>;
#else
  gLineJoin = lineJoin;
#endif
#if useSSBO == 1 && useUmiterLimit == 0
  gMiterLimit = float(getmiterLimit(lineData[iSSBOIndex].<miterLimit>));
#else
  gMiterLimit = float(getmiterLimit(miterLimit));
#endif
}
