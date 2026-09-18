/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Marks/fastSymbolDefines.glsl"
//</includes>
// GEOMETRY SHADER
// Inputs

// flat uint64_t gRowId[]
// flat float gPointSize[]
// flat vec4 gFillColor[]
// flat vec4 gStrokeColor[]
// flat float gStrokeWidth[]
// flat uint gBarbId[]
// flat float gDirection[] // only if not uniform
// flat float gAnchorScale[]
<GeometryShaderInputs>

// Outputs

// flat uint64_t fRowId
// vec2 fUV
// flat uint fBarbId
// flat float fPointScale
// flat float fPointSize
// flat float fApproxMaxCoverage
// flat float fAnchorScale
// flat vec4 fFillColor
// flat vec4 fStrokeColor
// flat float fStrokeWidth
out <FragmentShaderInputs>;

#define useUdirection <useUdirection>

layout(std430) uniform FAST_SYMBOL_GEOM_UBO_TYPE {
  mat3x2 uVPmatrix;
  float uInvViewportWidth;
  float uInvViewportHeight;
#if useUdirection == 1
  float uSinDirection;
  float uCosDirection;
#endif // useUdirection
};

#define NUM_BARB_TYPES <numBarbTypes>
#define DO_MIRROR_Y <doMirrorY>

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

void processVertex() {
  float point_size = gPointSize[0];

  float out_stroke_width = gStrokeWidth[0];

  // determine the approximate maximum pixel coverage for the symbol. This is needed
  // for sizes < 1.0
  // TODO: incorporate stroke width?
  float out_approx_max_coverage = min(1.0, point_size);

  // always round up otherwise we'll lose strokes that are < 0.5 pixels
  // on shapes that abut the edge of the quad
  float out_point_size = point_size + ceil(out_stroke_width);

#if useUdirection == 0
  #if DO_MIRROR_Y == 1
  float sin_dir = sin(gDirection[0]);
  float cos_dir = cos(gDirection[0]);
  #else
  float sin_dir = sin(-gDirection[0]);
  float cos_dir = cos(-gDirection[0]);
  #endif
  mat2 rot_tm = mat2(cos_dir * uInvViewportWidth, -sin_dir * uInvViewportWidth,
                     sin_dir * uInvViewportHeight, cos_dir * uInvViewportHeight);
#else // useUdirection
  mat2 rot_tm = mat2(uCosDirection * uInvViewportWidth, -uSinDirection * uInvViewportWidth,
                     uSinDirection * uInvViewportHeight, uCosDirection * uInvViewportHeight);
#endif

  vec2 raw_pos = gl_in[0].gl_Position.xy;
  vec2 pos = vec2((uVPmatrix[0][0] * raw_pos.x) + uVPmatrix[2][0],
                  (uVPmatrix[1][1] * raw_pos.y) + uVPmatrix[2][1]);

  vec2 pivot;
  if (gBarbId[0] == 0) {
    pivot = vec2(0, 0);
  } else {
    float pivot_x = 0.45;
#if DO_MIRROR_Y == 1
    float pivot_y = -0.45;
#else
    float pivot_y = 0.45;
#endif
    pivot = vec2((pivot_x * 2.0) * (point_size + NUM_PAD_PIXELS), (pivot_y * 2.0) * (point_size + NUM_PAD_PIXELS));
  }
  float point_size_padded = out_point_size + NUM_PAD_PIXELS;

  float point_scale = point_size_padded / point_size;

#define EMIT_UNIFORM_OUTPUTS                    \
  fRowId = gRowId[0];                           \
  fBarbId = gBarbId[0];                         \
  fPointScale = point_scale;                    \
  fPointSize = out_point_size;                  \
  fAnchorScale = gAnchorScale[0];               \
  fFillColor = gFillColor[0];                   \
  fStrokeColor = gStrokeColor[0];               \
  fStrokeWidth = out_stroke_width;              \
  fApproxMaxCoverage = out_approx_max_coverage; \

  float u_max, x_max;
  if (gBarbId[0] == 0){
    u_max = 1.0;
    x_max = point_size_padded;
  } else {
    u_max = 0.0;
    x_max = 0.0;
  }

#if DO_MIRROR_Y == 0
  #define V_MIN -1.0
  #define V_MAX 1.0
#else
  #define V_MIN 1.0
  #define V_MAX -1.0
#endif

  // VERTEX 0
  fUV = vec2(-1.0, V_MIN);
  gl_Position = vec4((vec2(-point_size_padded, -point_size_padded) + pivot) * rot_tm + pos, 0.0, 1.0);
  EMIT_UNIFORM_OUTPUTS;
  EmitVertex();

  // VERTEX 1
  fUV = vec2(u_max, V_MIN);
  gl_Position = vec4((vec2(x_max, -point_size_padded) + pivot) * rot_tm + pos, 0.0, 1.0);
  EMIT_UNIFORM_OUTPUTS;
  EmitVertex();

  // VERTEX 2
  fUV = vec2(-1.0, V_MAX);
  gl_Position = vec4((vec2(-point_size_padded, point_size_padded) + pivot) * rot_tm + pos, 0.0, 1.0);
  EMIT_UNIFORM_OUTPUTS;
  EmitVertex();

  // VERTEX 3
  fUV = vec2(u_max, V_MAX);
  gl_Position = vec4((vec2(x_max, point_size_padded) + pivot) * rot_tm + pos, 0.0, 1.0);
  EMIT_UNIFORM_OUTPUTS;
  EmitVertex();
}

void main() {
  // check for garbage vertex sentinel
  if (gBarbId[0] < NUM_BARB_TYPES) {
    processVertex();
  }
  EndPrimitive();
}
