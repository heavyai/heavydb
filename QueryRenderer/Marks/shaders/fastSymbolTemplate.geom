/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Marks/fastSymbolDefines.glsl"
//</includes>
// GEOMETRY SHADER
// Inputs

// flat uint64_t gRowId[]
// flat vec4 gFillColor[]
// flat vec4 gStrokeColor[]
// flat float gStrokeWidth[]
// flat uint gShapeType[]
// flat vec2 gPointSize[]
// flat float gAngle (only when not uniform)
// flat int gAccumIdx[] (accumulation render only)
<GeometryShaderInputs>

// Outputs

// flat uint64_t fRowId
// flat uint fShapeType
// flat vec4 fSymbolScale
// flat float fPointSize
// flat vec4 fFillColor
// flat vec4 fStrokeColor
// flat float fStrokeWidth
// flat float fApproxMaxCoverage
// vec2 fUV
// flat int accumIdx (accumulation render only)
out <FragmentShaderInputs>;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

#define doAccumIndex <doAccumIndex>
#define useUangle <useUangle>
#define doHeatmapEdgePad <doHeatmapEdgePad>

layout(std430) uniform FAST_SYMBOL_GEOM_UBO_TYPE {
  float uPivotx;
  float uPivoty;
  mat3x2 uVPmatrix;
  float uInvViewportWidth;
  float uInvViewportHeight;
#if useUangle == 1
  float uSinAngle;
  float uCosAngle;
#endif // useUangle
};

void processVertex() {
  vec2 point_size = gPointSize[0];

  // Grow the point slightly in order to guarantee there are no subpixel gaps between the
  // points when rendering binned heatmaps
#if doHeatmapEdgePad == 1
  point_size += 0.1;
#endif // doHeatmapEdgePad

  float max_size = max(point_size.x, point_size.y);

  // Don't emit for small points
  if (max_size < CULL_SIZE) {
    return;
  }
  float out_stroke_width = gStrokeWidth[0];

  // determine the approximate maximum pixel coverage for the symbol. This is needed
  // for sizes < 1.0
  // TODO: incorporate stroke width?
  float out_approx_max_coverage = min(1.0, point_size.x) * min(1.0, point_size.y);

  // always round up otherwise we'll lose strokes that are < 0.5 pixels
  // on shapes that abut the edge of the quad
  float out_point_size = max_size + ceil(out_stroke_width);

#if useUangle == 0
  float sin_angle = sin(-gAngle[0]);
  float cos_angle = cos(-gAngle[0]);
  mat2 rot_tm = mat2(cos_angle * uInvViewportWidth, -sin_angle * uInvViewportWidth,
                     sin_angle * uInvViewportHeight, cos_angle * uInvViewportHeight);
#else // useUangle
  mat2 rot_tm = mat2(uCosAngle * uInvViewportWidth, -uSinAngle * uInvViewportWidth,
                     uSinAngle * uInvViewportHeight, uCosAngle * uInvViewportHeight);
#endif

  vec2 raw_pos = gl_in[0].gl_Position.xy;
  vec2 pos = vec2((uVPmatrix[0][0] * raw_pos.x) + uVPmatrix[2][0],
                  (uVPmatrix[1][1] * raw_pos.y) + uVPmatrix[2][1]);

  // pivot for wedge is offset 30%
  // The pad multiplier of 0.3333 was determined via empirical testing because my brain hurted
  float pivot_y = gShapeType[0] == WEDGE ? uPivoty + (0.3333 - (NUM_PAD_PIXELS * 0.3333 / out_point_size)) : uPivoty;
  vec2 pivot = vec2((uPivotx * 2.0) * (point_size.x + NUM_PAD_PIXELS), (pivot_y * 2.0) * (point_size.y + NUM_PAD_PIXELS));

  float point_size_padded = out_point_size + NUM_PAD_PIXELS;

  vec4 symbol_scale;
  if (gShapeType[0] == CIRCLE) {
    symbol_scale = vec4(point_size_padded / point_size.x, point_size_padded / point_size.y,
                        point_size.x / point_size_padded, point_size.y / point_size_padded);
  } else {
    float d = 1.0 / point_size_padded;
    symbol_scale = vec4(d * point_size.x, d * point_size.y, 0.0, 0.0);
  }

#if doAccumIndex == 1
#define EMIT_ACCUM_INDEX accumIdx = gAccumIdx[0]
#else
#define EMIT_ACCUM_INDEX
#endif
#define EMIT_UNIFORM_OUTPUTS                    \
  fRowId = gRowId[0];                           \
  EMIT_ACCUM_INDEX;                             \
  fShapeType = gShapeType[0];                   \
  fSymbolScale = symbol_scale;                  \
  fPointSize = out_point_size;                  \
  fFillColor = gFillColor[0];                   \
  fStrokeColor = gStrokeColor[0];               \
  fStrokeWidth = out_stroke_width;              \
  fApproxMaxCoverage = out_approx_max_coverage; \

  // VERTEX 0
  fUV = vec2(0.0, 0.0);
  gl_Position = vec4((vec2(-point_size_padded, -point_size_padded) + pivot) * rot_tm + pos, 0.0, 1.0);
  EMIT_UNIFORM_OUTPUTS;
  EmitVertex();

  // VERTEX 1
  fUV = vec2(1.0, 0.0);
  gl_Position = vec4((vec2(point_size_padded, -point_size_padded) + pivot) * rot_tm + pos, 0.0, 1.0);
  EMIT_UNIFORM_OUTPUTS;
  EmitVertex();

  // VERTEX 2
  fUV = vec2(0.0, 1.0);
  gl_Position = vec4((vec2(-point_size_padded, point_size_padded) + pivot) * rot_tm + pos, 0.0, 1.0);
  EMIT_UNIFORM_OUTPUTS;
  EmitVertex();

  // VERTEX 3
  fUV = vec2(1.0, 1.0);
  gl_Position = vec4((vec2(point_size_padded, point_size_padded) + pivot) * rot_tm + pos, 0.0, 1.0);
  EMIT_UNIFORM_OUTPUTS;
  EmitVertex();
}

void main() {
  // check for garbage vertex sentinel
  if (gShapeType[0] < NUM_SYMBOL_TYPES) {
    processVertex();
  }
  EndPrimitive();
}
