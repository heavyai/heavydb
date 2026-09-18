/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<extensions>
#extension GL_ARB_shader_draw_parameters : require
//</extensions>
// GEOMETRY SHADER

// Alternative version of lineTemplate.geom for 1D cross-section terrain renders
// Drop-in replacement but ignores most line attributes and renders a solid quad
// for each line segment, with the bottom at the bottom of the image

// Inputs

// uint64_t gRowId[]
// int gAccumIdx[] (accumulation renders only)
// float gOpacity[]
// vec4 gColor[]
// float gStrokeWidth[]
// float gStrokeOpacity[]
// int gLineJoin[]
// float gMiterLimit[]
<GeometryShaderInputs>

// Outputs

// flat uint64_t fRowId
// flat int accumIdx (accumulation renders only)
// vec2 fNormDistCoords
// flat vec4 fColor
// flat float fOpacity
// flat float fStrokeOpacity
out <FragmentShaderInputs>;

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

struct Viewport
{
  int x;
  int y;
  int width;
  int height;
};

layout(std430, binding = 0) uniform SHARED_VIEWPORT_UBO {
  Viewport viewport;
};

#define doStrokeAccum <doStrokeAccum>

#define NO_DIST_COORDS vec2(0.0)

vec2 ScreentoNDC(in vec2 v) {
  float nx = ((2.0 / float(viewport.width)) * (v.x - viewport.x)) - 1.0;
  float ny = ((2.0 / float(viewport.height)) * (v.y - viewport.y)) - 1.0;
  return vec2(nx, ny);
}

void emitVertex(
  in vec2 pos,
  in vec2 zw,
  in vec2 norm_dist_coords,
  in bool is_bottom_vertex) {
  vec2 ndc_pos = ScreentoNDC(pos);
  if (is_bottom_vertex) {
    ndc_pos.y = -1.0;
  }
  gl_Position = vec4(ndc_pos, zw);
  fRowId = gRowId[1];
#if doStrokeAccum == 1
  accumIdx = gAccumIdx[1];
#endif
  fNormDistCoords = norm_dist_coords;
  fColor = gColor[1];
  fOpacity = gOpacity[1];
  fStrokeOpacity = gStrokeOpacity[1];
  EmitVertex();
}

void renderStroke(
  in vec2 pos0,
  in vec2 pos1,
  in vec2 pos2,
  in vec2 zw1,
  in vec2 zw2,
  in float begin_line_width,
  in float end_line_width,
  in int line_join,
  in float miter_limit) {
  // main segment
  vec2 segment_dir = pos2 - pos1;
  float segment_len = length(segment_dir);
  if (segment_len > 0.0) {
    // worth drawing this segment
    segment_dir = segment_dir / segment_len;

    // draw quad
    emitVertex(pos1, zw1, NO_DIST_COORDS, false);
    emitVertex(pos2, zw2, NO_DIST_COORDS, false);
    emitVertex(pos1, zw1, NO_DIST_COORDS, true);
    emitVertex(pos2, zw2, NO_DIST_COORDS, true);
    EndPrimitive();
  }
}

void main() {
#if doStrokeAccum == 1
  // always draw
  {
#else
  // draw if not transparent
  if (gColor[1].a > 0.0 || gColor[2].a > 0.0) {
#endif
    renderStroke(
      gl_in[0].gl_Position.xy,
      gl_in[1].gl_Position.xy,
      gl_in[2].gl_Position.xy,
      gl_in[1].gl_Position.zw,
      gl_in[2].gl_Position.zw,
      gStrokeWidth[1],
      gStrokeWidth[2],
      gLineJoin[1],
      gMiterLimit[1]);
  }
}
