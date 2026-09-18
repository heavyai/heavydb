/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Marks/typeDefines.glsl"
#include "Utils/colorConvertSubroutines.glsl"
#include "Marks/coordConvertSubroutines.glsl"
//</includes>
// VERTEX SHADER

// Input RenderProperties
<RenderPropertyTypeInfos>

// Vertex attribute inputs
<VertexProperties>

// Outputs

// flat uint64_t gRowId
// flat float gPointSize
// flat vec4 gFillColor
// flat vec4 gStrokeColor
// flat float gStrokeWidth
// flat uint gBarbId
// flat float gDirection // only if not uniform
// flat float gAnchorScale
out <GeometryShaderInputs>;

// Uniform RenderProperties
layout(std430) uniform <UniformProperties>;

// viewport data
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

// Color conversion stubs
vec4 transformfillColorToRGB(in vec4 c) { return c; }
vec4 transformstrokeColorToRGB(in vec4 c) { return c; }
vec4 unpackfillColor(in uint c) { return vec4(0); }
vec4 unpackstrokeColor(in uint c) { return vec4(0); }

// get* functions for properties
<PropertyGetters>

// project* functions for coord inputs
inTx projectx(in inTx x) {
  return x;
}

inTy projecty(in inTy y) {
  return y;
}

#define NUM_BARB_TYPES <numBarbTypes>
#define numid <numid>
#define useKey <useKey>

void processEmptyVertex() {
  gl_Position = vec4(-10, -10, -10, 0);
  // Use NUM_BARB_TYPES as a dead vertex sentinel
  gBarbId = NUM_BARB_TYPES;
}

// Grow primitive slightly to avoid multi-sampling issues at the edges
// This is compensated in the fragment shader which shrinks the circle diameter
// by 2 pixels
const float kMultiSamplingSizePad = 2.0;

void processVertex() {
  float speed = float(getspeed(speed));
  gBarbId = clamp(uint(round(speed / 5.0)), 0, NUM_BARB_TYPES - 1);

  float finalxPos = float(getx(projectx(x)));
  float finalyPos = float(gety(projecty(y)));

  float pointSize = float(getsize(size));
  if (pointSize < 0.1) {
    gBarbId = NUM_BARB_TYPES;
    return;
  }

#if IS_MULTISAMPLING == 1
  float pointSize += kMultiSamplingSizePad;
#endif
  gPointSize = pointSize;

  gl_Position = vec4(finalxPos, finalyPos, 0.0, 1.0);

#if useUdirection == 0
  gDirection = float(getdirection(direction));
  if (quantizeDirection == 1) {
    gDirection = float((int(gDirection) + 5) / 10) * 10.0;
  }
  gDirection = radians(gDirection);
#endif // useUdirection

  // limit anchor size so it doesn't cross billboard edge
  gAnchorScale = clamp(float(getanchorScale(anchorScale)), 0.0, 1.0) * 0.1;

  gFillColor = transformfillColorToRGB(getfillColor(fillColor));
  float out_opacity = float(getopacity(opacity));
  gFillColor.a *= out_opacity;

  gStrokeColor = transformstrokeColorToRGB(getstrokeColor(strokeColor));
  gStrokeColor.a *= out_opacity;

  gStrokeWidth = float(getstrokeWidth(strokeWidth));
}

void main() {
#if useKey == 1
  if (key != invalidKey) {
    processVertex();
  } else {
    processEmptyVertex();
  }
#else
  processVertex();
#endif  // useKey

  // ids from queries go from 0 to numrows-1, but since we're storing
  // the ids as unsigned ints, and there isn't a way to specify the
  // clear value for secondary buffers, we need to account for that
  // offset here
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
}
