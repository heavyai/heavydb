/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<extensions>
#extension GL_EXT_buffer_reference2 : require
//</extensions>
//<includes>
#include "Marks/typeDefines.glsl"
#include "Utils/colorConvertSubroutines.glsl"
#include "Marks/coordConvertSubroutines.glsl"
#include "Marks/slabAddressTable.glsl"
#include "Marks/fastSymbolDefines.glsl"
//</includes>
// VERTEX SHADER

// RenderProperty type info defines
<RenderPropertyTypeInfos>

// Vertex attribute inputs
<VertexProperties>

// Outputs

// flat uint64_t gRowId
// flat vec4 gFillColor
// flat vec4 gStrokeColor
// flat float gStrokeWidth
// flat uint gShapeType
// flat vec2 gPointSize
// flat float gAngle (only when not uniform)
// flat int gAccumIdx (accumulation render only)
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

// Set to 1 to handle accumulation rendering in a geometry shader
// The injected scale will add the necessary interface block output
// as "out int gAccumIdx"
#define supportsGeoAccum 1

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

inTx2 projectx2(in inTx2 x) {
  return x;
}

inTxc projectxc(in inTxc x) {
  return x;
}

inTy projecty(in inTy y) {
  return y;
}

inTy2 projecty2(in inTy2 y) {
  return y;
}

inTyc projectyc(in inTyc y) {
  return y;
}

#define numid <numid>
#define useKey <useKey>

#define computeX <computeX>
#define computeY <computeY>
#define computeWidth <computeWidth>
#define computeHeight <computeHeight>

void processEmptyVertex() {
  gl_Position = vec4(-10, -10, -10, 0);
  // Use NUM_SYMBOL_TYPES as a dead vertex sentinel
  gShapeType = NUM_SYMBOL_TYPES;
}

void processVertex() {
  gShapeType = uint(getshape(shape));

#if computeX == 0
#if inTxEnum == UNSIGNED_INT64_ARB
  float finalxPos = float(projectx(getx_ptr(x, propCompressionBits, PROP_COMPRESSION_BIT_X, 0)));
#else
  float finalxPos = float(getx(projectx(x)));
#endif
#elif computeX == 1
#if inTx2Enum == UNSIGNED_INT64_ARB
  float finalxPos = float(projectx2(getx_ptr(x2, propCompressionBits, PROP_COMPRESSION_BIT_X2, 0)));
#else
  float finalxPos = float(getx2(projectx2(x2)));
#endif
#else
#if inTxcEnum == UNSIGNED_INT64_ARB
  float finalxPos = float(projectxc(getx_ptr(xc, propCompressionBits, PROP_COMPRESSION_BIT_XC, 0)));
#else
  float finalxPos = float(getxc(projectxc(xc)));
#endif
#endif  // computeX
#if computeY == 0
#if inTyEnum == UNSIGNED_INT64_ARB
  float finalyPos = float(projecty(gety_ptr(y, propCompressionBits, PROP_COMPRESSION_BIT_Y, 0)));
#else
  float finalyPos = float(gety(projecty(y)));
#endif
#elif computeY == 1
#if inTy2Enum == UNSIGNED_INT64_ARB
  float finalyPos = float(projecty2(gety_ptr(y2, propCompressionBits, PROP_COMPRESSION_BIT_Y2, 0)));
#else
  float finalyPos = float(gety2(projecty2(y2)));
#endif
#else
#if inTycEnum == UNSIGNED_INT64_ARB
  float finalyPos = float(projectyc(gety_ptr(yc, propCompressionBits, PROP_COMPRESSION_BIT_YC, 0)));
#else
  float finalyPos = float(getyc(projectyc(yc)));
#endif
#endif  // computeY

#if computeWidth == 0
  gPointSize.x = float(getwidth(width));
#else
#if inTx2Enum == UNSIGNED_INT64_ARB
  gPointSize.x = float(projectx2(getx_ptr(x2, propCompressionBits, PROP_COMPRESSION_BIT_X2, 0))) - finalxPos;
#else
  gPointSize.x = float(getx2(projectx2(x2))) - finalxPos;
#endif
#endif  // computeWidth
#if computeHeight == 0
  gPointSize.y = float(getheight(height));
#else
#if inTy2Enum == UNSIGNED_INT64_ARB
  gPointSize.y = float(projecty2(gety_ptr(y2, propCompressionBits, PROP_COMPRESSION_BIT_Y2, 0))) - finalyPos;
#else
  gPointSize.y = float(gety2(projecty2(y2))) - finalyPos;
#endif
#endif  // computeHeight

  // Skip adding the pad and return if the point is too small to render
  // The geometry shader will check the point size and bail
  if (max(gPointSize.x, gPointSize.y) < 0.1) {
    return;
  }

  gl_Position = vec4(finalxPos, finalyPos, 0.0, 1.0);

#if useUangle == 0
  gAngle = float(getangle(angle));
  if (uint(getangleUnit(angleUnit)) == 1) {
    gAngle = radians(gAngle);
  }
#endif // useUangle

  gFillColor = transformfillColorToRGB(getfillColor(fillColor));
  gFillColor.a *= float(getopacity(opacity)) * float(getfillOpacity(fillOpacity));

  gStrokeWidth = float(getstrokeWidth(strokeWidth));

  gStrokeColor = transformstrokeColorToRGB(getstrokeColor(strokeColor));
  gStrokeColor.a *= float(getopacity(opacity)) * float(getstrokeOpacity(strokeOpacity));
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
