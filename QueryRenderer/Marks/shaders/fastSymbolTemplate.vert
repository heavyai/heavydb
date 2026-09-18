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

// flat uint64_t fRowId
// flat uint fShapeType
// flat vec4 fSymbolScale
// flat float fPointSize
// flat vec4 fFillColor
// flat vec4 fStrokeColor
// flat float fStrokeWidth
// flat float fApproxMaxCoverage
// flat int accumIdx (accumulation render only)
out <FragmentShaderInputs>;

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

// General #defines
#define numid <numid>
#define useKey <useKey>
#define doHeatmapEdgePad <doHeatmapEdgePad>

#define computeX <computeX>
#define computeY <computeY>
#define computeWidth <computeWidth>
#define computeHeight <computeHeight>

void processEmptyVertex() {
  gl_Position = vec4(-10, -10, -10, 0);
  fShapeType = 0;
  fPointSize = 0.0;
}

void processVertex() {
  fShapeType = uint(getshape(shape));

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

  vec2 size;
#if computeWidth == 0
  size.x = float(getwidth(width));
#else
#if inTx2Enum == UNSIGNED_INT64_ARB
  size.x = float(projectx2(getx_ptr(x2, propCompressionBits, PROP_COMPRESSION_BIT_X2, 0))) - finalxPos;
#else
  size.x = float(getx2(projectx2(x2))) - finalxPos;
#endif
#endif  // computeWidth
#if computeHeight == 0
  size.y = float(getheight(height));
#else
#if inTy2Enum == UNSIGNED_INT64_ARB
  size.y = float(projecty2(gety_ptr(y2, propCompressionBits, PROP_COMPRESSION_BIT_Y2, 0))) - finalyPos;
#else
  size.y = float(gety2(projecty2(y2))) - finalyPos;
#endif
#endif  // computeHeight

  // Cull if too small. This will catch NULLS
  if (max(size.x, size.y) < CULL_SIZE) {
    processEmptyVertex();
    return;
  }

  // Grow the point slightly in order to guarantee there are no subpixel gaps between the
  // points when rendering binned heatmaps
#if doHeatmapEdgePad == 1
  size += 0.1;
#endif // doHeatmapEdgePad

  float maxSize = max(size.x, size.y);
  float useStrokeWidth = float(getstrokeWidth(strokeWidth));

  // determine the approximate maximum pixel coverage for the symbol. This is needed
  // for sizes < 1.0
  // TODO: incorporate stroke width?
  fApproxMaxCoverage = min(1.0, size.x) * min(1.0, size.y);

  // always round up otherwise we'll lose strokes that are < 0.5 pixels
  // on shapes that abut the edge of the quad
  fPointSize = maxSize + ceil(useStrokeWidth);

  // TODO: uPivotx/y can be premultiplied
  float pivot_y = fShapeType == WEDGE ? uPivoty + 0.3333 : uPivoty;
  gl_Position = vec4(uPivotx * uVPmatrix[0][0] * size.x + uVPmatrix[0][0] * finalxPos + uVPmatrix[2][0],
                     pivot_y * uVPmatrix[1][1] * size.y + uVPmatrix[1][1] * finalyPos + uVPmatrix[2][1],
                     0.0, 1.0);

  float point_size_padded = fPointSize + NUM_PAD_PIXELS;
  if (fShapeType == CIRCLE) {
    fSymbolScale = vec4(point_size_padded / size.x, point_size_padded / size.y,
                        size.x / point_size_padded, size.y / point_size_padded);
  } else {
    float d = 1.0 / point_size_padded;
    fSymbolScale = vec4(d * size.x, d * size.y, 0.0, 0.0);
  }
  gl_PointSize = point_size_padded;

  fStrokeWidth = useStrokeWidth;
  fFillColor = transformfillColorToRGB(getfillColor(fillColor));
  fFillColor.a *= float(getopacity(opacity)) * float(getfillOpacity(fillOpacity));

  fStrokeColor = transformstrokeColorToRGB(getstrokeColor(strokeColor));
  fStrokeColor.a *= float(getopacity(opacity)) * float(getstrokeOpacity(strokeOpacity));
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
  fRowId = uint64_t(id + 1);
#endif
#if numid >= 2
  int total_bitshift = id_offsets[0];
  fRowId = (uint64_t(id1 + 1) << total_bitshift) | fRowId;
#endif
#if numid >= 3
  total_bitshift += id_offsets[1];
  fRowId = (uint64_t(id2 + 1) << total_bitshift) | fRowId;
#endif
}
