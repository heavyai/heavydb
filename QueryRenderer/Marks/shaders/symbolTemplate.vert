/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<extensions>
#extension GL_EXT_buffer_reference2 : require
//</extensions>
//<includes>
#include "Utils/colorConvertSubroutines.glsl"
#include "Marks/coordConvertSubroutines.glsl"
#include "Marks/typeDefines.glsl"
#include "Marks/slabAddressTable.glsl"
//</includes>
// VERTEX SHADER

// Input RenderProperties
<RenderPropertyTypeInfos>

// Vertex attribute inputs
<VertexProperties>

in float shapeposx;
in float shapeposy;
in float shapeu;
in float shapev;

// Uniform RenderProperties
layout(std430) uniform <UniformProperties>;

// viewport data
struct Viewport {
  int x;
  int y;
  int width;
  int height;
};

// get* functions for properties
<PropertyGetters>

layout(push_constant) uniform PUSH_CONSTANTS {
  uint32_t currShapeType;
} pushConstants;

layout(std430, binding = 0) uniform SHARED_VIEWPORT_UBO {
  Viewport viewport;
};

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
#define isFillPass <isFillPass>
#define isCircle <isCircle>

// Color conversion stubs
#if isFillPass == 1
vec4 transformfillColorToRGB(in vec4 c) {
  return c;
}
vec4 unpackfillColor(in uint c) {
  return vec4(0);
}
#else
vec4 transformstrokeColorToRGB(in vec4 c) {
  return c;
}
vec4 unpackstrokeColor(in uint c) {
  return vec4(0);
}
#endif

#if isFillPass == 0
vec2 NDCtoScreen(in vec2 p) {
  return vec2((p.x + 1) * (float(viewport.width) * 0.5) + viewport.x,
              (p.y + 1) * (float(viewport.height) * 0.5) + viewport.y);
}
#endif

////////////////////////////////////////////////////////////////
/**
 * Non-interpolated shader outputs.
 */
#if isFillPass == 1 || isCircle == 1
// flat uint64_t fRowId
// flat int accumIdx (accumulation renders only)
// flat vec4 fColor
// if isCircle
//   flat float fWidth
//   flat float fHeight
//   vec2 fouterUVCoord
//   if !isFillPass
//     vec2 finnerUVCoord
//     flat float fStrokeWidth
out <FragmentShaderInputs>;

#else // isFillPass || isCircle

// outputs for lineTemplate.geom
out <GeometryShaderInputs>;
#endif  // isFillPass || isCircle

#define computeX <computeX>
#define computeY <computeY>
#define computeWidth <computeWidth>
#define computeHeight <computeHeight>
void processEmptyVertex() {
  gl_Position = vec4(-10, -10, -10, 0);
#if isFillPass == 1 || isCircle == 1
  fColor = vec4(0, 0, 0, 0);
#if isCircle == 1
  fWidth = 0;
  fHeight = 0;
  fouterUVCoord = vec2(0);
#if isFillPass == 0
  finnerUVCoord = vec2(0);
  fStrokeWidth = 0;
#endif
#endif
#else
  gOpacity = 0.0;
  gColor = vec4(0, 0, 0, 0);
  gStrokeWidth = 0;
  gStrokeOpacity = 0;
  gLineJoin = 0;
  gMiterLimit = 0;
#endif  // isFillPass == 1
}

void processVertex() {
  uint shapeType = uint(getshape(shape));
  if (shapeType != pushConstants.currShapeType) {
    processEmptyVertex();
  } else {
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
    float widthToUse = float(getwidth(width));
#else
#if inTx2Enum == UNSIGNED_INT64_ARB
    float widthToUse = float(projectx2(getx_ptr(x2, propCompressionBits, PROP_COMPRESSION_BIT_X2, 0))) - finalxPos;
#else
    float widthToUse = float(getx2(projectx2(x2))) - finalxPos;
#endif
#endif  // computeWidth
#if computeHeight == 0
    float heightToUse = float(getheight(height));
#else
#if inTy2Enum == UNSIGNED_INT64_ARB
    float heightToUse = float(projecty2(gety_ptr(y2, propCompressionBits, PROP_COMPRESSION_BIT_Y2, 0))) - finalyPos;
#else
    float heightToUse = float(gety2(projecty2(y2))) - finalyPos;
#endif
#endif  // computeHeight

#if isFillPass == 0
    float finalStrokeWidth = float(getstrokeWidth(strokeWidth));
#if isCircle == 1
    // bulge the shape's dimensions to account for the circle's stroke
    // The circle is stroked in the fragment shader, unlike poly geom
    widthToUse += sign(widthToUse) * finalStrokeWidth;
    heightToUse += sign(heightToUse) * finalStrokeWidth;
#if computeX != 2
    finalxPos -= finalStrokeWidth / 2.0;
#endif
#if computeY != 2
    finalyPos -= finalStrokeWidth / 2.0;
#endif
    float strokewRatio = finalStrokeWidth / widthToUse;
    float strokewDenom = 1.0 - 2 * strokewRatio;
    float strokehRatio = finalStrokeWidth / heightToUse;
    float strokehDenom = 1.0 - 2 * strokehRatio;
    if (strokewDenom <= 0 || strokehDenom <= 0) {
      finnerUVCoord = vec2(0.0);
    } else {
      finnerUVCoord.x = (shapeu - strokewRatio) / strokewDenom;
      finnerUVCoord.y = (shapev - strokehRatio) / strokehDenom;
    }
#endif  // isCircle == 1
#endif  // isFillPass == 0
    vec2 vpScale = vec2(uViewProjMatrix[0][0], uViewProjMatrix[1][1]);
    vec2 vpTranslate = vec2(uViewProjMatrix[0][0] * finalxPos + uViewProjMatrix[2][0],
                            uViewProjMatrix[1][1] * finalyPos + uViewProjMatrix[2][1]);
#if useUangle == 0
    float theta = float(getangle(angle));
    if (uint(getangleUnit(angleUnit)) == 1) {
      theta = radians(theta);
    }
    float sinAngle = sin(-theta);
    float cosAngle = cos(-theta);
    mat2 modelViewTM = mat2(cosAngle * vpScale.x, -sinAngle * vpScale.x,
                            sinAngle * vpScale.y, cosAngle * vpScale.y);
#else  // useUangle
    mat2 modelViewTM = mat2(uCosAngle * vpScale.x, -uSinAngle * vpScale.x,
                            uSinAngle * vpScale.y, uCosAngle * vpScale.y);
#endif
    vec2 shape_pt = vec2((shapeposx + uPivotx) * widthToUse,
                         (shapeposy + uPivoty) * heightToUse) * modelViewTM + vpTranslate;

#if isFillPass == 1 || isCircle == 1
    gl_Position = vec4(
        shape_pt,
        clamp(float(gl_InstanceIndex + 1) / float(totalNumInstances), 0.0, 1.0),
        1.0);

#if isCircle == 1
    fWidth = widthToUse;
    fHeight = heightToUse;
    fouterUVCoord = vec2(shapeu, shapev);
#if isFillPass == 1
    fColor = transformfillColorToRGB(getfillColor(fillColor));
    fColor.a *= float(getopacity(opacity)) * float(getfillOpacity(fillOpacity));
#else
    fColor = transformstrokeColorToRGB(getstrokeColor(strokeColor));
    fColor.a *= float(getopacity(opacity)) * float(getstrokeOpacity(strokeOpacity));
    fStrokeWidth = finalStrokeWidth;
#endif
#else
    fColor = transformfillColorToRGB(getfillColor(fillColor));
    fColor.a *= float(getopacity(opacity)) * float(getfillOpacity(fillOpacity));
#endif
#else
    // start stroke output data
    gl_Position = vec4(
        NDCtoScreen(shape_pt),
        clamp(float(gl_InstanceIndex + 1) / float(totalNumInstances), 0.0, 1.0),
        1.0);

    gOpacity = float(getopacity(opacity));
    gColor = transformstrokeColorToRGB(getstrokeColor(strokeColor));

    gStrokeWidth = finalStrokeWidth;
    gStrokeOpacity = float(getstrokeOpacity(strokeOpacity));
    gLineJoin = lineJoin;
    gMiterLimit = float(getmiterLimit(miterLimit));
    // end stroke output data
#endif  // isFillPass
  }
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
  uint64_t outId = uint64_t(id + 1);
#if numid >= 2
  int total_bitshift = id_offsets[0];
  outId = (uint64_t(id1 + 1) << total_bitshift) | outId;
#endif
#if numid >= 3
  total_bitshift += id_offsets[1];
  outId = (uint64_t(id2 + 1) << total_bitshift) | outId;
#endif
#if isFillPass == 1 || isCircle == 1
  fRowId = outId;
#else
  gRowId = outId;
#endif
#endif
}
