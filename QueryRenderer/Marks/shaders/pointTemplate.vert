/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//</includes>
// VERTEX SHADER

// RenderProperty type info defines
<RenderPropertyTypeInfos>

// Vertex attribute inputs
<VertexProperties>

// Outputs

// flat uint64_t fRowId
// flat float fPointSize
// flat vec4 fColor
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

// Color conversion declaration stubs
vec4 transformfillColorToRGB(in vec4 c) { return c; }
vec4 unpackfillColor(in uint c) { return vec4(0); }

// get* functions for properties
<PropertyGetters>

// project* functions for coord inputs
inTx projectx(in inTx x) {
  return x;
}

inTy projecty(in inTy y) {
  return y;
}

// General #defines
#define numid <numid>
#define useKey <useKey>
#define IS_MULTISAMPLING <isMultiSampling>

////////////////////////////////////////////////////////////////
// Grow primitive slightly to avoid multi-sampling issues at the edges
// This is compensated in the fragment shader which shrinks the circle diameter
// by 2 pixels
const float kMultiSamplingSizePad = 2.0;

void main() {
#if useKey == 1
  if (key == invalidKey) {
    gl_Position = vec4(-10, -10, -10, 0);
    fPointSize = 0.0;
    gl_PointSize = 0.0;
    fColor = vec4(0, 0, 0, 0);
    fRowId = 0;
  } else
#endif // useKey == 1
  {
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
    gl_Position =
        vec4(px * uViewProjMatrix[0][0] + uViewProjMatrix[2][0],
             py * uViewProjMatrix[1][1] + uViewProjMatrix[2][1],
             0.5,
             1.0);
#if IS_MULTISAMPLING == 1
    float sz = float(getsize(size)) + kMultiSamplingSizePad;
#else
    float sz = float(getsize(size));
#endif
    fPointSize = sz;
    gl_PointSize = sz;

    fColor = transformfillColorToRGB(getfillColor(fillColor));
    fColor.a *= float(getopacity(opacity)) * float(getfillOpacity(fillOpacity));

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
}
