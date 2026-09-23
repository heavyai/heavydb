/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Color conversion stubs
vec4 transformfillColorToRGB(in vec4 c) { return c; }
vec4 transformstrokeColorToRGB(in vec4 c) { return c; }
vec4 unpackfillColor(in uint c) { return vec4(0); }
vec4 unpackstrokeColor(in uint c) { return vec4(0); }

// number of rowid to process
#define numid <numid>

// viewport ubo
struct Viewport {
  int x;
  int y;
  int width;
  int height;
};

layout(std430, binding = 0) uniform SHARED_VIEWPORT_UBO {
  Viewport viewport;
};


// RenderProperty ubo
layout(std430) uniform <UniformProperties>;

// get* functions for properties
<PropertyGetters>

// project* functions for coord inputs
inTx projectx(in inTx x) {
  return x;
}

inTy projecty(in inTy y) {
  return y;
}


// Outputs
layout(location = 0) flat out uint64_t fRowId;     // the row id in the table
layout(location = 1) out vec4 fColor;         // the output color of the primitive

void main() {
	gl_Position =
			vec4(float(getx(projectx(x))) * uViewProjMatrix[0][0] + uViewProjMatrix[2][0],
						float(gety(projecty(y))) * uViewProjMatrix[1][1] + uViewProjMatrix[2][1],
						0.5,
						1.0);

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
