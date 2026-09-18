/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Marks/sdfUtils.glsl"
//</includes>
// FRAGMENT SHADER
// Inputs

// flat uint64_t fRowId
// vec2 fUV (geometry shader variant)
// flat uint fBarbId
// flat float fPointScale
// flat float fPointSize
// flat float fApproxMaxCoverage
// flat float fAnchorScale
// flat vec4 fFillColor
// flat vec4 fStrokeColor
// flat float fStrokeWidth
in <FragmentShaderInputs>;

#define NUM_BARB_TYPES <numBarbTypes>
#define IS_MULTISAMPLING <isMultiSampling>
#define SHOW_BILLBOARD <showBillboard>

#if IS_MULTISAMPLING == 1
#define DISCARD_SAFETY_PAD 1.0
#else
#define DISCARD_SAFETY_PAD 0.0
#endif

#define DO_SMOOTHING 0

layout(std430) uniform BARB_DEF_UBO_TYPE {
  uint uBarbCounts[NUM_BARB_TYPES];
  uint uBarbOffsets[NUM_BARB_TYPES];
  uint uPennantCounts[NUM_BARB_TYPES];
  uint uPennantOffsets[NUM_BARB_TYPES];
  vec2 uBarbVerts[<numBarbVerts>];
};

float pdot(in vec2 a, in vec2 b, in vec2 c) {
  return (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y);
}

bool is_outside_triangle(in vec2 p, in int offset) {
  // handles CW and CCW winding
  // TODO(scb): early outs are possible at each step if winding
  // order is non-variable
  bool b1 = (pdot(p, uBarbVerts[offset], uBarbVerts[offset+1]) < 0.0);
  bool b2 = (pdot(p, uBarbVerts[offset+1], uBarbVerts[offset+2]) < 0.0);
  bool b3 = (pdot(p, uBarbVerts[offset+2], uBarbVerts[offset]) < 0.0);

  return !((b1 == b2) && (b2 == b3));
}

vec4 getFragColor(in vec4 tempColor) {
  return tempColor;
}

void getEdgeDistanceInPixels(out float dpix, out bool isOutside) {
  vec2 p = fUV;

#if IS_MULTISAMPLING == 1
  float pixsize = 1.0 / fPointSize;
  p = p + ((gl_SamplePosition - vec2(0.5)) * pixsize);
#endif

  isOutside = true;
  p = p * fPointScale;

  dpix = 2.0;
  if (fBarbId == 0) { // Calm circles
    vec2 c1 = normalize(p) * 0.15;
    vec2 c2 = normalize(p) * 0.025;
    float d1 = distance(p, c1);
    float d2 = distance(p, c2);
    dpix = min(d1, d2) * fPointSize;
  } else { // Wind barbs
    float d_anchor = distance(p, uBarbVerts[0]);
    if (d_anchor < fAnchorScale) {
      dpix = 0.0;
    } else {
      float dist = 2.0;

      // Pennants first
      bool is_outside_pennant = true;
      int offset = int(uPennantOffsets[fBarbId]);
      int pennant_count = int(uPennantCounts[fBarbId]);
      for (int i = 0; i < pennant_count; i++, offset += 3) {
        if (!is_outside_triangle(p, offset)) {
          is_outside_pennant = false;
          // skip remaining pennants but get edge distances first
          i = pennant_count;
        }
        dist = min(dist, squared_dist_to_segment(p, uBarbVerts[offset], uBarbVerts[offset + 1]));
        dist = min(dist, squared_dist_to_segment(p, uBarbVerts[offset], uBarbVerts[offset + 2]));
        dist = min(dist, squared_dist_to_segment(p, uBarbVerts[offset + 1], uBarbVerts[offset + 2]));
      }

      // Barb segments including staff
      int vertStart = int(uBarbOffsets[fBarbId]);
      int vertEnd = vertStart + int(uBarbCounts[fBarbId] * 2);
      // Find the closest line segment
      for (int i = vertStart; i < vertEnd; i+=2) {
        vec2 a = uBarbVerts[i];
        vec2 b = uBarbVerts[i + 1];
        dist = min(dist, squared_dist_to_segment(p, a, b));
      }
      dpix = sqrt(max(0.0, dist)) * fPointSize;
      isOutside = is_outside_pennant;
    }
  }
}

#if SHOW_BILLBOARD == 1
const vec4 transparent = vec4(1,0,1,0.1);
#else
const vec4 transparent = vec4(0);
#endif

vec4 mapDistanceToColor(in float dpix, in bool isOutside) {
  // be sure we are completely out of the pixel before discarding else other samples may be discarded
  if (isOutside) {
#if SHOW_BILLBOARD == 0
    if (dpix > (fStrokeWidth + DISCARD_SAFETY_PAD)) {
      discard;
    }
#endif
    return (dpix > fStrokeWidth ? transparent : fStrokeColor) * fApproxMaxCoverage;
  }
  return (dpix > fStrokeWidth ? fFillColor : fStrokeColor) * fApproxMaxCoverage;
}

//
// Entry points for injected main() implementations
//
vec4 getFragmentColorOrDiscard() {
  float dpix;
  bool isOutside;
  getEdgeDistanceInPixels(dpix, isOutside);
  return mapDistanceToColor(dpix, isOutside);
}

// Barbs don't support accumulation
void maybeDiscard() {}
