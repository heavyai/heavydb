/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Marks/fastSymbolDefines.glsl"
#include "Marks/sdfUtils.glsl"
//</includes>
// FRAGMENT SHADER
// Inputs

// flat uint64_t fRowId
// flat uint fShapeType
// flat vec4 fSymbolScale
// flat in float fPointSize
// flat vec4 fFillColor
// flat vec4 fStrokeColor
// flat float fStrokeWidth
// flat float fApproxMaxCoverage
// vec2 fUV (geometry shader mode only)
// flat int accumIdx (accumulation render only)
in <FragmentShaderInputs>;

#define USING_GEOM_OR_MESH_SHADER <usingGeomOrMeshShader>
#define IS_MULTISAMPLING <isMultiSampling>

#if IS_MULTISAMPLING == 1
#define DISCARD_SAFETY_PAD 1.0
#else
#define DISCARD_SAFETY_PAD 0.0
#endif

#define DO_SMOOTHING 0
#define USE_CROSSINGS_TEST 0 // Use crossings instead of winding number

layout(std430) uniform SYMBOL_DEF_UBO_TYPE {
  uint uSymbolFlags[NUM_SYMBOL_TYPES];
  uint uVertCounts[NUM_SYMBOL_TYPES];
  uint uVertOffsets[NUM_SYMBOL_TYPES];
  vec2 uSymbolVerts[<numSymbolVerts>];
};

// Stub for color mapping function (dynamically bound)
vec4 mapDistanceToColor(in float, in bool) {return vec4(0);}

float pdot(in vec2 a, in vec2 b, in vec2 c) {
  return (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y);
}

bool is_outside_triangle(in vec2 p, in int offset) {
  // handles CW and CCW winding
  // TODO(scb): early outs are possible at each step if winding
  // order is non-variable
  bool b1 = (pdot(p, uSymbolVerts[offset] * fSymbolScale.xy, uSymbolVerts[offset+1] * fSymbolScale.xy) < 0.0);
  bool b2 = (pdot(p, uSymbolVerts[offset+1] * fSymbolScale.xy, uSymbolVerts[offset+2] * fSymbolScale.xy) < 0.0);
  bool b3 = (pdot(p, uSymbolVerts[offset+2] * fSymbolScale.xy, uSymbolVerts[offset] * fSymbolScale.xy) < 0.0);

  return !((b1 == b2) && (b2 == b3));
}

#if USE_CROSSINGS_TEST == 0
// Get the winding number for p
// returns 0 if point is outside shape
// squared distance is returned in dist_squared
float get_winding_number(in vec2 p, in int first_vert, in int last_vert, out float dist_squared) {
  float wn = 0.0;
  dist_squared = 2.0; // max computed distance is ~1.0 so this guarantees min is less
  vec2 b = uSymbolVerts[first_vert] * fSymbolScale.xy;
  for(int i=first_vert; i<last_vert; i++) {
    vec2 a = b;
    b = uSymbolVerts[i+1] * fSymbolScale.xy;
    if (a.y <= p.y) {
      if (b.y > p.y) {
        // if (is_left_of_segment(p, a, b) > 0.0) { wn += 1.0; }
        wn += max(sign(is_left_of_segment(p, a, b)), 0.0);
      }
    }
    else {
      if (b.y <= p.y) {
        // if (is_left_of_segment(p, a, b) < 0.0) { wn -= 1.0; }
        wn += min(sign(is_left_of_segment(p, a, b)), 0.0);
      }
    }
    // find the minimum distance to the shape
    dist_squared = min(dist_squared, squared_dist_to_segment(p, a, b));
  }
  return wn;
}

#else // USE_CROSSINGS_TEST
// Get the crossing number for p
// point is outside if the return is an even number
// squared distance is returned in dist_squared
float get_crossing_number(in vec2 p, in int first_vert, in int last_vert, out float dist_squared)
{
  float cn = 0.0;
  dist_squared = 2.0; // max computed distance is ~1.0 so this guarantees min is less
  vec2 b = uSymbolVerts[first_vert] * fSymbolScale.xy;
  for(int i=first_vert; i<last_vert; i++) {
    vec2 a = b;
    b = uSymbolVerts[i+1] * fSymbolScale.xy;
    // check if we're within the line segment in y
    if (((a.y <= p.y) && (b.y > p.y)) || ((a.y > p.y) && (b.y <= p.y))) {
      // compute x intersection
      float vt = (p.y - a.y) / (b.y - a.y);
      if (p.x < a.x + vt * (b.x - a.x)) {
        cn += 1.0;
      }
    }
    // find the minimum distance to the shape
    dist_squared = min(dist_squared, squared_dist_to_segment(p, a, b));
  }
  return cn;
}
#endif // USE_CROSSINGS_TEST

vec4 getFragColor(in vec4 tempColor) {
  return tempColor;
}

void getEdgeDistanceInPixels(out float dpix, out bool isOutside) {
  uint symbolFlags = uSymbolFlags[fShapeType];

#if USING_GEOM_OR_MESH_SHADER == 1
  vec2 p = fUV;
#else
  vec2 p = gl_PointCoord;
#endif

#if IS_MULTISAMPLING == 1
  float pixsize = 1.0 / fPointSize;
  p = (p * 2.0 - vec2(1.0)) + ((gl_SamplePosition - vec2(0.5)) * pixsize);
#else
  p = p * 2.0 - vec2(1.0);
#endif

  isOutside = true;

  // TODO(scb): This is still too branchy. We really have 3 classes
  // of shapes: circle, symmetric line segments with no acute angles, and
  // triangles. We should be able to switch on that class and fold the
  // inside/outside determination into the distance calculation. For now
  // performance is sufficient but this is worth exploring in the future
  dpix = 2.0;
  if (fShapeType == CIRCLE) {
    float d = length(p * fSymbolScale.xy);
    if (d <= 1.0) {
      isOutside = false;
    }

    vec2 strokeCenter;
    if (abs(fSymbolScale.z - fSymbolScale.w) > 0.01) {
      strokeCenter = ellipse_point(p, fSymbolScale.z, fSymbolScale.w);
    } else {
      strokeCenter = normalize(p) * fSymbolScale.zw;
    }
    dpix = distance(p, strokeCenter) * fPointSize;
  } else {
    int segmentCount = int(uVertCounts[fShapeType]);
    int vertStart = int(uVertOffsets[fShapeType]);
    int vertEnd = vertStart + segmentCount;

    float dist = 2.0;
    if ((symbolFlags & FLAG_XY_SYMMETRY) == FLAG_XY_SYMMETRY) {
      p = abs(p);
    }
    else if ((symbolFlags & FLAG_X_SYMMETRY) == FLAG_X_SYMMETRY) {
      p.x = abs(p.x);
    }

    if ((symbolFlags & FLAG_USE_WINDING_NUMBER_TEST) == FLAG_USE_WINDING_NUMBER_TEST) {
#if USE_CROSSINGS_TEST == 0
      float wn = get_winding_number(p, vertStart, vertEnd, dist);
      if (abs(wn) > 0.1) {
        isOutside = false;
      }
#else // USE_CROSSINGS_TEST
      float cn = get_crossing_number(p, vertStart, vertEnd, dist);
      if (mod(cn, 2.0) > 0.5) {
        isOutside = false;
      }
#endif // USE_CROSSINGS_TEST
    } else {
      // Find the closest line segment, and check which side we're on to determine inside / outside
      // regardless of whether the closest point on the line is interior or an end point
      vec2 b = uSymbolVerts[vertStart] * fSymbolScale.xy;
      int closestSegment = 0;
      for (int i = vertStart; i < vertEnd; i++) {
        vec2 a = b;
        b = uSymbolVerts[i + 1] * fSymbolScale.xy;
        float segDist = squared_dist_to_segment(p, a, b);
        if (segDist < dist) {
          dist = segDist;
          closestSegment = i;
        }
      }
      if ((symbolFlags & FLAG_TRIANGLE) == FLAG_TRIANGLE) {
        isOutside = is_outside_triangle(p, vertStart);
      } else {
        isOutside = sign(is_left_of_segment(p,
                                            uSymbolVerts[closestSegment] * fSymbolScale.xy,
                                            uSymbolVerts[closestSegment + 1] * fSymbolScale.xy)) > 0.0 ? true : false;
      }
    }
    dpix = sqrt(max(0.0, dist)) * fPointSize;
  }
}

//
// Distance -> Color maps
//

// Fill and Stroke enabled
const vec4 transparent = vec4(0);
vec4 mapDistanceToColorFillAndStroke(in float dpix, in bool isOutside) {
#if DO_SMOOTHING
  if (isOutside && (dpix > fStrokeWidth + DISCARD_SAFETY_PAD)) {
    discard;
  }
  float blend = (1.0 - smoothstep(fStrokeWidth - 0.5, fStrokeWidth + 0.5, dpix)) * min(fStrokeWidth, 1.0);
  return mix(isOutside ? transparent:fFillColor, fStrokeColor, blend) * fApproxMaxCoverage;
#else
    // be sure we are completely out of the pixel before discarding else other samples may be discarded
  if (isOutside) {
    if (dpix > (fStrokeWidth + DISCARD_SAFETY_PAD)) {
      discard;
    }
    return (dpix > fStrokeWidth ? transparent : fStrokeColor) * fApproxMaxCoverage;
  }
  return (dpix > fStrokeWidth ? fFillColor : fStrokeColor) * fApproxMaxCoverage;
#endif
}

// Fill only
vec4 mapDistanceToColorFill(in float dpix, in bool isOutside) {
#if DO_SMOOTHING
  if (isOutside) {
    if (dpix > DISCARD_SAFETY_PAD) {
      discard;
    }
    dpix = -dpix;
  }
  dpix += 0.5;
  return fFillColor * smoothstep(0.0, 1.0, dpix) * fApproxMaxCoverage;
#else
  if (isOutside) {
    // be sure we are completely out of the pixel before discarding else other samples may be discarded
    if (dpix > DISCARD_SAFETY_PAD) {
      discard;
    }
    return transparent;
  }
  return fFillColor * fApproxMaxCoverage;
#endif
}

// Stroke only
vec4 mapDistanceToColorStroke(in float dpix, in bool isOutside) {
#if DO_SMOOTHING
  if (dpix > fStrokeWidth + DISCARD_SAFETY_PAD) {
    discard;
  }
 float blend = (1.0 - smoothstep(fStrokeWidth - 0.5, fStrokeWidth + 0.5, dpix)) * min(fStrokeWidth, 1.0);
 return fStrokeColor * blend;
#else
    // be sure we are completely out of the pixel before discarding else other samples may be discarded
  if (dpix > (fStrokeWidth + DISCARD_SAFETY_PAD)) {
    discard;
  }
  if (dpix < fStrokeWidth) {
    return fStrokeColor;
  }
  return transparent;
#endif
}

//
// maybeDiscard mappers
//

// Stub for maybe discard function (dynamically bound)
void maybeDiscardFunc(in float dpix, in bool isOutside) {}

void maybeDiscardFillAndStroke(in float dpix, in bool isOutside) {
  if (isOutside && (dpix > fStrokeWidth)) {
    discard;
  }
}

// Fill only
void maybeDiscardFill(in float dpix, in bool isOutside) {
  if (isOutside) {
    discard;
  }
}

// Stroke only
void maybeDiscardStroke(in float dpix, in bool isOutside) {
  if (dpix > fStrokeWidth) {
    discard;
  }
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

void maybeDiscard() {
  float dpix;
  bool isOutside;
  getEdgeDistanceInPixels(dpix, isOutside);
  maybeDiscardFunc(dpix, isOutside);
}
