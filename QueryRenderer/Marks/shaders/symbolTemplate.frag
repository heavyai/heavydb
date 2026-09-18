/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// FRAGMENT SHADER

#define isFillPass <isFillPass>
#define isCircle <isCircle>

// Inputs

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
#if isCircle == 1
const vec2 halfvec = vec2(0.5);
#endif

in <FragmentShaderInputs>;

float getAlphaOrDiscard() {
  float alpha = 1.0;
#if isCircle == 1
#if isFillPass == 1
  float minDim = min(fWidth, fHeight);
  if (minDim <= 1.0) {
    alpha = 0.5 * minDim;
  } else if (minDim <= 2.0) {
    alpha = 0.75 - (minDim - 1.0) * 0.25;
  } else {
    vec2 pixsize = vec2(1.0 / fWidth, 1.0 / fHeight);
    vec2 coord = fouterUVCoord + ((gl_SamplePosition - halfvec) * pixsize);
    float dist = distance(coord, halfvec);
    if (dist > 0.5) {
      discard;
    }
  }
#else
  vec2 pixsize = vec2(1.0 / fWidth, 1.0 / fHeight);
  vec2 offset = (gl_SamplePosition - halfvec) * pixsize;
  float dist = distance(fouterUVCoord + offset, halfvec);
  if (dist > 0.5) {
    discard;
  }
  dist = distance(finnerUVCoord + offset, halfvec);
  if (dist < 0.5) {
    discard;
  }
#endif
#endif
  return alpha;
}

void maybeDiscard() {
  getAlphaOrDiscard();
}

vec4 getFragmentColorOrDiscard() {
  return vec4(fColor.rgb, getAlphaOrDiscard());
}
