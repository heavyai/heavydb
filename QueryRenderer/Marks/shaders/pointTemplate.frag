/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// FRAGMENT SHADER

// Inputs

// flat uint64_t fRowId
// flat float fPointSize
// flat vec4 fColor
in <FragmentShaderInputs>;

#define IS_MULTISAMPLING <isMultiSampling>
#define DO_DEVICE_LOST_TEST_LOOP <doDeviceLostTestLoop>

#if DO_DEVICE_LOST_TEST_LOOP
layout(std430) uniform POINT_FRAG_UBO_TYPE {
  int uTestLoopEnd;
};
#endif


const vec2 halfvec = vec2(0.5);

void maybeDiscard() {
#if IS_MULTISAMPLING
  float pixsize = 1.0 / fPointSize;
  vec2 coord = gl_PointCoord + ((gl_SamplePosition - halfvec) * pixsize);
  // using (0.5 - pixsize) shrinks the circle diameter by 2 pixels to compensate for
  // the extra 2 pixel diameter pad to avoid multi-sampling issues
  if (distance(coord, halfvec) > (0.5 - pixsize)) {
    discard;
  }
#else
  if (distance(gl_PointCoord, halfvec) > 0.5) {
    discard;
  }
#endif
}

#if DO_DEVICE_LOST_TEST_LOOP
vec4 triggerDeviceLost() {
  vec4 rtnColor = fColor;
  int x = 1;
  float y = 1.0;
  // This loop should trigger device lost
  while (x > uTestLoopEnd) {
    y = float(x) * sqrt(y) + 4.0;
    x++;
    if (x++ > 65535) {
      x = 1;
    }
  }
  // Use the result to prevent dead code removal
  return rtnColor * y;
}
#endif

vec4 getFragmentColorOrDiscard() {
  maybeDiscard();

#if DO_DEVICE_LOST_TEST_LOOP
  return triggerDeviceLost();
#else
  return fColor;
#endif
}
