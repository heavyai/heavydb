/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) out vec4 color;

layout(location = 0) in float fPointSize;

layout(std430) uniform POINTS_SHADER_UBO_TYPE {
  vec2 imageSize;
};

const vec2 center = vec2(0.5, 0.5);

const vec2 pseudo_samples[4] = {vec2(0.05, 0.5), vec2(0.95, 0.5), vec2(0.5, 0.05), vec2(0.5,0.95)};

#if 1
// programmatic constants in case we want to generate several different
// tests for this shader
#define USE_DIFFERENTIALS <useDifferentials>
#define DO_SAMPLE_OFFSET <doSampleOffset>
#define USE_PSEUDO_SAMPLE_POSITIONS <usePseudoSamplePositions>
#define DO_MAGNIFY_SAMPLE_POSITIONS <doMagnifySamplePositions>
#define DO_DISCARD <doDiscard>
#define DO_SMOOTHSTEP <doSmoothstep>
#define DO_COLORS <doColors>
#else // Fixed shader constants for quick experiments (saves recompilation)
#define USE_DIFFERENTIALS            0
#define DO_SAMPLE_OFFSET             0
#define USE_PSEUDO_SAMPLE_POSITIONS  0
#define DO_MAGNIFY_SAMPLE_POSITIONS  0
#define DO_DISCARD                   1
#define DO_SMOOTHSTEP                1
#define DO_COLORS                    1
#endif

void main() {
  // If magnifying positions we need to shrink the circle so there's enough
  // room in the point primitive to see the individual sample circles
  float cutoff_distance = (DO_MAGNIFY_SAMPLE_POSITIONS==1) ? 0.2 : 0.5;

#if USE_DIFFERENTIALS == 1
  // Average partials so other math matches the non-differential method
  float duv = (dFdx(gl_PointCoord.x) + dFdy(gl_PointCoord.y)) * 0.5;
#else
  float duv = 1.0 / fPointSize;
#endif

#if DO_SAMPLE_OFFSET == 1

  // Use faux sample positions in order to isolate behavior of gl_SamplePosition
  // and allow testing of gl_SampleID
#if USE_PSEUDO_SAMPLE_POSITIONS == 1
  vec2 sample_pos = pseudo_samples[gl_SampleID];
#else
  vec2 sample_pos = gl_SamplePosition;
#endif
  vec2 sample_offset = vec2(sample_pos - vec2(0.5)) * duv;
#if DO_MAGNIFY_SAMPLE_POSITIONS == 1
  sample_offset *= 10.0; // magnify sample positions
#endif

#else // DO_SAMPLE_OFFSET
  vec2 sample_offset = vec2(0.0);
#endif // DO_SAMPLE_OFFSET

  // Compute distance from the center of the circle. Discard has side effects
  // and interacts with dF and flow control so it can be useful to skip it for
  // determinism testing
  float d = distance((gl_PointCoord) + sample_offset, center);
  if (d > cutoff_distance) {
#if DO_DISCARD == 1
    discard;
#else
    // Add a little color to the rest of the point so we can see the edges
    color = vec4(0.2,0.0,0.2,0.2);
#endif
  } else {

#if DO_SMOOTHSTEP == 1
  float a = 1.0 - smoothstep(cutoff_distance - duv, 0.5, d);
#else
  float a = 1.0;
#endif

#if DO_COLORS
  // Output the gl_PointCoord color to allow testing for issues with it and
  // to make things a little more interesting.
  color = vec4(gl_PointCoord * a, 0.0, a);
#else
  // Just output white (easier to see samples etc)
  color = vec4(a);
#endif
  }
}
