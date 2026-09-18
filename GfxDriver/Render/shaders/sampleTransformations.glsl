/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// sampleTransformations.glsl
//
// Function to warp uniform 2d sample positions to other distributions
//
// WARNING: These function can be costly, they are intended for experimental purposes
// For anything performance sensitive sequences should be precomputed on the host

//<includes>
#include "Shading/rtUtils.glsl"
//</includes>

#ifndef F_PI_4
#define F_PI_4 0.7853981634
#endif

vec2 rotate_sample(float r, vec2 s) {
  float angle = r * 6.283185307;
  float sa = sin(angle);
  float ca = cos(angle);
  return vec2(s.x * ca + s.y * sa, -s.x * sa + s.y * ca);
}

vec3 align_sample_to_vector(vec3 v, vec2 s) {
  vec3 dx, dy;
  compute_basis_vectors(v, dx, dy);
  return normalize(v + (s.x * dx) + (s.y * dy));
}

//
// Disc (concentric mapping)
//
vec2 map_disc_concentric(vec2 p) {
  float r1 = p.x * 2.0f - 1.0f;
  float r2 = p.y * 2.0f - 1.0f;
  float r, q;
  if ((r1 >= -r2) && (r1 > r2)) {
    if (r1 == 0.0f)
      r = 0.0f;
    else {
      q = F_PI_4 * r2 / r1;
      r = r1;
    }
  }
  if ((r1 <= r2) && (r1 > -r2)) {
    if (r2 == 0.0f)
      r = 0.0f;
    else {
      q = F_PI_4 * (2 - r1 / r2);
      r = r2;
    }
  }
  if ((r1 <= -r2) && (r1 <= r2)) {
    if (r1 == 0.0f)
      r = 0.0f;
    else {
      q = F_PI_4 * (4 + r2 / r1);
      r = -r1;
    }
  }
  if ((r1 >= r2) && (r1 <= -r2)) {
    if (r2 == 0.0f)
      r = 0.0f;
    else {
      q = F_PI_4 * (6 - r1 / r2);
      r = -r2;
    }
  }

  return vec2(cos(q) * r, sin(q) * r);
}
