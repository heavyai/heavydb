/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// rtUtils.glsl
//
// Utility functions for use with raytracing applications

//
// compute_basis_vectors
//

// Compute basis vectors given vector v
// outputs vectors x and y, which are perpendicular to v and each other
void compute_basis_vectors(const vec3 v, out vec3 x, out vec3 y) {
  const float yz = -v.y * v.z;
  y = normalize(((abs(v.z) > 0.99999f) ? vec3(-v.x * v.y, 1.0f - v.y * v.y, yz)
                                       : vec3(-v.x * v.z, yz, 1.0f - v.z * v.z)));
  x = cross(y, v);
}

//
// offset_ray_origin
//

// Offset a ray origin along normal to avoid self-intersection
// Use geometric normal (not interpolated)
// See Ray Tracing Gems Ch. 6
vec3 offset_ray_origin(vec3 p, vec3 n) {
  const float origin = 1.0 / 32.0;
  const float float_scale = 1.0 / 65536.0;
  const float int_scale = 256.0;

  ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
              abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
              abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}
