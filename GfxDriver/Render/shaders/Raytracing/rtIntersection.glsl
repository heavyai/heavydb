/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

struct Sphere {
  vec3 center;
  float radius;
};

float intersect_sphere(const Sphere s, const vec3 origin, const vec3 dir) {
  vec3 L = origin - s.center;
  float a = dot(dir, dir);
  float b = 2.0 * dot(L, dir);
  float c = dot(L, L) - (s.radius * s.radius);
  float d = b * b - 4.0 * a * c;
  if (d > 0) {
    return (-b - sqrt(d)) / (2.0 * a);
  }
  return -1;
}
