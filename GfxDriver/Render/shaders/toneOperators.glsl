/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Basic Reinhard
vec3 reinhard(vec3 v) {
  return v / (1.0 + v);
}

// Reinhard extended with white point support
vec3 reinhard_extended(vec3 v, float max_white) {
  vec3 num = v * (1.0 + (v / vec3(max_white * max_white)));
  return num / (1.0 + v);
}

// Reinhard modifying luminance
float luminance(vec3 v) {
  return dot(v, vec3(0.2126, 0.7152, 0.0722));
}

vec3 change_luminance(vec3 c_in, float l_out) {
  float l_in = luminance(c_in);
  return c_in * (l_out / l_in);
}

vec3 reinhard_extended_luminance(vec3 v, float max_white_l) {
  float l_old = luminance(v);
  float numerator = l_old * (1.0 + (l_old / (max_white_l * max_white_l)));
  float l_new = numerator / (1.0 + l_old);
  return change_luminance(v, l_new);
}

// Uncharted 2 filmic
vec3 uncharted2_tonemap_partial(vec3 x)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x *(A * x + B) + D * F)) - E / F;
}

vec3 uncharted2_filmic(vec3 v)
{
    float exposure_bias = 2.0;
    vec3 curr = uncharted2_tonemap_partial(v * exposure_bias);

    vec3 W = vec3(11.2);
    vec3 white_scale = vec3(1.0) / uncharted2_tonemap_partial(W);
    return curr * white_scale;
}
