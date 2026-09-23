/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(std430, binding = 0) uniform SHARED_UBO {
  mat4 viewTM;
};
layout(std430) uniform FRAG_UBO {
  float frag_x;
  int frag_y;
};

layout(std430, binding = 1) buffer SHARED_SSBO {
  float frag_ssbo_values[];
};
layout(std430) buffer FRAG_SSBO {
  int frag_shared_ssbo_values[];
};

layout(r32ui, binding = 2) uniform coherent uimage2D shared_image;
layout(r32ui) uniform coherent uimage2D frag_image;

layout(binding = 3) uniform sampler2D shared_sampler;
uniform sampler2D frag_sampler;
layout(binding = 4) uniform sampler2D shared_array_of_samplers[2];
uniform sampler2D frag_array_of_samplers[2];

layout(binding = 6) uniform sampler2DArray shared_sampler_array;
uniform sampler2DArray frag_sampler_array;
layout(binding = 7) uniform sampler2DArray shared_array_of_sampler_arrays[2];
uniform sampler2DArray frag_array_of_sampler_arrays[2];

layout(location = 0) in float in_depth;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec4 color;

void main() {
  color = vec4(in_color * in_depth * frag_x, float(frag_y));
}
