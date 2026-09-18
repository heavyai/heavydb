/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(std430, binding = 0) uniform SHARED_UBO {
  mat4 viewTM;
};
layout(std430) uniform VERT_UBO {
  float vert_x;
  int vert_y;
};

layout(std430, binding = 1) buffer SHARED_SSBO {
  float vert_ssbo_values[];
};
layout(std430) buffer VERT_SSBO {
  int vert_shared_ssbo_values[];
};

layout(r32ui, binding = 2) uniform coherent uimage2D shared_image;
layout(r32ui) uniform coherent uimage2D vert_image;

layout(binding = 3) uniform sampler2D shared_sampler;
uniform sampler2D vert_sampler;
layout(binding = 4) uniform sampler2D shared_array_of_samplers[2];
uniform sampler2D vert_array_of_samplers[2];

layout(binding = 6) uniform sampler2DArray shared_sampler_array;
uniform sampler2DArray vert_sampler_array;
layout(binding = 7) uniform sampler2DArray shared_array_of_sampler_arrays[2];
uniform sampler2DArray vert_array_of_sampler_arrays[2];

in vec3 in_position;

layout(location = 0) out float out_depth;
layout(location = 1) out vec3 out_color;

void main() {
  gl_Position = viewTM * vec4(in_position, 1.0);
  out_depth = in_position.z + vert_x + float(vert_y);
  out_color = in_position * 0.5 + 0.5;
}
