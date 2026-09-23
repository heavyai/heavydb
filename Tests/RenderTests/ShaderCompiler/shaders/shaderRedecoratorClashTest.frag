/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(std430, binding = 0) uniform SHARED_UBO {
  mat4 viewTM;
};

// this will clash with the image in the vert shader
layout(std430, binding = 1) buffer SHARED_SSBO {
  float frag_ssbo_values[];
};

layout(location = 0) in float in_depth;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec4 color;

void main() {
  color = vec4(in_color * in_depth * frag_ssbo_values[0], float(frag_ssbo_values[1]));
}
