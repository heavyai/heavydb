/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(std430, binding = 0) uniform SHARED_UBO {
  mat4 viewTM;
};

// this will clash with the SSBO in the frag shader
layout(r32ui, binding = 1) uniform coherent uimage2D shared_image;

in vec3 in_position;

layout(location = 0) out float out_depth;
layout(location = 1) out vec3 out_color;

void main() {
  gl_Position = viewTM * vec4(in_position, 1.0);
  out_depth = in_position.z;
  out_color = in_position * 0.5 + 0.5;
}
