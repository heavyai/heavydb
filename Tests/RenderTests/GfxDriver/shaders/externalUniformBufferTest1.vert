/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec2 in_position;
in vec3 in_color;

layout(location = 0) out vec4 fColor;

layout(std140, binding = 0) uniform EXTERNAL_UBO {
  float vert_value;
  float frag_value;
};

void main() {
  gl_Position = vec4(in_position.x + vert_value, in_position.y, 0.5, 1.0);
  fColor = vec4(in_color, 1.0);
}
