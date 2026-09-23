/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec2 in_position;
in vec3 in_color;

layout(location = 0) out vec4 fColor;

void main() {
  gl_Position = vec4(in_position.xy, 0.5, 1.0);
  fColor = vec4(in_color, 1.0);
}
