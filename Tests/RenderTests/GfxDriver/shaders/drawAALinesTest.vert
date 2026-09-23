/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#extension GL_ARB_shader_draw_parameters : require

in vec2 in_position;
in vec3 in_color;
in float in_stroke_width;
in float in_line_join;

layout(location = 0) out vec4 gColor;
layout(location = 1) out float gStrokeWidth;
layout(location = 2) out int gLineJoin;
layout(location = 3) out float gMiterLimit;

void main() {
  gl_Position = vec4(in_position, 0.5, 1.0);
  gColor = vec4(in_color, 1.0);
  gStrokeWidth = in_stroke_width;
  gLineJoin = int(in_line_join);
  gMiterLimit = 10.0;
}
