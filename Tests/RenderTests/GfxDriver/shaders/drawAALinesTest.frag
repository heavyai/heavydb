/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) out vec4 color;

// start input data
layout(location = 0) in vec2 fNormDistCoords;
layout(location = 1) flat in vec4 fColor;
// end input data

void main() {
  // clip round joint, avoiding sqrt
  if (dot(fNormDistCoords, fNormDistCoords) > 1.0) {
    discard;
  }
  color = vec4(fColor.rgb * fColor.a, fColor.a);
}
