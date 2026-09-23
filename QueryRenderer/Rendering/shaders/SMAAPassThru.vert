/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) out vec2 fTexCoord;

layout(std430) uniform SMAA_PASS_THRU_VERT_UBO_TYPE {
  vec2 VIEWPORT_SCALING;
};

void main() {
  float x = float((gl_VertexIndex & 1) << 2);
  float y = float((gl_VertexIndex & 2) << 1);
  fTexCoord = vec2(x * 0.5, y * 0.5);
  gl_Position = vec4((x * VIEWPORT_SCALING.x) - 1.0, (y * VIEWPORT_SCALING.y) - 1.0, 0.5, 1.0);
}

