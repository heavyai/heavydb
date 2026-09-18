/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) out vec4 color;

layout(std430) uniform DRAW_FULLSCREEN_FRAG_UBO_TYPE {
  vec2 imageSize;
  int invertY;
};

void main() {
  float y = gl_FragCoord.y / imageSize.y;
  if (invertY == 1) {
    y = 1.0 - y;
  }
  color = vec4(gl_FragCoord.x / imageSize.x, y, 0.0, 1.0);
}
