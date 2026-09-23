/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout (location=0) out vec4 color;

layout(std430) uniform DRAW_FULLSCREEN_FRAG_UBO_TYPE {
  vec2 imageSize;
  int invertY;
  int loopEnd;
};

void main() {
  int x = 1;
  float y = 0.0;
  // This loop should trigger device lost
  while (x > loopEnd) {
    y = float(x) * sqrt(y) + 4.0;
    x++;
  }
  color = vec4(gl_FragCoord.x / imageSize.x, y, 0.0, 1.0);
}
