/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) out vec4 color;

const int kModeSolid = 1;
const int kModeGrid = 2;
const int kModeEditor = 3;

// Flags from QueryRenderer
#define FLAG_XY_SYMMETRY (1 << 0)
#define FLAG_X_SYMMETRY (1 << 1)

layout(std430) uniform MULTISYMBOL_BG_TYPE {
  int mode;
  vec3 solidColor;
  vec3 gridColor;
  vec2 imageSize;
  uint symbolFlags;
};

float grid(vec2 frag_coord, float space, float grid_width)
{
    vec2 p  = frag_coord - vec2(0.5);
    vec2 size = vec2(grid_width - 0.5);

    vec2 a1 = mod(p - size, space);
    vec2 a2 = mod(p + size, space);
    vec2 a = a2 - a1;

    float g = min(a.x, a.y);
    return clamp(g, 0.0, 1.0);
}

void main() {
  if (mode == kModeSolid) {
    color = vec4(solidColor, 1);
  } else if (mode == kModeGrid || mode == kModeEditor) {
    vec2 p = (gl_FragCoord.xy - (imageSize / 2.0));
    float g1 = grid(p, imageSize.x / 4.0, 5.0);
    float g2 = grid(p, imageSize.x / 12.0, 2.5);
    float g3 = grid(p, imageSize.x / 36.0, 1.0);
    vec3 c = mix(gridColor, solidColor, clamp(g1 * g2 * g3, 0.0, 1.0));
    if (mode == kModeEditor) {
      if ((symbolFlags & FLAG_XY_SYMMETRY) == FLAG_XY_SYMMETRY) {
        if (p.x < 0.0 || p.y > 0.0) {
          c = c * 0.5;
        }
      }
      if ((symbolFlags & FLAG_X_SYMMETRY) == FLAG_X_SYMMETRY) {
        if (p.x < 0.0) {
          c = c * 0.5;
        }
      }
    }
    color = vec4(c, 1);
  } else {
    color = vec4(1, 0, 1, 1); // magenta = error
  }
}
