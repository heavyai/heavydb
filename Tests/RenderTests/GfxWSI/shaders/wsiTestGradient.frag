/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) out vec4 color;

layout(std430) uniform WSI_GRADIENT_UBO_TYPE {
  vec2 imageSize;
  vec4 gradColor1;
  vec4 gradColor2;
};

void main() {
  vec2 uv = vec2(gl_FragCoord.x / imageSize.x, gl_FragCoord.y / imageSize.y);

  // checker
  const float checker_scale = 20.0;
  float total = floor(uv.x * checker_scale) + floor(uv.y * checker_scale);
  bool is_even = mod(total, 2.0) == 0.0;
  vec3 checker_col = is_even ? vec3(0,0,0) : vec3(0.5,0.5,0.5);

  // gradient
  vec4 c1 = mix(vec4(0,0,0,1), gradColor1, uv.x);
  vec4 c2 = mix(vec4(0,0,0,1), gradColor2, uv.y);
  float a = c1.a * c2.a;

  color = vec4(mix(checker_col, vec3(c1.rgb) + vec3(c2.rgb), a), 1);
}
