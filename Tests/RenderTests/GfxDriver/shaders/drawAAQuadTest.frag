/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) in vec2 fUV;

layout(std430, binding=0) uniform QUAD_SHADER_UBO_TYPE {
  vec2 imageSize;
  vec2 screenOffset;
};

layout(location = 0) out vec4 color;

const vec2 center = vec2(0.5, 0.5);

void main() {
  vec2 sample_offset = (gl_SamplePosition - vec2(0.5)) / imageSize;
  float d = distance(fUV + sample_offset, center);
  if (d > 0.25) {
    discard;
  }
  color = vec4(fUV, 0.0, 1.0);
}
