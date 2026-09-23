/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec2 in_position;
in vec2 in_uv;

layout(std430, binding=0) uniform QUAD_SHADER_UBO_TYPE {
  vec2 imageSize;
  vec2 screenOffset;
};

layout(location = 0) out vec2 fUV;

void main() {
  float aspect = imageSize.x / imageSize.y;
  vec2 position = in_position + screenOffset;
  gl_Position = vec4(position.x, position.y * aspect, 0.5, 1.0);
  fUV = in_uv;
}
