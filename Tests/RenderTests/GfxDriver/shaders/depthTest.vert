/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec3 in_position;

layout(std430) uniform DEPTH_TEST_UBO_TYPE {
  mat4 viewTM;
};

layout(push_constant) uniform DEPTH_TEST_PUSH_CONSTANTS {
  vec3 translate;
  float scale;
} pushConstants;

layout(location = 0) out float fDepth;
layout(location = 1) out vec3 fColor;

void main() {
  vec4 pos = vec4(in_position.x * pushConstants.scale + pushConstants.translate.x,
                  in_position.y * pushConstants.scale + pushConstants.translate.y,
                  in_position.z * pushConstants.scale + pushConstants.translate.z,
                  1.0);
  gl_Position = viewTM * pos;
  fDepth = in_position.z;
  fColor = in_position * 0.5 + 0.5;
}
