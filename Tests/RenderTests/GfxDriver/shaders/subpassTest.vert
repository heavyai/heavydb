/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define NUM_OBJECTS 7

in vec3 in_position;

layout(std430) uniform SUBPASS_TEST_CONSTANT_UBO_TYPE {
  mat4 viewTM;
  float scale;
};

layout(std430) uniform SUBPASS_TEST_PER_OBJECT_UBO_TYPE {
  vec4 objPositions[NUM_OBJECTS];
};

layout(push_constant) uniform PUSH_CONSTANTS {
  uint32_t object_index;
} pushConstants;

layout(location = 0) out float fDepth;
layout(location = 1) out vec3 fColor;

void main() {
  vec3 translate = objPositions[pushConstants.object_index].xyz;
  vec4 pos = vec4(in_position.x * scale + translate.x,
                  in_position.y * scale + translate.y,
                  in_position.z * scale + translate.z,
                  1.0);
  gl_Position = viewTM * pos;
  fDepth = in_position.z;
  fColor = in_position * 0.5 + 0.5;
}
