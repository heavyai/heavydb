/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec3 in_position;

layout(std430) uniform DRAW_SPHERES_TEST_UBO_TYPE {
  vec3 translate;
  float scale;
  mat4 viewTM;
};

layout(location = 0) out float fDepth;
layout(location = 1) out vec3 fColor;

void main() {
  vec4 pos = vec4(in_position.x * scale + translate.x,
                  in_position.y * scale + translate.y,
                  in_position.z * scale + translate.z,
                  1.0);
  gl_Position = viewTM * pos;
  fDepth = in_position.z;
  fColor = translate * 0.5 + 0.5;
}
