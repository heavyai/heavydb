/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec3 in_position;

layout(std430) uniform PPLL_TEST_UBO_TYPE {
  mat4 viewTM;
};

layout(std430) buffer PPLL_TEST_SSBO_TYPE {
    vec4 sphereData[];
};

void main() {
  vec4 data = sphereData[gl_InstanceIndex];
  vec4 pos = vec4(in_position.x * data.w + data.x,
                  in_position.y * data.w + data.y,
                  in_position.z * data.w + data.z,
                  1.0);
  gl_Position = viewTM * pos;
}
