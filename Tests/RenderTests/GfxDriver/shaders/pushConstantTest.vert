/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec2 in_position;

layout(push_constant) uniform PUSH_CONSTANTS {
  uint32_t index;
} pushConstants;

const float offsets[5] = float[5](-0.6667, -0.3333, 0.0, 0.3333, 0.6667);

void main() {
  gl_Position = vec4(in_position.x + offsets[pushConstants.index], in_position.y, 0.5, 1.0);
}
