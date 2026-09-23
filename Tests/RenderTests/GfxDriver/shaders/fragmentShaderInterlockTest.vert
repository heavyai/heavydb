/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec2 in_position;
in float in_key;

layout (location = 0) out float out_key;

void main() {
  gl_Position = vec4(in_position.xy, 0.5, 1.0);
  gl_PointSize = 30.0;
  out_key = in_key;
}
