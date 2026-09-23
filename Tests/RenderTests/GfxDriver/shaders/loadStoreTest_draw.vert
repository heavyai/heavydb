/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec2 in_position;
in float in_pointSize;

void main() {
  gl_Position = vec4(in_position.xy, 0.5, 1.0);
  gl_PointSize = in_pointSize;
}
