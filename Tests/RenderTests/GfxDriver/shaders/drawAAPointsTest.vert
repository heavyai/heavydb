/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec3 in_position;
in float in_pointSize;

layout(location = 0) out float fPointSize;

void main() {
  gl_Position = vec4(in_position, 1.0);
  gl_PointSize = in_pointSize;
  fPointSize = in_pointSize;
}
