/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) in float fDepth;
layout(location = 1) in vec3 fColor;

layout(location = 0) out vec4 out_color;

#define SHOW_WIREFRAME 0

void main() {
  vec3 color = fColor * fDepth;
#if SHOW_WIREFRAME == 1
  int bits = bitCount(gl_SampleMaskIn[0]);
  out_color = (bits < 4) ? vec4(color * float(bits) * 0.25, 1.0) : vec4(color, 1.0);
#else
  out_color = vec4(color, 1.0);
#endif
}
