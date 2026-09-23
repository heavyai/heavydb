/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) in float fDepth;
layout(location = 1) in vec3 fColor;

layout(location = 0) out vec4 color;

void main() {
  color = vec4(fColor * fDepth, 1.0);
}
