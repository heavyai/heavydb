/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) in vec4 fColor;

layout(location = 0) out vec4 color;

layout(std140, binding = 0) uniform EXTERNAL_UBO {
  float vert_value;
  float frag_value;
};

void main() {
  color = vec4(fColor.r + frag_value, fColor.g, fColor.b, fColor.a);
}
