/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout (location = 0) in PerVertexData {
  vec4 color;
} f_in;

layout (location = 0) out vec4 fColor;

void main() {
  fColor = f_in.color;
}
