/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// VERTEX SHADER
void main() {
  float x = -1.0 + float((gl_VertexIndex & 1) << 2);
  float y = -1.0 + float((gl_VertexIndex & 2) << 1);
  gl_Position = vec4(x, y, 0.5, 1.0);
}
