/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

in vec2 in_position;

void main() {
  gl_Position = vec4(in_position.x,
                  in_position.y,
                  0.0,
                  1.0);
}
