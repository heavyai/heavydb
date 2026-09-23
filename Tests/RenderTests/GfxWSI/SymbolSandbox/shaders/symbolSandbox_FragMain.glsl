/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(location = 0) out vec4 color;

void main() {
  vec4 frag_color = getFragmentColorOrDiscard();
  color = vec4(frag_color.rgb * frag_color.a, frag_color.a);
}
