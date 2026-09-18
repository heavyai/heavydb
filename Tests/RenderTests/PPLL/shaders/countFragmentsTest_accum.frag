/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(r32ui) uniform restrict uimage2D fragment_count_image;

void main(void) {
  imageAtomicAdd(fragment_count_image, ivec2(gl_FragCoord.xy), 1);
}
