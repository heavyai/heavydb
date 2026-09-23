/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(r32ui) uniform restrict readonly uimage2DArray srcAccumTxArray;
layout(r32ui) uniform restrict coherent uimage2DArray inTxArrayPixelCounter;

layout(push_constant) uniform ACCUMULATION_COMPOSITE_PUSH_CONSTANTS {
  int numAccumTextures;
} pushConstants;

void main() {
  for (int i = 0; i < pushConstants.numAccumTextures; ++i) {
    uint cnt = imageLoad(srcAccumTxArray, ivec3(gl_FragCoord.xy, i)).r;
    imageAtomicAdd(inTxArrayPixelCounter, ivec3(gl_FragCoord.xy, i), cnt);
  }
}
