/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(r32ui) uniform restrict readonly uimage2D srcAccumTx;
layout(r32ui) uniform restrict coherent uimage2DArray inTxArrayPixelCounter;

layout(push_constant) uniform ACCUMULATION_COMPOSITE_PUSH_CONSTANTS {
  int layerIndex;
} pushConstants;

void main() {
  uint cnt = imageLoad(srcAccumTx, ivec2(gl_FragCoord.xy)).r;
  int32_t layer_index = pushConstants.layerIndex;
  imageAtomicAdd(inTxArrayPixelCounter, ivec3(gl_FragCoord.xy, layer_index), cnt);
}
