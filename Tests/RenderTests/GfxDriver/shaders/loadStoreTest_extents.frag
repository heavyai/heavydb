/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

layout(r32ui) uniform restrict readonly uimage2D pixelCounter;

layout(std430) buffer IMAGE_STATS_SSBO {
  uint64_t totalNonZeroCount;
  uint64_t totalSqrDiff;
  uint32_t minCount;
  uint32_t maxCount;
  uint32_t numNonZeroCount;
} imageStats;

void main() {
  uint32_t count = imageLoad(pixelCounter, ivec2(gl_FragCoord.xy)).r;
  if (count > 0u) {
    atomicMin(imageStats.minCount, count);
    atomicMax(imageStats.maxCount, count);
    atomicAdd(imageStats.totalNonZeroCount, uint64_t(count));
    atomicAdd(imageStats.numNonZeroCount, 1);
  }
}
