/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DriverTests/testUtils.glsl"

layout(location = 0) out vec4 densityColor;
layout(location = 1) out vec4 stdDevColor;

layout(r32ui) uniform restrict readonly uimage2D pixelCounter;

layout(std430) buffer IMAGE_STATS_SSBO {
  uint64_t totalNonZeroCount;
  uint64_t totalSqrDiff;
  uint32_t minCount;
  uint32_t maxCount;
  uint32_t numNonZeroCount;
} imageStats;

vec4 getDensityColor(in uint count) {
  float fmax = float(imageStats.maxCount);
  float fcount = float(count);
  float opacity_scale = 1.0 / fmax;
  return transformHSLtoRGB(vec4(fcount, 1.0, 0.5, fcount * opacity_scale));
}

vec4 getStdDevColor(in uint count) {
  float numPixels = float(imageStats.numNonZeroCount);
  float mean = float(imageStats.totalNonZeroCount) / numPixels;
  float variance = float(imageStats.totalSqrDiff) / numPixels;
  float stddev = sqrt(variance);
  float fmin = float(imageStats.minCount);
  float fmax = float(imageStats.maxCount);

  return vec4(stddev / float(count));
}

void main() {
  uint32_t count = imageLoad(pixelCounter, ivec2(gl_FragCoord.xy)).r;
  if (count > 0u) {
    densityColor = getDensityColor(count);
    stdDevColor = getStdDevColor(count);
  } else {
    densityColor = vec4(0);
    stdDevColor = vec4(0);
  }
}
