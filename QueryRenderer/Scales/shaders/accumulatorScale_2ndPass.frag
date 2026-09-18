/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//<includes>
#include "Marks/typeDefines.glsl"
//</includes>
// FRAGMENT SHADER
#define numAccumColors_<name> <numAccumColors>
#define numAccumTextures_<name> <numAccumTextures>

#define useExtentsSSBO <useExtentsSSBO>

layout(location = 0) out vec4 color;

layout(std430) uniform ACCUMULATOR_SCALE_2ND_PASS_UBO_TYPE_<name> {
  vec4 inColors[numAccumColors_<name>];
  uint minDensity;
  uint maxDensity;
  float minStdDev;
  float maxStdDev;
};

#if useExtentsSSBO == 1
layout(std430) buffer EXTENTS_SSBO {
  uint64_t totalNonZeroCount;
  uint64_t totalSqrDiff;
  uint32_t minCount;
  uint32_t maxCount;
  uint32_t numNonZeroCount;
} extents;
#endif // useExtentsSSBO

layout(r32ui) uniform restrict readonly uimage2DArray inTxArrayPixelCounter;

//
// Accumulated count handling
//
uint getAccumulatedCnt(uint idx) {
  return imageLoad(inTxArrayPixelCounter, ivec3(gl_FragCoord.xy, idx)).r;
}

//
// StdDev handling
//
// stub
void calcMeanStdDev(out float mean, out float stddev) {
  mean = 0.0;
  stddev = 0.0;
}

void getEmptyMeanStdDev(out float mean, out float stddev) {
  mean = 0.0;
  stddev = 0.0;
}

#if useExtentsSSBO == 1
void getTextureMeanStdDev(out float mean, out float stddev) {
  float numPixels = float(extents.numNonZeroCount);
  mean = float(extents.totalNonZeroCount) / numPixels;
  float variance = float(extents.totalSqrDiff) / numPixels;
  stddev = sqrt(variance);
}
#endif // useExtentsSSBO

//
// Min/Max density handling
//
// stubs
uint getMinDensity(in float, in float) { return 0; }
uint getMaxDensity(in float, in float) { return 0; }

uint getUniformMinDensity(in float mean, in float stddev) {
  return minDensity;
}

uint getUniformMaxDensity(in float mean, in float stddev) {
  return maxDensity;
}

#if useExtentsSSBO == 1
uint getTextureMinDensity(in float mean, in float stddev) {
  return extents.minCount;
}

uint getStdDevMinDensity(in float mean, in float stddev) {
  return uint(max(floor(mean - minStdDev * stddev), float(extents.minCount)));
}

uint getTextureMaxDensity(in float mean, in float stddev) {
  return extents.maxCount;
}

uint getStdDevMaxDensity(in float mean, in float stddev) {
  return uint(min(ceil(mean + maxStdDev * stddev), float(extents.maxCount)));
}
#endif // useExtentsSSBO

vec4 getDensityColor(in domainType_<name>_ACCUMULATION pct) {
  return float(pct) * inColors[0];
}

//
// Accumulated color handling. These are the primary entry points
//
// stub
vec4 getAccumulatedColor() { return vec4(0); }

vec4 getMinAccumulatedColor() {
  uint idx = 0;
  uint cnt;
  vec4 finalColor = vec4(0,0,0,0);

  uint cnt1, cnt2;
  int minIdx = -1;
  uint minCnt = 10000000;
  for (int i = 0; i < numAccumColors_<name>; i+=2) {
    idx = i / 2;

    if (idx < numAccumTextures_<name>) {
      cnt = getAccumulatedCnt(idx);
      cnt1 = cnt & uint(0x000000FF);
      if (cnt1 > 0 && cnt1 < minCnt) {
        minCnt = cnt1;
        minIdx = i;
      }
      cnt2 = (cnt & uint(0x00FF0000)) >> 16;
      if (cnt2 > 0 && cnt2 < minCnt) {
        minCnt = cnt2;
        minIdx = i + 1;
      }
    }
  }
  if (minIdx >= 0) {
    finalColor = inColors[minIdx];
  }
  return finalColor;
}

vec4 getMaxAccumulatedColor() {
  uint idx = 0;
  uint cnt;
  vec4 finalColor = vec4(0,0,0,0);

  uint cnt1, cnt2;
  int maxIdx = -1;
  uint maxCnt = 0;
  for (int i = 0; i < numAccumColors_<name>; i+=2) {
    idx = i / 2;

    if (idx < numAccumTextures_<name>) {
      cnt = getAccumulatedCnt(idx);
      cnt1 = cnt & uint(0x000000FF);
      if (cnt1 > 0 && cnt1 > maxCnt) {
        maxCnt = cnt1;
        maxIdx = i;
      }
      cnt2 = (cnt & uint(0x00FF0000)) >> 16;
      if (cnt2 > 0 && cnt2 > maxCnt) {
        maxCnt = cnt2;
        maxIdx = i + 1;
      }
    }
  }
  if (maxIdx >= 0) {
    finalColor = inColors[maxIdx];
  }
  return finalColor;
}

vec4 getBlendAccumulatedColor() {
  uint idx = 0;
  uint cnt;
  vec4 finalColor = vec4(0,0,0,0);

  uint cnts[numAccumColors_<name> + (numAccumColors_<name> % 2)];
  uint totalCnt = 0;
  for (uint i = 0; i < numAccumColors_<name>; i+=2) {
    idx = i / 2;

    if (idx < numAccumTextures_<name>) {
      cnt = getAccumulatedCnt(idx);
      cnts[i] = cnt & uint(0x000000FF);
      cnts[i+1] = (cnt & uint(0x00FF0000)) >> 16;
      totalCnt += cnts[i] + cnts[i+1];
    }
  }

  float sum = float(totalCnt);
  for (uint i = 0; i < numAccumColors_<name>; i++) {
    finalColor += (float(cnts[i]) / sum) * inColors[i];
  }
  return finalColor;
}

vec4 getDensityAccumulatedColor() {
  uint totalCnt = 0;

  // there should only be 1 counter texture for density accumulation
  totalCnt = getAccumulatedCnt(0);
  if (totalCnt == 0) {
    return vec4(0,0,0,0);
  }
  float mean, stddev;
  calcMeanStdDev(mean, stddev);
  uint myMinDensity = getMinDensity(mean, stddev);
  uint myMaxDensity = getMaxDensity(mean, stddev);
  // NOTE: had to convert totalCnt and myMinDensity to floats/doubles before the first subtraction
  // to care for the case when totalCnt < myMinDensity (if we cast after the subtraction, we'd get a huge number).
  // However, there's no need to do that in the myMaxDensity-myMinDensity case as max should always be >= min.
  domainType_<name>_ACCUMULATION pct;
  if (myMinDensity == myMaxDensity) {
    pct = domainType_<name>_ACCUMULATION(1);
  } else {
    pct = (domainType_<name>_ACCUMULATION(totalCnt) - domainType_<name>_ACCUMULATION(myMinDensity)) /
          domainType_<name>_ACCUMULATION(myMaxDensity - myMinDensity);
  }
  return getDensityColor(pct);
}

vec4 getPctColor(in domainType_<name>_ACCUMULATION pct) {
  return float(pct)*inColors[0];
}

vec4 getPctAccumulatedColor() {
  uint totalCnt = getAccumulatedCnt(0);
  uint itemCnt = 0;
  domainType_<name>_ACCUMULATION pct = domainType_<name>_ACCUMULATION(0);
  if (totalCnt > 0) {
    itemCnt = getAccumulatedCnt(1);
    pct = domainType_<name>_ACCUMULATION(itemCnt) / domainType_<name>_ACCUMULATION(totalCnt);
    return getPctColor(pct);
  }
  return vec4(0,0,0,0);
}

void main() {
  // get final color
  vec4 final_color = getAccumulatedColor();

  // is there a color contribution?
  if (final_color.a > 0.0) {
    // output premultiplied color
    color = vec4(final_color.rgb * final_color.a, final_color.a);
  } else {
    discard;
  }
}
