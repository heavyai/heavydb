/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// main() implementation for accumulation (mainTemplate_Accumulation.glsl)
//
#define numAccumTextures_<name> <numAccumTextures>
layout(r32ui) uniform coherent restrict uimage2DArray inTxArrayPixelCounter;

// stub
void accumulate() {}

void minMaxBlendAccumulate() {
    uint currIdx = accumIdx / 2;
    uint shiftIdx = accumIdx % uint(2);
    uint shift = shiftIdx * 16;

    uint incr = 1 << shift;
    imageAtomicAdd(inTxArrayPixelCounter, ivec3(gl_FragCoord.xy, currIdx), incr);
}

void pctAccumulate() {
    ivec2 loc = ivec2(gl_FragCoord.xy);
    imageAtomicAdd(inTxArrayPixelCounter, ivec3(loc, 0), 1);
#if numAccumTextures_<name> > 1
    if (accumIdx == 0) {
        imageAtomicAdd(inTxArrayPixelCounter, ivec3(loc, 1), 1);
    }
#endif
}

void densityAccumulate() {
    // NOTE: accumIdx should always be 0 in the density case
    imageAtomicAdd(inTxArrayPixelCounter, ivec3(gl_FragCoord.xy, 0), 1);
}

void main(void) {
  maybeDiscard();
  accumulate();
  writeOutput_ID(fRowId);
}
