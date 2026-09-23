/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// BEGIN SCALE quantizeScaleTemplate.vert

#define domainType_<name> <domainType>
#define rangeType_<name> <rangeType>

#define numRanges_<name> <numRanges>

layout(std430) uniform QUANTIZE_SCALE_UBO_TYPE_<name> {
  domainType_<name> uDomains_<name>[2];
  domainType_<name> nullDomainVal_<name>;
  rangeType_<name> uRanges_<name>[numRanges_<name>];
  rangeType_<name> nullRangeVal_<name>;
};

//
// Null value handling
//
// stub
bool isNullValFunc_<name>(in domainType_<name>) { return false; }

// passthrough
bool isNullValPassThru_<name>(in domainType_<name> val) {
  return false;
}

// check
bool isNullVal_<name>(in domainType_<name> val) {
  if (val == nullDomainVal_<name>) {
    return true;
  }
  return false;
}

#define doAccum_QuantizeScale_<name> <doAccum>

// TODO(croot): right now, if the quantize scale is a density
// accumulation, we still find the index via the following math
// unnecessarily. The better approach is to make a subroutine
// or an additional #if/#else ... but that will take a little
// bit of work. Right now (as of 07/15/16) I'm changing
// accumulatorScale_1stPass_frag.h to automatically set the
// accumulator texture index to 0 rather than using 'accumTx'
rangeType_<name> evalQuantizeScale_<name>(in domainType_<name> category) {
  int idx = -1;
  rangeType_<name> val = rangeType_<name>(0);
  if (isNullValFunc_<name>(category)) {
    idx = numRanges_<name>;
    val = nullRangeVal_<name>;
  } else {
    double diff;
    double d0 = double(uDomains_<name>[0]);
    double d1 = double(uDomains_<name>[1]);
    double dcat = double(category);
    if (uDomains_<name>[0] >= domainType_<name>(0)) {
      diff = min(dcat, dcat - d0);
    } else {
      diff = max(dcat, dcat - d0);
    }
    double quantizeDiff = (d1 - d0) / double(numRanges_<name>);
    idx = int(clamp(trunc(diff / quantizeDiff), double(0), double(numRanges_<name>-1)));
    val = uRanges_<name>[idx];
  }

#if doAccum_QuantizeScale_<name> == 1
  if (idx < 0) {
    idx = 0;
  }
#if supportsGeoAccum == 1
  gAccumIdx = idx;
#else
#if IS_MESH_SHADER
  v_out[g_output_index].accumIdx = idx;
#else
  accumIdx = idx;
#endif
#endif
  return vec4(0,0,0,0);
#else
  return val;
#endif
}

// END SCALE quantizeScaleTemplate.vert
