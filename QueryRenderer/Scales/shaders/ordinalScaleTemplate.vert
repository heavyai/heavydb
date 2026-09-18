/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// BEGIN SCALE ordinalScaleTemplate.vert

#define domainType_<name> <domainType>
#define rangeType_<name> <rangeType>

#define numDomains_<name> <numDomains>
#define numRanges_<name> <numRanges>

layout(std430) uniform ORDINAL_SCALE_UBO_TYPE_<name> {
  domainType_<name> uDomains_<name>[numDomains_<name>];
  domainType_<name> nullDomainVal_<name>;
  rangeType_<name> uRanges_<name>[numRanges_<name>];
  rangeType_<name> nullRangeVal_<name>;
  rangeType_<name> uDefault_<name>;
};

//
// Null value handling
//
// stub
bool isNullValFunc_<name>(in domainType_<name>) { return false; }

// pass-through
// TODO(scb): Doesn't need to be a 'subroutine'
bool isNullVal_<name>(in domainType_<name> val) {
  if (val == nullDomainVal_<name>) {
    return true;
  }
  return false;
}

#define doAccum_OrdinalScale_<name> <doAccum>

rangeType_<name> evalOrdinalScale_<name>(in domainType_<name> category) {
  int idx = -1;
  rangeType_<name> val = uDefault_<name>;
  if (isNullValFunc_<name>(category)) {
    idx = numDomains_<name> + 1;
    val = nullRangeVal_<name>;
  } else {
    int i = 0;
    while (i < numDomains_<name>) {
      if (uDomains_<name>[i] == category) {
        idx = i;
        break;
      }
      ++i;
    }

    if (idx >= 0) {
      idx = idx % numRanges_<name>;
      val = uRanges_<name>[idx];
    }
  }

#if doAccum_OrdinalScale_<name> == 1
  if (idx < 0) {
    idx = numDomains_<name>;
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
#endif

  return val;
}

// END SCALE ordinalScaleTemplate.vert
