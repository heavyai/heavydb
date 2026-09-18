/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// BEGIN SCALE accumulatorScalePct_1stPass.vert

#define domainType_<name> <domainType>
#define domainTypeEnum_<name> <domainTypeEnum>
#define rangeType_<name> <rangeType>
#define rangeTypeEnum_<name> <rangeTypeEnum>

layout(std430) uniform ACCUMULATOR_SCALE_PCT_1ST_PASS_UBO_TYPE_<name> {
  domainType_<name> uPctVal_<name>;
  domainType_<name> uPctMargin_<name>;
};

rangeType_<name> evalPctScale_<name>(in domainType_<name> inval) {
  int idx = -1;
  if (inval >= uPctVal_<name> - uPctMargin_<name> && inval <= uPctVal_<name> + uPctMargin_<name>) {
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
  return rangeType_<name>(0);
}

// END SCALE accumulatorScalePct_1stPass.vert
