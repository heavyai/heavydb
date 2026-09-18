/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// BEGIN SCALE thresholdScaleTemplate.vert

#define domainType_<name> <domainType>
#define rangeType_<name> <rangeType>

#define numRanges_<name> <numRanges>
#define numDomains_<name> <numDomains>

layout(std430) uniform THRESHOLD_SCALE_UBO_TYPE_<name> {
  domainType_<name> uDomains_<name>[numDomains_<name>];
  domainType_<name> nullDomainVal_<name>;
  rangeType_<name> uRanges_<name>[numRanges_<name>];
  rangeType_<name> nullRangeVal_<name>;
};

//
// Null value handling
//
// stub
bool isNullValFunc_<name>(in domainType_<name>) { return false; }

// pass-through
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

#define doAccum_ThresholdScale_<name> <doAccum>

rangeType_<name> evalThresholdScale_<name>(in domainType_<name> category) {
        int idx = -1;
        rangeType_<name> val = rangeType_<name>(0);
        if (isNullValFunc_<name>(category)) {
            idx = numRanges_<name>;
            val = nullRangeVal_<name>;
        } else {
            for(int i=0; i<numDomains_<name>; i++) {
                if(category < uDomains_<name>[i]) {
                idx = i;
                break;
                }
            }
            if (idx < 0) {
            idx = numRanges_<name> - 1;
            }
            val = uRanges_<name>[idx];
    }
    #if doAccum_ThresholdScale_<name> == 1
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

// END SCALE thresholdScaleTemplate.vert
