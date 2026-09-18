/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// BEGIN SCALE quantitativeScaleTemplate.vert

#define domainType_<name> <domainType>
#define domainTypeEnum_<name> <domainTypeEnum>
#define rangeType_<name> <rangeType>
#define rangeTypeEnum_<name> <rangeTypeEnum>

#define numDomains_<name> <numDomains>
#define numRanges_<name> <numRanges>
#define useClamp_<name> <useClamp>

layout(std430) uniform QUANTITATIVE_SCALE_UBO_TYPE_<name> {
  domainType_<name> uDomains_<name>[numDomains_<name>];
  domainType_<name> nullDomainVal_<name>;
  rangeType_<name> uRanges_<name>[numRanges_<name>];
  rangeType_<name> nullRangeVal_<name>;
  float uExponent_<name>;
};

//
// Null value handling
//
// stub
bool isNullValFunc_<name>(in domainType_<name> val) { return false; }

// Pass-through
bool isNullValPassThru_<name>(in domainType_<name> val) {
  return false;
}

// Check
bool isNullVal_<name>(in domainType_<name> val) {
  if (val == nullDomainVal_<name>) {
    return true;
  }
  return false;
}

#define doAccum_QuantitativeScale_<name> <doAccum>

//
// Not accumulating
//

#if doAccum_QuantitativeScale_<name> == 0
//
// Quantitative transform prototype
//
// Stub
domainType_<name> quantTransform_<name>(in domainType_<name> val) { return val; }

// pass-through
domainType_<name> passThruTransform_<name>(in domainType_<name> val) {
  return val;
}

// log
domainType_<name> logTransform_<name>(in domainType_<name> val) {
  return domainType_<name>(log(float(val)));
}

// pow
domainType_<name> powTransform_<name>(in domainType_<name> val) {
  return domainType_<name>(pow(float(val), uExponent_<name>));
}

// sqrt
domainType_<name> sqrtTransform_<name>(in domainType_<name> val) {
#if domainTypeEnum_<name> == DOUBLE
  return sqrt(val);
#elif domainTypeEnum_<name> == UNSIGNED_INT64_ARB || domainTypeEnum_<name> == INT64_ARB
  return domainType_<name>(sqrt(double(val)));
#else
  return domainType_<name>(sqrt(float(val)));
#endif
}

//
// Quantitative interpolation (Range)
//
// declaration
rangeType_<name> quantInterp_<name>(
  in rangeType_<name> v1,
  in rangeType_<name>,
#if domainTypeEnum_<name> == DOUBLE || domainTypeEnum_<name> == UNSIGNED_INT64_ARB || domainTypeEnum_<name> == INT64_ARB
  in double)
#else
  in float)
#endif
{ return v1; }

// default
rangeType_<name> defaultInterp_<name>(
  in rangeType_<name> v1,
  in rangeType_<name> v2,
#if domainTypeEnum_<name> == DOUBLE || domainTypeEnum_<name> == UNSIGNED_INT64_ARB || domainTypeEnum_<name> == INT64_ARB
  in double t) {
#else
  in float t) {
#endif

#if domainTypeEnum_<name> == DOUBLE || domainTypeEnum_<name> == UNSIGNED_INT64_ARB || domainTypeEnum_<name> == INT64_ARB
#if rangeTypeEnum_<name> == DOUBLE || rangeTypeEnum_<name> == DOUBLE_VEC2 || rangeTypeEnum_<name> == DOUBLE_VEC3 || rangeTypeEnum_<name> == DOUBLE_VEC4
  return mix(v1, v2, t);
#else
  return mix(v1, v2, float(t));
#endif
#else
  return mix(v1, v2, t);
#endif // if domainTypeEnum_<name> == DOUBLE
}

// Hsl / Hcl
rangeType_<name> colorInterpHslHcl_<name>(
  in rangeType_<name> v1,
  in rangeType_<name> v2,
#if domainTypeEnum_<name> == DOUBLE || domainTypeEnum_<name> == UNSIGNED_INT64_ARB || domainTypeEnum_<name> == INT64_ARB
  in double t) {
#else
  in float t) {
#endif
#if rangeTypeEnum_<name> == FLOAT_VEC4
  if (abs(v1[0] - v2[0]) > 180.0) {
    if (v1[0] < v2[0]) {
      v1[0] += 360.0;
    } else {
      v2[0] += 360.0;
    }
  }
#endif
  return defaultInterp_<name>(v1, v2, t);
}

// Hsl / Hcl long
rangeType_<name> colorInterpHslHclLong_<name>(
  in rangeType_<name> v1,
  in rangeType_<name> v2,
#if domainTypeEnum_<name> == DOUBLE || domainTypeEnum_<name> == UNSIGNED_INT64_ARB || domainTypeEnum_<name> == INT64_ARB
  in double t) {
#else
  in float t) {
#endif
#if rangeTypeEnum_<name> == FLOAT_VEC4
  float absval = abs(v1[0] - v2[0]);
  if (absval > 0 && absval < 180.0) {
    if (v1[0] < v2[0]) {
      v2[0] += 360.0;
    } else {
      v1[0] += 360.0;
    }
  }
#endif
  return defaultInterp_<name>(v1, v2, t);
}

#endif // doAccum_QuantitativeScale_<name> (not accumulating)

rangeType_<name> evalQuantitativeScale_<name>(in domainType_<name> domainVal) {
  domainType_<name> val1;
  domainType_<name> val2;
  int idx1, idx2;

#if doAccum_QuantitativeScale_<name> == 1
#if supportsGeoAccum == 1
  gAccumIdx = 0;
#else
#if IS_MESH_SHADER
  v_out[g_output_index].accumIdx = 0;
#else
  accumIdx = 0;
#endif
#endif
  return uRanges_<name>[0];
#else
  if (isNullValFunc_<name>(domainVal)) {
    return nullRangeVal_<name>;
  }

  domainType_<name> transformedVal = quantTransform_<name>(domainVal);
#if numDomains_<name> == 1 || numRanges_<name> == 1
  idx1 = 0;
  idx2 = 0;
  val1 = uDomains_<name>[0];
  val2 = uDomains_<name>[0];
#elif numDomains_<name> == 2 || numRanges_<name> == 2
  idx1 = 0;
  idx2 = 1;
  val1 = uDomains_<name>[0];
  val2 = uDomains_<name>[1];
#else
  int startIdx = 0;
  int endIdx = numDomains_<name> - 1;
  domainType_<name> midVal;

  if (transformedVal <= uDomains_<name>[startIdx]) {
    idx1 = startIdx;
    idx2 = startIdx+1;
  } else if (transformedVal >= uDomains_<name>[endIdx]) {
    idx1 = endIdx-1;
    idx2 = endIdx;
  } else {
    while (true) {
      int midIdx = startIdx + (endIdx - startIdx) / 2;
      if (midIdx == startIdx) {
        idx1 = midIdx;
        idx2 = endIdx;
        break;
      } else {
        midVal = uDomains_<name>[midIdx];
        if (transformedVal == midVal) {
          idx1 = midIdx;
          idx2 = midIdx+1;
          break;
        } else if (transformedVal > midVal) {
          startIdx = midIdx;
        } else {
          endIdx = midIdx;
        }
      }
    }
  }

  val1 = uDomains_<name>[idx1];
  val2 = uDomains_<name>[idx2];
#endif


#if domainTypeEnum_<name> == DOUBLE
  double t = (transformedVal - val1) / (val2 - val1);
#elif domainTypeEnum_<name> == INT64_ARB || domainTypeEnum_<name> == UNSIGNED_INT64_ARB
  double t = (double(transformedVal) - double(val1)) / (double(val2) - double(val1));
#else
  float t = (float(transformedVal) - float(val1)) / (float(val2) - float(val1));
#endif
  if (val2 == val1 || isnan(t)) {
    t = 0.5;
  }

#if useClamp_<name> == 1
  t = clamp(t, 0.0, 1.0);
#endif

  return quantInterp_<name>(uRanges_<name>[idx1], uRanges_<name>[idx2], t);
#endif // doAccum_QuantitativeScale_<name>
}

// END SCALE quantitativeScaleTemplate.vert
