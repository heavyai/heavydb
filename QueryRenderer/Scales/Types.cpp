/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/Types.h"

#include <string>

namespace QueryRenderer {

std::string to_string(const ScaleType scale_type) {
  switch (scale_type) {
    case ScaleType::kLinear:
      return "LINEAR";
    case ScaleType::kLog:
      return "LOG";
    case ScaleType::kPow:
      return "POW";
    case ScaleType::kSqrt:
      return "SQRT";
    case ScaleType::kOrdinal:
      return "ORDINAL";
    case ScaleType::kQuantize:
      return "QUANTIZE";
    case ScaleType::kThreshold:
      return "THRESHOLD";
    case ScaleType::kUndefined:
      return "UNDEFINED";
    default:
      return "<scale type " + std::to_string(static_cast<int>(scale_type)) + ">";
  }

  return "";
}

std::string to_string(const AccumulatorType accum_type) {
  switch (accum_type) {
    case AccumulatorType::kMin:
      return "MIN";
    case AccumulatorType::kMax:
      return "MAX";
    case AccumulatorType::kBlend:
      return "BLEND";
    case AccumulatorType::kPct:
      return "PCT";
    case AccumulatorType::kDensity:
      return "DENSITY";
    case AccumulatorType::kUndefined:
      return "UNDEFINED";
    case AccumulatorType::kAll:
      return "ALL";
  }

  return "";
}

std::string to_string(const ScaleInterpType interp_type) {
  switch (interp_type) {
    case ScaleInterpType::kRgb:
      return "InterpolateRGB";
    case ScaleInterpType::kHsl:
      return "InterpolateHsl";
    case ScaleInterpType::kHslLong:
      return "InterpolateHslLong";
    case ScaleInterpType::kLab:
      return "InterpolateLab";
    case ScaleInterpType::kHcl:
      return "InterpolateHcl";
    case ScaleInterpType::kHclLong:
      return "InterpolateHclLong";
    case ScaleInterpType::kUndefined:
      return "UNDEFINED";
    default:
      return "scale interpolator type " + std::to_string(static_cast<int>(interp_type)) +
             ">";
  }
  return "";
}

std::vector<std::string> getScaleInterpTypes(
    const std::vector<ScaleInterpType>& interps) {
  int num_interp_types = interps.size();
  bool use_arg = num_interp_types > 0;
  if (!use_arg) {
    num_interp_types = static_cast<int>(ScaleInterpType::kUndefined);
  }

  std::vector<std::string> rtn(num_interp_types);
  if (use_arg) {
    int i = 0;
    for (auto& interp : interps) {
      rtn[i++] = to_string(interp);
    }
  } else {
    for (int i = 0; i < num_interp_types; ++i) {
      rtn[i] = to_string(static_cast<ScaleInterpType>(i));
    }
  }
  return rtn;
}

bool isQuantitativeScale(const ScaleType type) {
  switch (type) {
    case ScaleType::kLinear:
    case ScaleType::kLog:
    case ScaleType::kPow:
    case ScaleType::kSqrt:
      return true;
    case ScaleType::kQuantize:
    case ScaleType::kThreshold:
    case ScaleType::kOrdinal:
    case ScaleType::kUndefined:
      break;
  }
  return false;
}

bool isContinuousDomainScale(const ScaleType type) {
  switch (type) {
    case ScaleType::kLinear:
    case ScaleType::kLog:
    case ScaleType::kPow:
    case ScaleType::kSqrt:
    case ScaleType::kQuantize:
    case ScaleType::kThreshold:
      return true;
    case ScaleType::kOrdinal:
    case ScaleType::kUndefined:
      break;
  }
  return false;
}

}  // namespace QueryRenderer
