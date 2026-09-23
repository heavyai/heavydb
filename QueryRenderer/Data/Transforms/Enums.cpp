/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Enums.h"

#include "GfxDriver/RenderError.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

std::string to_string(const XformType value) {
  switch (value) {
    case XformType::kAggregate:
      return "AGGREGATE";
    case XformType::kFormula:
      return "FORMULA";
    case XformType::kMaxXformType:
      CHECK(false);
  }

  return "";
}

int convert_string_to_xform_type_enum(const std::string& val) {
  auto uppercase = makeUpperCase(val);

  if (uppercase == to_string(XformType::kAggregate)) {
    return static_cast<int>(XformType::kAggregate);
  } else if (uppercase == to_string(XformType::kFormula)) {
    return static_cast<int>(XformType::kFormula);
  }

  return -1;
}

std::string get_xform_types_as_string() {
  return enum_to_string<XformType>(static_cast<XformType>(0),
                                   XformType::kMaxXformType,
                                   static_cast<std::string (*)(XformType)>(&to_string));
}

std::string to_string(const OpType value) {
  switch (value) {
    case OpType::kCount:
      return "COUNT";
    case OpType::kCountValid:
      return "VALID";
    case OpType::kCountMissing:
      return "MISSING";
    case OpType::kMin:
      return "MIN";
    case OpType::kMax:
      return "MAX";
    case OpType::kSum:
      return "SUM";
    case OpType::kAvg:
      return "AVG";
    case OpType::kVariance:
      return "VARIANCE";
    case OpType::kVarianceP:
      return "VARIANCEP";
    case OpType::kStdDev:
      return "STDDEV";
    case OpType::kStdDevP:
      return "STDDEVP";
    case OpType::kFormula:
      return "FORMULA";
    case OpType::kDistinct:
      return "DISTINCT";
    case OpType::kCountDistinct:
      return "COUNTDISTINCT";
    case OpType::kMedian:
      return "MEDIAN";
    case OpType::kQuantile:
      return "QUANTILE";
    case OpType::kSQDiffSum:
      return "SQDIFFSUM";
    case OpType::kDistinctHistogram:
      return "DISTINCT_HISTOGRAM";
    case OpType::kTopK:
      return "TOPK";
    case OpType::kBottomK:
      return "BOTTOMK";
    case OpType::kNonClientFacingSeparator:
    case OpType::kMaxOpType:
      CHECK(false);
  }

  return "";
}

int convert_string_to_op_type_enum(const std::string& val) {
  auto uppercase = makeUpperCase(val);

  if (uppercase == to_string(OpType::kCount)) {
    return static_cast<int>(OpType::kCount);
  } else if (uppercase == to_string(OpType::kCountValid)) {
    return static_cast<int>(OpType::kCountValid);
  } else if (uppercase == to_string(OpType::kCountMissing)) {
    return static_cast<int>(OpType::kCountMissing);
  } else if (uppercase == to_string(OpType::kMin)) {
    return static_cast<int>(OpType::kMin);
  } else if (uppercase == to_string(OpType::kMax)) {
    return static_cast<int>(OpType::kMax);
  } else if (uppercase == to_string(OpType::kSum)) {
    return static_cast<int>(OpType::kSum);
  } else if (uppercase == to_string(OpType::kAvg)) {
    return static_cast<int>(OpType::kAvg);
  } else if (uppercase == "MEAN") {
    return static_cast<int>(OpType::kMean);
  } else if (uppercase == "AVERAGE") {
    return static_cast<int>(OpType::kAverage);
  } else if (uppercase == to_string(OpType::kVariance)) {
    return static_cast<int>(OpType::kVariance);
  } else if (uppercase == to_string(OpType::kVarianceP)) {
    return static_cast<int>(OpType::kVarianceP);
  } else if (uppercase == to_string(OpType::kStdDev)) {
    return static_cast<int>(OpType::kStdDev);
  } else if (uppercase == to_string(OpType::kStdDevP)) {
    return static_cast<int>(OpType::kStdDevP);
  } else if (uppercase == to_string(OpType::kFormula)) {
    return static_cast<int>(OpType::kFormula);
  } else if (uppercase == to_string(OpType::kDistinct)) {
    return static_cast<int>(OpType::kDistinct);
  } else if (uppercase == to_string(OpType::kCountDistinct)) {
    return static_cast<int>(OpType::kCountDistinct);
  } else if (uppercase == to_string(OpType::kMedian)) {
    return static_cast<int>(OpType::kMedian);
  } else if (uppercase == to_string(OpType::kQuantile)) {
    return static_cast<int>(OpType::kQuantile);
  }

  return -1;
}

bool is_agg_op_type(const OpType op_type) {
  switch (op_type) {
    case OpType::kCount:
    case OpType::kCountValid:
    case OpType::kCountMissing:
    case OpType::kMin:
    case OpType::kMax:
    case OpType::kSum:
    case OpType::kAvg:
    case OpType::kVariance:
    case OpType::kVarianceP:
    case OpType::kStdDev:
    case OpType::kStdDevP:
    case OpType::kDistinct:
    case OpType::kCountDistinct:
    case OpType::kMedian:
    case OpType::kQuantile:
    case OpType::kTopK:
    case OpType::kBottomK:
    case OpType::kSQDiffSum:
    case OpType::kDistinctHistogram:
      return true;
    case OpType::kNonClientFacingSeparator:
    case OpType::kFormula:
    case OpType::kMaxOpType:
      return false;
  }
  return false;
}

bool is_client_facing_op_type(const OpType op_type) {
  return op_type < OpType::kNonClientFacingSeparator;
}

std::string get_agg_ops_as_string() {
  return enum_to_string<OpType>(static_cast<OpType>(0),
                                OpType::kMaxOpType,
                                static_cast<std::string (*)(OpType)>(&to_string),
                                [](const OpType val) {
                                  return is_client_facing_op_type(val) &&
                                         is_agg_op_type(val);
                                });
}

std::ostream& operator<<(std::ostream& os, const XformType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const OpType value) {
  os << to_string(value);
  return os;
}

}  // namespace QueryRenderer
