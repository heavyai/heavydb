/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/QuantileProps.h"

#include "QueryRenderer/Data/Transforms/Enums.h"
#include "QueryRenderer/Data/Transforms/Utils.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

QuantileProps::QuantileProps(const uint16_t in_num_quantiles,
                             const bool in_include_extrema,
                             const bool in_approximate,
                             const size_t in_num_bins)
    : DistinctHistogramProps(in_approximate, in_num_bins)
    , num_quantiles{in_num_quantiles}
    , include_extrema{in_include_extrema} {
  validate();
}

QuantileProps::QuantileProps(const JSONLocation& json_loc)
    : DistinctHistogramProps(json_loc)
    , num_quantiles{getNumQuantilesFromJSONObj(json_loc)}
    , include_extrema{getIncludeExtremaFromJSONObj(json_loc)} {
  validate();
}

void QuantileProps::validate() {
  if (num_quantiles == 0 || num_quantiles > 1000) {
    throw std::runtime_error("num quantiles must be in the range 1-1000");
  }
}

decltype(QuantileProps::num_quantiles) QuantileProps::getNumQuantilesFromJSONObj(
    const JSONLocation& json_loc) {
  CHECK(json_loc.isObject());

  auto const op_type_loc = json_loc.getMember(JSONSchema_v1::Xform::kTypeProp);
  CHECK(op_type_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
  CHECK(op_type_loc.isString());
  auto const which_op =
      static_cast<OpType>(convert_string_to_op_type_enum(op_type_loc.getString()));
  CHECK(which_op == OpType::kQuantile || which_op == OpType::kMedian) << which_op;

  unsigned int num_quantiles{2};
  auto const num_quant_loc = json_loc.getMember(JSONSchema_v1::Xform::kNumQuantilesProp);
  if (which_op == OpType::kMedian) {
    RUNTIME_EX_ASSERT(
        !num_quant_loc.isValid(),
        RapidJSONUtils::createJsonParseError(
            num_quant_loc,
            "The median aggregator operator " +
                RapidJSONUtils::getObjAsString(json_loc.getValueRef()) +
                " must not contain a \"" +
                std::string(JSONSchema_v1::Xform::kNumQuantilesProp) +
                "\" property. That is reserved for quantile operators. The number of "
                "quantiles for a median is always 2, so there's no need to declare \"" +
                std::string(JSONSchema_v1::Xform::kNumQuantilesProp) + "\"."));
  } else {
    RUNTIME_EX_ASSERT(
        num_quant_loc.isValid() && num_quant_loc.isUint() &&
            (num_quantiles = num_quant_loc.getUint()) >= 1 && num_quantiles <= 1000,
        RapidJSONUtils::createJsonParseError(
            num_quant_loc,
            "The quantile aggregator operator " +
                RapidJSONUtils::getObjAsString(json_loc.getValueRef()) +
                " must contain a \"" +
                std::string(JSONSchema_v1::Xform::kNumQuantilesProp) +
                "\" property and it must be an integer between 1 and 1000."));
  }

  return static_cast<uint16_t>(num_quantiles);
}

decltype(QuantileProps::include_extrema) QuantileProps::getIncludeExtremaFromJSONObj(
    const JSONLocation& json_loc) {
  CHECK(json_loc.isObject());

  auto const optype_loc = json_loc.getMember(JSONSchema_v1::Xform::kTypeProp);
  CHECK(optype_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
  CHECK(optype_loc.isString()) << RapidJSONUtils::getPointerPath(optype_loc.getPathRef());
  auto const which_op =
      static_cast<OpType>(convert_string_to_op_type_enum(optype_loc.getString()));
  CHECK(which_op == OpType::kQuantile || which_op == OpType::kMedian) << which_op;

  bool include_extrema = defaultIncludeExtrema();
  auto const inc_extrema_loc =
      json_loc.getMember(JSONSchema_v1::Xform::kIncludeExtremaProp);
  if (inc_extrema_loc.isValid()) {
    RUNTIME_EX_ASSERT(
        which_op != OpType::kMedian,
        RapidJSONUtils::createJsonParseError(
            inc_extrema_loc,
            "The \"" + std::string(JSONSchema_v1::Xform::kIncludeExtremaProp) +
                "\" property for the median aggregator operator is "
                "reserverd for quantile operators. For median operators, the extrema "
                "is "
                "not calculated."));

    RUNTIME_EX_ASSERT(
        inc_extrema_loc.isBool(),
        RapidJSONUtils::createJsonParseError(
            inc_extrema_loc,
            "The \"" + std::string(JSONSchema_v1::Xform::kIncludeExtremaProp) +
                "\" property for the quantile aggregator operator " +
                RapidJSONUtils::getObjAsString(json_loc.getValueRef()) +
                " must be a boolean."));
    include_extrema = inc_extrema_loc.getBool();
  }
  return include_extrema;
}

void QuantileProps::serialize(std::stringstream& ss) const {
  DistinctHistogramProps::serialize(ss);
  ss << " " << num_quantiles << " " << include_extrema;
}

std::vector<AnyDataType> QuantileProps::deserialize(std::istringstream& ss) {
  DistinctHistogramProps::deserialize(ss);
  decltype(num_quantiles) num_quantiles;
  decltype(include_extrema) include_extrema;
  ss >> num_quantiles;
  ss >> include_extrema;
  return {AnyDataType(QueryDataType::UINT, static_cast<unsigned int>(num_quantiles)),
          AnyDataType(QueryDataType::BOOL, include_extrema)};
}

}  // namespace QueryRenderer
