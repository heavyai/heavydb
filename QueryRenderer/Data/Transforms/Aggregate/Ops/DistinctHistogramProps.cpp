/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramProps.h"

#include "QueryRenderer/Data/Transforms/Utils.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

decltype(DistinctHistogramProps::approximate)
DistinctHistogramProps::getApproximateFromJSONObj(const JSONLocation& json_loc) {
  CHECK(json_loc.isObject());

  std::remove_const<decltype(num_bins)>::type json_approx = defaultApproximate();
  auto const approx_loc = json_loc.getMember(JSONSchema_v1::Xform::kApproximateProp);
  if (approx_loc.isValid()) {
    RUNTIME_EX_ASSERT(approx_loc.isBool(),
                      RapidJSONUtils::createJsonParseError(
                          approx_loc,
                          "The \"" + std::string(JSONSchema_v1::Xform::kApproximateProp) +
                              "\" property of the aggregator operator " +
                              RapidJSONUtils::getObjAsString(json_loc.getValueRef()) +
                              " must be a boolean."));
    json_approx = approx_loc.getBool();
  }

  return json_approx;
}

decltype(DistinctHistogramProps::num_bins) DistinctHistogramProps::getNumBinsFromJSONObj(
    const JSONLocation& json_loc) {
  CHECK(json_loc.isObject());

  std::remove_const<decltype(num_bins)>::type json_num_bins = defaultNumBins();
  auto const num_bins_loc = json_loc.getMember(JSONSchema_v1::Xform::kNumBinsProp);
  if (num_bins_loc.isValid()) {
    RUNTIME_EX_ASSERT(num_bins_loc.isUint(),
                      RapidJSONUtils::createJsonParseError(
                          num_bins_loc,
                          "The \"" + std::string(JSONSchema_v1::Xform::kNumBinsProp) +
                              "\" aggregator property must be an integer."));

    json_num_bins = num_bins_loc.getUint();
    RUNTIME_EX_ASSERT(json_num_bins >= 1 && json_num_bins <= 10000,
                      RapidJSONUtils::createJsonParseError(
                          num_bins_loc,
                          "The \"" + std::string(JSONSchema_v1::Xform::kNumBinsProp) +
                              "\" aggregator property must be between 1 and 10000."));
  }

  return json_num_bins;
}

DistinctHistogramProps::DistinctHistogramProps(const bool in_approximate,
                                               const size_t in_num_bins)
    : approximate(in_approximate), num_bins(in_num_bins) {
  validate();
}

DistinctHistogramProps::DistinctHistogramProps(const JSONLocation& json_loc)
    : DistinctHistogramProps(getApproximateFromJSONObj(json_loc),
                             getNumBinsFromJSONObj(json_loc)) {}

void DistinctHistogramProps::validate() {
  if (num_bins == 0 || num_bins > 10000) {
    throw std::runtime_error("num bins must be in the range 1-10000");
  }
}

DistinctHistogramProps::DistinctHistogramProps(
    const std::vector<AnyDataType>& deserialized_props)
    : DistinctHistogramProps() {
  CHECK_EQ(deserialized_props.size(), 2u);
  CHECK(deserialized_props[0].getType() == QueryDataType::BOOL);
  CHECK(deserialized_props[1].getType() == QueryDataType::UINT);
  const_cast<bool&>(approximate) = deserialized_props[0].getVal<bool>();
  const_cast<size_t&>(num_bins) = deserialized_props[1].getVal<size_t>();
}

void DistinctHistogramProps::serialize(std::stringstream& ss) const {
  ss << " " << approximate << " " << num_bins;
}

std::vector<AnyDataType> DistinctHistogramProps::deserialize(std::istringstream& ss) {
  std::remove_const<decltype(approximate)>::type approximate;
  std::remove_const<decltype(num_bins)>::type num_bins;
  ss >> approximate;
  ss >> num_bins;
  return {AnyDataType(QueryDataType::BOOL, approximate),
          AnyDataType(QueryDataType::UINT, static_cast<unsigned int>(num_bins))};
}

}  // namespace QueryRenderer
