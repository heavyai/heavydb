/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramProps.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/QuantileProps.h"

namespace QueryRenderer {

class QuantileOp : public AggDepOp {
 public:
  using Props = QuantileProps;

  QuantileOp(const XformShPtr& parent_xform,
             const LayoutAttrInfo& input_info,
             const VisitedInputsSetShPtr& visited_inputs,
             const uint16_t num_quantiles,
             const bool include_extrema = Props::defaultIncludeExtrema(),
             const bool approximate = DistinctHistogramProps::defaultApproximate(),
             const size_t num_bins = DistinctHistogramProps::defaultNumBins());

  QuantileOp(const XformShPtr& parent_xform,
             const LayoutAttrInfo& input_info,
             const VisitedInputsSetShPtr& visited_inputs,
             const bool approximate = DistinctHistogramProps::defaultApproximate(),
             const size_t num_bins = DistinctHistogramProps::defaultNumBins());

  QuantileOp(const XformShPtr& parent_xform,
             const LayoutAttrInfo& input_info,
             const VisitedInputsSetShPtr& visited_inputs,
             const JSONLocation& json_loc);

  ~QuantileOp() override = default;

  static OpType operatorType(const bool is_median) {
    return (is_median ? OpType::kMedian : OpType::kQuantile);
  }
  OpType getOpType() const final { return operatorType(is_median_); }

  // NOTE: quantile is a floating-pt value to properly handle the case
  // where the quantile lands in-between two values.
  SQLTypeInfo getOutputType() const final;
  gfx::BufferAttrType getOutputBufferAttrType() const final {
    return gfx::get_float_equivalent_type(AggOp::getOutputBufferAttrType());
  }

  DependencyOpTypeMap getRequiredDependencyInfo() const final;
  void setDependency(const XformOpShPtr& op) final;

  const Props& getProps() const { return props_; }

 private:
  const XformOp::OpResult executeOp(
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const DependencyOpResultsMap& dependency_results) final;

  const DependencyOpMap* getDependencyOps(
      const std::string* op_type = nullptr) const final;
  const DependencyOutputsMap* getDependencyOutputsMap() const final {
    return &dep_outputs_;
  }

  void serializeProps(std::stringstream& ss) const final { props_.serialize(ss); }
  static std::vector<AnyDataType> deserializeProps(std::istringstream& ss) {
    return Props::deserialize(ss);
  }
  static void serializePropsFromJSONObj(std::stringstream& ss,
                                        const JSONLocation& json_loc);

  const bool is_median_;
  const Props props_;
  DependencyOutputsMap dep_outputs_;

  friend XformOp;  // friending XformOp to expose the
                   // deserializeProps/serializePropsFromJSONObj static functions
};

template <>
struct OpSelector<OpType::kMedian> {
  using type = QuantileOp;
};

template <>
struct OpSelector<OpType::kQuantile> {
  using type = QuantileOp;
};

}  // namespace QueryRenderer
