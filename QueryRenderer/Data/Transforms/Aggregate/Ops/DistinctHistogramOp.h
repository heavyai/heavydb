/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramProps.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"

namespace QueryRenderer {

class DistinctHistogramOp : public AggOp,
                            public AggOpResultsInterface<DistinctHistogramOp>,
                            public ThrustOpInterface {
 public:
  using Props = DistinctHistogramProps;

  DistinctHistogramOp(const XformShPtr& parent_xform,
                      const LayoutAttrInfo& input_info,
                      const VisitedInputsSetShPtr& visited_inputs,
                      const bool approximate = Props::defaultApproximate(),
                      const size_t num_bins = Props::defaultNumBins())
      : AggOp(parent_xform, input_info, visited_inputs, true)
      , props_{approximate, num_bins} {
    validateInputs();
    validateInputBuffers();
  }
  ~DistinctHistogramOp() override = default;
  static OpType operatorType() { return OpType::kDistinctHistogram; }
  OpType getOpType() const final { return operatorType(); }

  static void serializeFromProps(std::stringstream& ss, const Props& props) {
    ss << serializeOpType(operatorType());
    props.serialize(ss);
  }

  static AggDataList createEmptyData(const QueryDataType data_type);
  static AggDataList createNullData(const QueryDataType data_type);
  static AggDataList flattenResults(std::vector<AggDataList>&& evaluator_results);

 private:
  void validateInputs() const final;

  AggDataList executeThrustOp(
      ThrustOpExecutor& executor,
      const InteropBufferInfo& interop_buffer_info,
      const LayoutAttrInfo& input_info,
      const DependencyOpResultsMap& dependency_results) const final;

  const XformOp::OpResult executeOp(
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const DependencyOpResultsMap& dependency_results) final;

  void serializeProps(std::stringstream& ss) const final { props_.serialize(ss); }
  static std::vector<AnyDataType> deserializeProps(std::istringstream& ss) {
    return Props::deserialize(ss);
  }
  static void serializePropsFromJSONObj(std::stringstream& ss,
                                        const JSONLocation& json_loc);

  const Props props_;
  friend XformOp;  // friending XformOp to expose the
                   // deserializeProps/serializePropsFromJSONObj static functions
};

template <>
struct OpSelector<OpType::kDistinctHistogram> {
  using type = DistinctHistogramOp;
};

}  // namespace QueryRenderer
