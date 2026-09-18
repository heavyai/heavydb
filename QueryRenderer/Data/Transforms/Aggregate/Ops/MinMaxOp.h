/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"

namespace QueryRenderer {

class MinOp : public AggOp,
              public AggOpResultsInterface<MinOp>,
              public ThrustOpInterface {
 public:
  MinOp(const XformShPtr& parent_xform,
        const LayoutAttrInfo& input_info,
        const VisitedInputsSetShPtr& visited_inputs)
      : AggOp(parent_xform, input_info, visited_inputs, false) {
    validateInputs();
    validateInputBuffers();
  }
  ~MinOp() override = default;
  static OpType operatorType() { return OpType::kMin; }
  OpType getOpType() const final { return operatorType(); }

  static AggDataList createEmptyData(const QueryDataType data_type);
  static AggDataList createNullData(const QueryDataType data_type);
  static void mergeResults(AggDataList& merged_result,
                           const AggDataList& result_to_merge);

 private:
  AggDataList executeThrustOp(
      ThrustOpExecutor& executor,
      const InteropBufferInfo& interop_buffer_info,
      const LayoutAttrInfo& input_info,
      const DependencyOpResultsMap& dependency_results) const final;

  const XformOp::OpResult executeOp(
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const DependencyOpResultsMap& dependency_results) final;
};

template <>
struct OpSelector<OpType::kMin> {
  using type = MinOp;
};

class MaxOp : public AggOp,
              public AggOpResultsInterface<MaxOp>,
              public ThrustOpInterface {
 public:
  MaxOp(const XformShPtr& parent_xform,
        const LayoutAttrInfo& input_info,
        const VisitedInputsSetShPtr& visited_inputs)
      : AggOp(parent_xform, input_info, visited_inputs, false) {
    validateInputs();
    validateInputBuffers();
  }
  ~MaxOp() override = default;
  static OpType operatorType() { return OpType::kMax; }
  OpType getOpType() const final { return operatorType(); }

  static AggDataList createNullData(const QueryDataType data_type);
  static AggDataList createEmptyData(const QueryDataType data_type);
  static void mergeResults(AggDataList& merged_result,
                           const AggDataList& result_to_merge);

 private:
  AggDataList executeThrustOp(
      ThrustOpExecutor& executor,
      const InteropBufferInfo& interop_buffer_info,
      const LayoutAttrInfo& input_info,
      const DependencyOpResultsMap& dependency_results) const final;

  const XformOp::OpResult executeOp(
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const DependencyOpResultsMap& dependency_results) final;
};

template <>
struct OpSelector<OpType::kMax> {
  using type = MaxOp;
};

}  // namespace QueryRenderer
