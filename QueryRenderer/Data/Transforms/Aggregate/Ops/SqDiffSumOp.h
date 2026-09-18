/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"

namespace QueryRenderer {

class SqDiffSumOp : public AggDepOp,
                    public AggOpResultsInterface<SqDiffSumOp>,
                    public ThrustOpInterface {
 public:
  SqDiffSumOp(const XformShPtr& parent_xform,
              const LayoutAttrInfo& input_info,
              const VisitedInputsSetShPtr& visited_inputs)
      : AggDepOp(parent_xform, input_info, visited_inputs, false) {
    validateInputs();
    validateInputBuffers();
  }
  ~SqDiffSumOp() override = default;
  static OpType operatorType() { return OpType::kSQDiffSum; }
  OpType getOpType() const final { return operatorType(); }
  SQLTypeInfo getOutputType() const final;
  gfx::BufferAttrType getOutputBufferAttrType() const final {
    return gfx::get_float_equivalent_type(AggOp::getOutputBufferAttrType());
  }
  DependencyOpTypeMap getRequiredDependencyInfo() const final;
  void setDependency(const XformOpShPtr& op) final;

  static AggDataList createEmptyData(const QueryDataType data_type);
  static AggDataList createNullData(const QueryDataType data_type);
  static void mergeResults(AggDataList& merged_result,
                           const AggDataList& result_to_merge);

 private:
  static const DependencyOutputsMap dep_outputs;

  AggDataList executeThrustOp(
      ThrustOpExecutor& executor,
      const InteropBufferInfo& interop_buffer_info,
      const LayoutAttrInfo& input_info,
      const DependencyOpResultsMap& dependency_results) const final;

  const XformOp::OpResult executeOp(
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const DependencyOpResultsMap& dependency_results) final;

  const DependencyOpMap* getDependencyOps(
      const std::string* op_type = nullptr) const final;

  const DependencyOutputsMap* getDependencyOutputsMap() const final {
    return &dep_outputs;
  }
};

template <>
struct OpSelector<OpType::kSQDiffSum> {
  using type = SqDiffSumOp;
};

}  // namespace QueryRenderer
