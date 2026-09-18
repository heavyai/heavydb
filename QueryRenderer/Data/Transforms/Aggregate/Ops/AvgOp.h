/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"

namespace QueryRenderer {

class AvgOp : public AggDepOp {
 public:
  AvgOp(const XformShPtr& parent_xform,
        const LayoutAttrInfo& input_info,
        const VisitedInputsSetShPtr& visited_inputs)
      : AggDepOp(parent_xform, input_info, visited_inputs, false)
      , dep_outputs(buildDependencyOutputs(input_info)) {
    validateInputs();
    validateInputBuffers();
  }
  ~AvgOp() override = default;
  static OpType operatorType() { return OpType::kAvg; }
  OpType getOpType() const final { return operatorType(); }
  SQLTypeInfo getOutputType() const final;
  gfx::BufferAttrType getOutputBufferAttrType() const final {
    return gfx::get_float_equivalent_type(AggOp::getOutputBufferAttrType());
  }
  DependencyOpTypeMap getRequiredDependencyInfo() const final;
  void setDependency(const XformOpShPtr& op) final;

 private:
  static DependencyOutputsMap buildDependencyOutputs(const LayoutAttrInfo& input_info);
  const DependencyOutputsMap dep_outputs;

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
struct OpSelector<OpType::kAvg> {
  using type = AvgOp;
};

}  // namespace QueryRenderer
