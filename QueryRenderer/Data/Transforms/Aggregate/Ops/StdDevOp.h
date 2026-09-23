/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"

namespace QueryRenderer {

class StdDevOp : public AggDepOp {
 public:
  StdDevOp(const XformShPtr& parent_xform,
           const LayoutAttrInfo& input_info,
           const VisitedInputsSetShPtr& visited_inputs,
           const bool is_pop)
      : AggDepOp(parent_xform, input_info, visited_inputs, false), is_pop_{is_pop} {
    validateInputs();
    validateInputBuffers();
  }
  ~StdDevOp() override = default;
  static OpType operatorType(const bool is_pop) {
    return (is_pop ? OpType::kStdDevP : OpType::kStdDev);
  }
  OpType getOpType() const override { return operatorType(is_pop_); }
  SQLTypeInfo getOutputType() const final;
  gfx::BufferAttrType getOutputBufferAttrType() const final {
    return gfx::get_float_equivalent_type(AggOp::getOutputBufferAttrType());
  }
  DependencyOpTypeMap getRequiredDependencyInfo() const override;
  void setDependency(const XformOpShPtr& op) override;

 private:
  static const DependencyOutputsMap dep_outputs;

  const XformOp::OpResult executeOp(
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const DependencyOpResultsMap& dependency_results) override;

  const DependencyOpMap* getDependencyOps(
      const std::string* op_type = nullptr) const override;
  const DependencyOutputsMap* getDependencyOutputsMap() const override {
    return &dep_outputs;
  }

  const bool is_pop_;
};

template <>
struct OpSelector<OpType::kStdDev> {
  using type = StdDevOp;
};

template <>
struct OpSelector<OpType::kStdDevP> {
  using type = StdDevOp;
};

}  // namespace QueryRenderer
