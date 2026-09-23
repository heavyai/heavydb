/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"

namespace QueryRenderer {

class VarianceOp : public AggDepOp {
 public:
  VarianceOp(const XformShPtr& parent_xform,
             const LayoutAttrInfo& input_info,
             const VisitedInputsSetShPtr& visited_inputs,
             const bool is_pop)
      : AggDepOp(parent_xform, input_info, visited_inputs, false)
      , dep_outputs{buildDependencyOutputs(input_info)}
      , is_pop_{is_pop} {
    validateInputs();
    validateInputBuffers();
  }
  ~VarianceOp() override = default;
  static OpType operatorType(const bool is_pop) {
    return (is_pop ? OpType::kVarianceP : OpType::kVariance);
  }
  OpType getOpType() const override { return operatorType(is_pop_); }
  SQLTypeInfo getOutputType() const final;
  gfx::BufferAttrType getOutputBufferAttrType() const final {
    return gfx::get_float_equivalent_type(AggOp::getOutputBufferAttrType());
  }
  DependencyOpTypeMap getRequiredDependencyInfo() const final;
  void setDependency(const XformOpShPtr& op) final;
  bool isPopulation() const { return is_pop_; }

 protected:
  static DependencyOutputsMap buildDependencyOutputs(const LayoutAttrInfo& input_info);
  const DependencyOutputsMap dep_outputs;

 private:
  const XformOp::OpResult executeOp(
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const DependencyOpResultsMap& dependency_results) override;

  const DependencyOpMap* getDependencyOps(
      const std::string* op_type = nullptr) const final;
  const DependencyOutputsMap* getDependencyOutputsMap() const final {
    return &dep_outputs;
  }

  const bool is_pop_;
};

template <>
struct OpSelector<OpType::kVariance> {
  using type = VarianceOp;
};

template <>
struct OpSelector<OpType::kVarianceP> {
  using type = VarianceOp;
};

}  // namespace QueryRenderer
