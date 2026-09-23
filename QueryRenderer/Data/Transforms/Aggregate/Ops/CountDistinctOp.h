/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"

namespace QueryRenderer {

class CountDistinctOp : public AggDepOp {
 public:
  CountDistinctOp(const XformShPtr& parent_xform,
                  const LayoutAttrInfo& input_info,
                  const VisitedInputsSetShPtr& visited_inputs)
      : AggDepOp(parent_xform, input_info, visited_inputs, false) {
    validateInputs();
    validateInputBuffers();
  }
  ~CountDistinctOp() override = default;
  static OpType operatorType() { return OpType::kCountDistinct; }

  OpType getOpType() const final { return operatorType(); }
  SQLTypeInfo getOutputType() const final { return SQLTypeInfo(kINT, true); }
  gfx::BufferAttrType getOutputBufferAttrType() const final {
    return gfx::BufferAttrType::kUint;
  }
  DependencyOpTypeMap getRequiredDependencyInfo() const final;
  void setDependency(const XformOpShPtr& op) final;

 private:
  static const DependencyOutputsMap dep_outputs;

  void validateInputs() const final;
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
struct OpSelector<OpType::kCountDistinct> {
  using type = CountDistinctOp;
};

}  // namespace QueryRenderer
