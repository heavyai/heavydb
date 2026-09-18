/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/XformOp.h"

namespace QueryRenderer {

class FormulaXformOp : public XformDepOp {
 public:
  FormulaXformOp(const XformShPtr& parent_xform,
                 const std::string& formula_str,
                 const DependencyOpMap& dependencies);
  ~FormulaXformOp() override {}
  OpType getOpType() const final { return OpType::kFormula; }
  SQLTypeInfo getOutputType() const final;
  gfx::BufferAttrType getOutputBufferAttrType() const final;
  std::string getOpTypeAsStr() const final {
    return to_string(getOpType()) + " - \"" + formula_str_ + "\"";
  }
  DependencyOpTypeMap getRequiredDependencyInfo() const final;
  void setDependency(const XformOpShPtr& op) final;

 private:
  const std::string formula_str_;
  DependencyOpMap dependencies_;
  DependencyOutputsMap dep_defs_;
  mutable gfx::BufferAttrType cached_output_type_;

  const XformOp::OpResult executeOp(
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const DependencyOpResultsMap& dependency_results) final;
  const DependencyOpMap* getDependencyOps(
      const std::string* op_type = nullptr) const final {
    return &dependencies_;
  }
  const DependencyOutputsMap* getDependencyOutputsMap() const final { return &dep_defs_; }
};

template <>
struct OpSelector<OpType::kFormula> {
  using type = FormulaXformOp;
};

}  // namespace QueryRenderer
