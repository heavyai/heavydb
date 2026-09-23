/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXform.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"

namespace QueryRenderer {

class FormulaXform : public BaseXform {
 public:
  explicit FormulaXform(const QueryRendererContext& ctx, const BaseDataTableShPtr& data);
  ~FormulaXform() override {}

  bool hasData() const final { return true; }

  XformType getXformType() const final { return XformType::kFormula; }
  bool hasOutput(const std::string& output) const final;
  const XformOpShPtr getOutputOp(const std::string& output) const final;
  std::set<std::string> getAllOutputNames() const final;

  void initialize(const XformShPtr& ptr, const JSONLocation& obj_loc) final;

 private:
  void initFromJSONObj(const XformShPtr& ptr, const JSONLocation& obj_loc);
  void markAggOpsDirtyAfterRenderStepInternal(const std::string& evaluator_name) final {}

  XformOpShPtr my_op_;
  std::string my_output_;

  friend XformShPtr createTransform(const QueryRendererContext&,
                                    const BaseDataTableShPtr&,
                                    const JSONLocation&);
};

}  // namespace QueryRenderer
