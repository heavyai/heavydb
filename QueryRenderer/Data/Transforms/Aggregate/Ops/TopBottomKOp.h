/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"

namespace QueryRenderer {

class TopBottomKOp : public AggDepOp,
                     public AggOpResultsInterface<TopBottomKOp>,
                     public ThrustDependencyOpInterface {
 public:
  TopBottomKOp(const bool ascending,
               const uint16_t k,
               const XformShPtr& parent_xform,
               const LayoutAttrInfo& input_info,
               const VisitedInputsSetShPtr& visited_inputs)
      : AggDepOp(parent_xform, input_info, visited_inputs, true), asc_{ascending}, k_{k} {
    validateInputs();
    validateInputBuffers();
  }
  ~TopBottomKOp() override = default;

  static OpType operatorType(const bool asc) {
    return (asc ? OpType::kBottomK : OpType::kTopK);
  }
  OpType getOpType() const final { return operatorType(asc_); }

  DependencyOpTypeMap getRequiredDependencyInfo() const final;
  void setDependency(const XformOpShPtr& op) final;

  static AggDataList createEmptyData(const QueryDataType data_type);
  static AggDataList createNullData(const QueryDataType data_type);
  static AggDataList flattenResults(std::vector<AggDataList>&& evaluator_results);

 private:
  static const DependencyOutputsMap dep_outputs;
  const bool asc_;
  const uint16_t k_;

  void validateInputs() const final;

  AggDataList executeThrustDependencyOp(
      ThrustOpExecutor& executor,
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
struct OpSelector<OpType::kTopK> {
  using type = TopBottomKOp;
};

template <>
struct OpSelector<OpType::kBottomK> {
  using type = TopBottomKOp;
};

}  // namespace QueryRenderer
