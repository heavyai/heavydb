/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"

namespace QueryRenderer {

class DistinctOp : public AggOp,
                   public AggOpResultsInterface<DistinctOp>,
                   public ThrustOpInterface {
 public:
  DistinctOp(const XformShPtr& parent_xform,
             const LayoutAttrInfo& input_info,
             const VisitedInputsSetShPtr& visited_inputs)
      : AggOp(parent_xform, input_info, visited_inputs, true) {
    validateInputs();
    validateInputBuffers();
  }
  ~DistinctOp() override = default;
  static OpType operatorType() { return OpType::kDistinct; }
  OpType getOpType() const final { return operatorType(); }

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
};

template <>
struct OpSelector<OpType::kDistinct> {
  using type = DistinctOp;
};

}  // namespace QueryRenderer
