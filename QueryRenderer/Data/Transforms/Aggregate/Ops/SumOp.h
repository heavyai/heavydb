/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"

namespace QueryRenderer {

class SumOp : public AggOp,
              public AggOpResultsInterface<SumOp>,
              public ThrustOpInterface {
 public:
  SumOp(const XformShPtr& parent_xform,
        const LayoutAttrInfo& input_info,
        const VisitedInputsSetShPtr& visited_inputs)
      : AggOp(parent_xform, input_info, visited_inputs, false) {
    validateInputs();
    validateInputBuffers();
  }
  ~SumOp() override = default;
  static OpType operatorType() { return OpType::kSum; }
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
struct OpSelector<OpType::kSum> {
  using type = SumOp;
};

}  // namespace QueryRenderer
