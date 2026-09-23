/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"

namespace QueryRenderer {

class CountOp : public AggOp,
                public AggOpResultsInterface<CountOp>,
                public ThrustOpInterface {
 public:
  CountOp(const XformShPtr& parent_xform,
          const LayoutAttrInfo& input_info,
          const VisitedInputsSetShPtr& visited_inputs)
      : AggOp(parent_xform, input_info, visited_inputs, false) {
    validateInputs();
    validateInputBuffers();
  }

  ~CountOp() override = default;

  static OpType operatorType() { return OpType::kCount; }
  OpType getOpType() const final { return operatorType(); }
  SQLTypeInfo getOutputType() const final { return SQLTypeInfo(kINT, true); }
  gfx::BufferAttrType getOutputBufferAttrType() const final {
    return gfx::BufferAttrType::kUint;
  }

  static AggDataList createEmptyData(const QueryDataType data_type);
  static AggDataList createNullData(const QueryDataType data_type);
  static void mergeResults(AggDataList& merged_result,
                           const AggDataList& result_to_merge);

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
struct OpSelector<OpType::kCount> {
  using type = CountOp;
};

}  // namespace QueryRenderer
