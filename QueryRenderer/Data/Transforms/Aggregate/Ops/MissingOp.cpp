/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/MissingOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/ValidateUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"

namespace QueryRenderer {

void MissingOp::validateInputs() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  ValidateUtils::validateNumOrDictEncodedStrInput(in_data, inputs_, this);
}

AggDataList MissingOp::executeThrustOp(ThrustOpExecutor& executor,
                                       const InteropBufferInfo& interop_buffer_info,
                                       const LayoutAttrInfo& input_info,
                                       const DependencyOpResultsMap&) const {
  return executor.executeMissingOp(interop_buffer_info, input_info);
}

const XformOp::OpResult MissingOp::executeOp(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const DependencyOpResultsMap& dependency_results) {
  return OpExecuteUtils::executeThrustOp(*this,
                                         *this,
                                         parent_xform_.lock(),
                                         inputs_,
                                         getDataMgr(),
                                         getRenderContextNonConst(),
                                         evaluator_name,
                                         mapped_buffers,
                                         dependency_results);
}

AggDataList MissingOp::createEmptyData(const QueryDataType data_type) {
  return CountOp::createEmptyData(data_type);
}

AggDataList MissingOp::createNullData(const QueryDataType data_type) {
  return CountOp::createNullData(data_type);
}

void MissingOp::mergeResults(AggDataList& merged_results,
                             const AggDataList& results_to_merge) {
  CountOp::mergeResults(merged_results, results_to_merge);
}

}  // namespace QueryRenderer
