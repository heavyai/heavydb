/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/SumOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"

namespace QueryRenderer {

AggDataList SumOp::executeThrustOp(ThrustOpExecutor& executor,
                                   const InteropBufferInfo& interop_buffer_info,
                                   const LayoutAttrInfo& input_info,
                                   const DependencyOpResultsMap&) const {
  return executor.executeSumOp(interop_buffer_info, input_info);
}

const XformOp::OpResult SumOp::executeOp(
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

AggDataList SumOp::createEmptyData(const QueryDataType data_type) {
  return ThrustOpResultUtils::createSingularValueFromType(data_type);
}

AggDataList SumOp::createNullData(const QueryDataType data_type) {
  return ThrustOpResultUtils::createSingularNullFromType(data_type);
}

void SumOp::mergeResults(AggDataList& merged_results,
                         const AggDataList& results_to_merge) {
  CountOp::mergeResults(merged_results, results_to_merge);
}

}  // namespace QueryRenderer
