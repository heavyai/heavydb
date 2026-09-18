/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/MinMaxOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"

namespace QueryRenderer {

AggDataList MinOp::executeThrustOp(ThrustOpExecutor& executor,
                                   const InteropBufferInfo& interop_buffer_info,
                                   const LayoutAttrInfo& input_info,
                                   const DependencyOpResultsMap&) const {
  return executor.executeMinOp(interop_buffer_info, input_info);
}

const XformOp::OpResult MinOp::executeOp(
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

void MinOp::mergeResults(AggDataList& merged_results,
                         const AggDataList& results_to_merge) {
  ThrustOpResultUtils::checkSingleValueResultsForMerge(merged_results, results_to_merge);
  *(merged_results[0]) =
      std::min(*(merged_results[0]), *(results_to_merge[0]), AnyDataType::minCompare);
}

AggDataList MinOp::createEmptyData(const QueryDataType data_type) {
  return createNullData(data_type);
}

AggDataList MinOp::createNullData(const QueryDataType data_type) {
  return ThrustOpResultUtils::createSingularNullFromType(data_type);
}

AggDataList MaxOp::executeThrustOp(ThrustOpExecutor& executor,
                                   const InteropBufferInfo& interop_buffer_info,
                                   const LayoutAttrInfo& input_info,
                                   const DependencyOpResultsMap&) const {
  return executor.executeMaxOp(interop_buffer_info, input_info);
}

const XformOp::OpResult MaxOp::executeOp(
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

AggDataList MaxOp::createEmptyData(const QueryDataType data_type) {
  return MinOp::createEmptyData(data_type);
}

AggDataList MaxOp::createNullData(const QueryDataType data_type) {
  return MinOp::createNullData(data_type);
}

void MaxOp::mergeResults(AggDataList& merged_results,
                         const AggDataList& results_to_merge) {
  ThrustOpResultUtils::checkSingleValueResultsForMerge(merged_results, results_to_merge);
  *(merged_results[0]) =
      std::max(*(merged_results[0]), *(results_to_merge[0]), AnyDataType::maxCompare);
}

}  // namespace QueryRenderer
