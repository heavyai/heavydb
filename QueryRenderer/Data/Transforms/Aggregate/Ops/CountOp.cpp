/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/ValidateUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"

namespace QueryRenderer {

/*********************** COUNT evaluation ***********************/
void CountOp::validateInputs() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  ValidateUtils::validateNumOrDictEncodedStrInput(in_data, inputs_, this);
}

AggDataList CountOp::executeThrustOp(ThrustOpExecutor& executor,
                                     const InteropBufferInfo& interop_buffer_info,
                                     const LayoutAttrInfo& input_info,
                                     const DependencyOpResultsMap&) const {
  return executor.executeCountOp(interop_buffer_info, input_info);
}

const XformOp::OpResult CountOp::executeOp(
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

AggDataList CountOp::createEmptyData(const QueryDataType data_type) {
  return {std::make_shared<AnyDataType>(QueryDataType::UINT, uint32_t(0))};
}

AggDataList CountOp::createNullData(const QueryDataType data_type) {
  return createEmptyData(data_type);
}

void CountOp::mergeResults(AggDataList& merged_results,
                           const AggDataList& results_to_merge) {
  ThrustOpResultUtils::checkSingleValueResultsForMerge(merged_results, results_to_merge);
  *(merged_results[0]) += *(results_to_merge[0]);
}

}  // namespace QueryRenderer
