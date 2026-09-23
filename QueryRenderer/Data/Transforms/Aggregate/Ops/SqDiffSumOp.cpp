/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/SqDiffSumOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/AvgOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"
#include "QueryRenderer/Utils/TypeUtils.h"

namespace QueryRenderer {

SQLTypeInfo SqDiffSumOp::getOutputType() const {
  return get_float_equivalent_type(AggOp::getOutputType());
}

AggDataList SqDiffSumOp::executeThrustOp(
    ThrustOpExecutor& executor,
    const InteropBufferInfo& interop_buffer_info,
    const LayoutAttrInfo& input_info,
    const DependencyOpResultsMap& dependency_results) const {
  auto avg_itr =
      dependency_results.find(dep_outputs.at(XformOp::serializeOpType(OpType::kAvg)));
  CHECK(avg_itr != dependency_results.end());
  auto const& avg_results = avg_itr->second;
  CHECK_EQ(avg_results.size(), 1u);
  CHECK(avg_results[0]);
  return executor.executeSqDiffSumOp(interop_buffer_info, input_info, *avg_results[0]);
}

const XformOp::OpResult SqDiffSumOp::executeOp(
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

AggDataList SqDiffSumOp::createEmptyData(const QueryDataType data_type) {
  return ThrustOpResultUtils::createSingularValueFromType(
      get_float_equivalent_type(data_type));
}

AggDataList SqDiffSumOp::createNullData(const QueryDataType data_type) {
  return ThrustOpResultUtils::createSingularNullFromType(
      get_float_equivalent_type(data_type));
}

void SqDiffSumOp::mergeResults(AggDataList& merged_results,
                               const AggDataList& results_to_merge) {
  CountOp::mergeResults(merged_results, results_to_merge);
}

/*********************** dependency info ***********************/
const XformOp::DependencyOutputsMap SqDiffSumOp::dep_outputs = {
    {XformOp::serializeOpType(OpType::kAvg), "avgval"}};

XformOp::DependencyOpTypeMap SqDiffSumOp::getRequiredDependencyInfo() const {
  return generateInputDependencyInfo(this, dep_outputs);
}

void SqDiffSumOp::setDependency(const XformOpShPtr& op) {
  setInputDependency(this, dep_outputs, dependent_ops_, op);
}

const XformOp::DependencyOpMap* SqDiffSumOp::getDependencyOps(
    const std::string* op_type) const {
  auto op = OpType::kMaxOpType;
  std::vector<AnyDataType> args;
  if (op_type) {
    std::tie(op, args) = XformOp::deserializeOperatorAndProps(*op_type);
  }
  CHECK_EQ(args.size(), 0u);
  if (!op_type || op == OpType::kAvg) {
    auto const& output = dep_outputs.at(XformOp::serializeOpType(OpType::kAvg));
    if (dependent_ops_.find(output) == dependent_ops_.end()) {
      CHECK(dependent_ops_
                .try_emplace(output,
                             std::make_shared<AvgOp>(
                                 parent_xform_.lock(), getInputInfo(), visited_inputs_))
                .second);
    }
  }

  return &dependent_ops_;
}

}  // namespace QueryRenderer
