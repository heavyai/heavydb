/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/TopBottomKOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/ValidateUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"

namespace QueryRenderer {

void TopBottomKOp::validateInputs() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  ValidateUtils::validateNumOrDictEncodedStrInput(in_data, inputs_, this);
}

AggDataList TopBottomKOp::executeThrustDependencyOp(
    ThrustOpExecutor& executor,
    const DependencyOpResultsMap& dependency_results) const {
  auto distinct_hist_itr = dependency_results.find(
      dep_outputs.at(XformOp::serializeOpType(OpType::kDistinctHistogram)));
  CHECK(distinct_hist_itr != dependency_results.end());

  return executor.executeTopBottomKOp(distinct_hist_itr->second, asc_, k_);
}

const XformOp::OpResult TopBottomKOp::executeOp(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const DependencyOpResultsMap& dependency_results) {
  return OpExecuteUtils::executeThrustDependencyOp(*this,
                                                   *this,
                                                   parent_xform_.lock(),
                                                   inputs_,
                                                   getDataMgr(),
                                                   getRenderContextNonConst(),
                                                   evaluator_name,
                                                   dependency_results);
}

AggDataList TopBottomKOp::createEmptyData(const QueryDataType data_type) {
  return ThrustOpResultUtils::createEmptyVectorType(data_type);
}

AggDataList TopBottomKOp::createNullData(const QueryDataType data_type) {
  return createEmptyData(data_type);
}

AggDataList TopBottomKOp::flattenResults(std::vector<AggDataList>&&) {
  THROW_RUNTIME_EX("TopK/BottomK merging is to be implemented.");
  return {};
}

/********************* dependency info ********************/
const XformOp::DependencyOutputsMap TopBottomKOp::dep_outputs = {
    {XformOp::serializeOpType(OpType::kDistinctHistogram), "distincthistogramval"}};

XformOp::DependencyOpTypeMap TopBottomKOp::getRequiredDependencyInfo() const {
  return generateInputDependencyInfo(this, dep_outputs);
}

void TopBottomKOp::setDependency(const XformOpShPtr& op) {
  setInputDependency(this, dep_outputs, dependent_ops_, op);
}

const XformOp::DependencyOpMap* TopBottomKOp::getDependencyOps(
    const std::string* op_type) const {
  auto op = OpType::kMaxOpType;
  std::vector<AnyDataType> args;
  if (op_type) {
    std::tie(op, args) = XformOp::deserializeOperatorAndProps(*op_type);
  }
  CHECK_EQ(args.size(), 0u);
  if (!op_type || op == OpType::kDistinctHistogram) {
    auto const& output =
        dep_outputs.at(XformOp::serializeOpType(OpType::kDistinctHistogram));
    if (dependent_ops_.find(output) == dependent_ops_.end()) {
      CHECK(dependent_ops_
                .try_emplace(
                    output,
                    std::make_shared<DistinctHistogramOp>(
                        parent_xform_.lock(), getInputInfo(), visited_inputs_, false))
                .second);
    }
  }
  return &dependent_ops_;
}

}  // namespace QueryRenderer
