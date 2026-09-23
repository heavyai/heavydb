/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountDistinctOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/AggError.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/ValidateUtils.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"

namespace QueryRenderer {

namespace {

AggDataList execute_count_distinct_op(
    const XformOp::DependencyOpResultsMap& dependency_results,
    const std::string& distinct_attr) {
  auto distinct_itr = dependency_results.find(distinct_attr);
  CHECK(distinct_itr != dependency_results.end());

  // NOTE: not checking that varresult are defined
  // because the getResult() call does that
  auto const& distinct_result = distinct_itr->second;
  CHECK_EQ(distinct_result.size(), 1u);
  CHECK(distinct_result[0]);
  CHECK(distinct_result[0]->isVector());

  return {std::make_shared<AnyDataType>(
      QueryDataType::UINT, static_cast<unsigned int>(distinct_result[0]->size()))};
}

}  // namespace

void CountDistinctOp::validateInputs() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  ValidateUtils::validateNumOrDictEncodedStrInput(in_data, inputs_, this);
}

const XformOp::OpResult CountDistinctOp::executeOp(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const DependencyOpResultsMap& dependency_results) {
  try {
    return {true,
            execute_count_distinct_op(
                dependency_results,
                dep_outputs.at(XformOp::serializeOpType(OpType::kDistinct)))};
  } catch (...) {
    LOG_AGG_INFO_AND_THROW(*this);
  }
  return {true, {}};
}

/******************** dependency info ********************/
const XformOp::DependencyOutputsMap CountDistinctOp::dep_outputs = {
    {XformOp::serializeOpType(OpType::kDistinct), "distinctval"}};

XformOp::DependencyOpTypeMap CountDistinctOp::getRequiredDependencyInfo() const {
  return generateInputDependencyInfo(this, dep_outputs);
}

void CountDistinctOp::setDependency(const XformOpShPtr& op) {
  setInputDependency(this, dep_outputs, dependent_ops_, op);
}

const XformOp::DependencyOpMap* CountDistinctOp::getDependencyOps(
    const std::string* op_type) const {
  auto op = OpType::kMaxOpType;
  std::vector<AnyDataType> args;
  if (op_type) {
    std::tie(op, args) = XformOp::deserializeOperatorAndProps(*op_type);
  }
  CHECK_EQ(args.size(), 0u);
  if (!op_type || op == OpType::kDistinct) {
    auto const& output = dep_outputs.at(XformOp::serializeOpType(OpType::kDistinct));
    if (dependent_ops_.find(output) == dependent_ops_.end()) {
      CHECK(dependent_ops_
                .try_emplace(output,
                             std::make_shared<DistinctOp>(
                                 parent_xform_.lock(), getInputInfo(), visited_inputs_))
                .second);
    }
  }
  return &dependent_ops_;
}

}  // namespace QueryRenderer
