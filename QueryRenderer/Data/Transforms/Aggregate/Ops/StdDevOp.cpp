/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/StdDevOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/AggError.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/VarianceOp.h"
#include "QueryRenderer/Utils/TypeUtils.h"

namespace QueryRenderer {

namespace {

AggDataList execute_stddev_op(const StdDevOp& parent_xform_op,
                              const XformOp::DependencyOpResultsMap& dependency_results,
                              const std::string& variance_attr,
                              const bool is_pop) {
  AggDataList results;

  auto variance_itr = dependency_results.find(variance_attr);
  CHECK(variance_itr != dependency_results.end());

  auto const& var_result = variance_itr->second;
  CHECK_EQ(var_result.size(), 1u);
  CHECK(var_result[0]);

  switch (var_result[0]->getType()) {
    case QueryDataType::FLOAT: {
      float var = var_result[0]->getVal<float>();
      if (isNullValue(var)) {
        results = {std::make_shared<AnyDataType>(QueryDataType::FLOAT, var)};
        break;
      }
      float val = 0.0f;
      LOG_IF(WARNING, var < 0.0)
          << "Attempting to calculate a stddev with a variance < 0 for operator "
          << std::string(parent_xform_op);
      if (var > 0) {
        val = sqrt(var);
      }
      results = {std::make_shared<AnyDataType>(QueryDataType::FLOAT, val)};
      break;
    }
    case QueryDataType::DOUBLE: {
      double var = var_result[0]->getVal<double>();
      if (isNullValue(var)) {
        results = {std::make_shared<AnyDataType>(QueryDataType::DOUBLE, var)};
        break;
      }
      double val = 0.0f;
      LOG_IF(WARNING, var < 0.0)
          << "Attempting to calculate a stddev with a variance < 0 for operator "
          << std::string(parent_xform_op);
      if (var > 0) {
        val = sqrt(var);
      }
      results = {std::make_shared<AnyDataType>(QueryDataType::DOUBLE, val)};
      break;
    }
    default:
      throw std::runtime_error("Data of type " + to_string(var_result[0]->getType()) +
                               ". Only floats & doubles are currently supported for "
                               "calculating standard deviation.");
  }

  return results;
}

}  // namespace

SQLTypeInfo StdDevOp::getOutputType() const {
  return get_float_equivalent_type(AggOp::getOutputType());
}

const XformOp::OpResult StdDevOp::executeOp(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const DependencyOpResultsMap& dependency_results) {
  try {
    return {true,
            execute_stddev_op(*this,
                              dependency_results,
                              dep_outputs.at(XformOp::serializeOpType(OpType::kVariance)),
                              is_pop_)};
  } catch (...) {
    LOG_AGG_INFO_AND_THROW(*this);
  }
  return {true, {}};
}

/*********************** dependency info ***********************/
const XformOp::DependencyOutputsMap StdDevOp::dep_outputs = {
    {XformOp::serializeOpType(OpType::kVariance), "varval"}};

XformOp::DependencyOpTypeMap StdDevOp::getRequiredDependencyInfo() const {
  return generateInputDependencyInfo(this, dep_outputs);
}

void StdDevOp::setDependency(const XformOpShPtr& op) {
  setInputDependency(this, dep_outputs, dependent_ops_, op);
}

const XformOp::DependencyOpMap* StdDevOp::getDependencyOps(
    const std::string* op_type) const {
  auto op = OpType::kMaxOpType;
  std::vector<AnyDataType> args;
  if (op_type) {
    std::tie(op, args) = XformOp::deserializeOperatorAndProps(*op_type);
  }
  CHECK_EQ(args.size(), 0u);
  if (!op_type || op == OpType::kVariance) {
    auto const& output = dep_outputs.at(XformOp::serializeOpType(OpType::kVariance));
    if (dependent_ops_.find(output) == dependent_ops_.end()) {
      CHECK(dependent_ops_
                .try_emplace(
                    output,
                    std::make_shared<VarianceOp>(
                        parent_xform_.lock(), getInputInfo(), visited_inputs_, is_pop_))
                .second);
    }
  }
  return &dependent_ops_;
}

}  // namespace QueryRenderer
