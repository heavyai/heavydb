/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/AvgOp.h"

#include "QueryRenderer/Data/Transforms/Aggregate/AggError.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/SumOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/ValidOp.h"
#include "QueryRenderer/Utils/TypeUtils.h"

namespace QueryRenderer {

namespace {

AggDataList execute_avg_op(const XformOp::DependencyOpResultsMap& dependency_results,
                           const std::string& count_attr,
                           const std::string& sum_attr) {
  AggDataList results;

  auto cnt_itr = dependency_results.find(count_attr);
  auto sum_itr = dependency_results.find(sum_attr);
  CHECK(cnt_itr != dependency_results.end());
  CHECK(sum_itr != dependency_results.end());

  // NOTE: not checking that cntresult/sumresult are defined
  // because the getResult() call does that
  auto const& cnt_result = cnt_itr->second;
  auto const& sum_result = sum_itr->second;
  CHECK_EQ(cnt_result.size(), sum_result.size());
  CHECK_EQ(cnt_result.size(), 1u);
  CHECK(cnt_result[0]);
  CHECK(sum_result[0]);

  switch (sum_result[0]->getType()) {
    case QueryDataType::INT:
    case QueryDataType::UINT:
    case QueryDataType::FLOAT: {
      auto denom = cnt_result[0]->getVal<float>();
      CHECK_NE(denom, getNullValue<float>());
      if (!denom) {
        results = {
            std::make_shared<AnyDataType>(QueryDataType::FLOAT, getNullValue<float>())};
      } else {
        results = {std::make_shared<AnyDataType>(QueryDataType::FLOAT,
                                                 sum_result[0]->getVal<float>() / denom)};
      }
      break;
    }
    case QueryDataType::INT64:
    case QueryDataType::UINT64:
    case QueryDataType::DOUBLE: {
      auto denom = cnt_result[0]->getVal<double>();
      CHECK_NE(denom, getNullValue<double>());
      if (!denom) {
        results = {
            std::make_shared<AnyDataType>(QueryDataType::DOUBLE, getNullValue<double>())};
      } else {
        results = {std::make_shared<AnyDataType>(
            QueryDataType::DOUBLE, sum_result[0]->getVal<double>() / denom)};
      }
      break;
    }
    default:
      throw std::runtime_error(
          "Data of type " + to_string(sum_result[0]->getType()) +
          ". Only single-value types are currently supported for calculating averages.");
  }

  return results;
}

}  // namespace

SQLTypeInfo AvgOp::getOutputType() const {
  return get_float_equivalent_type(AggOp::getOutputType());
}

const XformOp::OpResult AvgOp::executeOp(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const DependencyOpResultsMap& dependency_results) {
  try {
    return {true,
            execute_avg_op(dependency_results,
                           dep_outputs.at(XformOp::serializeOpType(OpType::kCountValid)),
                           dep_outputs.at(XformOp::serializeOpType(OpType::kSum)))};
  } catch (...) {
    LOG_AGG_INFO_AND_THROW(*this);
  }
  return {true, {}};
}

/*********************** dependency info ***********************/
XformOp::DependencyOutputsMap AvgOp::buildDependencyOutputs(
    const LayoutAttrInfo& input_info) {
  DependencyOutputsMap rtn = {{XformOp::serializeOpType(OpType::kCountValid), "cntval"},
                              {XformOp::serializeOpType(OpType::kSum), "sumval"}};
  return rtn;
}

XformOp::DependencyOpTypeMap AvgOp::getRequiredDependencyInfo() const {
  return generateInputDependencyInfo(this, dep_outputs);
}

void AvgOp::setDependency(const XformOpShPtr& op) {
  setInputDependency(this, dep_outputs, dependent_ops_, op);
}

const XformOp::DependencyOpMap* AvgOp::getDependencyOps(
    const std::string* op_type) const {
  auto op = OpType::kMaxOpType;
  std::vector<AnyDataType> args;
  if (op_type) {
    std::tie(op, args) = XformOp::deserializeOperatorAndProps(*op_type);
  }
  CHECK_EQ(args.size(), 0u);
  if (!op_type || op == OpType::kCountValid) {
    auto const& output = dep_outputs.at(XformOp::serializeOpType(OpType::kCountValid));
    if (dependent_ops_.find(output) == dependent_ops_.end()) {
      CHECK(dependent_ops_
                .try_emplace(output,
                             std::make_shared<ValidOp>(
                                 parent_xform_.lock(), getInputInfo(), visited_inputs_))
                .second);
    }
  }

  if (!op_type || op == OpType::kSum) {
    auto const& output = dep_outputs.at(XformOp::serializeOpType(OpType::kSum));
    if (dependent_ops_.find(output) == dependent_ops_.end()) {
      CHECK(dependent_ops_
                .try_emplace(output,
                             std::make_shared<SumOp>(
                                 parent_xform_.lock(), getInputInfo(), visited_inputs_))
                .second);
    }
  }

  return &dependent_ops_;
}

}  // namespace QueryRenderer
