/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/XformOp.h"

#include <boost/algorithm/string/join.hpp>

#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/AvgOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountDistinctOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/MinMaxOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/MissingOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/QuantileOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/SqDiffSumOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/StdDevOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/SumOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/TopBottomKOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/ValidOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/VarianceOp.h"
#include "QueryRenderer/Data/Transforms/BaseXform.h"
#include "QueryRenderer/Data/Transforms/Formula/FormulaXformOp.h"
#include "QueryRenderer/Data/Transforms/Utils.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

using ::gfx::BufferAttrType;
using ::gfx::BufferLayoutShPtr;

void XformOp::validateNumericAttrType(const BaseDataTableShPtr& in_data_ptr,
                                      const LayoutAttrInfoSet& inputs,
                                      const XformOp* current_op) {
  for (auto const& input_info : inputs) {
    auto const attr_type = in_data_ptr->getAttributeBufferType(input_info.attr_name);
    RUNTIME_EX_ASSERT(
        attr_type == BufferAttrType::kUint || attr_type == BufferAttrType::kInt ||
            attr_type == BufferAttrType::kFloat || attr_type == BufferAttrType::kDouble ||
            attr_type == BufferAttrType::kUint64 || attr_type == BufferAttrType::kInt64,
        "Input \"" + input_info.attr_name + "\" is of type " +
            ::gfx::to_string(attr_type) +
            ". Only numeric types are currently supported as generic inputs for "
            "transform operator " +
            to_string(current_op->getOpType()) + ".");
  }
}

std::pair<OpType, std::vector<AnyDataType>> XformOp::deserializeOperatorAndProps(
    const std::string& serialized_str) {
  std::istringstream ss(serialized_str);
  int i_op_type;
  ss >> i_op_type;
  auto op_type = static_cast<OpType>(i_op_type);
  std::vector<AnyDataType> props;
  switch (op_type) {
    case OpType::kCount:
      props = OpSelector<OpType::kCount>::type::deserializeProps(ss);
      break;
    case OpType::kCountValid:
      props = OpSelector<OpType::kCountValid>::type::deserializeProps(ss);
      break;
    case OpType::kCountMissing:
      props = OpSelector<OpType::kCountMissing>::type::deserializeProps(ss);
      break;
    case OpType::kSum:
      props = OpSelector<OpType::kSum>::type::deserializeProps(ss);
      break;
    case OpType::kSQDiffSum:
      props = OpSelector<OpType::kSQDiffSum>::type::deserializeProps(ss);
      break;
    case OpType::kMin:
      props = OpSelector<OpType::kMin>::type::deserializeProps(ss);
      break;
    case OpType::kMax:
      props = OpSelector<OpType::kMax>::type::deserializeProps(ss);
      break;
    case OpType::kDistinct:
      props = OpSelector<OpType::kDistinct>::type::deserializeProps(ss);
      break;
    case OpType::kDistinctHistogram:
      props = OpSelector<OpType::kDistinctHistogram>::type::deserializeProps(ss);
      break;
    case OpType::kAvg:
      props = OpSelector<OpType::kAvg>::type::deserializeProps(ss);
      break;
    case OpType::kVariance:
      props = OpSelector<OpType::kVariance>::type::deserializeProps(ss);
      break;
    case OpType::kVarianceP:
      props = OpSelector<OpType::kVarianceP>::type::deserializeProps(ss);
      break;
    case OpType::kStdDev:
      props = OpSelector<OpType::kStdDev>::type::deserializeProps(ss);
      break;
    case OpType::kStdDevP:
      props = OpSelector<OpType::kStdDevP>::type::deserializeProps(ss);
      break;
    case OpType::kCountDistinct:
      props = OpSelector<OpType::kCountDistinct>::type::deserializeProps(ss);
      break;
    case OpType::kMedian:
      props = OpSelector<OpType::kMedian>::type::deserializeProps(ss);
      break;
    case OpType::kQuantile:
      props = OpSelector<OpType::kQuantile>::type::deserializeProps(ss);
      break;
    case OpType::kTopK:
      props = OpSelector<OpType::kTopK>::type::deserializeProps(ss);
      break;
    case OpType::kBottomK:
      props = OpSelector<OpType::kBottomK>::type::deserializeProps(ss);
      break;
    case OpType::kFormula:
      props = OpSelector<OpType::kFormula>::type::deserializeProps(ss);
      break;
    case OpType::kNonClientFacingSeparator:
    case OpType::kMaxOpType:
      CHECK(false);
  }

  return std::make_pair(op_type, std::move(props));
}

std::string XformOp::serializeOperatorProps(const XformOp* op) {
  std::stringstream stream;
  op->serialize(stream);
  return stream.str();
}

std::string XformOp::serializeOperatorProps(const XformOp& op) {
  return serializeOperatorProps(&op);
}

std::pair<OpType, std::string> XformOp::getOpTypeAndSerializeOperatorFromJSONObj(
    const JSONLocation& json_loc) {
  std::string which_op_str;
  if (json_loc.isObject()) {
    auto const optype_loc = json_loc.getMember(JSONSchema_v1::Xform::kTypeProp);
    if (!optype_loc.isValid() || !optype_loc.isString()) {
      throw std::runtime_error("Invalid transform operator: " +
                               RapidJSONUtils::getObjAsString(json_loc.getValueRef()) +
                               ". Transform operator objects must contain a \"" +
                               std::string(JSONSchema_v1::Xform::kTypeProp) +
                               "\" string property.");
    }
    which_op_str = optype_loc.getString();
  } else if (json_loc.isString()) {
    which_op_str = json_loc.getString();
  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        json_loc,
        "Invalid transform operator. Transform operator objects must be a string or an "
        "object."));
  }

  auto const which_op = convert_string_to_op_type_enum(which_op_str);
  RUNTIME_EX_ASSERT(
      which_op >= 0,
      RapidJSONUtils::createJsonParseError(
          json_loc,
          "Invalid transform operator type: \"" + which_op_str + "\" from json" +
              RapidJSONUtils::getObjAsString(json_loc.getValueRef()) + "."));

  auto op_type = static_cast<OpType>(which_op);
  std::stringstream ss;
  ss << serializeOpType(static_cast<OpType>(which_op));
  if (json_loc.isObject()) {
    switch (op_type) {
      case OpType::kCount:
        OpSelector<OpType::kCount>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kCountValid:
        OpSelector<OpType::kCountValid>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kCountMissing:
        OpSelector<OpType::kCountMissing>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kSum:
        OpSelector<OpType::kSum>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kSQDiffSum:
        OpSelector<OpType::kSQDiffSum>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kMin:
        OpSelector<OpType::kMin>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kMax:
        OpSelector<OpType::kMax>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kDistinct:
        OpSelector<OpType::kDistinct>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kDistinctHistogram:
        OpSelector<OpType::kDistinctHistogram>::type::serializePropsFromJSONObj(ss,
                                                                                json_loc);
        break;
      case OpType::kAvg:
        OpSelector<OpType::kAvg>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kVariance:
        OpSelector<OpType::kVariance>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kVarianceP:
        OpSelector<OpType::kVarianceP>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kStdDev:
        OpSelector<OpType::kStdDev>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kStdDevP:
        OpSelector<OpType::kStdDevP>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kCountDistinct:
        OpSelector<OpType::kCountDistinct>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kMedian:
        OpSelector<OpType::kMedian>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kQuantile:
        OpSelector<OpType::kQuantile>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kTopK:
        OpSelector<OpType::kTopK>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kBottomK:
        OpSelector<OpType::kBottomK>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kFormula:
        OpSelector<OpType::kFormula>::type::serializePropsFromJSONObj(ss, json_loc);
        break;
      case OpType::kNonClientFacingSeparator:
      case OpType::kMaxOpType:
        CHECK(false);
    }
  }
  return std::make_pair(op_type, ss.str());
}

void XformOp::serialize(std::stringstream& ss) const {
  ss << serializeOpType(getOpType());
  serializeProps(ss);
}

XformOp::XformOp(const XformShPtr& parent_xform,
                 const LayoutAttrInfoSet& inputs,
                 const bool is_vector_op)
    : parent_xform_{parent_xform}
    , inputs_{inputs}
    , cached_result_{false, {}}
    , dirty_{true}
    , is_vector_{is_vector_op} {}

XformOp::XformOp(const XformShPtr& parent_xform,
                 const LayoutAttrInfo& input_info,
                 const bool is_vector_op)
    : parent_xform_{parent_xform}
    , cached_result_{false, {}}
    , dirty_{true}
    , is_vector_{is_vector_op} {
  inputs_.emplace(input_info);
}

XformOp::XformOp(const XformShPtr& parent_xform,
                 const std::vector<LayoutAttrInfo>& all_input_info,
                 const bool is_vector_op)
    : parent_xform_{parent_xform}
    , cached_result_{false, {}}
    , dirty_{true}
    , is_vector_{is_vector_op} {
  std::for_each(all_input_info.begin(),
                all_input_info.end(),
                [this](auto const& input_info) { inputs_.emplace(input_info); });
}

std::unordered_set<std::string> XformOp::getInputAttrNames() const {
  std::unordered_set<std::string> rtn;
  std::transform(inputs_.begin(),
                 inputs_.end(),
                 std::inserter(rtn, rtn.end()),
                 [](auto const& input) { return input.attr_name; });
  return rtn;
}

XformOp::operator std::string() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  return "{source: \"" + parent_xform->getSourceDataTableName() + "\", inputs: [" +
         boost::algorithm::join(getInputAttrNames(), ", ") +
         "], operator: " + getOpTypeAsStr() + "}";
}

void XformOp::validateInputs() const {
  // Defaults to checking basic numeric input types
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(in_data);
  if (data) {
    for (auto const& input_info : inputs_) {
      auto query_data_layout = getDataLayoutForAttribute(in_data, input_info.attr_name);
      CHECK(query_data_layout);
      auto const& type_info =
          query_data_layout->getAttrSQLTypeInfoRef(input_info.attr_name);
      RUNTIME_EX_ASSERT(type_info.is_number(),
                        "Input \"" + input_info.attr_name + "\" is of type " +
                            type_info.get_type_name() +
                            ". Only numeric types are currently supported as inputs for "
                            "aggregate transform operator " +
                            to_string(getOpType()) + ".");
    }
  } else {
    validateNumericAttrType(in_data, inputs_, this);
  }
}

template <>
XformOp::TypedOpArrayResult<std::string> XformOp::evaluateVector<std::string>(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers) {
  RUNTIME_EX_ASSERT(
      is_vector_,
      std::string(*this) +
          ": Cannot evaluate to a vector. This op results in singular value.");

  auto const type_info = getOutputType();
  RUNTIME_EX_ASSERT(
      type_info.is_string() && type_info.get_compression() == kENCODING_DICT,
      std::string(*this) +
          ": Cannot evaluate operator to a vector of strings. The output is not a "
          "dict-encoded string. It is a " +
          type_info.get_type_name() +
          ". Only dict-encoded strings are supported for evaluating to string.");

  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto in_data = parent_xform->getInputDataTable();
  CHECK(in_data);
  auto sql_data_table = dynamic_cast<BaseQueryDataTableSQLJSON*>(in_data.get());
  CHECK(sql_data_table) << std::string(*this) << ": invalid input table";
  RUNTIME_EX_ASSERT(inputs_.size() == 1,
                    std::string(*this) +
                        ": Cannot evaluate operator to a vector of strings. The output "
                        "has multiple inputs. Only "
                        "outputs with a single input can currently be evaluated to a "
                        "vector of strings.");

  auto const itr = inputs_.begin();
  auto layout = getDataLayoutForAttribute(in_data, itr->attr_name);
  CHECK(layout) << std::string(*this) << ": Layout can't be found for: " << itr->attr_name
                << " in table " << sql_data_table->getName();

  auto eval_result = evaluate(evaluator_name, mapped_buffers);
  if (!eval_result.is_execution_complete) {
    return {false, std::vector<std::string>()};
  }
  auto data_results = eval_result.getDataResults();
  CHECK_EQ(data_results.size(), 1u);
  CHECK(data_results[0]);
  CHECK(data_results[0]->isVector());
  auto ids = data_results[0]->getVectorVal<int>();

  auto const* render_query_runner = parent_xform->ctx_.getRenderQueryRunner();
  CHECK(render_query_runner);

  return {eval_result.is_execution_complete,
          render_query_runner->getStringsFromIds(
              *layout, itr->attr_name, ids, *sql_data_table->getResultSet())};
}

const XformOp::OpResult XformOp::evaluate(const std::string& evaluator_name,
                                          InteropBufferMgr* mapped_buffers) {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);

  std::unique_ptr<InteropBufferMgr> my_mapped_buffers;
  InteropBufferMgr* my_mapped_buffers_ptr = mapped_buffers;
  if (!my_mapped_buffers_ptr) {
    my_mapped_buffers = std::make_unique<InteropBufferMgr>(
        parent_xform->ctx_.getCudaMgr(), parent_xform->ctx_.getGlobalContext());
    my_mapped_buffers_ptr = my_mapped_buffers.get();
  }

  return evaluateInternal(evaluator_name, my_mapped_buffers_ptr);
}

Data_Namespace::DataMgr& XformOp::getDataMgr() {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  auto data_mgr = parent_xform->ctx_.getDataMgr();
  CHECK(data_mgr);
  return *data_mgr;
}

const CudaMgr_Namespace::CudaMgr* XformOp::getCudaMgr() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  return parent_xform->ctx_.getCudaMgr();
}

const QueryRendererContext& XformOp::getRenderContext() const {
  auto const parent_xform = parent_xform_.lock();
  CHECK(parent_xform);
  return parent_xform->ctx_;
}

QueryRendererContext& XformOp::getRenderContextNonConst() {
  return const_cast<QueryRendererContext&>(getRenderContext());
}

const XformOp::OpResult XformOp::evaluateInternal(const std::string& evaluator_name,
                                                  InteropBufferMgr* mapped_buffers) {
  if (dirty_) {
    DependencyOpResultsMap dependent_vals;
    auto* dep_info = getDependencyOps();

    dirty_ = false;

    auto& is_execution_complete = cached_result_.is_execution_complete;
    is_execution_complete = true;
    if (dep_info) {
      for (auto& item : *dep_info) {
        auto eval_info = item.second->evaluateInternal(evaluator_name, mapped_buffers);
        is_execution_complete = is_execution_complete && eval_info.is_execution_complete;

        if (is_execution_complete) {
          auto insert =
              dependent_vals.insert({item.first, std::move(eval_info.op_results)});
          CHECK(insert.second);
        }
      }
    }

    if (!is_execution_complete) {
      return {is_execution_complete, {}};
    }

    cached_result_ = executeOp(evaluator_name, mapped_buffers, dependent_vals);
  }

  return cached_result_;
}

XformOpShPtr XformOp::getDependency(const std::string& dep_input_attr,
                                    const std::string& dep_op_type) const {
  auto const dep_outputs = getDependencyOutputsMap();
  RUNTIME_EX_ASSERT(
      dep_outputs,
      "The operator: " + std::string(*this) + " does not support a dependencies");

  RUNTIME_EX_ASSERT(isValidDepInput(dep_input_attr),
                    "The operator: " + std::string(*this) +
                        " does not support a dependency with input name: \"" +
                        dep_input_attr + "\".");

  auto attr_itr = dep_outputs->find(dep_op_type);
  RUNTIME_EX_ASSERT(attr_itr != dep_outputs->end(),
                    "The operator: " + std::string(*this) +
                        " does not support a dependency of type: \"" + dep_op_type +
                        "\".");

  auto op_map = getDependencyOps(&dep_op_type);
  CHECK(op_map) << "Ops with dependencies need this defined";
  return op_map->at(attr_itr->second);
}

void XformOp::setDirty() {
  dirty_ = true;

  bool cleanup = false;
  for (auto& item : dependents_) {
    auto ptr = item.second.lock();
    if (ptr) {
      ptr->setDirty();
    } else {
      cleanup = true;
    }
  }
  if (cleanup) {
    cleanupDependents();
  }
}

void XformOp::clearCacheWithoutPropagatingDirtyFlag() {
  // NOTE: we're not propagating the dirty flag here to this op's dependents
  // This needs to be called in the specific instance where this data isn't
  // necessary anymore, but a dependent's data is.
  // A node is deemed unnecessary if all it's dependent nodes are not dirty.
  // This is used to clean up memory in cases where an op node is hanging onto a lot
  // of data that a user isn't directly using (such as DistinctHistogramOp, which
  // is a hidden op)
  // TODO(croot): should we really be doing this? Or should we still hang onto the
  // memory? There could be cases where haning onto it might be useful, but I've
  // yet to see a compelling case, tho going to a proper reactive dataflow representation
  // may change that.
  if (!cached_result_.is_execution_complete) {
    return;
  }

  for (auto& item : dependents_) {
    auto ptr = item.second.lock();
    if (ptr && (ptr->dirty_ || !ptr->cached_result_.is_execution_complete)) {
      return;
    }
  }
  cached_result_.reset();
  dirty_ = true;
}

void XformOp::setDependent(const XformOpWkPtr& op) {
  auto op_ptr = op.lock();
  if (op_ptr) {
    dependents_.insert({op_ptr.get(), op_ptr});
  }

  cleanupDependents();
}

void XformOp::cleanupDependents() {
  auto itr = dependents_.begin();
  while (itr != dependents_.end()) {
    if (itr->second.expired()) {
      itr = dependents_.erase(itr);
    } else {
      ++itr;
    }
  }
}

}  // namespace QueryRenderer
