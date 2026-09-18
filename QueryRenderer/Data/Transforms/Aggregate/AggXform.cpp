/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/AggXform.h"

#include "GfxDriver/Resources/BufferLayout.h"
#include "QueryRenderer/AggregationContext.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/AvgOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountDistinctOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/CountOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/MinMaxOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/MissingOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/QuantileOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/StdDevOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/SumOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/ValidOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/VarianceOp.h"
#include "QueryRenderer/Data/Transforms/Utils.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

using ::gfx::BufferAttrType;
using ::gfx::InterleavedBufferLayout;
using ::gfx::InterleavedBufferLayoutShPtr;

namespace {

void build_agg_dependency_graph(
    AggXform::OperatorMap_by_NameAndType& op_name_type_map,
    AggXform::OperatorMap_by_NameAndType& tmp_op_name_type_map,
    const AggOpShPtr& op) {
  if (op->isDependentOp()) {
    auto all_dep_info = op->getRequiredDependencyInfo();
    for (auto& dep_info : all_dep_info) {
      for (auto& dep_op : dep_info.second) {
        auto key = std::make_tuple(dep_info.first, dep_op);
        auto itr = op_name_type_map.find(key);
        if (itr != op_name_type_map.end()) {
          op->setDependency(*itr);
          (*itr)->setDependent(op);
        } else if ((itr = tmp_op_name_type_map.find(key)) != tmp_op_name_type_map.end()) {
          op->setDependency(*itr);
          (*itr)->setDependent(op);
        } else {
          auto dep = op->getDependency(dep_info.first, dep_op);
          dep->setDependent(op);
          auto agg_dep = std::dynamic_pointer_cast<AggOp>(dep);
          CHECK(agg_dep);
          CHECK(tmp_op_name_type_map.insert(agg_dep).second);
          build_agg_dependency_graph(op_name_type_map, tmp_op_name_type_map, agg_dep);
        }
      }
    }
  }
}

}  // namespace

AggXform::AggXform(const QueryRendererContext& ctx, const BaseDataTableShPtr& data)
    : BaseXform(ctx, data) {}

void AggXform::initialize(const XformShPtr& ptr, const JSONLocation& obj_loc) {
  initFromJSONObj(ptr, obj_loc);
}

bool AggXform::hasOutput(const std::string& output) const {
  return outputs_.find(output) != outputs_.end();
}

const XformOpShPtr AggXform::getOutputOp(const std::string& output) const {
  auto itr = outputs_.find(output);
  RUNTIME_EX_ASSERT(itr != outputs_.end(),
                    "Transform of type " + to_string(getXformType()) +
                        " does not have output \"" + output + "\".");
  return itr->second.second;
}

std::set<std::string> AggXform::getAllOutputNames() const {
  std::set<std::string> rtn;
  std::transform(outputs_.begin(),
                 outputs_.end(),
                 std::inserter(rtn, rtn.begin()),
                 [](auto const& item) { return item.first; });
  return rtn;
}

void AggXform::initFromJSONObj(const XformShPtr& ptr, const JSONLocation& obj_loc) {
  CHECK(data_);
  RUNTIME_EX_ASSERT(
      isExternallySourcedInputFormat(data_->getInputFormat()),
      RapidJSONUtils::createJsonParseError(
          obj_loc,
          "Aggregator transform objects only support aggregation on input data of one of "
          "these types " +
              getDataInputFormatsAsStr() + ". The input data table is of type " +
              to_string(data_->getInputFormat()) + "."));

  auto const field_loc = obj_loc.getMember(JSONSchema_v1::Xform::kFieldProp);
  RUNTIME_EX_ASSERT(field_loc.isValid() && field_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        field_loc.isValid() ? field_loc : obj_loc,
                        "Aggregator transform object must contain a \"" +
                            std::string(JSONSchema_v1::Xform::kFieldProp) +
                            "\" property and it must be an array of strings."));

  auto const ops_loc = obj_loc.getMember(JSONSchema_v1::Xform::kOpsProp);
  RUNTIME_EX_ASSERT(
      ops_loc.isValid() && ops_loc.isArray() && ops_loc.size() == field_loc.size(),
      RapidJSONUtils::createJsonParseError(
          ops_loc.isValid() ? ops_loc : obj_loc,
          "Aggregator transform object must contain an \"" +
              std::string(JSONSchema_v1::Xform::kOpsProp) +
              "\" property and it must be an array of strings with the "
              "same number of elements as the \"" +
              std::string(JSONSchema_v1::Xform::kFieldProp) + "\" property."));

  auto const as_loc = obj_loc.getMember(JSONSchema_v1::Xform::kAsProp);
  RUNTIME_EX_ASSERT(
      as_loc.isValid() && as_loc.isArray() && as_loc.size() == field_loc.size(),
      RapidJSONUtils::createJsonParseError(
          as_loc.isValid() ? as_loc : obj_loc,
          "Aggregator transform object must contain an \"" +
              std::string(std::string(JSONSchema_v1::Xform::kAsProp)) +
              "\" property and it must be an array of strings with the "
              "same number of elements as the \"" +
              std::string(JSONSchema_v1::Xform::kFieldProp) + "\" property."));

  // TODO(croot): have a setting to switch between interleaved/sequential buffer
  // layouts?
  // clear out existing outputs
  ops_.clear();
  tmp_ops_.clear();
  outputs_.clear();

  std::vector<QueryDataLayout::AttrAliasInfo> attr_info;
  AggOp::VisitedInputsSetShPtr visited_inputs =
      std::make_shared<std::unordered_set<std::string>>();
  for (size_t i = 0; i < field_loc.size(); ++i) {
    auto const field_item_loc = field_loc[i];
    RUNTIME_EX_ASSERT(
        field_item_loc.isString(),
        RapidJSONUtils::createJsonParseError(
            field_item_loc,
            "The value of index " + std::to_string(i) + " in the \"" +
                std::string(JSONSchema_v1::Xform::kFieldProp) +
                "\" array has the value " +
                RapidJSONUtils::getObjAsString(field_item_loc.getValueRef()) +
                " which is not a string. All elements of the \"" +
                std::string(JSONSchema_v1::Xform::kFieldProp) +
                "\" property array must be strings referencing columns in the data "
                "source. The available columns are " +
                to_string(data_->getAllAttrNames())));

    auto const col_name = std::string(field_item_loc.getString());
    const bool attr_exists_in_data = data_->hasAttribute(col_name);
    QueryDataLayoutShPtr distrib_agg_layout =
        (!attr_exists_in_data ? getDataLayoutForAttribute(data_, col_name) : nullptr);
    RUNTIME_EX_ASSERT(
        attr_exists_in_data || distrib_agg_layout,
        RapidJSONUtils::createJsonParseError(
            field_item_loc,
            "The column \"" + col_name + "\" specified at index " + std::to_string(i) +
                " in the \"" + std::string(JSONSchema_v1::Xform::kFieldProp) +
                "\" property array does not exist in the data source. The "
                "available columns are " +
                to_string(data_->getAllAttrNames())));

    auto const ops_item_loc = ops_loc[i];
    bool is_obj_defined_op = ops_item_loc.isObject();
    OpType op_type;
    std::string serialized_op;
    try {
      std::tie(op_type, serialized_op) =
          XformOp::getOpTypeAndSerializeOperatorFromJSONObj(ops_item_loc);
    } catch (std::runtime_error& err) {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          ops_item_loc,
          "Error parsing the value of index " + std::to_string(i) + " in the \"" +
              std::string(JSONSchema_v1::Xform::kOpsProp) + "\" array. " + err.what()));
    }

    RUNTIME_EX_ASSERT(
        is_agg_op_type(op_type),
        RapidJSONUtils::createJsonParseError(
            ops_item_loc,
            "The value of index " + std::to_string(i) + " in the \"" +
                std::string(JSONSchema_v1::Xform::kOpsProp) + "\" array has the value " +
                RapidJSONUtils::getObjAsString(ops_item_loc.getValueRef()) +
                " but it is not an aggregation operator. It must be one of the "
                "strings: " +
                get_agg_ops_as_string() + "."));

    auto const as_item_loc = as_loc[i];
    RUNTIME_EX_ASSERT(
        as_item_loc.isString(),
        RapidJSONUtils::createJsonParseError(
            as_item_loc,
            "The value of index " + std::to_string(i) + " in the \"" +
                std::string(JSONSchema_v1::Xform::kAsProp) + "\" array has the value " +
                RapidJSONUtils::getObjAsString(as_item_loc.getValueRef()) +
                " which is not a string. All elements of the \"" +
                std::string(JSONSchema_v1::Xform::kAsProp) +
                "\" property array must be unique strings."));

    auto out_itr = outputs_.find(makeLowerCase(as_item_loc.getString()));
    RUNTIME_EX_ASSERT(
        out_itr == outputs_.end(),
        RapidJSONUtils::createJsonParseError(
            as_item_loc,
            "The value of index " + std::to_string(i) + " in the \"" +
                std::string(JSONSchema_v1::Xform::kAsProp) + "\" array with the value\"" +
                RapidJSONUtils::getObjAsString(as_item_loc.getValueRef()) +
                "\" is a duplicate of the element at index " +
                std::to_string(out_itr->second.first) + ". All elements of the \"" +
                std::string(JSONSchema_v1::Xform::kAsProp) +
                "\" property array must be unique strings."));

    auto attr_data_layout = data_->getAttributeBufferLayout(col_name);
    CHECK(attr_data_layout);

    auto const type_info = data_->getAttributeTypeInfo(col_name);
    const LayoutAttrInfo input_info{col_name, type_info, attr_data_layout};

    auto& operator_name_type_map = ops_.get<OpByNameAndType>();
    auto op_map_itr =
        operator_name_type_map.find(std::make_tuple(col_name, serialized_op));
    if (op_map_itr == operator_name_type_map.end()) {
      auto op_insert = std::make_pair(op_map_itr, false);
      try {
        switch (op_type) {
          case OpType::kCount:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<CountOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kCountValid:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<ValidOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kCountMissing:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<MissingOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kMin:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<MinOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kMax:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<MaxOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kSum:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<SumOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kAvg:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<AvgOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kVariance:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<VarianceOp>(ptr, input_info, visited_inputs, false));
            break;
          case OpType::kVarianceP:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<VarianceOp>(ptr, input_info, visited_inputs, true));
            break;
          case OpType::kStdDev:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<StdDevOp>(ptr, input_info, visited_inputs, false));
            break;
          case OpType::kStdDevP:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<StdDevOp>(ptr, input_info, visited_inputs, true));
            break;
          case OpType::kDistinct:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<DistinctOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kCountDistinct:
            op_insert = operator_name_type_map.emplace(
                std::make_shared<CountDistinctOp>(ptr, input_info, visited_inputs));
            break;
          case OpType::kMedian:
            if (is_obj_defined_op) {
              op_insert = operator_name_type_map.emplace(std::make_shared<QuantileOp>(
                  ptr, input_info, visited_inputs, ops_item_loc));
            } else {
              op_insert = operator_name_type_map.emplace(std::make_shared<QuantileOp>(
                  ptr, input_info, visited_inputs, 2, false));
            }
            break;
          case OpType::kQuantile:
            if (is_obj_defined_op) {
              op_insert = operator_name_type_map.emplace(std::make_shared<QuantileOp>(
                  ptr, input_info, visited_inputs, ops_item_loc));
            } else {
              op_insert = operator_name_type_map.emplace(std::make_shared<QuantileOp>(
                  ptr, input_info, visited_inputs, 4, false));
            }
            break;
          case OpType::kNonClientFacingSeparator:
          case OpType::kSQDiffSum:
          case OpType::kDistinctHistogram:
          case OpType::kFormula:
          case OpType::kTopK:
          case OpType::kBottomK:
          case OpType::kMaxOpType:
            CHECK(false);
        }
      } catch (std::runtime_error& err) {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
            obj_loc,
            "The column \"" + col_name + "\" specified at index " + std::to_string(i) +
                " in the \"" + std::string(JSONSchema_v1::Xform::kFieldProp) +
                "\" property array is not valid. " + err.what()));
      }
      CHECK(op_insert.second);
      op_map_itr = op_insert.first;
    }
    auto insert = outputs_.insert({as_item_loc.getString(), {i, *op_map_itr}});
    CHECK(insert.second);

    attr_info.emplace_back(insert.first->first, (*op_map_itr)->getOutputType());
  }

  // update ops with dependencies that are being asked to be executed by the user
  // to avoid duplicating calculations
  auto& op_name_type_map = ops_.get<OpByNameAndType>();
  auto& tmp_op_name_type_map = tmp_ops_.get<OpByNameAndType>();
  for (auto& item : op_name_type_map) {
    build_agg_dependency_graph(op_name_type_map, tmp_op_name_type_map, item);
  }

  data_layout_ = std::make_shared<QueryDataLayout>(std::move(attr_info));
}

void AggXform::markAggOpsDirtyAfterRenderStepInternal(
    const std::string& /*evaluator_name*/) {}

void AggXform::clearNonPublicFacingOpDataAfterVegaUpdateInternal() {
  // go thru all the temporary (on non-public facing) ops in the
  // operator dependency graph and clear out their cached data
  // This should be called after a successful JSON update to clear out
  // unused data as that cached data isn't necessary anymore.
  // But in doing so, don't propogate the dirty flag.
  for (auto& tmp_op : tmp_ops_) {
    tmp_op->clearCacheWithoutPropagatingDirtyFlag();
  }
}

}  // namespace QueryRenderer
