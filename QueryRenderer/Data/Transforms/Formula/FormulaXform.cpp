/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Formula/FormulaXform.h"

#include <muparserx/mpParser.h>

#include "QueryRenderer/Data/Transforms/Formula/FormulaXformOp.h"
#include "QueryRenderer/Data/Transforms/Utils.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

using ::gfx::BufferAttrType;
using ::gfx::InterleavedBufferLayout;
using ::gfx::InterleavedBufferLayoutShPtr;

FormulaXform::FormulaXform(const QueryRendererContext& ctx,
                           const BaseDataTableShPtr& data)
    : BaseXform(ctx, data) {}

void FormulaXform::initialize(const XformShPtr& ptr, const JSONLocation& obj_loc) {
  initFromJSONObj(ptr, obj_loc);
}

bool FormulaXform::hasOutput(const std::string& output) const {
  if (output == my_output_) {
    return true;
  }

  auto parent = dynamic_cast<BaseXform*>(data_.get());
  if (parent) {
    return parent->hasOutput(output);
  }
  // TODO(croot): add support for running expressions against BaseQueryDataTableJSON
  // objects

  return false;
}

const XformOpShPtr FormulaXform::getOutputOp(const std::string& output) const {
  if (output == my_output_) {
    return my_op_;
  }

  auto parent = dynamic_cast<BaseXform*>(data_.get());
  if (parent) {
    return parent->getOutputOp(output);
  }
  // TODO(croot): add support for running expressions against BaseQueryDataTableJSON
  // objects using thrust/cuda or executing a subquery

  THROW_RUNTIME_EX("Transform " + to_string(getXformType()) + " does not have output \"" +
                   output + "\".");
  return nullptr;
}

std::set<std::string> FormulaXform::getAllOutputNames() const {
  std::set<std::string> rtn;
  auto parent = dynamic_cast<BaseXform*>(data_.get());
  if (parent) {
    rtn = parent->getAllOutputNames();
  }
  // TODO(croot): add support for running expressions against BaseQueryDataTableJSON
  // objects

  rtn.insert(my_output_);
  return rtn;
}

void FormulaXform::initFromJSONObj(const XformShPtr& ptr, const JSONLocation& obj_loc) {
  CHECK(data_);

  RUNTIME_EX_ASSERT(
      data_->getInputFormat() == DataInputFormat::kTransformed,
      RapidJSONUtils::createJsonParseError(
          obj_loc,
          "Formula transform objects only currently support expressions run against "
          "other transform data table inputs. The input data table is of type " +
              to_string(data_->getInputFormat()) + "."));

  auto parent = dynamic_cast<BaseXform*>(data_.get());
  CHECK(parent);

  RUNTIME_EX_ASSERT(
      parent->getXformType() == XformType::kAggregate ||
          parent->getXformType() == XformType::kFormula,
      RapidJSONUtils::createJsonParseError(
          obj_loc,
          "Formula transform objects only currently support expressions run against "
          "aggregate/formula transform inputs. The input transform table is of type " +
              to_string(parent->getXformType()) + "."));

  auto const expr_loc = obj_loc.getMember(JSONSchema_v1::Xform::kExprProp);
  RUNTIME_EX_ASSERT(expr_loc.isValid() && expr_loc.isString(),
                    RapidJSONUtils::createJsonParseError(
                        expr_loc.isValid() ? expr_loc : obj_loc,
                        "Formula transform object must contain an \"" +
                            std::string(JSONSchema_v1::Xform::kExprProp) +
                            "\" property and it must be a string."));

  auto const as_loc = obj_loc.getMember(JSONSchema_v1::Xform::kAsProp);
  RUNTIME_EX_ASSERT(as_loc.isValid() && as_loc.isString(),
                    RapidJSONUtils::createJsonParseError(
                        as_loc.isValid() ? as_loc : obj_loc,
                        "Formula transform object must contain an \"" +
                            std::string(JSONSchema_v1::Xform::kAsProp) +
                            "\" property and it must be a string."));

  auto const expr_str = std::string(expr_loc.getString());
  auto const as_str = std::string(as_loc.getString());
  RUNTIME_EX_ASSERT(
      !parent->hasOutput(as_str),
      RapidJSONUtils::createJsonParseError(
          as_loc,
          "The formula transform object already has the output \"" + as_str +
              "\" specified by the \"" + std::string(JSONSchema_v1::Xform::kAsProp) +
              "\" property. A unique name must be used. The existing outputs are " +
              to_string(parent->getAllOutputNames())));

  ::mup::ParserX p(::mup::pckCOMMON | ::mup::pckUNIT | ::mup::pckNON_COMPLEX);
  ::mup::var_maptype vmap;
  try {
    p.SetExpr(expr_str);
    vmap = p.GetExprVar();
  } catch (::mup::ParserError& err) {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        expr_loc,
        "Could not properly compile the expression \"" + expr_str +
            "\". Error: " + err.GetMsg()));
  } catch (std::exception& err) {
    CHECK(false) << err.what();
  }

  XformOp::DependencyOpMap dep_ops;
  std::for_each(
      vmap.begin(), vmap.end(), [&parent, &expr_loc, &dep_ops](auto const& item) {
        RUNTIME_EX_ASSERT(
            parent->hasOutput(item.first),
            RapidJSONUtils::createJsonParseError(
                expr_loc,
                "The expression variable \"" + item.first + "\" cannot be found. The \"" +
                    std::string(JSONSchema_v1::Xform::kExprProp) +
                    "\" property is invalid. The available variable names are " +
                    to_string(parent->getAllOutputNames())));

        auto output_op = parent->getOutputOp(item.first);
        CHECK(output_op) << "Issue with op: " << item.first;

        RUNTIME_EX_ASSERT(!output_op->isVectorOp(),
                          RapidJSONUtils::createJsonParseError(
                              expr_loc,
                              "The expression variable \"" + item.first +
                                  "\" results in an array. Arrays cannot be used in "
                                  "expressions. The \"" +
                                  std::string(JSONSchema_v1::Xform::kExprProp) +
                                  "\" property is invalid."));

        dep_ops.insert({item.first, parent->getOutputOp(item.first)});
      });

  my_output_ = as_str;
  my_op_ = std::make_shared<FormulaXformOp>(ptr, expr_str, dep_ops);

  // TODO(croot): have a setting to switch between interleaved/sequential buffer
  // layouts?
  auto parent_layout = parent->getOutputsDataLayout();
  auto attr_info = parent_layout->getAllAttrInfo();
  attr_info.emplace_back(as_str, my_op_->getOutputType());
  data_layout_ = std::make_shared<QueryDataLayout>(std::move(attr_info));

  for (auto& dep_item : dep_ops) {
    dep_item.second->setDependent(my_op_);
  }
}

}  // namespace QueryRenderer
