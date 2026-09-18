/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Formula/FormulaXformOp.h"

#include <muparserx/mpParser.h>

#include "QueryRenderer/Data/Transforms/BaseXform.h"
#include "QueryRenderer/Utils/NumericUtils.h"
#include "QueryRenderer/Utils/TypeUtils.h"

using ::gfx::BufferAttrType;

namespace QueryRenderer {

namespace {

void validate_parser_with_deps(FormulaXformOp* op,
                               const std::string& formula_str,
                               const XformOp::DependencyOpMap& deps,
                               ::mup::ParserX& parser) {
  ::mup::var_maptype vmap;
  try {
    parser.SetExpr(formula_str);
    vmap = parser.GetExprVar();
  } catch (::mup::ParserError& err) {
    THROW_RUNTIME_EX(std::string(*op) +
                     " Could not properly compile the formula expression \"" +
                     formula_str + "\". Error: " + err.GetMsg());
  } catch (std::exception& err) {
    CHECK(false) << err.what();
  }

  for (auto variable = vmap.begin(); variable != vmap.end(); ++variable) {
    RUNTIME_EX_ASSERT(deps.find(variable->first) != deps.end(),
                      std::string(*op) + " Could not find the variable \"" +
                          variable->first + "\" from the formula expression \"" +
                          formula_str + "\" in the operator's dependencies.");
  }
}

BufferAttrType get_higher_priority_buffer_attr_type(
    const BufferAttrType base_data_type,
    const BufferAttrType check_data_type) {
  if (base_data_type == check_data_type) {
    return base_data_type;
  }

  if (base_data_type == BufferAttrType::kCOUNT) {
    return check_data_type;
  } else if (check_data_type == BufferAttrType::kCOUNT) {
    return base_data_type;
  }

  switch (base_data_type) {
    case BufferAttrType::kFloat:
      if (check_data_type == BufferAttrType::kInt ||
          check_data_type == BufferAttrType::kUint) {
        return BufferAttrType::kFloat;
      } else {
        return BufferAttrType::kDouble;
      }
    case BufferAttrType::kDouble:
      return BufferAttrType::kDouble;
    case BufferAttrType::kInt:
      if (check_data_type == BufferAttrType::kUint) {
        // if int < 0, so int64 will hold both
        return BufferAttrType::kInt64;
      } else if (check_data_type == BufferAttrType::kUint64) {
        return BufferAttrType::kDouble;
      } else if (check_data_type == BufferAttrType::kFloat ||
                 check_data_type == BufferAttrType::kInt64 ||
                 check_data_type == BufferAttrType::kDouble) {
        return check_data_type;
      }
      break;
    case BufferAttrType::kUint:
      if (check_data_type == BufferAttrType::kInt) {
        // if the int < 0, so int64 will hold both
        return BufferAttrType::kInt64;
      } else if (check_data_type == BufferAttrType::kFloat ||
                 check_data_type == BufferAttrType::kInt64 ||
                 check_data_type == BufferAttrType::kUint64 ||
                 check_data_type == BufferAttrType::kDouble) {
        return check_data_type;
      }
      break;
    case BufferAttrType::kInt64:
      if (check_data_type == BufferAttrType::kInt ||
          check_data_type == BufferAttrType::kUint) {
        return BufferAttrType::kInt64;
      } else if (check_data_type == BufferAttrType::kFloat ||
                 check_data_type == BufferAttrType::kDouble) {
        return BufferAttrType::kDouble;
      } else if (check_data_type == BufferAttrType::kUint64) {
        return BufferAttrType::kDouble;
      }
      break;
    case BufferAttrType::kUint64:
      if (check_data_type == BufferAttrType::kInt) {
        return BufferAttrType::kDouble;
      } else if (check_data_type == BufferAttrType::kUint) {
        return BufferAttrType::kUint64;
      } else if (check_data_type == BufferAttrType::kFloat ||
                 check_data_type == BufferAttrType::kDouble) {
        return BufferAttrType::kDouble;
      } else if (check_data_type == BufferAttrType::kInt64) {
        return BufferAttrType::kDouble;
      }
      break;

    default:
      break;
  }

  THROW_RUNTIME_EX("Determining the higher priority type between " +
                   to_string(base_data_type) + " and " + to_string(check_data_type) +
                   " is unsupported");
  return base_data_type;
}

BufferAttrType get_buffer_attr_type_from_value(const ::mup::Value& val) {
  auto char_type = val.GetType();
  switch (char_type) {
    case 'i':
      return BufferAttrType::kInt;
    case 'f':
      return BufferAttrType::kDouble;
    default:
      THROW_RUNTIME_EX("Cannot convert muparserx type to a valid buffer type. '" +
                       std::string(&char_type, 1) + "' is not supported.");
  }
  return BufferAttrType::kInt;
}

}  // namespace

FormulaXformOp::FormulaXformOp(const XformShPtr& parent_xform,
                               const std::string& formula_str,
                               const DependencyOpMap& dependencies)
    : XformDepOp(parent_xform, LayoutAttrInfoSet(), false)
    , formula_str_{formula_str}
    , dependencies_{dependencies}
    , cached_output_type_{BufferAttrType::kCOUNT} {
  validateInputs();
  ::mup::ParserX exp_parser(::mup::pckCOMMON | ::mup::pckUNIT | ::mup::pckNON_COMPLEX);
  validate_parser_with_deps(this, formula_str_, dependencies_, exp_parser);
  std::transform(dependencies_.begin(),
                 dependencies_.end(),
                 std::inserter(dep_defs_, dep_defs_.end()),
                 [](auto const& item) {
                   return std::make_pair(
                       XformOp::serializeOperatorProps(item.second.get()), item.first);
                 });
}

XformOp::DependencyOpTypeMap FormulaXformOp::getRequiredDependencyInfo() const {
  XformOp::DependencyOpTypeMap rtn;
  std::transform(
      dependencies_.begin(),
      dependencies_.end(),
      std::inserter(rtn, rtn.end()),
      [](auto const& item) {
        return std::make_pair(
            item.first,
            OpTypeContainer({XformOp::serializeOperatorProps(item.second.get())}));
      });
  return rtn;
}

void FormulaXformOp::setDependency(const XformOpShPtr& op) {
  THROW_RUNTIME_EX(std::string(*this) +
                   ": Cannot explicitly set a dependency for a formula operator.");
}

SQLTypeInfo FormulaXformOp::getOutputType() const {
  return render_type_to_sql_type(getOutputBufferAttrType());
}

BufferAttrType FormulaXformOp::getOutputBufferAttrType() const {
  if (cached_output_type_ == BufferAttrType::kCOUNT) {
    ::mup::ParserX p(::mup::pckCOMMON | ::mup::pckUNIT | ::mup::pckNON_COMPLEX);
    ::mup::var_maptype vmap, cmap;
    try {
      p.SetExpr(formula_str_);
      vmap = p.GetExprVar();
      cmap = p.GetConst();
    } catch (::mup::ParserError& err) {
      THROW_RUNTIME_EX("Could not properly compile the expression \"" + formula_str_ +
                       "\". Error: " + err.GetMsg());
    } catch (std::exception& err) {
      CHECK(false) << err.what();
    }

    // TODO(croot): iterating through the types used in the expressions to get a guess
    // of what the output type would be, but this is not accurate.
    // For instance, an int / int should probably be a float, unless we require for
    // casting in the expression, which muparserx can do.
    // Nevertheless, without a way to have a callback called with the arguments used for
    // each function, we can't make determinations like this, so getting the types from
    // the variables and constants in the expression is currently the best way without
    // fully evaluating the expression. We don't want to do that tho because
    // we want evaluation to be lazy
    XformOp::DependencyOpMap dep_ops;
    auto const parent_xform = parent_xform_.lock();
    CHECK(parent_xform);
    std::for_each(vmap.begin(), vmap.end(), [this, &parent_xform](auto const& item) {
      auto const op = parent_xform->getOutputOp(item.first);
      CHECK(op) << "couldn't find op \"" << item.first << "\"";
      cached_output_type_ = get_higher_priority_buffer_attr_type(
          cached_output_type_, op->getOutputBufferAttrType());
    });

    std::for_each(cmap.begin(), cmap.end(), [this](auto const& item) {
      auto val = (::mup::Value&)(*(item.second));
      cached_output_type_ = get_higher_priority_buffer_attr_type(
          cached_output_type_, get_buffer_attr_type_from_value(val));
    });
  }

  return cached_output_type_;
}

namespace {

AggDataList execute_formula_op(const XformOp::DependencyOpResultsMap& dependency_results,
                               const std::string& formula_str) {
  ::mup::ParserX exp_parser(::mup::pckCOMMON | ::mup::pckUNIT | ::mup::pckNON_COMPLEX);
  bool has_null = false;
  for (auto const& dep_result : dependency_results) {
    auto const& dep_data_results = dep_result.second;
    CHECK_EQ(dep_data_results.size(), 1u);
    CHECK(dep_data_results[0]);
    auto const data_type = dep_data_results[0]->getType();
    switch (data_type) {
      case QueryDataType::INT:
      case QueryDataType::INT64:
      case QueryDataType::UINT:
      case QueryDataType::UINT64: {
        auto const val = dep_data_results[0]->getVal<int64_t>();
        if (isNullValue(val)) {
          has_null = true;
        } else if (!std::isfinite(val)) {
          throw std::runtime_error("The result of executing dependency \"" +
                                   dep_result.first + "\" is " +
                                   (std::isinf(val) ? "infinite" : "not a number") + ".");
        }
        exp_parser.DefineConst(dep_result.first, val);
        break;
      }
      case QueryDataType::FLOAT: {
        auto val = dep_data_results[0]->getVal<float>();
        if (isNullValue(val)) {
          has_null = true;
        } else if (!std::isfinite(val)) {
          throw std::runtime_error("The result of executing dependency \"" +
                                   dep_result.first + "\" is " +
                                   (std::isinf(val) ? "infinite" : "not a number") + ".");
        }
        exp_parser.DefineConst(dep_result.first, val);
        break;
      }
      case QueryDataType::DOUBLE: {
        auto val = dep_data_results[0]->getVal<double>();
        if (isNullValue(val)) {
          has_null = true;
        } else if (!std::isfinite(val)) {
          throw std::runtime_error("The result of executing dependency \"" +
                                   dep_result.first + "\" is " +
                                   (std::isinf(val) ? "infinite" : "not a number") + ".");
        }
        exp_parser.DefineConst(dep_result.first, val);
        break;
      }
      default:
        throw std::runtime_error(
            "Cannot evaluate formula expression. Unsupported data type " +
            to_string(dep_data_results[0]->getType()));
    }
  }

  AggDataList op_results;
  try {
    exp_parser.SetExpr(formula_str);
    auto& result = exp_parser.Eval();

    auto char_type = result.GetType();
    switch (char_type) {
      case 'i': {
        auto val = result.GetInteger();
        if (!std::isfinite(val)) {
          throw std::runtime_error(
              "The result of evaluating expression is " +
              std::string(std::isinf(val) ? "infinite" : "not a number") + ".");
        }
        op_results = {std::make_shared<AnyDataType>(
            QueryDataType::INT64, has_null ? getNullValue<int64_t>() : val)};
        break;
      }
      case 'f': {
        auto val = result.GetFloat();
        if (!std::isfinite(val)) {
          throw std::runtime_error(
              "The result of evaluating expression is " +
              std::string(std::isinf(val) ? "infinite" : "not a number") + ".");
        }
        op_results = {std::make_shared<AnyDataType>(
            QueryDataType::DOUBLE, has_null ? getNullValue<double>() : val)};
        break;
      }
      default:
        throw std::runtime_error("Cannot convert muparserx type to valid data type. '" +
                                 std::string(&char_type, 1) + "' is not supported.");
    }
  } catch (::mup::ParserError& err) {
    throw std::runtime_error(
        "Could not properly execute the expression expression. Error: " + err.GetMsg() +
        "." + (has_null ? " FWIW a null is found in the expression." : ""));
  } catch (std::runtime_error& err) {
    throw err;
  } catch (std::exception& err) {
    CHECK(false) << err.what() << " has_null:" << has_null;
  }
  return op_results;
}

}  // namespace

const XformOp::OpResult FormulaXformOp::executeOp(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const DependencyOpResultsMap& dependency_results) {
  CHECK_EQ(dependency_results.size(), dependencies_.size());
  try {
    return {true, execute_formula_op(dependency_results, formula_str_)};
  } catch (std::runtime_error& err) {
    THROW_RUNTIME_EX(std::string(*this) +
                     ": Got an error during operator evaluation: " + err.what());
  }
  return {true, {}};
}

}  // namespace QueryRenderer
