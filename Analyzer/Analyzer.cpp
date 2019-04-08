/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    Analyzer.cpp
 * @author  Wei Hong <wei@map-d.com>
 * @brief   Analyzer functions
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "Analyzer.h"
#include <glog/logging.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "../Catalog/Catalog.h"
#include "../Shared/DateConversions.h"
#include "../Shared/sql_type_to_string.h"
#include "../Shared/sqltypes.h"
#include "../Shared/unreachable.h"

namespace Analyzer {

Constant::~Constant() {
  if ((type_info.is_string() || type_info.is_geometry()) && !is_null) {
    delete constval.stringval;
  }
}

Subquery::~Subquery() {
  delete parsetree;
}

RangeTblEntry::~RangeTblEntry() {
  if (view_query != nullptr) {
    delete view_query;
  }
}

Query::~Query() {
  for (auto p : rangetable) {
    delete p;
  }
  delete order_by;
  delete next_query;
}

std::shared_ptr<Analyzer::Expr> ColumnVar::deep_copy() const {
  return makeExpr<ColumnVar>(type_info, table_id, column_id, rte_idx);
}

void ExpressionTuple::collect_rte_idx(std::set<int>& rte_idx_set) const {
  for (const auto column : tuple_) {
    column->collect_rte_idx(rte_idx_set);
  }
}

std::shared_ptr<Analyzer::Expr> ExpressionTuple::deep_copy() const {
  std::vector<std::shared_ptr<Expr>> tuple_deep_copy;
  for (const auto& column : tuple_) {
    const auto column_deep_copy =
        std::dynamic_pointer_cast<Analyzer::ColumnVar>(column->deep_copy());
    CHECK(column_deep_copy);
    tuple_deep_copy.push_back(column_deep_copy);
  }
  return makeExpr<ExpressionTuple>(tuple_deep_copy);
}

std::shared_ptr<Analyzer::Expr> Var::deep_copy() const {
  return makeExpr<Var>(type_info, table_id, column_id, rte_idx, which_row, varno);
}

std::shared_ptr<Analyzer::Expr> Constant::deep_copy() const {
  Datum d = constval;
  if ((type_info.is_string() || type_info.is_geometry()) && !is_null) {
    d.stringval = new std::string(*constval.stringval);
  }
  if (type_info.get_type() == kARRAY) {
    return makeExpr<Constant>(type_info, is_null, value_list);
  }
  return makeExpr<Constant>(type_info, is_null, d);
}

std::shared_ptr<Analyzer::Expr> UOper::deep_copy() const {
  return makeExpr<UOper>(type_info, contains_agg, optype, operand->deep_copy());
}

std::shared_ptr<Analyzer::Expr> BinOper::deep_copy() const {
  return makeExpr<BinOper>(type_info,
                           contains_agg,
                           optype,
                           qualifier,
                           left_operand->deep_copy(),
                           right_operand->deep_copy());
}

std::shared_ptr<Analyzer::Expr> Subquery::deep_copy() const {
  // not supported yet.
  CHECK(false);
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> InValues::deep_copy() const {
  std::list<std::shared_ptr<Analyzer::Expr>> new_value_list;
  for (auto p : value_list) {
    new_value_list.push_back(p->deep_copy());
  }
  return makeExpr<InValues>(arg->deep_copy(), new_value_list);
}

std::shared_ptr<Analyzer::Expr> CharLengthExpr::deep_copy() const {
  return makeExpr<CharLengthExpr>(arg->deep_copy(), calc_encoded_length);
}

std::shared_ptr<Analyzer::Expr> LikeExpr::deep_copy() const {
  return makeExpr<LikeExpr>(arg->deep_copy(),
                            like_expr->deep_copy(),
                            escape_expr ? escape_expr->deep_copy() : nullptr,
                            is_ilike,
                            is_simple);
}

std::shared_ptr<Analyzer::Expr> RegexpExpr::deep_copy() const {
  return makeExpr<RegexpExpr>(arg->deep_copy(),
                              pattern_expr->deep_copy(),
                              escape_expr ? escape_expr->deep_copy() : nullptr);
}

std::shared_ptr<Analyzer::Expr> LikelihoodExpr::deep_copy() const {
  return makeExpr<LikelihoodExpr>(arg->deep_copy(), likelihood);
}
std::shared_ptr<Analyzer::Expr> AggExpr::deep_copy() const {
  return makeExpr<AggExpr>(type_info,
                           aggtype,
                           arg == nullptr ? nullptr : arg->deep_copy(),
                           is_distinct,
                           error_rate);
}

std::shared_ptr<Analyzer::Expr> CaseExpr::deep_copy() const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      new_list;
  for (auto p : expr_pair_list) {
    new_list.emplace_back(p.first->deep_copy(), p.second->deep_copy());
  }
  return makeExpr<CaseExpr>(type_info,
                            contains_agg,
                            new_list,
                            else_expr == nullptr ? nullptr : else_expr->deep_copy());
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::deep_copy() const {
  return makeExpr<ExtractExpr>(type_info, contains_agg, field_, from_expr_->deep_copy());
}

std::shared_ptr<Analyzer::Expr> DateaddExpr::deep_copy() const {
  return makeExpr<DateaddExpr>(
      type_info, field_, number_->deep_copy(), datetime_->deep_copy());
}

std::shared_ptr<Analyzer::Expr> DatediffExpr::deep_copy() const {
  return makeExpr<DatediffExpr>(
      type_info, field_, start_->deep_copy(), end_->deep_copy());
}

std::shared_ptr<Analyzer::Expr> DatetruncExpr::deep_copy() const {
  return makeExpr<DatetruncExpr>(
      type_info, contains_agg, field_, from_expr_->deep_copy());
}

std::shared_ptr<Analyzer::Expr> OffsetInFragment::deep_copy() const {
  return makeExpr<OffsetInFragment>();
}

ExpressionPtr ArrayExpr::deep_copy() const {
  return makeExpr<Analyzer::ArrayExpr>(type_info, contained_expressions_, expr_index_);
}

SQLTypeInfo BinOper::analyze_type_info(SQLOps op,
                                       const SQLTypeInfo& left_type,
                                       const SQLTypeInfo& right_type,
                                       SQLTypeInfo* new_left_type,
                                       SQLTypeInfo* new_right_type) {
  SQLTypeInfo result_type;
  SQLTypeInfo common_type;
  *new_left_type = left_type;
  *new_right_type = right_type;
  if (IS_LOGIC(op)) {
    if (left_type.get_type() != kBOOLEAN || right_type.get_type() != kBOOLEAN) {
      throw std::runtime_error(
          "non-boolean operands cannot be used in logic operations.");
    }
    result_type = SQLTypeInfo(kBOOLEAN, false);
  } else if (IS_COMPARISON(op)) {
    if (left_type != right_type) {
      if (left_type.is_number() && right_type.is_number()) {
        common_type = common_numeric_type(left_type, right_type);
        *new_left_type = common_type;
        new_left_type->set_notnull(left_type.get_notnull());
        *new_right_type = common_type;
        new_right_type->set_notnull(right_type.get_notnull());
      } else if (left_type.is_time() && right_type.is_time()) {
        switch (left_type.get_type()) {
          case kTIMESTAMP:
            switch (right_type.get_type()) {
              case kTIME:
                throw std::runtime_error("Cannont compare between TIMESTAMP and TIME.");
                break;
              case kDATE:
                *new_left_type = SQLTypeInfo(left_type.get_type(),
                                             left_type.get_dimension(),
                                             0,
                                             left_type.get_notnull());
                *new_right_type = *new_left_type;
                new_right_type->set_notnull(right_type.get_notnull());
                break;
              case kTIMESTAMP:
                *new_left_type = SQLTypeInfo(
                    kTIMESTAMP,
                    std::max(left_type.get_dimension(), right_type.get_dimension()),
                    0,
                    left_type.get_notnull());
                *new_right_type = SQLTypeInfo(
                    kTIMESTAMP,
                    std::max(left_type.get_dimension(), right_type.get_dimension()),
                    0,
                    right_type.get_notnull());
                break;
              default:
                CHECK(false);
            }
            break;
          case kTIME:
            switch (right_type.get_type()) {
              case kTIMESTAMP:
                throw std::runtime_error("Cannont compare between TIME and TIMESTAMP.");
                break;
              case kDATE:
                throw std::runtime_error("Cannont compare between TIME and DATE.");
                break;
              case kTIME:
                *new_left_type = SQLTypeInfo(
                    kTIME,
                    std::max(left_type.get_dimension(), right_type.get_dimension()),
                    0,
                    left_type.get_notnull());
                *new_right_type = SQLTypeInfo(
                    kTIME,
                    std::max(left_type.get_dimension(), right_type.get_dimension()),
                    0,
                    right_type.get_notnull());
                break;
              default:
                CHECK(false);
            }
            break;
          case kDATE:
            switch (right_type.get_type()) {
              case kTIMESTAMP:
                *new_left_type = SQLTypeInfo(right_type.get_type(),
                                             right_type.get_dimension(),
                                             0,
                                             left_type.get_notnull());
                *new_right_type = *new_left_type;
                new_right_type->set_notnull(right_type.get_notnull());
                break;
              case kDATE:
                *new_left_type = SQLTypeInfo(left_type.get_type(),
                                             left_type.get_dimension(),
                                             0,
                                             left_type.get_notnull());
                *new_right_type = *new_left_type;
                new_right_type->set_notnull(right_type.get_notnull());
                break;
              case kTIME:
                throw std::runtime_error("Cannont compare between DATE and TIME.");
                break;
              default:
                CHECK(false);
            }
            break;
          default:
            CHECK(false);
        }
      } else if (left_type.is_string() && right_type.is_time()) {
        *new_left_type = right_type;
        new_left_type->set_notnull(left_type.get_notnull());
        *new_right_type = right_type;
      } else if (left_type.is_time() && right_type.is_string()) {
        *new_left_type = left_type;
        *new_right_type = left_type;
        new_right_type->set_notnull(right_type.get_notnull());
      } else if (left_type.is_string() && right_type.is_string()) {
        *new_left_type = left_type;
        *new_right_type = right_type;
      } else if (left_type.is_boolean() && right_type.is_boolean()) {
        const bool notnull = left_type.get_notnull() && right_type.get_notnull();
        common_type = SQLTypeInfo(kBOOLEAN, notnull);
        *new_left_type = common_type;
        *new_right_type = common_type;
      } else {
        throw std::runtime_error("Cannot compare between " + left_type.get_type_name() +
                                 " and " + right_type.get_type_name());
      }
    }
    result_type = SQLTypeInfo(kBOOLEAN, false);
  } else if (op == kMINUS &&
             (left_type.get_type() == kDATE || left_type.get_type() == kTIMESTAMP) &&
             right_type.is_timeinterval()) {
    *new_left_type = left_type;
    *new_right_type = right_type;
    result_type = left_type;
  } else if (IS_ARITHMETIC(op)) {
    if (!(left_type.is_number() || left_type.is_timeinterval()) ||
        !(right_type.is_number() || right_type.is_timeinterval())) {
      throw std::runtime_error("non-numeric operands in arithmetic operations.");
    }
    if (op == kMODULO && (!left_type.is_integer() || !right_type.is_integer())) {
      throw std::runtime_error("non-integer operands in modulo operation.");
    }
    common_type = common_numeric_type(left_type, right_type);
    if (common_type.is_decimal()) {
      if (op == kMULTIPLY) {
        // Decimal multiplication requires common_type adjustment:
        // dimension and scale of the result should be increased.
        auto new_dimension = left_type.get_dimension() + right_type.get_dimension();
        // If new dimension is over 20 digits, the result may overflow, or it may not.
        // Rely on the runtime overflow detection rather than a static check here.
        if (common_type.get_dimension() < new_dimension) {
          common_type.set_dimension(new_dimension);
        }
        common_type.set_scale(left_type.get_scale() + right_type.get_scale());
      } else if (op == kPLUS || op == kMINUS) {
        // Scale should remain the same but dimension could actually go up
        common_type.set_dimension(common_type.get_dimension() + 1);
      }
    }
    *new_left_type = common_type;
    new_left_type->set_notnull(left_type.get_notnull());
    *new_right_type = common_type;
    new_right_type->set_notnull(right_type.get_notnull());
    if (op == kMULTIPLY) {
      new_left_type->set_scale(left_type.get_scale());
      new_right_type->set_scale(right_type.get_scale());
    }
    result_type = common_type;
  } else {
    throw std::runtime_error("invalid binary operator type.");
  }
  result_type.set_notnull(left_type.get_notnull() && right_type.get_notnull());
  return result_type;
}

SQLTypeInfo BinOper::common_string_type(const SQLTypeInfo& type1,
                                        const SQLTypeInfo& type2) {
  SQLTypeInfo common_type;
  EncodingType comp = kENCODING_NONE;
  int comp_param = 0;
  CHECK(type1.is_string() && type2.is_string());
  // if type1 and type2 have the same DICT encoding then keep it
  // otherwise, they must be decompressed
  if (type1.get_compression() == kENCODING_DICT &&
      type2.get_compression() == kENCODING_DICT) {
    if (type1.get_comp_param() == type2.get_comp_param() ||
        type1.get_comp_param() == TRANSIENT_DICT(type2.get_comp_param())) {
      comp = kENCODING_DICT;
      comp_param = std::min(type1.get_comp_param(), type2.get_comp_param());
    }
  } else if (type1.get_compression() == kENCODING_DICT &&
             type2.get_compression() == kENCODING_NONE) {
    comp_param = type1.get_comp_param();
  } else if (type1.get_compression() == kENCODING_NONE &&
             type2.get_compression() == kENCODING_DICT) {
    comp_param = type2.get_comp_param();
  } else {
    comp_param = std::max(type1.get_comp_param(),
                          type2.get_comp_param());  // preserve previous comp_param if set
  }
  const bool notnull = type1.get_notnull() && type2.get_notnull();
  if (type1.get_type() == kTEXT || type2.get_type() == kTEXT) {
    common_type = SQLTypeInfo(kTEXT, 0, 0, notnull, comp, comp_param, kNULLT);
    return common_type;
  }
  common_type = SQLTypeInfo(kVARCHAR,
                            std::max(type1.get_dimension(), type2.get_dimension()),
                            0,
                            notnull,
                            comp,
                            comp_param,
                            kNULLT);
  return common_type;
}

SQLTypeInfo BinOper::common_numeric_type(const SQLTypeInfo& type1,
                                         const SQLTypeInfo& type2) {
  SQLTypeInfo common_type;
  const bool notnull = type1.get_notnull() && type2.get_notnull();
  if (type1.get_type() == type2.get_type()) {
    CHECK(((type1.is_number() || type1.is_timeinterval()) &&
           (type2.is_number() || type2.is_timeinterval())) ||
          (type1.is_boolean() && type2.is_boolean()));
    common_type = SQLTypeInfo(type1.get_type(),
                              std::max(type1.get_dimension(), type2.get_dimension()),
                              std::max(type1.get_scale(), type2.get_scale()),
                              notnull);
    return common_type;
  }
  std::string timeinterval_op_error{
      "Operator type not supported for time interval arithmetic: "};
  if (type1.is_timeinterval()) {
    if (!type2.is_integer()) {
      throw std::runtime_error(timeinterval_op_error + type2.get_type_name());
    }
    return type1;
  }
  if (type2.is_timeinterval()) {
    if (!type1.is_integer()) {
      throw std::runtime_error(timeinterval_op_error + type1.get_type_name());
    }
    return type2;
  }
  CHECK(type1.is_number() && type2.is_number());
  switch (type1.get_type()) {
    case kTINYINT:
      switch (type2.get_type()) {
        case kSMALLINT:
          common_type = SQLTypeInfo(kSMALLINT, notnull);
          break;
        case kINT:
          common_type = SQLTypeInfo(kINT, notnull);
          break;
        case kBIGINT:
          common_type = SQLTypeInfo(kBIGINT, notnull);
          break;
        case kFLOAT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kDOUBLE:
          common_type = SQLTypeInfo(kDOUBLE, notnull);
          break;
        case kNUMERIC:
        case kDECIMAL:
          common_type =
              SQLTypeInfo(kDECIMAL,
                          std::max(5 + type2.get_scale(), type2.get_dimension()),
                          type2.get_scale(),
                          notnull);
          break;
        default:
          CHECK(false);
      }
      break;
    case kSMALLINT:
      switch (type2.get_type()) {
        case kTINYINT:
          common_type = SQLTypeInfo(kSMALLINT, notnull);
          break;
        case kINT:
          common_type = SQLTypeInfo(kINT, notnull);
          break;
        case kBIGINT:
          common_type = SQLTypeInfo(kBIGINT, notnull);
          break;
        case kFLOAT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kDOUBLE:
          common_type = SQLTypeInfo(kDOUBLE, notnull);
          break;
        case kNUMERIC:
        case kDECIMAL:
          common_type =
              SQLTypeInfo(kDECIMAL,
                          std::max(5 + type2.get_scale(), type2.get_dimension()),
                          type2.get_scale(),
                          notnull);
          break;
        default:
          CHECK(false);
      }
      break;
    case kINT:
      switch (type2.get_type()) {
        case kTINYINT:
          common_type = SQLTypeInfo(kINT, notnull);
          break;
        case kSMALLINT:
          common_type = SQLTypeInfo(kINT, notnull);
          break;
        case kBIGINT:
          common_type = SQLTypeInfo(kBIGINT, notnull);
          break;
        case kFLOAT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kDOUBLE:
          common_type = SQLTypeInfo(kDOUBLE, notnull);
          break;
        case kNUMERIC:
        case kDECIMAL:
          common_type = SQLTypeInfo(
              kDECIMAL,
              std::max(std::min(19, 10 + type2.get_scale()), type2.get_dimension()),
              type2.get_scale(),
              notnull);
          break;
        default:
          CHECK(false);
      }
      break;
    case kBIGINT:
      switch (type2.get_type()) {
        case kTINYINT:
          common_type = SQLTypeInfo(kBIGINT, notnull);
          break;
        case kSMALLINT:
          common_type = SQLTypeInfo(kBIGINT, notnull);
          break;
        case kINT:
          common_type = SQLTypeInfo(kBIGINT, notnull);
          break;
        case kFLOAT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kDOUBLE:
          common_type = SQLTypeInfo(kDOUBLE, notnull);
          break;
        case kNUMERIC:
        case kDECIMAL:
          common_type = SQLTypeInfo(kDECIMAL, 19, type2.get_scale(), notnull);
          break;
        default:
          CHECK(false);
      }
      break;
    case kFLOAT:
      switch (type2.get_type()) {
        case kTINYINT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kSMALLINT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kINT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kBIGINT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kDOUBLE:
          common_type = SQLTypeInfo(kDOUBLE, notnull);
          break;
        case kNUMERIC:
        case kDECIMAL:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        default:
          CHECK(false);
      }
      break;
    case kDOUBLE:
      switch (type2.get_type()) {
        case kTINYINT:
        case kSMALLINT:
        case kINT:
        case kBIGINT:
        case kFLOAT:
        case kNUMERIC:
        case kDECIMAL:
          common_type = SQLTypeInfo(kDOUBLE, notnull);
          break;
        default:
          CHECK(false);
      }
      break;
    case kNUMERIC:
    case kDECIMAL:
      switch (type2.get_type()) {
        case kSMALLINT:
          common_type =
              SQLTypeInfo(kDECIMAL,
                          std::max(5 + type1.get_scale(), type1.get_dimension()),
                          type1.get_scale(),
                          notnull);
          break;
        case kINT:
          common_type = SQLTypeInfo(
              kDECIMAL,
              std::max(std::min(19, 10 + type1.get_scale()), type2.get_dimension()),
              type1.get_scale(),
              notnull);
          break;
        case kBIGINT:
          common_type = SQLTypeInfo(kDECIMAL, 19, type1.get_scale(), notnull);
          break;
        case kFLOAT:
          common_type = SQLTypeInfo(kFLOAT, notnull);
          break;
        case kDOUBLE:
          common_type = SQLTypeInfo(kDOUBLE, notnull);
          break;
        case kNUMERIC:
        case kDECIMAL: {
          int common_scale = std::max(type1.get_scale(), type2.get_scale());
          common_type = SQLTypeInfo(kDECIMAL,
                                    std::max(type1.get_dimension() - type1.get_scale(),
                                             type2.get_dimension() - type2.get_scale()) +
                                        common_scale,
                                    common_scale,
                                    notnull);
        } break;
        default:
          CHECK(false);
      }
      break;
    default:
      CHECK(false);
  }
  common_type.set_fixed_size();
  return common_type;
}

std::shared_ptr<Analyzer::Expr> Expr::decompress() {
  if (type_info.get_compression() == kENCODING_NONE) {
    return shared_from_this();
  }
  SQLTypeInfo new_type_info(type_info.get_type(),
                            type_info.get_dimension(),
                            type_info.get_scale(),
                            type_info.get_notnull(),
                            kENCODING_NONE,
                            0,
                            type_info.get_subtype());
  return makeExpr<UOper>(new_type_info, contains_agg, kCAST, shared_from_this());
}

std::shared_ptr<Analyzer::Expr> Expr::add_cast(const SQLTypeInfo& new_type_info) {
  if (new_type_info == type_info) {
    return shared_from_this();
  }
  if (new_type_info.is_string() && type_info.is_string() &&
      new_type_info.get_compression() == kENCODING_DICT &&
      type_info.get_compression() == kENCODING_DICT &&
      (new_type_info.get_comp_param() == type_info.get_comp_param() ||
       new_type_info.get_comp_param() == TRANSIENT_DICT(type_info.get_comp_param()))) {
    return shared_from_this();
  }
  if (!type_info.is_castable(new_type_info)) {
    throw std::runtime_error("Cannot CAST from " + type_info.get_type_name() + " to " +
                             new_type_info.get_type_name());
  }
  // @TODO(wei) temporary restriction until executor can support this.
  if (typeid(*this) != typeid(Constant) && new_type_info.is_string() &&
      new_type_info.get_compression() == kENCODING_DICT &&
      new_type_info.get_comp_param() <= TRANSIENT_DICT_ID) {
    if (type_info.is_string() && type_info.get_compression() != kENCODING_DICT) {
      throw std::runtime_error(
          "Cannot group by string columns which are not dictionary encoded.");
    }
    throw std::runtime_error(
        "Internal error: Cannot apply transient dictionary encoding to non-literal "
        "expression "
        "yet.");
  }
  return makeExpr<UOper>(new_type_info, contains_agg, kCAST, shared_from_this());
}

namespace {

struct IntFracRepr {
  const int64_t integral;
  const int64_t fractional;
  const int64_t scale;
};

IntFracRepr decimal_to_int_frac(const int64_t dec, const SQLTypeInfo& ti) {
  if (ti.get_scale() > 18) {
    auto truncated_ti = ti;
    truncated_ti.set_scale(18);
    auto truncated_dec = dec;
    for (int i = 0; i < ti.get_scale() - 18; ++i) {
      truncated_dec /= 10;
    }
    return decimal_to_int_frac(truncated_dec, truncated_ti);
  }
  int64_t integral_part = dec;
  int64_t scale = 1;
  for (int i = 0; i < ti.get_scale(); i++) {
    integral_part /= 10;
    scale *= 10;
  }
  return {integral_part, dec - integral_part * scale, scale};
}

}  // namespace

void Constant::cast_number(const SQLTypeInfo& new_type_info) {
  switch (type_info.get_type()) {
    case kTINYINT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          break;
        case kINT:
          constval.intval = (int32_t)constval.tinyintval;
          break;
        case kSMALLINT:
          constval.smallintval = (int16_t)constval.tinyintval;
          break;
        case kBIGINT:
          constval.bigintval = (int64_t)constval.tinyintval;
          break;
        case kDOUBLE:
          constval.doubleval = (double)constval.tinyintval;
          break;
        case kFLOAT:
          constval.floatval = (float)constval.tinyintval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = (int64_t)constval.tinyintval;
          for (int i = 0; i < new_type_info.get_scale(); i++) {
            constval.bigintval *= 10;
          }
          break;
        default:
          CHECK(false);
      }
      break;
    case kINT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = (int8_t)constval.intval;
          break;
        case kINT:
          break;
        case kSMALLINT:
          constval.smallintval = (int16_t)constval.intval;
          break;
        case kBIGINT:
          constval.bigintval = (int64_t)constval.intval;
          break;
        case kDOUBLE:
          constval.doubleval = (double)constval.intval;
          break;
        case kFLOAT:
          constval.floatval = (float)constval.intval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = (int64_t)constval.intval;
          for (int i = 0; i < new_type_info.get_scale(); i++) {
            constval.bigintval *= 10;
          }
          break;
        default:
          CHECK(false);
      }
      break;
    case kSMALLINT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = (int8_t)constval.smallintval;
          break;
        case kINT:
          constval.intval = (int32_t)constval.smallintval;
          break;
        case kSMALLINT:
          break;
        case kBIGINT:
          constval.bigintval = (int64_t)constval.smallintval;
          break;
        case kDOUBLE:
          constval.doubleval = (double)constval.smallintval;
          break;
        case kFLOAT:
          constval.floatval = (float)constval.smallintval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = (int64_t)constval.smallintval;
          for (int i = 0; i < new_type_info.get_scale(); i++) {
            constval.bigintval *= 10;
          }
          break;
        default:
          CHECK(false);
      }
      break;
    case kBIGINT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = (int8_t)constval.bigintval;
          break;
        case kINT:
          constval.intval = (int32_t)constval.bigintval;
          break;
        case kSMALLINT:
          constval.smallintval = (int16_t)constval.bigintval;
          break;
        case kBIGINT:
          break;
        case kDOUBLE:
          constval.doubleval = (double)constval.bigintval;
          break;
        case kFLOAT:
          constval.floatval = (float)constval.bigintval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          for (int i = 0; i < new_type_info.get_scale(); i++) {
            constval.bigintval *= 10;
          }
          break;
        default:
          CHECK(false);
      }
      break;
    case kDOUBLE:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = (int8_t)constval.doubleval;
          break;
        case kINT:
          constval.intval = (int32_t)constval.doubleval;
          break;
        case kSMALLINT:
          constval.smallintval = (int16_t)constval.doubleval;
          break;
        case kBIGINT:
          constval.bigintval = (int64_t)constval.doubleval;
          break;
        case kDOUBLE:
          break;
        case kFLOAT:
          constval.floatval = (float)constval.doubleval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          for (int i = 0; i < new_type_info.get_scale(); i++) {
            constval.doubleval *= 10;
          }
          constval.bigintval = (int64_t)constval.doubleval;
          break;
        default:
          CHECK(false);
      }
      break;
    case kFLOAT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = (int8_t)constval.floatval;
          break;
        case kINT:
          constval.intval = (int32_t)constval.floatval;
          break;
        case kSMALLINT:
          constval.smallintval = (int16_t)constval.floatval;
          break;
        case kBIGINT:
          constval.bigintval = (int64_t)constval.floatval;
          break;
        case kDOUBLE:
          constval.doubleval = (double)constval.floatval;
          break;
        case kFLOAT:
          break;
        case kNUMERIC:
        case kDECIMAL:
          for (int i = 0; i < new_type_info.get_scale(); i++) {
            constval.floatval *= 10;
          }
          constval.bigintval = (int64_t)constval.floatval;
          break;
        default:
          CHECK(false);
      }
      break;
    case kNUMERIC:
    case kDECIMAL:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          for (int i = 0; i < type_info.get_scale(); i++) {
            constval.bigintval /= 10;
          }
          constval.tinyintval = (int8_t)constval.bigintval;
          break;
        case kINT:
          for (int i = 0; i < type_info.get_scale(); i++) {
            constval.bigintval /= 10;
          }
          constval.intval = (int32_t)constval.bigintval;
          break;
        case kSMALLINT:
          for (int i = 0; i < type_info.get_scale(); i++) {
            constval.bigintval /= 10;
          }
          constval.smallintval = (int16_t)constval.bigintval;
          break;
        case kBIGINT:
          for (int i = 0; i < type_info.get_scale(); i++) {
            constval.bigintval /= 10;
          }
          break;
        case kDOUBLE: {
          const auto int_frac = decimal_to_int_frac(constval.bigintval, type_info);
          constval.doubleval = int_frac.integral +
                               static_cast<double>(int_frac.fractional) / int_frac.scale;
          break;
        }
        case kFLOAT: {
          const auto int_frac = decimal_to_int_frac(constval.bigintval, type_info);
          constval.floatval = int_frac.integral +
                              static_cast<double>(int_frac.fractional) / int_frac.scale;
          break;
        }
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = convert_decimal_value_to_scale(
              constval.bigintval, type_info, new_type_info);
          break;
        default:
          CHECK(false);
      }
      break;
    case kTIMESTAMP:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = (int8_t)constval.timeval;
          break;
        case kINT:
          constval.intval = (int32_t)constval.timeval;
          break;
        case kSMALLINT:
          constval.smallintval = (int16_t)constval.timeval;
          break;
        case kBIGINT:
          constval.bigintval = (int64_t)constval.timeval;
          break;
        case kDOUBLE:
          constval.doubleval = (double)constval.timeval;
          break;
        case kFLOAT:
          constval.floatval = (float)constval.timeval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = (int64_t)constval.timeval;
          for (int i = 0; i < new_type_info.get_scale(); i++) {
            constval.bigintval *= 10;
          }
          break;
        default:
          CHECK(false);
      }
      break;
    case kBOOLEAN:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = constval.boolval ? 1 : 0;
          break;
        case kINT:
          constval.intval = constval.boolval ? 1 : 0;
          break;
        case kSMALLINT:
          constval.smallintval = constval.boolval ? 1 : 0;
          break;
        case kBIGINT:
          constval.bigintval = constval.boolval ? 1 : 0;
          break;
        case kDOUBLE:
          constval.doubleval = constval.boolval ? 1 : 0;
          break;
        case kFLOAT:
          constval.floatval = constval.boolval ? 1 : 0;
          break;
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = constval.boolval ? 1 : 0;
          for (int i = 0; i < new_type_info.get_scale(); i++) {
            constval.bigintval *= 10;
          }
          break;
        default:
          CHECK(false);
      }
      break;
    default:
      CHECK(false);
  }
  type_info = new_type_info;
}

void Constant::cast_string(const SQLTypeInfo& new_type_info) {
  std::string* s = constval.stringval;
  if (s != nullptr && new_type_info.get_type() != kTEXT &&
      static_cast<size_t>(new_type_info.get_dimension()) < s->length()) {
    // truncate string
    constval.stringval = new std::string(s->substr(0, new_type_info.get_dimension()));
    delete s;
  }
  type_info = new_type_info;
}

void Constant::cast_from_string(const SQLTypeInfo& new_type_info) {
  std::string* s = constval.stringval;
  SQLTypeInfo ti = new_type_info;
  constval = StringToDatum(*s, ti);
  delete s;
  type_info = new_type_info;
}

void Constant::cast_to_string(const SQLTypeInfo& str_type_info) {
  const auto str_val = DatumToString(constval, type_info);
  constval.stringval = new std::string(str_val);
  if (str_type_info.get_type() != kTEXT &&
      constval.stringval->length() > static_cast<size_t>(str_type_info.get_dimension())) {
    // truncate the string
    *constval.stringval = constval.stringval->substr(0, str_type_info.get_dimension());
  }
  type_info = str_type_info;
}

void Constant::do_cast(const SQLTypeInfo& new_type_info) {
  if (type_info == new_type_info) {
    return;
  }
  if (new_type_info.is_number() &&
      (type_info.is_number() || type_info.get_type() == kTIMESTAMP ||
       type_info.get_type() == kBOOLEAN)) {
    cast_number(new_type_info);
  } else if (new_type_info.is_geometry() && type_info.is_string()) {
    type_info = new_type_info;
  } else if (new_type_info.is_geometry() &&
             type_info.get_type() == new_type_info.get_type()) {
    type_info = new_type_info;
  } else if (new_type_info.is_boolean() && type_info.is_boolean()) {
    type_info = new_type_info;
  } else if (new_type_info.is_string() && type_info.is_string()) {
    cast_string(new_type_info);
  } else if (type_info.is_string() || type_info.get_type() == kVARCHAR) {
    cast_from_string(new_type_info);
  } else if (new_type_info.is_string()) {
    cast_to_string(new_type_info);
  } else if (new_type_info.get_type() == kDATE && type_info.get_type() == kDATE) {
    if (new_type_info.is_date_in_days()) {
      const int64_t val = constval.timeval;
      constval.timeval = DateConverters::get_epoch_days_from_seconds(val);
    }
    type_info = new_type_info;
  } else if (new_type_info.get_type() == kDATE && type_info.get_type() == kTIMESTAMP) {
    type_info = new_type_info;
    constval.timeval = type_info.is_high_precision_timestamp()
                           ? ExtractFromTimeHighPrecision(
                                 kEPOCH,
                                 constval.timeval,
                                 get_timestamp_precision_scale(type_info.get_dimension()))
                           : DateTruncate(dtDAY, constval.timeval);
    if (new_type_info.is_date_in_days()) {
      const int64_t val = constval.timeval;
      constval.timeval = DateConverters::get_epoch_days_from_seconds(val);
    }
  } else if (type_info.get_type() == kTIMESTAMP &&
             new_type_info.get_type() == kTIMESTAMP) {
    if (type_info.get_dimension() != new_type_info.get_dimension()) {
      const auto result_unit =
          timestamp_precisions_lookup_.find(new_type_info.get_dimension());
      constval.timeval =
          type_info.get_dimension() < new_type_info.get_dimension()
              ? DateTruncateAlterPrecisionScaleUp(
                    result_unit->second,
                    constval.timeval,
                    get_timestamp_precision_scale(new_type_info.get_dimension() -
                                                  type_info.get_dimension()))
              : DateTruncateAlterPrecisionScaleDown(
                    result_unit->second,
                    constval.timeval,
                    get_timestamp_precision_scale(type_info.get_dimension() -
                                                  new_type_info.get_dimension()));
    }
    type_info = new_type_info;
  } else if (new_type_info.is_array() && type_info.is_array()) {
    auto new_sub_ti = new_type_info.get_elem_type();
    for (auto& v : value_list) {
      auto c = std::dynamic_pointer_cast<Analyzer::Constant>(v);
      if (!c) {
        throw std::runtime_error("Invalid array cast.");
      }
      c->do_cast(new_sub_ti);
    }
    type_info = new_type_info;
  } else if (get_is_null() && (new_type_info.is_number() || new_type_info.is_time() ||
                               new_type_info.is_string())) {
    type_info = new_type_info;
    set_null_value();
  } else {
    throw std::runtime_error("Invalid cast.");
  }
}

void Constant::set_null_value() {
  switch (type_info.get_type()) {
    case kBOOLEAN:
      constval.boolval = NULL_BOOLEAN;
      break;
    case kTINYINT:
      constval.tinyintval = NULL_TINYINT;
      break;
    case kINT:
      constval.intval = NULL_INT;
      break;
    case kSMALLINT:
      constval.smallintval = NULL_SMALLINT;
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      constval.bigintval = NULL_BIGINT;
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
// @TODO(alex): store it as 64 bit on ARMv7l and remove the ifdef
#ifdef __ARM_ARCH_7A__
      static_assert(sizeof(time_t) == 4, "Unsupported time_t size");
      constval.timeval = NULL_INT;
#else
      static_assert(sizeof(time_t) == 8, "Unsupported time_t size");
      constval.timeval = NULL_BIGINT;
#endif
      break;
    case kVARCHAR:
    case kCHAR:
    case kTEXT:
      constval.stringval = nullptr;
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      constval.stringval = nullptr;
      break;
    case kFLOAT:
      constval.floatval = NULL_FLOAT;
      break;
    case kDOUBLE:
      constval.doubleval = NULL_DOUBLE;
      break;
    case kNULLT:
      constval.bigintval = 0;
      break;
    default:
      CHECK(false);
  }
}

std::shared_ptr<Analyzer::Expr> Constant::add_cast(const SQLTypeInfo& new_type_info) {
  if (is_null) {
    type_info = new_type_info;
    set_null_value();
    return shared_from_this();
  }
  if (new_type_info.get_compression() != type_info.get_compression()) {
    if (new_type_info.get_compression() != kENCODING_NONE) {
      SQLTypeInfo new_ti = new_type_info;
      if (new_ti.get_compression() != kENCODING_DATE_IN_DAYS) {
        new_ti.set_compression(kENCODING_NONE);
      }
      do_cast(new_ti);
    }
    return Expr::add_cast(new_type_info);
  }
  do_cast(new_type_info);
  return shared_from_this();
}

std::shared_ptr<Analyzer::Expr> UOper::add_cast(const SQLTypeInfo& new_type_info) {
  if (optype != kCAST) {
    return Expr::add_cast(new_type_info);
  }
  if (type_info.is_string() && new_type_info.is_string() &&
      new_type_info.get_compression() == kENCODING_DICT &&
      type_info.get_compression() == kENCODING_NONE) {
    const SQLTypeInfo oti = operand->get_type_info();
    if (oti.is_string() && oti.get_compression() == kENCODING_DICT &&
        (oti.get_comp_param() == new_type_info.get_comp_param() ||
         oti.get_comp_param() == TRANSIENT_DICT(new_type_info.get_comp_param()))) {
      auto result = operand;
      operand = nullptr;
      return result;
    }
  }
  return Expr::add_cast(new_type_info);
}

std::shared_ptr<Analyzer::Expr> CaseExpr::add_cast(const SQLTypeInfo& new_type_info) {
  SQLTypeInfo ti = new_type_info;
  if (new_type_info.is_string() && new_type_info.get_compression() == kENCODING_DICT &&
      new_type_info.get_comp_param() == TRANSIENT_DICT_ID && type_info.is_string() &&
      type_info.get_compression() == kENCODING_NONE &&
      type_info.get_comp_param() > TRANSIENT_DICT_ID) {
    ti.set_comp_param(TRANSIENT_DICT(type_info.get_comp_param()));
  }

  for (auto& p : expr_pair_list) {
    p.second = p.second->add_cast(ti);
  }
  if (else_expr != nullptr) {
    else_expr = else_expr->add_cast(ti);
  }
  type_info = ti;
  return shared_from_this();
}

std::shared_ptr<Analyzer::Expr> Subquery::add_cast(const SQLTypeInfo& new_type_info) {
  // not supported yet.
  CHECK(false);
  return nullptr;
}

void RangeTblEntry::add_all_column_descs(const Catalog_Namespace::Catalog& catalog) {
  column_descs =
      catalog.getAllColumnMetadataForTable(table_desc->tableId, true, true, true);
}

void RangeTblEntry::expand_star_in_targetlist(
    const Catalog_Namespace::Catalog& catalog,
    std::vector<std::shared_ptr<TargetEntry>>& tlist,
    int rte_idx) {
  column_descs =
      catalog.getAllColumnMetadataForTable(table_desc->tableId, false, true, true);
  for (auto col_desc : column_descs) {
    auto cv = makeExpr<ColumnVar>(
        col_desc->columnType, table_desc->tableId, col_desc->columnId, rte_idx);
    auto tle = std::make_shared<TargetEntry>(col_desc->columnName, cv, false);
    tlist.push_back(tle);
  }
}

const ColumnDescriptor* RangeTblEntry::get_column_desc(
    const Catalog_Namespace::Catalog& catalog,
    const std::string& name) {
  for (auto cd : column_descs) {
    if (cd->columnName == name) {
      return cd;
    }
  }
  const ColumnDescriptor* cd = catalog.getMetadataForColumn(table_desc->tableId, name);
  if (cd != nullptr) {
    column_descs.push_back(cd);
  }
  return cd;
}

int Query::get_rte_idx(const std::string& name) const {
  int rte_idx = 0;
  for (auto rte : rangetable) {
    if (rte->get_rangevar() == name) {
      return rte_idx;
    }
    rte_idx++;
  }
  return -1;
}

void Query::add_rte(RangeTblEntry* rte) {
  rangetable.push_back(rte);
}

void ColumnVar::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  if (!groupby.empty()) {
    for (auto e : groupby) {
      auto c = std::dynamic_pointer_cast<ColumnVar>(e);
      if (c && table_id == c->get_table_id() && column_id == c->get_column_id()) {
        return;
      }
    }
  }
  throw std::runtime_error(
      "expressions in the SELECT or HAVING clause must be an aggregate function or an "
      "expression "
      "over GROUP BY columns.");
}

void Var::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  if (which_row != kGROUPBY) {
    throw std::runtime_error("Internal error: invalid VAR in GROUP BY or HAVING.");
  }
}

void UOper::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  operand->check_group_by(groupby);
}

void BinOper::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  left_operand->check_group_by(groupby);
  right_operand->check_group_by(groupby);
}

namespace {

template <class T>
bool expr_is(const std::shared_ptr<Analyzer::Expr>& expr) {
  return std::dynamic_pointer_cast<T>(expr) != nullptr;
}

}  // namespace

bool BinOper::simple_predicate_has_simple_cast(
    const std::shared_ptr<Analyzer::Expr> cast_operand,
    const std::shared_ptr<Analyzer::Expr> const_operand) {
  if (expr_is<UOper>(cast_operand) && expr_is<Constant>(const_operand)) {
    auto u_expr = std::dynamic_pointer_cast<UOper>(cast_operand);
    if (u_expr->get_optype() != kCAST) {
      return false;
    }
    if (!(expr_is<Analyzer::ColumnVar>(u_expr->get_own_operand()) &&
          !expr_is<Analyzer::Var>(u_expr->get_own_operand()))) {
      return false;
    }
    const auto& ti = u_expr->get_type_info();
    if (ti.is_time() && u_expr->get_operand()->get_type_info().is_time()) {
      // Allow casts between time types to pass through
      return true;
    } else if (ti.is_integer() && u_expr->get_operand()->get_type_info().is_integer()) {
      // Allow casts between integer types to pass through
      return true;
    }
  }
  return false;
}

std::shared_ptr<Analyzer::Expr> BinOper::normalize_simple_predicate(int& rte_idx) const {
  rte_idx = -1;
  if (!IS_COMPARISON(optype) || qualifier != kONE) {
    return nullptr;
  }
  if (expr_is<UOper>(left_operand)) {
    if (BinOper::simple_predicate_has_simple_cast(left_operand, right_operand)) {
      auto uo = std::dynamic_pointer_cast<UOper>(left_operand);
      auto cv = std::dynamic_pointer_cast<ColumnVar>(uo->get_own_operand());
      rte_idx = cv->get_rte_idx();
      return this->deep_copy();
    }
  } else if (expr_is<UOper>(right_operand)) {
    if (BinOper::simple_predicate_has_simple_cast(right_operand, left_operand)) {
      auto uo = std::dynamic_pointer_cast<UOper>(right_operand);
      auto cv = std::dynamic_pointer_cast<ColumnVar>(uo->get_own_operand());
      rte_idx = cv->get_rte_idx();
      return makeExpr<BinOper>(type_info,
                               contains_agg,
                               COMMUTE_COMPARISON(optype),
                               qualifier,
                               right_operand->deep_copy(),
                               left_operand->deep_copy());
    }
  } else if (expr_is<ColumnVar>(left_operand) && !expr_is<Var>(left_operand) &&
             expr_is<Constant>(right_operand)) {
    auto cv = std::dynamic_pointer_cast<ColumnVar>(left_operand);
    rte_idx = cv->get_rte_idx();
    return this->deep_copy();
  } else if (expr_is<Constant>(left_operand) && expr_is<ColumnVar>(right_operand) &&
             !expr_is<Var>(right_operand)) {
    auto cv = std::dynamic_pointer_cast<ColumnVar>(right_operand);
    rte_idx = cv->get_rte_idx();
    return makeExpr<BinOper>(type_info,
                             contains_agg,
                             COMMUTE_COMPARISON(optype),
                             qualifier,
                             right_operand->deep_copy(),
                             left_operand->deep_copy());
  }
  return nullptr;
}

void ColumnVar::group_predicates(std::list<const Expr*>& scan_predicates,
                                 std::list<const Expr*>& join_predicates,
                                 std::list<const Expr*>& const_predicates) const {
  if (type_info.get_type() == kBOOLEAN) {
    scan_predicates.push_back(this);
  }
}

void UOper::group_predicates(std::list<const Expr*>& scan_predicates,
                             std::list<const Expr*>& join_predicates,
                             std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  operand->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void BinOper::group_predicates(std::list<const Expr*>& scan_predicates,
                               std::list<const Expr*>& join_predicates,
                               std::list<const Expr*>& const_predicates) const {
  if (optype == kAND) {
    left_operand->group_predicates(scan_predicates, join_predicates, const_predicates);
    right_operand->group_predicates(scan_predicates, join_predicates, const_predicates);
    return;
  }
  std::set<int> rte_idx_set;
  left_operand->collect_rte_idx(rte_idx_set);
  right_operand->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

namespace {

bool is_expr_nullable(const Analyzer::Expr* expr) {
  const auto const_expr = dynamic_cast<const Analyzer::Constant*>(expr);
  if (const_expr) {
    return const_expr->get_is_null();
  }
  const auto& expr_ti = expr->get_type_info();
  return !expr_ti.get_notnull();
}

bool is_in_values_nullable(const std::shared_ptr<Analyzer::Expr>& a,
                           const std::list<std::shared_ptr<Analyzer::Expr>>& l) {
  if (is_expr_nullable(a.get())) {
    return true;
  }
  for (const auto& v : l) {
    if (is_expr_nullable(v.get())) {
      return true;
    }
  }
  return false;
}

}  // namespace

InValues::InValues(std::shared_ptr<Analyzer::Expr> a,
                   const std::list<std::shared_ptr<Analyzer::Expr>>& l)
    : Expr(kBOOLEAN, !is_in_values_nullable(a, l)), arg(a), value_list(l) {}

void InValues::group_predicates(std::list<const Expr*>& scan_predicates,
                                std::list<const Expr*>& join_predicates,
                                std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  arg->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

InIntegerSet::InIntegerSet(const std::shared_ptr<const Analyzer::Expr> a,
                           const std::vector<int64_t>& l,
                           const bool not_null)
    : Expr(kBOOLEAN, not_null), arg(a), value_list(l) {}

void CharLengthExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                      std::list<const Expr*>& join_predicates,
                                      std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  arg->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void LikeExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                std::list<const Expr*>& join_predicates,
                                std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  arg->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void RegexpExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                  std::list<const Expr*>& join_predicates,
                                  std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  arg->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void LikelihoodExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                      std::list<const Expr*>& join_predicates,
                                      std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  arg->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void AggExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                               std::list<const Expr*>& join_predicates,
                               std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  arg->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void CaseExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                std::list<const Expr*>& join_predicates,
                                std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  for (auto p : expr_pair_list) {
    p.first->collect_rte_idx(rte_idx_set);
    p.second->collect_rte_idx(rte_idx_set);
  }
  if (else_expr != nullptr) {
    else_expr->collect_rte_idx(rte_idx_set);
  }
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void ExtractExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                   std::list<const Expr*>& join_predicates,
                                   std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  from_expr_->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void DateaddExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                   std::list<const Expr*>& join_predicates,
                                   std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  number_->collect_rte_idx(rte_idx_set);
  datetime_->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void DatediffExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                    std::list<const Expr*>& join_predicates,
                                    std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  start_->collect_rte_idx(rte_idx_set);
  end_->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

void DatetruncExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                     std::list<const Expr*>& join_predicates,
                                     std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  from_expr_->collect_rte_idx(rte_idx_set);
  if (rte_idx_set.size() > 1) {
    join_predicates.push_back(this);
  } else if (rte_idx_set.size() == 1) {
    scan_predicates.push_back(this);
  } else {
    const_predicates.push_back(this);
  }
}

std::shared_ptr<Analyzer::Expr> ColumnVar::rewrite_with_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  for (auto tle : tlist) {
    const Expr* e = tle->get_expr();
    const ColumnVar* colvar = dynamic_cast<const ColumnVar*>(e);
    if (colvar != nullptr) {
      if (table_id == colvar->get_table_id() && column_id == colvar->get_column_id()) {
        return colvar->deep_copy();
      }
    }
  }
  throw std::runtime_error("Internal error: cannot find ColumnVar in targetlist.");
}

std::shared_ptr<Analyzer::Expr> ColumnVar::rewrite_with_child_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  int varno = 1;
  for (auto tle : tlist) {
    const Expr* e = tle->get_expr();
    const ColumnVar* colvar = dynamic_cast<const ColumnVar*>(e);
    if (colvar == nullptr) {
      throw std::runtime_error(
          "Internal Error: targetlist in rewrite_with_child_targetlist is not all "
          "columns.");
    }
    if (table_id == colvar->get_table_id() && column_id == colvar->get_column_id()) {
      return makeExpr<Var>(colvar->get_type_info(),
                           colvar->get_table_id(),
                           colvar->get_column_id(),
                           colvar->get_rte_idx(),
                           Var::kINPUT_OUTER,
                           varno);
    }
    varno++;
  }
  throw std::runtime_error("Internal error: cannot find ColumnVar in child targetlist.");
}

std::shared_ptr<Analyzer::Expr> ColumnVar::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  int varno = 1;
  for (auto tle : tlist) {
    const Expr* e = tle->get_expr();
    if (typeid(*e) != typeid(AggExpr)) {
      const ColumnVar* colvar = dynamic_cast<const ColumnVar*>(e);
      if (colvar == nullptr) {
        throw std::runtime_error(
            "Internal Error: targetlist in rewrite_agg_to_var is not all columns and "
            "aggregates.");
      }
      if (table_id == colvar->get_table_id() && column_id == colvar->get_column_id()) {
        return makeExpr<Var>(colvar->get_type_info(),
                             colvar->get_table_id(),
                             colvar->get_column_id(),
                             colvar->get_rte_idx(),
                             Var::kINPUT_OUTER,
                             varno);
      }
    }
    varno++;
  }
  throw std::runtime_error(
      "Internal error: cannot find ColumnVar from having clause in targetlist.");
}

std::shared_ptr<Analyzer::Expr> Var::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  int varno = 1;
  for (auto tle : tlist) {
    const Expr* e = tle->get_expr();
    if (*e == *this) {
      return makeExpr<Var>(e->get_type_info(), Var::kINPUT_OUTER, varno);
    }
    varno++;
  }
  throw std::runtime_error(
      "Internal error: cannot find Var from having clause in targetlist.");
}

std::shared_ptr<Analyzer::Expr> InValues::rewrite_with_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  std::list<std::shared_ptr<Analyzer::Expr>> new_value_list;
  for (auto v : value_list) {
    new_value_list.push_back(v->deep_copy());
  }
  return makeExpr<InValues>(arg->rewrite_with_targetlist(tlist), new_value_list);
}

std::shared_ptr<Analyzer::Expr> InValues::rewrite_with_child_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  std::list<std::shared_ptr<Analyzer::Expr>> new_value_list;
  for (auto v : value_list) {
    new_value_list.push_back(v->deep_copy());
  }
  return makeExpr<InValues>(arg->rewrite_with_child_targetlist(tlist), new_value_list);
}

std::shared_ptr<Analyzer::Expr> InValues::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  std::list<std::shared_ptr<Analyzer::Expr>> new_value_list;
  for (auto v : value_list) {
    new_value_list.push_back(v->rewrite_agg_to_var(tlist));
  }
  return makeExpr<InValues>(arg->rewrite_agg_to_var(tlist), new_value_list);
}

std::shared_ptr<Analyzer::Expr> AggExpr::rewrite_with_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  for (auto tle : tlist) {
    const Expr* e = tle->get_expr();
    if (typeid(*e) == typeid(AggExpr)) {
      const AggExpr* agg = dynamic_cast<const AggExpr*>(e);
      if (*this == *agg) {
        return agg->deep_copy();
      }
    }
  }
  throw std::runtime_error("Internal error: cannot find AggExpr in targetlist.");
}

std::shared_ptr<Analyzer::Expr> AggExpr::rewrite_with_child_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<AggExpr>(type_info,
                           aggtype,
                           arg ? arg->rewrite_with_child_targetlist(tlist) : nullptr,
                           is_distinct,
                           error_rate);
}

std::shared_ptr<Analyzer::Expr> AggExpr::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  int varno = 1;
  for (auto tle : tlist) {
    const Expr* e = tle->get_expr();
    if (typeid(*e) == typeid(AggExpr)) {
      const AggExpr* agg_expr = dynamic_cast<const AggExpr*>(e);
      if (*this == *agg_expr) {
        return makeExpr<Var>(agg_expr->get_type_info(), Var::kINPUT_OUTER, varno);
      }
    }
    varno++;
  }
  throw std::runtime_error(
      "Internal error: cannot find AggExpr from having clause in targetlist.");
}

std::shared_ptr<Analyzer::Expr> CaseExpr::rewrite_with_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      epair_list;
  for (auto p : expr_pair_list) {
    epair_list.emplace_back(p.first->rewrite_with_targetlist(tlist),
                            p.second->rewrite_with_targetlist(tlist));
  }
  return makeExpr<CaseExpr>(
      type_info,
      contains_agg,
      epair_list,
      else_expr ? else_expr->rewrite_with_targetlist(tlist) : nullptr);
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::rewrite_with_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<ExtractExpr>(
      type_info, contains_agg, field_, from_expr_->rewrite_with_targetlist(tlist));
}

std::shared_ptr<Analyzer::Expr> DateaddExpr::rewrite_with_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DateaddExpr>(type_info,
                               field_,
                               number_->rewrite_with_targetlist(tlist),
                               datetime_->rewrite_with_targetlist(tlist));
}

std::shared_ptr<Analyzer::Expr> DatediffExpr::rewrite_with_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DatediffExpr>(type_info,
                                field_,
                                start_->rewrite_with_targetlist(tlist),
                                end_->rewrite_with_targetlist(tlist));
}

std::shared_ptr<Analyzer::Expr> DatetruncExpr::rewrite_with_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DatetruncExpr>(
      type_info, contains_agg, field_, from_expr_->rewrite_with_targetlist(tlist));
}

std::shared_ptr<Analyzer::Expr> CaseExpr::rewrite_with_child_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      epair_list;
  for (auto p : expr_pair_list) {
    epair_list.emplace_back(p.first->rewrite_with_child_targetlist(tlist),
                            p.second->rewrite_with_child_targetlist(tlist));
  }
  return makeExpr<CaseExpr>(
      type_info,
      contains_agg,
      epair_list,
      else_expr ? else_expr->rewrite_with_child_targetlist(tlist) : nullptr);
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::rewrite_with_child_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<ExtractExpr>(
      type_info, contains_agg, field_, from_expr_->rewrite_with_child_targetlist(tlist));
}

std::shared_ptr<Analyzer::Expr> DateaddExpr::rewrite_with_child_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DateaddExpr>(type_info,
                               field_,
                               number_->rewrite_with_child_targetlist(tlist),
                               datetime_->rewrite_with_child_targetlist(tlist));
}

std::shared_ptr<Analyzer::Expr> DatediffExpr::rewrite_with_child_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DatediffExpr>(type_info,
                                field_,
                                start_->rewrite_with_child_targetlist(tlist),
                                end_->rewrite_with_child_targetlist(tlist));
}

std::shared_ptr<Analyzer::Expr> DatetruncExpr::rewrite_with_child_targetlist(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DatetruncExpr>(
      type_info, contains_agg, field_, from_expr_->rewrite_with_child_targetlist(tlist));
}

std::shared_ptr<Analyzer::Expr> CaseExpr::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      epair_list;
  for (auto p : expr_pair_list) {
    epair_list.emplace_back(p.first->rewrite_agg_to_var(tlist),
                            p.second->rewrite_agg_to_var(tlist));
  }
  return makeExpr<CaseExpr>(type_info,
                            contains_agg,
                            epair_list,
                            else_expr ? else_expr->rewrite_agg_to_var(tlist) : nullptr);
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<ExtractExpr>(
      type_info, contains_agg, field_, from_expr_->rewrite_agg_to_var(tlist));
}

std::shared_ptr<Analyzer::Expr> DateaddExpr::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DateaddExpr>(type_info,
                               field_,
                               number_->rewrite_agg_to_var(tlist),
                               datetime_->rewrite_agg_to_var(tlist));
}

std::shared_ptr<Analyzer::Expr> DatediffExpr::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DatediffExpr>(type_info,
                                field_,
                                start_->rewrite_agg_to_var(tlist),
                                end_->rewrite_agg_to_var(tlist));
}

std::shared_ptr<Analyzer::Expr> DatetruncExpr::rewrite_agg_to_var(
    const std::vector<std::shared_ptr<TargetEntry>>& tlist) const {
  return makeExpr<DatetruncExpr>(
      type_info, contains_agg, field_, from_expr_->rewrite_agg_to_var(tlist));
}

bool ColumnVar::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(ColumnVar) && typeid(rhs) != typeid(Var)) {
    return false;
  }
  const ColumnVar& rhs_cv = dynamic_cast<const ColumnVar&>(rhs);
  if (rte_idx != -1) {
    return (table_id == rhs_cv.get_table_id()) && (column_id == rhs_cv.get_column_id()) &&
           (rte_idx == rhs_cv.get_rte_idx());
  }
  const Var* v = dynamic_cast<const Var*>(this);
  if (v == nullptr) {
    return false;
  }
  const Var* rv = dynamic_cast<const Var*>(&rhs);
  if (rv == nullptr) {
    return false;
  }
  return (v->get_which_row() == rv->get_which_row()) &&
         (v->get_varno() == rv->get_varno());
}

bool ExpressionTuple::operator==(const Expr& rhs) const {
  const auto rhs_tuple = dynamic_cast<const ExpressionTuple*>(&rhs);
  if (!rhs_tuple) {
    return false;
  }
  const auto& rhs_tuple_cols = rhs_tuple->getTuple();
  if (rhs_tuple_cols.size() != tuple_.size()) {
    return false;
  }
  for (size_t i = 0; i < tuple_.size(); ++i) {
    if (!(*tuple_[i] == *rhs_tuple_cols[i])) {
      return false;
    }
  }
  return true;
}

bool Datum_equal(const SQLTypeInfo& ti, Datum val1, Datum val2) {
  switch (ti.get_type()) {
    case kBOOLEAN:
      return val1.boolval == val2.boolval;
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      return *val1.stringval == *val2.stringval;
    case kNUMERIC:
    case kDECIMAL:
    case kBIGINT:
      return val1.bigintval == val2.bigintval;
    case kINT:
      return val1.intval == val2.intval;
    case kSMALLINT:
      return val1.smallintval == val2.smallintval;
    case kTINYINT:
      return val1.tinyintval == val2.tinyintval;
    case kFLOAT:
      return val1.floatval == val2.floatval;
    case kDOUBLE:
      return val1.doubleval == val2.doubleval;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return val1.timeval == val2.timeval;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      return *val1.stringval == *val2.stringval;
    default:
      throw std::runtime_error("Unrecognized type for Constant Datum equality: " +
                               ti.get_type_name());
  }
  UNREACHABLE();
  return false;
}

bool Constant::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(Constant)) {
    return false;
  }
  const Constant& rhs_c = dynamic_cast<const Constant&>(rhs);
  if (type_info != rhs_c.get_type_info() || is_null != rhs_c.get_is_null()) {
    return false;
  }
  if (is_null && rhs_c.get_is_null()) {
    return true;
  }
  if (type_info.is_array()) {
    return false;
  }
  return Datum_equal(type_info, constval, rhs_c.get_constval());
}

bool UOper::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(UOper)) {
    return false;
  }
  const UOper& rhs_uo = dynamic_cast<const UOper&>(rhs);
  return optype == rhs_uo.get_optype() && *operand == *rhs_uo.get_operand();
}

bool BinOper::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(BinOper)) {
    return false;
  }
  const BinOper& rhs_bo = dynamic_cast<const BinOper&>(rhs);
  return optype == rhs_bo.get_optype() && *left_operand == *rhs_bo.get_left_operand() &&
         *right_operand == *rhs_bo.get_right_operand();
}

bool CharLengthExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(CharLengthExpr)) {
    return false;
  }
  const CharLengthExpr& rhs_cl = dynamic_cast<const CharLengthExpr&>(rhs);
  if (!(*arg == *rhs_cl.get_arg()) ||
      calc_encoded_length != rhs_cl.get_calc_encoded_length()) {
    return false;
  }
  return true;
}

bool LikeExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(LikeExpr)) {
    return false;
  }
  const LikeExpr& rhs_lk = dynamic_cast<const LikeExpr&>(rhs);
  if (!(*arg == *rhs_lk.get_arg()) || !(*like_expr == *rhs_lk.get_like_expr()) ||
      is_ilike != rhs_lk.get_is_ilike()) {
    return false;
  }
  if (escape_expr.get() == rhs_lk.get_escape_expr()) {
    return true;
  }
  if (escape_expr != nullptr && rhs_lk.get_escape_expr() != nullptr &&
      *escape_expr == *rhs_lk.get_escape_expr()) {
    return true;
  }
  return false;
}

bool RegexpExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(RegexpExpr)) {
    return false;
  }
  const RegexpExpr& rhs_re = dynamic_cast<const RegexpExpr&>(rhs);
  if (!(*arg == *rhs_re.get_arg()) || !(*pattern_expr == *rhs_re.get_pattern_expr())) {
    return false;
  }
  if (escape_expr.get() == rhs_re.get_escape_expr()) {
    return true;
  }
  if (escape_expr != nullptr && rhs_re.get_escape_expr() != nullptr &&
      *escape_expr == *rhs_re.get_escape_expr()) {
    return true;
  }
  return false;
}

bool LikelihoodExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(LikelihoodExpr)) {
    return false;
  }
  const LikelihoodExpr& rhs_l = dynamic_cast<const LikelihoodExpr&>(rhs);
  if (!(*arg == *rhs_l.get_arg())) {
    return false;
  }
  if (likelihood != rhs_l.get_likelihood()) {
    return false;
  }
  return true;
}

bool InValues::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(InValues)) {
    return false;
  }
  const InValues& rhs_iv = dynamic_cast<const InValues&>(rhs);
  if (!(*arg == *rhs_iv.get_arg())) {
    return false;
  }
  if (value_list.size() != rhs_iv.get_value_list().size()) {
    return false;
  }
  auto q = rhs_iv.get_value_list().begin();
  for (auto p : value_list) {
    if (!(*p == **q)) {
      return false;
    }
    q++;
  }
  return true;
}

bool AggExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(AggExpr)) {
    return false;
  }
  const AggExpr& rhs_ae = dynamic_cast<const AggExpr&>(rhs);
  if (aggtype != rhs_ae.get_aggtype() || is_distinct != rhs_ae.get_is_distinct()) {
    return false;
  }
  if (arg.get() == rhs_ae.get_arg()) {
    return true;
  }
  if (arg == nullptr || rhs_ae.get_arg() == nullptr) {
    return false;
  }
  return *arg == *rhs_ae.get_arg();
}

bool CaseExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(CaseExpr)) {
    return false;
  }
  const CaseExpr& rhs_ce = dynamic_cast<const CaseExpr&>(rhs);
  if (expr_pair_list.size() != rhs_ce.get_expr_pair_list().size()) {
    return false;
  }
  if ((else_expr == nullptr && rhs_ce.get_else_expr() != nullptr) ||
      (else_expr != nullptr && rhs_ce.get_else_expr() == nullptr)) {
    return false;
  }
  auto it = rhs_ce.get_expr_pair_list().cbegin();
  for (auto p : expr_pair_list) {
    if (!(*p.first == *it->first) || !(*p.second == *it->second)) {
      return false;
    }
    ++it;
  }
  return else_expr == nullptr ||
         (else_expr != nullptr && *else_expr == *rhs_ce.get_else_expr());
}

bool ExtractExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(ExtractExpr)) {
    return false;
  }
  const ExtractExpr& rhs_ee = dynamic_cast<const ExtractExpr&>(rhs);
  return field_ == rhs_ee.get_field() && *from_expr_ == *rhs_ee.get_from_expr();
}

bool DateaddExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(DateaddExpr)) {
    return false;
  }
  const DateaddExpr& rhs_ee = dynamic_cast<const DateaddExpr&>(rhs);
  return field_ == rhs_ee.get_field() && *number_ == *rhs_ee.get_number_expr() &&
         *datetime_ == *rhs_ee.get_datetime_expr();
}

bool DatediffExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(DatediffExpr)) {
    return false;
  }
  const DatediffExpr& rhs_ee = dynamic_cast<const DatediffExpr&>(rhs);
  return field_ == rhs_ee.get_field() && *start_ == *rhs_ee.get_start_expr() &&
         *end_ == *rhs_ee.get_end_expr();
}

bool DatetruncExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(DatetruncExpr)) {
    return false;
  }
  const DatetruncExpr& rhs_ee = dynamic_cast<const DatetruncExpr&>(rhs);
  return field_ == rhs_ee.get_field() && *from_expr_ == *rhs_ee.get_from_expr();
}

bool OffsetInFragment::operator==(const Expr& rhs) const {
  return typeid(rhs) == typeid(OffsetInFragment);
}

bool ArrayExpr::operator==(Expr const& rhs) const {
  if (typeid(rhs) != typeid(ArrayExpr)) {
    return false;
  }
  ArrayExpr const& casted_rhs = static_cast<ArrayExpr const&>(rhs);
  for (unsigned i = 0; i < contained_expressions_.size(); i++) {
    auto& lhs_expr = contained_expressions_[i];
    auto& rhs_expr = casted_rhs.contained_expressions_[i];
    if (!(lhs_expr == rhs_expr)) {
      return false;
    }
  }
  return true;
}

std::string ColumnVar::toString() const {
  return "(ColumnVar table: " + std::to_string(table_id) +
         " column: " + std::to_string(column_id) + " rte: " + std::to_string(rte_idx) +
         ") ";
}

std::string ExpressionTuple::toString() const {
  std::string str{"< "};
  for (const auto& column : tuple_) {
    str += column->toString();
  }
  str += "> ";
  return str;
}

std::string Var::toString() const {
  return "(Var table: " + std::to_string(table_id) +
         " column: " + std::to_string(column_id) + " rte: " + std::to_string(rte_idx) +
         " which_row: " + std::to_string(which_row) + " varno: " + std::to_string(varno) +
         ") ";
}

std::string Constant::toString() const {
  std::string str{"(Const "};
  if (is_null) {
    str += "NULL";
  } else if (type_info.is_array()) {
    const auto& elem_ti = type_info.get_elem_type();
    str += sql_type_to_str(type_info.get_type()) + ": " +
           sql_type_to_str(elem_ti.get_type());
  } else {
    str += DatumToString(constval, type_info);
  }
  str += ") ";
  return str;
}

std::string UOper::toString() const {
  std::string op;
  switch (optype) {
    case kNOT:
      op = "NOT ";
      break;
    case kUMINUS:
      op = "- ";
      break;
    case kISNULL:
      op = "IS NULL ";
      break;
    case kEXISTS:
      op = "EXISTS ";
      break;
    case kCAST:
      op = "CAST " + type_info.get_type_name() + "(" +
           std::to_string(type_info.get_precision()) + "," +
           std::to_string(type_info.get_scale()) + ") " +
           type_info.get_compression_name() + "(" +
           std::to_string(type_info.get_comp_param()) + ") ";
      break;
    case kUNNEST:
      op = "UNNEST ";
      break;
    default:
      break;
  }
  return "(" + op + operand->toString() + ") ";
}

std::string BinOper::toString() const {
  std::string op;
  switch (optype) {
    case kEQ:
      op = "= ";
      break;
    case kNE:
      op = "<> ";
      break;
    case kLT:
      op = "< ";
      break;
    case kLE:
      op = "<= ";
      break;
    case kGT:
      op = "> ";
      break;
    case kGE:
      op = ">= ";
      break;
    case kAND:
      op = "AND ";
      break;
    case kOR:
      op = "OR ";
      break;
    case kMINUS:
      op = "- ";
      break;
    case kPLUS:
      op = "+ ";
      break;
    case kMULTIPLY:
      op = "* ";
      break;
    case kDIVIDE:
      op = "/ ";
      break;
    case kMODULO:
      op = "% ";
      break;
    case kARRAY_AT:
      op = "[] ";
      break;
    default:
      break;
  }
  std::string str{"("};
  str += op;
  if (qualifier == kANY) {
    str += "ANY ";
  } else if (qualifier == kALL) {
    str += "ALL ";
  }
  str += left_operand->toString();
  str += right_operand->toString();
  str += ") ";
  return str;
}

std::string Subquery::toString() const {
  return "(Subquery ) ";
}

std::string InValues::toString() const {
  std::string str{"(IN "};
  str += arg->toString();
  str += "(";
  for (auto e : value_list) {
    str += e->toString();
  }
  str += ") ";
  return str;
}

std::shared_ptr<Analyzer::Expr> InIntegerSet::deep_copy() const {
  return std::make_shared<InIntegerSet>(arg, value_list, get_type_info().get_notnull());
}

bool InIntegerSet::operator==(const Expr& rhs) const {
  if (!dynamic_cast<const InIntegerSet*>(&rhs)) {
    return false;
  }
  const auto& rhs_in_integer_set = static_cast<const InIntegerSet&>(rhs);
  return *arg == *rhs_in_integer_set.arg && value_list == rhs_in_integer_set.value_list;
}

std::string InIntegerSet::toString() const {
  std::string str{"(IN_INTEGER_SET "};
  str += arg->toString();
  str += "( ";
  for (const auto e : value_list) {
    str += std::to_string(e) + " ";
  }
  str += ") ";
  return str;
}

std::string CharLengthExpr::toString() const {
  std::string str;
  if (calc_encoded_length) {
    str += "CHAR_LENGTH(";
  } else {
    str += "LENGTH(";
  }
  str += arg->toString();
  str += ") ";
  return str;
}

std::string LikeExpr::toString() const {
  std::string str{"(LIKE "};
  str += arg->toString();
  str += like_expr->toString();
  if (escape_expr) {
    str += escape_expr->toString();
  }
  str += ") ";
  return str;
}

std::string RegexpExpr::toString() const {
  std::string str{"(REGEXP "};
  str += arg->toString();
  str += pattern_expr->toString();
  if (escape_expr) {
    str += escape_expr->toString();
  }
  str += ") ";
  return str;
}

std::string LikelihoodExpr::toString() const {
  std::string str{"(LIKELIHOOD "};
  str += arg->toString();
  return str + " " + std::to_string(likelihood) + ") ";
}

std::string AggExpr::toString() const {
  std::string agg;
  switch (aggtype) {
    case kAVG:
      agg = "AVG ";
      break;
    case kMIN:
      agg = "MIN ";
      break;
    case kMAX:
      agg = "MAX ";
      break;
    case kSUM:
      agg = "SUM ";
      break;
    case kCOUNT:
      agg = "COUNT ";
      break;
    case kAPPROX_COUNT_DISTINCT:
      agg = "APPROX_COUNT_DISTINCT";
      break;
    case kSAMPLE:
      agg = "SAMPLE";
      break;
  }
  std::string str{"(" + agg};
  if (is_distinct) {
    str += "DISTINCT ";
  }
  if (arg) {
    str += arg->toString();
  } else {
    str += "*";
  }
  return str + ") ";
}

std::string CaseExpr::toString() const {
  std::string str{"CASE "};
  for (auto p : expr_pair_list) {
    str += "(";
    str += p.first->toString();
    str += ", ";
    str += p.second->toString();
    str += ") ";
  }
  if (else_expr) {
    str += "ELSE ";
    str += else_expr->toString();
  }
  str += " END ";
  return str;
}

std::string ExtractExpr::toString() const {
  return "EXTRACT(" + std::to_string(field_) + " FROM " + from_expr_->toString() + ") ";
}

std::string DateaddExpr::toString() const {
  return "DATEADD(" + std::to_string(field_) + " NUMBER " + number_->toString() +
         " DATETIME " + datetime_->toString() + ") ";
}

std::string DatediffExpr::toString() const {
  return "DATEDIFF(" + std::to_string(field_) + " START " + start_->toString() + " END " +
         end_->toString() + ") ";
}

std::string DatetruncExpr::toString() const {
  return "DATE_TRUNC(" + std::to_string(field_) + " , " + from_expr_->toString() + ") ";
}

std::string OffsetInFragment::toString() const {
  return "(OffsetInFragment) ";
}

std::string ArrayExpr::toString() const {
  std::string str{"ARRAY["};

  auto iter(contained_expressions_.begin());
  while (iter != contained_expressions_.end()) {
    str += (*iter)->toString();
    if (iter + 1 != contained_expressions_.end()) {
      str += ", ";
    }
    iter++;
  }
  str += "]";
  return str;
}

std::string TargetEntry::toString() const {
  std::string str{"(" + resname + " "};
  str += expr->toString();
  if (unnest) {
    str += " UNNEST";
  }
  str += ") ";
  return str;
}

std::string OrderEntry::toString() const {
  std::string str{std::to_string(tle_no)};
  if (is_desc) {
    str += " desc";
  }
  if (nulls_first) {
    str += " nulls first";
  }
  str += " ";
  return str;
}

void Expr::add_unique(std::list<const Expr*>& expr_list) const {
  // only add unique instances to the list
  for (auto e : expr_list) {
    if (*e == *this) {
      return;
    }
  }
  expr_list.push_back(this);
}

void BinOper::find_expr(bool (*f)(const Expr*), std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  left_operand->find_expr(f, expr_list);
  right_operand->find_expr(f, expr_list);
}

void UOper::find_expr(bool (*f)(const Expr*), std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  operand->find_expr(f, expr_list);
}

void InValues::find_expr(bool (*f)(const Expr*),
                         std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  arg->find_expr(f, expr_list);
  for (auto e : value_list) {
    e->find_expr(f, expr_list);
  }
}

void CharLengthExpr::find_expr(bool (*f)(const Expr*),
                               std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  arg->find_expr(f, expr_list);
}

void LikeExpr::find_expr(bool (*f)(const Expr*),
                         std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  arg->find_expr(f, expr_list);
  like_expr->find_expr(f, expr_list);
  if (escape_expr != nullptr) {
    escape_expr->find_expr(f, expr_list);
  }
}

void RegexpExpr::find_expr(bool (*f)(const Expr*),
                           std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  arg->find_expr(f, expr_list);
  pattern_expr->find_expr(f, expr_list);
  if (escape_expr != nullptr) {
    escape_expr->find_expr(f, expr_list);
  }
}

void LikelihoodExpr::find_expr(bool (*f)(const Expr*),
                               std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  arg->find_expr(f, expr_list);
}

void AggExpr::find_expr(bool (*f)(const Expr*), std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  if (arg != nullptr) {
    arg->find_expr(f, expr_list);
  }
}

void CaseExpr::find_expr(bool (*f)(const Expr*),
                         std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  for (auto p : expr_pair_list) {
    p.first->find_expr(f, expr_list);
    p.second->find_expr(f, expr_list);
  }
  if (else_expr != nullptr) {
    else_expr->find_expr(f, expr_list);
  }
}

void ExtractExpr::find_expr(bool (*f)(const Expr*),
                            std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  from_expr_->find_expr(f, expr_list);
}

void DateaddExpr::find_expr(bool (*f)(const Expr*),
                            std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  number_->find_expr(f, expr_list);
  datetime_->find_expr(f, expr_list);
}

void DatediffExpr::find_expr(bool (*f)(const Expr*),
                             std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  start_->find_expr(f, expr_list);
  end_->find_expr(f, expr_list);
}

void DatetruncExpr::find_expr(bool (*f)(const Expr*),
                              std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  from_expr_->find_expr(f, expr_list);
}

void CaseExpr::collect_rte_idx(std::set<int>& rte_idx_set) const {
  for (auto p : expr_pair_list) {
    p.first->collect_rte_idx(rte_idx_set);
    p.second->collect_rte_idx(rte_idx_set);
  }
  if (else_expr != nullptr) {
    else_expr->collect_rte_idx(rte_idx_set);
  }
}

void ExtractExpr::collect_rte_idx(std::set<int>& rte_idx_set) const {
  from_expr_->collect_rte_idx(rte_idx_set);
}

void DateaddExpr::collect_rte_idx(std::set<int>& rte_idx_set) const {
  number_->collect_rte_idx(rte_idx_set);
  datetime_->collect_rte_idx(rte_idx_set);
}

void DatediffExpr::collect_rte_idx(std::set<int>& rte_idx_set) const {
  start_->collect_rte_idx(rte_idx_set);
  end_->collect_rte_idx(rte_idx_set);
}

void DatetruncExpr::collect_rte_idx(std::set<int>& rte_idx_set) const {
  from_expr_->collect_rte_idx(rte_idx_set);
}

void CaseExpr::collect_column_var(
    std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>& colvar_set,
    bool include_agg) const {
  for (auto p : expr_pair_list) {
    p.first->collect_column_var(colvar_set, include_agg);
    p.second->collect_column_var(colvar_set, include_agg);
  }
  if (else_expr != nullptr) {
    else_expr->collect_column_var(colvar_set, include_agg);
  }
}

void ExtractExpr::collect_column_var(
    std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>& colvar_set,
    bool include_agg) const {
  from_expr_->collect_column_var(colvar_set, include_agg);
}

void DateaddExpr::collect_column_var(
    std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>& colvar_set,
    bool include_agg) const {
  number_->collect_column_var(colvar_set, include_agg);
  datetime_->collect_column_var(colvar_set, include_agg);
}

void DatediffExpr::collect_column_var(
    std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>& colvar_set,
    bool include_agg) const {
  start_->collect_column_var(colvar_set, include_agg);
  end_->collect_column_var(colvar_set, include_agg);
}

void DatetruncExpr::collect_column_var(
    std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>& colvar_set,
    bool include_agg) const {
  from_expr_->collect_column_var(colvar_set, include_agg);
}

void CaseExpr::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  for (auto p : expr_pair_list) {
    p.first->check_group_by(groupby);
    p.second->check_group_by(groupby);
  }
  if (else_expr != nullptr) {
    else_expr->check_group_by(groupby);
  }
}

void ExtractExpr::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  from_expr_->check_group_by(groupby);
}

void DateaddExpr::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  number_->check_group_by(groupby);
  datetime_->check_group_by(groupby);
}

void DatediffExpr::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  start_->check_group_by(groupby);
  end_->check_group_by(groupby);
}

void DatetruncExpr::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  from_expr_->check_group_by(groupby);
}

void CaseExpr::get_domain(DomainSet& domain_set) const {
  for (const auto& p : expr_pair_list) {
    const auto c = std::dynamic_pointer_cast<const Constant>(p.second);
    if (c != nullptr) {
      c->add_unique(domain_set);
    } else {
      const auto v = std::dynamic_pointer_cast<const ColumnVar>(p.second);
      if (v != nullptr) {
        v->add_unique(domain_set);
      } else {
        const auto cast = std::dynamic_pointer_cast<const UOper>(p.second);
        if (cast != nullptr && cast->get_optype() == kCAST) {
          const Constant* c = dynamic_cast<const Constant*>(cast->get_operand());
          if (c != nullptr) {
            cast->add_unique(domain_set);
            continue;
          } else {
            const auto v = std::dynamic_pointer_cast<const ColumnVar>(p.second);
            if (v != nullptr) {
              v->add_unique(domain_set);
              continue;
            }
          }
        }
        p.second->get_domain(domain_set);
        if (domain_set.empty()) {
          return;
        }
      }
    }
  }
  if (else_expr != nullptr) {
    const auto c = std::dynamic_pointer_cast<const Constant>(else_expr);
    if (c != nullptr) {
      c->add_unique(domain_set);
    } else {
      const auto v = std::dynamic_pointer_cast<const ColumnVar>(else_expr);
      if (v != nullptr) {
        v->add_unique(domain_set);
      } else {
        const auto cast = std::dynamic_pointer_cast<const UOper>(else_expr);
        if (cast != nullptr && cast->get_optype() == kCAST) {
          const Constant* c = dynamic_cast<const Constant*>(cast->get_operand());
          if (c != nullptr) {
            c->add_unique(domain_set);
          } else {
            const auto v = std::dynamic_pointer_cast<const ColumnVar>(else_expr);
            if (v != nullptr) {
              v->add_unique(domain_set);
            }
          }
        } else {
          else_expr->get_domain(domain_set);
        }
      }
    }
  }
}

std::shared_ptr<Analyzer::Expr> FunctionOper::deep_copy() const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args_copy;
  for (size_t i = 0; i < getArity(); ++i) {
    args_copy.push_back(getArg(i)->deep_copy());
  }
  return makeExpr<Analyzer::FunctionOper>(type_info, getName(), args_copy);
}

bool FunctionOper::operator==(const Expr& rhs) const {
  if (type_info != rhs.get_type_info()) {
    return false;
  }
  const auto rhs_func_oper = dynamic_cast<const FunctionOper*>(&rhs);
  if (!rhs_func_oper) {
    return false;
  }
  if (getName() != rhs_func_oper->getName()) {
    return false;
  }
  if (getArity() != rhs_func_oper->getArity()) {
    return false;
  }
  for (size_t i = 0; i < getArity(); ++i) {
    if (!(*getArg(i) == *(rhs_func_oper->getArg(i)))) {
      return false;
    }
  }
  return true;
}

std::string FunctionOper::toString() const {
  std::string str{"(" + name_ + " "};
  for (const auto arg : args_) {
    str += arg->toString();
  }
  str += ")";
  return str;
}

std::shared_ptr<Analyzer::Expr> FunctionOperWithCustomTypeHandling::deep_copy() const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args_copy;
  for (size_t i = 0; i < getArity(); ++i) {
    args_copy.push_back(getArg(i)->deep_copy());
  }
  return makeExpr<Analyzer::FunctionOperWithCustomTypeHandling>(
      type_info, getName(), args_copy);
}

bool FunctionOperWithCustomTypeHandling::operator==(const Expr& rhs) const {
  if (type_info != rhs.get_type_info()) {
    return false;
  }
  const auto rhs_func_oper =
      dynamic_cast<const FunctionOperWithCustomTypeHandling*>(&rhs);
  if (!rhs_func_oper) {
    return false;
  }
  if (getName() != rhs_func_oper->getName()) {
    return false;
  }
  if (getArity() != rhs_func_oper->getArity()) {
    return false;
  }
  for (size_t i = 0; i < getArity(); ++i) {
    if (!(*getArg(i) == *(rhs_func_oper->getArg(i)))) {
      return false;
    }
  }
  return true;
}

}  // namespace Analyzer
