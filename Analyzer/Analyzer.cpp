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

#include "Analyzer/Analyzer.h"
#include "Geospatial/Conversion.h"
#include "Geospatial/Types.h"
#include "QueryEngine/DateTimeUtils.h"
#include "Shared/DateConverters.h"
#include "Shared/misc.h"
#include "Shared/sqltypes.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace Analyzer {

Constant::~Constant() {
  if (type_info.is_string() && !is_null) {
    delete constval.stringval;
  }
}

Subquery::~Subquery() {
  delete parsetree;
}

Query::~Query() {
  delete order_by;
  delete next_query;
}

std::shared_ptr<Analyzer::Expr> ColumnVar::deep_copy() const {
  return makeExpr<ColumnVar>(col_info_, rte_idx);
}

void ExpressionTuple::collect_rte_idx(std::set<int>& rte_idx_set) const {
  for (const auto& column : tuple_) {
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
  return makeExpr<Var>(col_info_, rte_idx, which_row, varno);
}

std::shared_ptr<Analyzer::Expr> Constant::deep_copy() const {
  Datum d = constval;
  if (type_info.is_string() && !is_null) {
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

std::shared_ptr<Analyzer::Expr> RangeOper::deep_copy() const {
  return makeExpr<RangeOper>(left_inclusive_,
                             right_inclusive_,
                             left_operand_->deep_copy(),
                             right_operand_->deep_copy());
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

std::shared_ptr<Analyzer::Expr> KeyForStringExpr::deep_copy() const {
  return makeExpr<KeyForStringExpr>(arg->deep_copy());
}

std::shared_ptr<Analyzer::Expr> SampleRatioExpr::deep_copy() const {
  return makeExpr<SampleRatioExpr>(arg->deep_copy());
}

std::shared_ptr<Analyzer::Expr> LowerExpr::deep_copy() const {
  return makeExpr<LowerExpr>(arg->deep_copy());
}

std::shared_ptr<Analyzer::Expr> CardinalityExpr::deep_copy() const {
  return makeExpr<CardinalityExpr>(arg->deep_copy());
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

std::shared_ptr<Analyzer::Expr> WidthBucketExpr::deep_copy() const {
  return makeExpr<WidthBucketExpr>(target_value_->deep_copy(),
                                   lower_bound_->deep_copy(),
                                   upper_bound_->deep_copy(),
                                   partition_count_->deep_copy());
}

std::shared_ptr<Analyzer::Expr> LikelihoodExpr::deep_copy() const {
  return makeExpr<LikelihoodExpr>(arg->deep_copy(), likelihood);
}

std::shared_ptr<Analyzer::Expr> AggExpr::deep_copy() const {
  return makeExpr<AggExpr>(
      type_info, aggtype, arg ? arg->deep_copy() : nullptr, is_distinct, arg1);
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

std::shared_ptr<Analyzer::Expr> WindowFunction::deep_copy() const {
  return makeExpr<WindowFunction>(
      type_info, kind_, args_, partition_keys_, order_keys_, collation_);
}

ExpressionPtr ArrayExpr::deep_copy() const {
  return makeExpr<Analyzer::ArrayExpr>(
      type_info, contained_expressions_, is_null_, local_alloc_);
}

std::shared_ptr<Analyzer::Expr> GeoUOper::deep_copy() const {
  if (op_ == Geospatial::GeoBase::GeoOp::kPROJECTION) {
    return makeExpr<GeoUOper>(op_, type_info, type_info, args0_);
  }
  return makeExpr<GeoUOper>(op_, type_info, ti0_, args0_);
}

std::shared_ptr<Analyzer::Expr> GeoBinOper::deep_copy() const {
  return makeExpr<GeoBinOper>(op_, type_info, ti0_, ti1_, args0_, args1_);
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
        case kTINYINT:
          common_type =
              SQLTypeInfo(kDECIMAL,
                          std::max(3 + type1.get_scale(), type1.get_dimension()),
                          type1.get_scale(),
                          notnull);
          break;
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
          break;
        }
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

// Return dec * 10^-scale
template <typename T>
T floatFromDecimal(int64_t const dec, unsigned const scale) {
  static_assert(std::is_floating_point_v<T>);
  return static_cast<T>(dec) / shared::power10(scale);
}

// Q: Why is there a maxRound() but no minRound()?
// A: The numerical value of std::numeric_limits<int64_t>::min() is unchanged when cast
// to either float or double, but std::numeric_limits<intXX_t>::max() is incremented to
// 2^(XX-1) when cast to float/double for XX in {32,64}, which is an invalid intXX_t
// value. Thus the maximum float/double that can be cast to a valid integer type must be
// calculated directly, and not just compared to std::numeric_limits<intXX_t>::max().
template <typename FLOAT_TYPE, typename INT_TYPE>
constexpr FLOAT_TYPE maxRound() {
  static_assert(std::is_integral_v<INT_TYPE> && std::is_floating_point_v<FLOAT_TYPE>);
  constexpr int dd =
      std::numeric_limits<INT_TYPE>::digits - std::numeric_limits<FLOAT_TYPE>::digits;
  if constexpr (0 < dd) {
    return static_cast<FLOAT_TYPE>(std::numeric_limits<INT_TYPE>::max() - (1ll << dd));
  } else {
    return static_cast<FLOAT_TYPE>(std::numeric_limits<INT_TYPE>::max());
  }
}

template <typename TO, typename FROM>
TO safeNarrow(FROM const from) {
  static_assert(std::is_integral_v<TO> && std::is_integral_v<FROM>);
  static_assert(sizeof(TO) < sizeof(FROM));
  if (from < static_cast<FROM>(std::numeric_limits<TO>::min()) ||
      static_cast<FROM>(std::numeric_limits<TO>::max()) < from) {
    throw std::runtime_error("Overflow or underflow");
  }
  return static_cast<TO>(from);
}

template <typename T>
T roundDecimal(int64_t n, unsigned scale) {
  static_assert(std::is_integral_v<T>);
  constexpr size_t max_scale = std::numeric_limits<uint64_t>::digits10;  // 19
  constexpr auto pow10 = shared::powersOf<uint64_t, max_scale + 1>(10);
  if (scale == 0) {
    if constexpr (sizeof(T) < sizeof(int64_t)) {
      return safeNarrow<T>(n);
    } else {
      return n;
    }
  } else if (max_scale < scale) {
    return 0;  // 0.09223372036854775807 rounds to 0
  }
  uint64_t const u = std::abs(n);
  uint64_t const pow = pow10[scale];
  uint64_t div = u / pow;
  uint64_t rem = u % pow;
  div += pow / 2 <= rem;
  if constexpr (sizeof(T) < sizeof(int64_t)) {
    return safeNarrow<T>(static_cast<int64_t>(n < 0 ? -div : div));
  } else {
    return n < 0 ? -div : div;
  }
}

template <typename TO, typename FROM>
TO safeRound(FROM const from) {
  static_assert(std::is_integral_v<TO> && std::is_floating_point_v<FROM>);
  constexpr FROM max_float = maxRound<FROM, TO>();
  FROM const n = std::round(from);
  if (n < static_cast<FROM>(std::numeric_limits<TO>::min()) || max_float < n) {
    throw std::runtime_error("Overflow or underflow");
  }
  return static_cast<TO>(n);
}

// Return numeric/decimal representation of from with given scale.
template <typename T>
int64_t safeScale(T from, unsigned const scale) {
  static_assert(std::is_arithmetic_v<T>);
  constexpr size_t max_scale = std::numeric_limits<int64_t>::digits10;  // 18
  constexpr auto pow10 = shared::powersOf<int64_t, max_scale + 1>(10);
  if constexpr (std::is_integral_v<T>) {
    int64_t retval;
    if (scale < pow10.size() && !__builtin_mul_overflow(from, pow10[scale], &retval)) {
      return retval;
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    if (scale < pow10.size()) {
      return safeRound<int64_t>(from * pow10[scale]);
    }
  }
  if (from == 0) {
    return 0;
  }
  throw std::runtime_error("Overflow or underflow");
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
        case kTIMESTAMP:
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
          constval.bigintval = safeScale(constval.tinyintval, new_type_info.get_scale());
          break;
        default:
          CHECK(false);
      }
      break;
    case kINT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = safeNarrow<int8_t>(constval.intval);
          break;
        case kINT:
          break;
        case kSMALLINT:
          constval.smallintval = safeNarrow<int16_t>(constval.intval);
          break;
        case kBIGINT:
        case kTIMESTAMP:
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
          constval.bigintval = safeScale(constval.intval, new_type_info.get_scale());
          break;
        default:
          CHECK(false);
      }
      break;
    case kSMALLINT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = safeNarrow<int8_t>(constval.smallintval);
          break;
        case kINT:
          constval.intval = (int32_t)constval.smallintval;
          break;
        case kSMALLINT:
          break;
        case kBIGINT:
        case kTIMESTAMP:
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
          constval.bigintval = safeScale(constval.smallintval, new_type_info.get_scale());
          break;
        default:
          CHECK(false);
      }
      break;
    case kBIGINT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = safeNarrow<int8_t>(constval.bigintval);
          break;
        case kINT:
          constval.intval = safeNarrow<int32_t>(constval.bigintval);
          break;
        case kSMALLINT:
          constval.smallintval = safeNarrow<int16_t>(constval.bigintval);
          break;
        case kBIGINT:
        case kTIMESTAMP:
          break;
        case kDOUBLE:
          constval.doubleval = (double)constval.bigintval;
          break;
        case kFLOAT:
          constval.floatval = (float)constval.bigintval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = safeScale(constval.bigintval, new_type_info.get_scale());
          break;
        default:
          CHECK(false);
      }
      break;
    case kDOUBLE:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = safeRound<int8_t>(constval.doubleval);
          break;
        case kINT:
          constval.intval = safeRound<int32_t>(constval.doubleval);
          break;
        case kSMALLINT:
          constval.smallintval = safeRound<int16_t>(constval.doubleval);
          break;
        case kBIGINT:
        case kTIMESTAMP:
          constval.bigintval = safeRound<int64_t>(constval.doubleval);
          break;
        case kDOUBLE:
          break;
        case kFLOAT:
          constval.floatval = (float)constval.doubleval;
          break;
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = safeScale(constval.doubleval, new_type_info.get_scale());
          break;
        default:
          CHECK(false);
      }
      break;
    case kFLOAT:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval = safeRound<int8_t>(constval.floatval);
          break;
        case kINT:
          constval.intval = safeRound<int32_t>(constval.floatval);
          break;
        case kSMALLINT:
          constval.smallintval = safeRound<int16_t>(constval.floatval);
          break;
        case kBIGINT:
        case kTIMESTAMP:
          constval.bigintval = safeRound<int64_t>(constval.floatval);
          break;
        case kDOUBLE:
          constval.doubleval = (double)constval.floatval;
          break;
        case kFLOAT:
          break;
        case kNUMERIC:
        case kDECIMAL:
          constval.bigintval = safeScale(constval.floatval, new_type_info.get_scale());
          break;
        default:
          CHECK(false);
      }
      break;
    case kNUMERIC:
    case kDECIMAL:
      switch (new_type_info.get_type()) {
        case kTINYINT:
          constval.tinyintval =
              roundDecimal<int8_t>(constval.bigintval, type_info.get_scale());
          break;
        case kINT:
          constval.intval =
              roundDecimal<int32_t>(constval.bigintval, type_info.get_scale());
          break;
        case kSMALLINT:
          constval.smallintval =
              roundDecimal<int16_t>(constval.bigintval, type_info.get_scale());
          break;
        case kBIGINT:
        case kTIMESTAMP:
          constval.bigintval =
              roundDecimal<int64_t>(constval.bigintval, type_info.get_scale());
          break;
        case kDOUBLE:
          constval.doubleval =
              floatFromDecimal<double>(constval.bigintval, type_info.get_scale());
          break;
        case kFLOAT:
          constval.floatval =
              floatFromDecimal<float>(constval.bigintval, type_info.get_scale());
          break;
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
          constval.tinyintval = safeNarrow<int8_t>(constval.bigintval);
          break;
        case kINT:
          constval.intval = safeNarrow<int32_t>(constval.bigintval);
          break;
        case kSMALLINT:
          constval.smallintval = safeNarrow<int16_t>(constval.bigintval);
          break;
        case kBIGINT:
        case kTIMESTAMP:
          break;
        case kDOUBLE:
          constval.doubleval = static_cast<double>(constval.bigintval);
          break;
        case kFLOAT:
          constval.floatval = static_cast<float>(constval.bigintval);
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
        case kTIMESTAMP:
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

namespace {

// TODO(adb): we should revisit this, as one could argue a Datum should never contain
// a null sentinel. In fact, if we bundle Datum with a null boolean ("NullableDatum"),
// the logic becomes more explicit. There are likely other bugs associated with the
// current logic -- for example, boolean is set to -128 which is likely UB
inline bool is_null_value(const SQLTypeInfo& ti, const Datum& constval) {
  switch (ti.get_type()) {
    case kBOOLEAN:
      return constval.tinyintval == NULL_BOOLEAN;
    case kTINYINT:
      return constval.tinyintval == NULL_TINYINT;
    case kINT:
      return constval.intval == NULL_INT;
    case kSMALLINT:
      return constval.smallintval == NULL_SMALLINT;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      return constval.bigintval == NULL_BIGINT;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return constval.bigintval == NULL_BIGINT;
    case kVARCHAR:
    case kCHAR:
    case kTEXT:
      return constval.stringval == nullptr;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      return constval.stringval == nullptr;
    case kFLOAT:
      return constval.floatval == NULL_FLOAT;
    case kDOUBLE:
      return constval.doubleval == NULL_DOUBLE;
    case kNULLT:
      return constval.bigintval == 0;
    case kARRAY:
      return constval.arrayval == nullptr;
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return false;
}

}  // namespace

void Constant::do_cast(const SQLTypeInfo& new_type_info) {
  if (type_info == new_type_info) {
    return;
  }
  if (is_null && !new_type_info.get_notnull()) {
    type_info = new_type_info;
    set_null_value();
    return;
  }
  if ((new_type_info.is_number() || new_type_info.get_type() == kTIMESTAMP) &&
      (new_type_info.get_type() != kTIMESTAMP || type_info.get_type() != kTIMESTAMP) &&
      (type_info.is_number() || type_info.get_type() == kTIMESTAMP ||
       type_info.get_type() == kBOOLEAN)) {
    cast_number(new_type_info);
  } else if (new_type_info.is_boolean() && type_info.is_boolean()) {
    type_info = new_type_info;
  } else if (new_type_info.is_string() && type_info.is_string()) {
    cast_string(new_type_info);
  } else if (type_info.is_string() || type_info.get_type() == kVARCHAR) {
    cast_from_string(new_type_info);
  } else if (new_type_info.is_string()) {
    cast_to_string(new_type_info);
  } else if (new_type_info.get_type() == kDATE && type_info.get_type() == kDATE) {
    type_info = new_type_info;
  } else if (new_type_info.get_type() == kDATE && type_info.get_type() == kTIMESTAMP) {
    constval.bigintval =
        type_info.is_high_precision_timestamp()
            ? truncate_high_precision_timestamp_to_date(
                  constval.bigintval,
                  DateTimeUtils::get_timestamp_precision_scale(type_info.get_dimension()))
            : DateTruncate(dtDAY, constval.bigintval);
    type_info = new_type_info;
  } else if ((type_info.get_type() == kTIMESTAMP || type_info.get_type() == kDATE) &&
             new_type_info.get_type() == kTIMESTAMP) {
    const auto dimen = (type_info.get_type() == kDATE) ? 0 : type_info.get_dimension();
    if (dimen != new_type_info.get_dimension()) {
      constval.bigintval = dimen < new_type_info.get_dimension()
                               ? DateTimeUtils::get_datetime_scaled_epoch(
                                     DateTimeUtils::ScalingType::ScaleUp,
                                     constval.bigintval,
                                     new_type_info.get_dimension() - dimen)
                               : DateTimeUtils::get_datetime_scaled_epoch(
                                     DateTimeUtils::ScalingType::ScaleDown,
                                     constval.bigintval,
                                     dimen - new_type_info.get_dimension());
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
                               new_type_info.is_string() || new_type_info.is_boolean())) {
    type_info = new_type_info;
    set_null_value();
  } else if (!is_null_value(type_info, constval) &&
             get_nullable_type_info(type_info) == new_type_info) {
    CHECK(!is_null);
    // relax nullability
    type_info = new_type_info;
    return;
  } else {
    throw std::runtime_error("Cast from " + type_info.get_type_name() + " to " +
                             new_type_info.get_type_name() + " not supported");
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
      constval.bigintval = NULL_BIGINT;
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
    case kARRAY:
      constval.arrayval = nullptr;
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
  const auto is_integral_type =
      new_type_info.is_integer() || new_type_info.is_decimal() || new_type_info.is_fp();
  if (is_integral_type && (type_info.is_time() || type_info.is_date())) {
    // Let the codegen phase deal with casts from date/time to a number.
    return makeExpr<UOper>(new_type_info, contains_agg, kCAST, shared_from_this());
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

  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      new_expr_pair_list;
  for (auto& p : expr_pair_list) {
    new_expr_pair_list.emplace_back(
        std::make_pair(p.first, p.second->deep_copy()->add_cast(ti)));
  }

  if (else_expr != nullptr) {
    else_expr = else_expr->add_cast(ti);
  }
  // Replace the current WHEN THEN pair list once we are sure all casts have succeeded
  expr_pair_list = new_expr_pair_list;

  type_info = ti;
  return shared_from_this();
}

std::shared_ptr<Analyzer::Expr> Subquery::add_cast(const SQLTypeInfo& new_type_info) {
  // not supported yet.
  CHECK(false);
  return nullptr;
}

void ColumnVar::check_group_by(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby) const {
  if (!groupby.empty()) {
    for (auto e : groupby) {
      auto c = std::dynamic_pointer_cast<ColumnVar>(e);
      if (c && get_table_id() == c->get_table_id() &&
          get_column_id() == c->get_column_id()) {
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

void KeyForStringExpr::group_predicates(std::list<const Expr*>& scan_predicates,
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

void SampleRatioExpr::group_predicates(std::list<const Expr*>& scan_predicates,
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

void LowerExpr::group_predicates(std::list<const Expr*>& scan_predicates,
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

void CardinalityExpr::group_predicates(std::list<const Expr*>& scan_predicates,
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

void WidthBucketExpr::group_predicates(std::list<const Expr*>& scan_predicates,
                                       std::list<const Expr*>& join_predicates,
                                       std::list<const Expr*>& const_predicates) const {
  std::set<int> rte_idx_set;
  target_value_->collect_rte_idx(rte_idx_set);
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
      if (get_table_id() == colvar->get_table_id() &&
          get_column_id() == colvar->get_column_id()) {
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
    if (get_table_id() == colvar->get_table_id() &&
        get_column_id() == colvar->get_column_id()) {
      return makeExpr<Var>(
          colvar->get_column_info(), colvar->get_rte_idx(), Var::kINPUT_OUTER, varno);
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
      if (get_table_id() == colvar->get_table_id() &&
          get_column_id() == colvar->get_column_id()) {
        return makeExpr<Var>(
            colvar->get_column_info(), colvar->get_rte_idx(), Var::kINPUT_OUTER, varno);
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
                           arg1);
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
    return (get_table_id() == rhs_cv.get_table_id()) &&
           (get_column_id() == rhs_cv.get_column_id()) &&
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
  return expr_list_match(tuple_, rhs_tuple_cols);
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
      return val1.bigintval == val2.bigintval;
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

bool RangeOper::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(RangeOper)) {
    return false;
  }
  const RangeOper& rhs_rg = dynamic_cast<const RangeOper&>(rhs);
  return left_inclusive_ == rhs_rg.left_inclusive_ &&
         right_inclusive_ == rhs_rg.right_inclusive_ &&
         *left_operand_ == *rhs_rg.left_operand_ &&
         *right_operand_ == *rhs_rg.right_operand_;
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

bool KeyForStringExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(KeyForStringExpr)) {
    return false;
  }
  const KeyForStringExpr& rhs_cl = dynamic_cast<const KeyForStringExpr&>(rhs);
  if (!(*arg == *rhs_cl.get_arg())) {
    return false;
  }
  return true;
}

bool SampleRatioExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(SampleRatioExpr)) {
    return false;
  }
  const SampleRatioExpr& rhs_cl = dynamic_cast<const SampleRatioExpr&>(rhs);
  if (!(*arg == *rhs_cl.get_arg())) {
    return false;
  }
  return true;
}

bool LowerExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(LowerExpr)) {
    return false;
  }

  return *arg == *dynamic_cast<const LowerExpr&>(rhs).get_arg();
}

bool CardinalityExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(CardinalityExpr)) {
    return false;
  }
  const CardinalityExpr& rhs_ca = dynamic_cast<const CardinalityExpr&>(rhs);
  if (!(*arg == *rhs_ca.get_arg())) {
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

bool WidthBucketExpr::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(WidthBucketExpr)) {
    return false;
  }
  const WidthBucketExpr& rhs_l = dynamic_cast<const WidthBucketExpr&>(rhs);
  if (!(*target_value_ == *rhs_l.get_target_value())) {
    return false;
  }
  if (!(*lower_bound_ == *rhs_l.get_lower_bound())) {
    return false;
  }
  if (!(*upper_bound_ == *rhs_l.get_upper_bound())) {
    return false;
  }
  if (!(*partition_count_ == *rhs_l.get_partition_count())) {
    return false;
  }
  return true;
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

bool WindowFunction::operator==(const Expr& rhs) const {
  const auto rhs_window = dynamic_cast<const WindowFunction*>(&rhs);
  if (!rhs_window) {
    return false;
  }
  if (kind_ != rhs_window->kind_ || args_.size() != rhs_window->args_.size() ||
      partition_keys_.size() != rhs_window->partition_keys_.size() ||
      order_keys_.size() != rhs_window->order_keys_.size()) {
    return false;
  }
  return expr_list_match(args_, rhs_window->args_) &&
         expr_list_match(partition_keys_, rhs_window->partition_keys_) &&
         expr_list_match(order_keys_, rhs_window->order_keys_);
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
  if (isNull() != casted_rhs.isNull()) {
    return false;
  }

  return true;
  ;
}

bool GeoUOper::operator==(const Expr& rhs) const {
  const auto rhs_geo = dynamic_cast<const GeoUOper*>(&rhs);
  if (!rhs_geo) {
    return false;
  }
  if (op_ != rhs_geo->getOp() || ti0_ != rhs_geo->getTypeInfo0() ||
      args0_.size() != rhs_geo->getArgs0().size()) {
    return false;
  }
  return expr_list_match(args0_, rhs_geo->getArgs0());
}

bool GeoBinOper::operator==(const Expr& rhs) const {
  const auto rhs_geo = dynamic_cast<const GeoBinOper*>(&rhs);
  if (!rhs_geo) {
    return false;
  }
  if (op_ != rhs_geo->getOp() || ti0_ != rhs_geo->getTypeInfo0() ||
      args0_.size() != rhs_geo->getArgs0().size()) {
    return false;
  }
  if (ti1_ != rhs_geo->getTypeInfo1() || args1_.size() != rhs_geo->getArgs1().size()) {
    return false;
  }
  return expr_list_match(args0_, rhs_geo->getArgs0()) ||
         expr_list_match(args1_, rhs_geo->getArgs1());
}

std::string ColumnVar::toString() const {
  return "(ColumnVar table: " + std::to_string(get_table_id()) +
         " column: " + std::to_string(get_column_id()) +
         " rte: " + std::to_string(rte_idx) + " " + get_type_info().get_type_name() +
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
  return "(Var table: " + std::to_string(get_table_id()) +
         " column: " + std::to_string(get_column_id()) +
         " rte: " + std::to_string(rte_idx) + " which_row: " + std::to_string(which_row) +
         " varno: " + std::to_string(varno) + ") ";
}

std::string Constant::toString() const {
  std::string str{"(Const "};
  if (is_null) {
    str += "NULL";
  } else if (type_info.is_array()) {
    const auto& elem_ti = type_info.get_elem_type();
    str += ::toString(type_info.get_type()) + ": " + ::toString(elem_ti.get_type());
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

std::string RangeOper::toString() const {
  const std::string lhs = left_inclusive_ ? "[" : "(";
  const std::string rhs = right_inclusive_ ? "]" : ")";
  return "(RangeOper " + lhs + " " + left_operand_->toString() + " , " +
         right_operand_->toString() + " " + rhs + " )";
}

std::string Subquery::toString() const {
  return "(Subquery ) ";
}

std::string InValues::toString() const {
  std::string str{"(IN "};
  str += arg->toString();
  str += "(";
  int cnt = 0;
  bool shorted_value_list_str = false;
  for (auto e : value_list) {
    str += e->toString();
    cnt++;
    if (cnt > 4) {
      shorted_value_list_str = true;
      break;
    }
  }
  if (shorted_value_list_str) {
    str += "... | ";
    str += "Total # values: ";
    str += std::to_string(value_list.size());
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
  int cnt = 0;
  bool shorted_value_list_str = false;
  for (const auto e : value_list) {
    str += std::to_string(e) + " ";
    cnt++;
    if (cnt > 4) {
      shorted_value_list_str = true;
      break;
    }
  }
  if (shorted_value_list_str) {
    str += "... | ";
    str += "Total # values: ";
    str += std::to_string(value_list.size());
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

std::string KeyForStringExpr::toString() const {
  std::string str{"KEY_FOR_STRING("};
  str += arg->toString();
  str += ") ";
  return str;
}

std::string SampleRatioExpr::toString() const {
  std::string str{"SAMPLE_RATIO("};
  str += arg->toString();
  str += ") ";
  return str;
}

std::string LowerExpr::toString() const {
  return "LOWER(" + arg->toString() + ") ";
}

std::string CardinalityExpr::toString() const {
  std::string str{"CARDINALITY("};
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

std::string WidthBucketExpr::toString() const {
  std::string str{"(WIDTH_BUCKET "};
  str += target_value_->toString();
  str += lower_bound_->toString();
  str += upper_bound_->toString();
  str += partition_count_->toString();
  return str + ") ";
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
    case kAPPROX_QUANTILE:
      agg = "APPROX_PERCENTILE";
      break;
    case kSINGLE_VALUE:
      agg = "SINGLE_VALUE";
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

std::string WindowFunction::toString() const {
  std::string result = "WindowFunction(" + ::toString(kind_);
  for (const auto& arg : args_) {
    result += " " + arg->toString();
  }
  return result + ") ";
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

std::string GeoUOper::toString() const {
  std::string fn;
  switch (op_) {
    case Geospatial::GeoBase::GeoOp::kPROJECTION:
      fn = "Geo";
      break;
    case Geospatial::GeoBase::GeoOp::kISEMPTY:
      fn = "ST_IsEmpty";
      break;
    case Geospatial::GeoBase::GeoOp::kISVALID:
      fn = "ST_IsValid";
      break;
    default:
      fn = "Geo_UNKNOWN";
      break;
  }
  std::string result = fn + "(";
  for (const auto& arg : args0_) {
    result += " " + arg->toString();
  }
  return result + " ) ";
}

std::string GeoBinOper::toString() const {
  std::string fn;
  switch (op_) {
    case Geospatial::GeoBase::GeoOp::kINTERSECTION:
      fn = "ST_Intersection";
      break;
    case Geospatial::GeoBase::GeoOp::kDIFFERENCE:
      fn = "ST_Difference";
      break;
    case Geospatial::GeoBase::GeoOp::kUNION:
      fn = "ST_Union";
      break;
    case Geospatial::GeoBase::GeoOp::kBUFFER:
      fn = "ST_Buffer";
      break;
    default:
      fn = "Geo_UNKNOWN";
      break;
  }
  std::string result = fn + "(";
  // TODO: generate wkt
  for (const auto& arg : args0_) {
    result += " " + arg->toString();
  }
  for (const auto& arg : args1_) {
    result += " " + arg->toString();
  }
  return result + " ) ";
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

void KeyForStringExpr::find_expr(bool (*f)(const Expr*),
                                 std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  arg->find_expr(f, expr_list);
}

void SampleRatioExpr::find_expr(bool (*f)(const Expr*),
                                std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  arg->find_expr(f, expr_list);
}

void LowerExpr::find_expr(bool (*f)(const Expr*),
                          std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
  } else {
    arg->find_expr(f, expr_list);
  }
}

void CardinalityExpr::find_expr(bool (*f)(const Expr*),
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

void WidthBucketExpr::find_expr(bool (*f)(const Expr*),
                                std::list<const Expr*>& expr_list) const {
  if (f(this)) {
    add_unique(expr_list);
    return;
  }
  target_value_->find_expr(f, expr_list);
  lower_bound_->find_expr(f, expr_list);
  upper_bound_->find_expr(f, expr_list);
  partition_count_->find_expr(f, expr_list);
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

void ArrayExpr::collect_rte_idx(std::set<int>& rte_idx_set) const {
  for (unsigned i = 0; i < getElementCount(); i++) {
    const auto expr = getElement(i);
    expr->collect_rte_idx(rte_idx_set);
  }
}

void FunctionOper::collect_rte_idx(std::set<int>& rte_idx_set) const {
  for (unsigned i = 0; i < getArity(); i++) {
    const auto expr = getArg(i);
    expr->collect_rte_idx(rte_idx_set);
  }
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

void ArrayExpr::collect_column_var(
    std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>& colvar_set,
    bool include_agg) const {
  for (unsigned i = 0; i < getElementCount(); i++) {
    const auto expr = getElement(i);
    expr->collect_column_var(colvar_set, include_agg);
  }
}

void FunctionOper::collect_column_var(
    std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>& colvar_set,
    bool include_agg) const {
  for (unsigned i = 0; i < getArity(); i++) {
    const auto expr = getArg(i);
    expr->collect_column_var(colvar_set, include_agg);
  }
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
  for (const auto& arg : args_) {
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

double WidthBucketExpr::get_bound_val(const Analyzer::Expr* bound_expr) const {
  CHECK(bound_expr);
  auto copied_expr = bound_expr->deep_copy();
  auto casted_expr = copied_expr->add_cast(SQLTypeInfo(kDOUBLE, false));
  CHECK(casted_expr);
  auto casted_constant = std::dynamic_pointer_cast<const Analyzer::Constant>(casted_expr);
  CHECK(casted_constant);
  return casted_constant->get_constval().doubleval;
}

int32_t WidthBucketExpr::get_partition_count_val() const {
  auto const_partition_count_expr =
      dynamic_cast<const Analyzer::Constant*>(partition_count_.get());
  if (!const_partition_count_expr) {
    return -1;
  }
  auto d = const_partition_count_expr->get_constval();
  switch (const_partition_count_expr->get_type_info().get_type()) {
    case kTINYINT:
      return d.tinyintval;
    case kSMALLINT:
      return d.smallintval;
    case kINT:
      return d.intval;
    case kBIGINT: {
      auto bi = d.bigintval;
      if (bi < 1 || bi > INT32_MAX) {
        return -1;
      }
      return bi;
    }
    default:
      return -1;
  }
}

namespace {

SQLTypes get_ti_from_geo(const Geospatial::GeoBase* geo) {
  CHECK(geo);
  switch (geo->getType()) {
    case Geospatial::GeoBase::GeoType::kPOINT: {
      return kPOINT;
    }
    case Geospatial::GeoBase::GeoType::kLINESTRING: {
      return kLINESTRING;
    }
    case Geospatial::GeoBase::GeoType::kPOLYGON: {
      return kPOLYGON;
    }
    case Geospatial::GeoBase::GeoType::kMULTIPOLYGON: {
      return kMULTIPOLYGON;
    }
    default:
      UNREACHABLE();
      return kNULLT;
  }
}

}  // namespace

// TODO: fixup null
GeoConstant::GeoConstant(std::unique_ptr<Geospatial::GeoBase>&& geo,
                         const SQLTypeInfo& ti)
    : GeoExpr(ti), geo_(std::move(geo)) {
  CHECK(geo_);
  if (get_ti_from_geo(geo_.get()) != ti.get_type()) {
    throw std::runtime_error("Conflicting types for geo data " + geo_->getWktString() +
                             " (type provided: " + ti.get_type_name() + ")");
  }
}

std::shared_ptr<Analyzer::Expr> GeoConstant::deep_copy() const {
  CHECK(geo_);
  return makeExpr<GeoConstant>(geo_->clone(), type_info);
}

std::string GeoConstant::toString() const {
  std::string str{"(GeoConstant "};
  CHECK(geo_);
  str += geo_->getWktString();
  str += ") ";
  return str;
}

std::string GeoConstant::getWKTString() const {
  CHECK(geo_);
  return geo_->getWktString();
}

bool GeoConstant::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(GeoConstant)) {
    return false;
  }
  const GeoConstant& rhs_c = dynamic_cast<const GeoConstant&>(rhs);
  if (type_info != rhs_c.get_type_info() /*|| is_null != rhs_c.get_is_null()*/) {
    return false;
  }
  /* TODO: constant nulls
  if (is_null && rhs_c.get_is_null()) {
    return true;
  }

  */
  return *geo_ == *rhs_c.geo_;
}

size_t GeoConstant::physicalCols() const {
  UNREACHABLE();
  return type_info.get_physical_coord_cols();
}

std::shared_ptr<Analyzer::Constant> GeoConstant::makePhysicalConstant(
    const size_t index) const {
  // TODO: handle bounds, etc
  const auto num_phys_coords = type_info.get_physical_coord_cols();
  CHECK_GE(num_phys_coords, 0);
  CHECK_LE(index, size_t(num_phys_coords));
  SQLTypeInfo ti = type_info;

  std::vector<double> coords;
  std::vector<double> bounds;
  std::vector<int> ring_sizes;
  std::vector<int> poly_rings;

  Geospatial::GeoTypesFactory::getGeoColumns(
      geo_->getWktString(), ti, coords, bounds, ring_sizes, poly_rings);

  switch (index) {
    case 0:  // coords
      return Geospatial::convert_coords(coords, ti);
    case 1:  // ring sizes
      return Geospatial::convert_rings(ring_sizes);
    case 2:  // poly rings
      return Geospatial::convert_rings(poly_rings);
    default:
      UNREACHABLE();
  }

  UNREACHABLE();
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> GeoConstant::add_cast(const SQLTypeInfo& new_type_info) {
  // TODO: we should eliminate the notion of input and output SRIDs on a type. A type can
  // only have 1 SRID. A cast or transforms changes the SRID of the type.
  // NOTE: SRID 0 indicates set srid, skip cast
  if (!(get_type_info().get_input_srid() == 0) &&
      (get_type_info().get_input_srid() != new_type_info.get_output_srid())) {
    // run cast
    CHECK(geo_);
    if (!geo_->transform(get_type_info().get_input_srid(),
                         new_type_info.get_output_srid())) {
      throw std::runtime_error("Failed to transform constant geometry: " + toString());
    }
  }
  return makeExpr<GeoConstant>(std::move(geo_), new_type_info);
}

GeoOperator::GeoOperator(const SQLTypeInfo& ti,
                         const std::string& name,
                         const std::vector<std::shared_ptr<Analyzer::Expr>>& args,
                         const std::optional<int>& output_srid_override)
    : GeoExpr(ti)
    , name_(name)
    , args_(args)
    , output_srid_override_(output_srid_override) {}

std::shared_ptr<Analyzer::Expr> GeoOperator::deep_copy() const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  for (size_t i = 0; i < args_.size(); i++) {
    args.push_back(args_[i]->deep_copy());
  }
  return makeExpr<GeoOperator>(type_info, name_, args, output_srid_override_);
}

void GeoOperator::collect_rte_idx(std::set<int>& rte_idx_set) const {
  for (size_t i = 0; i < size(); i++) {
    const auto expr = getOperand(i);
    expr->collect_rte_idx(rte_idx_set);
  }
}

void GeoOperator::collect_column_var(
    std::set<const ColumnVar*, bool (*)(const ColumnVar*, const ColumnVar*)>& colvar_set,
    bool include_agg) const {
  for (size_t i = 0; i < size(); i++) {
    const auto expr = getOperand(i);
    expr->collect_column_var(colvar_set, include_agg);
  }
}

std::string GeoOperator::toString() const {
  std::string str{"(" + name_ + " "};
  for (const auto& arg : args_) {
    str += arg->toString();
  }
  str += ")";
  return str;
}

bool GeoOperator::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(GeoOperator)) {
    return false;
  }
  const GeoOperator& rhs_go = dynamic_cast<const GeoOperator&>(rhs);
  if (getName() != rhs_go.getName()) {
    return false;
  }
  if (rhs_go.size() != size()) {
    return false;
  }
  for (size_t i = 0; i < size(); i++) {
    if (args_[i].get() != rhs_go.getOperand(i)) {
      return false;
    }
  }
  return true;
}

size_t GeoOperator::size() const {
  return args_.size();
}

Analyzer::Expr* GeoOperator::getOperand(const size_t index) const {
  CHECK_LT(index, args_.size());
  return args_[index].get();
}

std::shared_ptr<Analyzer::Expr> GeoOperator::add_cast(const SQLTypeInfo& new_type_info) {
  UNREACHABLE();
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> GeoTransformOperator::deep_copy() const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  for (size_t i = 0; i < args_.size(); i++) {
    args.push_back(args_[i]->deep_copy());
  }
  return makeExpr<GeoTransformOperator>(
      type_info, name_, args, input_srid_, output_srid_);
}

std::string GeoTransformOperator::toString() const {
  std::string str{"(" + name_ + " "};
  for (const auto& arg : args_) {
    str += arg->toString();
  }
  str +=
      " : " + std::to_string(input_srid_) + " -> " + std::to_string(output_srid_) + " ";
  str += ")";
  return str;
}

bool GeoTransformOperator::operator==(const Expr& rhs) const {
  if (typeid(rhs) != typeid(GeoTransformOperator)) {
    return false;
  }
  const GeoTransformOperator& rhs_gto = dynamic_cast<const GeoTransformOperator&>(rhs);
  if (getName() != rhs_gto.getName()) {
    return false;
  }
  if (rhs_gto.size() != size()) {
    return false;
  }
  for (size_t i = 0; i < size(); i++) {
    if (args_[i].get() != rhs_gto.getOperand(i)) {
      return false;
    }
  }
  if (input_srid_ != rhs_gto.input_srid_) {
    return false;
  }
  if (output_srid_ != rhs_gto.output_srid_) {
    return false;
  }
  return true;
}

}  // namespace Analyzer

bool expr_list_match(const std::vector<std::shared_ptr<Analyzer::Expr>>& lhs,
                     const std::vector<std::shared_ptr<Analyzer::Expr>>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!(*lhs[i] == *rhs[i])) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<Analyzer::Expr> remove_cast(const std::shared_ptr<Analyzer::Expr>& expr) {
  const auto uoper = dynamic_cast<const Analyzer::UOper*>(expr.get());
  if (!uoper || uoper->get_optype() != kCAST) {
    return expr;
  }
  return uoper->get_own_operand();
}
