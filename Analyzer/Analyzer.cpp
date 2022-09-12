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
#include "DataProvider/DataProvider.h"
#include "IR/TypeUtils.h"
#include "QueryEngine/DateTimeUtils.h"
#include "QueryEngine/Execute.h"  // TODO: remove
#include "Shared/DateConverters.h"
#include "Shared/misc.h"
#include "Shared/sqltypes.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

bool expr_is_null(const hdk::ir::Expr* expr) {
  if (expr->type()->isNull()) {
    return true;
  }
  const auto const_expr = dynamic_cast<const hdk::ir::Constant*>(expr);
  return const_expr && const_expr->get_is_null();
}

void check_like_expr(const std::string& like_str, char escape_char) {
  if (like_str.back() == escape_char) {
    throw std::runtime_error("LIKE pattern must not end with escape character.");
  }
}

bool translate_to_like_pattern(std::string& pattern_str, char escape_char) {
  char prev_char = '\0';
  char prev_prev_char = '\0';
  std::string like_str;
  for (char& cur_char : pattern_str) {
    if (prev_char == escape_char || isalnum(cur_char) || cur_char == ' ' ||
        cur_char == '.') {
      like_str.push_back((cur_char == '.') ? '_' : cur_char);
      prev_prev_char = prev_char;
      prev_char = cur_char;
      continue;
    }
    if (prev_char == '.' && prev_prev_char != escape_char) {
      if (cur_char == '*' || cur_char == '+') {
        if (cur_char == '*') {
          like_str.pop_back();
        }
        // .* --> %
        // .+ --> _%
        like_str.push_back('%');
        prev_prev_char = prev_char;
        prev_char = cur_char;
        continue;
      }
    }
    return false;
  }
  pattern_str = like_str;
  return true;
}

bool test_is_simple_expr(const std::string& like_str, char escape_char) {
  // if not bounded by '%' then not a simple string
  if (like_str.size() < 2 || like_str[0] != '%' || like_str[like_str.size() - 1] != '%') {
    return false;
  }
  // if the last '%' is escaped then not a simple string
  if (like_str[like_str.size() - 2] == escape_char &&
      like_str[like_str.size() - 3] != escape_char) {
    return false;
  }
  for (size_t i = 1; i < like_str.size() - 1; i++) {
    if (like_str[i] == '%' || like_str[i] == '_' || like_str[i] == '[' ||
        like_str[i] == ']') {
      if (like_str[i - 1] != escape_char) {
        return false;
      }
    }
  }
  return true;
}
void erase_cntl_chars(std::string& like_str, char escape_char) {
  char prev_char = '\0';
  // easier to create new string of allowable chars
  // rather than erase chars from
  // existing string
  std::string new_str;
  for (char& cur_char : like_str) {
    if (cur_char == '%' || cur_char == escape_char) {
      if (prev_char != escape_char) {
        prev_char = cur_char;
        continue;
      }
    }
    new_str.push_back(cur_char);
    prev_char = cur_char;
  }
  like_str = new_str;
}

int precisionOrZero(const hdk::ir::Type* type) {
  return type->isDecimal() ? type->as<hdk::ir::DecimalType>()->precision() : 0;
}

int scaleOrZero(const hdk::ir::Type* type) {
  return type->isDecimal() ? type->as<hdk::ir::DecimalType>()->scale() : 0;
}

}  // namespace

namespace Analyzer {

const hdk::ir::Type* analyze_type_info(SQLOps op,
                                       const hdk::ir::Type* left_type,
                                       const hdk::ir::Type* right_type,
                                       const hdk::ir::Type** new_left_type,
                                       const hdk::ir::Type** new_right_type) {
  auto& ctx = left_type->ctx();
  const hdk::ir::Type* result_type;
  const hdk::ir::Type* common_type;
  *new_left_type = left_type;
  *new_right_type = right_type;
  if (IS_LOGIC(op)) {
    if (!left_type->isBoolean() || !right_type->isBoolean()) {
      throw std::runtime_error(
          "non-boolean operands cannot be used in logic operations.");
    }
    result_type = ctx.boolean();
  } else if (IS_COMPARISON(op)) {
    if (!left_type->equal(right_type)) {
      if (left_type->isNumber() && right_type->isNumber()) {
        common_type = common_numeric_type(left_type, right_type);
        *new_left_type = common_type->withNullable(left_type->nullable());
        *new_right_type = common_type->withNullable(right_type->nullable());
      } else if (left_type->isDateTime() && right_type->isDateTime()) {
        switch (left_type->id()) {
          case hdk::ir::Type::kTimestamp:
            switch (right_type->id()) {
              case hdk::ir::Type::kTime:
                throw std::runtime_error("Cannont compare between TIMESTAMP and TIME.");
                break;
              case hdk::ir::Type::kDate:
                *new_left_type = left_type;
                *new_right_type = (*new_left_type)->withNullable(right_type->nullable());
                break;
              case hdk::ir::Type::kTimestamp:
                *new_left_type = ctx.timestamp(
                    std::max(left_type->as<hdk::ir::DateTimeBaseType>()->unit(),
                             right_type->as<hdk::ir::DateTimeBaseType>()->unit()),
                    left_type->nullable());
                *new_right_type = ctx.timestamp(
                    std::max(left_type->as<hdk::ir::DateTimeBaseType>()->unit(),
                             right_type->as<hdk::ir::DateTimeBaseType>()->unit()),
                    right_type->nullable());
                break;
              default:
                CHECK(false);
            }
            break;
          case hdk::ir::Type::kTime:
            switch (right_type->id()) {
              case hdk::ir::Type::kTimestamp:
                throw std::runtime_error("Cannont compare between TIME and TIMESTAMP.");
                break;
              case hdk::ir::Type::kDate:
                throw std::runtime_error("Cannont compare between TIME and DATE.");
                break;
              case hdk::ir::Type::kTime:
                *new_left_type = ctx.time64(
                    std::max(left_type->as<hdk::ir::DateTimeBaseType>()->unit(),
                             right_type->as<hdk::ir::DateTimeBaseType>()->unit()),
                    left_type->nullable());
                *new_right_type = ctx.time64(
                    std::max(left_type->as<hdk::ir::DateTimeBaseType>()->unit(),
                             right_type->as<hdk::ir::DateTimeBaseType>()->unit()),
                    right_type->nullable());
                break;
              default:
                CHECK(false);
            }
            break;
          case hdk::ir::Type::kDate:
            switch (right_type->id()) {
              case hdk::ir::Type::kTimestamp:
                *new_left_type = right_type->withNullable(left_type->nullable());
                *new_right_type = right_type;
                break;
              case hdk::ir::Type::kDate:
                *new_left_type =
                    ctx.date64(hdk::ir::TimeUnit::kSecond, left_type->nullable());
                *new_right_type =
                    ctx.date64(hdk::ir::TimeUnit::kSecond, right_type->nullable());
                break;
              case hdk::ir::Type::kTime:
                throw std::runtime_error("Cannont compare between DATE and TIME.");
                break;
              default:
                CHECK(false);
            }
            break;
          default:
            CHECK(false);
        }
      } else if ((left_type->isString() || left_type->isExtDictionary()) &&
                 right_type->isTime()) {
        *new_left_type = right_type->withNullable(left_type->nullable());
        *new_right_type = right_type;
      } else if (left_type->isTime() &&
                 (right_type->isString() || right_type->isExtDictionary())) {
        *new_left_type = left_type;
        *new_right_type = left_type->withNullable(right_type->nullable());
      } else if ((left_type->isString() || left_type->isExtDictionary()) &&
                 (right_type->isString() || right_type->isExtDictionary())) {
        *new_left_type = left_type;
        *new_right_type = right_type;
      } else if (left_type->isBoolean() && right_type->isBoolean()) {
        const bool nullable = left_type->nullable() || right_type->nullable();
        common_type = ctx.boolean(nullable);
        *new_left_type = common_type;
        *new_right_type = common_type;
      } else {
        throw std::runtime_error("Cannot compare between " + left_type->toString() +
                                 " and " + right_type->toString());
      }
    }
    result_type = ctx.boolean();
  } else if (op == kMINUS && (left_type->isDate() || left_type->isTimestamp()) &&
             right_type->isInterval()) {
    *new_left_type = left_type;
    *new_right_type = right_type;
    result_type = left_type;
  } else if (IS_ARITHMETIC(op)) {
    if (!(left_type->isNumber() || left_type->isInterval()) ||
        !(right_type->isNumber() || right_type->isInterval())) {
      throw std::runtime_error("non-numeric operands in arithmetic operations.");
    }
    if (op == kMODULO && (!left_type->isInteger() || !right_type->isInteger())) {
      throw std::runtime_error("non-integer operands in modulo operation.");
    }
    common_type = common_numeric_type(left_type, right_type);
    if (common_type->isDecimal()) {
      if (op == kMULTIPLY) {
        // Decimal multiplication requires common_type adjustment:
        // dimension and scale of the result should be increased.
        auto left_precision = precisionOrZero(left_type);
        auto left_scale = scaleOrZero(left_type);
        auto right_precision = precisionOrZero(right_type);
        auto right_scale = scaleOrZero(right_type);
        auto new_precision = left_precision + right_precision;
        // If new dimension is over 20 digits, the result may overflow, or it may not.
        // Rely on the runtime overflow detection rather than a static check here.
        new_precision =
            std::max(common_type->as<hdk::ir::DecimalType>()->precision(), new_precision);
        common_type =
            ctx.decimal(common_type->size(), new_precision, left_scale + right_scale);
      } else if (op == kPLUS || op == kMINUS) {
        // Scale should remain the same but dimension could actually go up
        common_type =
            ctx.decimal(common_type->size(),
                        common_type->as<hdk::ir::DecimalType>()->precision() + 1,
                        common_type->as<hdk::ir::DecimalType>()->scale());
      }
    }
    *new_left_type = common_type->withNullable(left_type->nullable());
    *new_right_type = common_type->withNullable(right_type->nullable());
    if (common_type->isDecimal() && op == kMULTIPLY) {
      if ((*new_left_type)->isDecimal()) {
        *new_left_type = ctx.decimal((*new_left_type)->size(),
                                     precisionOrZero(*new_left_type),
                                     scaleOrZero(left_type),
                                     left_type->nullable());
      }
      if ((*new_right_type)->isDecimal()) {
        *new_right_type = ctx.decimal((*new_right_type)->size(),
                                      precisionOrZero(*new_right_type),
                                      scaleOrZero(right_type),
                                      right_type->nullable());
      }
    }
    result_type = common_type;
  } else {
    throw std::runtime_error("invalid binary operator type.");
  }
  result_type =
      result_type->withNullable(left_type->nullable() || right_type->nullable());
  return result_type;
}

const hdk::ir::Type* common_string_type(const hdk::ir::Type* type1,
                                        const hdk::ir::Type* type2) {
  auto& ctx = type1->ctx();
  const hdk::ir::Type* common_type;
  auto nullable = type1->nullable() || type2->nullable();
  CHECK(type1->isString() || type1->isExtDictionary());
  CHECK(type2->isString() || type2->isExtDictionary());
  // if type1 and type2 have the same DICT encoding then keep it
  // otherwise, they must be decompressed
  if (type1->isExtDictionary() && type2->isExtDictionary()) {
    auto dict_id1 = type1->as<hdk::ir::ExtDictionaryType>()->dictId();
    auto dict_id2 = type2->as<hdk::ir::ExtDictionaryType>()->dictId();
    if (dict_id1 == dict_id2 || dict_id1 == TRANSIENT_DICT(dict_id2)) {
      common_type = ctx.extDict(ctx.text(), std::min(dict_id1, dict_id2), 4, nullable);
    } else {
      common_type = ctx.text();
    }
  } else if (type1->isVarChar() && type2->isVarChar()) {
    auto length = std::max(type1->as<hdk::ir::VarCharType>()->maxLength(),
                           type2->as<hdk::ir::VarCharType>()->maxLength());
    common_type = ctx.varChar(length, nullable);
  } else {
    common_type = ctx.text();
  }

  return common_type;
}

const hdk::ir::Type* common_numeric_type(const hdk::ir::Type* type1,
                                         const hdk::ir::Type* type2) {
  auto& ctx = type1->ctx();
  bool nullable = type1->nullable() || type2->nullable();
  if (type1->id() == type2->id()) {
    CHECK(((type1->isNumber() || type1->isInterval()) &&
           (type2->isNumber() || type2->isInterval())) ||
          (type1->isBoolean() && type2->isBoolean()));

    if (type1->isDecimal()) {
      auto precision = std::max(type1->as<hdk::ir::DecimalType>()->precision(),
                                type2->as<hdk::ir::DecimalType>()->precision());
      auto scale = std::max(type1->as<hdk::ir::DecimalType>()->scale(),
                            type2->as<hdk::ir::DecimalType>()->scale());
      return ctx.decimal(
          std::max(type1->size(), type2->size()), precision, scale, nullable);
    }
    if (type1->isInteger()) {
      return ctx.integer(std::max(type1->size(), type2->size()), nullable);
    }
    if (type1->isFloatingPoint()) {
      if (type1->size() != type2->size()) {
        return ctx.fp64(nullable);
      }
      return type1->withNullable(nullable);
    }
    if (type1->isInterval()) {
      auto unit = std::max(type1->as<hdk::ir::IntervalType>()->unit(),
                           type2->as<hdk::ir::IntervalType>()->unit());
      return ctx.interval(std::max(type1->size(), type2->size()), unit);
    }
    CHECK(type1->isBoolean());
    return type1->withNullable(nullable);
  }
  std::string timeinterval_op_error{
      "Operator type not supported for time interval arithmetic: "};
  if (type1->isInterval()) {
    if (!type2->isInteger()) {
      throw std::runtime_error(timeinterval_op_error + type2->toString());
    }
    return type1->withNullable(nullable);
  }
  if (type2->isInterval()) {
    if (!type1->isInteger()) {
      throw std::runtime_error(timeinterval_op_error + type1->toString());
    }
    return type2->withNullable(nullable);
  }
  CHECK(type1->isNumber() && type2->isNumber());
  // Don't allow decimal in type2 to reduce number of option in the following switch.
  if (type2->isDecimal()) {
    std::swap(type1, type2);
  }
  switch (type1->id()) {
    case hdk::ir::Type::kInteger:
      switch (type2->id()) {
        case hdk::ir::Type::kFloatingPoint:
          return type2->withNullable(nullable);
        default:
          CHECK(false);
      }
      break;
    case hdk::ir::Type::kFloatingPoint:
      return type1->withNullable(nullable);
    case hdk::ir::Type::kDecimal:
      switch (type2->id()) {
        case hdk::ir::Type::kInteger: {
          auto precision = type1->as<hdk::ir::DecimalType>()->precision();
          auto scale = type1->as<hdk::ir::DecimalType>()->scale();
          if (type2->size() == 8) {
            precision = 19;
          } else if (type2->size() == 4) {
            precision = std::max(std::min(19, 10 + scale), precision);
          } else {
            CHECK(type2->size() <= 2);
            precision = std::max(5 + scale, precision);
          }
          return ctx.decimal(type1->size(), precision, scale, nullable);
        }
        case hdk::ir::Type::kFloatingPoint:
          return type2->withNullable(nullable);
        default:
          CHECK(false);
      }
      break;
    default:
      CHECK(false);
  }
  return nullptr;
}

hdk::ir::ExprPtr analyzeIntValue(const int64_t intval) {
  SQLTypes t;
  Datum d;
  if (intval >= INT16_MIN && intval <= INT16_MAX) {
    t = kSMALLINT;
    d.smallintval = (int16_t)intval;
  } else if (intval >= INT32_MIN && intval <= INT32_MAX) {
    t = kINT;
    d.intval = (int32_t)intval;
  } else {
    t = kBIGINT;
    d.bigintval = intval;
  }
  return hdk::ir::makeExpr<hdk::ir::Constant>(t, false, d);
}

hdk::ir::ExprPtr analyzeFixedPtValue(const int64_t numericval,
                                     const int scale,
                                     const int precision) {
  auto type = hdk::ir::Context::defaultCtx().decimal64(precision, scale);
  Datum d;
  d.bigintval = numericval;
  return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, d);
}

hdk::ir::ExprPtr analyzeStringValue(const std::string& stringval) {
  auto type =
      hdk::ir::Context::defaultCtx().varChar(static_cast<int>(stringval.length()));
  Datum d;
  d.stringval = new std::string(stringval);
  return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, d);
}

bool exprs_share_one_and_same_rte_idx(const hdk::ir::ExprPtr& lhs_expr,
                                      const hdk::ir::ExprPtr& rhs_expr) {
  std::set<int> lhs_rte_idx;
  lhs_expr->collect_rte_idx(lhs_rte_idx);
  CHECK(!lhs_rte_idx.empty());
  std::set<int> rhs_rte_idx;
  rhs_expr->collect_rte_idx(rhs_rte_idx);
  CHECK(!rhs_rte_idx.empty());
  return lhs_rte_idx.size() == 1UL && lhs_rte_idx == rhs_rte_idx;
}

const hdk::ir::Type* get_str_dict_cast_type(const hdk::ir::Type* lhs_type,
                                            const hdk::ir::Type* rhs_type,
                                            const Executor* executor) {
  CHECK(lhs_type->isExtDictionary());
  CHECK(rhs_type->isExtDictionary());
  const auto lhs_dict_id = lhs_type->as<hdk::ir::ExtDictionaryType>()->dictId();
  const auto rhs_dict_id = rhs_type->as<hdk::ir::ExtDictionaryType>()->dictId();
  CHECK_NE(lhs_dict_id, rhs_dict_id);
  if (lhs_dict_id == TRANSIENT_DICT_ID) {
    return rhs_type;
  }
  if (rhs_dict_id == TRANSIENT_DICT_ID) {
    return lhs_type;
  }
  // When translator is used from DAG builder, executor is not available.
  // In this case, simply choose LHS and revise the decision later.
  if (!executor) {
    return lhs_type;
  }
  // If here then neither lhs or rhs type was transient, we should see which
  // type has the largest dictionary and make that the destination type
  const auto lhs_sdp = executor->getStringDictionaryProxy(lhs_dict_id, true);
  const auto rhs_sdp = executor->getStringDictionaryProxy(rhs_dict_id, true);
  return lhs_sdp->entryCount() >= rhs_sdp->entryCount() ? lhs_type : rhs_type;
}

const hdk::ir::Type* common_string_type(const hdk::ir::Type* type1,
                                        const hdk::ir::Type* type2,
                                        const Executor* executor) {
  auto& ctx = type1->ctx();
  const hdk::ir::Type* common_type;
  auto nullable = type1->nullable() || type2->nullable();
  CHECK(type1->isString() || type1->isExtDictionary());
  CHECK(type2->isString() || type2->isExtDictionary());
  // if type1 and type2 have the same DICT encoding then keep it
  // otherwise, they must be decompressed
  if (type1->isExtDictionary() && type2->isExtDictionary()) {
    auto dict_id1 = type1->as<hdk::ir::ExtDictionaryType>()->dictId();
    auto dict_id2 = type2->as<hdk::ir::ExtDictionaryType>()->dictId();
    if (dict_id1 == dict_id2 || dict_id1 == TRANSIENT_DICT(dict_id2)) {
      common_type = ctx.extDict(ctx.text(), std::min(dict_id1, dict_id2), 4, nullable);
    } else {
      common_type = get_str_dict_cast_type(type1, type2, executor);
    }
  } else if (type1->isVarChar() && type2->isVarChar()) {
    auto length = std::max(type1->as<hdk::ir::VarCharType>()->maxLength(),
                           type2->as<hdk::ir::VarCharType>()->maxLength());
    common_type = ctx.varChar(length, nullable);
  } else {
    common_type = ctx.text();
  }

  return common_type;
}

hdk::ir::ExprPtr normalizeOperExpr(const SQLOps optype,
                                   const SQLQualifier qual,
                                   hdk::ir::ExprPtr left_expr,
                                   hdk::ir::ExprPtr right_expr,
                                   const Executor* executor) {
  if ((left_expr->type()->isDate() &&
       left_expr->type()->as<hdk::ir::DateType>()->unit() == hdk::ir::TimeUnit::kDay) ||
      (right_expr->type()->isDate() &&
       right_expr->type()->as<hdk::ir::DateType>()->unit() == hdk::ir::TimeUnit::kDay)) {
    // Do not propogate encoding
    left_expr = left_expr->decompress();
    right_expr = right_expr->decompress();
  }
  auto left_type = left_expr->type();
  auto right_type = right_expr->type();
  auto& ctx = left_type->ctx();
  if (qual != kONE) {
    // subquery not supported yet.
    CHECK(!std::dynamic_pointer_cast<hdk::ir::ScalarSubquery>(right_expr));
    if (!right_type->isArray()) {
      throw std::runtime_error(
          "Existential or universal qualifiers can only be used in front of a subquery "
          "or an "
          "expression of array type.");
    }
    right_type = right_type->as<hdk::ir::ArrayBaseType>()->elemType();
  }
  const hdk::ir::Type* new_left_type;
  const hdk::ir::Type* new_right_type;
  auto result_type =
      analyze_type_info(optype, left_type, right_type, &new_left_type, &new_right_type);
  if (result_type->isInterval()) {
    return hdk::ir::makeExpr<hdk::ir::BinOper>(
        result_type, false, optype, qual, left_expr, right_expr);
  }
  // No executor means we build an expression for further normalization.
  // Required casts will be added at normalization. Double normalization
  // with casts may cause a change in the resulting type. E.g. it would
  // increase dimension for decimals on each normalization for arithmetic
  // operations.
  if (executor) {
    if (!left_type->equal(new_left_type)) {
      left_expr = left_expr->add_cast(new_left_type);
    }
    if (!right_type->equal(new_right_type)) {
      if (qual == kONE) {
        right_expr = right_expr->add_cast(new_right_type);
      } else {
        right_expr = right_expr->add_cast(
            ctx.arrayVarLen(new_right_type, 4, new_right_type->nullable()));
      }
    }
  }

  // No executor means we are building DAG. Skip normalization at this
  // step and perform it later on execution unit build.
  if (IS_COMPARISON(optype) && executor) {
    if (new_left_type->isExtDictionary() && new_right_type->isExtDictionary()) {
      if (new_left_type->as<hdk::ir::ExtDictionaryType>()->dictId() !=
          new_right_type->as<hdk::ir::ExtDictionaryType>()->dictId()) {
        // Join framework does its own string dictionary translation
        // (at least partly since the rhs table projection does not use
        // the normal runtime execution framework), do if we detect
        // that the rte idxs of the two tables are different, bail
        // on translating
        const bool should_translate_strings =
            exprs_share_one_and_same_rte_idx(left_expr, right_expr);
        if (should_translate_strings && (optype == kEQ || optype == kNE)) {
          CHECK(executor);
          // Make the type we're casting to the transient dictionary, if it exists,
          // otherwise the largest dictionary in terms of number of entries
          auto type = get_str_dict_cast_type(new_left_type, new_right_type, executor);
          auto& expr_to_cast = type->equal(new_left_type) ? right_expr : left_expr;
          expr_to_cast = expr_to_cast->add_cast(type, true);
        } else {  // Ordered comparison operator
          // We do not currently support ordered (i.e. >, <=) comparisons between
          // dictionary-encoded columns, and need to decompress when translation
          // is turned off even for kEQ and KNE
          left_expr = left_expr->decompress();
          right_expr = right_expr->decompress();
        }
      } else {  // Strings shared comp param
        if (!(optype == kEQ || optype == kNE)) {
          // We do not currently support ordered (i.e. >, <=) comparisons between
          // encoded columns, so try to decode (will only succeed with watchdog off)
          left_expr = left_expr->decompress();
          right_expr = right_expr->decompress();
        } else {
          // do nothing, can directly support equals/non-equals comparisons between two
          // dictionary encoded columns sharing the same dictionary as these are
          // effectively integer comparisons in the same dictionary space
        }
      }
    } else if (new_left_type->isExtDictionary() && !new_right_type->isExtDictionary()) {
      CHECK(new_right_type->isString());
      auto type = ctx.extDict(new_right_type,
                              new_left_type->as<hdk::ir::ExtDictionaryType>()->dictId(),
                              4,
                              new_right_type->nullable());
      right_expr = right_expr->add_cast(type);
    } else if (new_right_type->isExtDictionary() && !new_left_type->isExtDictionary()) {
      CHECK(new_left_type->isString());
      auto type = ctx.extDict(new_left_type,
                              new_right_type->as<hdk::ir::ExtDictionaryType>()->dictId(),
                              4,
                              new_left_type->nullable());
      left_expr = left_expr->add_cast(type);
    } else {
      left_expr = left_expr->decompress();
      right_expr = right_expr->decompress();
    }
  } else if (!IS_COMPARISON(optype)) {
    left_expr = left_expr->decompress();
    right_expr = right_expr->decompress();
  }
  bool has_agg = (left_expr->get_contains_agg() || right_expr->get_contains_agg());
  return hdk::ir::makeExpr<hdk::ir::BinOper>(
      result_type, has_agg, optype, qual, left_expr, right_expr);
}

hdk::ir::ExprPtr normalizeCaseExpr(
    const std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>>& expr_pair_list,
    const hdk::ir::ExprPtr else_e_in,
    const Executor* executor) {
  const hdk::ir::Type* type = nullptr;
  bool has_agg = false;
  // We need to keep track of whether there was at
  // least one none-encoded string literal expression
  // type among any of the case sub-expressions separately
  // from rest of type determination logic, as it will
  // be casted to the output dictionary type if all output
  // types are either dictionary encoded or none-encoded
  // literals, or kept as none-encoded if all sub-expression
  // types are none-encoded (column or literal)
  const hdk::ir::Type* none_encoded_literal_type = nullptr;

  for (auto& p : expr_pair_list) {
    auto e1 = p.first;
    CHECK(e1->type()->isBoolean());
    auto e2 = p.second;
    if (e2->get_contains_agg()) {
      has_agg = true;
    }
    const auto& e2_type = e2->type();
    if (e2_type->isString() && !std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(e2)) {
      none_encoded_literal_type =
          none_encoded_literal_type
              ? common_string_type(none_encoded_literal_type, e2_type, executor)
              : e2_type;
      continue;
    }

    if (type == nullptr) {
      type = e2_type;
    } else if (e2_type->isNull()) {
      type = type->withNullable(true);
      e2->set_type_info(type);
    } else if (!type->equal(e2_type)) {
      if ((type->isString() || type->isExtDictionary()) &&
          (e2_type->isString() || e2_type->isExtDictionary())) {
        // Executor is needed to determine which dictionary is the largest
        // in case of two dictionary types with different encodings
        type = common_string_type(type, e2_type, executor);
      } else if (type->isNumber() && e2_type->isNumber()) {
        type = common_numeric_type(type, e2_type);
      } else if (type->isBoolean() && e2_type->isBoolean()) {
        type = common_numeric_type(type, e2_type);
      } else {
        throw std::runtime_error(
            "expressions in THEN clause must be of the same or compatible types.");
      }
    }
  }
  auto else_e = else_e_in;
  if (else_e) {
    const auto& else_type = else_e->type();
    if (else_e->get_contains_agg()) {
      has_agg = true;
    }
    if (else_type->isString() &&
        !std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(else_e)) {
      none_encoded_literal_type =
          none_encoded_literal_type
              ? common_string_type(none_encoded_literal_type, else_type, executor)
              : else_type;
    } else {
      if (type == nullptr) {
        type = else_type;
      } else if (expr_is_null(else_e.get())) {
        type = type->withNullable(true);
        else_e->set_type_info(type);
      } else if (!type->equal(else_type)) {
        type = type->withNullable(true);
        if ((type->isString() || type->isExtDictionary()) &&
            (else_type->isString() || else_type->isExtDictionary())) {
          // Executor is needed to determine which dictionary is the largest
          // in case of two dictionary types with different encodings
          type = common_string_type(type, else_type, executor);
        } else if (type->isNumber() && else_type->isNumber()) {
          type = common_numeric_type(type, else_type);
        } else if (type->isBoolean() && else_type->isBoolean()) {
          type = common_numeric_type(type, else_type);
        } else if (!logicalType(type)->equal(logicalType(else_type))) {
          throw std::runtime_error(
              // types differing by encoding will be resolved at decode
              "Expressions in ELSE clause must be of the same or compatible types as "
              "those in the THEN clauses.");
        }
      }
    }
  } else if (type) {
    type = type->withNullable(true);
  }

  if ((!type || type->isNull()) && none_encoded_literal_type) {
    // If we haven't set a type so far it's because
    // every case sub-expression has a none-encoded
    // literal output. Make this our output type
    type = none_encoded_literal_type;
  }

  if (!type || type->isNull()) {
    throw std::runtime_error(
        "Cannot deduce the type for case expressions, all branches null");
  }

  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> cast_expr_pair_list;
  for (auto p : expr_pair_list) {
    cast_expr_pair_list.emplace_back(p.first,
                                     executor ? p.second->add_cast(type) : p.second);
  }
  if (else_e != nullptr) {
    else_e = executor ? else_e->add_cast(type) : else_e;
  } else {
    Datum d;
    // always create an else expr so that executor doesn't need to worry about it
    else_e = hdk::ir::makeExpr<hdk::ir::Constant>(type, true, d);
  }

  auto case_expr = hdk::ir::makeExpr<hdk::ir::CaseExpr>(
      type, has_agg, std::move(cast_expr_pair_list), else_e);
  return case_expr;
}

hdk::ir::ExprPtr getLikeExpr(hdk::ir::ExprPtr arg_expr,
                             hdk::ir::ExprPtr like_expr,
                             hdk::ir::ExprPtr escape_expr,
                             const bool is_ilike,
                             const bool is_not) {
  if (!arg_expr->type()->isString() && !arg_expr->type()->isExtDictionary()) {
    throw std::runtime_error("expression before LIKE must be of a string type.");
  }
  if (!like_expr->type()->isString() && !like_expr->type()->isExtDictionary()) {
    throw std::runtime_error("expression after LIKE must be of a string type.");
  }
  char escape_char = '\\';
  if (escape_expr != nullptr) {
    if (!escape_expr->type()->isString()) {
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    }
    auto c = std::dynamic_pointer_cast<hdk::ir::Constant>(escape_expr);
    if (c != nullptr && c->get_constval().stringval->length() > 1) {
      throw std::runtime_error("String after ESCAPE must have a single character.");
    }
    escape_char = (*c->get_constval().stringval)[0];
  }
  auto c = std::dynamic_pointer_cast<hdk::ir::Constant>(like_expr);
  bool is_simple = false;
  if (c != nullptr) {
    std::string& pattern = *c->get_constval().stringval;
    if (is_ilike) {
      std::transform(pattern.begin(), pattern.end(), pattern.begin(), ::tolower);
    }
    check_like_expr(pattern, escape_char);
    is_simple = test_is_simple_expr(pattern, escape_char);
    if (is_simple) {
      erase_cntl_chars(pattern, escape_char);
    }
  }
  hdk::ir::ExprPtr result = hdk::ir::makeExpr<hdk::ir::LikeExpr>(
      arg_expr->decompress(), like_expr, escape_expr, is_ilike, is_simple);
  if (is_not) {
    result = hdk::ir::makeExpr<hdk::ir::UOper>(kBOOLEAN, kNOT, result);
  }
  return result;
}

hdk::ir::ExprPtr getRegexpExpr(hdk::ir::ExprPtr arg_expr,
                               hdk::ir::ExprPtr pattern_expr,
                               hdk::ir::ExprPtr escape_expr,
                               const bool is_not) {
  if (!arg_expr->type()->isString() && !arg_expr->type()->isExtDictionary()) {
    throw std::runtime_error("expression before REGEXP must be of a string type.");
  }
  if (!pattern_expr->type()->isString() && !pattern_expr->type()->isExtDictionary()) {
    throw std::runtime_error("expression after REGEXP must be of a string type.");
  }
  char escape_char = '\\';
  if (escape_expr != nullptr) {
    if (!escape_expr->type()->isString()) {
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    }
    if (!escape_expr->type()->isString()) {
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    }
    auto c = std::dynamic_pointer_cast<hdk::ir::Constant>(escape_expr);
    if (c != nullptr && c->get_constval().stringval->length() > 1) {
      throw std::runtime_error("String after ESCAPE must have a single character.");
    }
    escape_char = (*c->get_constval().stringval)[0];
    if (escape_char != '\\') {
      throw std::runtime_error("Only supporting '\\' escape character.");
    }
  }
  auto c = std::dynamic_pointer_cast<hdk::ir::Constant>(pattern_expr);
  if (c != nullptr) {
    std::string& pattern = *c->get_constval().stringval;
    if (translate_to_like_pattern(pattern, escape_char)) {
      return getLikeExpr(arg_expr, pattern_expr, escape_expr, false, is_not);
    }
  }
  hdk::ir::ExprPtr result = hdk::ir::makeExpr<hdk::ir::RegexpExpr>(
      arg_expr->decompress(), pattern_expr, escape_expr);
  if (is_not) {
    result = hdk::ir::makeExpr<hdk::ir::UOper>(kBOOLEAN, kNOT, result);
  }
  return result;
}

hdk::ir::ExprPtr getUserLiteral(const std::string& user) {
  Datum d;
  d.stringval = new std::string(user);
  return hdk::ir::makeExpr<hdk::ir::Constant>(kTEXT, false, d, false);
}

hdk::ir::ExprPtr getTimestampLiteral(const int64_t timestampval) {
  Datum d;
  d.bigintval = timestampval;
  return hdk::ir::makeExpr<hdk::ir::Constant>(kTIMESTAMP, false, d, false);
}

}  // namespace Analyzer

hdk::ir::ExprPtr remove_cast(const hdk::ir::ExprPtr& expr) {
  const auto uoper = dynamic_cast<const hdk::ir::UOper*>(expr.get());
  if (!uoper || uoper->get_optype() != kCAST) {
    return expr;
  }
  return uoper->get_own_operand();
}
