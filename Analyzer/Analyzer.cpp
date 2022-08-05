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
  if (expr->get_type_info().get_type() == kNULLT) {
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

}  // namespace

namespace Analyzer {

SQLTypeInfo analyze_type_info(SQLOps op,
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

SQLTypeInfo common_string_type(const SQLTypeInfo& type1, const SQLTypeInfo& type2) {
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

SQLTypeInfo common_numeric_type(const SQLTypeInfo& type1, const SQLTypeInfo& type2) {
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
  SQLTypeInfo ti(kNUMERIC, 0, 0, false);
  ti.set_scale(scale);
  ti.set_precision(precision);
  Datum d;
  d.bigintval = numericval;
  return hdk::ir::makeExpr<hdk::ir::Constant>(ti, false, d);
}

hdk::ir::ExprPtr analyzeStringValue(const std::string& stringval) {
  SQLTypeInfo ti(kVARCHAR, stringval.length(), 0, true);
  Datum d;
  d.stringval = new std::string(stringval);
  return hdk::ir::makeExpr<hdk::ir::Constant>(ti, false, d);
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

SQLTypeInfo const& get_str_dict_cast_type(const SQLTypeInfo& lhs_type_info,
                                          const SQLTypeInfo& rhs_type_info,
                                          const Executor* executor) {
  CHECK(lhs_type_info.is_string());
  CHECK(lhs_type_info.get_compression() == kENCODING_DICT);
  CHECK(rhs_type_info.is_string());
  CHECK(rhs_type_info.get_compression() == kENCODING_DICT);
  const auto lhs_comp_param = lhs_type_info.get_comp_param();
  const auto rhs_comp_param = rhs_type_info.get_comp_param();
  CHECK_NE(lhs_comp_param, rhs_comp_param);
  if (lhs_type_info.get_comp_param() == TRANSIENT_DICT_ID) {
    return rhs_type_info;
  }
  if (rhs_type_info.get_comp_param() == TRANSIENT_DICT_ID) {
    return lhs_type_info;
  }
  // When translator is used from DAG builder, executor is not available.
  // In this case, simply choose LHS and revise the decision later.
  if (!executor) {
    return lhs_type_info;
  }
  // If here then neither lhs or rhs type was transient, we should see which
  // type has the largest dictionary and make that the destination type
  const auto lhs_sdp = executor->getStringDictionaryProxy(lhs_comp_param, true);
  const auto rhs_sdp = executor->getStringDictionaryProxy(rhs_comp_param, true);
  return lhs_sdp->entryCount() >= rhs_sdp->entryCount() ? lhs_type_info : rhs_type_info;
}

SQLTypeInfo common_string_type(const SQLTypeInfo& lhs_type_info,
                               const SQLTypeInfo& rhs_type_info,
                               const Executor* executor) {
  CHECK(lhs_type_info.is_string());
  CHECK(rhs_type_info.is_string());
  const auto lhs_comp_param = lhs_type_info.get_comp_param();
  const auto rhs_comp_param = rhs_type_info.get_comp_param();
  if (lhs_type_info.is_dict_encoded_string() && rhs_type_info.is_dict_encoded_string()) {
    if (lhs_comp_param == rhs_comp_param ||
        lhs_comp_param == TRANSIENT_DICT(rhs_comp_param)) {
      return lhs_comp_param <= rhs_comp_param ? lhs_type_info : rhs_type_info;
    }
    return get_str_dict_cast_type(lhs_type_info, rhs_type_info, executor);
  }
  CHECK(lhs_type_info.is_none_encoded_string() || rhs_type_info.is_none_encoded_string());
  SQLTypeInfo ret_ti =
      lhs_type_info.is_none_encoded_string() ? lhs_type_info : rhs_type_info;
  ret_ti.set_dimension(
      std::max(lhs_type_info.get_dimension(), rhs_type_info.get_dimension()));
  return ret_ti;
}

hdk::ir::ExprPtr normalizeOperExpr(const SQLOps optype,
                                   const SQLQualifier qual,
                                   hdk::ir::ExprPtr left_expr,
                                   hdk::ir::ExprPtr right_expr,
                                   const Executor* executor) {
  if (left_expr->get_type_info().is_date_in_days() ||
      right_expr->get_type_info().is_date_in_days()) {
    // Do not propogate encoding
    left_expr = left_expr->decompress();
    right_expr = right_expr->decompress();
  }
  const auto& left_type = left_expr->get_type_info();
  auto right_type = right_expr->get_type_info();
  if (qual != kONE) {
    // subquery not supported yet.
    CHECK(!std::dynamic_pointer_cast<hdk::ir::ScalarSubquery>(right_expr));
    if (right_type.get_type() != kARRAY) {
      throw std::runtime_error(
          "Existential or universal qualifiers can only be used in front of a subquery "
          "or an "
          "expression of array type.");
    }
    right_type = right_type.get_elem_type();
  }
  SQLTypeInfo new_left_type;
  SQLTypeInfo new_right_type;
  auto result_type =
      analyze_type_info(optype, left_type, right_type, &new_left_type, &new_right_type);
  if (result_type.is_timeinterval()) {
    return hdk::ir::makeExpr<hdk::ir::BinOper>(
        result_type, false, optype, qual, left_expr, right_expr);
  }
  // No executor means we build an expression for further normalization.
  // Required casts will be added at normalization. Double normalization
  // with casts may cause a change in the resulting type. E.g. it would
  // increase dimension for decimals on each normalization for arithmetic
  // operations.
  if (executor) {
    if (left_type != new_left_type) {
      left_expr = left_expr->add_cast(new_left_type);
    }
    if (right_type != new_right_type) {
      if (qual == kONE) {
        right_expr = right_expr->add_cast(new_right_type);
      } else {
        right_expr = right_expr->add_cast(new_right_type.get_array_type());
      }
    }
  }

  // No executor means we are building DAG. Skip normalization at this
  // step and perform it later on execution unit build.
  if (IS_COMPARISON(optype) && executor) {
    if (new_left_type.get_compression() == kENCODING_DICT &&
        new_right_type.get_compression() == kENCODING_DICT) {
      if (new_left_type.get_comp_param() != new_right_type.get_comp_param()) {
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
          SQLTypeInfo ti(get_str_dict_cast_type(new_left_type, new_right_type, executor));
          auto& expr_to_cast = ti == new_left_type ? right_expr : left_expr;
          ti.set_fixed_size();
          ti.set_dict_intersection();
          expr_to_cast = expr_to_cast->add_cast(ti);
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
    } else if (new_left_type.get_compression() == kENCODING_DICT &&
               new_right_type.get_compression() == kENCODING_NONE) {
      SQLTypeInfo ti(new_right_type);
      ti.set_compression(new_left_type.get_compression());
      ti.set_comp_param(new_left_type.get_comp_param());
      ti.set_fixed_size();
      right_expr = right_expr->add_cast(ti);
    } else if (new_right_type.get_compression() == kENCODING_DICT &&
               new_left_type.get_compression() == kENCODING_NONE) {
      SQLTypeInfo ti(new_left_type);
      ti.set_compression(new_right_type.get_compression());
      ti.set_comp_param(new_right_type.get_comp_param());
      ti.set_fixed_size();
      left_expr = left_expr->add_cast(ti);
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
  SQLTypeInfo ti;
  bool has_agg = false;
  // We need to keep track of whether there was at
  // least one none-encoded string literal expression
  // type among any of the case sub-expressions separately
  // from rest of type determination logic, as it will
  // be casted to the output dictionary type if all output
  // types are either dictionary encoded or none-encoded
  // literals, or kept as none-encoded if all sub-expression
  // types are none-encoded (column or literal)
  SQLTypeInfo none_encoded_literal_ti;

  for (auto& p : expr_pair_list) {
    auto e1 = p.first;
    CHECK(e1->get_type_info().is_boolean());
    auto e2 = p.second;
    if (e2->get_contains_agg()) {
      has_agg = true;
    }
    const auto& e2_ti = e2->get_type_info();
    if (e2_ti.is_string() && !e2_ti.is_dict_encoded_string() &&
        !std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(e2)) {
      CHECK(e2_ti.is_none_encoded_string());
      none_encoded_literal_ti =
          none_encoded_literal_ti.get_type() == kNULLT
              ? e2_ti
              : common_string_type(none_encoded_literal_ti, e2_ti, executor);
      continue;
    }

    if (ti.get_type() == kNULLT) {
      ti = e2_ti;
    } else if (e2_ti.get_type() == kNULLT) {
      ti.set_notnull(false);
      e2->set_type_info(ti);
    } else if (ti != e2_ti) {
      if (ti.is_string() && e2_ti.is_string()) {
        // Executor is needed to determine which dictionary is the largest
        // in case of two dictionary types with different encodings
        ti = common_string_type(ti, e2_ti, executor);
      } else if (ti.is_number() && e2_ti.is_number()) {
        ti = common_numeric_type(ti, e2_ti);
      } else if (ti.is_boolean() && e2_ti.is_boolean()) {
        ti = common_numeric_type(ti, e2_ti);
      } else {
        throw std::runtime_error(
            "expressions in THEN clause must be of the same or compatible types.");
      }
    }
  }
  auto else_e = else_e_in;
  const auto& else_ti = else_e->get_type_info();
  if (else_e) {
    if (else_e->get_contains_agg()) {
      has_agg = true;
    }
    if (else_ti.is_string() && !else_ti.is_dict_encoded_string() &&
        !std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(else_e)) {
      CHECK(else_ti.is_none_encoded_string());
      none_encoded_literal_ti =
          none_encoded_literal_ti.get_type() == kNULLT
              ? else_ti
              : common_string_type(none_encoded_literal_ti, else_ti, executor);
    } else {
      if (ti.get_type() == kNULLT) {
        ti = else_ti;
      } else if (expr_is_null(else_e.get())) {
        ti.set_notnull(false);
        else_e->set_type_info(ti);
      } else if (ti != else_ti) {
        ti.set_notnull(false);
        if (ti.is_string() && else_ti.is_string()) {
          // Executor is needed to determine which dictionary is the largest
          // in case of two dictionary types with different encodings
          ti = common_string_type(ti, else_ti, executor);
        } else if (ti.is_number() && else_ti.is_number()) {
          ti = common_numeric_type(ti, else_ti);
        } else if (ti.is_boolean() && else_ti.is_boolean()) {
          ti = common_numeric_type(ti, else_ti);
        } else if (get_logical_type_info(ti) != get_logical_type_info(else_ti)) {
          throw std::runtime_error(
              // types differing by encoding will be resolved at decode
              "Expressions in ELSE clause must be of the same or compatible types as "
              "those in the THEN clauses.");
        }
      }
    }
  } else {
    ti.set_notnull(false);
  }

  if (ti.get_type() == kNULLT && none_encoded_literal_ti.get_type() != kNULLT) {
    // If we haven't set a type so far it's because
    // every case sub-expression has a none-encoded
    // literal output. Make this our output type
    ti = none_encoded_literal_ti;
  }

  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> cast_expr_pair_list;
  for (auto p : expr_pair_list) {
    cast_expr_pair_list.emplace_back(p.first,
                                     executor ? p.second->add_cast(ti) : p.second);
  }
  if (else_e != nullptr) {
    else_e = executor ? else_e->add_cast(ti) : else_e;
  } else {
    Datum d;
    // always create an else expr so that executor doesn't need to worry about it
    else_e = hdk::ir::makeExpr<hdk::ir::Constant>(ti, true, d);
  }
  if (ti.get_type() == kNULLT) {
    throw std::runtime_error(
        "Cannot deduce the type for case expressions, all branches null");
  }

  auto case_expr =
      hdk::ir::makeExpr<hdk::ir::CaseExpr>(ti, has_agg, cast_expr_pair_list, else_e);
  return case_expr;
}

hdk::ir::ExprPtr getLikeExpr(hdk::ir::ExprPtr arg_expr,
                             hdk::ir::ExprPtr like_expr,
                             hdk::ir::ExprPtr escape_expr,
                             const bool is_ilike,
                             const bool is_not) {
  if (!arg_expr->get_type_info().is_string()) {
    throw std::runtime_error("expression before LIKE must be of a string type.");
  }
  if (!like_expr->get_type_info().is_string()) {
    throw std::runtime_error("expression after LIKE must be of a string type.");
  }
  char escape_char = '\\';
  if (escape_expr != nullptr) {
    if (!escape_expr->get_type_info().is_string()) {
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    }
    if (!escape_expr->get_type_info().is_string()) {
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
  if (!arg_expr->get_type_info().is_string()) {
    throw std::runtime_error("expression before REGEXP must be of a string type.");
  }
  if (!pattern_expr->get_type_info().is_string()) {
    throw std::runtime_error("expression after REGEXP must be of a string type.");
  }
  char escape_char = '\\';
  if (escape_expr != nullptr) {
    if (!escape_expr->get_type_info().is_string()) {
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    }
    if (!escape_expr->get_type_info().is_string()) {
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
  return hdk::ir::makeExpr<hdk::ir::Constant>(kTEXT, false, d);
}

hdk::ir::ExprPtr getTimestampLiteral(const int64_t timestampval) {
  Datum d;
  d.bigintval = timestampval;
  return hdk::ir::makeExpr<hdk::ir::Constant>(kTIMESTAMP, false, d);
}

}  // namespace Analyzer

hdk::ir::ExprPtr remove_cast(const hdk::ir::ExprPtr& expr) {
  const auto uoper = dynamic_cast<const hdk::ir::UOper*>(expr.get());
  if (!uoper || uoper->get_optype() != kCAST) {
    return expr;
  }
  return uoper->get_own_operand();
}
