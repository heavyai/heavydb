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
 * @file    ParserNode.cpp
 * @author  Wei Hong <wei@map-d.com>
 * @brief   Functions for ParserNode classes
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cassert>
#include <stdexcept>
#include <typeinfo>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include "../Catalog/Catalog.h"
#include "../Catalog/SharedDictionaryValidator.h"
#include "ParserNode.h"
#include "ReservedKeywords.h"
#include "../Planner/Planner.h"
#include "../QueryEngine/Execute.h"
#include "../Fragmenter/InsertOrderFragmenter.h"
#include "../Import/Importer.h"
#include "../Shared/measure.h"
#include "../Shared/mapd_glob.h"
#include "parser.h"

#include "../QueryEngine/ExtensionFunctionsWhitelist.h"
#include "../QueryEngine/RelAlgExecutor.h"
#include "../QueryEngine/CalciteAdapter.h"

size_t g_leaf_count{0};

namespace Parser {

std::shared_ptr<Analyzer::Expr> NullLiteral::analyze(const Catalog_Namespace::Catalog& catalog,
                                                     Analyzer::Query& query,
                                                     TlistRefType allow_tlist_ref) const {
  return makeExpr<Analyzer::Constant>(kNULLT, true);
}

std::shared_ptr<Analyzer::Expr> StringLiteral::analyze(const Catalog_Namespace::Catalog& catalog,
                                                       Analyzer::Query& query,
                                                       TlistRefType allow_tlist_ref) const {
  return analyzeValue(*stringval);
}

std::shared_ptr<Analyzer::Expr> StringLiteral::analyzeValue(const std::string& stringval) {
  SQLTypeInfo ti(kVARCHAR, stringval.length(), 0, true);
  Datum d;
  d.stringval = new std::string(stringval);
  return makeExpr<Analyzer::Constant>(ti, false, d);
}

std::shared_ptr<Analyzer::Expr> IntLiteral::analyze(const Catalog_Namespace::Catalog& catalog,
                                                    Analyzer::Query& query,
                                                    TlistRefType allow_tlist_ref) const {
  return analyzeValue(intval);
}

std::shared_ptr<Analyzer::Expr> IntLiteral::analyzeValue(const int64_t intval) {
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
  return makeExpr<Analyzer::Constant>(t, false, d);
}

std::shared_ptr<Analyzer::Expr> FixedPtLiteral::analyze(const Catalog_Namespace::Catalog& catalog,
                                                        Analyzer::Query& query,
                                                        TlistRefType allow_tlist_ref) const {
  SQLTypeInfo ti(kNUMERIC, 0, 0, false);
  Datum d = StringToDatum(*fixedptval, ti);
  return makeExpr<Analyzer::Constant>(ti, false, d);
}

std::shared_ptr<Analyzer::Expr> FixedPtLiteral::analyzeValue(const int64_t numericval,
                                                             const int scale,
                                                             const int precision) {
  SQLTypeInfo ti(kNUMERIC, 0, 0, false);
  ti.set_scale(scale);
  ti.set_precision(precision);
  Datum d;
  d.bigintval = numericval;
  return makeExpr<Analyzer::Constant>(ti, false, d);
}

std::shared_ptr<Analyzer::Expr> FloatLiteral::analyze(const Catalog_Namespace::Catalog& catalog,
                                                      Analyzer::Query& query,
                                                      TlistRefType allow_tlist_ref) const {
  Datum d;
  d.floatval = floatval;
  return makeExpr<Analyzer::Constant>(kFLOAT, false, d);
}

std::shared_ptr<Analyzer::Expr> DoubleLiteral::analyze(const Catalog_Namespace::Catalog& catalog,
                                                       Analyzer::Query& query,
                                                       TlistRefType allow_tlist_ref) const {
  Datum d;
  d.doubleval = doubleval;
  return makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
}

std::shared_ptr<Analyzer::Expr> TimestampLiteral::analyze(const Catalog_Namespace::Catalog& catalog,
                                                          Analyzer::Query& query,
                                                          TlistRefType allow_tlist_ref) const {
  return get(timestampval);
}

std::shared_ptr<Analyzer::Expr> TimestampLiteral::get(const time_t timestampval) {
  Datum d;
  d.timeval = timestampval;
  return makeExpr<Analyzer::Constant>(kTIMESTAMP, false, d);
}

std::shared_ptr<Analyzer::Expr> UserLiteral::analyze(const Catalog_Namespace::Catalog& catalog,
                                                     Analyzer::Query& query,
                                                     TlistRefType allow_tlist_ref) const {
  throw std::runtime_error("USER literal not supported yet.");
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> OperExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref) const {
  auto left_expr = left->analyze(catalog, query, allow_tlist_ref);
  const auto& left_type = left_expr->get_type_info();
  if (right == nullptr) {
    return makeExpr<Analyzer::UOper>(left_type, left_expr->get_contains_agg(), optype, left_expr->decompress());
  }
  if (optype == kARRAY_AT) {
    if (left_type.get_type() != kARRAY)
      throw std::runtime_error(left->to_string() + " is not of array type.");
    auto right_expr = right->analyze(catalog, query, allow_tlist_ref);
    const auto& right_type = right_expr->get_type_info();
    if (!right_type.is_integer())
      throw std::runtime_error(right->to_string() + " is not of integer type.");
    return makeExpr<Analyzer::BinOper>(left_type.get_elem_type(), false, kARRAY_AT, kONE, left_expr, right_expr);
  }
  auto right_expr = right->analyze(catalog, query, allow_tlist_ref);
  return normalize(optype, opqualifier, left_expr, right_expr);
}

std::shared_ptr<Analyzer::Expr> OperExpr::normalize(const SQLOps optype,
                                                    const SQLQualifier qual,
                                                    std::shared_ptr<Analyzer::Expr> left_expr,
                                                    std::shared_ptr<Analyzer::Expr> right_expr) {
  const auto& left_type = left_expr->get_type_info();
  auto right_type = right_expr->get_type_info();
  if (qual != kONE) {
    // subquery not supported yet.
    CHECK(!std::dynamic_pointer_cast<Analyzer::Subquery>(right_expr));
    if (right_type.get_type() != kARRAY)
      throw std::runtime_error(
          "Existential or universal qualifiers can only be used in front of a subquery or an "
          "expression of array type.");
    right_type = right_type.get_elem_type();
  }
  SQLTypeInfo new_left_type;
  SQLTypeInfo new_right_type;
  auto result_type =
      Analyzer::BinOper::analyze_type_info(optype, left_type, right_type, &new_left_type, &new_right_type);
  if (result_type.is_timeinterval()) {
    return makeExpr<Analyzer::BinOper>(result_type, false, optype, qual, left_expr, right_expr);
  }
  if (left_type != new_left_type)
    left_expr = left_expr->add_cast(new_left_type);
  if (right_type != new_right_type) {
    if (qual == kONE)
      right_expr = right_expr->add_cast(new_right_type);
    else
      right_expr = right_expr->add_cast(new_right_type.get_array_type());
  }
  if (IS_EQUIVALENCE(optype) || optype == kNE) {
    if (new_left_type.get_compression() == kENCODING_DICT && new_right_type.get_compression() == kENCODING_DICT) {
      // do nothing
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
  } else {
    left_expr = left_expr->decompress();
    right_expr = right_expr->decompress();
  }
  bool has_agg = (left_expr->get_contains_agg() || right_expr->get_contains_agg());
  if (left_type.is_decimal() && right_type.is_decimal() && optype == kMULTIPLY) {
    result_type.set_scale(left_type.get_scale() + right_type.get_scale());
  }
  return makeExpr<Analyzer::BinOper>(result_type, has_agg, optype, qual, left_expr, right_expr);
}

std::shared_ptr<Analyzer::Expr> SubqueryExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                      Analyzer::Query& query,
                                                      TlistRefType allow_tlist_ref) const {
  throw std::runtime_error("Subqueries are not supported yet.");
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> IsNullExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                    Analyzer::Query& query,
                                                    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  auto result = makeExpr<Analyzer::UOper>(kBOOLEAN, kISNULL, arg_expr);
  if (is_not)
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  return result;
}

std::shared_ptr<Analyzer::Expr> InSubquery::analyze(const Catalog_Namespace::Catalog& catalog,
                                                    Analyzer::Query& query,
                                                    TlistRefType allow_tlist_ref) const {
  throw std::runtime_error("Subqueries are not supported yet.");
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> InValues::analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  SQLTypeInfo ti = arg_expr->get_type_info();
  bool dict_comp = ti.get_compression() == kENCODING_DICT;
  std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
  for (auto& p : value_list) {
    auto e = p->analyze(catalog, query, allow_tlist_ref);
    if (ti != e->get_type_info()) {
      if (ti.is_string() && e->get_type_info().is_string())
        ti = Analyzer::BinOper::common_string_type(ti, e->get_type_info());
      else if (ti.is_number() && e->get_type_info().is_number())
        ti = Analyzer::BinOper::common_numeric_type(ti, e->get_type_info());
      else
        throw std::runtime_error("IN expressions must contain compatible types.");
    }
    if (dict_comp)
      value_exprs.push_back(e->add_cast(arg_expr->get_type_info()));
    else
      value_exprs.push_back(e);
  }
  if (!dict_comp) {
    arg_expr = arg_expr->decompress();
    arg_expr = arg_expr->add_cast(ti);
    std::list<std::shared_ptr<Analyzer::Expr>> cast_vals;
    for (auto p : value_exprs) {
      cast_vals.push_back(p->add_cast(ti));
    }
    value_exprs.swap(cast_vals);
  }
  std::shared_ptr<Analyzer::Expr> result = makeExpr<Analyzer::InValues>(arg_expr, value_exprs);
  if (is_not)
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  return result;
}

std::shared_ptr<Analyzer::Expr> BetweenExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                     Analyzer::Query& query,
                                                     TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  auto lower_expr = lower->analyze(catalog, query, allow_tlist_ref);
  auto upper_expr = upper->analyze(catalog, query, allow_tlist_ref);
  SQLTypeInfo new_left_type, new_right_type;
  (void)Analyzer::BinOper::analyze_type_info(
      kGE, arg_expr->get_type_info(), lower_expr->get_type_info(), &new_left_type, &new_right_type);
  auto lower_pred = makeExpr<Analyzer::BinOper>(kBOOLEAN,
                                                kGE,
                                                kONE,
                                                arg_expr->add_cast(new_left_type)->decompress(),
                                                lower_expr->add_cast(new_right_type)->decompress());
  (void)Analyzer::BinOper::analyze_type_info(
      kLE, arg_expr->get_type_info(), lower_expr->get_type_info(), &new_left_type, &new_right_type);
  auto upper_pred = makeExpr<Analyzer::BinOper>(kBOOLEAN,
                                                kLE,
                                                kONE,
                                                arg_expr->deep_copy()->add_cast(new_left_type)->decompress(),
                                                upper_expr->add_cast(new_right_type)->decompress());
  std::shared_ptr<Analyzer::Expr> result = makeExpr<Analyzer::BinOper>(kBOOLEAN, kAND, kONE, lower_pred, upper_pred);
  if (is_not)
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  return result;
}

std::shared_ptr<Analyzer::Expr> CharLengthExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                        Analyzer::Query& query,
                                                        TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  if (!arg_expr->get_type_info().is_string())
    throw std::runtime_error("expression in char_length clause must be of a string type.");
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::CharLengthExpr>(arg_expr->decompress(), calc_encoded_length);
  return result;
}

void LikeExpr::check_like_expr(const std::string& like_str, char escape_char) {
  if (like_str.back() == escape_char)
    throw std::runtime_error("LIKE pattern must not end with escape character.");
}

bool LikeExpr::test_is_simple_expr(const std::string& like_str, char escape_char) {
  // if not bounded by '%' then not a simple string
  if (like_str.size() < 2 || like_str[0] != '%' || like_str[like_str.size() - 1] != '%')
    return false;
  // if the last '%' is escaped then not a simple string
  if (like_str[like_str.size() - 2] == escape_char && like_str[like_str.size() - 3] != escape_char)
    return false;
  for (size_t i = 1; i < like_str.size() - 1; i++) {
    if (like_str[i] == '%' || like_str[i] == '_' || like_str[i] == '[' || like_str[i] == ']') {
      if (like_str[i - 1] != escape_char)
        return false;
    }
  }
  return true;
}

void LikeExpr::erase_cntl_chars(std::string& like_str, char escape_char) {
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

std::shared_ptr<Analyzer::Expr> LikeExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  auto like_expr = like_string->analyze(catalog, query, allow_tlist_ref);
  auto escape_expr = escape_string == nullptr ? nullptr : escape_string->analyze(catalog, query, allow_tlist_ref);
  return LikeExpr::get(arg_expr, like_expr, escape_expr, is_ilike, is_not);
}

std::shared_ptr<Analyzer::Expr> LikeExpr::get(std::shared_ptr<Analyzer::Expr> arg_expr,
                                              std::shared_ptr<Analyzer::Expr> like_expr,
                                              std::shared_ptr<Analyzer::Expr> escape_expr,
                                              const bool is_ilike,
                                              const bool is_not) {
  if (!arg_expr->get_type_info().is_string())
    throw std::runtime_error("expression before LIKE must be of a string type.");
  if (!like_expr->get_type_info().is_string())
    throw std::runtime_error("expression after LIKE must be of a string type.");
  char escape_char = '\\';
  if (escape_expr != nullptr) {
    if (!escape_expr->get_type_info().is_string())
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    if (!escape_expr->get_type_info().is_string())
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    auto c = std::dynamic_pointer_cast<Analyzer::Constant>(escape_expr);
    if (c != nullptr && c->get_constval().stringval->length() > 1)
      throw std::runtime_error("String after ESCAPE must have a single character.");
    escape_char = (*c->get_constval().stringval)[0];
  }
  auto c = std::dynamic_pointer_cast<Analyzer::Constant>(like_expr);
  bool is_simple = false;
  if (c != nullptr) {
    std::string& pattern = *c->get_constval().stringval;
    if (is_ilike)
      std::transform(pattern.begin(), pattern.end(), pattern.begin(), ::tolower);
    check_like_expr(pattern, escape_char);
    is_simple = test_is_simple_expr(pattern, escape_char);
    if (is_simple) {
      erase_cntl_chars(pattern, escape_char);
    }
  }
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::LikeExpr>(arg_expr->decompress(), like_expr, escape_expr, is_ilike, is_simple);
  if (is_not)
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  return result;
}

void RegexpExpr::check_pattern_expr(const std::string& pattern_str, char escape_char) {
  if (pattern_str.back() == escape_char)
    throw std::runtime_error("REGEXP pattern must not end with escape character.");
}

bool RegexpExpr::translate_to_like_pattern(std::string& pattern_str, char escape_char) {
  char prev_char = '\0';
  char prev_prev_char = '\0';
  std::string like_str;
  for (char& cur_char : pattern_str) {
    if (prev_char == escape_char || isalnum(cur_char) || cur_char == ' ' || cur_char == '.') {
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

std::shared_ptr<Analyzer::Expr> RegexpExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                    Analyzer::Query& query,
                                                    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  auto pattern_expr = pattern_string->analyze(catalog, query, allow_tlist_ref);
  auto escape_expr = escape_string == nullptr ? nullptr : escape_string->analyze(catalog, query, allow_tlist_ref);
  return RegexpExpr::get(arg_expr, pattern_expr, escape_expr, is_not);
}

std::shared_ptr<Analyzer::Expr> RegexpExpr::get(std::shared_ptr<Analyzer::Expr> arg_expr,
                                                std::shared_ptr<Analyzer::Expr> pattern_expr,
                                                std::shared_ptr<Analyzer::Expr> escape_expr,
                                                const bool is_not) {
  if (!arg_expr->get_type_info().is_string())
    throw std::runtime_error("expression before REGEXP must be of a string type.");
  if (!pattern_expr->get_type_info().is_string())
    throw std::runtime_error("expression after REGEXP must be of a string type.");
  char escape_char = '\\';
  if (escape_expr != nullptr) {
    if (!escape_expr->get_type_info().is_string())
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    if (!escape_expr->get_type_info().is_string())
      throw std::runtime_error("expression after ESCAPE must be of a string type.");
    auto c = std::dynamic_pointer_cast<Analyzer::Constant>(escape_expr);
    if (c != nullptr && c->get_constval().stringval->length() > 1)
      throw std::runtime_error("String after ESCAPE must have a single character.");
    escape_char = (*c->get_constval().stringval)[0];
    if (escape_char != '\\')
      throw std::runtime_error("Only supporting '\\' escape character.");
  }
  auto c = std::dynamic_pointer_cast<Analyzer::Constant>(pattern_expr);
  if (c != nullptr) {
    std::string& pattern = *c->get_constval().stringval;
    if (translate_to_like_pattern(pattern, escape_char)) {
      return LikeExpr::get(arg_expr, pattern_expr, escape_expr, false, is_not);
    }
  }
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::RegexpExpr>(arg_expr->decompress(), pattern_expr, escape_expr);
  if (is_not)
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  return result;
}

std::shared_ptr<Analyzer::Expr> LikelihoodExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                        Analyzer::Query& query,
                                                        TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  return LikelihoodExpr::get(arg_expr, likelihood, is_not);
}

std::shared_ptr<Analyzer::Expr> LikelihoodExpr::get(std::shared_ptr<Analyzer::Expr> arg_expr,
                                                    float likelihood,
                                                    const bool is_not) {
  if (!arg_expr->get_type_info().is_boolean())
    throw std::runtime_error("likelihood expression expects boolean type.");
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::LikelihoodExpr>(arg_expr->decompress(), is_not ? 1 - likelihood : likelihood);
  return result;
}

std::shared_ptr<Analyzer::Expr> ExistsExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                    Analyzer::Query& query,
                                                    TlistRefType allow_tlist_ref) const {
  throw std::runtime_error("Subqueries are not supported yet.");
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> ColumnRef::analyze(const Catalog_Namespace::Catalog& catalog,
                                                   Analyzer::Query& query,
                                                   TlistRefType allow_tlist_ref) const {
  int table_id{0};
  int rte_idx{0};
  const ColumnDescriptor* cd{nullptr};
  if (column == nullptr)
    throw std::runtime_error("invalid column name *.");
  if (table != nullptr) {
    rte_idx = query.get_rte_idx(*table);
    if (rte_idx < 0)
      throw std::runtime_error("range variable or table name " + *table + " does not exist.");
    Analyzer::RangeTblEntry* rte = query.get_rte(rte_idx);
    cd = rte->get_column_desc(catalog, *column);
    if (cd == nullptr)
      throw std::runtime_error("Column name " + *column + " does not exist.");
    table_id = rte->get_table_id();
  } else {
    bool found = false;
    int i = 0;
    for (auto rte : query.get_rangetable()) {
      cd = rte->get_column_desc(catalog, *column);
      if (cd != nullptr && !found) {
        found = true;
        rte_idx = i;
        table_id = rte->get_table_id();
      } else if (cd != nullptr && found)
        throw std::runtime_error("Column name " + *column + " is ambiguous.");
      i++;
    }
    if (cd == nullptr && allow_tlist_ref != TlistRefType::TLIST_NONE) {
      // check if this is a reference to a targetlist entry
      bool found = false;
      int varno = -1;
      int i = 1;
      std::shared_ptr<Analyzer::TargetEntry> tle;
      for (auto p : query.get_targetlist()) {
        if (*column == p->get_resname() && !found) {
          found = true;
          varno = i;
          tle = p;
        } else if (*column == p->get_resname() && found)
          throw std::runtime_error("Output alias " + *column + " is ambiguous.");
        i++;
      }
      if (found) {
        if (dynamic_cast<Analyzer::Var*>(tle->get_expr())) {
          Analyzer::Var* v = static_cast<Analyzer::Var*>(tle->get_expr());
          if (v->get_which_row() == Analyzer::Var::kGROUPBY)
            return v->deep_copy();
        }
        if (allow_tlist_ref == TlistRefType::TLIST_COPY)
          return tle->get_expr()->deep_copy();
        else
          return makeExpr<Analyzer::Var>(tle->get_expr()->get_type_info(), Analyzer::Var::kOUTPUT, varno);
      }
    }
    if (cd == nullptr)
      throw std::runtime_error("Column name " + *column + " does not exist.");
  }
  return makeExpr<Analyzer::ColumnVar>(cd->columnType, table_id, cd->columnId, rte_idx);
}

std::shared_ptr<Analyzer::Expr> FunctionRef::analyze(const Catalog_Namespace::Catalog& catalog,
                                                     Analyzer::Query& query,
                                                     TlistRefType allow_tlist_ref) const {
  SQLTypeInfo result_type;
  SQLAgg agg_type;
  std::shared_ptr<Analyzer::Expr> arg_expr;
  bool is_distinct = false;
  if (boost::iequals(*name, "count")) {
    result_type = SQLTypeInfo(kBIGINT, false);
    agg_type = kCOUNT;
    if (arg) {
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      const SQLTypeInfo& ti = arg_expr->get_type_info();
      if (ti.is_string() && (ti.get_compression() != kENCODING_DICT || !distinct))
        throw std::runtime_error("Strings must be dictionary-encoded in COUNT(DISTINCT).");
      if (ti.get_type() == kARRAY && !distinct)
        throw std::runtime_error("Only COUNT(DISTINCT) is supported on arrays.");
    }
    is_distinct = distinct;
  } else {
    if (!arg) {
      throw std::runtime_error("Cannot compute " + *name + " with argument '*'.");
    }
    if (boost::iequals(*name, "min")) {
      agg_type = kMIN;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      arg_expr = arg_expr->decompress();
      result_type = arg_expr->get_type_info();
    } else if (boost::iequals(*name, "max")) {
      agg_type = kMAX;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      arg_expr = arg_expr->decompress();
      result_type = arg_expr->get_type_info();
    } else if (boost::iequals(*name, "avg")) {
      agg_type = kAVG;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      if (!arg_expr->get_type_info().is_number())
        throw std::runtime_error("Cannot compute AVG on non-number-type arguments.");
      arg_expr = arg_expr->decompress();
      result_type = SQLTypeInfo(kDOUBLE, false);
    } else if (boost::iequals(*name, "sum")) {
      agg_type = kSUM;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      if (!arg_expr->get_type_info().is_number())
        throw std::runtime_error("Cannot compute SUM on non-number-type arguments.");
      arg_expr = arg_expr->decompress();
      result_type = arg_expr->get_type_info().is_integer() ? SQLTypeInfo(kBIGINT, false) : arg_expr->get_type_info();
    } else if (boost::iequals(*name, "unnest")) {
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      const SQLTypeInfo& arg_ti = arg_expr->get_type_info();
      if (arg_ti.get_type() != kARRAY)
        throw std::runtime_error(arg->to_string() + " is not of array type.");
      return makeExpr<Analyzer::UOper>(arg_ti.get_elem_type(), false, kUNNEST, arg_expr);
    } else
      throw std::runtime_error("invalid function name: " + *name);
    if (arg_expr->get_type_info().is_string() || arg_expr->get_type_info().get_type() == kARRAY)
      throw std::runtime_error("Only COUNT(DISTINCT ) aggregate is supported on strings and arrays.");
  }
  int naggs = query.get_num_aggs();
  query.set_num_aggs(naggs + 1);
  return makeExpr<Analyzer::AggExpr>(result_type, agg_type, arg_expr, is_distinct);
}

std::shared_ptr<Analyzer::Expr> CastExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref) const {
  target_type->check_type();
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  SQLTypeInfo ti(target_type->get_type(),
                 target_type->get_param1(),
                 target_type->get_param2(),
                 arg_expr->get_type_info().get_notnull());
  if (arg_expr->get_type_info().get_type() != target_type->get_type() &&
      arg_expr->get_type_info().get_compression() != kENCODING_NONE)
    arg_expr->decompress();
  return arg_expr->add_cast(ti);
}

std::shared_ptr<Analyzer::Expr> CaseExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref) const {
  SQLTypeInfo ti;
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>> expr_pair_list;
  for (auto& p : when_then_list) {
    auto e1 = p->get_expr1()->analyze(catalog, query, allow_tlist_ref);
    if (e1->get_type_info().get_type() != kBOOLEAN)
      throw std::runtime_error("Only boolean expressions can be used after WHEN.");
    auto e2 = p->get_expr2()->analyze(catalog, query, allow_tlist_ref);
    expr_pair_list.emplace_back(e1, e2);
  }
  auto else_e = else_expr ? else_expr->analyze(catalog, query, allow_tlist_ref) : nullptr;
  return normalize(expr_pair_list, else_e);
}

namespace {

bool expr_is_null(const Analyzer::Expr* expr) {
  if (expr->get_type_info().get_type() == kNULLT) {
    return true;
  }
  const auto const_expr = dynamic_cast<const Analyzer::Constant*>(expr);
  return const_expr && const_expr->get_is_null();
}

}  // namespace

std::shared_ptr<Analyzer::Expr> CaseExpr::normalize(
    const std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>& expr_pair_list,
    const std::shared_ptr<Analyzer::Expr> else_e_in) {
  SQLTypeInfo ti;
  bool has_agg = false;
  for (auto& p : expr_pair_list) {
    auto e1 = p.first;
    CHECK(e1->get_type_info().is_boolean());
    auto e2 = p.second;
    if (ti.get_type() == kNULLT)
      ti = e2->get_type_info();
    else if (e2->get_type_info().get_type() == kNULLT) {
      ti.set_notnull(false);
      e2->set_type_info(ti);
    } else if (ti != e2->get_type_info()) {
      if (ti.is_string() && e2->get_type_info().is_string())
        ti = Analyzer::BinOper::common_string_type(ti, e2->get_type_info());
      else if (ti.is_number() && e2->get_type_info().is_number())
        ti = Analyzer::BinOper::common_numeric_type(ti, e2->get_type_info());
      else if (ti.is_boolean() && e2->get_type_info().is_boolean())
        ti = Analyzer::BinOper::common_numeric_type(ti, e2->get_type_info());
      else
        throw std::runtime_error("expressions in THEN clause must be of the same or compatible types.");
    }
    if (e2->get_contains_agg())
      has_agg = true;
  }
  auto else_e = else_e_in;
  if (else_e) {
    if (else_e->get_contains_agg())
      has_agg = true;
    if (expr_is_null(else_e.get())) {
      ti.set_notnull(false);
      else_e->set_type_info(ti);
    } else if (ti != else_e->get_type_info()) {
      ti.set_notnull(false);
      if (ti.is_string() && else_e->get_type_info().is_string())
        ti = Analyzer::BinOper::common_string_type(ti, else_e->get_type_info());
      else if (ti.is_number() && else_e->get_type_info().is_number())
        ti = Analyzer::BinOper::common_numeric_type(ti, else_e->get_type_info());
      else if (ti.is_boolean() && else_e->get_type_info().is_boolean())
        ti = Analyzer::BinOper::common_numeric_type(ti, else_e->get_type_info());
      else
        throw std::runtime_error(
            "expressions in ELSE clause must be of the same or compatible types as those in the "
            "THEN clauses.");
    }
  }
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>> cast_expr_pair_list;
  for (auto p : expr_pair_list) {
    ti.set_notnull(false);
    cast_expr_pair_list.push_back(std::make_pair(p.first, p.second->add_cast(ti)));
  }
  if (else_e != nullptr) {
    else_e = else_e->add_cast(ti);
  } else {
    Datum d;
    // always create an else expr so that executor doesn't need to worry about it
    ti.set_notnull(false);
    else_e = makeExpr<Analyzer::Constant>(ti, true, d);
  }
  if (ti.get_type() == kNULLT) {
    throw std::runtime_error("Can't deduce the type for case expressions, all branches null");
  }
  return makeExpr<Analyzer::CaseExpr>(ti, has_agg, cast_expr_pair_list, else_e);
}

std::string CaseExpr::to_string() const {
  std::string str("CASE ");
  for (auto& p : when_then_list) {
    str += "WHEN " + p->get_expr1()->to_string() + " THEN " + p->get_expr2()->to_string() + " ";
  }
  if (else_expr != nullptr)
    str += "ELSE " + else_expr->to_string();
  str += " END";
  return str;
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                     Analyzer::Query& query,
                                                     TlistRefType allow_tlist_ref) const {
  auto from_expr = from_arg->analyze(catalog, query, allow_tlist_ref);
  return get(from_expr, *field);
}

std::shared_ptr<Analyzer::Expr> ExtractExpr::get(const std::shared_ptr<Analyzer::Expr> from_expr,
                                                 const std::string& field) {
  return ExtractExpr::get(from_expr, to_extract_field(field));
}

namespace {

std::string extract_field_name(const ExtractField fieldno) {
  switch (fieldno) {
    case kYEAR:
      return "year";
    case kQUARTER:
      return "quarter";
    case kMONTH:
      return "month";
    case kDAY:
      return "day";
    case kHOUR:
      return "hour";
    case kMINUTE:
      return "minute";
    case kSECOND:
      return "second";
    case kDOW:
      return "dow";
    case kISODOW:
      return "isodow";
    case kDOY:
      return "doy";
    case kEPOCH:
      return "epoch";
    case kQUARTERDAY:
      return "quarterday";
    case kWEEK:
      return "week";
  }
  CHECK(false);
  return "";
}

}  // namespace

std::shared_ptr<Analyzer::Expr> ExtractExpr::get(const std::shared_ptr<Analyzer::Expr> from_expr,
                                                 const ExtractField fieldno) {
  if (!from_expr->get_type_info().is_time())
    throw std::runtime_error("Only TIME, TIMESTAMP and DATE types can be in EXTRACT function.");
  switch (from_expr->get_type_info().get_type()) {
    case kTIME:
      if (fieldno != kHOUR && fieldno != kMINUTE && fieldno != kSECOND)
        throw std::runtime_error("Cannot EXTRACT " + extract_field_name(fieldno) + " from TIME.");
      break;
    default:
      break;
  }
  SQLTypeInfo ti(kBIGINT, 0, 0, from_expr->get_type_info().get_notnull());
  auto c = std::dynamic_pointer_cast<Analyzer::Constant>(from_expr);
  if (c != nullptr) {
    c->set_type_info(ti);
    Datum d;
    d.bigintval = ExtractFromTime(fieldno, c->get_constval().timeval);
    c->set_constval(d);
    return c;
  }
  return makeExpr<Analyzer::ExtractExpr>(ti, from_expr->get_contains_agg(), fieldno, from_expr->decompress());
}

ExtractField ExtractExpr::to_extract_field(const std::string& field) {
  ExtractField fieldno;
  if (boost::iequals(field, "year"))
    fieldno = kYEAR;
  else if (boost::iequals(field, "quarter"))
    fieldno = kQUARTER;
  else if (boost::iequals(field, "month"))
    fieldno = kMONTH;
  else if (boost::iequals(field, "day"))
    fieldno = kDAY;
  else if (boost::iequals(field, "quarterday"))
    fieldno = kQUARTERDAY;
  else if (boost::iequals(field, "hour"))
    fieldno = kHOUR;
  else if (boost::iequals(field, "minute"))
    fieldno = kMINUTE;
  else if (boost::iequals(field, "second"))
    fieldno = kSECOND;
  else if (boost::iequals(field, "dow"))
    fieldno = kDOW;
  else if (boost::iequals(field, "isodow"))
    fieldno = kISODOW;
  else if (boost::iequals(field, "doy"))
    fieldno = kDOY;
  else if (boost::iequals(field, "epoch"))
    fieldno = kEPOCH;
  else if (boost::iequals(field, "week"))
    fieldno = kWEEK;
  else
    throw std::runtime_error("Invalid field in EXTRACT function " + field);
  return fieldno;
}

std::string ExtractExpr::to_string() const {
  std::string str("EXTRACT(");
  str += *field + " FROM " + from_arg->to_string() + ")";
  return str;
}

/*
 * year
 * month
 * day
 * hour
 * minute
 * second
 *
 * millennium
 * century
 * decade
 * milliseconds
 * microseconds
 * week
 */
std::shared_ptr<Analyzer::Expr> DatetruncExpr::analyze(const Catalog_Namespace::Catalog& catalog,
                                                       Analyzer::Query& query,
                                                       TlistRefType allow_tlist_ref) const {
  auto from_expr = from_arg->analyze(catalog, query, allow_tlist_ref);
  return get(from_expr, *field);
}

std::shared_ptr<Analyzer::Expr> DatetruncExpr::get(const std::shared_ptr<Analyzer::Expr> from_expr,
                                                   const std::string& field) {
  const auto fieldno = to_date_trunc_field(field);
  return DatetruncExpr::get(from_expr, fieldno);
}

std::string dt_field_name(const DatetruncField fieldno) {
  switch (fieldno) {
    case dtYEAR:
      return "year";
    case dtQUARTER:
      return "quarter";
    case dtMONTH:
      return "month";
    case dtQUARTERDAY:
      return "quarterday";
    case dtDAY:
      return "day";
    case dtHOUR:
      return "hour";
    case dtMINUTE:
      return "minute";
    case dtSECOND:
      return "second";
    case dtMILLENNIUM:
      return "millennium";
    case dtCENTURY:
      return "century";
    case dtDECADE:
      return "decade";
    case dtMILLISECOND:
      return "millisecond";
    case dtMICROSECOND:
      return "microsecond";
    case dtWEEK:
      return "week";
    case dtINVALID:
      CHECK(false);
  }
  CHECK(false);
  return "";
}

std::shared_ptr<Analyzer::Expr> DatetruncExpr::get(const std::shared_ptr<Analyzer::Expr> from_expr,
                                                   const DatetruncField fieldno) {
  if (!from_expr->get_type_info().is_time())
    throw std::runtime_error("Only TIME, TIMESTAMP and DATE types can be in DATE_TRUNC function.");
  switch (from_expr->get_type_info().get_type()) {
    case kTIME:
      if (fieldno != dtHOUR && fieldno != dtMINUTE && fieldno != dtSECOND)
        throw std::runtime_error("Cannot DATE_TRUNC " + dt_field_name(fieldno) + " from TIME.");
      break;
    default:
      break;
  }
  SQLTypeInfo ti(kTIMESTAMP, 0, 0, from_expr->get_type_info().get_notnull());
  auto c = std::dynamic_pointer_cast<Analyzer::Constant>(from_expr);
  if (c != nullptr) {
    c->set_type_info(ti);
    Datum d;
    d.bigintval = DateTruncate(fieldno, c->get_constval().timeval);
    c->set_constval(d);
    return c;
  }
  return makeExpr<Analyzer::DatetruncExpr>(ti, from_expr->get_contains_agg(), fieldno, from_expr->decompress());
}

DatetruncField DatetruncExpr::to_date_trunc_field(const std::string& field) {
  // TODO(alex): unify with the similar function in ExtractExpr?
  DatetruncField fieldno;
  if (boost::iequals(field, "year"))
    fieldno = dtYEAR;
  else if (boost::iequals(field, "quarter"))
    fieldno = dtQUARTER;
  else if (boost::iequals(field, "month"))
    fieldno = dtMONTH;
  else if (boost::iequals(field, "quarterday"))
    fieldno = dtQUARTERDAY;
  else if (boost::iequals(field, "day"))
    fieldno = dtDAY;
  else if (boost::iequals(field, "hour"))
    fieldno = dtHOUR;
  else if (boost::iequals(field, "minute"))
    fieldno = dtMINUTE;
  else if (boost::iequals(field, "second"))
    fieldno = dtSECOND;
  else if (boost::iequals(field, "millennium"))
    fieldno = dtMILLENNIUM;
  else if (boost::iequals(field, "century"))
    fieldno = dtCENTURY;
  else if (boost::iequals(field, "decade"))
    fieldno = dtDECADE;
  else if (boost::iequals(field, "millisecond"))
    fieldno = dtMILLISECOND;
  else if (boost::iequals(field, "microsecond"))
    fieldno = dtMICROSECOND;
  else if (boost::iequals(field, "week"))
    fieldno = dtWEEK;
  else
    throw std::runtime_error("Invalid field in DATE_TRUNC function " + field);
  return fieldno;
}

std::string DatetruncExpr::to_string() const {
  std::string str("DATE_TRUNC(");
  str += *field + " , " + from_arg->to_string() + ")";
  return str;
}

void UnionQuery::analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  left->analyze(catalog, query);
  Analyzer::Query* right_query = new Analyzer::Query();
  right->analyze(catalog, *right_query);
  query.set_next_query(right_query);
  query.set_is_unionall(is_unionall);
}

void QuerySpec::analyze_having_clause(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  std::shared_ptr<Analyzer::Expr> p;
  if (having_clause != nullptr) {
    p = having_clause->analyze(catalog, query, Expr::TlistRefType::TLIST_COPY);
    if (p->get_type_info().get_type() != kBOOLEAN)
      throw std::runtime_error("Only boolean expressions can be in HAVING clause.");
    p->check_group_by(query.get_group_by());
  }
  query.set_having_predicate(p);
}

void QuerySpec::analyze_group_by(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  std::list<std::shared_ptr<Analyzer::Expr>> groupby;
  if (!groupby_clause.empty()) {
    int gexpr_no = 1;
    std::shared_ptr<Analyzer::Expr> gexpr;
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist = query.get_targetlist();
    for (auto& c : groupby_clause) {
      // special-case ordinal numbers in GROUP BY
      if (dynamic_cast<Literal*>(c.get())) {
        IntLiteral* i = dynamic_cast<IntLiteral*>(c.get());
        if (!i)
          throw std::runtime_error("Invalid literal in GROUP BY clause.");
        int varno = (int)i->get_intval();
        if (varno <= 0 || varno > static_cast<int>(tlist.size()))
          throw std::runtime_error("Invalid ordinal number in GROUP BY clause.");
        if (tlist[varno - 1]->get_expr()->get_contains_agg())
          throw std::runtime_error(
              "Ordinal number in GROUP BY cannot reference an expression containing aggregate "
              "functions.");
        gexpr = makeExpr<Analyzer::Var>(tlist[varno - 1]->get_expr()->get_type_info(), Analyzer::Var::kOUTPUT, varno);
      } else {
        gexpr = c->analyze(catalog, query, Expr::TlistRefType::TLIST_REF);
      }
      const SQLTypeInfo gti = gexpr->get_type_info();
      bool set_new_type = false;
      SQLTypeInfo ti(gti);
      if (gti.is_string() && gti.get_compression() == kENCODING_NONE) {
        set_new_type = true;
        ti.set_compression(kENCODING_DICT);
        ti.set_comp_param(TRANSIENT_DICT_ID);
        ti.set_fixed_size();
      }
      std::shared_ptr<Analyzer::Var> v;
      if (std::dynamic_pointer_cast<Analyzer::Var>(gexpr)) {
        v = std::static_pointer_cast<Analyzer::Var>(gexpr);
        int n = v->get_varno();
        gexpr = tlist[n - 1]->get_own_expr();
        auto cv = std::dynamic_pointer_cast<Analyzer::ColumnVar>(gexpr);
        if (cv != nullptr) {
          // inherit all ColumnVar info for lineage.
          *std::static_pointer_cast<Analyzer::ColumnVar>(v) = *cv;
        }
        v->set_which_row(Analyzer::Var::kGROUPBY);
        v->set_varno(gexpr_no);
        tlist[n - 1]->set_expr(v);
      }
      if (set_new_type) {
        auto new_e = gexpr->add_cast(ti);
        groupby.push_back(new_e);
        if (v != nullptr)
          v->set_type_info(new_e->get_type_info());
      } else
        groupby.push_back(gexpr);
      gexpr_no++;
    }
  }
  if (query.get_num_aggs() > 0 || !groupby.empty()) {
    for (auto t : query.get_targetlist()) {
      auto e = t->get_expr();
      e->check_group_by(groupby);
    }
  }
  query.set_group_by(groupby);
}

void QuerySpec::analyze_where_clause(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  if (where_clause == nullptr) {
    query.set_where_predicate(nullptr);
    return;
  }
  auto p = where_clause->analyze(catalog, query, Expr::TlistRefType::TLIST_COPY);
  if (p->get_type_info().get_type() != kBOOLEAN)
    throw std::runtime_error("Only boolean expressions can be in WHERE clause.");
  query.set_where_predicate(p);
}

void QuerySpec::analyze_select_clause(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist = query.get_targetlist_nonconst();
  if (select_clause.empty()) {
    // this means SELECT *
    int rte_idx = 0;
    for (auto rte : query.get_rangetable()) {
      rte->expand_star_in_targetlist(catalog, tlist, rte_idx++);
    }
  } else {
    for (auto& p : select_clause) {
      const Parser::Expr* select_expr = p->get_select_expr();
      // look for the case of range_var.*
      if (typeid(*select_expr) == typeid(ColumnRef) &&
          dynamic_cast<const ColumnRef*>(select_expr)->get_column() == nullptr) {
        const std::string* range_var_name = dynamic_cast<const ColumnRef*>(select_expr)->get_table();
        int rte_idx = query.get_rte_idx(*range_var_name);
        if (rte_idx < 0)
          throw std::runtime_error("invalid range variable name: " + *range_var_name);
        Analyzer::RangeTblEntry* rte = query.get_rte(rte_idx);
        rte->expand_star_in_targetlist(catalog, tlist, rte_idx);
      } else {
        auto e = select_expr->analyze(catalog, query);
        std::string resname;

        if (p->get_alias() != nullptr)
          resname = *p->get_alias();
        else if (std::dynamic_pointer_cast<Analyzer::ColumnVar>(e) && !std::dynamic_pointer_cast<Analyzer::Var>(e)) {
          auto colvar = std::static_pointer_cast<Analyzer::ColumnVar>(e);
          const ColumnDescriptor* col_desc =
              catalog.getMetadataForColumn(colvar->get_table_id(), colvar->get_column_id());
          resname = col_desc->columnName;
        }
        if (e->get_type_info().get_type() == kNULLT)
          throw std::runtime_error("Untyped NULL in SELECT clause.  Use CAST to specify a type.");
        auto o = std::static_pointer_cast<Analyzer::UOper>(e);
        bool unnest = (o != nullptr && o->get_optype() == kUNNEST);
        auto tle = std::make_shared<Analyzer::TargetEntry>(resname, e, unnest);
        tlist.push_back(tle);
      }
    }
  }
}

void QuerySpec::analyze_from_clause(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  Analyzer::RangeTblEntry* rte;
  for (auto& p : from_clause) {
    const TableDescriptor* table_desc;
    table_desc = catalog.getMetadataForTable(*p->get_table_name());
    if (table_desc == nullptr)
      throw std::runtime_error("Table " + *p->get_table_name() + " does not exist.");
    std::string range_var;
    if (p->get_range_var() == nullptr)
      range_var = *p->get_table_name();
    else
      range_var = *p->get_range_var();
    rte = new Analyzer::RangeTblEntry(range_var, table_desc, nullptr);
    query.add_rte(rte);
  }
}

void QuerySpec::analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  query.set_is_distinct(is_distinct);
  analyze_from_clause(catalog, query);
  analyze_select_clause(catalog, query);
  analyze_where_clause(catalog, query);
  analyze_group_by(catalog, query);
  analyze_having_clause(catalog, query);
}

void SelectStmt::analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  query.set_stmt_type(kSELECT);
  query.set_limit(limit);
  if (offset < 0)
    throw std::runtime_error("OFFSET cannot be negative.");
  query.set_offset(offset);
  query_expr->analyze(catalog, query);
  if (orderby_clause.empty() && !query.get_is_distinct()) {
    query.set_order_by(nullptr);
    return;
  }
  const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist = query.get_targetlist();
  std::list<Analyzer::OrderEntry>* order_by = new std::list<Analyzer::OrderEntry>();
  if (!orderby_clause.empty()) {
    for (auto& p : orderby_clause) {
      int tle_no = p->get_colno();
      if (tle_no == 0) {
        // use column name
        // search through targetlist for matching name
        const std::string* name = p->get_column()->get_column();
        tle_no = 1;
        bool found = false;
        for (auto tle : tlist) {
          if (tle->get_resname() == *name) {
            found = true;
            break;
          }
          tle_no++;
        }
        if (!found)
          throw std::runtime_error("invalid name in order by: " + *name);
      }
      order_by->push_back(Analyzer::OrderEntry(tle_no, p->get_is_desc(), p->get_nulls_first()));
    }
  }
  if (query.get_is_distinct()) {
    // extend order_by to include all targetlist entries.
    for (int i = 1; i <= static_cast<int>(tlist.size()); i++) {
      bool in_orderby = false;
      std::for_each(order_by->begin(), order_by->end(), [&in_orderby, i](const Analyzer::OrderEntry& oe) {
        in_orderby = in_orderby || (i == oe.tle_no);
      });
      if (!in_orderby)
        order_by->push_back(Analyzer::OrderEntry(i, false, false));
    }
  }
  query.set_order_by(order_by);
}

std::string SQLType::to_string() const {
  std::string str;
  switch (type) {
    case kBOOLEAN:
      str = "BOOLEAN";
      break;
    case kCHAR:
      str = "CHAR(" + boost::lexical_cast<std::string>(param1) + ")";
      break;
    case kVARCHAR:
      str = "VARCHAR(" + boost::lexical_cast<std::string>(param1) + ")";
      break;
    case kTEXT:
      str = "TEXT";
      break;
    case kNUMERIC:
      str = "NUMERIC(" + boost::lexical_cast<std::string>(param1);
      if (param2 > 0)
        str += ", " + boost::lexical_cast<std::string>(param2);
      str += ")";
      break;
    case kDECIMAL:
      str = "DECIMAL(" + boost::lexical_cast<std::string>(param1);
      if (param2 > 0)
        str += ", " + boost::lexical_cast<std::string>(param2);
      str += ")";
      break;
    case kBIGINT:
      str = "BIGINT";
      break;
    case kINT:
      str = "INT";
      break;
    case kSMALLINT:
      str = "SMALLINT";
      break;
    case kFLOAT:
      str = "FLOAT";
      break;
    case kDOUBLE:
      str = "DOUBLE";
      break;
    case kTIME:
      str = "TIME";
      if (param1 < 6)
        str += "(" + boost::lexical_cast<std::string>(param1) + ")";
      break;
    case kTIMESTAMP:
      str = "TIMESTAMP";
      if (param1 < 6)
        str += "(" + boost::lexical_cast<std::string>(param1) + ")";
      break;
    case kDATE:
      str = "DATE";
      break;
    default:
      assert(false);
      break;
  }
  if (is_array)
    str += "[]";
  return str;
}

std::string SelectEntry::to_string() const {
  std::string str = select_expr->to_string();
  if (alias != nullptr)
    str += " AS " + *alias;
  return str;
}

std::string TableRef::to_string() const {
  std::string str = *table_name;
  if (range_var != nullptr)
    str += " " + *range_var;
  return str;
}

std::string ColumnRef::to_string() const {
  std::string str;
  if (table == nullptr)
    str = *column;
  else if (column == nullptr)
    str = *table + ".*";
  else
    str = *table + "." + *column;
  return str;
}

std::string OperExpr::to_string() const {
  std::string op_str[] = {"=", "===", "<>", "<", ">", "<=", ">=", " AND ", " OR ", "NOT", "-", "+", "*", "/"};
  std::string str;
  if (optype == kUMINUS)
    str = "-(" + left->to_string() + ")";
  else if (optype == kNOT)
    str = "NOT (" + left->to_string() + ")";
  else if (optype == kARRAY_AT)
    str = left->to_string() + "[" + right->to_string() + "]";
  else if (optype == kUNNEST)
    str = "UNNEST(" + left->to_string() + ")";
  else if (optype == kIN)
    str = "(" + left->to_string() + " IN " + right->to_string() + ")";
  else
    str = "(" + left->to_string() + op_str[optype] + right->to_string() + ")";
  return str;
}

std::string InExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not)
    str += " NOT IN ";
  else
    str += " IN ";
  return str;
}

std::string ExistsExpr::to_string() const {
  return "EXISTS (" + query->to_string() + ")";
}

std::string SubqueryExpr::to_string() const {
  std::string str;
  str = "(";
  str += query->to_string();
  str += ")";
  return str;
}

std::string IsNullExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not)
    str += " IS NOT NULL";
  else
    str += " IS NULL";
  return str;
}

std::string InSubquery::to_string() const {
  std::string str = InExpr::to_string();
  str += subquery->to_string();
  return str;
}

std::string InValues::to_string() const {
  std::string str = InExpr::to_string() + "(";
  bool notfirst = false;
  for (auto& p : value_list) {
    if (notfirst)
      str += ", ";
    else
      notfirst = true;
    str += p->to_string();
  }
  str += ")";
  return str;
}

std::string BetweenExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not)
    str += " NOT BETWEEN ";
  else
    str += " BETWEEN ";
  str += lower->to_string() + " AND " + upper->to_string();
  return str;
}

std::string CharLengthExpr::to_string() const {
  std::string str;
  if (calc_encoded_length)
    str = "CHAR_LENGTH (" + arg->to_string() + ")";
  else
    str = "LENGTH (" + arg->to_string() + ")";
  return str;
}

std::string LikeExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not)
    str += " NOT LIKE ";
  else
    str += " LIKE ";
  str += like_string->to_string();
  if (escape_string != nullptr)
    str += " ESCAPE " + escape_string->to_string();
  return str;
}

std::string RegexpExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not)
    str += " NOT REGEXP ";
  else
    str += " REGEXP ";
  str += pattern_string->to_string();
  if (escape_string != nullptr)
    str += " ESCAPE " + escape_string->to_string();
  return str;
}

std::string LikelihoodExpr::to_string() const {
  std::string str = " LIKELIHOOD ";
  str += arg->to_string();
  str += " ";
  str += boost::lexical_cast<std::string>(is_not ? 1.0 - likelihood : likelihood);
  return str;
}

std::string FunctionRef::to_string() const {
  std::string str = *name + "(";
  if (distinct)
    str += "DISTINCT ";
  if (arg == nullptr)
    str += "*)";
  else
    str += arg->to_string() + ")";
  return str;
}

std::string QuerySpec::to_string() const {
  std::string query_str = "SELECT ";
  if (is_distinct)
    query_str += "DISTINCT ";
  if (select_clause.empty())
    query_str += "* ";
  else {
    bool notfirst = false;
    for (auto& p : select_clause) {
      if (notfirst)
        query_str += ", ";
      else
        notfirst = true;
      query_str += p->to_string();
    }
  }
  query_str += " FROM ";
  bool notfirst = false;
  for (auto& p : from_clause) {
    if (notfirst)
      query_str += ", ";
    else
      notfirst = true;
    query_str += p->to_string();
  }
  if (where_clause)
    query_str += " WHERE " + where_clause->to_string();
  if (!groupby_clause.empty()) {
    query_str += " GROUP BY ";
    bool notfirst = false;
    for (auto& p : groupby_clause) {
      if (notfirst)
        query_str += ", ";
      else
        notfirst = true;
      query_str += p->to_string();
    }
  }
  if (having_clause) {
    query_str += " HAVING " + having_clause->to_string();
  }
  query_str += ";";
  return query_str;
}

void InsertStmt::analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  query.set_stmt_type(kINSERT);
  const TableDescriptor* td = catalog.getMetadataForTable(*table);
  if (td == nullptr)
    throw std::runtime_error("Table " + *table + " does not exist.");
  if (td->isView)
    throw std::runtime_error("Insert to views is not supported yet.");
  query.set_result_table_id(td->tableId);
  std::list<int> result_col_list;
  if (column_list.empty()) {
    const std::list<const ColumnDescriptor*> all_cols = catalog.getAllColumnMetadataForTable(td->tableId, false, false);
    for (auto cd : all_cols) {
      result_col_list.push_back(cd->columnId);
    }
  } else {
    for (auto& c : column_list) {
      const ColumnDescriptor* cd = catalog.getMetadataForColumn(td->tableId, *c);
      if (cd == nullptr)
        throw std::runtime_error("Column " + *c + " does not exist.");
      result_col_list.push_back(cd->columnId);
    }
    if (catalog.getAllColumnMetadataForTable(td->tableId, false, false).size() != result_col_list.size())
      throw std::runtime_error("Insert into a subset of columns is not supported yet.");
  }
  query.set_result_col_list(result_col_list);
}

void InsertValuesStmt::analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  InsertStmt::analyze(catalog, query);
  std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist = query.get_targetlist_nonconst();
  if (query.get_result_col_list().size() != value_list.size())
    throw std::runtime_error("Insert has more target columns than expressions.");
  std::list<int>::const_iterator it = query.get_result_col_list().begin();
  for (auto& v : value_list) {
    auto e = v->analyze(catalog, query);
    const ColumnDescriptor* cd = catalog.getMetadataForColumn(query.get_result_table_id(), *it);
    assert(cd != nullptr);
    if (cd->columnType.get_notnull()) {
      auto c = std::dynamic_pointer_cast<Analyzer::Constant>(e);
      if (c != nullptr && c->get_is_null())
        throw std::runtime_error("Cannot insert NULL into column " + cd->columnName);
    }
    e = e->add_cast(cd->columnType);
    tlist.emplace_back(new Analyzer::TargetEntry("", e, false));
    ++it;
  }
}

void InsertQueryStmt::analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& insert_query) const {
  InsertStmt::analyze(catalog, insert_query);
  query->analyze(catalog, insert_query);
}

void UpdateStmt::analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  throw std::runtime_error("UPDATE statement not supported yet.");
}

void DeleteStmt::analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const {
  throw std::runtime_error("DELETE statement not supported yet.");
}

void SQLType::check_type() {
  switch (type) {
    case kCHAR:
    case kVARCHAR:
      if (param1 <= 0)
        throw std::runtime_error("CHAR and VARCHAR must have a positive dimension.");
      break;
    case kDECIMAL:
    case kNUMERIC:
      if (param1 <= 0)
        throw std::runtime_error("DECIMAL and NUMERIC must have a positive precision.");
      else if (param1 > 19)
        throw std::runtime_error("DECIMAL and NUMERIC precision cannot be larger than 19.");
      else if (param1 <= param2)
        throw std::runtime_error("DECIMAL and NUMERIC must have precision larger than scale.");
      break;
    case kTIMESTAMP:
    case kTIME:
      if (param1 == -1)
        param1 = 0;      // default precision is 0
      if (param1 > 0) {  // @TODO(wei) support sub-second precision later.
        if (type == kTIMESTAMP)
          throw std::runtime_error("Only TIMESTAMP(0) is supported now.");
        else
          throw std::runtime_error("Only TIME(0) is supported now.");
      }
      break;
    default:
      param1 = 0;
      break;
  }
}

namespace {

size_t shard_column_index(const std::string& name, const std::list<ColumnDescriptor>& columns) {
  size_t index = 1;
  for (const auto& cd : columns) {
    if (cd.columnName == name) {
      return index;
    }
    ++index;
  }
  // Not found, return 0
  return 0;
}

void validate_shard_column_type(const size_t shard_column_id, const std::list<ColumnDescriptor>& columns) {
  CHECK_NE(size_t(0), shard_column_id);
  CHECK_LE(shard_column_id, columns.size());
  auto column_it = columns.begin();
  std::advance(column_it, shard_column_id - 1);
  const auto& col_ti = column_it->columnType;
  if (col_ti.is_integer() || (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT)) {
    return;
  }
  throw std::runtime_error("Cannot shard on type " + col_ti.get_type_name() + ", encoding " +
                           col_ti.get_compression_name());
}

void set_string_field(rapidjson::Value& obj,
                      const std::string& field_name,
                      const std::string& field_value,
                      rapidjson::Document& document) {
  rapidjson::Value field_name_json_str;
  field_name_json_str.SetString(field_name.c_str(), field_name.size(), document.GetAllocator());
  rapidjson::Value field_value_json_str;
  field_value_json_str.SetString(field_value.c_str(), field_value.size(), document.GetAllocator());
  obj.AddMember(field_name_json_str, field_value_json_str, document.GetAllocator());
}

std::string serialize_key_metainfo(const ShardKeyDef* shard_key_def,
                                   const std::vector<SharedDictionaryDef>& shared_dict_defs) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();
  rapidjson::Value arr(rapidjson::kArrayType);
  if (shard_key_def) {
    rapidjson::Value shard_key_obj(rapidjson::kObjectType);
    set_string_field(shard_key_obj, "type", "SHARD KEY", document);
    set_string_field(shard_key_obj, "name", shard_key_def->get_column(), document);
    arr.PushBack(shard_key_obj, allocator);
  }
  for (const auto& shared_dict_def : shared_dict_defs) {
    rapidjson::Value shared_dict_obj(rapidjson::kObjectType);
    set_string_field(shared_dict_obj, "type", "SHARED DICTIONARY", document);
    set_string_field(shared_dict_obj, "name", shared_dict_def.get_column(), document);
    set_string_field(shared_dict_obj, "foreign_table", shared_dict_def.get_foreign_table(), document);
    set_string_field(shared_dict_obj, "foreign_column", shared_dict_def.get_foreign_column(), document);
    arr.PushBack(shared_dict_obj, allocator);
  }
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  arr.Accept(writer);
  return buffer.GetString();
}

}  // namespace

void CreateTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  // check access privileges
  if (!session.checkDBAccessPrivileges({false, false, true})) {  // CREATE
    throw std::runtime_error("Table " + *table + " will not be created. User has no create privileges.");
  }

  if (catalog.getMetadataForTable(*table) != nullptr) {
    if (if_not_exists_)
      return;
    throw std::runtime_error("Table " + *table + " already exists.");
  }
  std::list<ColumnDescriptor> columns;
  std::unordered_set<std::string> uc_col_names;
  std::vector<SharedDictionaryDef> shared_dict_defs;
  const ShardKeyDef* shard_key_def{nullptr};
  for (auto& e : table_element_list) {
    if (dynamic_cast<SharedDictionaryDef*>(e.get())) {
      auto shared_dict_def = static_cast<SharedDictionaryDef*>(e.get());
      validate_shared_dictionary(this, shared_dict_def, columns, shared_dict_defs, catalog);
      shared_dict_defs.push_back(*shared_dict_def);
      continue;
    }
    if (dynamic_cast<ShardKeyDef*>(e.get())) {
      if (shard_key_def) {
        throw std::runtime_error("Specified more than one shard key");
      }
      shard_key_def = static_cast<const ShardKeyDef*>(e.get());
      continue;
    }
    if (!dynamic_cast<ColumnDef*>(e.get()))
      throw std::runtime_error("Table constraints are not supported yet.");
    ColumnDef* coldef = static_cast<ColumnDef*>(e.get());
    ColumnDescriptor cd;
    cd.columnName = *coldef->get_column_name();
    const auto uc_col_name = boost::to_upper_copy<std::string>(cd.columnName);
    if (reserved_keywords.find(uc_col_name) != reserved_keywords.end()) {
      throw std::runtime_error("Cannot create column with reserved keyword '" + cd.columnName + "'");
    }
    const auto it_ok = uc_col_names.insert(uc_col_name);
    if (!it_ok.second) {
      throw std::runtime_error("Column '" + cd.columnName + "' defined more than once");
    }
    SQLType* t = coldef->get_column_type();
    t->check_type();
    if (t->get_is_array()) {
      cd.columnType.set_type(kARRAY);
      cd.columnType.set_subtype(t->get_type());
    } else
      cd.columnType.set_type(t->get_type());
    cd.columnType.set_dimension(t->get_param1());
    cd.columnType.set_scale(t->get_param2());
    const ColumnConstraintDef* cc = coldef->get_column_constraint();
    if (cc == nullptr)
      cd.columnType.set_notnull(false);
    else {
      cd.columnType.set_notnull(cc->get_notnull());
    }
    const CompressDef* compression = coldef->get_compression();
    if (compression == nullptr) {
      // Change default TEXT column behaviour to be DICT encoded
      if (cd.columnType.is_string() || cd.columnType.is_string_array()) {
        // default to 32-bits
        cd.columnType.set_compression(kENCODING_DICT);
        cd.columnType.set_comp_param(32);
      } else {
        cd.columnType.set_compression(kENCODING_NONE);
        cd.columnType.set_comp_param(0);
      }
    } else {
      const std::string& comp = *compression->get_encoding_name();
      int comp_param;
      if (boost::iequals(comp, "fixed")) {
        if (!cd.columnType.is_integer() && !cd.columnType.is_time())
          throw std::runtime_error(cd.columnName + ": Fixed encoding is only supported for integer or time columns.");
        // fixed-bits encoding
        SQLTypes type = cd.columnType.get_type();
        if (type == kARRAY)
          type = cd.columnType.get_subtype();
        switch (type) {
          case kSMALLINT:
            if (compression->get_encoding_param() != 8)
              throw std::runtime_error(cd.columnName +
                                       ": Compression parameter for Fixed encoding on SMALLINT must be 8.");
            break;
          case kINT:
            if (compression->get_encoding_param() != 8 && compression->get_encoding_param() != 16)
              throw std::runtime_error(cd.columnName +
                                       ": Compression parameter for Fixed encoding on INTEGER must be 8 or 16.");
            break;
          case kBIGINT:
            if (compression->get_encoding_param() != 8 && compression->get_encoding_param() != 16 &&
                compression->get_encoding_param() != 32)
              throw std::runtime_error(cd.columnName +
                                       ": Compression parameter for Fixed encoding on BIGINT must be 8 or 16 or 32.");
            break;
          case kTIMESTAMP:
          case kDATE:
          case kTIME:
            if (compression->get_encoding_param() != 32)
              throw std::runtime_error(
                  cd.columnName + ": Compression parameter for Fixed encoding on TIME, DATE or TIMESTAMP must 32.");
            break;
          default:
            throw std::runtime_error(cd.columnName + ": Cannot apply FIXED encoding to " + t->to_string());
        }
        cd.columnType.set_compression(kENCODING_FIXED);
        cd.columnType.set_comp_param(compression->get_encoding_param());
      } else if (boost::iequals(comp, "rl")) {
        // run length encoding
        cd.columnType.set_compression(kENCODING_RL);
        cd.columnType.set_comp_param(0);
        // throw std::runtime_error("RL(Run Length) encoding not supported yet.");
      } else if (boost::iequals(comp, "diff")) {
        // differential encoding
        cd.columnType.set_compression(kENCODING_DIFF);
        cd.columnType.set_comp_param(0);
        // throw std::runtime_error("DIFF(differential) encoding not supported yet.");
      } else if (boost::iequals(comp, "dict")) {
        if (!cd.columnType.is_string() && !cd.columnType.is_string_array())
          throw std::runtime_error(cd.columnName +
                                   ": Dictionary encoding is only supported on string or string array columns.");
        if (compression->get_encoding_param() == 0)
          comp_param = 32;  // default to 32-bits
        else
          comp_param = compression->get_encoding_param();
        if (cd.columnType.is_string_array() && comp_param != 32) {
          throw std::runtime_error(cd.columnName + ": Compression parameter for string arrays must be 32");
        }
        if (comp_param != 8 && comp_param != 16 && comp_param != 32)
          throw std::runtime_error(cd.columnName +
                                   ": Compression parameter for Dictionary encoding must be 8 or 16 or 32.");
        // diciontary encoding
        cd.columnType.set_compression(kENCODING_DICT);
        cd.columnType.set_comp_param(comp_param);
      } else if (boost::iequals(comp, "NONE")) {
        if (!cd.columnType.is_string() && !cd.columnType.is_string_array())
          throw std::runtime_error(cd.columnName +
                                   ": None encoding is only supported on string or string array columns.");
        cd.columnType.set_compression(kENCODING_NONE);
        cd.columnType.set_comp_param(0);
      } else if (boost::iequals(comp, "sparse")) {
        // sparse column encoding with mostly NULL values
        if (cd.columnType.get_notnull())
          throw std::runtime_error(cd.columnName + ": Cannot do sparse column encoding on a NOT NULL column.");
        if (compression->get_encoding_param() == 0 || compression->get_encoding_param() % 8 != 0 ||
            compression->get_encoding_param() > 48)
          throw std::runtime_error(cd.columnName +
                                   "Must specify number of bits as 8, 16, 24, 32 or 48 as the parameter to "
                                   "sparse-column encoding.");
        cd.columnType.set_compression(kENCODING_SPARSE);
        cd.columnType.set_comp_param(compression->get_encoding_param());
        // throw std::runtime_error("SPARSE encoding not supported yet.");
      } else
        throw std::runtime_error(cd.columnName + ": Invalid column compression scheme " + comp);
    }
    if (cd.columnType.is_string_array() && cd.columnType.get_compression() != kENCODING_DICT)
      throw std::runtime_error(cd.columnName + ": Array of strings must be dictionary encoded. Specify ENCODING DICT");
    cd.columnType.set_fixed_size();
    cd.isSystemCol = false;
    cd.isVirtualCol = false;
    columns.push_back(cd);
  }

  TableDescriptor td;
  td.tableName = *table;
  td.nColumns = columns.size();
  td.isView = false;
  td.fragmenter = nullptr;
  td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
  td.maxFragRows = DEFAULT_FRAGMENT_ROWS;
  td.maxChunkSize = DEFAULT_MAX_CHUNK_SIZE;
  td.fragPageSize = DEFAULT_PAGE_SIZE;
  td.maxRows = DEFAULT_MAX_ROWS;
  if (shard_key_def) {
    td.shardedColumnId = shard_column_index(shard_key_def->get_column(), columns);
    if (!td.shardedColumnId) {
      throw std::runtime_error("Specified shard column " + shard_key_def->get_column() + " doesn't exist");
    }
    validate_shard_column_type(td.shardedColumnId, columns);
  }
  if (is_temporary_)
    td.persistenceLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
  else
    td.persistenceLevel = Data_Namespace::MemoryLevel::DISK_LEVEL;
  if (!storage_options.empty()) {
    for (auto& p : storage_options) {
      if (boost::iequals(*p->get_name(), "fragment_size")) {
        if (!dynamic_cast<const IntLiteral*>(p->get_value()))
          throw std::runtime_error("FRAGMENT_SIZE must be an integer literal.");
        int frag_size = static_cast<const IntLiteral*>(p->get_value())->get_intval();
        if (frag_size <= 0)
          throw std::runtime_error("FRAGMENT_SIZE must be a positive number.");
        td.maxFragRows = frag_size;
      } else if (boost::iequals(*p->get_name(), "max_chunk_size")) {
        if (!dynamic_cast<const IntLiteral*>(p->get_value()))
          throw std::runtime_error("MAX_CHUNK_SIZE must be an integer literal.");
        int64_t max_chunk_size = static_cast<const IntLiteral*>(p->get_value())->get_intval();
        if (max_chunk_size <= 0)
          throw std::runtime_error("MAX_CHUNK_SIZE must be a positive number.");
        td.maxChunkSize = max_chunk_size;
      } else if (boost::iequals(*p->get_name(), "page_size")) {
        if (!dynamic_cast<const IntLiteral*>(p->get_value()))
          throw std::runtime_error("PAGE_SIZE must be an integer literal.");
        int page_size = static_cast<const IntLiteral*>(p->get_value())->get_intval();
        if (page_size <= 0)
          throw std::runtime_error("PAGE_SIZE must be a positive number.");
        td.fragPageSize = page_size;
      } else if (boost::iequals(*p->get_name(), "max_rows")) {
        auto max_rows = static_cast<const IntLiteral*>(p->get_value())->get_intval();
        if (max_rows <= 0) {
          throw std::runtime_error("MAX_ROWS must be a positive number.");
        }
        td.maxRows = max_rows;
      } else if (boost::iequals(*p->get_name(), "partitions")) {
        const auto partitions = static_cast<const StringLiteral*>(p->get_value())->get_stringval();
        CHECK(partitions);
        const auto partitions_uc = boost::to_upper_copy<std::string>(*partitions);
        if (partitions_uc != "SHARDED" && partitions_uc != "REPLICATED") {
          throw std::runtime_error("PARTITIONS must be SHARDED or REPLICATED");
        }
        if (shard_key_def && partitions_uc == "REPLICATED") {
          throw std::runtime_error("A table cannot be sharded and replicated at the same time");
        }
        td.partitions = partitions_uc;
      } else if (boost::iequals(*p->get_name(), "shard_count")) {
        if (!dynamic_cast<const IntLiteral*>(p->get_value()))
          throw std::runtime_error("SHARD_COUNT must be an integer literal.");
        int shard_count = static_cast<const IntLiteral*>(p->get_value())->get_intval();
        if (shard_count <= 0)
          throw std::runtime_error("SHARD_COUNT must be a positive number.");
        if (g_leaf_count) {
          if (shard_count % g_leaf_count) {
            throw std::runtime_error("SHARD_COUNT must be a multiple of the number of leaves in the cluster.");
          }
          td.nShards = shard_count / g_leaf_count;
        } else {
          td.nShards = shard_count;
        }
      } else {
        throw std::runtime_error("Invalid CREATE TABLE option " + *p->get_name() +
                                 ".  Should be FRAGMENT_SIZE, PAGE_SIZE, MAX_ROWS, PARTITIONS or SHARD_COUNT.");
      }
    }
  }
  if (shard_key_def && !td.nShards) {
    throw std::runtime_error("Must specify the number of shards through the SHARD_COUNT option");
  }
  td.keyMetainfo = serialize_key_metainfo(shard_key_def, shared_dict_defs);
  catalog.createShardedTable(td, columns, shared_dict_defs);

  if (catalog.isAccessPrivCheckEnabled()) {
    auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
    syscat.createDBObject(session.get_currentUser(), td.tableName, catalog);
  }
}

std::shared_ptr<ResultSet> getResultRows(const Catalog_Namespace::SessionInfo& session,
                                         const std::string select_stmt,
                                         std::vector<TargetMetaInfo>& targets) {
  auto& catalog = session.get_catalog();

  auto executor = Executor::getExecutor(catalog.get_currentDB().dbId);

#ifdef HAVE_CUDA
  const auto device_type = session.get_executor_device_type();
#else
  const auto device_type = ExecutorDeviceType::CPU;
#endif  // HAVE_CUDA
  auto& calcite_mgr = catalog.get_calciteMgr();

  const auto query_ra = calcite_mgr.process(session, pg_shim(select_stmt), true, false);
  CompilationOptions co = {device_type, true, ExecutorOptLevel::LoopStrengthReduction, false};
  ExecutionOptions eo = {false, true, false, true, false, false, false, false, 10000};
  RelAlgExecutor ra_executor(executor.get(), catalog);
  ExecutionResult result{
      std::make_shared<ResultSet>(
          std::vector<TargetInfo>{}, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, nullptr),
      {}};
  result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
  targets = result.getTargetsMeta();

  return result.getRows();
}

namespace {

template <class T>
void translate_strings(T* dest_ids_buffer,
                       StringDictionary* dest_dict,
                       const T* source_ids_buffer,
                       const StringDictionary* source_dict,
                       const size_t num_rows,
                       const SQLTypeInfo& ti) {
  const auto source_ids = reinterpret_cast<const T*>(source_ids_buffer);
  auto dest_ids = reinterpret_cast<T*>(dest_ids_buffer);
  for (size_t i = 0; i < num_rows; ++i) {
    if (source_ids[i] != inline_fixed_encoding_null_val(ti)) {
      dest_ids[i] = dest_dict->getOrAdd(source_dict->getString(source_ids[i]));
    } else {
      dest_ids[i] = source_ids[i];
    }
  }
}

int8_t* fill_dict_column(std::vector<std::unique_ptr<int8_t[]>>& dest_string_ids_owner,
                         StringDictionary* dest_dict,
                         const int8_t* source_ids_buffer,
                         const StringDictionary* source_dict,
                         const size_t num_rows,
                         const SQLTypeInfo& ti) {
  dest_string_ids_owner.emplace_back(new int8_t[num_rows * ti.get_size()]);
  switch (ti.get_size()) {
    case 1:
      translate_strings(reinterpret_cast<uint8_t*>(dest_string_ids_owner.back().get()),
                        dest_dict,
                        reinterpret_cast<const uint8_t*>(source_ids_buffer),
                        source_dict,
                        num_rows,
                        ti);
      break;
    case 2:
      translate_strings(reinterpret_cast<uint16_t*>(dest_string_ids_owner.back().get()),
                        dest_dict,
                        reinterpret_cast<const uint16_t*>(source_ids_buffer),
                        source_dict,
                        num_rows,
                        ti);
      break;
    case 4: {
      translate_strings(reinterpret_cast<int32_t*>(dest_string_ids_owner.back().get()),
                        dest_dict,
                        reinterpret_cast<const int32_t*>(source_ids_buffer),
                        source_dict,
                        num_rows,
                        ti);
      break;
    }
    default:
      CHECK(false);
  }
  return dest_string_ids_owner.back().get();
}

}  // namespace

void CreateTableAsSelectStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  if (g_cluster) {
    throw std::runtime_error("Distributed CTAS not supported yet");
  }
  auto& catalog = session.get_catalog();

  // check access privileges
  if (!session.checkDBAccessPrivileges({false, false, true})) {  // CREATE
    throw std::runtime_error("CTAS failed. Table " + table_name_ +
                             " will not be created. User has no create privileges.");
  }

  if (catalog.getMetadataForTable(table_name_) != nullptr) {
    throw std::runtime_error("Table " + table_name_ + " already exists.");
  }
  std::vector<TargetMetaInfo> target_metainfos;
  const auto result_rows = getResultRows(session, select_query_, target_metainfos);
  std::list<ColumnDescriptor> column_descriptors;
  std::vector<SQLTypeInfo> logical_column_types;
  for (const auto& target_metainfo : target_metainfos) {
    ColumnDescriptor cd;
    cd.columnName = target_metainfo.get_resname();
    cd.columnType = target_metainfo.get_type_info();
    if (cd.columnType.get_size() < 0) {
      throw std::runtime_error("Variable-length type " + cd.columnType.get_type_name() + " not supported in CTAS");
    }
    if (cd.columnType.get_compression() == kENCODING_FIXED) {
      throw std::runtime_error("Fixed encoding integers not supported in CTAS");
    }
    if (cd.columnType.get_compression() == kENCODING_DICT) {
      cd.columnType.set_comp_param(cd.columnType.get_size() * 8);
      logical_column_types.push_back(target_metainfo.get_type_info());
    } else {
      logical_column_types.push_back(get_logical_type_info(target_metainfo.get_type_info()));
    }
    column_descriptors.push_back(cd);
  }
  TableDescriptor td;
  td.tableName = table_name_;
  td.nColumns = column_descriptors.size();
  td.isView = false;
  td.fragmenter = nullptr;
  td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
  td.maxFragRows = DEFAULT_FRAGMENT_ROWS;
  td.maxChunkSize = DEFAULT_MAX_CHUNK_SIZE;
  td.fragPageSize = DEFAULT_PAGE_SIZE;
  td.maxRows = DEFAULT_MAX_ROWS;
  td.keyMetainfo = "[]";
  if (is_temporary_) {
    td.persistenceLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
  } else {
    td.persistenceLevel = Data_Namespace::MemoryLevel::DISK_LEVEL;
  }
  catalog.createTable(td, column_descriptors, {}, true);
  if (result_rows->definitelyHasNoRows()) {
    return;
  }
  const TableDescriptor* created_td{nullptr};
  try {
    const auto row_set_mem_owner = result_rows->getRowSetMemOwner();
    ColumnarResults columnar_results(row_set_mem_owner, *result_rows, column_descriptors.size(), logical_column_types);
    Fragmenter_Namespace::InsertData insert_data;
    insert_data.databaseId = catalog.get_currentDB().dbId;
    created_td = catalog.getMetadataForTable(table_name_);
    CHECK(created_td);
    insert_data.tableId = created_td->tableId;
    std::vector<int> column_ids;
    const auto& column_buffers = columnar_results.getColumnBuffers();
    CHECK_EQ(column_descriptors.size(), column_buffers.size());
    size_t col_idx = 0;
    std::vector<std::unique_ptr<int8_t[]>> dest_string_ids_owner;
    for (const auto cd : column_descriptors) {
      DataBlockPtr p;
      const auto created_cd = catalog.getMetadataForColumn(insert_data.tableId, cd.columnName);
      CHECK(created_cd);
      column_ids.push_back(created_cd->columnId);
      if (created_cd->columnType.get_compression() == kENCODING_DICT) {
        const auto source_dd = catalog.getMetadataForDict(target_metainfos[col_idx].get_type_info().get_comp_param());
        CHECK(source_dd);
        const auto dest_dd = catalog.getMetadataForDict(created_cd->columnType.get_comp_param());
        CHECK(dest_dd);
        const auto dest_ids = fill_dict_column(dest_string_ids_owner,
                                               dest_dd->stringDict.get(),
                                               column_buffers[col_idx],
                                               source_dd->stringDict.get(),
                                               columnar_results.size(),
                                               created_cd->columnType);
        p.numbersPtr = reinterpret_cast<int8_t*>(dest_ids);
      } else {
        p.numbersPtr = const_cast<int8_t*>(column_buffers[col_idx]);
      }
      insert_data.data.push_back(p);
      ++col_idx;
    }
    insert_data.columnIds = column_ids;
    insert_data.numRows = columnar_results.size();
    created_td->fragmenter->insertData(insert_data);
  } catch (...) {
    if (created_td) {
      catalog.dropTable(created_td);
    }
    throw;
  }
  if (catalog.isAccessPrivCheckEnabled()) {
    auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
    syscat.createDBObject(session.get_currentUser(), td.tableName, catalog);
  }
}

void DropTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();

  // check access privileges
  if (!session.checkDBAccessPrivileges({false, false, true})) {  // CREATE, which is being used for DROP as well
    throw std::runtime_error("Table " + *table + " will not be dropped. User has no proper privileges.");
  }

  const TableDescriptor* td = catalog.getMetadataForTable(*table);
  if (td == nullptr) {
    if (if_exists)
      return;
    throw std::runtime_error("Table " + *table + " does not exist.");
  }
  if (td->isView)
    throw std::runtime_error(*table + " is a view.  Use DROP VIEW.");
  catalog.dropTable(td);
}

void TruncateTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  const TableDescriptor* td = catalog.getMetadataForTable(*table);
  if (td == nullptr) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  }

  // check access privileges
  if (catalog.isAccessPrivCheckEnabled()) {
    std::vector<DBObject> privObjects;
    DBObject dbObject(*table, TableDBObjectType);
    static_cast<Catalog_Namespace::SysCatalog&>(catalog).populateDBObjectKey(dbObject, catalog);
    std::vector<bool> privs{false, false, false, true};  // TRUNCATE
    dbObject.setPrivileges(privs);
    privObjects.push_back(dbObject);
    if (!(static_cast<Catalog_Namespace::SysCatalog&>(catalog))
             .checkPrivileges(session.get_currentUser(), privObjects)) {
      throw std::runtime_error("Table " + *table + " will not be trancated. User " +
                               session.get_currentUser().userName + " has no proper privileges.");
    }
  }

  if (td->isView)
    throw std::runtime_error(*table + " is a view.  Cannot Truncate.");
  catalog.truncateTable(td);
}

void RenameTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  const TableDescriptor* td = catalog.getMetadataForTable(*table);

  if (td == nullptr) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  }
  if (catalog.getMetadataForTable(*new_table_name) != nullptr) {
    throw std::runtime_error("Table or View " + *new_table_name + " already exists.");
  }
  catalog.renameTable(td, *new_table_name);
}

void RenameColumnStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  const TableDescriptor* td = catalog.getMetadataForTable(*table);
  if (td == nullptr) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  }
  const ColumnDescriptor* cd = catalog.getMetadataForColumn(td->tableId, *column);
  if (cd == nullptr) {
    throw std::runtime_error("Column " + *column + " does not exist.");
  }
  if (catalog.getMetadataForColumn(td->tableId, *new_column_name) != nullptr) {
    throw std::runtime_error("Column " + *new_column_name + " already exists.");
  }
  if (reserved_keywords.find(boost::to_upper_copy<std::string>(*new_column_name)) != reserved_keywords.end()) {
    throw std::runtime_error("Cannot create column with reserved keyword '" + *new_column_name + "'");
  }
  catalog.renameColumn(td, cd, *new_column_name);
}

void CopyTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto importer_factory = [](const Catalog_Namespace::Catalog& catalog,
                             const TableDescriptor* td,
                             const std::string& file_path,
                             const Importer_NS::CopyParams& copy_params) {
    return boost::make_unique<Importer_NS::Importer>(catalog, td, file_path, copy_params);
  };
  return execute(session, importer_factory);
}

void CopyTableStmt::execute(
    const Catalog_Namespace::SessionInfo& session,
    const std::function<std::unique_ptr<Importer_NS::Importer>(const Catalog_Namespace::Catalog&,
                                                               const TableDescriptor*,
                                                               const std::string&,
                                                               const Importer_NS::CopyParams&)>& importer_factory) {
  size_t rows_completed = 0;
  size_t rows_rejected = 0;
  size_t total_time = 0;
  bool load_truncated = false;

  auto& catalog = session.get_catalog();
  const TableDescriptor* td = catalog.getMetadataForTable(*table);
  if (td == nullptr)
    throw std::runtime_error("Table " + *table + " does not exist.");

  // check access privileges
  if (catalog.isAccessPrivCheckEnabled()) {
    std::vector<DBObject> privObjects;
    DBObject dbObject(*table, TableDBObjectType);
    static_cast<Catalog_Namespace::SysCatalog&>(catalog).populateDBObjectKey(dbObject, catalog);
    std::vector<bool> privs{false, true, false, false};  // INSERT
    dbObject.setPrivileges(privs);
    privObjects.push_back(dbObject);
    if (!(static_cast<Catalog_Namespace::SysCatalog&>(catalog))
             .checkPrivileges(session.get_currentUser(), privObjects)) {
      throw std::runtime_error("Violation of access privileges: user " + session.get_currentUser().userName +
                               " has no insert privileges for table " + *table + ".");
    }
  }

  auto file_paths = mapd_glob(*file_pattern);
  if (file_paths.size() == 0)
    throw std::runtime_error("File " + *file_pattern + " does not exist.");
  for (auto file_path : file_paths) {
    if (!boost::filesystem::exists(file_path))
      throw std::runtime_error("File " + file_path + " does not exist.");
    Importer_NS::CopyParams copy_params;
    if (!options.empty()) {
      for (auto& p : options) {
        if (boost::iequals(*p->get_name(), "max_reject")) {
          const IntLiteral* int_literal = dynamic_cast<const IntLiteral*>(p->get_value());
          if (int_literal == nullptr)
            throw std::runtime_error("max_reject option must be an integer.");
          copy_params.max_reject = int_literal->get_intval();
        } else if (boost::iequals(*p->get_name(), "threads")) {
          const IntLiteral* int_literal = dynamic_cast<const IntLiteral*>(p->get_value());
          if (int_literal == nullptr)
            throw std::runtime_error("Threads option must be an integer.");
          copy_params.threads = int_literal->get_intval();
        } else if (boost::iequals(*p->get_name(), "delimiter")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Delimiter option must be a string.");
          else if (str_literal->get_stringval()->length() != 1)
            throw std::runtime_error("Delimiter must be a single character string.");
          copy_params.delimiter = (*str_literal->get_stringval())[0];
        } else if (boost::iequals(*p->get_name(), "nulls")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Nulls option must be a string.");
          copy_params.null_str = *str_literal->get_stringval();
        } else if (boost::iequals(*p->get_name(), "header")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Header option must be a boolean.");
          const std::string* s = str_literal->get_stringval();
          if (*s == "t" || *s == "true" || *s == "T" || *s == "True")
            copy_params.has_header = true;
          else if (*s == "f" || *s == "false" || *s == "F" || *s == "False")
            copy_params.has_header = false;
          else
            throw std::runtime_error("Invalid string for boolean " + *s);
        } else if (boost::iequals(*p->get_name(), "quote")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Quote option must be a string.");
          else if (str_literal->get_stringval()->length() != 1)
            throw std::runtime_error("Quote must be a single character string.");
          copy_params.quote = (*str_literal->get_stringval())[0];
        } else if (boost::iequals(*p->get_name(), "escape")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Escape option must be a string.");
          else if (str_literal->get_stringval()->length() != 1)
            throw std::runtime_error("Escape must be a single character string.");
          copy_params.escape = (*str_literal->get_stringval())[0];
        } else if (boost::iequals(*p->get_name(), "line_delimiter")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Line_delimiter option must be a string.");
          else if (str_literal->get_stringval()->length() != 1)
            throw std::runtime_error("Line_delimiter must be a single character string.");
          copy_params.line_delim = (*str_literal->get_stringval())[0];
        } else if (boost::iequals(*p->get_name(), "quoted")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Quoted option must be a boolean.");
          const std::string* s = str_literal->get_stringval();
          if (*s == "t" || *s == "true" || *s == "T" || *s == "True")
            copy_params.quoted = true;
          else if (*s == "f" || *s == "false" || *s == "F" || *s == "False")
            copy_params.quoted = false;
          else
            throw std::runtime_error("Invalid string for boolean " + *s);
        } else if (boost::iequals(*p->get_name(), "array")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Array option must be a string.");
          else if (str_literal->get_stringval()->length() != 2)
            throw std::runtime_error("Array option must be exactly two characters.  Default is {}.");
          copy_params.array_begin = (*str_literal->get_stringval())[0];
          copy_params.array_end = (*str_literal->get_stringval())[1];
        } else if (boost::iequals(*p->get_name(), "array_delimiter")) {
          const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
          if (str_literal == nullptr)
            throw std::runtime_error("Array Delimiter option must be a string.");
          else if (str_literal->get_stringval()->length() != 1)
            throw std::runtime_error("Array Delimiter must be a single character string.");
          copy_params.array_delim = (*str_literal->get_stringval())[0];
        } else
          throw std::runtime_error("Invalid option for COPY: " + *p->get_name());
      }
    }
    auto importer = importer_factory(catalog, td, file_path, copy_params);

    auto ms = measure<>::execution([&]() {
      auto res = importer->import();
      rows_completed += res.rows_completed;
      rows_rejected += res.rows_rejected;
      load_truncated = res.load_truncated;
    });
    total_time += ms;
    if (load_truncated || rows_rejected > copy_params.max_reject) {
      LOG(ERROR) << "COPY exited early due to reject records count during multi file processing ";
      // if we have crossed the truncated load threshold
      load_truncated = true;
      break;
    }
  }
  std::string tr;
  if (!load_truncated) {
    tr = std::string("Loaded: " + std::to_string(rows_completed) + " recs, Rejected: " + std::to_string(rows_rejected) +
                     " recs in " + std::to_string((double)total_time / 1000.0) + " secs");
  } else {
    tr = std::string("Loader truncated due to reject count.  Processed : " + std::to_string(rows_completed) +
                     " recs, Rejected: " + std::to_string(rows_rejected) + " recs in " +
                     std::to_string((double)total_time / 1000.0) + " secs");
  }

  return_message.reset(new std::string(tr));
  LOG(INFO) << tr;
}

// CREATE ROLE payroll_dept_role;
void CreateRoleStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (!catalog.isAccessPrivCheckEnabled()) {
    throw std::runtime_error(
        "CREATE ROLE " + get_role() +
        " failed. This command may be executed only when DB object level access privileges check turned on.");
  }

  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB) {
    throw std::runtime_error("CREATE ROLE " + get_role() + " failed. Must be in the system database ('" +
                             std::string(MAPD_SYSTEM_DB) + "') to create roles.");
  }
  const auto& currentUser = session.get_currentUser();
  if (!currentUser.isSuper) {
    throw std::runtime_error("CREATE ROLE " + get_role() + " failed. It can only be executed by super user.");
  }
  const auto role_name = boost::to_lower_copy<std::string>(get_role());
  if (!role_name.compare(MAPD_DEFAULT_ROOT_USER_ROLE) || !role_name.compare(MAPD_DEFAULT_USER_ROLE)) {
    throw std::runtime_error("CREATE ROLE " + get_role() +
                             " failed because this role name is reserved and can't be used.");
  }

  auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  syscat.createRole(get_role());
}

// DROP ROLE payroll_dept_role;
void DropRoleStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (!catalog.isAccessPrivCheckEnabled()) {
    throw std::runtime_error(
        "DROP ROLE " + get_role() +
        " failed. This command may be executed only when DB object level access privileges check turned on.");
  }

  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB) {
    throw std::runtime_error("DROP ROLE " + get_role() + " failed. Must be in the system database ('" +
                             std::string(MAPD_SYSTEM_DB) + "') to drop roles.");
  }
  const auto& currentUser = session.get_currentUser();
  if (!currentUser.isSuper) {
    throw std::runtime_error("DROP ROLE " + get_role() + "failed. It can only be executed by super user.");
  }

  auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  if (syscat.getMetadataForRole(get_role()) == nullptr) {
    throw std::runtime_error("DROP ROLE " + get_role() + " failed because role with this name does not exist.");
  }
  syscat.dropRole(get_role());
}

std::vector<std::string> splitObjectHierName(const std::string& hierName) {
  std::vector<std::string> componentNames;
  boost::split(componentNames, hierName, boost::is_any_of("."));
  return componentNames;
}

std::string extractObjectNameFromHierName(const std::string& objectHierName,
                                          const std::string& objectType,
                                          const Catalog_Namespace::Catalog& cat) {
  std::string objectName;
  std::vector<std::string> componentNames = splitObjectHierName(objectHierName);
  if (objectType.compare("DATABASE") == 0) {
    if (componentNames.size() == 1) {
      objectName = componentNames[0];
    } else {
      throw std::runtime_error("DB object name is not correct " + objectHierName);
    }
  } else {
    if (objectType.compare("TABLE") == 0) {
      switch (componentNames.size()) {
        case (1): {
          objectName = componentNames[0];
          break;
        }
        case (2): {
          objectName = componentNames[1];
          break;
        }
        default: { throw std::runtime_error("DB object name is not correct " + objectHierName); }
      }
    } else {
      throw std::runtime_error("DB object type " + objectType + " is not supported.");
    }
  }
  return objectName;
}

// GRANT SELECT/INSERT/CREATE ON TABLE payroll_table TO payroll_dept_role;
void GrantPrivilegesStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (!catalog.isAccessPrivCheckEnabled()) {
    throw std::runtime_error(
        "GRANT " + get_priv() +
        " failed. This command may be executed only when DB object level access privileges check turned on.");
  }

  const auto& currentUser = session.get_currentUser();
  auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  const auto parserObjectType = boost::to_upper_copy<std::string>(get_object_type());
  const auto objectName = extractObjectNameFromHierName(get_object(), parserObjectType, catalog);
  DBObjectType objectType = TableDBObjectType;
  if (parserObjectType.compare("DATABASE") == 0) {
    objectType = DatabaseDBObjectType;
  } else {
    if (parserObjectType.compare("TABLE") != 0) {
      throw std::runtime_error("DB object type " + parserObjectType + " is not supported.");
    }
  }
  DBObject dbObject(objectName, objectType);

  /* verify object ownership if not suser */
  if (!currentUser.isSuper) {
    if (!syscat.verifyDBObjectOwnership(currentUser, dbObject, catalog)) {
      throw std::runtime_error("GRANT " + get_priv() +
                               " failed. It can only be executed by super user or owner of the object.");
    }
  }

  /* set proper values of privileges & grant them to the object */
  std::vector<bool> privs;
  const auto priv = boost::to_upper_copy<std::string>(get_priv());
  if (priv.compare("ALL") == 0) {
    if (objectType == DatabaseDBObjectType) {
      privs = {true, true, true, true};
    } else {  // TableDBObjectType
      privs = {true, true, false, true};
    }
  } else {
    if (priv.compare("SELECT") == 0) {
      privs = {true, false, false, false};
    } else {
      if (priv.compare("INSERT") == 0) {
        privs = {false, true, false, false};
      } else {
        if ((priv.compare("CREATE") == 0) &&
            (objectType == DatabaseDBObjectType)) {  // CREATE privs allowed only on DB level
          privs = {false, false, true, false};
        } else {
          if (priv.compare("TRUNCATE") == 0) {
            privs = {false, false, false, true};
          } else {
            throw std::runtime_error("Privileges " + get_priv() + " being granted to DB object " + get_object() +
                                     " are not correct.");
          }
        }
      }
    }
  }
  dbObject.setPrivileges(privs);
  syscat.grantDBObjectPrivileges(get_role(), dbObject, catalog);
}

// REVOKE SELECT/INSERT/CREATE ON TABLE payroll_table FROM payroll_dept_role;
void RevokePrivilegesStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (!catalog.isAccessPrivCheckEnabled()) {
    throw std::runtime_error(
        "REVOKE " + get_priv() +
        " failed. This command may be executed only when DB object level access privileges check turned on.");
  }

  const auto& currentUser = session.get_currentUser();
  auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  const auto parserObjectType = boost::to_upper_copy<std::string>(get_object_type());
  const auto objectName = extractObjectNameFromHierName(get_object(), parserObjectType, catalog);
  DBObjectType objectType = TableDBObjectType;
  if (parserObjectType.compare("DATABASE") == 0) {
    objectType = DatabaseDBObjectType;
  } else {
    if (parserObjectType.compare("TABLE") != 0) {
      throw std::runtime_error("DB object type " + parserObjectType + " is not supported.");
    }
  }
  DBObject dbObject(objectName, objectType);

  /* verify object ownership if not suser */
  if (!currentUser.isSuper) {
    if (!syscat.verifyDBObjectOwnership(currentUser, dbObject, catalog)) {
      throw std::runtime_error("REVOKE " + get_priv() +
                               " failed. It can only be executed by super user or owner of the object.");
    }
  }

  /* set proper values of privileges & revoke them from the object */
  std::vector<bool> privs;
  const auto priv = boost::to_upper_copy<std::string>(get_priv());
  if (priv.compare("ALL") == 0) {
    if (objectType == DatabaseDBObjectType) {
      privs = {true, true, true, true};
    } else {  // TableDBObjectType
      privs = {true, true, false, true};
    }
  } else {
    if (priv.compare("SELECT") == 0) {
      privs = {true, false, false, false};
    } else {
      if (priv.compare("INSERT") == 0) {
        privs = {false, true, false, false};
      } else {
        if ((priv.compare("CREATE") == 0) &&
            (objectType == DatabaseDBObjectType)) {  // CREATE privs allowed only on DB level
          privs = {false, false, true, false};
        } else {
          if (priv.compare("TRUNCATE") == 0) {
            privs = {false, false, false, true};
          } else {
            throw std::runtime_error("Privileges " + get_priv() + " being revoked from DB object " + get_object() +
                                     " are not correct.");
          }
        }
      }
    }
  }
  dbObject.setPrivileges(privs);
  syscat.revokeDBObjectPrivileges(get_role(), dbObject, catalog);
}

// SHOW ON TABLE payroll_table FOR payroll_dept_role;
void ShowPrivilegesStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (!catalog.isAccessPrivCheckEnabled()) {
    throw std::runtime_error(
        "SHOW ON " + get_object() + " FOR " + get_role() +
        " failed. This command may be executed only when DB object level access privileges check turned on.");
  }

  const auto& currentUser = session.get_currentUser();
  auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  const auto parserObjectType = boost::to_upper_copy<std::string>(get_object_type());
  const auto objectName = extractObjectNameFromHierName(get_object(), parserObjectType, catalog);
  DBObjectType objectType = TableDBObjectType;
  if (parserObjectType.compare("DATABASE") == 0) {
    objectType = DatabaseDBObjectType;
  } else {
    if (parserObjectType.compare("TABLE") != 0) {
      throw std::runtime_error("DB object type " + parserObjectType + " is not supported.");
    }
  }
  DBObject dbObject(objectName, objectType);

  /* verify object ownership if not suser */
  if (!currentUser.isSuper) {
    if (!syscat.verifyDBObjectOwnership(currentUser, dbObject, catalog)) {
      throw std::runtime_error("SHOW ON " + get_object() + " FOR " + get_role() +
                               " failed. It can only be executed by super user or owner of the object.");
    }
  }

  /* get values of privileges for the object and report them */
  syscat.getDBObjectPrivileges(get_role(), dbObject, catalog);
  std::vector<bool> privs = dbObject.getPrivileges();
  printf("\nPRIVILEGES ON %s FOR %s ARE SET AS FOLLOWING: ", get_object().c_str(), get_role().c_str());
  size_t i = 0;
  while (i < privs.size()) {
    if (privs[i]) {
      switch (i) {
        case 0: {
          printf(" SELECT");
          break;
        }
        case 1: {
          printf(" INSERT");
          break;
        }
        case 2: {
          printf(" CREATE");
          break;
        }
        case 3: {
          printf(" TRUNCATE");
          break;
        }
        default: { CHECK(false); }
      }
    }
    i++;
  }
  printf(".\n");
}

// GRANT payroll_dept_role TO joe;
void GrantRoleStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (!catalog.isAccessPrivCheckEnabled()) {
    throw std::runtime_error(
        "GRANT " + get_role() + " TO " + get_user() +
        " failed. This command may be executed only when DB object level access privileges check turned on.");
  }

  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB) {
    throw std::runtime_error("GRANT " + get_role() + " TO " + get_user() +
                             "; failed. Must be in the system database ('" + std::string(MAPD_SYSTEM_DB) +
                             "') to grant roles.");
  }
  const auto& currentUser = session.get_currentUser();
  if (!currentUser.isSuper) {
    throw std::runtime_error("GRANT " + get_role() + " TO " + get_user() +
                             "; failed, because it can only be executed by super user.");
  }
  auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  if (!get_user().compare(MAPD_ROOT_USER)) {
    throw std::runtime_error("Request to grant role " + get_role() +
                             " failed because mapd root user has all privileges by default.");
  }
  syscat.grantRole(get_role(), get_user());
}

// REVOKE payroll_dept_role FROM joe;
void RevokeRoleStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (!catalog.isAccessPrivCheckEnabled()) {
    throw std::runtime_error(
        "REVOKE " + get_role() + " FROM " + get_user() +
        " failed. This command may be executed only when DB object level access privileges check turned on.");
  }

  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB) {
    throw std::runtime_error("REVOKE " + get_role() + " TO " + get_user() +
                             "; failed. Must be in the system database ('" + std::string(MAPD_SYSTEM_DB) +
                             "') to revoke roles.");
  }
  const auto& currentUser = session.get_currentUser();
  if (!currentUser.isSuper) {
    throw std::runtime_error("REVOKE " + get_role() + " FROM " + get_user() +
                             "; failed, because it can only be executed by super user.");
  }
  auto& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  if (!get_user().compare(MAPD_ROOT_USER)) {
    throw std::runtime_error("Request to revoke role " + get_role() +
                             " failed because privileges can not be revoked from mapd root user.");
  }
  syscat.revokeRole(get_role(), get_user());
}

void ExportQueryStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  if (g_cluster) {
    throw std::runtime_error("Distributed export not supported yet");
  }
  auto& catalog = session.get_catalog();
  Importer_NS::CopyParams copy_params;
  if (!options.empty()) {
    for (auto& p : options) {
      if (boost::iequals(*p->get_name(), "delimiter")) {
        const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr)
          throw std::runtime_error("Delimiter option must be a string.");
        else if (str_literal->get_stringval()->length() != 1)
          throw std::runtime_error("Delimiter must be a single character string.");
        copy_params.delimiter = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "nulls")) {
        const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr)
          throw std::runtime_error("Nulls option must be a string.");
        copy_params.null_str = *str_literal->get_stringval();
      } else if (boost::iequals(*p->get_name(), "header")) {
        const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr)
          throw std::runtime_error("Header option must be a boolean.");
        const std::string* s = str_literal->get_stringval();
        if (*s == "t" || *s == "true" || *s == "T" || *s == "True")
          copy_params.has_header = true;
        else if (*s == "f" || *s == "false" || *s == "F" || *s == "False")
          copy_params.has_header = false;
        else
          throw std::runtime_error("Invalid string for boolean " + *s);
      } else if (boost::iequals(*p->get_name(), "quote")) {
        const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr)
          throw std::runtime_error("Quote option must be a string.");
        else if (str_literal->get_stringval()->length() != 1)
          throw std::runtime_error("Quote must be a single character string.");
        copy_params.quote = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "escape")) {
        const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr)
          throw std::runtime_error("Escape option must be a string.");
        else if (str_literal->get_stringval()->length() != 1)
          throw std::runtime_error("Escape must be a single character string.");
        copy_params.escape = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "line_delimiter")) {
        const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr)
          throw std::runtime_error("Line_delimiter option must be a string.");
        else if (str_literal->get_stringval()->length() != 1)
          throw std::runtime_error("Line_delimiter must be a single character string.");
        copy_params.line_delim = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "quoted")) {
        const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr)
          throw std::runtime_error("Quoted option must be a boolean.");
        const std::string* s = str_literal->get_stringval();
        if (*s == "t" || *s == "true" || *s == "T" || *s == "True")
          copy_params.quoted = true;
        else if (*s == "f" || *s == "false" || *s == "F" || *s == "False")
          copy_params.quoted = false;
        else
          throw std::runtime_error("Invalid string for boolean " + *s);
      } else if (boost::iequals(*p->get_name(), "header")) {
        const StringLiteral* str_literal = dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr)
          throw std::runtime_error("Header option must be a boolean.");
        const std::string* s = str_literal->get_stringval();
        if (*s == "t" || *s == "true" || *s == "T" || *s == "True")
          copy_params.has_header = true;
        else if (*s == "f" || *s == "false" || *s == "F" || *s == "False")
          copy_params.has_header = false;
        else
          throw std::runtime_error("Invalid string for boolean " + *s);
      } else
        throw std::runtime_error("Invalid option for COPY: " + *p->get_name());
    }
  }
  std::vector<TargetMetaInfo> targets;
  const auto results = getResultRows(session, *select_stmt, targets);
  TargetMetaInfo* td = targets.data();

  std::ofstream outfile;
  if (file_path->empty() || !boost::filesystem::path(*file_path).is_absolute()) {
    std::string file_name;
    if (file_path->empty()) {
      file_name = session.get_session_id() + ".txt";
    } else {
      file_name = boost::filesystem::path(*file_path).filename().string();
    }
    *file_path = catalog.get_basePath() + "/mapd_export/" + session.get_session_id() + "/";
    if (!boost::filesystem::exists(*file_path))
      if (!boost::filesystem::create_directories(*file_path))
        throw std::runtime_error("Directory " + *file_path + " cannot be created.");
    *file_path += file_name;
  }
  outfile.open(*file_path);
  if (!outfile)
    throw std::runtime_error("Cannot open file: " + *file_path);
  if (copy_params.has_header) {
    bool not_first = false;
    size_t i = 0;
    for (const auto& target : targets) {
      std::string col_name = target.get_resname();
      if (col_name.empty()) {
        col_name = "result_" + std::to_string(i + 1);
      }
      if (not_first)
        outfile << copy_params.delimiter;
      else
        not_first = true;
      outfile << col_name;
      ++i;
    }
    outfile << copy_params.line_delim;
  }
  while (true) {
    const auto crt_row = results->getNextRow(true, true);
    if (crt_row.empty()) {
      break;
    }
    bool not_first = false;
    for (size_t i = 0; i < results->colCount(); ++i) {
      bool is_null;
      const auto tv = crt_row[i];
      const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
      if (not_first)
        outfile << copy_params.delimiter;
      else
        not_first = true;
      if (copy_params.quoted)
        outfile << copy_params.quote;
      const auto& ti = td[i].get_type_info();
      if (!scalar_tv) {
        outfile << datum_to_string(crt_row[i], ti, " | ");
        if (copy_params.quoted) {
          outfile << copy_params.quote;
        }
        continue;
      }
      if (boost::get<int64_t>(scalar_tv)) {
        auto int_val = *(boost::get<int64_t>(scalar_tv));
        switch (ti.get_type()) {
          case kBOOLEAN:
            is_null = (int_val == NULL_BOOLEAN);
            break;
          case kSMALLINT:
            is_null = (int_val == NULL_SMALLINT);
            break;
          case kINT:
            is_null = (int_val == NULL_INT);
            break;
          case kBIGINT:
            is_null = (int_val == NULL_BIGINT);
            break;
          case kTIME:
          case kTIMESTAMP:
          case kDATE:
            if (sizeof(time_t) == 4)
              is_null = (int_val == NULL_INT);
            else
              is_null = (int_val == NULL_BIGINT);
            break;
          default:
            is_null = false;
        }
        if (is_null)
          outfile << copy_params.null_str;
        else if (ti.get_type() == kTIME) {
          time_t t = int_val;
          std::tm tm_struct;
          gmtime_r(&t, &tm_struct);
          char buf[9];
          strftime(buf, 9, "%T", &tm_struct);
          outfile << buf;
        } else
          outfile << int_val;
      } else if (boost::get<double>(scalar_tv)) {
        auto real_val = *(boost::get<double>(scalar_tv));
        if (ti.get_type() == kFLOAT) {
          is_null = (real_val == NULL_FLOAT);
        } else {
          is_null = (real_val == NULL_DOUBLE);
        }
        if (is_null)
          outfile << copy_params.null_str;
        else
          outfile << real_val;
      } else if (boost::get<float>(scalar_tv)) {
        CHECK_EQ(kFLOAT, ti.get_type());
        auto real_val = *(boost::get<float>(scalar_tv));
        if (real_val == NULL_FLOAT)
          outfile << copy_params.null_str;
        else
          outfile << real_val;
      } else {
        auto s = boost::get<NullableString>(scalar_tv);
        is_null = !s || boost::get<void*>(s);
        if (is_null)
          outfile << copy_params.null_str;
        else {
          auto s_notnull = boost::get<std::string>(s);
          CHECK(s_notnull);
          if (!copy_params.quoted)
            outfile << *s_notnull;
          else {
            size_t q = s_notnull->find(copy_params.quote);
            if (q == std::string::npos)
              outfile << *s_notnull;
            else {
              std::string str(*s_notnull);
              while (q != std::string::npos) {
                str.insert(q, 1, copy_params.escape);
                q = str.find(copy_params.quote, q + 2);
              }
              outfile << str;
            }
          }
        }
      }
      if (copy_params.quoted) {
        outfile << copy_params.quote;
      }
    }
    outfile << copy_params.line_delim;
  }
  outfile.close();
}

void CreateViewStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (catalog.getMetadataForTable(view_name_) != nullptr) {
    if (if_not_exists_)
      return;
    throw std::runtime_error("Table or View " + view_name_ + " already exists.");
  }
  const auto query_after_shim = pg_shim(select_query_);
  catalog.get_calciteMgr().process(session, query_after_shim, true, true);
  TableDescriptor td;
  td.tableName = view_name_;
  td.nColumns = 0;
  td.isView = true;
  td.viewSQL = query_after_shim;
  td.fragmenter = nullptr;
  td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
  td.maxFragRows = DEFAULT_FRAGMENT_ROWS;    // @todo this stuff should not be InsertOrderFragmenter
  td.maxChunkSize = DEFAULT_MAX_CHUNK_SIZE;  // @todo this stuff should not be InsertOrderFragmenter
  td.fragPageSize = DEFAULT_PAGE_SIZE;
  td.maxRows = DEFAULT_MAX_ROWS;
  catalog.createTable(td, {}, {}, true);
}

void DropViewStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  const TableDescriptor* td = catalog.getMetadataForTable(*view_name);
  if (td == nullptr) {
    if (if_exists)
      return;
    throw std::runtime_error("View " + *view_name + " does not exist.");
  }
  if (!td->isView)
    throw std::runtime_error(*view_name + " is a table.  Use DROP TABLE.");
  catalog.dropTable(td);
}

void CreateDBStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
    throw std::runtime_error("Must be in the system database ('" + std::string(MAPD_SYSTEM_DB) +
                             "') to create databases.");
  Catalog_Namespace::SysCatalog& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  int ownerId = session.get_currentUser().userId;
  if (!name_value_list.empty()) {
    for (auto& p : name_value_list) {
      if (boost::iequals(*p->get_name(), "owner")) {
        if (!dynamic_cast<const StringLiteral*>(p->get_value()))
          throw std::runtime_error("Owner name must be a string literal.");
        const std::string* str = static_cast<const StringLiteral*>(p->get_value())->get_stringval();
        Catalog_Namespace::UserMetadata user;
        if (!syscat.getMetadataForUser(*str, user))
          throw std::runtime_error("User " + *str + " does not exist.");
        ownerId = user.userId;
      } else
        throw std::runtime_error("Invalid CREATE DATABASE option " + *p->get_name() + ". Only OWNER supported.");
    }
  }
  syscat.createDatabase(*db_name, ownerId);
}

void DropDBStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
    throw std::runtime_error("Must be in the system database ('" + std::string(MAPD_SYSTEM_DB) +
                             "') to drop databases.");
  Catalog_Namespace::SysCatalog& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  Catalog_Namespace::DBMetadata db;
  if (!syscat.getMetadataForDB(*db_name, db))
    throw std::runtime_error("Database " + *db_name + " does not exist.");

  if (!session.get_currentUser().isSuper && session.get_currentUser().userId != db.dbOwner)
    throw std::runtime_error("Only the super user or the owner can drop database.");

  syscat.dropDatabase(db.dbId, *db_name);
}

void CreateUserStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  std::string passwd;
  bool is_super = false;
  for (auto& p : name_value_list) {
    if (boost::iequals(*p->get_name(), "password")) {
      if (!dynamic_cast<const StringLiteral*>(p->get_value()))
        throw std::runtime_error("Password must be a string literal.");
      passwd = *static_cast<const StringLiteral*>(p->get_value())->get_stringval();
    } else if (boost::iequals(*p->get_name(), "is_super")) {
      if (!dynamic_cast<const StringLiteral*>(p->get_value()))
        throw std::runtime_error("IS_SUPER option must be a string literal.");
      const std::string* str = static_cast<const StringLiteral*>(p->get_value())->get_stringval();
      if (boost::iequals(*str, "true"))
        is_super = true;
      else if (boost::iequals(*str, "false"))
        is_super = false;
      else
        throw std::runtime_error("Value to IS_SUPER must be TRUE or FALSE.");
    } else
      throw std::runtime_error("Invalid CREATE USER option " + *p->get_name() + ".  Should be PASSWORD or IS_SUPER.");
  }
  if (passwd.empty())
    throw std::runtime_error("Must have a password for CREATE USER.");
  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
    throw std::runtime_error("Must be in the system database ('" + std::string(MAPD_SYSTEM_DB) + "') to create users.");
  if (!session.get_currentUser().isSuper)
    throw std::runtime_error("Only super user can create new users.");
  Catalog_Namespace::SysCatalog& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  syscat.createUser(*user_name, passwd, is_super);
}

void AlterUserStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (catalog.isAccessPrivCheckEnabled() && !session.get_currentUser().isSuper) {
    throw std::runtime_error("ALTER USER command failed. It can only be executed by super user.");
  }
  const std::string* passwd = nullptr;
  const std::string* insertaccessDB = nullptr;
  bool is_super = false;
  bool* is_superp = nullptr;
  for (auto& p : name_value_list) {
    if (boost::iequals(*p->get_name(), "password")) {
      if (!dynamic_cast<const StringLiteral*>(p->get_value()))
        throw std::runtime_error("Password must be a string literal.");
      passwd = static_cast<const StringLiteral*>(p->get_value())->get_stringval();
    } else if (boost::iequals(*p->get_name(), "insertaccess")) {
      if (!dynamic_cast<const StringLiteral*>(p->get_value()))
        throw std::runtime_error("INSERTACCESS must be a string literal.");
      insertaccessDB = static_cast<const StringLiteral*>(p->get_value())->get_stringval();
    } else if (boost::iequals(*p->get_name(), "is_super")) {
      if (!dynamic_cast<const StringLiteral*>(p->get_value()))
        throw std::runtime_error("IS_SUPER option must be a string literal.");
      const std::string* str = static_cast<const StringLiteral*>(p->get_value())->get_stringval();
      if (boost::iequals(*str, "true")) {
        is_super = true;
        is_superp = &is_super;
      } else if (boost::iequals(*str, "false")) {
        is_super = false;
        is_superp = &is_super;
      } else
        throw std::runtime_error("Value to IS_SUPER must be TRUE or FALSE.");
    } else
      throw std::runtime_error("Invalid CREATE USER option " + *p->get_name() +
                               ".  Should be PASSWORD, INSERTACCESS or IS_SUPER.");
  }
  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
    throw std::runtime_error("Must be in the system database ('" + std::string(MAPD_SYSTEM_DB) + "') to alter users.");
  Catalog_Namespace::SysCatalog& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  Catalog_Namespace::UserMetadata user;
  if (!syscat.getMetadataForUser(*user_name, user))
    throw std::runtime_error("User " + *user_name + " does not exist.");
  if (!session.get_currentUser().isSuper && session.get_currentUser().userId != user.userId)
    throw std::runtime_error("Only user super can change another user's attributes.");
  if (insertaccessDB) {
    LOG(INFO) << " InsertAccess being given to user " << user.userId << " for db " << *insertaccessDB;
    Catalog_Namespace::DBMetadata db;
    if (!syscat.getMetadataForDB(*insertaccessDB, db))
      throw std::runtime_error("Database " + *insertaccessDB + " does not exist.");
    Privileges privs;
    privs.insert_ = true;
    privs.select_ = true;
    syscat.grantPrivileges(user.userId, db.dbId, privs);
  }
  if (passwd || is_superp) {
    syscat.alterUser(user.userId, passwd, is_superp);
  }
}

void DropUserStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.get_catalog();
  if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
    throw std::runtime_error("Must be in the system database ('" + std::string(MAPD_SYSTEM_DB) + "') to drop users.");
  if (!session.get_currentUser().isSuper)
    throw std::runtime_error("Only super user can drop users.");
  Catalog_Namespace::SysCatalog& syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
  syscat.dropUser(*user_name);
}
}
