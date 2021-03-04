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

#include "ParserNode.h"

#include <boost/algorithm/string.hpp>
#include <boost/core/null_deleter.hpp>
#include <boost/filesystem.hpp>
#include <boost/function.hpp>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <regex>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>

#include "Analyzer/RangeTableEntry.h"
#include "Catalog/Catalog.h"
#include "Catalog/DataframeTableDescriptor.h"
#include "Catalog/SharedDictionaryValidator.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Fragmenter/SortedOrderFragmenter.h"
#include "Fragmenter/TargetValueConvertersFactories.h"
#include "Geospatial/Compression.h"
#include "Geospatial/Types.h"
#include "ImportExport/Importer.h"
#include "LockMgr/LockMgr.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/CalciteDeserializerUtils.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "ReservedKeywords.h"
#include "Shared/StringTransform.h"
#include "Shared/measure.h"
#include "Shared/shard_key.h"
#include "Shared/thread_count.h"
#include "TableArchiver/TableArchiver.h"
#include "Utils/FsiUtils.h"

#include "gen-cpp/CalciteServer.h"
#include "parser.h"

bool g_enable_calcite_ddl_parser{true};

size_t g_leaf_count{0};
bool g_test_drop_column_rollback{false};
extern bool g_enable_experimental_string_functions;
extern bool g_enable_fsi;

using Catalog_Namespace::SysCatalog;
using namespace std::string_literals;

using TableDefFuncPtr = boost::function<void(TableDescriptor&,
                                             const NameValueAssign*,
                                             const std::list<ColumnDescriptor>& columns)>;

using DataframeDefFuncPtr =
    boost::function<void(DataframeTableDescriptor&,
                         const NameValueAssign*,
                         const std::list<ColumnDescriptor>& columns)>;

namespace Parser {
std::shared_ptr<Analyzer::Expr> NullLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  return makeExpr<Analyzer::Constant>(kNULLT, true);
}

std::shared_ptr<Analyzer::Expr> StringLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  return analyzeValue(*stringval);
}

std::shared_ptr<Analyzer::Expr> StringLiteral::analyzeValue(
    const std::string& stringval) {
  SQLTypeInfo ti(kVARCHAR, stringval.length(), 0, true);
  Datum d;
  d.stringval = new std::string(stringval);
  return makeExpr<Analyzer::Constant>(ti, false, d);
}

std::shared_ptr<Analyzer::Expr> IntLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
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

std::shared_ptr<Analyzer::Expr> FixedPtLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
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

std::shared_ptr<Analyzer::Expr> FloatLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  Datum d;
  d.floatval = floatval;
  return makeExpr<Analyzer::Constant>(kFLOAT, false, d);
}

std::shared_ptr<Analyzer::Expr> DoubleLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  Datum d;
  d.doubleval = doubleval;
  return makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
}

std::shared_ptr<Analyzer::Expr> TimestampLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  return get(timestampval_);
}

std::shared_ptr<Analyzer::Expr> TimestampLiteral::get(const int64_t timestampval) {
  Datum d;
  d.bigintval = timestampval;
  return makeExpr<Analyzer::Constant>(kTIMESTAMP, false, d);
}

std::shared_ptr<Analyzer::Expr> UserLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  Datum d;
  return makeExpr<Analyzer::Constant>(kTEXT, false, d);
}

std::shared_ptr<Analyzer::Expr> UserLiteral::get(const std::string& user) {
  Datum d;
  d.stringval = new std::string(user);
  return makeExpr<Analyzer::Constant>(kTEXT, false, d);
}

std::shared_ptr<Analyzer::Expr> ArrayLiteral::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  SQLTypeInfo ti = SQLTypeInfo(kARRAY, true);
  bool set_subtype = true;
  std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
  for (auto& p : value_list) {
    auto e = p->analyze(catalog, query, allow_tlist_ref);
    CHECK(e);
    auto c = std::dynamic_pointer_cast<Analyzer::Constant>(e);
    if (c != nullptr && c->get_is_null()) {
      value_exprs.push_back(c);
      continue;
    }
    auto subtype = e->get_type_info().get_type();
    if (subtype == kNULLT) {
      // NULL element
    } else if (set_subtype) {
      ti.set_subtype(subtype);
      set_subtype = false;
    }
    value_exprs.push_back(e);
  }
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::Constant>(ti, false, value_exprs);
  return result;
}

std::string ArrayLiteral::to_string() const {
  std::string str = "{";
  bool notfirst = false;
  for (auto& p : value_list) {
    if (notfirst) {
      str += ", ";
    } else {
      notfirst = true;
    }
    str += p->to_string();
  }
  str += "}";
  return str;
}

std::shared_ptr<Analyzer::Expr> OperExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto left_expr = left->analyze(catalog, query, allow_tlist_ref);
  const auto& left_type = left_expr->get_type_info();
  if (right == nullptr) {
    return makeExpr<Analyzer::UOper>(
        left_type, left_expr->get_contains_agg(), optype, left_expr->decompress());
  }
  if (optype == kARRAY_AT) {
    if (left_type.get_type() != kARRAY) {
      throw std::runtime_error(left->to_string() + " is not of array type.");
    }
    auto right_expr = right->analyze(catalog, query, allow_tlist_ref);
    const auto& right_type = right_expr->get_type_info();
    if (!right_type.is_integer()) {
      throw std::runtime_error(right->to_string() + " is not of integer type.");
    }
    return makeExpr<Analyzer::BinOper>(
        left_type.get_elem_type(), false, kARRAY_AT, kONE, left_expr, right_expr);
  }
  auto right_expr = right->analyze(catalog, query, allow_tlist_ref);
  return normalize(optype, opqualifier, left_expr, right_expr);
}

std::shared_ptr<Analyzer::Expr> OperExpr::normalize(
    const SQLOps optype,
    const SQLQualifier qual,
    std::shared_ptr<Analyzer::Expr> left_expr,
    std::shared_ptr<Analyzer::Expr> right_expr) {
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
    CHECK(!std::dynamic_pointer_cast<Analyzer::Subquery>(right_expr));
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
  auto result_type = Analyzer::BinOper::analyze_type_info(
      optype, left_type, right_type, &new_left_type, &new_right_type);
  if (result_type.is_timeinterval()) {
    return makeExpr<Analyzer::BinOper>(
        result_type, false, optype, qual, left_expr, right_expr);
  }
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

  if (IS_COMPARISON(optype)) {
    if (optype != kOVERLAPS && new_left_type.is_geometry() &&
        new_right_type.is_geometry()) {
      throw std::runtime_error(
          "Comparison operators are not yet supported for geospatial types.");
    }

    if (new_left_type.get_compression() == kENCODING_DICT &&
        new_right_type.get_compression() == kENCODING_DICT &&
        new_left_type.get_comp_param() == new_right_type.get_comp_param()) {
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
  return makeExpr<Analyzer::BinOper>(
      result_type, has_agg, optype, qual, left_expr, right_expr);
}

std::shared_ptr<Analyzer::Expr> SubqueryExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  throw std::runtime_error("Subqueries are not supported yet.");
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> IsNullExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  auto result = makeExpr<Analyzer::UOper>(kBOOLEAN, kISNULL, arg_expr);
  if (is_not) {
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  }
  return result;
}

std::shared_ptr<Analyzer::Expr> InSubquery::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  throw std::runtime_error("Subqueries are not supported yet.");
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> InValues::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  SQLTypeInfo ti = arg_expr->get_type_info();
  bool dict_comp = ti.get_compression() == kENCODING_DICT;
  std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
  for (auto& p : value_list) {
    auto e = p->analyze(catalog, query, allow_tlist_ref);
    if (ti != e->get_type_info()) {
      if (ti.is_string() && e->get_type_info().is_string()) {
        ti = Analyzer::BinOper::common_string_type(ti, e->get_type_info());
      } else if (ti.is_number() && e->get_type_info().is_number()) {
        ti = Analyzer::BinOper::common_numeric_type(ti, e->get_type_info());
      } else {
        throw std::runtime_error("IN expressions must contain compatible types.");
      }
    }
    if (dict_comp) {
      value_exprs.push_back(e->add_cast(arg_expr->get_type_info()));
    } else {
      value_exprs.push_back(e);
    }
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
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::InValues>(arg_expr, value_exprs);
  if (is_not) {
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  }
  return result;
}

std::shared_ptr<Analyzer::Expr> BetweenExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  auto lower_expr = lower->analyze(catalog, query, allow_tlist_ref);
  auto upper_expr = upper->analyze(catalog, query, allow_tlist_ref);
  SQLTypeInfo new_left_type, new_right_type;
  (void)Analyzer::BinOper::analyze_type_info(kGE,
                                             arg_expr->get_type_info(),
                                             lower_expr->get_type_info(),
                                             &new_left_type,
                                             &new_right_type);
  auto lower_pred =
      makeExpr<Analyzer::BinOper>(kBOOLEAN,
                                  kGE,
                                  kONE,
                                  arg_expr->add_cast(new_left_type)->decompress(),
                                  lower_expr->add_cast(new_right_type)->decompress());
  (void)Analyzer::BinOper::analyze_type_info(kLE,
                                             arg_expr->get_type_info(),
                                             lower_expr->get_type_info(),
                                             &new_left_type,
                                             &new_right_type);
  auto upper_pred = makeExpr<Analyzer::BinOper>(
      kBOOLEAN,
      kLE,
      kONE,
      arg_expr->deep_copy()->add_cast(new_left_type)->decompress(),
      upper_expr->add_cast(new_right_type)->decompress());
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::BinOper>(kBOOLEAN, kAND, kONE, lower_pred, upper_pred);
  if (is_not) {
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  }
  return result;
}

std::shared_ptr<Analyzer::Expr> CharLengthExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  if (!arg_expr->get_type_info().is_string()) {
    throw std::runtime_error(
        "expression in char_length clause must be of a string type.");
  }
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::CharLengthExpr>(arg_expr->decompress(), calc_encoded_length);
  return result;
}

std::shared_ptr<Analyzer::Expr> CardinalityExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  if (!arg_expr->get_type_info().is_array()) {
    throw std::runtime_error(
        "expression in cardinality clause must be of an array type.");
  }
  std::shared_ptr<Analyzer::Expr> result =
      makeExpr<Analyzer::CardinalityExpr>(arg_expr->decompress());
  return result;
}

void LikeExpr::check_like_expr(const std::string& like_str, char escape_char) {
  if (like_str.back() == escape_char) {
    throw std::runtime_error("LIKE pattern must not end with escape character.");
  }
}

bool LikeExpr::test_is_simple_expr(const std::string& like_str, char escape_char) {
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

std::shared_ptr<Analyzer::Expr> LikeExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  auto like_expr = like_string->analyze(catalog, query, allow_tlist_ref);
  auto escape_expr = escape_string == nullptr
                         ? nullptr
                         : escape_string->analyze(catalog, query, allow_tlist_ref);
  return LikeExpr::get(arg_expr, like_expr, escape_expr, is_ilike, is_not);
}

std::shared_ptr<Analyzer::Expr> LikeExpr::get(std::shared_ptr<Analyzer::Expr> arg_expr,
                                              std::shared_ptr<Analyzer::Expr> like_expr,
                                              std::shared_ptr<Analyzer::Expr> escape_expr,
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
    auto c = std::dynamic_pointer_cast<Analyzer::Constant>(escape_expr);
    if (c != nullptr && c->get_constval().stringval->length() > 1) {
      throw std::runtime_error("String after ESCAPE must have a single character.");
    }
    escape_char = (*c->get_constval().stringval)[0];
  }
  auto c = std::dynamic_pointer_cast<Analyzer::Constant>(like_expr);
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
  std::shared_ptr<Analyzer::Expr> result = makeExpr<Analyzer::LikeExpr>(
      arg_expr->decompress(), like_expr, escape_expr, is_ilike, is_simple);
  if (is_not) {
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  }
  return result;
}

void RegexpExpr::check_pattern_expr(const std::string& pattern_str, char escape_char) {
  if (pattern_str.back() == escape_char) {
    throw std::runtime_error("REGEXP pattern must not end with escape character.");
  }
}

bool RegexpExpr::translate_to_like_pattern(std::string& pattern_str, char escape_char) {
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

std::shared_ptr<Analyzer::Expr> RegexpExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  auto pattern_expr = pattern_string->analyze(catalog, query, allow_tlist_ref);
  auto escape_expr = escape_string == nullptr
                         ? nullptr
                         : escape_string->analyze(catalog, query, allow_tlist_ref);
  return RegexpExpr::get(arg_expr, pattern_expr, escape_expr, is_not);
}

std::shared_ptr<Analyzer::Expr> RegexpExpr::get(
    std::shared_ptr<Analyzer::Expr> arg_expr,
    std::shared_ptr<Analyzer::Expr> pattern_expr,
    std::shared_ptr<Analyzer::Expr> escape_expr,
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
    auto c = std::dynamic_pointer_cast<Analyzer::Constant>(escape_expr);
    if (c != nullptr && c->get_constval().stringval->length() > 1) {
      throw std::runtime_error("String after ESCAPE must have a single character.");
    }
    escape_char = (*c->get_constval().stringval)[0];
    if (escape_char != '\\') {
      throw std::runtime_error("Only supporting '\\' escape character.");
    }
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
  if (is_not) {
    result = makeExpr<Analyzer::UOper>(kBOOLEAN, kNOT, result);
  }
  return result;
}

std::shared_ptr<Analyzer::Expr> LikelihoodExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  return LikelihoodExpr::get(arg_expr, likelihood, is_not);
}

std::shared_ptr<Analyzer::Expr> LikelihoodExpr::get(
    std::shared_ptr<Analyzer::Expr> arg_expr,
    float likelihood,
    const bool is_not) {
  if (!arg_expr->get_type_info().is_boolean()) {
    throw std::runtime_error("likelihood expression expects boolean type.");
  }
  std::shared_ptr<Analyzer::Expr> result = makeExpr<Analyzer::LikelihoodExpr>(
      arg_expr->decompress(), is_not ? 1 - likelihood : likelihood);
  return result;
}

std::shared_ptr<Analyzer::Expr> ExistsExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  throw std::runtime_error("Subqueries are not supported yet.");
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> ColumnRef::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  int table_id{0};
  int rte_idx{0};
  const ColumnDescriptor* cd{nullptr};
  if (column == nullptr) {
    throw std::runtime_error("invalid column name *.");
  }
  if (table != nullptr) {
    rte_idx = query.get_rte_idx(*table);
    if (rte_idx < 0) {
      throw std::runtime_error("range variable or table name " + *table +
                               " does not exist.");
    }
    Analyzer::RangeTableEntry* rte = query.get_rte(rte_idx);
    cd = rte->get_column_desc(catalog, *column);
    if (cd == nullptr) {
      throw std::runtime_error("Column name " + *column + " does not exist.");
    }
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
      } else if (cd != nullptr && found) {
        throw std::runtime_error("Column name " + *column + " is ambiguous.");
      }
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
        } else if (*column == p->get_resname() && found) {
          throw std::runtime_error("Output alias " + *column + " is ambiguous.");
        }
        i++;
      }
      if (found) {
        if (dynamic_cast<Analyzer::Var*>(tle->get_expr())) {
          Analyzer::Var* v = static_cast<Analyzer::Var*>(tle->get_expr());
          if (v->get_which_row() == Analyzer::Var::kGROUPBY) {
            return v->deep_copy();
          }
        }
        if (allow_tlist_ref == TlistRefType::TLIST_COPY) {
          return tle->get_expr()->deep_copy();
        } else {
          return makeExpr<Analyzer::Var>(
              tle->get_expr()->get_type_info(), Analyzer::Var::kOUTPUT, varno);
        }
      }
    }
    if (cd == nullptr) {
      throw std::runtime_error("Column name " + *column + " does not exist.");
    }
  }
  return makeExpr<Analyzer::ColumnVar>(cd->columnType, table_id, cd->columnId, rte_idx);
}

std::shared_ptr<Analyzer::Expr> FunctionRef::analyze(
    const Catalog_Namespace::Catalog& catalog,
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
      if (ti.is_string() && (ti.get_compression() != kENCODING_DICT || !distinct)) {
        throw std::runtime_error(
            "Strings must be dictionary-encoded in COUNT(DISTINCT).");
      }
      if (ti.get_type() == kARRAY && !distinct) {
        throw std::runtime_error("Only COUNT(DISTINCT) is supported on arrays.");
      }
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
      if (!arg_expr->get_type_info().is_number()) {
        throw std::runtime_error("Cannot compute AVG on non-number-type arguments.");
      }
      arg_expr = arg_expr->decompress();
      result_type = SQLTypeInfo(kDOUBLE, false);
    } else if (boost::iequals(*name, "sum")) {
      agg_type = kSUM;
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      if (!arg_expr->get_type_info().is_number()) {
        throw std::runtime_error("Cannot compute SUM on non-number-type arguments.");
      }
      arg_expr = arg_expr->decompress();
      result_type = arg_expr->get_type_info().is_integer() ? SQLTypeInfo(kBIGINT, false)
                                                           : arg_expr->get_type_info();
    } else if (boost::iequals(*name, "unnest")) {
      arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
      const SQLTypeInfo& arg_ti = arg_expr->get_type_info();
      if (arg_ti.get_type() != kARRAY) {
        throw std::runtime_error(arg->to_string() + " is not of array type.");
      }
      return makeExpr<Analyzer::UOper>(arg_ti.get_elem_type(), false, kUNNEST, arg_expr);
    } else {
      throw std::runtime_error("invalid function name: " + *name);
    }
    if (arg_expr->get_type_info().is_string() ||
        arg_expr->get_type_info().get_type() == kARRAY) {
      throw std::runtime_error(
          "Only COUNT(DISTINCT ) aggregate is supported on strings and arrays.");
    }
  }
  int naggs = query.get_num_aggs();
  query.set_num_aggs(naggs + 1);
  return makeExpr<Analyzer::AggExpr>(
      result_type, agg_type, arg_expr, is_distinct, nullptr);
}

std::shared_ptr<Analyzer::Expr> CastExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  target_type->check_type();
  auto arg_expr = arg->analyze(catalog, query, allow_tlist_ref);
  SQLTypeInfo ti(target_type->get_type(),
                 target_type->get_param1(),
                 target_type->get_param2(),
                 arg_expr->get_type_info().get_notnull());
  if (arg_expr->get_type_info().get_type() != target_type->get_type() &&
      arg_expr->get_type_info().get_compression() != kENCODING_NONE) {
    arg_expr->decompress();
  }
  return arg_expr->add_cast(ti);
}

std::shared_ptr<Analyzer::Expr> CaseExpr::analyze(
    const Catalog_Namespace::Catalog& catalog,
    Analyzer::Query& query,
    TlistRefType allow_tlist_ref) const {
  SQLTypeInfo ti;
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      expr_pair_list;
  for (auto& p : when_then_list) {
    auto e1 = p->get_expr1()->analyze(catalog, query, allow_tlist_ref);
    if (e1->get_type_info().get_type() != kBOOLEAN) {
      throw std::runtime_error("Only boolean expressions can be used after WHEN.");
    }
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

bool bool_from_string_literal(const Parser::StringLiteral* str_literal) {
  const std::string* s = str_literal->get_stringval();
  if (*s == "t" || *s == "true" || *s == "T" || *s == "True") {
    return true;
  } else if (*s == "f" || *s == "false" || *s == "F" || *s == "False") {
    return false;
  } else {
    throw std::runtime_error("Invalid string for boolean " + *s);
  }
}

}  // namespace

std::shared_ptr<Analyzer::Expr> CaseExpr::normalize(
    const std::list<std::pair<std::shared_ptr<Analyzer::Expr>,
                              std::shared_ptr<Analyzer::Expr>>>& expr_pair_list,
    const std::shared_ptr<Analyzer::Expr> else_e_in) {
  SQLTypeInfo ti;
  bool has_agg = false;
  std::set<int> dictionary_ids;

  for (auto& p : expr_pair_list) {
    auto e1 = p.first;
    CHECK(e1->get_type_info().is_boolean());
    auto e2 = p.second;
    if (e2->get_type_info().is_dict_encoded_string()) {
      dictionary_ids.insert(e2->get_type_info().get_comp_param());
    }
    if (ti.get_type() == kNULLT) {
      ti = e2->get_type_info();
    } else if (e2->get_type_info().get_type() == kNULLT) {
      ti.set_notnull(false);
      e2->set_type_info(ti);
    } else if (ti != e2->get_type_info()) {
      if (ti.is_string() && e2->get_type_info().is_string()) {
        ti = Analyzer::BinOper::common_string_type(ti, e2->get_type_info());
      } else if (ti.is_number() && e2->get_type_info().is_number()) {
        ti = Analyzer::BinOper::common_numeric_type(ti, e2->get_type_info());
      } else if (ti.is_boolean() && e2->get_type_info().is_boolean()) {
        ti = Analyzer::BinOper::common_numeric_type(ti, e2->get_type_info());
      } else {
        throw std::runtime_error(
            "expressions in THEN clause must be of the same or compatible types.");
      }
    }
    if (e2->get_contains_agg()) {
      has_agg = true;
    }
  }
  auto else_e = else_e_in;
  if (else_e) {
    if (else_e->get_contains_agg()) {
      has_agg = true;
    }
    if (expr_is_null(else_e.get())) {
      ti.set_notnull(false);
      else_e->set_type_info(ti);
    } else if (ti != else_e->get_type_info()) {
      if (else_e->get_type_info().is_dict_encoded_string()) {
        dictionary_ids.insert(else_e->get_type_info().get_comp_param());
      }
      ti.set_notnull(false);
      if (ti.is_string() && else_e->get_type_info().is_string()) {
        ti = Analyzer::BinOper::common_string_type(ti, else_e->get_type_info());
      } else if (ti.is_number() && else_e->get_type_info().is_number()) {
        ti = Analyzer::BinOper::common_numeric_type(ti, else_e->get_type_info());
      } else if (ti.is_boolean() && else_e->get_type_info().is_boolean()) {
        ti = Analyzer::BinOper::common_numeric_type(ti, else_e->get_type_info());
      } else if (get_logical_type_info(ti) !=
                 get_logical_type_info(else_e->get_type_info())) {
        throw std::runtime_error(
            // types differing by encoding will be resolved at decode

            "expressions in ELSE clause must be of the same or compatible types as those "
            "in the THEN clauses.");
      }
    }
  }
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      cast_expr_pair_list;
  for (auto p : expr_pair_list) {
    ti.set_notnull(false);
    cast_expr_pair_list.emplace_back(p.first, p.second->add_cast(ti));
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
    throw std::runtime_error(
        "Can't deduce the type for case expressions, all branches null");
  }

  auto case_expr = makeExpr<Analyzer::CaseExpr>(ti, has_agg, cast_expr_pair_list, else_e);
  if (ti.get_compression() != kENCODING_DICT && dictionary_ids.size() == 1 &&
      *(dictionary_ids.begin()) > 0) {
    // the above logic makes two assumptions when strings are present. 1) that all types
    // in the case statement are either null or strings, and 2) that none-encoded strings
    // will always win out over dict encoding. If we only have one dictionary, and that
    // dictionary is not a transient dictionary, we can cast the entire case to be dict
    // encoded and use transient dictionaries for any literals
    ti.set_compression(kENCODING_DICT);
    ti.set_comp_param(*dictionary_ids.begin());
    case_expr->add_cast(ti);
  }
  return case_expr;
}

std::string CaseExpr::to_string() const {
  std::string str("CASE ");
  for (auto& p : when_then_list) {
    str += "WHEN " + p->get_expr1()->to_string() + " THEN " +
           p->get_expr2()->to_string() + " ";
  }
  if (else_expr != nullptr) {
    str += "ELSE " + else_expr->to_string();
  }
  str += " END";
  return str;
}

void UnionQuery::analyze(const Catalog_Namespace::Catalog& catalog,
                         Analyzer::Query& query) const {
  left->analyze(catalog, query);
  Analyzer::Query* right_query = new Analyzer::Query();
  right->analyze(catalog, *right_query);
  query.set_next_query(right_query);
  query.set_is_unionall(is_unionall);
}

void QuerySpec::analyze_having_clause(const Catalog_Namespace::Catalog& catalog,
                                      Analyzer::Query& query) const {
  std::shared_ptr<Analyzer::Expr> p;
  if (having_clause != nullptr) {
    p = having_clause->analyze(catalog, query, Expr::TlistRefType::TLIST_COPY);
    if (p->get_type_info().get_type() != kBOOLEAN) {
      throw std::runtime_error("Only boolean expressions can be in HAVING clause.");
    }
    p->check_group_by(query.get_group_by());
  }
  query.set_having_predicate(p);
}

void QuerySpec::analyze_group_by(const Catalog_Namespace::Catalog& catalog,
                                 Analyzer::Query& query) const {
  std::list<std::shared_ptr<Analyzer::Expr>> groupby;
  if (!groupby_clause.empty()) {
    int gexpr_no = 1;
    std::shared_ptr<Analyzer::Expr> gexpr;
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist =
        query.get_targetlist();
    for (auto& c : groupby_clause) {
      // special-case ordinal numbers in GROUP BY
      if (dynamic_cast<Literal*>(c.get())) {
        IntLiteral* i = dynamic_cast<IntLiteral*>(c.get());
        if (!i) {
          throw std::runtime_error("Invalid literal in GROUP BY clause.");
        }
        int varno = (int)i->get_intval();
        if (varno <= 0 || varno > static_cast<int>(tlist.size())) {
          throw std::runtime_error("Invalid ordinal number in GROUP BY clause.");
        }
        if (tlist[varno - 1]->get_expr()->get_contains_agg()) {
          throw std::runtime_error(
              "Ordinal number in GROUP BY cannot reference an expression containing "
              "aggregate "
              "functions.");
        }
        gexpr = makeExpr<Analyzer::Var>(
            tlist[varno - 1]->get_expr()->get_type_info(), Analyzer::Var::kOUTPUT, varno);
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
        if (v != nullptr) {
          v->set_type_info(new_e->get_type_info());
        }
      } else {
        groupby.push_back(gexpr);
      }
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

void QuerySpec::analyze_where_clause(const Catalog_Namespace::Catalog& catalog,
                                     Analyzer::Query& query) const {
  if (where_clause == nullptr) {
    query.set_where_predicate(nullptr);
    return;
  }
  auto p = where_clause->analyze(catalog, query, Expr::TlistRefType::TLIST_COPY);
  if (p->get_type_info().get_type() != kBOOLEAN) {
    throw std::runtime_error("Only boolean expressions can be in WHERE clause.");
  }
  query.set_where_predicate(p);
}

void QuerySpec::analyze_select_clause(const Catalog_Namespace::Catalog& catalog,
                                      Analyzer::Query& query) const {
  std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist =
      query.get_targetlist_nonconst();
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
        const std::string* range_var_name =
            dynamic_cast<const ColumnRef*>(select_expr)->get_table();
        int rte_idx = query.get_rte_idx(*range_var_name);
        if (rte_idx < 0) {
          throw std::runtime_error("invalid range variable name: " + *range_var_name);
        }
        Analyzer::RangeTableEntry* rte = query.get_rte(rte_idx);
        rte->expand_star_in_targetlist(catalog, tlist, rte_idx);
      } else {
        auto e = select_expr->analyze(catalog, query);
        std::string resname;

        if (p->get_alias() != nullptr) {
          resname = *p->get_alias();
        } else if (std::dynamic_pointer_cast<Analyzer::ColumnVar>(e) &&
                   !std::dynamic_pointer_cast<Analyzer::Var>(e)) {
          auto colvar = std::static_pointer_cast<Analyzer::ColumnVar>(e);
          const ColumnDescriptor* col_desc = catalog.getMetadataForColumn(
              colvar->get_table_id(), colvar->get_column_id());
          resname = col_desc->columnName;
        }
        if (e->get_type_info().get_type() == kNULLT) {
          throw std::runtime_error(
              "Untyped NULL in SELECT clause.  Use CAST to specify a type.");
        }
        auto o = std::static_pointer_cast<Analyzer::UOper>(e);
        bool unnest = (o != nullptr && o->get_optype() == kUNNEST);
        auto tle = std::make_shared<Analyzer::TargetEntry>(resname, e, unnest);
        tlist.push_back(tle);
      }
    }
  }
}

void QuerySpec::analyze_from_clause(const Catalog_Namespace::Catalog& catalog,
                                    Analyzer::Query& query) const {
  Analyzer::RangeTableEntry* rte;
  for (auto& p : from_clause) {
    const TableDescriptor* table_desc;
    table_desc = catalog.getMetadataForTable(*p->get_table_name());
    if (table_desc == nullptr) {
      throw std::runtime_error("Table " + *p->get_table_name() + " does not exist.");
    }
    std::string range_var;
    if (p->get_range_var() == nullptr) {
      range_var = *p->get_table_name();
    } else {
      range_var = *p->get_range_var();
    }
    rte = new Analyzer::RangeTableEntry(range_var, table_desc, nullptr);
    query.add_rte(rte);
  }
}

void QuerySpec::analyze(const Catalog_Namespace::Catalog& catalog,
                        Analyzer::Query& query) const {
  query.set_is_distinct(is_distinct);
  analyze_from_clause(catalog, query);
  analyze_select_clause(catalog, query);
  analyze_where_clause(catalog, query);
  analyze_group_by(catalog, query);
  analyze_having_clause(catalog, query);
}

void SelectStmt::analyze(const Catalog_Namespace::Catalog& catalog,
                         Analyzer::Query& query) const {
  query.set_stmt_type(kSELECT);
  query.set_limit(limit);
  if (offset < 0) {
    throw std::runtime_error("OFFSET cannot be negative.");
  }
  query.set_offset(offset);
  query_expr->analyze(catalog, query);
  if (orderby_clause.empty() && !query.get_is_distinct()) {
    query.set_order_by(nullptr);
    return;
  }
  const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist =
      query.get_targetlist();
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
        if (!found) {
          throw std::runtime_error("invalid name in order by: " + *name);
        }
      }
      order_by->push_back(
          Analyzer::OrderEntry(tle_no, p->get_is_desc(), p->get_nulls_first()));
    }
  }
  if (query.get_is_distinct()) {
    // extend order_by to include all targetlist entries.
    for (int i = 1; i <= static_cast<int>(tlist.size()); i++) {
      bool in_orderby = false;
      std::for_each(order_by->begin(),
                    order_by->end(),
                    [&in_orderby, i](const Analyzer::OrderEntry& oe) {
                      in_orderby = in_orderby || (i == oe.tle_no);
                    });
      if (!in_orderby) {
        order_by->push_back(Analyzer::OrderEntry(i, false, false));
      }
    }
  }
  query.set_order_by(order_by);
}

std::string SelectEntry::to_string() const {
  std::string str = select_expr->to_string();
  if (alias != nullptr) {
    str += " AS " + *alias;
  }
  return str;
}

std::string TableRef::to_string() const {
  std::string str = *table_name;
  if (range_var != nullptr) {
    str += " " + *range_var;
  }
  return str;
}

std::string ColumnRef::to_string() const {
  std::string str;
  if (table == nullptr) {
    str = *column;
  } else if (column == nullptr) {
    str = *table + ".*";
  } else {
    str = *table + "." + *column;
  }
  return str;
}

std::string OperExpr::to_string() const {
  std::string op_str[] = {
      "=", "===", "<>", "<", ">", "<=", ">=", " AND ", " OR ", "NOT", "-", "+", "*", "/"};
  std::string str;
  if (optype == kUMINUS) {
    str = "-(" + left->to_string() + ")";
  } else if (optype == kNOT) {
    str = "NOT (" + left->to_string() + ")";
  } else if (optype == kARRAY_AT) {
    str = left->to_string() + "[" + right->to_string() + "]";
  } else if (optype == kUNNEST) {
    str = "UNNEST(" + left->to_string() + ")";
  } else if (optype == kIN) {
    str = "(" + left->to_string() + " IN " + right->to_string() + ")";
  } else {
    str = "(" + left->to_string() + op_str[optype] + right->to_string() + ")";
  }
  return str;
}

std::string InExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not) {
    str += " NOT IN ";
  } else {
    str += " IN ";
  }
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
  if (is_not) {
    str += " IS NOT NULL";
  } else {
    str += " IS NULL";
  }
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
    if (notfirst) {
      str += ", ";
    } else {
      notfirst = true;
    }
    str += p->to_string();
  }
  str += ")";
  return str;
}

std::string BetweenExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not) {
    str += " NOT BETWEEN ";
  } else {
    str += " BETWEEN ";
  }
  str += lower->to_string() + " AND " + upper->to_string();
  return str;
}

std::string CharLengthExpr::to_string() const {
  std::string str;
  if (calc_encoded_length) {
    str = "CHAR_LENGTH (" + arg->to_string() + ")";
  } else {
    str = "LENGTH (" + arg->to_string() + ")";
  }
  return str;
}

std::string CardinalityExpr::to_string() const {
  std::string str = "CARDINALITY(" + arg->to_string() + ")";
  return str;
}

std::string LikeExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not) {
    str += " NOT LIKE ";
  } else {
    str += " LIKE ";
  }
  str += like_string->to_string();
  if (escape_string != nullptr) {
    str += " ESCAPE " + escape_string->to_string();
  }
  return str;
}

std::string RegexpExpr::to_string() const {
  std::string str = arg->to_string();
  if (is_not) {
    str += " NOT REGEXP ";
  } else {
    str += " REGEXP ";
  }
  str += pattern_string->to_string();
  if (escape_string != nullptr) {
    str += " ESCAPE " + escape_string->to_string();
  }
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
  if (distinct) {
    str += "DISTINCT ";
  }
  if (arg == nullptr) {
    str += "*)";
  } else {
    str += arg->to_string() + ")";
  }
  return str;
}

std::string QuerySpec::to_string() const {
  std::string query_str = "SELECT ";
  if (is_distinct) {
    query_str += "DISTINCT ";
  }
  if (select_clause.empty()) {
    query_str += "* ";
  } else {
    bool notfirst = false;
    for (auto& p : select_clause) {
      if (notfirst) {
        query_str += ", ";
      } else {
        notfirst = true;
      }
      query_str += p->to_string();
    }
  }
  query_str += " FROM ";
  bool notfirst = false;
  for (auto& p : from_clause) {
    if (notfirst) {
      query_str += ", ";
    } else {
      notfirst = true;
    }
    query_str += p->to_string();
  }
  if (where_clause) {
    query_str += " WHERE " + where_clause->to_string();
  }
  if (!groupby_clause.empty()) {
    query_str += " GROUP BY ";
    bool notfirst = false;
    for (auto& p : groupby_clause) {
      if (notfirst) {
        query_str += ", ";
      } else {
        notfirst = true;
      }
      query_str += p->to_string();
    }
  }
  if (having_clause) {
    query_str += " HAVING " + having_clause->to_string();
  }
  query_str += ";";
  return query_str;
}

void InsertStmt::analyze(const Catalog_Namespace::Catalog& catalog,
                         Analyzer::Query& query) const {
  query.set_stmt_type(kINSERT);
  const TableDescriptor* td = catalog.getMetadataForTable(*table);
  if (td == nullptr) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  }
  if (td->isView) {
    throw std::runtime_error("Insert to views is not supported yet.");
  }
  foreign_storage::validate_non_foreign_table_write(td);
  query.set_result_table_id(td->tableId);
  std::list<int> result_col_list;
  if (column_list.empty()) {
    const std::list<const ColumnDescriptor*> all_cols =
        catalog.getAllColumnMetadataForTable(td->tableId, false, false, true);
    for (auto cd : all_cols) {
      result_col_list.push_back(cd->columnId);
    }
  } else {
    for (auto& c : column_list) {
      const ColumnDescriptor* cd = catalog.getMetadataForColumn(td->tableId, *c);
      if (cd == nullptr) {
        throw std::runtime_error("Column " + *c + " does not exist.");
      }
      result_col_list.push_back(cd->columnId);
      const auto& col_ti = cd->columnType;
      if (col_ti.get_physical_cols() > 0) {
        CHECK(cd->columnType.is_geometry());
        for (auto i = 1; i <= col_ti.get_physical_cols(); i++) {
          const ColumnDescriptor* pcd =
              catalog.getMetadataForColumn(td->tableId, cd->columnId + i);
          if (pcd == nullptr) {
            throw std::runtime_error("Column " + *c + "'s metadata is incomplete.");
          }
          result_col_list.push_back(pcd->columnId);
        }
      }
    }
    if (catalog.getAllColumnMetadataForTable(td->tableId, false, false, true).size() !=
        result_col_list.size()) {
      throw std::runtime_error("Insert into a subset of columns is not supported yet.");
    }
  }
  query.set_result_col_list(result_col_list);
}

size_t InsertValuesStmt::determineLeafIndex(const Catalog_Namespace::Catalog& catalog,
                                            size_t num_leafs) {
  const TableDescriptor* td = catalog.getMetadataForTable(*table);
  if (td == nullptr) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  }
  if (td->isView) {
    throw std::runtime_error("Insert to views is not supported yet.");
  }
  foreign_storage::validate_non_foreign_table_write(td);
  if (td->partitions == "REPLICATED") {
    throw std::runtime_error("Cannot determine leaf on replicated table.");
  }

  if (0 == td->nShards) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dis;
    const auto leaf_idx = dis(gen) % num_leafs;
    return leaf_idx;
  }

  size_t indexOfShardColumn = 0;
  const ColumnDescriptor* shardColumn = catalog.getShardColumnMetadataForTable(td);
  CHECK(shardColumn);

  if (column_list.empty()) {
    auto all_cols =
        catalog.getAllColumnMetadataForTable(td->tableId, false, false, false);
    auto iter = std::find(all_cols.begin(), all_cols.end(), shardColumn);
    CHECK(iter != all_cols.end());
    indexOfShardColumn = std::distance(all_cols.begin(), iter);
  } else {
    for (auto& c : column_list) {
      if (*c == shardColumn->columnName) {
        break;
      }
      indexOfShardColumn++;
    }

    if (indexOfShardColumn == column_list.size()) {
      throw std::runtime_error("No value defined for shard column.");
    }
  }

  if (indexOfShardColumn >= value_list.size()) {
    throw std::runtime_error("No value defined for shard column.");
  }

  auto& shardColumnValueExpr = *(std::next(value_list.begin(), indexOfShardColumn));

  Analyzer::Query query;
  auto e = shardColumnValueExpr->analyze(catalog, query);
  e = e->add_cast(shardColumn->columnType);
  const Analyzer::Constant* con = dynamic_cast<Analyzer::Constant*>(e.get());
  if (!con) {
    auto col_cast = dynamic_cast<const Analyzer::UOper*>(e.get());
    CHECK(col_cast);
    CHECK_EQ(kCAST, col_cast->get_optype());
    con = dynamic_cast<const Analyzer::Constant*>(col_cast->get_operand());
  }
  CHECK(con);

  Datum d = con->get_constval();

  auto shard_count = td->nShards * num_leafs;
  int64_t shardId = 0;

  if (con->get_is_null()) {
    shardId = SHARD_FOR_KEY(inline_fixed_encoding_null_val(shardColumn->columnType),
                            shard_count);
  } else if (shardColumn->columnType.is_string()) {
    auto dictDesc =
        catalog.getMetadataForDict(shardColumn->columnType.get_comp_param(), true);
    auto str_id = dictDesc->stringDict->getOrAdd(*d.stringval);
    bool invalid = false;

    if (4 == shardColumn->columnType.get_size()) {
      invalid = str_id > max_valid_int_value<int32_t>();
    } else if (2 == shardColumn->columnType.get_size()) {
      invalid = str_id > max_valid_int_value<uint16_t>();
    } else if (1 == shardColumn->columnType.get_size()) {
      invalid = str_id > max_valid_int_value<uint8_t>();
    }

    if (invalid || str_id == inline_int_null_value<int32_t>()) {
      str_id = inline_fixed_encoding_null_val(shardColumn->columnType);
    }
    shardId = SHARD_FOR_KEY(str_id, shard_count);
  } else {
    switch (shardColumn->columnType.get_logical_size()) {
      case 8:
        shardId = SHARD_FOR_KEY(d.bigintval, shard_count);
        break;
      case 4:
        shardId = SHARD_FOR_KEY(d.intval, shard_count);
        break;
      case 2:
        shardId = SHARD_FOR_KEY(d.smallintval, shard_count);
        break;
      case 1:
        shardId = SHARD_FOR_KEY(d.tinyintval, shard_count);
        break;
      default:
        CHECK(false);
    }
  }

  return shardId / td->nShards;
}

void InsertValuesStmt::analyze(const Catalog_Namespace::Catalog& catalog,
                               Analyzer::Query& query) const {
  InsertStmt::analyze(catalog, query);
  std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist =
      query.get_targetlist_nonconst();
  const auto tableId = query.get_result_table_id();
  const std::list<const ColumnDescriptor*> non_phys_cols =
      catalog.getAllColumnMetadataForTable(tableId, false, false, false);
  if (non_phys_cols.size() != value_list.size()) {
    throw std::runtime_error("Insert has more target columns than expressions.");
  }
  std::list<int>::const_iterator it = query.get_result_col_list().begin();
  for (auto& v : value_list) {
    auto e = v->analyze(catalog, query);
    const ColumnDescriptor* cd =
        catalog.getMetadataForColumn(query.get_result_table_id(), *it);
    CHECK(cd);
    if (cd->columnType.get_notnull()) {
      auto c = std::dynamic_pointer_cast<Analyzer::Constant>(e);
      if (c != nullptr && c->get_is_null()) {
        throw std::runtime_error("Cannot insert NULL into column " + cd->columnName);
      }
    }
    e = e->add_cast(cd->columnType);
    tlist.emplace_back(new Analyzer::TargetEntry("", e, false));
    ++it;

    const auto& col_ti = cd->columnType;
    if (col_ti.get_physical_cols() > 0) {
      CHECK(cd->columnType.is_geometry());
      auto c = dynamic_cast<const Analyzer::Constant*>(e.get());
      if (!c) {
        auto uoper = std::dynamic_pointer_cast<Analyzer::UOper>(e);
        if (uoper && uoper->get_optype() == kCAST) {
          c = dynamic_cast<const Analyzer::Constant*>(uoper->get_operand());
        }
      }
      bool is_null = false;
      std::string* geo_string{nullptr};
      if (c) {
        is_null = c->get_is_null();
        if (!is_null) {
          geo_string = c->get_constval().stringval;
        }
      }
      if (!is_null && !geo_string) {
        throw std::runtime_error("Expecting a WKT or WKB hex string for column " +
                                 cd->columnName);
      }
      std::vector<double> coords;
      std::vector<double> bounds;
      std::vector<int> ring_sizes;
      std::vector<int> poly_rings;
      int render_group =
          0;  // @TODO simon.eves where to get render_group from in this context?!
      SQLTypeInfo import_ti{cd->columnType};
      if (!is_null) {
        if (!Geospatial::GeoTypesFactory::getGeoColumns(
                *geo_string, import_ti, coords, bounds, ring_sizes, poly_rings)) {
          throw std::runtime_error("Cannot read geometry to insert into column " +
                                   cd->columnName);
        }
        if (coords.empty()) {
          // Importing from geo_string WKT resulted in empty coords: dealing with a NULL
          is_null = true;
        }
        if (cd->columnType.get_type() != import_ti.get_type()) {
          // allow POLYGON to be inserted into MULTIPOLYGON column
          if (!(import_ti.get_type() == SQLTypes::kPOLYGON &&
                cd->columnType.get_type() == SQLTypes::kMULTIPOLYGON)) {
            throw std::runtime_error(
                "Imported geometry doesn't match the type of column " + cd->columnName);
          }
        }
      } else {
        // Special case for NULL POINT, push NULL representation to coords
        if (cd->columnType.get_type() == kPOINT) {
          if (!coords.empty()) {
            throw std::runtime_error("NULL POINT with unexpected coordinates in column " +
                                     cd->columnName);
          }
          coords.push_back(NULL_ARRAY_DOUBLE);
          coords.push_back(NULL_DOUBLE);
        }
      }

      // TODO: check if import SRID matches columns SRID, may need to transform before
      // inserting

      int nextColumnOffset = 1;

      const ColumnDescriptor* cd_coords = catalog.getMetadataForColumn(
          query.get_result_table_id(), cd->columnId + nextColumnOffset);
      CHECK(cd_coords);
      CHECK_EQ(cd_coords->columnType.get_type(), kARRAY);
      CHECK_EQ(cd_coords->columnType.get_subtype(), kTINYINT);
      std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
      if (!is_null || cd->columnType.get_type() == kPOINT) {
        auto compressed_coords = Geospatial::compress_coords(coords, col_ti);
        for (auto cc : compressed_coords) {
          Datum d;
          d.tinyintval = cc;
          auto e = makeExpr<Analyzer::Constant>(kTINYINT, false, d);
          value_exprs.push_back(e);
        }
      }
      tlist.emplace_back(new Analyzer::TargetEntry(
          "",
          makeExpr<Analyzer::Constant>(cd_coords->columnType, is_null, value_exprs),
          false));
      ++it;
      nextColumnOffset++;

      if (cd->columnType.get_type() == kPOLYGON ||
          cd->columnType.get_type() == kMULTIPOLYGON) {
        // Put ring sizes array into separate physical column
        const ColumnDescriptor* cd_ring_sizes = catalog.getMetadataForColumn(
            query.get_result_table_id(), cd->columnId + nextColumnOffset);
        CHECK(cd_ring_sizes);
        CHECK_EQ(cd_ring_sizes->columnType.get_type(), kARRAY);
        CHECK_EQ(cd_ring_sizes->columnType.get_subtype(), kINT);
        std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
        if (!is_null) {
          for (auto c : ring_sizes) {
            Datum d;
            d.intval = c;
            auto e = makeExpr<Analyzer::Constant>(kINT, false, d);
            value_exprs.push_back(e);
          }
        }
        tlist.emplace_back(new Analyzer::TargetEntry(
            "",
            makeExpr<Analyzer::Constant>(cd_ring_sizes->columnType, is_null, value_exprs),
            false));
        ++it;
        nextColumnOffset++;

        if (cd->columnType.get_type() == kMULTIPOLYGON) {
          // Put poly_rings array into separate physical column
          const ColumnDescriptor* cd_poly_rings = catalog.getMetadataForColumn(
              query.get_result_table_id(), cd->columnId + nextColumnOffset);
          CHECK(cd_poly_rings);
          CHECK_EQ(cd_poly_rings->columnType.get_type(), kARRAY);
          CHECK_EQ(cd_poly_rings->columnType.get_subtype(), kINT);
          std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
          if (!is_null) {
            for (auto c : poly_rings) {
              Datum d;
              d.intval = c;
              auto e = makeExpr<Analyzer::Constant>(kINT, false, d);
              value_exprs.push_back(e);
            }
          }
          tlist.emplace_back(new Analyzer::TargetEntry(
              "",
              makeExpr<Analyzer::Constant>(
                  cd_poly_rings->columnType, is_null, value_exprs),
              false));
          ++it;
          nextColumnOffset++;
        }
      }

      if (cd->columnType.get_type() == kLINESTRING ||
          cd->columnType.get_type() == kPOLYGON ||
          cd->columnType.get_type() == kMULTIPOLYGON) {
        const ColumnDescriptor* cd_bounds = catalog.getMetadataForColumn(
            query.get_result_table_id(), cd->columnId + nextColumnOffset);
        CHECK(cd_bounds);
        CHECK_EQ(cd_bounds->columnType.get_type(), kARRAY);
        CHECK_EQ(cd_bounds->columnType.get_subtype(), kDOUBLE);
        std::list<std::shared_ptr<Analyzer::Expr>> value_exprs;
        if (!is_null) {
          for (auto b : bounds) {
            Datum d;
            d.doubleval = b;
            auto e = makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
            value_exprs.push_back(e);
          }
        }
        tlist.emplace_back(new Analyzer::TargetEntry(
            "",
            makeExpr<Analyzer::Constant>(cd_bounds->columnType, is_null, value_exprs),
            false));
        ++it;
        nextColumnOffset++;
      }

      if (cd->columnType.get_type() == kPOLYGON ||
          cd->columnType.get_type() == kMULTIPOLYGON) {
        // Put render group into separate physical column
        const ColumnDescriptor* cd_render_group = catalog.getMetadataForColumn(
            query.get_result_table_id(), cd->columnId + nextColumnOffset);
        CHECK(cd_render_group);
        CHECK_EQ(cd_render_group->columnType.get_type(), kINT);
        Datum d;
        d.intval = render_group;
        tlist.emplace_back(new Analyzer::TargetEntry(
            "",
            makeExpr<Analyzer::Constant>(cd_render_group->columnType, is_null, d),
            false));
        ++it;
        nextColumnOffset++;
      }
    }
  }
}

void InsertValuesStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  if (!session.checkDBAccessPrivileges(
          DBObjectType::TableDBObjectType, AccessPrivileges::INSERT_INTO_TABLE, *table)) {
    throw std::runtime_error("User has no insert privileges on " + *table + ".");
  }

  auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  Analyzer::Query query;
  analyze(catalog, query);

  //  Acquire schema write lock -- leave data lock so the fragmenter can checkpoint. For
  //  singleton inserts we just take a write lock on the schema, which prevents concurrent
  //  inserts.
  auto result_table_id = query.get_result_table_id();
  const auto td_with_lock =
      lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
          catalog, result_table_id);
  auto td = td_with_lock();
  CHECK(td);

  if (td->isView) {
    throw std::runtime_error("Singleton inserts on views is not supported.");
  }
  foreign_storage::validate_non_foreign_table_write(td);

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  RelAlgExecutor ra_executor(executor.get(), catalog);

  ra_executor.executeSimpleInsert(query);
}

void UpdateStmt::analyze(const Catalog_Namespace::Catalog& catalog,
                         Analyzer::Query& query) const {
  throw std::runtime_error("UPDATE statement not supported yet.");
}

void DeleteStmt::analyze(const Catalog_Namespace::Catalog& catalog,
                         Analyzer::Query& query) const {
  throw std::runtime_error("DELETE statement not supported yet.");
}

namespace {

void validate_shard_column_type(const ColumnDescriptor& cd) {
  const auto& col_ti = cd.columnType;
  if (col_ti.is_integer() ||
      (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT) ||
      col_ti.is_time()) {
    return;
  }
  throw std::runtime_error("Cannot shard on type " + col_ti.get_type_name() +
                           ", encoding " + col_ti.get_compression_name());
}

size_t shard_column_index(const std::string& name,
                          const std::list<ColumnDescriptor>& columns) {
  size_t index = 1;
  for (const auto& cd : columns) {
    if (cd.columnName == name) {
      validate_shard_column_type(cd);
      return index;
    }
    ++index;
    if (cd.columnType.is_geometry()) {
      index += cd.columnType.get_physical_cols();
    }
  }
  // Not found, return 0
  return 0;
}

size_t sort_column_index(const std::string& name,
                         const std::list<ColumnDescriptor>& columns) {
  size_t index = 1;
  for (const auto& cd : columns) {
    if (boost::to_upper_copy<std::string>(cd.columnName) == name) {
      return index;
    }
    ++index;
    if (cd.columnType.is_geometry()) {
      index += cd.columnType.get_physical_cols();
    }
  }
  // Not found, return 0
  return 0;
}

void set_string_field(rapidjson::Value& obj,
                      const std::string& field_name,
                      const std::string& field_value,
                      rapidjson::Document& document) {
  rapidjson::Value field_name_json_str;
  field_name_json_str.SetString(
      field_name.c_str(), field_name.size(), document.GetAllocator());
  rapidjson::Value field_value_json_str;
  field_value_json_str.SetString(
      field_value.c_str(), field_value.size(), document.GetAllocator());
  obj.AddMember(field_name_json_str, field_value_json_str, document.GetAllocator());
}

std::string serialize_key_metainfo(
    const ShardKeyDef* shard_key_def,
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
    set_string_field(
        shared_dict_obj, "foreign_table", shared_dict_def.get_foreign_table(), document);
    set_string_field(shared_dict_obj,
                     "foreign_column",
                     shared_dict_def.get_foreign_column(),
                     document);
    arr.PushBack(shared_dict_obj, allocator);
  }
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  arr.Accept(writer);
  return buffer.GetString();
}

template <typename LITERAL_TYPE,
          typename ASSIGNMENT,
          typename VALIDATE = DefaultValidate<LITERAL_TYPE>>
decltype(auto) get_property_value(const NameValueAssign* p,
                                  ASSIGNMENT op,
                                  VALIDATE validate = VALIDATE()) {
  const auto val = validate(p);
  return op(val);
}

decltype(auto) get_storage_type(TableDescriptor& td,
                                const NameValueAssign* p,
                                const std::list<ColumnDescriptor>& columns) {
  auto assignment = [&td](const auto val) { td.storageType = val; };
  return get_property_value<StringLiteral, decltype(assignment), CaseSensitiveValidate>(
      p, assignment);
}

decltype(auto) get_frag_size_def(TableDescriptor& td,
                                 const NameValueAssign* p,
                                 const std::list<ColumnDescriptor>& columns) {
  return get_property_value<IntLiteral>(p,
                                        [&td](const auto val) { td.maxFragRows = val; });
}

decltype(auto) get_frag_size_dataframe_def(DataframeTableDescriptor& df_td,
                                           const NameValueAssign* p,
                                           const std::list<ColumnDescriptor>& columns) {
  return get_property_value<IntLiteral>(
      p, [&df_td](const auto val) { df_td.maxFragRows = val; });
}

decltype(auto) get_max_chunk_size_def(TableDescriptor& td,
                                      const NameValueAssign* p,
                                      const std::list<ColumnDescriptor>& columns) {
  return get_property_value<IntLiteral>(p,
                                        [&td](const auto val) { td.maxChunkSize = val; });
}

decltype(auto) get_max_chunk_size_dataframe_def(
    DataframeTableDescriptor& df_td,
    const NameValueAssign* p,
    const std::list<ColumnDescriptor>& columns) {
  return get_property_value<IntLiteral>(
      p, [&df_td](const auto val) { df_td.maxChunkSize = val; });
}

decltype(auto) get_delimiter_def(DataframeTableDescriptor& df_td,
                                 const NameValueAssign* p,
                                 const std::list<ColumnDescriptor>& columns) {
  return get_property_value<StringLiteral>(p, [&df_td](const auto val) {
    if (val.size() != 1) {
      throw std::runtime_error("Length of DELIMITER must be equal to 1.");
    }
    df_td.delimiter = val;
  });
}

decltype(auto) get_header_def(DataframeTableDescriptor& df_td,
                              const NameValueAssign* p,
                              const std::list<ColumnDescriptor>& columns) {
  return get_property_value<StringLiteral>(p, [&df_td](const auto val) {
    if (val == "FALSE") {
      df_td.hasHeader = false;
    } else if (val == "TRUE") {
      df_td.hasHeader = true;
    } else {
      throw std::runtime_error("Option HEADER support only 'true' or 'false' values.");
    }
  });
}

decltype(auto) get_page_size_def(TableDescriptor& td,
                                 const NameValueAssign* p,
                                 const std::list<ColumnDescriptor>& columns) {
  return get_property_value<IntLiteral>(p,
                                        [&td](const auto val) { td.fragPageSize = val; });
}
decltype(auto) get_max_rows_def(TableDescriptor& td,
                                const NameValueAssign* p,
                                const std::list<ColumnDescriptor>& columns) {
  return get_property_value<IntLiteral>(p, [&td](const auto val) { td.maxRows = val; });
}

decltype(auto) get_skip_rows_def(DataframeTableDescriptor& df_td,
                                 const NameValueAssign* p,
                                 const std::list<ColumnDescriptor>& columns) {
  return get_property_value<IntLiteral>(
      p, [&df_td](const auto val) { df_td.skipRows = val; });
}

decltype(auto) get_partions_def(TableDescriptor& td,
                                const NameValueAssign* p,
                                const std::list<ColumnDescriptor>& columns) {
  return get_property_value<StringLiteral>(p, [&td](const auto partitions_uc) {
    if (partitions_uc != "SHARDED" && partitions_uc != "REPLICATED") {
      throw std::runtime_error("PARTITIONS must be SHARDED or REPLICATED");
    }
    if (td.shardedColumnId != 0 && partitions_uc == "REPLICATED") {
      throw std::runtime_error(
          "A table cannot be sharded and replicated at the same time");
    };
    td.partitions = partitions_uc;
  });
}
decltype(auto) get_shard_count_def(TableDescriptor& td,
                                   const NameValueAssign* p,
                                   const std::list<ColumnDescriptor>& columns) {
  if (!td.shardedColumnId) {
    throw std::runtime_error("SHARD KEY must be defined.");
  }
  return get_property_value<IntLiteral>(p, [&td](const auto shard_count) {
    if (g_leaf_count && shard_count % g_leaf_count) {
      throw std::runtime_error(
          "SHARD_COUNT must be a multiple of the number of leaves in the cluster.");
    }
    td.nShards = g_leaf_count ? shard_count / g_leaf_count : shard_count;
    if (!td.shardedColumnId && !td.nShards) {
      throw std::runtime_error(
          "Must specify the number of shards through the SHARD_COUNT option");
    };
  });
}

decltype(auto) get_vacuum_def(TableDescriptor& td,
                              const NameValueAssign* p,
                              const std::list<ColumnDescriptor>& columns) {
  return get_property_value<StringLiteral>(p, [&td](const auto vacuum_uc) {
    if (vacuum_uc != "IMMEDIATE" && vacuum_uc != "DELAYED") {
      throw std::runtime_error("VACUUM must be IMMEDIATE or DELAYED");
    }
    td.hasDeletedCol = boost::iequals(vacuum_uc, "IMMEDIATE") ? false : true;
  });
}

decltype(auto) get_sort_column_def(TableDescriptor& td,
                                   const NameValueAssign* p,
                                   const std::list<ColumnDescriptor>& columns) {
  return get_property_value<StringLiteral>(p, [&td, &columns](const auto sort_upper) {
    td.sortedColumnId = sort_column_index(sort_upper, columns);
    if (!td.sortedColumnId) {
      throw std::runtime_error("Specified sort column " + sort_upper + " doesn't exist");
    }
  });
}

decltype(auto) get_max_rollback_epochs_def(TableDescriptor& td,
                                           const NameValueAssign* p,
                                           const std::list<ColumnDescriptor>& columns) {
  auto assignment = [&td](const auto val) {
    td.maxRollbackEpochs =
        val < 0 ? -1 : val;  // Anything < 0 means unlimited rollbacks. Note that 0
                             // still means keeping a shadow copy of data/metdata
                             // between epochs so bad writes can be rolled back
  };
  return get_property_value<IntLiteral, decltype(assignment), PositiveOrZeroValidate>(
      p, assignment);
}

static const std::map<const std::string, const TableDefFuncPtr> tableDefFuncMap = {
    {"fragment_size"s, get_frag_size_def},
    {"max_chunk_size"s, get_max_chunk_size_def},
    {"page_size"s, get_page_size_def},
    {"max_rows"s, get_max_rows_def},
    {"partitions"s, get_partions_def},
    {"shard_count"s, get_shard_count_def},
    {"vacuum"s, get_vacuum_def},
    {"sort_column"s, get_sort_column_def},
    {"storage_type"s, get_storage_type},
    {"max_rollback_epochs", get_max_rollback_epochs_def}};

void get_table_definitions(TableDescriptor& td,
                           const std::unique_ptr<NameValueAssign>& p,
                           const std::list<ColumnDescriptor>& columns) {
  const auto it = tableDefFuncMap.find(boost::to_lower_copy<std::string>(*p->get_name()));
  if (it == tableDefFuncMap.end()) {
    throw std::runtime_error(
        "Invalid CREATE TABLE option " + *p->get_name() +
        ". Should be FRAGMENT_SIZE, MAX_CHUNK_SIZE, PAGE_SIZE, MAX_ROLLBACK_EPOCHS, "
        "MAX_ROWS, "
        "PARTITIONS, SHARD_COUNT, VACUUM, SORT_COLUMN, STORAGE_TYPE.");
  }
  return it->second(td, p.get(), columns);
}

void get_table_definitions_for_ctas(TableDescriptor& td,
                                    const std::unique_ptr<NameValueAssign>& p,
                                    const std::list<ColumnDescriptor>& columns) {
  const auto it = tableDefFuncMap.find(boost::to_lower_copy<std::string>(*p->get_name()));
  if (it == tableDefFuncMap.end()) {
    throw std::runtime_error(
        "Invalid CREATE TABLE AS option " + *p->get_name() +
        ". Should be FRAGMENT_SIZE, MAX_CHUNK_SIZE, PAGE_SIZE, MAX_ROLLBACK_EPOCHS, "
        "MAX_ROWS, "
        "PARTITIONS, SHARD_COUNT, VACUUM, SORT_COLUMN, STORAGE_TYPE or "
        "USE_SHARED_DICTIONARIES.");
  }
  return it->second(td, p.get(), columns);
}

static const std::map<const std::string, const DataframeDefFuncPtr> dataframeDefFuncMap =
    {{"fragment_size"s, get_frag_size_dataframe_def},
     {"max_chunk_size"s, get_max_chunk_size_dataframe_def},
     {"skip_rows"s, get_skip_rows_def},
     {"delimiter"s, get_delimiter_def},
     {"header"s, get_header_def}};

void get_dataframe_definitions(DataframeTableDescriptor& df_td,
                               const std::unique_ptr<NameValueAssign>& p,
                               const std::list<ColumnDescriptor>& columns) {
  const auto it =
      dataframeDefFuncMap.find(boost::to_lower_copy<std::string>(*p->get_name()));
  if (it == dataframeDefFuncMap.end()) {
    throw std::runtime_error(
        "Invalid CREATE DATAFRAME option " + *p->get_name() +
        ". Should be FRAGMENT_SIZE, MAX_CHUNK_SIZE, SKIP_ROWS, DELIMITER or HEADER.");
  }
  return it->second(df_td, p.get(), columns);
}

}  // namespace

CreateTableStmt::CreateTableStmt(const rapidjson::Value& payload) {
  CHECK(payload.HasMember("name"));
  table_ = std::make_unique<std::string>(json_str(payload["name"]));
  CHECK(payload.HasMember("elements"));
  CHECK(payload["elements"].IsArray());

  // TODO: support temporary tables
  is_temporary_ = false;

  if_not_exists_ = false;
  if (payload.HasMember("ifNotExists")) {
    if_not_exists_ = json_bool(payload["ifNotExists"]);
  }

  const auto elements = payload["elements"].GetArray();
  for (const auto& element : elements) {
    CHECK(element.IsObject());
    CHECK(element.HasMember("type"));
    if (json_str(element["type"]) == "SQL_COLUMN_DECLARATION") {
      CHECK(element.HasMember("name"));
      auto col_name = std::make_unique<std::string>(json_str(element["name"]));
      CHECK(element.HasMember("sqltype"));
      const auto sql_types = to_sql_type(json_str(element["sqltype"]));

      // decimal / numeric precision / scale
      int precision = -1;
      int scale = -1;
      if (element.HasMember("precision")) {
        precision = json_i64(element["precision"]);
      }
      if (element.HasMember("scale")) {
        scale = json_i64(element["scale"]);
      }

      std::optional<int64_t> array_size;
      if (element.HasMember("arraySize")) {
        // We do not yet support geo arrays
        array_size = json_i64(element["arraySize"]);
      }
      std::unique_ptr<SQLType> sql_type;
      if (element.HasMember("subtype")) {
        CHECK(element.HasMember("coordinateSystem"));
        const auto subtype_sql_types = to_sql_type(json_str(element["subtype"]));
        sql_type = std::make_unique<SQLType>(
            subtype_sql_types,
            static_cast<int>(sql_types),
            static_cast<int>(json_i64(element["coordinateSystem"])),
            false);
      } else if (precision > 0 && scale > 0) {
        sql_type = std::make_unique<SQLType>(sql_types,
                                             precision,
                                             scale,
                                             /*is_array=*/array_size.has_value(),
                                             array_size ? *array_size : -1);
      } else if (precision > 0) {
        sql_type = std::make_unique<SQLType>(sql_types,
                                             precision,
                                             0,
                                             /*is_array=*/array_size.has_value(),
                                             array_size ? *array_size : -1);
      } else {
        sql_type = std::make_unique<SQLType>(sql_types,
                                             /*is_array=*/array_size.has_value(),
                                             array_size ? *array_size : -1);
      }
      CHECK(sql_type);

      CHECK(element.HasMember("nullable"));
      const auto nullable = json_bool(element["nullable"]);
      std::unique_ptr<ColumnConstraintDef> constraint_def;
      if (!nullable) {
        constraint_def = std::make_unique<ColumnConstraintDef>(/*notnull=*/true,
                                                               /*unique=*/false,
                                                               /*primarykey=*/false,
                                                               /*defaultval=*/nullptr);
      }
      std::unique_ptr<CompressDef> compress_def;
      if (element.HasMember("encodingType") && !element["encodingType"].IsNull()) {
        std::string encoding_type = json_str(element["encodingType"]);
        CHECK(element.HasMember("encodingSize"));
        auto encoding_name =
            std::make_unique<std::string>(json_str(element["encodingType"]));
        compress_def = std::make_unique<CompressDef>(encoding_name.release(),
                                                     json_i64(element["encodingSize"]));
      }
      auto col_def = std::make_unique<ColumnDef>(
          col_name.release(),
          sql_type.release(),
          compress_def ? compress_def.release() : nullptr,
          constraint_def ? constraint_def.release() : nullptr);
      table_element_list_.emplace_back(std::move(col_def));
    } else if (json_str(element["type"]) == "SQL_COLUMN_CONSTRAINT") {
      CHECK(element.HasMember("name"));
      if (json_str(element["name"]) == "SHARD_KEY") {
        CHECK(element.HasMember("columns"));
        CHECK(element["columns"].IsArray());
        const auto& columns = element["columns"].GetArray();
        if (columns.Size() != size_t(1)) {
          throw std::runtime_error("Only one shard column is currently supported.");
        }
        auto shard_key_def = std::make_unique<ShardKeyDef>(json_str(columns[0]));
        table_element_list_.emplace_back(std::move(shard_key_def));
      } else if (json_str(element["name"]) == "SHARED_DICT") {
        CHECK(element.HasMember("columns"));
        CHECK(element["columns"].IsArray());
        const auto& columns = element["columns"].GetArray();
        if (columns.Size() != size_t(1)) {
          throw std::runtime_error(
              R"(Only one column per shared dictionary entry is currently supported. Use multiple SHARED DICT statements to share dictionaries from multiple columns.)");
        }
        CHECK(element.HasMember("references") && element["references"].IsObject());
        const auto& references = element["references"].GetObject();
        std::string references_table_name;
        if (references.HasMember("table")) {
          references_table_name = json_str(references["table"]);
        } else {
          references_table_name = *table_;
        }
        CHECK(references.HasMember("column"));

        auto shared_dict_def = std::make_unique<SharedDictionaryDef>(
            json_str(columns[0]), references_table_name, json_str(references["column"]));
        table_element_list_.emplace_back(std::move(shared_dict_def));

      } else {
        LOG(FATAL) << "Unsupported type for SQL_COLUMN_CONSTRAINT: "
                   << json_str(element["name"]);
      }
    } else {
      LOG(FATAL) << "Unsupported element type for CREATE TABLE: "
                 << element["type"].GetString();
    }
  }

  CHECK(payload.HasMember("options"));
  if (payload["options"].IsObject()) {
    for (const auto& option : payload["options"].GetObject()) {
      auto option_name = std::make_unique<std::string>(json_str(option.name));
      std::unique_ptr<Literal> literal_value;
      if (option.value.IsString()) {
        auto literal_string = std::make_unique<std::string>(json_str(option.value));
        literal_value = std::make_unique<StringLiteral>(literal_string.release());
      } else if (option.value.IsInt() || option.value.IsInt64()) {
        literal_value = std::make_unique<IntLiteral>(json_i64(option.value));
      } else if (option.value.IsNull()) {
        literal_value = std::make_unique<NullLiteral>();
      } else {
        throw std::runtime_error("Unable to handle literal for " + *option_name);
      }
      CHECK(literal_value);

      storage_options_.emplace_back(std::make_unique<NameValueAssign>(
          option_name.release(), literal_value.release()));
    }
  } else {
    CHECK(payload["options"].IsNull());
  }
}

void CreateTableStmt::executeDryRun(const Catalog_Namespace::SessionInfo& session,
                                    TableDescriptor& td,
                                    std::list<ColumnDescriptor>& columns,
                                    std::vector<SharedDictionaryDef>& shared_dict_defs) {
  std::unordered_set<std::string> uc_col_names;
  const auto& catalog = session.getCatalog();
  const ShardKeyDef* shard_key_def{nullptr};
  for (auto& e : table_element_list_) {
    if (dynamic_cast<SharedDictionaryDef*>(e.get())) {
      auto shared_dict_def = static_cast<SharedDictionaryDef*>(e.get());
      validate_shared_dictionary(
          this, shared_dict_def, columns, shared_dict_defs, catalog);
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
    if (!dynamic_cast<ColumnDef*>(e.get())) {
      throw std::runtime_error("Table constraints are not supported yet.");
    }
    ColumnDef* coldef = static_cast<ColumnDef*>(e.get());
    ColumnDescriptor cd;
    cd.columnName = *coldef->get_column_name();
    ddl_utils::validate_non_duplicate_column(cd.columnName, uc_col_names);
    setColumnDescriptor(cd, coldef);
    columns.push_back(cd);
  }

  ddl_utils::set_default_table_attributes(*table_, td, columns.size());

  if (shard_key_def) {
    td.shardedColumnId = shard_column_index(shard_key_def->get_column(), columns);
    if (!td.shardedColumnId) {
      throw std::runtime_error("Specified shard column " + shard_key_def->get_column() +
                               " doesn't exist");
    }
  }
  if (is_temporary_) {
    td.persistenceLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
  } else {
    td.persistenceLevel = Data_Namespace::MemoryLevel::DISK_LEVEL;
  }
  if (!storage_options_.empty()) {
    for (auto& p : storage_options_) {
      get_table_definitions(td, p, columns);
    }
  }
  if (td.shardedColumnId && !td.nShards) {
    throw std::runtime_error("SHARD_COUNT needs to be specified with SHARD_KEY.");
  }
  td.keyMetainfo = serialize_key_metainfo(shard_key_def, shared_dict_defs);
}

void CreateTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  // check access privileges
  if (!session.checkDBAccessPrivileges(DBObjectType::TableDBObjectType,
                                       AccessPrivileges::CREATE_TABLE)) {
    throw std::runtime_error("Table " + *table_ +
                             " will not be created. User has no create privileges.");
  }

  if (!catalog.validateNonExistentTableOrView(*table_, if_not_exists_)) {
    return;
  }

  TableDescriptor td;
  std::list<ColumnDescriptor> columns;
  std::vector<SharedDictionaryDef> shared_dict_defs;

  executeDryRun(session, td, columns, shared_dict_defs);
  td.userId = session.get_currentUser().userId;

  catalog.createShardedTable(td, columns, shared_dict_defs);
  // TODO (max): It's transactionally unsafe, should be fixed: we may create object w/o
  // privileges
  SysCatalog::instance().createDBObject(
      session.get_currentUser(), td.tableName, TableDBObjectType, catalog);
}

void CreateDataframeStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  // check access privileges
  if (!session.checkDBAccessPrivileges(DBObjectType::TableDBObjectType,
                                       AccessPrivileges::CREATE_TABLE)) {
    throw std::runtime_error("Table " + *table_ +
                             " will not be created. User has no create privileges.");
  }

  if (catalog.getMetadataForTable(*table_) != nullptr) {
    throw std::runtime_error("Table " + *table_ + " already exists.");
  }
  DataframeTableDescriptor df_td;
  std::list<ColumnDescriptor> columns;
  std::vector<SharedDictionaryDef> shared_dict_defs;

  std::unordered_set<std::string> uc_col_names;
  for (auto& e : table_element_list_) {
    if (dynamic_cast<SharedDictionaryDef*>(e.get())) {
      auto shared_dict_def = static_cast<SharedDictionaryDef*>(e.get());
      validate_shared_dictionary(
          this, shared_dict_def, columns, shared_dict_defs, catalog);
      shared_dict_defs.push_back(*shared_dict_def);
      continue;
    }
    if (!dynamic_cast<ColumnDef*>(e.get())) {
      throw std::runtime_error("Table constraints are not supported yet.");
    }
    ColumnDef* coldef = static_cast<ColumnDef*>(e.get());
    ColumnDescriptor cd;
    cd.columnName = *coldef->get_column_name();
    const auto uc_col_name = boost::to_upper_copy<std::string>(cd.columnName);
    const auto it_ok = uc_col_names.insert(uc_col_name);
    if (!it_ok.second) {
      throw std::runtime_error("Column '" + cd.columnName + "' defined more than once");
    }

    setColumnDescriptor(cd, coldef);
    columns.push_back(cd);
  }

  df_td.tableName = *table_;
  df_td.nColumns = columns.size();
  df_td.isView = false;
  df_td.fragmenter = nullptr;
  df_td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
  df_td.maxFragRows = DEFAULT_FRAGMENT_ROWS;
  df_td.maxChunkSize = DEFAULT_MAX_CHUNK_SIZE;
  df_td.fragPageSize = DEFAULT_PAGE_SIZE;
  df_td.maxRows = DEFAULT_MAX_ROWS;
  df_td.persistenceLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
  if (!storage_options_.empty()) {
    for (auto& p : storage_options_) {
      get_dataframe_definitions(df_td, p, columns);
    }
  }
  df_td.keyMetainfo = serialize_key_metainfo(nullptr, shared_dict_defs);
  df_td.userId = session.get_currentUser().userId;
  df_td.storageType = *filename_;

  catalog.createShardedTable(df_td, columns, shared_dict_defs);
  // TODO (max): It's transactionally unsafe, should be fixed: we may create object w/o
  // privileges
  SysCatalog::instance().createDBObject(
      session.get_currentUser(), df_td.tableName, TableDBObjectType, catalog);
}

std::shared_ptr<ResultSet> getResultSet(QueryStateProxy query_state_proxy,
                                        const std::string select_stmt,
                                        std::vector<TargetMetaInfo>& targets,
                                        bool validate_only = false,
                                        std::vector<size_t> outer_fragment_indices = {}) {
  auto const session = query_state_proxy.getQueryState().getConstSessionInfo();
  auto& catalog = session->getCatalog();

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
#ifdef HAVE_CUDA
  const auto device_type = session->get_executor_device_type();
#else
  const auto device_type = ExecutorDeviceType::CPU;
#endif  // HAVE_CUDA
  auto calcite_mgr = catalog.getCalciteMgr();

  // TODO MAT this should actually get the global or the session parameter for
  // view optimization
  const auto query_ra =
      calcite_mgr
          ->process(query_state_proxy, pg_shim(select_stmt), {}, true, false, false, true)
          .plan_result;
  RelAlgExecutor ra_executor(executor.get(), catalog, query_ra);
  CompilationOptions co = CompilationOptions::defaults(device_type);
  const auto& query_hints = ra_executor.getParsedQueryHints();
  if (query_hints.cpu_mode) {
    co.device_type = ExecutorDeviceType::CPU;
  }
  co.opt_level = ExecutorOptLevel::LoopStrengthReduction;
  // TODO(adb): Need a better method of dropping constants into this ExecutionOptions
  // struct
  ExecutionOptions eo = {false,
                         true,
                         false,
                         true,
                         false,
                         false,
                         validate_only,
                         false,
                         10000,
                         false,
                         false,
                         0.9,
                         false,
                         1000,
                         ExecutorType::Native,
                         outer_fragment_indices};
  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     ExecutorDeviceType::CPU,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     nullptr,
                                                     0,
                                                     0),
                         {}};
  result = ra_executor.executeRelAlgQuery(co, eo, false, nullptr);
  targets = result.getTargetsMeta();

  return result.getRows();
}

size_t LocalConnector::getOuterFragmentCount(QueryStateProxy query_state_proxy,
                                             std::string& sql_query_string) {
  auto const session = query_state_proxy.getQueryState().getConstSessionInfo();
  auto& catalog = session->getCatalog();

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
#ifdef HAVE_CUDA
  const auto device_type = session->get_executor_device_type();
#else
  const auto device_type = ExecutorDeviceType::CPU;
#endif  // HAVE_CUDA
  auto calcite_mgr = catalog.getCalciteMgr();

  // TODO MAT this should actually get the global or the session parameter for
  // view optimization
  const auto query_ra =
      calcite_mgr
          ->process(
              query_state_proxy, pg_shim(sql_query_string), {}, true, false, false, true)
          .plan_result;
  RelAlgExecutor ra_executor(executor.get(), catalog, query_ra);
  const auto& query_hints = ra_executor.getParsedQueryHints();
  CompilationOptions co = {query_hints.cpu_mode ? ExecutorDeviceType::CPU : device_type,
                           true,
                           ExecutorOptLevel::LoopStrengthReduction,
                           false};
  // TODO(adb): Need a better method of dropping constants into this ExecutionOptions
  // struct
  ExecutionOptions eo = {
      false, true, false, true, false, false, false, false, 10000, false, false, 0.9};
  return ra_executor.getOuterFragmentCount(co, eo);
}

AggregatedResult LocalConnector::query(QueryStateProxy query_state_proxy,
                                       std::string& sql_query_string,
                                       std::vector<size_t> outer_frag_indices,
                                       bool validate_only) {
  auto const session = query_state_proxy.getQueryState().getConstSessionInfo();
  // TODO(PS): Should we be using the shimmed query in getResultSet?
  std::string pg_shimmed_select_query = pg_shim(sql_query_string);

  std::vector<TargetMetaInfo> target_metainfos;
  auto result_rows = getResultSet(query_state_proxy,
                                  sql_query_string,
                                  target_metainfos,
                                  validate_only,
                                  outer_frag_indices);
  AggregatedResult res = {result_rows, target_metainfos};
  return res;
}

std::vector<AggregatedResult> LocalConnector::query(
    QueryStateProxy query_state_proxy,
    std::string& sql_query_string,
    std::vector<size_t> outer_frag_indices) {
  auto res = query(query_state_proxy, sql_query_string, outer_frag_indices, false);
  return {res};
}

void LocalConnector::insertDataToLeaf(const Catalog_Namespace::SessionInfo& session,
                                      const size_t leaf_idx,
                                      Fragmenter_Namespace::InsertData& insert_data) {
  CHECK(leaf_idx == 0);
  auto& catalog = session.getCatalog();
  auto created_td = catalog.getMetadataForTable(insert_data.tableId);
  ChunkKey chunkKey = {catalog.getCurrentDB().dbId, created_td->tableId};
  // TODO(adb): Ensure that we have previously obtained a write lock for this table's
  // data.
  created_td->fragmenter->insertDataNoCheckpoint(insert_data);
}

void LocalConnector::checkpoint(const Catalog_Namespace::SessionInfo& session,
                                int table_id) {
  auto& catalog = session.getCatalog();
  catalog.checkpointWithAutoRollback(table_id);
}

void LocalConnector::rollback(const Catalog_Namespace::SessionInfo& session,
                              int table_id) {
  auto& catalog = session.getCatalog();
  auto db_id = catalog.getDatabaseId();
  auto table_epochs = catalog.getTableEpochs(db_id, table_id);
  catalog.setTableEpochs(db_id, table_epochs);
}

std::list<ColumnDescriptor> LocalConnector::getColumnDescriptors(AggregatedResult& result,
                                                                 bool for_create) {
  std::list<ColumnDescriptor> column_descriptors;
  std::list<ColumnDescriptor> column_descriptors_for_create;

  int rowid_suffix = 0;
  for (const auto& target_metainfo : result.targets_meta) {
    ColumnDescriptor cd;
    cd.columnName = target_metainfo.get_resname();
    if (cd.columnName == "rowid") {
      cd.columnName += std::to_string(rowid_suffix++);
    }
    cd.columnType = target_metainfo.get_physical_type_info();

    ColumnDescriptor cd_for_create = cd;

    if (cd.columnType.get_compression() == kENCODING_DICT) {
      // we need to reset the comp param (as this points to the actual dictionary)
      if (cd.columnType.is_array()) {
        // for dict encoded arrays, it is always 4 bytes
        cd_for_create.columnType.set_comp_param(32);
      } else {
        cd_for_create.columnType.set_comp_param(cd.columnType.get_size() * 8);
      }
    }

    if (cd.columnType.is_date() && !cd.columnType.is_date_in_days()) {
      // default to kENCODING_DATE_IN_DAYS encoding
      cd_for_create.columnType.set_compression(kENCODING_DATE_IN_DAYS);
      cd_for_create.columnType.set_comp_param(0);
    }

    column_descriptors_for_create.push_back(cd_for_create);
    column_descriptors.push_back(cd);
  }

  if (for_create) {
    return column_descriptors_for_create;
  }

  return column_descriptors;
}

void InsertIntoTableAsSelectStmt::populateData(QueryStateProxy query_state_proxy,
                                               const TableDescriptor* td,
                                               bool validate_table) {
  auto const session = query_state_proxy.getQueryState().getConstSessionInfo();
  auto& catalog = session->getCatalog();
  foreign_storage::validate_non_foreign_table_write(td);

  LocalConnector local_connector;
  bool populate_table = false;

  if (leafs_connector_) {
    populate_table = true;
  } else {
    leafs_connector_ = &local_connector;
    if (!g_cluster) {
      populate_table = true;
    }
  }

  auto get_target_column_descriptors = [this, &catalog](const TableDescriptor* td) {
    std::vector<const ColumnDescriptor*> target_column_descriptors;
    if (column_list_.empty()) {
      auto list = catalog.getAllColumnMetadataForTable(td->tableId, false, false, false);
      target_column_descriptors = {std::begin(list), std::end(list)};

    } else {
      for (auto& c : column_list_) {
        const ColumnDescriptor* cd = catalog.getMetadataForColumn(td->tableId, *c);
        if (cd == nullptr) {
          throw std::runtime_error("Column " + *c + " does not exist.");
        }
        target_column_descriptors.push_back(cd);
      }
    }

    return target_column_descriptors;
  };

  bool is_temporary = table_is_temporary(td);

  if (validate_table) {
    // check access privileges
    if (!td) {
      throw std::runtime_error("Table " + table_name_ + " does not exist.");
    }
    if (td->isView) {
      throw std::runtime_error("Insert to views is not supported yet.");
    }

    if (!session->checkDBAccessPrivileges(DBObjectType::TableDBObjectType,
                                          AccessPrivileges::INSERT_INTO_TABLE,
                                          table_name_)) {
      throw std::runtime_error("User has no insert privileges on " + table_name_ + ".");
    }

    // only validate the select query so we get the target types
    // correctly, but do not populate the result set
    auto result = local_connector.query(query_state_proxy, select_query_, {}, true);
    auto source_column_descriptors = local_connector.getColumnDescriptors(result, false);

    std::vector<const ColumnDescriptor*> target_column_descriptors =
        get_target_column_descriptors(td);
    if (catalog.getAllColumnMetadataForTable(td->tableId, false, false, false).size() !=
        target_column_descriptors.size()) {
      throw std::runtime_error("Insert into a subset of columns is not supported yet.");
    }

    if (source_column_descriptors.size() != target_column_descriptors.size()) {
      throw std::runtime_error("The number of source and target columns does not match.");
    }

    for (int i = 0; i < source_column_descriptors.size(); i++) {
      const ColumnDescriptor* source_cd =
          &(*std::next(source_column_descriptors.begin(), i));
      const ColumnDescriptor* target_cd = target_column_descriptors.at(i);

      if (source_cd->columnType.get_type() != target_cd->columnType.get_type()) {
        auto type_cannot_be_cast = [](const auto& col_type) {
          return (col_type.is_time() || col_type.is_geometry() || col_type.is_array() ||
                  col_type.is_boolean());
        };

        if (type_cannot_be_cast(source_cd->columnType) ||
            type_cannot_be_cast(target_cd->columnType)) {
          throw std::runtime_error("Source '" + source_cd->columnName + " " +
                                   source_cd->columnType.get_type_name() +
                                   "' and target '" + target_cd->columnName + " " +
                                   target_cd->columnType.get_type_name() +
                                   "' column types do not match.");
        }
      }
      if (source_cd->columnType.is_array()) {
        if (source_cd->columnType.get_subtype() != target_cd->columnType.get_subtype()) {
          throw std::runtime_error("Source '" + source_cd->columnName + " " +
                                   source_cd->columnType.get_type_name() +
                                   "' and target '" + target_cd->columnName + " " +
                                   target_cd->columnType.get_type_name() +
                                   "' array column element types do not match.");
        }
      }

      if (source_cd->columnType.is_decimal() ||
          source_cd->columnType.get_elem_type().is_decimal()) {
        SQLTypeInfo sourceType = source_cd->columnType;
        SQLTypeInfo targetType = target_cd->columnType;

        if (source_cd->columnType.is_array()) {
          sourceType = source_cd->columnType.get_elem_type();
          targetType = target_cd->columnType.get_elem_type();
        }

        if (sourceType.get_scale() != targetType.get_scale()) {
          throw std::runtime_error("Source '" + source_cd->columnName + " " +
                                   source_cd->columnType.get_type_name() +
                                   "' and target '" + target_cd->columnName + " " +
                                   target_cd->columnType.get_type_name() +
                                   "' decimal columns scales do not match.");
        }
      }

      if (source_cd->columnType.is_string()) {
        if (!target_cd->columnType.is_string()) {
          throw std::runtime_error("Source '" + source_cd->columnName + " " +
                                   source_cd->columnType.get_type_name() +
                                   "' and target '" + target_cd->columnName + " " +
                                   target_cd->columnType.get_type_name() +
                                   "' column types do not match.");
        }
        if (source_cd->columnType.get_compression() !=
            target_cd->columnType.get_compression()) {
          throw std::runtime_error("Source '" + source_cd->columnName + " " +
                                   source_cd->columnType.get_type_name() +
                                   "' and target '" + target_cd->columnName + " " +
                                   target_cd->columnType.get_type_name() +
                                   "' columns string encodings do not match.");
        }
      }

      if (source_cd->columnType.is_timestamp() && target_cd->columnType.is_timestamp()) {
        if (source_cd->columnType.get_dimension() !=
            target_cd->columnType.get_dimension()) {
          throw std::runtime_error("Source '" + source_cd->columnName + " " +
                                   source_cd->columnType.get_type_name() +
                                   "' and target '" + target_cd->columnName + " " +
                                   target_cd->columnType.get_type_name() +
                                   "' timestamp column precisions do not match.");
        }
      }

      if (!source_cd->columnType.is_string() && !source_cd->columnType.is_geometry() &&
          !source_cd->columnType.is_integer() && !source_cd->columnType.is_decimal() &&
          !source_cd->columnType.is_date() && !source_cd->columnType.is_time() &&
          !source_cd->columnType.is_timestamp() &&
          source_cd->columnType.get_size() > target_cd->columnType.get_size()) {
        throw std::runtime_error("Source '" + source_cd->columnName + " " +
                                 source_cd->columnType.get_type_name() +
                                 "' and target '" + target_cd->columnName + " " +
                                 target_cd->columnType.get_type_name() +
                                 "' column encoding sizes do not match.");
      }
    }
  }

  if (!populate_table) {
    return;
  }

  int64_t total_row_count = 0;
  int64_t total_source_query_time_ms = 0;
  int64_t total_target_value_translate_time_ms = 0;
  int64_t total_data_load_time_ms = 0;

  Fragmenter_Namespace::InsertDataLoader insertDataLoader(*leafs_connector_);
  auto target_column_descriptors = get_target_column_descriptors(td);

  auto outer_frag_count =
      leafs_connector_->getOuterFragmentCount(query_state_proxy, select_query_);

  size_t outer_frag_end = outer_frag_count == 0 ? 1 : outer_frag_count;

  try {
    for (size_t outer_frag_idx = 0; outer_frag_idx < outer_frag_end; outer_frag_idx++) {
      std::vector<size_t> allowed_outer_fragment_indices;

      if (outer_frag_count) {
        allowed_outer_fragment_indices.push_back(outer_frag_idx);
      }

      const auto query_clock_begin = timer_start();
      std::vector<AggregatedResult> query_results = leafs_connector_->query(
          query_state_proxy, select_query_, allowed_outer_fragment_indices);
      total_source_query_time_ms += timer_stop(query_clock_begin);

      for (auto& res : query_results) {
        auto result_rows = res.rs;
        result_rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
        const auto num_rows = result_rows->rowCount();

        if (0 == num_rows) {
          continue;
        }

        total_row_count += num_rows;

        size_t leaf_count = leafs_connector_->leafCount();

        size_t max_number_of_rows_per_package =
            std::min(num_rows / leaf_count, 64UL * 1024UL);

        size_t start_row = 0;
        size_t num_rows_to_process = std::min(num_rows, max_number_of_rows_per_package);

        // ensure that at least one row is being processed
        num_rows_to_process = std::max(num_rows_to_process, 1UL);

        std::vector<std::unique_ptr<TargetValueConverter>> value_converters;

        TargetValueConverterFactory factory;

        const int num_worker_threads = cpu_threads();

        std::vector<size_t> thread_start_idx(num_worker_threads),
            thread_end_idx(num_worker_threads);
        bool can_go_parallel = !result_rows->isTruncated() && num_rows_to_process > 20000;

        std::atomic<size_t> row_idx{0};

        auto convert_function = [&result_rows,
                                 &value_converters,
                                 &row_idx,
                                 &num_rows_to_process,
                                 &thread_start_idx,
                                 &thread_end_idx](const int thread_id) {
          const int num_cols = value_converters.size();
          const size_t start = thread_start_idx[thread_id];
          const size_t end = thread_end_idx[thread_id];
          size_t idx = 0;
          for (idx = start; idx < end; ++idx) {
            const auto result_row = result_rows->getRowAtNoTranslations(idx);
            if (!result_row.empty()) {
              size_t target_row = row_idx.fetch_add(1);

              if (target_row >= num_rows_to_process) {
                break;
              }

              for (unsigned int col = 0; col < num_cols; col++) {
                const auto& mapd_variant = result_row[col];
                value_converters[col]->convertToColumnarFormat(target_row, &mapd_variant);
              }
            }
          }

          thread_start_idx[thread_id] = idx;
        };

        auto single_threaded_convert_function = [&result_rows,
                                                 &value_converters,
                                                 &row_idx,
                                                 &num_rows_to_process,
                                                 &thread_start_idx,
                                                 &thread_end_idx](const int thread_id) {
          const int num_cols = value_converters.size();
          const size_t start = thread_start_idx[thread_id];
          const size_t end = thread_end_idx[thread_id];
          size_t idx = 0;
          for (idx = start; idx < end; ++idx) {
            size_t target_row = row_idx.fetch_add(1);

            if (target_row >= num_rows_to_process) {
              break;
            }
            const auto result_row = result_rows->getNextRow(false, false);
            CHECK(!result_row.empty());
            for (unsigned int col = 0; col < num_cols; col++) {
              const auto& mapd_variant = result_row[col];
              value_converters[col]->convertToColumnarFormat(target_row, &mapd_variant);
            }
          }

          thread_start_idx[thread_id] = idx;
        };

        if (can_go_parallel) {
          const size_t entryCount = result_rows->entryCount();
          for (size_t i = 0,
                      start_entry = 0,
                      stride = (entryCount + num_worker_threads - 1) / num_worker_threads;
               i < num_worker_threads && start_entry < entryCount;
               ++i, start_entry += stride) {
            const auto end_entry = std::min(start_entry + stride, entryCount);
            thread_start_idx[i] = start_entry;
            thread_end_idx[i] = end_entry;
          }

        } else {
          thread_start_idx[0] = 0;
          thread_end_idx[0] = result_rows->entryCount();
        }

        std::shared_ptr<Executor> executor;

        if (g_enable_experimental_string_functions) {
          executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
        }

        while (start_row < num_rows) {
          value_converters.clear();
          row_idx = 0;
          int colNum = 0;
          for (const auto targetDescriptor : target_column_descriptors) {
            auto sourceDataMetaInfo = res.targets_meta[colNum++];

            ConverterCreateParameter param{
                num_rows_to_process,
                catalog,
                sourceDataMetaInfo,
                targetDescriptor,
                targetDescriptor->columnType,
                !targetDescriptor->columnType.get_notnull(),
                result_rows->getRowSetMemOwner()->getLiteralStringDictProxy(),
                g_enable_experimental_string_functions
                    ? executor->getStringDictionaryProxy(
                          sourceDataMetaInfo.get_type_info().get_comp_param(),
                          result_rows->getRowSetMemOwner(),
                          true)
                    : nullptr};
            auto converter = factory.create(param);
            value_converters.push_back(std::move(converter));
          }

          const auto translate_clock_begin = timer_start();
          if (can_go_parallel) {
            std::vector<std::future<void>> worker_threads;
            for (int i = 0; i < num_worker_threads; ++i) {
              worker_threads.push_back(
                  std::async(std::launch::async, convert_function, i));
            }

            for (auto& child : worker_threads) {
              child.wait();
            }
            for (auto& child : worker_threads) {
              child.get();
            }

          } else {
            single_threaded_convert_function(0);
          }

          // finalize the insert data
          {
            auto finalizer_func =
                [](std::unique_ptr<TargetValueConverter>::pointer targetValueConverter) {
                  targetValueConverter->finalizeDataBlocksForInsertData();
                };
            std::vector<std::future<void>> worker_threads;
            for (auto& converterPtr : value_converters) {
              worker_threads.push_back(
                  std::async(std::launch::async, finalizer_func, converterPtr.get()));
            }

            for (auto& child : worker_threads) {
              child.wait();
            }
            for (auto& child : worker_threads) {
              child.get();
            }
          }

          Fragmenter_Namespace::InsertData insert_data;
          insert_data.databaseId = catalog.getCurrentDB().dbId;
          CHECK(td);
          insert_data.tableId = td->tableId;
          insert_data.numRows = num_rows_to_process;

          for (int col_idx = 0; col_idx < target_column_descriptors.size(); col_idx++) {
            value_converters[col_idx]->addDataBlocksToInsertData(insert_data);
          }
          total_target_value_translate_time_ms += timer_stop(translate_clock_begin);

          const auto data_load_clock_begin = timer_start();
          insertDataLoader.insertData(*session, insert_data);
          total_data_load_time_ms += timer_stop(data_load_clock_begin);

          start_row += num_rows_to_process;
          num_rows_to_process =
              std::min(num_rows - start_row, max_number_of_rows_per_package);
        }
      }
    }
  } catch (...) {
    try {
      leafs_connector_->rollback(*session, td->tableId);
    } catch (std::exception& e) {
      LOG(ERROR) << "An error occurred during ITAS rollback attempt. Table id: "
                 << td->tableId << ", Error: " << e.what();
    }
    throw;
  }

  int64_t total_time_ms = total_source_query_time_ms +
                          total_target_value_translate_time_ms + total_data_load_time_ms;

  VLOG(1) << "CTAS/ITAS " << total_row_count << " rows loaded in " << total_time_ms
          << "ms (outer_frag_count=" << outer_frag_count
          << ", query_time=" << total_source_query_time_ms
          << "ms, translation_time=" << total_target_value_translate_time_ms
          << "ms, data_load_time=" << total_data_load_time_ms
          << "ms)\nquery: " << select_query_;

  if (!is_temporary) {
    leafs_connector_->checkpoint(*session, td->tableId);
  }
}

void InsertIntoTableAsSelectStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto session_copy = session;
  auto session_ptr = std::shared_ptr<Catalog_Namespace::SessionInfo>(
      &session_copy, boost::null_deleter());
  auto query_state = query_state::QueryState::create(session_ptr, select_query_);
  auto stdlog = STDLOG(query_state);

  auto& catalog = session_ptr->getCatalog();

  const auto execute_read_lock = mapd_shared_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  lockmgr::LockedTableDescriptors locks;
  std::vector<std::string> tables;

  // get the table info
  auto calcite_mgr = catalog.getCalciteMgr();

  // TODO MAT this should actually get the global or the session parameter for
  // view optimization
  const auto result = calcite_mgr->process(query_state->createQueryStateProxy(),
                                           pg_shim(select_query_),
                                           {},
                                           true,
                                           false,
                                           false,
                                           true);

  tables.insert(tables.end(),
                result.resolved_accessed_objects.tables_selected_from.begin(),
                result.resolved_accessed_objects.tables_selected_from.end());
  tables.emplace_back(table_name_);

  // force sort into tableid order in case of name change to guarantee fixed order of
  // mutex access
  std::sort(tables.begin(),
            tables.end(),
            [&catalog](const std::string& a, const std::string& b) {
              return catalog.getMetadataForTable(a, false)->tableId <
                     catalog.getMetadataForTable(b, false)->tableId;
            });

  tables.erase(unique(tables.begin(), tables.end()), tables.end());
  for (const auto& table : tables) {
    locks.emplace_back(
        std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>>(
            lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
                catalog, table)));
    if (table == table_name_) {
      // Aquire an insert data lock for updates/deletes, consistent w/ insert. The
      // table data lock will be aquired in the fragmenter during checkpoint.
      locks.emplace_back(
          std::make_unique<lockmgr::TableInsertLockContainer<lockmgr::WriteLock>>(
              lockmgr::TableInsertLockContainer<lockmgr::WriteLock>::acquire(
                  catalog.getDatabaseId(), (*locks.back())())));
    } else {
      locks.emplace_back(
          std::make_unique<lockmgr::TableDataLockContainer<lockmgr::ReadLock>>(
              lockmgr::TableDataLockContainer<lockmgr::ReadLock>::acquire(
                  catalog.getDatabaseId(), (*locks.back())())));
    }
  }

  const TableDescriptor* td = catalog.getMetadataForTable(table_name_);

  populateData(query_state->createQueryStateProxy(), td, true);
}

void CreateTableAsSelectStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto session_copy = session;
  auto session_ptr = std::shared_ptr<Catalog_Namespace::SessionInfo>(
      &session_copy, boost::null_deleter());
  auto query_state = query_state::QueryState::create(session_ptr, select_query_);
  auto stdlog = STDLOG(query_state);

  LocalConnector local_connector;
  auto& catalog = session.getCatalog();
  bool create_table = nullptr == leafs_connector_;

  std::set<std::string> select_tables;
  if (create_table) {
    const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
        *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
            legacylockmgr::ExecutorOuterLock, true));

    // check access privileges
    if (!session.checkDBAccessPrivileges(DBObjectType::TableDBObjectType,
                                         AccessPrivileges::CREATE_TABLE)) {
      throw std::runtime_error("CTAS failed. Table " + table_name_ +
                               " will not be created. User has no create privileges.");
    }

    if (catalog.getMetadataForTable(table_name_) != nullptr) {
      if (if_not_exists_) {
        return;
      }
      throw std::runtime_error("Table " + table_name_ +
                               " already exists and no data was loaded.");
    }

    // get the table info
    auto calcite_mgr = catalog.getCalciteMgr();

    // TODO MAT this should actually get the global or the session parameter for
    // view optimization
    const auto result = calcite_mgr->process(query_state->createQueryStateProxy(),
                                             pg_shim(select_query_),
                                             {},
                                             true,
                                             false,
                                             false,
                                             true);

    select_tables.insert(result.resolved_accessed_objects.tables_selected_from.begin(),
                         result.resolved_accessed_objects.tables_selected_from.end());

    // only validate the select query so we get the target types
    // correctly, but do not populate the result set
    // we currently have exclusive access to the system so this is safe
    auto validate_result = local_connector.query(
        query_state->createQueryStateProxy(), select_query_, {}, true);

    const auto column_descriptors_for_create =
        local_connector.getColumnDescriptors(validate_result, true);

    // some validation as the QE might return some out of range column types
    for (auto& cd : column_descriptors_for_create) {
      if (cd.columnType.is_decimal() && cd.columnType.get_precision() > 18) {
        throw std::runtime_error(cd.columnName + ": Precision too high, max 18.");
      }
    }

    TableDescriptor td;
    td.tableName = table_name_;
    td.userId = session.get_currentUser().userId;
    td.nColumns = column_descriptors_for_create.size();
    td.isView = false;
    td.fragmenter = nullptr;
    td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
    td.maxFragRows = DEFAULT_FRAGMENT_ROWS;
    td.maxChunkSize = DEFAULT_MAX_CHUNK_SIZE;
    td.fragPageSize = DEFAULT_PAGE_SIZE;
    td.maxRows = DEFAULT_MAX_ROWS;
    td.maxRollbackEpochs = DEFAULT_MAX_ROLLBACK_EPOCHS;
    if (is_temporary_) {
      td.persistenceLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
    } else {
      td.persistenceLevel = Data_Namespace::MemoryLevel::DISK_LEVEL;
    }

    bool use_shared_dictionaries = true;

    if (!storage_options_.empty()) {
      for (auto& p : storage_options_) {
        if (boost::to_lower_copy<std::string>(*p->get_name()) ==
            "use_shared_dictionaries") {
          const StringLiteral* literal =
              dynamic_cast<const StringLiteral*>(p->get_value());
          if (nullptr == literal) {
            throw std::runtime_error(
                "USE_SHARED_DICTIONARIES must be a string parameter");
          }
          std::string val = boost::to_lower_copy<std::string>(*literal->get_stringval());
          use_shared_dictionaries = val == "true" || val == "1" || val == "t";
        } else {
          get_table_definitions_for_ctas(td, p, column_descriptors_for_create);
        }
      }
    }

    std::vector<SharedDictionaryDef> sharedDictionaryRefs;

    if (use_shared_dictionaries) {
      const auto source_column_descriptors =
          local_connector.getColumnDescriptors(validate_result, false);
      const auto mapping = catalog.getDictionaryToColumnMapping();

      for (auto& source_cd : source_column_descriptors) {
        const auto& ti = source_cd.columnType;
        if (ti.is_string()) {
          if (ti.get_compression() == kENCODING_DICT) {
            int dict_id = ti.get_comp_param();
            auto it = mapping.find(dict_id);
            if (mapping.end() != it) {
              const auto targetColumn = it->second;
              auto targetTable =
                  catalog.getMetadataForTable(targetColumn->tableId, false);
              CHECK(targetTable);
              LOG(INFO) << "CTAS: sharing text dictionary on column "
                        << source_cd.columnName << " with " << targetTable->tableName
                        << "." << targetColumn->columnName;
              sharedDictionaryRefs.push_back(
                  SharedDictionaryDef(source_cd.columnName,
                                      targetTable->tableName,
                                      targetColumn->columnName));
            }
          }
        }
      }
    }

    // currently no means of defining sharding in CTAS
    td.keyMetainfo = serialize_key_metainfo(nullptr, sharedDictionaryRefs);

    catalog.createTable(td, column_descriptors_for_create, sharedDictionaryRefs, true);
    // TODO (max): It's transactionally unsafe, should be fixed: we may create object
    // w/o privileges
    SysCatalog::instance().createDBObject(
        session.get_currentUser(), td.tableName, TableDBObjectType, catalog);
  }

  // note there is a time where we do not have any executor outer lock here. someone could
  // come along and mess with the data or other tables.
  const auto execute_read_lock = mapd_shared_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  lockmgr::LockedTableDescriptors locks;
  std::vector<std::string> tables;
  tables.insert(tables.end(), select_tables.begin(), select_tables.end());
  CHECK_EQ(tables.size(), select_tables.size());
  tables.emplace_back(table_name_);
  // force sort into tableid order in case of name change to guarantee fixed order of
  // mutex access
  std::sort(tables.begin(),
            tables.end(),
            [&catalog](const std::string& a, const std::string& b) {
              return catalog.getMetadataForTable(a, false)->tableId <
                     catalog.getMetadataForTable(b, false)->tableId;
            });
  tables.erase(unique(tables.begin(), tables.end()), tables.end());
  for (const auto& table : tables) {
    locks.emplace_back(
        std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>>(
            lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
                catalog, table)));
    if (table == table_name_) {
      // Aquire an insert data lock for updates/deletes, consistent w/ insert. The
      // table data lock will be aquired in the fragmenter during checkpoint.
      locks.emplace_back(
          std::make_unique<lockmgr::TableInsertLockContainer<lockmgr::WriteLock>>(
              lockmgr::TableInsertLockContainer<lockmgr::WriteLock>::acquire(
                  catalog.getDatabaseId(), (*locks.back())())));
    } else {
      locks.emplace_back(
          std::make_unique<lockmgr::TableDataLockContainer<lockmgr::ReadLock>>(
              lockmgr::TableDataLockContainer<lockmgr::ReadLock>::acquire(
                  catalog.getDatabaseId(), (*locks.back())())));
    }
  }

  const TableDescriptor* td = catalog.getMetadataForTable(table_name_);

  try {
    populateData(query_state->createQueryStateProxy(), td, false);
  } catch (...) {
    if (!g_cluster) {
      const TableDescriptor* created_td = catalog.getMetadataForTable(table_name_);
      if (created_td) {
        catalog.dropTable(created_td);
      }
    }
    throw;
  }
}

DropTableStmt::DropTableStmt(const rapidjson::Value& payload) {
  CHECK(payload.HasMember("tableName"));
  table = std::make_unique<std::string>(json_str(payload["tableName"]));

  if_exists = false;
  if (payload.HasMember("ifExists")) {
    if_exists = json_bool(payload["ifExists"]);
  }
}

void DropTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  // TODO(adb): the catalog should be handling this locking.
  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  const TableDescriptor* td{nullptr};
  std::unique_ptr<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>> td_with_lock;

  try {
    td_with_lock =
        std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>>(
            lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
                catalog, *table, false));
    td = (*td_with_lock)();
  } catch (const std::runtime_error& e) {
    if (if_exists) {
      return;
    } else {
      throw e;
    }
  }

  CHECK(td);
  CHECK(td_with_lock);

  // check access privileges
  if (!session.checkDBAccessPrivileges(
          DBObjectType::TableDBObjectType, AccessPrivileges::DROP_TABLE, *table)) {
    throw std::runtime_error("Table " + *table +
                             " will not be dropped. User has no proper privileges.");
  }

  ddl_utils::validate_table_type(td, ddl_utils::TableType::TABLE, "DROP");

  auto table_data_write_lock =
      lockmgr::TableDataLockMgr::getWriteLockForTable(catalog, *table);
  catalog.dropTable(td);

  // invalidate cached hashtable
  DeleteTriggeredCacheInvalidator::invalidateCaches();
}

void TruncateTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  // TODO: Removal of the FileMgr is not thread safe. Take a global system write lock
  // when truncating a table
  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  const auto td_with_lock =
      lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
          catalog, *table, true);
  const auto td = td_with_lock();
  if (!td) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  }

  // check access privileges
  std::vector<DBObject> privObjects;
  DBObject dbObject(*table, TableDBObjectType);
  dbObject.loadKey(catalog);
  dbObject.setPrivileges(AccessPrivileges::TRUNCATE_TABLE);
  privObjects.push_back(dbObject);
  if (!SysCatalog::instance().checkPrivileges(session.get_currentUser(), privObjects)) {
    throw std::runtime_error("Table " + *table + " will not be truncated. User " +
                             session.get_currentUser().userLoggable() +
                             " has no proper privileges.");
  }

  if (td->isView) {
    throw std::runtime_error(*table + " is a view.  Cannot Truncate.");
  }
  foreign_storage::validate_non_foreign_table_write(td);
  auto table_data_write_lock =
      lockmgr::TableDataLockMgr::getWriteLockForTable(catalog, *table);
  catalog.truncateTable(td);

  // invalidate cached hashtable
  DeleteTriggeredCacheInvalidator::invalidateCaches();
}

void check_alter_table_privilege(const Catalog_Namespace::SessionInfo& session,
                                 const TableDescriptor* td) {
  if (session.get_currentUser().isSuper ||
      session.get_currentUser().userId == td->userId) {
    return;
  }
  std::vector<DBObject> privObjects;
  DBObject dbObject(td->tableName, TableDBObjectType);
  dbObject.loadKey(session.getCatalog());
  dbObject.setPrivileges(AccessPrivileges::ALTER_TABLE);
  privObjects.push_back(dbObject);
  if (!SysCatalog::instance().checkPrivileges(session.get_currentUser(), privObjects)) {
    throw std::runtime_error("Current user does not have the privilege to alter table: " +
                             td->tableName);
  }
}

void RenameUserStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  if (!session.get_currentUser().isSuper) {
    throw std::runtime_error("Only a super user can rename users.");
  }

  Catalog_Namespace::UserMetadata user;
  if (!SysCatalog::instance().getMetadataForUser(*username_, user)) {
    throw std::runtime_error("User " + *username_ + " does not exist.");
  }

  SysCatalog::instance().renameUser(*username_, *new_username_);
}

void RenameDatabaseStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  Catalog_Namespace::DBMetadata db;

  // TODO: use database lock instead
  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  if (!SysCatalog::instance().getMetadataForDB(*database_name_, db)) {
    throw std::runtime_error("Database " + *database_name_ + " does not exist.");
  }

  if (!session.get_currentUser().isSuper &&
      session.get_currentUser().userId != db.dbOwner) {
    throw std::runtime_error("Only a super user or the owner can rename the database.");
  }

  SysCatalog::instance().renameDatabase(*database_name_, *new_database_name_);
}

void RenameTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  // TODO(adb): the catalog should be handling this locking (see AddColumStmt)
  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  const auto td_with_lock =
      lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
          catalog, *table, false);
  const auto td = td_with_lock();
  CHECK(td);
  validate_table_type(td, ddl_utils::TableType::TABLE, "ALTER");

  check_alter_table_privilege(session, td);
  if (catalog.getMetadataForTable(*new_table_name) != nullptr) {
    throw std::runtime_error("Table or View " + *new_table_name + " already exists.");
  }
  catalog.renameTable(td, *new_table_name);
}

void DDLStmt::setColumnDescriptor(ColumnDescriptor& cd, const ColumnDef* coldef) {
  bool not_null;
  const ColumnConstraintDef* cc = coldef->get_column_constraint();
  if (cc == nullptr) {
    not_null = false;
  } else {
    not_null = cc->get_notnull();
  }
  ddl_utils::set_column_descriptor(*coldef->get_column_name(),
                                   cd,
                                   coldef->get_column_type(),
                                   not_null,
                                   coldef->get_compression());
}

void AddColumnStmt::check_executable(const Catalog_Namespace::SessionInfo& session,
                                     const TableDescriptor* td) {
  auto& catalog = session.getCatalog();
  if (!td) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  } else {
    if (td->isView) {
      throw std::runtime_error("Adding columns to a view is not supported.");
    }
    validate_table_type(td, ddl_utils::TableType::TABLE, "ALTER");
    if (table_is_temporary(td)) {
      throw std::runtime_error(
          "Adding columns to temporary tables is not yet supported.");
    }
  };

  check_alter_table_privilege(session, td);

  if (0 == coldefs.size()) {
    coldefs.push_back(std::move(coldef));
  }

  for (const auto& coldef : coldefs) {
    auto& new_column_name = *coldef->get_column_name();
    if (catalog.getMetadataForColumn(td->tableId, new_column_name) != nullptr) {
      throw std::runtime_error("Column " + new_column_name + " already exists.");
    }
  }
}

void AddColumnStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  // TODO(adb): the catalog should be handling this locking.
  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  const auto td_with_lock =
      lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
          catalog, *table, true);
  const auto td = td_with_lock();

  check_executable(session, td);

  CHECK(td->fragmenter);
  if (std::dynamic_pointer_cast<Fragmenter_Namespace::SortedOrderFragmenter>(
          td->fragmenter)) {
    throw std::runtime_error(
        "Adding columns to a table is not supported when using the \"sort_column\" "
        "option.");
  }

  // Do not take a data write lock, as the fragmenter may call `deleteFragments` during
  // a cap operation. Note that the schema write lock will prevent concurrent inserts
  // along with all other queries.

  catalog.getSqliteConnector().query("BEGIN TRANSACTION");
  try {
    std::map<const std::string, const ColumnDescriptor> cds;
    std::map<const int, const ColumnDef*> cid_coldefs;
    for (const auto& coldef : coldefs) {
      ColumnDescriptor cd;
      setColumnDescriptor(cd, coldef.get());
      catalog.addColumn(*td, cd);
      cds.emplace(*coldef->get_column_name(), cd);
      cid_coldefs.emplace(cd.columnId, coldef.get());

      // expand geo column to phy columns
      if (cd.columnType.is_geometry()) {
        std::list<ColumnDescriptor> phy_geo_columns;
        catalog.expandGeoColumn(cd, phy_geo_columns);
        for (auto& cd : phy_geo_columns) {
          catalog.addColumn(*td, cd);
          cds.emplace(cd.columnName, cd);
          cid_coldefs.emplace(cd.columnId, nullptr);
        }
      }
    }

    std::unique_ptr<import_export::Loader> loader(new import_export::Loader(catalog, td));
    auto import_buffers = import_export::setup_column_loaders(td, loader.get());
    loader->setReplicating(true);

    // set_geo_physical_import_buffer below needs a sorted import_buffers
    std::sort(import_buffers.begin(),
              import_buffers.end(),
              [](decltype(import_buffers[0])& a, decltype(import_buffers[0])& b) {
                return a->getColumnDesc()->columnId < b->getColumnDesc()->columnId;
              });

    size_t nrows = td->fragmenter->getNumRows();
    // if sharded, get total nrows from all sharded tables
    if (td->nShards > 0) {
      const auto physical_tds = catalog.getPhysicalTablesDescriptors(td);
      nrows = 0;
      std::for_each(physical_tds.begin(), physical_tds.end(), [&nrows](const auto& td) {
        nrows += td->fragmenter->getNumRows();
      });
    }
    if (nrows > 0) {
      int skip_physical_cols = 0;
      for (const auto cit : cid_coldefs) {
        const auto cd = catalog.getMetadataForColumn(td->tableId, cit.first);
        const auto coldef = cit.second;
        const auto column_constraint = coldef ? coldef->get_column_constraint() : nullptr;
        std::string defaultval = "";
        if (column_constraint) {
          auto defaultlp = column_constraint->get_defaultval();
          auto defaultsp = dynamic_cast<const StringLiteral*>(defaultlp);
          defaultval = defaultsp ? *defaultsp->get_stringval()
                                 : defaultlp ? defaultlp->to_string() : "";
        }
        bool isnull = column_constraint ? (0 == defaultval.size()) : true;
        if (boost::to_upper_copy<std::string>(defaultval) == "NULL") {
          isnull = true;
        }

        if (isnull) {
          if (column_constraint && column_constraint->get_notnull()) {
            throw std::runtime_error("Default value required for column " +
                                     cd->columnName + " (NULL value not supported)");
          }
        }

        for (auto it = import_buffers.begin(); it < import_buffers.end(); ++it) {
          auto& import_buffer = *it;
          if (cd->columnId == import_buffer->getColumnDesc()->columnId) {
            if (coldef != nullptr ||
                skip_physical_cols-- <= 0) {  // skip non-null phy col
              import_buffer->add_value(
                  cd, defaultval, isnull, import_export::CopyParams(), nrows);
              if (cd->columnType.is_geometry()) {
                std::vector<double> coords, bounds;
                std::vector<int> ring_sizes, poly_rings;
                int render_group = 0;
                SQLTypeInfo tinfo{cd->columnType};
                if (!Geospatial::GeoTypesFactory::getGeoColumns(defaultval,
                                                                tinfo,
                                                                coords,
                                                                bounds,
                                                                ring_sizes,
                                                                poly_rings,
                                                                false)) {
                  throw std::runtime_error("Bad geometry data: '" + defaultval + "'");
                }
                size_t col_idx = 1 + std::distance(import_buffers.begin(), it);
                import_export::Importer::set_geo_physical_import_buffer(catalog,
                                                                        cd,
                                                                        import_buffers,
                                                                        col_idx,
                                                                        coords,
                                                                        bounds,
                                                                        ring_sizes,
                                                                        poly_rings,
                                                                        render_group,
                                                                        nrows);
                // skip following phy cols
                skip_physical_cols = cd->columnType.get_physical_cols();
              }
            }
            break;
          }
        }
      }
    }

    if (!loader->loadNoCheckpoint(import_buffers, nrows)) {
      throw std::runtime_error("loadNoCheckpoint failed!");
    }
    catalog.roll(true);
    loader->checkpoint();
    catalog.getSqliteConnector().query("END TRANSACTION");
  } catch (...) {
    catalog.roll(false);
    catalog.getSqliteConnector().query("ROLLBACK TRANSACTION");
    throw;
  }
}

void DropColumnStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  // TODO(adb): the catalog should be handling this locking.
  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  const auto td_with_lock =
      lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
          catalog, *table, true);
  const auto td = td_with_lock();
  if (!td) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  }
  validate_table_type(td, ddl_utils::TableType::TABLE, "ALTER");
  if (td->isView) {
    throw std::runtime_error("Dropping a column from a view is not supported.");
  }
  if (table_is_temporary(td)) {
    throw std::runtime_error(
        "Dropping a column from a temporary table is not yet supported.");
  }

  check_alter_table_privilege(session, td);

  for (const auto& column : columns) {
    if (nullptr == catalog.getMetadataForColumn(td->tableId, *column)) {
      throw std::runtime_error("Column " + *column + " does not exist.");
    }
  }

  if (td->nColumns <= (td->hasDeletedCol ? 3 : 2)) {
    throw std::runtime_error("Table " + *table + " has only one column.");
  }

  catalog.getSqliteConnector().query("BEGIN TRANSACTION");
  try {
    std::vector<int> columnIds;
    for (const auto& column : columns) {
      ColumnDescriptor cd = *catalog.getMetadataForColumn(td->tableId, *column);
      if (td->nShards > 0 && td->shardedColumnId == cd.columnId) {
        throw std::runtime_error("Dropping sharding column " + cd.columnName +
                                 " is not supported.");
      }
      catalog.dropColumn(*td, cd);
      columnIds.push_back(cd.columnId);
      for (int i = 0; i < cd.columnType.get_physical_cols(); i++) {
        const auto pcd = catalog.getMetadataForColumn(td->tableId, cd.columnId + i + 1);
        CHECK(pcd);
        catalog.dropColumn(*td, *pcd);
        columnIds.push_back(cd.columnId + i + 1);
      }
    }

    for (auto shard : catalog.getPhysicalTablesDescriptors(td)) {
      shard->fragmenter->dropColumns(columnIds);
    }
    // if test forces to rollback
    if (g_test_drop_column_rollback) {
      throw std::runtime_error("lol!");
    }
    catalog.roll(true);
    if (td->persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
      catalog.checkpoint(td->tableId);
    }
    catalog.getSqliteConnector().query("END TRANSACTION");
  } catch (...) {
    catalog.setForReload(td->tableId);
    catalog.roll(false);
    catalog.getSqliteConnector().query("ROLLBACK TRANSACTION");
    throw;
  }

  // invalidate cached hashtable
  DeleteTriggeredCacheInvalidator::invalidateCaches();
}

void RenameColumnStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  const auto td_with_lock =
      lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
          catalog, *table, false);
  const auto td = td_with_lock();
  CHECK(td);
  validate_table_type(td, ddl_utils::TableType::TABLE, "ALTER");

  check_alter_table_privilege(session, td);
  const ColumnDescriptor* cd = catalog.getMetadataForColumn(td->tableId, *column);
  if (cd == nullptr) {
    throw std::runtime_error("Column " + *column + " does not exist.");
  }
  if (catalog.getMetadataForColumn(td->tableId, *new_column_name) != nullptr) {
    throw std::runtime_error("Column " + *new_column_name + " already exists.");
  }
  catalog.renameColumn(td, cd, *new_column_name);
}

void AlterTableParamStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  enum TableParamType { MaxRollbackEpochs, Epoch };
  static const std::unordered_map<std::string, TableParamType> param_map = {
      {"max_rollback_epochs", TableParamType::MaxRollbackEpochs},
      {"epoch", TableParamType::Epoch}};
  // Below is to ensure that executor is not currently executing on table when we might be
  // changing it's storage. Question: will/should catalog write lock take care of this?
  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));
  auto& catalog = session.getCatalog();
  const auto td_with_lock =
      lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
          catalog, *table, false);
  const auto td = td_with_lock();
  if (!td) {
    throw std::runtime_error("Table " + *table + " does not exist.");
  }
  if (td->isView) {
    throw std::runtime_error("Setting parameters for a view is not supported.");
  }
  if (table_is_temporary(td)) {
    throw std::runtime_error(
        "Setting parameters for a temporary table is not yet supported.");
  }
  check_alter_table_privilege(session, td);

  // Only allow max_rollback_epochs for now
  std::string param_name(*param->get_name());
  boost::algorithm::to_lower(param_name);
  const IntLiteral* val_int_literal = dynamic_cast<const IntLiteral*>(param->get_value());
  if (val_int_literal == nullptr) {
    throw std::runtime_error("Table parameters should be integers.");
  }
  const int32_t param_val = val_int_literal->get_intval();

  const auto param_it = param_map.find(param_name);
  if (param_it == param_map.end()) {
    throw std::runtime_error(param_name + " is not a settable table parameter.");
  }
  switch (param_it->second) {
    case MaxRollbackEpochs: {
      catalog.setMaxRollbackEpochs(td->tableId, param_val);
      break;
    }
    case Epoch: {
      catalog.setTableEpoch(catalog.getDatabaseId(), td->tableId, param_val);
    }
  }
}

void CopyTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto importer_factory = [](Catalog_Namespace::Catalog& catalog,
                             const TableDescriptor* td,
                             const std::string& file_path,
                             const import_export::CopyParams& copy_params) {
    return std::make_unique<import_export::Importer>(catalog, td, file_path, copy_params);
  };
  return execute(session, importer_factory);
}

void CopyTableStmt::execute(const Catalog_Namespace::SessionInfo& session,
                            const std::function<std::unique_ptr<import_export::Importer>(
                                Catalog_Namespace::Catalog&,
                                const TableDescriptor*,
                                const std::string&,
                                const import_export::CopyParams&)>& importer_factory) {
  boost::regex non_local_file_regex{R"(^\s*(s3|http|https)://.+)",
                                    boost::regex::extended | boost::regex::icase};
  if (!boost::regex_match(*file_pattern, non_local_file_regex)) {
    ddl_utils::validate_allowed_file_path(
        *file_pattern, ddl_utils::DataTransferType::IMPORT, true);
  }

  size_t rows_completed = 0;
  size_t rows_rejected = 0;
  size_t total_time = 0;
  bool load_truncated = false;

  // Prevent simultaneous import / truncate (see TruncateTableStmt::execute)
  const auto execute_read_lock = mapd_shared_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  const TableDescriptor* td{nullptr};
  std::unique_ptr<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>> td_with_lock;
  std::unique_ptr<lockmgr::WriteLock> insert_data_lock;

  auto& catalog = session.getCatalog();

  try {
    td_with_lock = std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>>(
        lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
            catalog, *table));
    td = (*td_with_lock)();
    insert_data_lock = std::make_unique<lockmgr::WriteLock>(
        lockmgr::InsertDataLockMgr::getWriteLockForTable(catalog, *table));
  } catch (const std::runtime_error& e) {
    // noop
    // TODO(adb): We're really only interested in whether the table exists or not.
    // Create a more refined exception.
  }

  // if the table already exists, it's locked, so check access privileges
  if (td) {
    std::vector<DBObject> privObjects;
    DBObject dbObject(*table, TableDBObjectType);
    dbObject.loadKey(catalog);
    dbObject.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE);
    privObjects.push_back(dbObject);
    if (!SysCatalog::instance().checkPrivileges(session.get_currentUser(), privObjects)) {
      throw std::runtime_error("Violation of access privileges: user " +
                               session.get_currentUser().userLoggable() +
                               " has no insert privileges for table " + *table + ".");
    }
  }

  // since we'll have not only posix file names but also s3/hdfs/... url
  // we do not expand wildcard or check file existence here.
  // from here on, file_path contains something which may be a url
  // or a wildcard of file names;
  std::string file_path = *file_pattern;
  import_export::CopyParams copy_params;
  if (!options.empty()) {
    for (auto& p : options) {
      if (boost::iequals(*p->get_name(), "max_reject")) {
        const IntLiteral* int_literal = dynamic_cast<const IntLiteral*>(p->get_value());
        if (int_literal == nullptr) {
          throw std::runtime_error("max_reject option must be an integer.");
        }
        copy_params.max_reject = int_literal->get_intval();
      } else if (boost::iequals(*p->get_name(), "buffer_size")) {
        const IntLiteral* int_literal = dynamic_cast<const IntLiteral*>(p->get_value());
        if (int_literal == nullptr) {
          throw std::runtime_error("buffer_size option must be an integer.");
        }
        copy_params.buffer_size = int_literal->get_intval();
      } else if (boost::iequals(*p->get_name(), "threads")) {
        const IntLiteral* int_literal = dynamic_cast<const IntLiteral*>(p->get_value());
        if (int_literal == nullptr) {
          throw std::runtime_error("Threads option must be an integer.");
        }
        copy_params.threads = int_literal->get_intval();
      } else if (boost::iequals(*p->get_name(), "delimiter")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Delimiter option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Delimiter must be a single character string.");
        }
        copy_params.delimiter = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "nulls")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Nulls option must be a string.");
        }
        copy_params.null_str = *str_literal->get_stringval();
      } else if (boost::iequals(*p->get_name(), "header")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Header option must be a boolean.");
        }
        copy_params.has_header = bool_from_string_literal(str_literal)
                                     ? import_export::ImportHeaderRow::HAS_HEADER
                                     : import_export::ImportHeaderRow::NO_HEADER;
#ifdef ENABLE_IMPORT_PARQUET
      } else if (boost::iequals(*p->get_name(), "parquet")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Parquet option must be a boolean.");
        }
        if (bool_from_string_literal(str_literal)) {
          // not sure a parquet "table" type is proper, but to make code
          // look consistent in some places, let's set "table" type too
          copy_params.file_type = import_export::FileType::PARQUET;
        }
#endif  // ENABLE_IMPORT_PARQUET
      } else if (boost::iequals(*p->get_name(), "s3_access_key")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Option s3_access_key must be a string.");
        }
        copy_params.s3_access_key = *str_literal->get_stringval();
      } else if (boost::iequals(*p->get_name(), "s3_secret_key")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Option s3_secret_key must be a string.");
        }
        copy_params.s3_secret_key = *str_literal->get_stringval();
      } else if (boost::iequals(*p->get_name(), "s3_region")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Option s3_region must be a string.");
        }
        copy_params.s3_region = *str_literal->get_stringval();
      } else if (boost::iequals(*p->get_name(), "s3_endpoint")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Option s3_endpoint must be a string.");
        }
        copy_params.s3_endpoint = *str_literal->get_stringval();
      } else if (boost::iequals(*p->get_name(), "quote")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Quote option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Quote must be a single character string.");
        }
        copy_params.quote = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "escape")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Escape option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Escape must be a single character string.");
        }
        copy_params.escape = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "line_delimiter")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Line_delimiter option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Line_delimiter must be a single character string.");
        }
        copy_params.line_delim = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "quoted")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Quoted option must be a boolean.");
        }
        copy_params.quoted = bool_from_string_literal(str_literal);
      } else if (boost::iequals(*p->get_name(), "plain_text")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("plain_text option must be a boolean.");
        }
        copy_params.plain_text = bool_from_string_literal(str_literal);
      } else if (boost::iequals(*p->get_name(), "array_marker")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Array Marker option must be a string.");
        } else if (str_literal->get_stringval()->length() != 2) {
          throw std::runtime_error(
              "Array Marker option must be exactly two characters.  Default is {}.");
        }
        copy_params.array_begin = (*str_literal->get_stringval())[0];
        copy_params.array_end = (*str_literal->get_stringval())[1];
      } else if (boost::iequals(*p->get_name(), "array_delimiter")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Array Delimiter option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Array Delimiter must be a single character string.");
        }
        copy_params.array_delim = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "lonlat")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Lonlat option must be a boolean.");
        }
        copy_params.lonlat = bool_from_string_literal(str_literal);
      } else if (boost::iequals(*p->get_name(), "geo")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Geo option must be a boolean.");
        }
        copy_params.file_type = bool_from_string_literal(str_literal)
                                    ? import_export::FileType::POLYGON
                                    : import_export::FileType::DELIMITED;
      } else if (boost::iequals(*p->get_name(), "geo_coords_type")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("'geo_coords_type' option must be a string");
        }
        const std::string* s = str_literal->get_stringval();
        if (boost::iequals(*s, "geography")) {
          throw std::runtime_error(
              "GEOGRAPHY coords type not yet supported. Please use GEOMETRY.");
          // copy_params.geo_coords_type = kGEOGRAPHY;
        } else if (boost::iequals(*s, "geometry")) {
          copy_params.geo_coords_type = kGEOMETRY;
        } else {
          throw std::runtime_error(
              "Invalid string for 'geo_coords_type' option (must be 'GEOGRAPHY' or "
              "'GEOMETRY'): " +
              *s);
        }
      } else if (boost::iequals(*p->get_name(), "geo_coords_encoding")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("'geo_coords_encoding' option must be a string");
        }
        const std::string* s = str_literal->get_stringval();
        if (boost::iequals(*s, "none")) {
          copy_params.geo_coords_encoding = kENCODING_NONE;
          copy_params.geo_coords_comp_param = 0;
        } else if (boost::iequals(*s, "compressed(32)")) {
          copy_params.geo_coords_encoding = kENCODING_GEOINT;
          copy_params.geo_coords_comp_param = 32;
        } else {
          throw std::runtime_error(
              "Invalid string for 'geo_coords_encoding' option (must be 'NONE' or "
              "'COMPRESSED(32)'): " +
              *s);
        }
      } else if (boost::iequals(*p->get_name(), "geo_coords_srid")) {
        const IntLiteral* int_literal = dynamic_cast<const IntLiteral*>(p->get_value());
        if (int_literal == nullptr) {
          throw std::runtime_error("'geo_coords_srid' option must be an integer");
        }
        const int srid = int_literal->get_intval();
        if (srid == 4326 || srid == 3857 || srid == 900913) {
          copy_params.geo_coords_srid = srid;
        } else {
          throw std::runtime_error(
              "Invalid value for 'geo_coords_srid' option (must be 4326, 3857, or "
              "900913): " +
              std::to_string(srid));
        }
      } else if (boost::iequals(*p->get_name(), "geo_layer_name")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("'geo_layer_name' option must be a string");
        }
        const std::string* layer_name = str_literal->get_stringval();
        if (layer_name) {
          copy_params.geo_layer_name = *layer_name;
        } else {
          throw std::runtime_error("Invalid value for 'geo_layer_name' option");
        }
      } else if (boost::iequals(*p->get_name(), "partitions")) {
        if (copy_params.file_type == import_export::FileType::POLYGON) {
          const auto partitions =
              static_cast<const StringLiteral*>(p->get_value())->get_stringval();
          CHECK(partitions);
          const auto partitions_uc = boost::to_upper_copy<std::string>(*partitions);
          if (partitions_uc != "REPLICATED") {
            throw std::runtime_error("PARTITIONS must be REPLICATED for geo COPY");
          }
          _geo_copy_from_partitions = partitions_uc;
        } else {
          throw std::runtime_error("PARTITIONS option not supported for non-geo COPY: " +
                                   *p->get_name());
        }
      } else if (boost::iequals(*p->get_name(), "geo_assign_render_groups")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("geo_assign_render_groups option must be a boolean.");
        }
        copy_params.geo_assign_render_groups = bool_from_string_literal(str_literal);
      } else if (boost::iequals(*p->get_name(), "geo_explode_collections")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("geo_explode_collections option must be a boolean.");
        }
        copy_params.geo_explode_collections = bool_from_string_literal(str_literal);
      } else if (boost::iequals(*p->get_name(), "source_srid")) {
        const IntLiteral* int_literal = dynamic_cast<const IntLiteral*>(p->get_value());
        if (int_literal == nullptr) {
          throw std::runtime_error("'source_srid' option must be an integer");
        }
        const int srid = int_literal->get_intval();
        if (copy_params.file_type == import_export::FileType::DELIMITED) {
          copy_params.source_srid = srid;
        } else {
          throw std::runtime_error(
              "'source_srid' option can only be used on csv/tsv files");
        }
      } else {
        throw std::runtime_error("Invalid option for COPY: " + *p->get_name());
      }
    }
  }

  std::string tr;
  if (copy_params.file_type == import_export::FileType::POLYGON) {
    // geo import
    // we do nothing here, except stash the parameters so we can
    // do the import when we unwind to the top of the handler
    _geo_copy_from_file_name = file_path;
    _geo_copy_from_copy_params = copy_params;
    _was_geo_copy_from = true;

    // the result string
    // @TODO simon.eves put something more useful in here
    // except we really can't because we haven't done the import yet!
    if (td) {
      tr = std::string("Appending geo to table '") + *table + std::string("'...");
    } else {
      tr = std::string("Creating table '") + *table +
           std::string("' and importing geo...");
    }
  } else {
    if (td) {
      CHECK(td_with_lock);

      // regular import
      auto importer = importer_factory(catalog, td, file_path, copy_params);
      auto ms = measure<>::execution([&]() {
        auto res = importer->import();
        rows_completed += res.rows_completed;
        rows_rejected += res.rows_rejected;
        load_truncated = res.load_truncated;
      });
      total_time += ms;

      // results
      if (load_truncated || rows_rejected > copy_params.max_reject) {
        LOG(ERROR) << "COPY exited early due to reject records count during multi file "
                      "processing ";
        // if we have crossed the truncated load threshold
        load_truncated = true;
        success = false;
      }
      if (!load_truncated) {
        tr = std::string("Loaded: " + std::to_string(rows_completed) +
                         " recs, Rejected: " + std::to_string(rows_rejected) +
                         " recs in " + std::to_string((double)total_time / 1000.0) +
                         " secs");
      } else {
        tr = std::string("Loader truncated due to reject count.  Processed : " +
                         std::to_string(rows_completed) + " recs, Rejected: " +
                         std::to_string(rows_rejected) + " recs in " +
                         std::to_string((double)total_time / 1000.0) + " secs");
      }
    } else {
      throw std::runtime_error("Table '" + *table + "' must exist before COPY FROM");
    }
  }

  return_message.reset(new std::string(tr));
  LOG(INFO) << tr;
}

// CREATE ROLE payroll_dept_role;
void CreateRoleStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  const auto& currentUser = session.get_currentUser();
  if (!currentUser.isSuper) {
    throw std::runtime_error("CREATE ROLE " + get_role() +
                             " failed. It can only be executed by super user.");
  }
  SysCatalog::instance().createRole(get_role());
}

// DROP ROLE payroll_dept_role;
void DropRoleStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  const auto& currentUser = session.get_currentUser();
  if (!currentUser.isSuper) {
    throw std::runtime_error("DROP ROLE " + get_role() +
                             " failed. It can only be executed by super user.");
  }
  auto* rl = SysCatalog::instance().getRoleGrantee(get_role());
  if (!rl) {
    throw std::runtime_error("DROP ROLE " + get_role() +
                             " failed because role with this name does not exist.");
  }
  SysCatalog::instance().dropRole(get_role());
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
    if (objectType.compare("TABLE") == 0 || objectType.compare("DASHBOARD") == 0 ||
        objectType.compare("VIEW") == 0 || objectType.compare("SERVER") == 0) {
      switch (componentNames.size()) {
        case (1): {
          objectName = componentNames[0];
          break;
        }
        case (2): {
          objectName = componentNames[1];
          break;
        }
        default: {
          throw std::runtime_error("DB object name is not correct " + objectHierName);
        }
      }
    } else {
      throw std::runtime_error("DB object type " + objectType + " is not supported.");
    }
  }
  return objectName;
}

static std::pair<AccessPrivileges, DBObjectType> parseStringPrivs(
    const std::string& privs,
    const DBObjectType& objectType,
    const std::string& object_name) {
  static const std::map<std::pair<const std::string, const DBObjectType>,
                        std::pair<const AccessPrivileges, const DBObjectType>>
      privileges_lookup{
          {{"ALL"s, DatabaseDBObjectType},
           {AccessPrivileges::ALL_DATABASE, DatabaseDBObjectType}},
          {{"ALL"s, TableDBObjectType}, {AccessPrivileges::ALL_TABLE, TableDBObjectType}},
          {{"ALL"s, DashboardDBObjectType},
           {AccessPrivileges::ALL_DASHBOARD, DashboardDBObjectType}},
          {{"ALL"s, ViewDBObjectType}, {AccessPrivileges::ALL_VIEW, ViewDBObjectType}},
          {{"ALL"s, ServerDBObjectType},
           {AccessPrivileges::ALL_SERVER, ServerDBObjectType}},

          {{"CREATE TABLE"s, DatabaseDBObjectType},
           {AccessPrivileges::CREATE_TABLE, TableDBObjectType}},
          {{"CREATE"s, DatabaseDBObjectType},
           {AccessPrivileges::CREATE_TABLE, TableDBObjectType}},
          {{"SELECT"s, DatabaseDBObjectType},
           {AccessPrivileges::SELECT_FROM_TABLE, TableDBObjectType}},
          {{"INSERT"s, DatabaseDBObjectType},
           {AccessPrivileges::INSERT_INTO_TABLE, TableDBObjectType}},
          {{"TRUNCATE"s, DatabaseDBObjectType},
           {AccessPrivileges::TRUNCATE_TABLE, TableDBObjectType}},
          {{"UPDATE"s, DatabaseDBObjectType},
           {AccessPrivileges::UPDATE_IN_TABLE, TableDBObjectType}},
          {{"DELETE"s, DatabaseDBObjectType},
           {AccessPrivileges::DELETE_FROM_TABLE, TableDBObjectType}},
          {{"DROP"s, DatabaseDBObjectType},
           {AccessPrivileges::DROP_TABLE, TableDBObjectType}},
          {{"ALTER"s, DatabaseDBObjectType},
           {AccessPrivileges::ALTER_TABLE, TableDBObjectType}},

          {{"SELECT"s, TableDBObjectType},
           {AccessPrivileges::SELECT_FROM_TABLE, TableDBObjectType}},
          {{"INSERT"s, TableDBObjectType},
           {AccessPrivileges::INSERT_INTO_TABLE, TableDBObjectType}},
          {{"TRUNCATE"s, TableDBObjectType},
           {AccessPrivileges::TRUNCATE_TABLE, TableDBObjectType}},
          {{"UPDATE"s, TableDBObjectType},
           {AccessPrivileges::UPDATE_IN_TABLE, TableDBObjectType}},
          {{"DELETE"s, TableDBObjectType},
           {AccessPrivileges::DELETE_FROM_TABLE, TableDBObjectType}},
          {{"DROP"s, TableDBObjectType},
           {AccessPrivileges::DROP_TABLE, TableDBObjectType}},
          {{"ALTER"s, TableDBObjectType},
           {AccessPrivileges::ALTER_TABLE, TableDBObjectType}},

          {{"CREATE VIEW"s, DatabaseDBObjectType},
           {AccessPrivileges::CREATE_VIEW, ViewDBObjectType}},
          {{"SELECT VIEW"s, DatabaseDBObjectType},
           {AccessPrivileges::SELECT_FROM_VIEW, ViewDBObjectType}},
          {{"DROP VIEW"s, DatabaseDBObjectType},
           {AccessPrivileges::DROP_VIEW, ViewDBObjectType}},
          {{"SELECT"s, ViewDBObjectType},
           {AccessPrivileges::SELECT_FROM_VIEW, ViewDBObjectType}},
          {{"DROP"s, ViewDBObjectType}, {AccessPrivileges::DROP_VIEW, ViewDBObjectType}},

          {{"CREATE DASHBOARD"s, DatabaseDBObjectType},
           {AccessPrivileges::CREATE_DASHBOARD, DashboardDBObjectType}},
          {{"EDIT DASHBOARD"s, DatabaseDBObjectType},
           {AccessPrivileges::EDIT_DASHBOARD, DashboardDBObjectType}},
          {{"VIEW DASHBOARD"s, DatabaseDBObjectType},
           {AccessPrivileges::VIEW_DASHBOARD, DashboardDBObjectType}},
          {{"DELETE DASHBOARD"s, DatabaseDBObjectType},
           {AccessPrivileges::DELETE_DASHBOARD, DashboardDBObjectType}},
          {{"VIEW"s, DashboardDBObjectType},
           {AccessPrivileges::VIEW_DASHBOARD, DashboardDBObjectType}},
          {{"EDIT"s, DashboardDBObjectType},
           {AccessPrivileges::EDIT_DASHBOARD, DashboardDBObjectType}},
          {{"DELETE"s, DashboardDBObjectType},
           {AccessPrivileges::DELETE_DASHBOARD, DashboardDBObjectType}},

          {{"CREATE SERVER"s, DatabaseDBObjectType},
           {AccessPrivileges::CREATE_SERVER, ServerDBObjectType}},
          {{"DROP SERVER"s, DatabaseDBObjectType},
           {AccessPrivileges::DROP_SERVER, ServerDBObjectType}},
          {{"DROP"s, ServerDBObjectType},
           {AccessPrivileges::DROP_SERVER, ServerDBObjectType}},
          {{"ALTER SERVER"s, DatabaseDBObjectType},
           {AccessPrivileges::ALTER_SERVER, ServerDBObjectType}},
          {{"ALTER"s, ServerDBObjectType},
           {AccessPrivileges::ALTER_SERVER, ServerDBObjectType}},

          {{"VIEW SQL EDITOR"s, DatabaseDBObjectType},
           {AccessPrivileges::VIEW_SQL_EDITOR, DatabaseDBObjectType}},
          {{"ACCESS"s, DatabaseDBObjectType},
           {AccessPrivileges::ACCESS, DatabaseDBObjectType}}};

  auto result = privileges_lookup.find(std::make_pair(privs, objectType));
  if (result == privileges_lookup.end()) {
    throw std::runtime_error("Privileges " + privs + " on DB object " + object_name +
                             " are not correct.");
  }
  return result->second;
}

static DBObject createObject(const std::string& objectName, DBObjectType objectType) {
  if (objectType == DashboardDBObjectType) {
    int32_t dashboard_id = -1;
    if (!objectName.empty()) {
      try {
        dashboard_id = stoi(objectName);
      } catch (const std::exception&) {
        throw std::runtime_error(
            "Privileges on dashboards should be changed via integer dashboard ID");
      }
    }
    return DBObject(dashboard_id, objectType);
  } else {
    return DBObject(objectName, objectType);
  }
}

// GRANT SELECT/INSERT/CREATE ON TABLE payroll_table TO payroll_dept_role;
void GrantPrivilegesStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();
  const auto& currentUser = session.get_currentUser();
  const auto parserObjectType = boost::to_upper_copy<std::string>(get_object_type());
  const auto objectName =
      extractObjectNameFromHierName(get_object(), parserObjectType, catalog);
  auto objectType = DBObjectTypeFromString(parserObjectType);
  if (objectType == ServerDBObjectType && !g_enable_fsi) {
    throw std::runtime_error("GRANT failed. SERVER object unrecognized.");
  }
  DBObject dbObject = createObject(objectName, objectType);
  /* verify object ownership if not suser */
  if (!currentUser.isSuper) {
    if (!SysCatalog::instance().verifyDBObjectOwnership(currentUser, dbObject, catalog)) {
      throw std::runtime_error(
          "GRANT failed. It can only be executed by super user or owner of the "
          "object.");
    }
  }
  /* set proper values of privileges & grant them to the object */
  std::vector<DBObject> objects(get_privs().size(), dbObject);
  for (size_t i = 0; i < get_privs().size(); ++i) {
    std::pair<AccessPrivileges, DBObjectType> priv = parseStringPrivs(
        boost::to_upper_copy<std::string>(get_privs()[i]), objectType, get_object());
    objects[i].setPrivileges(priv.first);
    objects[i].setPermissionType(priv.second);
    if (priv.second == ServerDBObjectType && !g_enable_fsi) {
      throw std::runtime_error("GRANT failed. SERVER object unrecognized.");
    }
  }
  SysCatalog::instance().grantDBObjectPrivilegesBatch(grantees, objects, catalog);
}

// REVOKE SELECT/INSERT/CREATE ON TABLE payroll_table FROM payroll_dept_role;
void RevokePrivilegesStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();
  const auto& currentUser = session.get_currentUser();
  const auto parserObjectType = boost::to_upper_copy<std::string>(get_object_type());
  const auto objectName =
      extractObjectNameFromHierName(get_object(), parserObjectType, catalog);
  auto objectType = DBObjectTypeFromString(parserObjectType);
  if (objectType == ServerDBObjectType && !g_enable_fsi) {
    throw std::runtime_error("REVOKE failed. SERVER object unrecognized.");
  }
  DBObject dbObject = createObject(objectName, objectType);
  /* verify object ownership if not suser */
  if (!currentUser.isSuper) {
    if (!SysCatalog::instance().verifyDBObjectOwnership(currentUser, dbObject, catalog)) {
      throw std::runtime_error(
          "REVOKE failed. It can only be executed by super user or owner of the "
          "object.");
    }
  }
  /* set proper values of privileges & grant them to the object */
  std::vector<DBObject> objects(get_privs().size(), dbObject);
  for (size_t i = 0; i < get_privs().size(); ++i) {
    std::pair<AccessPrivileges, DBObjectType> priv = parseStringPrivs(
        boost::to_upper_copy<std::string>(get_privs()[i]), objectType, get_object());
    objects[i].setPrivileges(priv.first);
    objects[i].setPermissionType(priv.second);
    if (priv.second == ServerDBObjectType && !g_enable_fsi) {
      throw std::runtime_error("REVOKE failed. SERVER object unrecognized.");
    }
  }
  SysCatalog::instance().revokeDBObjectPrivilegesBatch(grantees, objects, catalog);
}

// NOTE: not used currently, will we ever use it?
// SHOW ON TABLE payroll_table FOR payroll_dept_role;
void ShowPrivilegesStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();
  const auto& currentUser = session.get_currentUser();
  const auto parserObjectType = boost::to_upper_copy<std::string>(get_object_type());
  const auto objectName =
      extractObjectNameFromHierName(get_object(), parserObjectType, catalog);
  auto objectType = DBObjectTypeFromString(parserObjectType);
  DBObject dbObject = createObject(objectName, objectType);
  /* verify object ownership if not suser */
  if (!currentUser.isSuper) {
    if (!SysCatalog::instance().verifyDBObjectOwnership(currentUser, dbObject, catalog)) {
      throw std::runtime_error(
          "SHOW ON " + get_object() + " FOR " + get_role() +
          " failed. It can only be executed by super user or owner of the object.");
    }
  }
  /* get values of privileges for the object and report them */
  SysCatalog::instance().getDBObjectPrivileges(get_role(), dbObject, catalog);
  AccessPrivileges privs = dbObject.getPrivileges();
  printf("\nPRIVILEGES ON %s FOR %s ARE SET AS FOLLOWING: ",
         get_object().c_str(),
         get_role().c_str());

  if (objectType == DBObjectType::DatabaseDBObjectType) {
    if (privs.hasPermission(DatabasePrivileges::CREATE_DATABASE)) {
      printf(" CREATE");
    }
    if (privs.hasPermission(DatabasePrivileges::DROP_DATABASE)) {
      printf(" DROP");
    }
  } else if (objectType == DBObjectType::TableDBObjectType) {
    if (privs.hasPermission(TablePrivileges::CREATE_TABLE)) {
      printf(" CREATE");
    }
    if (privs.hasPermission(TablePrivileges::DROP_TABLE)) {
      printf(" DROP");
    }
    if (privs.hasPermission(TablePrivileges::SELECT_FROM_TABLE)) {
      printf(" SELECT");
    }
    if (privs.hasPermission(TablePrivileges::INSERT_INTO_TABLE)) {
      printf(" INSERT");
    }
    if (privs.hasPermission(TablePrivileges::UPDATE_IN_TABLE)) {
      printf(" UPDATE");
    }
    if (privs.hasPermission(TablePrivileges::DELETE_FROM_TABLE)) {
      printf(" DELETE");
    }
    if (privs.hasPermission(TablePrivileges::TRUNCATE_TABLE)) {
      printf(" TRUNCATE");
    }
    if (privs.hasPermission(TablePrivileges::ALTER_TABLE)) {
      printf(" ALTER");
    }
  } else if (objectType == DBObjectType::DashboardDBObjectType) {
    if (privs.hasPermission(DashboardPrivileges::CREATE_DASHBOARD)) {
      printf(" CREATE");
    }
    if (privs.hasPermission(DashboardPrivileges::DELETE_DASHBOARD)) {
      printf(" DELETE");
    }
    if (privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD)) {
      printf(" VIEW");
    }
    if (privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD)) {
      printf(" EDIT");
    }
  } else if (objectType == DBObjectType::ViewDBObjectType) {
    if (privs.hasPermission(ViewPrivileges::CREATE_VIEW)) {
      printf(" CREATE");
    }
    if (privs.hasPermission(ViewPrivileges::DROP_VIEW)) {
      printf(" DROP");
    }
    if (privs.hasPermission(ViewPrivileges::SELECT_FROM_VIEW)) {
      printf(" SELECT");
    }
    if (privs.hasPermission(ViewPrivileges::INSERT_INTO_VIEW)) {
      printf(" INSERT");
    }
    if (privs.hasPermission(ViewPrivileges::UPDATE_IN_VIEW)) {
      printf(" UPDATE");
    }
    if (privs.hasPermission(ViewPrivileges::DELETE_FROM_VIEW)) {
      printf(" DELETE");
    }
  }
  printf(".\n");
}

// GRANT payroll_dept_role TO joe;
void GrantRoleStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  const auto& currentUser = session.get_currentUser();
  if (!currentUser.isSuper) {
    throw std::runtime_error(
        "GRANT failed, because it can only be executed by super user.");
  }
  if (std::find(get_grantees().begin(), get_grantees().end(), OMNISCI_ROOT_USER) !=
      get_grantees().end()) {
    throw std::runtime_error(
        "Request to grant role failed because mapd root user has all privileges by "
        "default.");
  }
  SysCatalog::instance().grantRoleBatch(get_roles(), get_grantees());
}

// REVOKE payroll_dept_role FROM joe;get_users
void RevokeRoleStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  const auto& currentUser = session.get_currentUser();
  if (!currentUser.isSuper) {
    throw std::runtime_error(
        "REVOKE failed, because it can only be executed by super user.");
  }
  if (std::find(get_grantees().begin(), get_grantees().end(), OMNISCI_ROOT_USER) !=
      get_grantees().end()) {
    throw std::runtime_error(
        "Request to revoke role failed because privileges can not be revoked from mapd "
        "root user.");
  }
  SysCatalog::instance().revokeRoleBatch(get_roles(), get_grantees());
}

void ShowCreateTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  using namespace Catalog_Namespace;

  const auto execute_read_lock = mapd_shared_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  auto& catalog = session.getCatalog();
  const TableDescriptor* td = catalog.getMetadataForTable(*table_, false);
  if (!td) {
    throw std::runtime_error("Table/View " + *table_ + " does not exist.");
  }

  DBObject dbObject(td->tableName, td->isView ? ViewDBObjectType : TableDBObjectType);
  dbObject.loadKey(catalog);
  std::vector<DBObject> privObjects = {dbObject};

  if (!SysCatalog::instance().hasAnyPrivileges(session.get_currentUser(), privObjects)) {
    throw std::runtime_error("Table/View " + *table_ + " does not exist.");
  }
  if (td->isView && !session.get_currentUser().isSuper) {
    // TODO: we need to run a validate query to ensure the user has access to the
    // underlying table, but we do not have any of the machinery in here. Disable for
    // now, unless the current user is a super user.
    throw std::runtime_error("SHOW CREATE TABLE not yet supported for views");
  }

  create_stmt_ = catalog.dumpCreateTable(td);
}

void ExportQueryStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto session_copy = session;
  auto session_ptr = std::shared_ptr<Catalog_Namespace::SessionInfo>(
      &session_copy, boost::null_deleter());
  auto query_state = query_state::QueryState::create(session_ptr, *select_stmt);
  auto stdlog = STDLOG(query_state);
  auto query_state_proxy = query_state->createQueryStateProxy();

  auto& catalog = session.getCatalog();

  LocalConnector local_connector;

  if (!leafs_connector_) {
    leafs_connector_ = &local_connector;
  }

  import_export::CopyParams copy_params;
  // @TODO(se) move rest to CopyParams when we have a Thrift endpoint
  import_export::QueryExporter::FileType file_type =
      import_export::QueryExporter::FileType::kCSV;
  std::string layer_name;
  import_export::QueryExporter::FileCompression file_compression =
      import_export::QueryExporter::FileCompression::kNone;
  import_export::QueryExporter::ArrayNullHandling array_null_handling =
      import_export::QueryExporter::ArrayNullHandling::kAbortWithWarning;

  parseOptions(copy_params, file_type, layer_name, file_compression, array_null_handling);

  if (file_path->empty()) {
    throw std::runtime_error("Invalid file path for COPY TO");
  } else if (!boost::filesystem::path(*file_path).is_absolute()) {
    std::string file_name = boost::filesystem::path(*file_path).filename().string();
    std::string file_dir =
        catalog.getBasePath() + "/mapd_export/" + session.get_session_id() + "/";
    if (!boost::filesystem::exists(file_dir)) {
      if (!boost::filesystem::create_directories(file_dir)) {
        throw std::runtime_error("Directory " + file_dir + " cannot be created.");
      }
    }
    *file_path = file_dir + file_name;
  } else {
    // Above branch will create a new file in the mapd_export directory. If that path is
    // not exercised, go through applicable file path validations.
    ddl_utils::validate_allowed_file_path(*file_path,
                                          ddl_utils::DataTransferType::EXPORT);
  }

  // get column info
  auto column_info_result =
      local_connector.query(query_state_proxy, *select_stmt, {}, true);

  // create exporter for requested file type
  auto query_exporter = import_export::QueryExporter::create(file_type);

  // default layer name to file path stem if it wasn't specified
  if (layer_name.size() == 0) {
    layer_name = boost::filesystem::path(*file_path).stem().string();
  }

  // begin export
  query_exporter->beginExport(*file_path,
                              layer_name,
                              copy_params,
                              column_info_result.targets_meta,
                              file_compression,
                              array_null_handling);

  // how many fragments?
  size_t outer_frag_count =
      leafs_connector_->getOuterFragmentCount(query_state_proxy, *select_stmt);
  size_t outer_frag_end = outer_frag_count == 0 ? 1 : outer_frag_count;

  // loop fragments
  for (size_t outer_frag_idx = 0; outer_frag_idx < outer_frag_end; outer_frag_idx++) {
    // limit the query to just this fragment
    std::vector<size_t> allowed_outer_fragment_indices;
    if (outer_frag_count) {
      allowed_outer_fragment_indices.push_back(outer_frag_idx);
    }

    // run the query
    std::vector<AggregatedResult> query_results = leafs_connector_->query(
        query_state_proxy, *select_stmt, allowed_outer_fragment_indices);

    // export the results
    query_exporter->exportResults(query_results);
  }

  // end export
  query_exporter->endExport();
}

void ExportQueryStmt::parseOptions(
    import_export::CopyParams& copy_params,
    import_export::QueryExporter::FileType& file_type,
    std::string& layer_name,
    import_export::QueryExporter::FileCompression& file_compression,
    import_export::QueryExporter::ArrayNullHandling& array_null_handling) {
  // defaults for non-CopyParams values
  file_type = import_export::QueryExporter::FileType::kCSV;
  layer_name.clear();
  file_compression = import_export::QueryExporter::FileCompression::kNone;

  if (!options.empty()) {
    for (auto& p : options) {
      if (boost::iequals(*p->get_name(), "delimiter")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Delimiter option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Delimiter must be a single character string.");
        }
        copy_params.delimiter = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "nulls")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Nulls option must be a string.");
        }
        copy_params.null_str = *str_literal->get_stringval();
      } else if (boost::iequals(*p->get_name(), "header")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Header option must be a boolean.");
        }
        copy_params.has_header = bool_from_string_literal(str_literal)
                                     ? import_export::ImportHeaderRow::HAS_HEADER
                                     : import_export::ImportHeaderRow::NO_HEADER;
      } else if (boost::iequals(*p->get_name(), "quote")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Quote option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Quote must be a single character string.");
        }
        copy_params.quote = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "escape")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Escape option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Escape must be a single character string.");
        }
        copy_params.escape = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "line_delimiter")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Line_delimiter option must be a string.");
        } else if (str_literal->get_stringval()->length() != 1) {
          throw std::runtime_error("Line_delimiter must be a single character string.");
        }
        copy_params.line_delim = (*str_literal->get_stringval())[0];
      } else if (boost::iequals(*p->get_name(), "quoted")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Quoted option must be a boolean.");
        }
        copy_params.quoted = bool_from_string_literal(str_literal);
      } else if (boost::iequals(*p->get_name(), "file_type")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("File Type option must be a string.");
        }
        auto file_type_str =
            boost::algorithm::to_lower_copy(*str_literal->get_stringval());
        if (file_type_str == "csv") {
          file_type = import_export::QueryExporter::FileType::kCSV;
        } else if (file_type_str == "geojson") {
          file_type = import_export::QueryExporter::FileType::kGeoJSON;
        } else if (file_type_str == "geojsonl") {
          file_type = import_export::QueryExporter::FileType::kGeoJSONL;
        } else if (file_type_str == "shapefile") {
          file_type = import_export::QueryExporter::FileType::kShapefile;
        } else {
          throw std::runtime_error(
              "File Type option must be 'CSV', 'GeoJSON', 'GeoJSONL' or 'Shapefile'");
        }
      } else if (boost::iequals(*p->get_name(), "layer_name")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Layer Name option must be a string.");
        }
        layer_name = *str_literal->get_stringval();
      } else if (boost::iequals(*p->get_name(), "file_compression")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("File Compression option must be a string.");
        }
        auto file_compression_str =
            boost::algorithm::to_lower_copy(*str_literal->get_stringval());
        if (file_compression_str == "none") {
          file_compression = import_export::QueryExporter::FileCompression::kNone;
        } else if (file_compression_str == "gzip") {
          file_compression = import_export::QueryExporter::FileCompression::kGZip;
        } else if (file_compression_str == "zip") {
          file_compression = import_export::QueryExporter::FileCompression::kZip;
        } else {
          throw std::runtime_error(
              "File Compression option must be 'None', 'GZip', or 'Zip'");
        }
      } else if (boost::iequals(*p->get_name(), "array_null_handling")) {
        const StringLiteral* str_literal =
            dynamic_cast<const StringLiteral*>(p->get_value());
        if (str_literal == nullptr) {
          throw std::runtime_error("Array Null Handling option must be a string.");
        }
        auto array_null_handling_str =
            boost::algorithm::to_lower_copy(*str_literal->get_stringval());
        if (array_null_handling_str == "abort") {
          array_null_handling =
              import_export::QueryExporter::ArrayNullHandling::kAbortWithWarning;
        } else if (array_null_handling_str == "raw") {
          array_null_handling =
              import_export::QueryExporter::ArrayNullHandling::kExportSentinels;
        } else if (array_null_handling_str == "zero") {
          array_null_handling =
              import_export::QueryExporter::ArrayNullHandling::kExportZeros;
        } else if (array_null_handling_str == "nullfield") {
          array_null_handling =
              import_export::QueryExporter::ArrayNullHandling::kNullEntireField;
        } else {
          throw std::runtime_error(
              "Array Null Handling option must be 'Abort', 'Raw', 'Zero', or "
              "'NullField'");
        }
      } else {
        throw std::runtime_error("Invalid option for COPY: " + *p->get_name());
      }
    }
  }
}

CreateViewStmt::CreateViewStmt(const rapidjson::Value& payload) {
  CHECK(payload.HasMember("name"));
  view_name_ = json_str(payload["name"]);

  if_not_exists_ = false;
  if (payload.HasMember("ifNotExists")) {
    if_not_exists_ = json_bool(payload["ifNotExists"]);
  }

  CHECK(payload.HasMember("query"));
  select_query_ = json_str(payload["query"]);
  std::regex newline_re("\\n");
  select_query_ = std::regex_replace(select_query_, newline_re, " ");
  // ensure a trailing semicolon is present on the select query
  if (select_query_.back() != ';') {
    select_query_.push_back(';');
  }
}

void CreateViewStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto session_copy = session;
  auto session_ptr = std::shared_ptr<Catalog_Namespace::SessionInfo>(
      &session_copy, boost::null_deleter());
  auto query_state = query_state::QueryState::create(session_ptr, select_query_);
  auto stdlog = STDLOG(query_state);
  auto& catalog = session.getCatalog();

  if (!catalog.validateNonExistentTableOrView(view_name_, if_not_exists_)) {
    return;
  }
  if (!session.checkDBAccessPrivileges(DBObjectType::ViewDBObjectType,
                                       AccessPrivileges::CREATE_VIEW)) {
    throw std::runtime_error("View " + view_name_ +
                             " will not be created. User has no create view privileges.");
  }

  const auto query_after_shim = pg_shim(select_query_);

  // this now also ensures that access permissions are checked
  catalog.getCalciteMgr()->process(query_state->createQueryStateProxy(),
                                   query_after_shim,
                                   {},
                                   true,
                                   false,
                                   false,
                                   true);

  // Take write lock after the query is processed to ensure no deadlocks
  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  TableDescriptor td;
  td.tableName = view_name_;
  td.userId = session.get_currentUser().userId;
  td.nColumns = 0;
  td.isView = true;
  td.viewSQL = query_after_shim;
  td.fragmenter = nullptr;
  td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
  td.maxFragRows =
      DEFAULT_FRAGMENT_ROWS;  // @todo this stuff should not be InsertOrderFragmenter
  td.maxChunkSize =
      DEFAULT_MAX_CHUNK_SIZE;  // @todo this stuff should not be InsertOrderFragmenter
  td.fragPageSize = DEFAULT_PAGE_SIZE;
  td.maxRows = DEFAULT_MAX_ROWS;
  catalog.createTable(td, {}, {}, true);

  // TODO (max): It's transactionally unsafe, should be fixed: we may create object w/o
  // privileges
  SysCatalog::instance().createDBObject(
      session.get_currentUser(), view_name_, ViewDBObjectType, catalog);
}

DropViewStmt::DropViewStmt(const rapidjson::Value& payload) {
  CHECK(payload.HasMember("viewName"));
  view_name = std::make_unique<std::string>(json_str(payload["viewName"]));

  if_exists = false;
  if (payload.HasMember("ifExists")) {
    if_exists = json_bool(payload["ifExists"]);
  }
}

void DropViewStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();

  const TableDescriptor* td{nullptr};
  std::unique_ptr<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>> td_with_lock;

  try {
    td_with_lock =
        std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>>(
            lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
                catalog, *view_name, false));
    td = (*td_with_lock)();
  } catch (const std::runtime_error& e) {
    if (if_exists) {
      return;
    } else {
      throw e;
    }
  }

  CHECK(td);
  CHECK(td_with_lock);

  if (!session.checkDBAccessPrivileges(
          DBObjectType::ViewDBObjectType, AccessPrivileges::DROP_VIEW, *view_name)) {
    throw std::runtime_error("View " + *view_name +
                             " will not be dropped. User has no drop view privileges.");
  }

  ddl_utils::validate_table_type(td, ddl_utils::TableType::VIEW, "DROP");
  catalog.dropTable(td);
}

static void checkStringLiteral(const std::string& option_name,
                               const std::unique_ptr<NameValueAssign>& p) {
  CHECK(p);
  if (!dynamic_cast<const StringLiteral*>(p->get_value())) {
    throw std::runtime_error(option_name + " option must be a string literal.");
  }
}

void CreateDBStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  if (!session.get_currentUser().isSuper) {
    throw std::runtime_error(
        "CREATE DATABASE command can only be executed by super user.");
  }

  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  Catalog_Namespace::DBMetadata db_meta;
  if (SysCatalog::instance().getMetadataForDB(*db_name, db_meta) && if_not_exists_) {
    return;
  }
  int ownerId = session.get_currentUser().userId;
  if (!name_value_list.empty()) {
    for (auto& p : name_value_list) {
      if (boost::iequals(*p->get_name(), "owner")) {
        checkStringLiteral("Owner name", p);
        const std::string* str =
            static_cast<const StringLiteral*>(p->get_value())->get_stringval();
        Catalog_Namespace::UserMetadata user;
        if (!SysCatalog::instance().getMetadataForUser(*str, user)) {
          throw std::runtime_error("User " + *str + " does not exist.");
        }
        ownerId = user.userId;
      } else {
        throw std::runtime_error("Invalid CREATE DATABASE option " + *p->get_name() +
                                 ". Only OWNER supported.");
      }
    }
  }
  SysCatalog::instance().createDatabase(*db_name, ownerId);
}

void DropDBStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  if (!session.get_currentUser().isSuper) {
    throw std::runtime_error("DROP DATABASE command can only be executed by super user.");
  }

  const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));

  Catalog_Namespace::DBMetadata db;
  if (!SysCatalog::instance().getMetadataForDB(*db_name, db)) {
    if (if_exists_) {
      return;
    }
    throw std::runtime_error("Database " + *db_name + " does not exist.");
  }

  if (!session.get_currentUser().isSuper &&
      session.get_currentUser().userId != db.dbOwner) {
    throw std::runtime_error("Only the super user or the owner can drop database.");
  }

  SysCatalog::instance().dropDatabase(db);
}

static bool readBooleanLiteral(const std::string& option_name,
                               const std::unique_ptr<NameValueAssign>& p) {
  CHECK(p);
  const std::string* str =
      static_cast<const StringLiteral*>(p->get_value())->get_stringval();
  if (boost::iequals(*str, "true")) {
    return true;
  } else if (boost::iequals(*str, "false")) {
    return false;
  } else {
    throw std::runtime_error("Value to " + option_name + " must be TRUE or FALSE.");
  }
}

void CreateUserStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  std::string passwd;
  bool is_super = false;
  std::string default_db;
  bool can_login = true;
  for (auto& p : name_value_list) {
    if (boost::iequals(*p->get_name(), "password")) {
      checkStringLiteral("Password", p);
      passwd = *static_cast<const StringLiteral*>(p->get_value())->get_stringval();
    } else if (boost::iequals(*p->get_name(), "is_super")) {
      checkStringLiteral("IS_SUPER", p);
      is_super = readBooleanLiteral("IS_SUPER", p);
    } else if (boost::iequals(*p->get_name(), "default_db")) {
      checkStringLiteral("DEFAULT_DB", p);
      default_db = *static_cast<const StringLiteral*>(p->get_value())->get_stringval();
    } else if (boost::iequals(*p->get_name(), "can_login")) {
      checkStringLiteral("CAN_LOGIN", p);
      can_login = readBooleanLiteral("can_login", p);
    } else {
      throw std::runtime_error("Invalid CREATE USER option " + *p->get_name() +
                               ". Should be PASSWORD, IS_SUPER, CAN_LOGIN"
                               " or DEFAULT_DB.");
    }
  }
  if (!session.get_currentUser().isSuper) {
    throw std::runtime_error("Only super user can create new users.");
  }
  SysCatalog::instance().createUser(*user_name, passwd, is_super, default_db, can_login);
}

void AlterUserStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  // Parse the statement
  const std::string* passwd = nullptr;
  bool is_super = false;
  bool* is_superp = nullptr;
  const std::string* default_db = nullptr;
  bool can_login = true;
  bool* can_loginp = nullptr;
  for (auto& p : name_value_list) {
    if (boost::iequals(*p->get_name(), "password")) {
      checkStringLiteral("Password", p);
      passwd = static_cast<const StringLiteral*>(p->get_value())->get_stringval();
    } else if (boost::iequals(*p->get_name(), "is_super")) {
      checkStringLiteral("IS_SUPER", p);
      is_super = readBooleanLiteral("IS_SUPER", p);
      is_superp = &is_super;
    } else if (boost::iequals(*p->get_name(), "default_db")) {
      if (dynamic_cast<const StringLiteral*>(p->get_value())) {
        default_db = static_cast<const StringLiteral*>(p->get_value())->get_stringval();
      } else if (dynamic_cast<const NullLiteral*>(p->get_value())) {
        static std::string blank;
        default_db = &blank;
      } else {
        throw std::runtime_error(
            "DEFAULT_DB option must be either a string literal or a NULL literal.");
      }
    } else if (boost::iequals(*p->get_name(), "can_login")) {
      checkStringLiteral("CAN_LOGIN", p);
      can_login = readBooleanLiteral("CAN_LOGIN", p);
      can_loginp = &can_login;
    } else {
      throw std::runtime_error("Invalid ALTER USER option " + *p->get_name() +
                               ". Should be PASSWORD, DEFAULT_DB, CAN_LOGIN"
                               " or IS_SUPER.");
    }
  }

  // Check if the user is authorized to execute ALTER USER statement
  Catalog_Namespace::UserMetadata user;
  if (!SysCatalog::instance().getMetadataForUser(*user_name, user)) {
    throw std::runtime_error("User " + *user_name + " does not exist.");
  }
  if (!session.get_currentUser().isSuper) {
    if (session.get_currentUser().userId != user.userId) {
      throw std::runtime_error("Only super user can change another user's attributes.");
    } else if (is_superp || can_loginp) {
      throw std::runtime_error(
          "A user can only update their own password or default database.");
    }
  }

  if (passwd || is_superp || default_db || can_loginp) {
    SysCatalog::instance().alterUser(
        user.userId, passwd, is_superp, default_db, can_loginp);
  }
}

void DropUserStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  if (!session.get_currentUser().isSuper) {
    throw std::runtime_error("Only super user can drop users.");
  }
  SysCatalog::instance().dropUser(*user_name);
}

void DumpTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  // check access privileges
  if (!session.checkDBAccessPrivileges(
          DBObjectType::TableDBObjectType, AccessPrivileges::SELECT_FROM_TABLE, *table)) {
    throw std::runtime_error("Table " + *table +
                             " will not be dumped. User has no select privileges.");
  }
  if (!session.checkDBAccessPrivileges(DBObjectType::TableDBObjectType,
                                       AccessPrivileges::CREATE_TABLE)) {
    throw std::runtime_error("Table " + *table +
                             " will not be dumped. User has no create privileges.");
  }
  auto& catalog = session.getCatalog();
  const TableDescriptor* td = catalog.getMetadataForTable(*table);
  TableArchiver table_archiver(&catalog);
  table_archiver.dumpTable(td, *path, compression);
}

void RestoreTableStmt::execute(const Catalog_Namespace::SessionInfo& session) {
  auto& catalog = session.getCatalog();
  const TableDescriptor* td = catalog.getMetadataForTable(*table, false);
  if (td) {
    // TODO: v1.0 simply throws to avoid accidentally overwrite target table.
    // Will add a REPLACE TABLE to explictly replace target table.
    // catalog.restoreTable(session, td, *path, compression);
    throw std::runtime_error("Table " + *table + " exists.");
  } else {
    // check access privileges
    if (!session.checkDBAccessPrivileges(DBObjectType::TableDBObjectType,
                                         AccessPrivileges::CREATE_TABLE)) {
      throw std::runtime_error("Table " + *table +
                               " will not be restored. User has no create privileges.");
    }
    TableArchiver table_archiver(&catalog);
    table_archiver.restoreTable(session, *table, *path, compression);
  }
}

void execute_calcite_ddl(
    const std::string& ddl_statement,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  CHECK(!ddl_statement.empty());
  VLOG(2) << "Parsing JSON DDL from Calcite: " << ddl_statement;
  rapidjson::Document ddl_query;
  ddl_query.Parse(ddl_statement);
  CHECK(ddl_query.IsObject());
  CHECK(ddl_query.HasMember("payload"));
  CHECK(ddl_query["payload"].IsObject());
  const auto& payload = ddl_query["payload"].GetObject();
  CHECK(payload.HasMember("command"));
  CHECK(payload["command"].IsString());

  const auto& ddl_command = std::string_view(payload["command"].GetString());
  if (ddl_command == "CREATE_TABLE") {
    auto create_table_stmt = Parser::CreateTableStmt(payload);
    create_table_stmt.execute(*session_ptr);
  } else if (ddl_command == "DROP_TABLE") {
    auto drop_table_stmt = Parser::DropTableStmt(payload);
    drop_table_stmt.execute(*session_ptr);
  } else if (ddl_command == "CREATE_VIEW") {
    auto create_view_stmt = Parser::CreateViewStmt(payload);
    create_view_stmt.execute(*session_ptr);
  } else if (ddl_command == "DROP_VIEW") {
    auto drop_view_stmt = Parser::DropViewStmt(payload);
    drop_view_stmt.execute(*session_ptr);
  } else {
    throw std::runtime_error("Unsupported DDL command");
  }
}

}  // namespace Parser
