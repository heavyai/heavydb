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
 * @file    TargetInfo.h
 * @author  Alex Suhan <alex@mapd.com>
 */

#ifndef QUERYENGINE_TARGETINFO_H
#define QUERYENGINE_TARGETINFO_H

#include "sqldefs.h"
#include "sqltypes.h"

#include "../Analyzer/Analyzer.h"

inline const Analyzer::AggExpr* cast_to_agg_expr(const Analyzer::Expr* target_expr) {
  return dynamic_cast<const Analyzer::AggExpr*>(target_expr);
}

inline const Analyzer::AggExpr* cast_to_agg_expr(
    const std::shared_ptr<Analyzer::Expr> target_expr) {
  return dynamic_cast<const Analyzer::AggExpr*>(target_expr.get());
}

inline bool target_expr_has_varlen_projection(const Analyzer::Expr* target_expr) {
  return !(dynamic_cast<const Analyzer::GeoExpr*>(target_expr) == nullptr);
}

inline bool target_expr_has_varlen_projection(
    const std::shared_ptr<Analyzer::Expr> target_expr) {
  return !(dynamic_cast<const Analyzer::GeoExpr*>(target_expr.get()) == nullptr);
}

struct TargetInfo {
  bool is_agg;
  SQLAgg agg_kind;
  SQLTypeInfo sql_type;
  SQLTypeInfo agg_arg_type;
  bool skip_null_val;
  bool is_distinct;
  bool is_varlen_projection;
#ifndef __CUDACC__
 public:
  inline std::string toString() const {
    auto result = std::string("TargetInfo(");
    result += "is_agg=" + std::string(is_agg ? "true" : "false") + ", ";
    result += "agg_kind=" + ::toString(agg_kind) + ", ";
    result += "sql_type=" + sql_type.to_string() + ", ";
    result += "agg_arg_type=" + agg_arg_type.to_string() + ", ";
    result += "skip_null_val=" + std::string(skip_null_val ? "true" : "false") + ", ";
    result += "is_distinct=" + std::string(is_distinct ? "true" : "false") + ")";
    result +=
        "is_varlen_projection=" + std::string(is_varlen_projection ? "true" : "false") +
        ")";
    return result;
  }
#endif
};

/**
 * Returns true if the aggregate function always returns a value in the domain of the
 * argument. Returns false otherwise.
 */
inline bool is_agg_domain_range_equivalent(const SQLAgg& agg_kind) {
  switch (agg_kind) {
    case kMIN:
    case kMAX:
    case kSINGLE_VALUE:
    case kSAMPLE:
      return true;
    default:
      break;
  }
  return false;
}

template <class PointerType>
inline TargetInfo get_target_info(const PointerType target_expr,
                                  const bool bigint_count) {
  const auto agg_expr = cast_to_agg_expr(target_expr);
  bool notnull = target_expr->get_type_info().get_notnull();
  if (!agg_expr) {
    const bool is_varlen_projection = target_expr_has_varlen_projection(target_expr);
    auto target_ti = target_expr ? get_logical_type_info(target_expr->get_type_info())
                                 : SQLTypeInfo(kBIGINT, notnull);
    return {false,
            kMIN,
            target_ti,
            SQLTypeInfo(kNULLT, false),
            false,
            false,
            is_varlen_projection};
  }
  const auto agg_type = agg_expr->get_aggtype();
  const auto agg_arg = agg_expr->get_arg();
  if (!agg_arg) {
    CHECK_EQ(kCOUNT, agg_type);
    CHECK(!agg_expr->get_is_distinct());
    return {true,
            kCOUNT,
            SQLTypeInfo(bigint_count ? kBIGINT : kINT, notnull),
            SQLTypeInfo(kNULLT, false),
            false,
            false,
            false};
  }

  const auto& agg_arg_ti = agg_arg->get_type_info();
  bool is_distinct{false};
  if (agg_expr->get_aggtype() == kCOUNT) {
    is_distinct = agg_expr->get_is_distinct();
  }

  if (agg_type == kAVG) {
    // Upcast the target type for AVG, so that the integer argument does not overflow the
    // sum
    return {true,
            agg_expr->get_aggtype(),
            agg_arg_ti.is_integer() ? SQLTypeInfo(kBIGINT, agg_arg_ti.get_notnull())
                                    : agg_arg_ti,
            agg_arg_ti,
            !agg_arg_ti.get_notnull(),
            is_distinct,
            false};
  }

  return {
      true,
      agg_expr->get_aggtype(),
      agg_type == kCOUNT
          ? SQLTypeInfo((is_distinct || bigint_count) ? kBIGINT : kINT, notnull)
          : agg_expr->get_type_info(),
      agg_arg_ti,
      agg_type == kCOUNT && agg_arg_ti.is_varlen() ? false : !agg_arg_ti.get_notnull(),
      is_distinct,
      false};
}

inline bool is_distinct_target(const TargetInfo& target_info) {
  return target_info.is_distinct || target_info.agg_kind == kAPPROX_COUNT_DISTINCT;
}

inline bool takes_float_argument(const TargetInfo& target_info) {
  return target_info.is_agg &&
         (target_info.agg_kind == kAVG || target_info.agg_kind == kSUM ||
          target_info.agg_kind == kMIN || target_info.agg_kind == kMAX ||
          target_info.agg_kind == kSINGLE_VALUE) &&
         target_info.agg_arg_type.get_type() == kFLOAT;
}

#endif  // QUERYENGINE_TARGETINFO_H
