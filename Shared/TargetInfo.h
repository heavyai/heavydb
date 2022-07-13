/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @brief
 *
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

namespace target_info {
TargetInfo get_target_info_impl(const Analyzer::Expr* target_expr,
                                const bool bigint_count);
}

inline TargetInfo get_target_info(const Analyzer::Expr* target_expr,
                                  const bool bigint_count) {
  return target_info::get_target_info_impl(target_expr, bigint_count);
}

inline TargetInfo get_target_info(const std::shared_ptr<Analyzer::Expr> target_expr,
                                  const bool bigint_count) {
  return target_info::get_target_info_impl(target_expr.get(), bigint_count);
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
