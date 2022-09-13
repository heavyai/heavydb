/*
 * Copyright 2022 Heavy.AI, Inc.
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

#include "TargetInfo.h"

namespace target_info {
TargetInfo get_target_info_impl(const Analyzer::Expr* target_expr,
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
      shared::is_any<kCOUNT, kCOUNT_IF>(agg_type)
          ? SQLTypeInfo((is_distinct || bigint_count) ? kBIGINT : kINT, notnull)
          : agg_expr->get_type_info(),
      agg_arg_ti,
      agg_type == kCOUNT && agg_arg_ti.is_varlen() ? false : !agg_arg_ti.get_notnull(),
      is_distinct,
      false};
}
}  // namespace target_info
