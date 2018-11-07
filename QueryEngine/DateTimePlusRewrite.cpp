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

#include "DateTimePlusRewrite.h"
#include "Execute.h"

#include "../Analyzer/Analyzer.h"
#include "../Parser/ParserNode.h"

#include <glog/logging.h>

namespace {

const Analyzer::Expr* remove_truncate_int(const Analyzer::Expr* expr) {
  if (!expr) {
    return nullptr;
  }
  const auto func_oper = dynamic_cast<const Analyzer::FunctionOper*>(expr);
  if (!func_oper || func_oper->getName() != "TRUNCATE") {
    return nullptr;
  }
  CHECK_EQ(size_t(2), func_oper->getArity());
  const auto arg = func_oper->getArg(0);
  const auto& arg_ti = arg->get_type_info();
  return arg_ti.is_integer() ? arg : nullptr;
}

bool match_const_integer(const Analyzer::Expr* expr, const int64_t v) {
  const auto const_expr = dynamic_cast<const Analyzer::Constant*>(expr);
  if (!const_expr) {
    return false;
  }
  const auto& const_ti = const_expr->get_type_info();
  if (!const_ti.is_integer()) {
    return false;
  }
  const auto& datum = const_expr->get_constval();
  switch (const_ti.get_type()) {
    case kTINYINT:
      return v == datum.tinyintval;
    case kSMALLINT:
      return v == datum.smallintval;
    case kINT:
      return v == datum.intval;
    case kBIGINT:
      return v == datum.bigintval;
    default:
      break;
  }
  return false;
}

DatetruncField get_dt_field(const Analyzer::Expr* ts,
                            const Analyzer::Expr* interval_multiplier,
                            const bool dt_hour) {
  if (dt_hour) {
    const auto extract_fn =
        dynamic_cast<const Analyzer::ExtractExpr*>(interval_multiplier);
    return (extract_fn && extract_fn->get_field() == kHOUR &&
            *extract_fn->get_from_expr() == *ts)
               ? dtHOUR
               : dtINVALID;
  }
  const auto interval_multiplier_fn =
      remove_truncate_int(remove_cast_to_int(interval_multiplier));
  if (!interval_multiplier_fn) {
    return dtINVALID;
  }
  const auto interval_multiplier_mul =
      dynamic_cast<const Analyzer::BinOper*>(interval_multiplier_fn);
  if (!interval_multiplier_mul || interval_multiplier_mul->get_optype() != kMULTIPLY ||
      !match_const_integer(interval_multiplier_mul->get_left_operand(), -1)) {
    return dtINVALID;
  }
  const auto extract_minus_one = dynamic_cast<const Analyzer::BinOper*>(
      interval_multiplier_mul->get_right_operand());
  if (!extract_minus_one || extract_minus_one->get_optype() != kMINUS ||
      !match_const_integer(extract_minus_one->get_right_operand(), 1)) {
    return dtINVALID;
  }
  const auto extract_fn =
      dynamic_cast<const Analyzer::ExtractExpr*>(extract_minus_one->get_left_operand());
  if (!extract_fn || !(*extract_fn->get_from_expr() == *ts)) {
    return dtINVALID;
  }
  switch (extract_fn->get_field()) {
    case kDAY:
      return dtMONTH;
    case kDOY:
      return dtYEAR;
    default:
      break;
  }
  return dtINVALID;
}

DatetruncField get_dt_field(const Analyzer::Expr* ts, const Analyzer::Expr* off_arg) {
  const auto mul_by_interval = dynamic_cast<const Analyzer::BinOper*>(off_arg);
  if (!mul_by_interval) {
    return dtINVALID;
  }
  auto interval =
      dynamic_cast<const Analyzer::Constant*>(mul_by_interval->get_right_operand());
  auto interval_multiplier = mul_by_interval->get_left_operand();
  if (!interval) {
    interval =
        dynamic_cast<const Analyzer::Constant*>(mul_by_interval->get_left_operand());
    if (!interval) {
      return dtINVALID;
    }
    interval_multiplier = mul_by_interval->get_right_operand();
  }
  const auto& interval_ti = interval->get_type_info();
  if (interval_ti.get_type() != kINTERVAL_DAY_TIME) {
    return dtINVALID;
  }
  const auto& datum = interval->get_constval();
  switch (datum.timeval) {
    case 86400000:
      return get_dt_field(ts, interval_multiplier, false);
    case 3600000:
      return get_dt_field(ts, interval_multiplier, true);
    default:
      break;
  }
  return dtINVALID;
}

std::shared_ptr<Analyzer::Expr> remove_cast_to_date(const Analyzer::Expr* expr) {
  if (!expr) {
    return nullptr;
  }
  const auto uoper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (!uoper || uoper->get_optype() != kCAST) {
    return nullptr;
  }
  const auto& operand_ti = uoper->get_operand()->get_type_info();
  const auto& target_ti = uoper->get_type_info();
  if (operand_ti.get_type() != kTIMESTAMP || target_ti.get_type() != kDATE) {
    return nullptr;
  }
  return uoper->get_own_operand();
}

}  // namespace

std::shared_ptr<Analyzer::Expr> rewrite_to_date_trunc(
    const Analyzer::FunctionOper* dt_plus) {
  CHECK_EQ("DATETIME_PLUS", dt_plus->getName());
  CHECK_EQ(size_t(2), dt_plus->getArity());
  const auto ts = remove_cast_to_date(dt_plus->getArg(0));
  if (!ts) {
    return nullptr;
  }
  const auto off_arg = dt_plus->getArg(1);
  const auto dt_field = get_dt_field(ts.get(), off_arg);
  if (dt_field == dtINVALID) {
    return nullptr;
  }
  return Parser::DatetruncExpr::get(ts, dt_field);
}
