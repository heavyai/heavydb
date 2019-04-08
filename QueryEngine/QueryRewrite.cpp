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

#include "QueryRewrite.h"
#include "ExpressionRange.h"
#include "ExpressionRewrite.h"

#include <glog/logging.h>

RelAlgExecutionUnit QueryRewriter::rewrite(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  auto rewritten_exe_unit = rewriteConstrainedByIn(ra_exe_unit_in);
  return rewriteOverlapsJoin(rewritten_exe_unit);
}

RelAlgExecutionUnit QueryRewriter::rewriteOverlapsJoin(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  if (!g_enable_overlaps_hashjoin) {
    return ra_exe_unit_in;
  }
  if (ra_exe_unit_in.join_quals.empty()) {
    return ra_exe_unit_in;
  }

  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  quals.insert(quals.end(), ra_exe_unit_in.quals.begin(), ra_exe_unit_in.quals.end());

  JoinQualsPerNestingLevel join_condition_per_nesting_level;
  for (const auto& join_condition_in : ra_exe_unit_in.join_quals) {
    JoinCondition join_condition{{}, join_condition_in.type};

    for (const auto& join_qual_expr_in : join_condition_in.quals) {
      auto new_overlaps_quals = rewrite_overlaps_conjunction(join_qual_expr_in);
      if (new_overlaps_quals) {
        const auto& overlaps_quals = *new_overlaps_quals;
        join_condition.quals.insert(join_condition.quals.end(),
                                    overlaps_quals.join_quals.begin(),
                                    overlaps_quals.join_quals.end());

        quals.insert(
            quals.end(), overlaps_quals.quals.begin(), overlaps_quals.quals.end());
      } else {
        join_condition.quals.push_back(join_qual_expr_in);
      }
    }
    join_condition_per_nesting_level.push_back(join_condition);
  }
  return {ra_exe_unit_in.input_descs,
          ra_exe_unit_in.input_col_descs,
          ra_exe_unit_in.simple_quals,
          quals,
          join_condition_per_nesting_level,
          ra_exe_unit_in.groupby_exprs,
          ra_exe_unit_in.target_exprs,
          ra_exe_unit_in.estimator,
          ra_exe_unit_in.sort_info,
          ra_exe_unit_in.scan_limit,
          ra_exe_unit_in.query_features};
}

RelAlgExecutionUnit QueryRewriter::rewriteConstrainedByIn(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  if (ra_exe_unit_in.groupby_exprs.empty()) {
    return ra_exe_unit_in;
  }
  if (ra_exe_unit_in.groupby_exprs.size() == 1 && !ra_exe_unit_in.groupby_exprs.front()) {
    return ra_exe_unit_in;
  }
  if (!ra_exe_unit_in.simple_quals.empty()) {
    return ra_exe_unit_in;
  }
  if (ra_exe_unit_in.quals.size() != 1) {
    return ra_exe_unit_in;
  }
  auto in_vals =
      std::dynamic_pointer_cast<Analyzer::InValues>(ra_exe_unit_in.quals.front());
  if (!in_vals) {
    in_vals = std::dynamic_pointer_cast<Analyzer::InValues>(
        rewrite_expr(ra_exe_unit_in.quals.front().get()));
  }
  if (!in_vals || in_vals->get_value_list().empty()) {
    return ra_exe_unit_in;
  }
  for (const auto in_val : in_vals->get_value_list()) {
    if (!std::dynamic_pointer_cast<Analyzer::Constant>(in_val)) {
      break;
    }
  }
  if (dynamic_cast<const Analyzer::CaseExpr*>(in_vals->get_arg())) {
    return ra_exe_unit_in;
  }
  auto case_expr = generateCaseForDomainValues(in_vals.get());
  return rewriteConstrainedByInImpl(ra_exe_unit_in, case_expr, in_vals.get());
}

RelAlgExecutionUnit QueryRewriter::rewriteConstrainedByInImpl(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const std::shared_ptr<Analyzer::CaseExpr> case_expr,
    const Analyzer::InValues* in_vals) const {
  std::list<std::shared_ptr<Analyzer::Expr>> new_groupby_list;
  std::vector<Analyzer::Expr*> new_target_exprs;
  bool rewrite{false};
  size_t groupby_idx{0};
  auto it = ra_exe_unit_in.groupby_exprs.begin();
  for (const auto group_expr : ra_exe_unit_in.groupby_exprs) {
    CHECK(group_expr);
    ++groupby_idx;
    if (*group_expr == *in_vals->get_arg()) {
      const auto expr_range = getExpressionRange(it->get(), query_infos_, executor_);
      if (expr_range.getType() != ExpressionRangeType::Integer) {
        ++it;
        continue;
      }
      const size_t range_sz = expr_range.getIntMax() - expr_range.getIntMin() + 1;
      if (range_sz <= in_vals->get_value_list().size() * g_constrained_by_in_threshold) {
        ++it;
        continue;
      }
      new_groupby_list.push_back(case_expr);
      for (size_t i = 0; i < ra_exe_unit_in.target_exprs.size(); ++i) {
        const auto target = ra_exe_unit_in.target_exprs[i];
        if (*target == *in_vals->get_arg()) {
          auto var_case_expr = makeExpr<Analyzer::Var>(
              case_expr->get_type_info(), Analyzer::Var::kGROUPBY, groupby_idx);
          target_exprs_owned_.push_back(var_case_expr);
          new_target_exprs.push_back(var_case_expr.get());
        } else {
          new_target_exprs.push_back(target);
        }
      }
      rewrite = true;
    } else {
      new_groupby_list.push_back(group_expr);
    }
    ++it;
  }
  if (!rewrite) {
    return ra_exe_unit_in;
  }
  return {ra_exe_unit_in.input_descs,
          ra_exe_unit_in.input_col_descs,
          ra_exe_unit_in.simple_quals,
          ra_exe_unit_in.quals,
          ra_exe_unit_in.join_quals,
          new_groupby_list,
          new_target_exprs,
          nullptr,
          ra_exe_unit_in.sort_info,
          ra_exe_unit_in.scan_limit};
}

std::shared_ptr<Analyzer::CaseExpr> QueryRewriter::generateCaseForDomainValues(
    const Analyzer::InValues* in_vals) {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      case_expr_list;
  auto in_val_arg = in_vals->get_arg()->deep_copy();
  for (const auto in_val : in_vals->get_value_list()) {
    auto case_cond = makeExpr<Analyzer::BinOper>(
        SQLTypeInfo(kBOOLEAN, true), false, kEQ, kONE, in_val_arg, in_val);
    auto in_val_copy = in_val->deep_copy();
    auto ti = in_val_copy->get_type_info();
    if (ti.is_string() && ti.get_compression() == kENCODING_DICT) {
      ti.set_comp_param(0);
    }
    in_val_copy->set_type_info(ti);
    case_expr_list.emplace_back(case_cond, in_val_copy);
  }
  // TODO(alex): refine the expression range for case with empty else expression;
  //             for now, add a dummy else which should never be taken
  auto else_expr = case_expr_list.front().second;
  return makeExpr<Analyzer::CaseExpr>(
      case_expr_list.front().second->get_type_info(), false, case_expr_list, else_expr);
}
