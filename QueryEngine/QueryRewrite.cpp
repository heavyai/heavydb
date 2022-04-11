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

#include <algorithm>
#include <memory>
#include <vector>

#include "ExpressionRange.h"
#include "ExpressionRewrite.h"
#include "Logger/Logger.h"
#include "Shared/sqltypes.h"

extern size_t g_constrained_by_in_threshold;
extern bool g_enable_overlaps_hashjoin;

RelAlgExecutionUnit QueryRewriter::rewrite(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  auto rewritten_exe_unit = rewriteConstrainedByIn(ra_exe_unit_in);
  auto rewritten_exe_unit_for_agg_on_gby_col =
      rewriteAggregateOnGroupByColumn(rewritten_exe_unit);
  return rewriteOverlapsJoin(rewritten_exe_unit_for_agg_on_gby_col);
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
      join_condition.quals.push_back(join_qual_expr_in);
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
          ra_exe_unit_in.query_hint,
          ra_exe_unit_in.query_plan_dag,
          ra_exe_unit_in.hash_table_build_plan_dag,
          ra_exe_unit_in.table_id_to_node_map,
          ra_exe_unit_in.use_bump_allocator};
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
  for (const auto& in_val : in_vals->get_value_list()) {
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
  for (const auto& group_expr : ra_exe_unit_in.groupby_exprs) {
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
          ra_exe_unit_in.scan_limit,
          ra_exe_unit_in.query_hint,
          ra_exe_unit_in.query_plan_dag,
          ra_exe_unit_in.hash_table_build_plan_dag,
          ra_exe_unit_in.table_id_to_node_map};
}

std::shared_ptr<Analyzer::CaseExpr> QueryRewriter::generateCaseForDomainValues(
    const Analyzer::InValues* in_vals) {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      case_expr_list;
  auto in_val_arg = in_vals->get_arg()->deep_copy();
  for (const auto& in_val : in_vals->get_value_list()) {
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

std::shared_ptr<Analyzer::CaseExpr>
QueryRewriter::generateCaseExprForCountDistinctOnGroupByCol(
    std::shared_ptr<Analyzer::Expr> expr) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      case_expr_list;
  auto is_null = std::make_shared<Analyzer::UOper>(kBOOLEAN, kISNULL, expr);
  auto is_not_null = std::make_shared<Analyzer::UOper>(kBOOLEAN, kNOT, is_null);
  Datum then_d;
  then_d.bigintval = 1;
  const auto then_constant = makeExpr<Analyzer::Constant>(kBIGINT, false, then_d);
  case_expr_list.emplace_back(is_not_null, then_constant);
  Datum else_d;
  else_d.bigintval = 0;
  const auto else_constant = makeExpr<Analyzer::Constant>(kBIGINT, false, else_d);
  auto case_expr = makeExpr<Analyzer::CaseExpr>(
      then_constant->get_type_info(), false, case_expr_list, else_constant);
  return case_expr;
}

std::pair<bool, std::set<size_t>> QueryRewriter::is_all_groupby_exprs_are_col_var(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs) const {
  std::set<size_t> gby_col_exprs_hash;
  for (auto gby_expr : groupby_exprs) {
    if (auto gby_col_var = std::dynamic_pointer_cast<Analyzer::ColumnVar>(gby_expr)) {
      gby_col_exprs_hash.insert(boost::hash_value(gby_col_var->toString()));
    } else {
      return {false, {}};
    }
  }
  return {true, gby_col_exprs_hash};
}

RelAlgExecutionUnit QueryRewriter::rewriteAggregateOnGroupByColumn(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  auto check_precond = is_all_groupby_exprs_are_col_var(ra_exe_unit_in.groupby_exprs);
  auto is_expr_on_gby_col = [&check_precond](const Analyzer::AggExpr* agg_expr) {
    CHECK(agg_expr);
    if (agg_expr->get_arg()) {
      // some expr does not have its own arg, i.e., count(*)
      auto agg_expr_hash = boost::hash_value(agg_expr->get_arg()->toString());
      // a valid expr should have hashed value > 0
      CHECK_GT(agg_expr_hash, 0u);
      if (check_precond.second.count(agg_expr_hash)) {
        return true;
      }
    }
    return false;
  };
  if (!check_precond.first) {
    // return the input ra_exe_unit if we have gby expr which is not col_var
    // i.e., group by x+1, y instead of group by x, y
    // todo (yoonmin) : can we relax this with a simple analysis of groupby / agg exprs?
    return ra_exe_unit_in;
  }

  std::vector<Analyzer::Expr*> new_target_exprs;
  for (auto expr : ra_exe_unit_in.target_exprs) {
    bool rewritten = false;
    if (auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(expr)) {
      if (is_expr_on_gby_col(agg_expr)) {
        auto target_expr = agg_expr->get_arg();
        // we have some issues when this rewriting is applied to float_type groupby column
        // in subquery, i.e., SELECT MIN(v1) FROM (SELECT v1, AGG(v1) FROM T GROUP BY v1);
        if (target_expr && target_expr->get_type_info().get_type() != SQLTypes::kFLOAT) {
          switch (agg_expr->get_aggtype()) {
            case SQLAgg::kCOUNT:
            case SQLAgg::kAPPROX_COUNT_DISTINCT: {
              if (agg_expr->get_aggtype() == SQLAgg::kCOUNT &&
                  !agg_expr->get_is_distinct()) {
                break;
              }
              auto case_expr =
                  generateCaseExprForCountDistinctOnGroupByCol(agg_expr->get_own_arg());
              new_target_exprs.push_back(case_expr.get());
              target_exprs_owned_.emplace_back(case_expr);
              rewritten = true;
              break;
            }
            case SQLAgg::kAPPROX_QUANTILE:
            case SQLAgg::kAVG:
            case SQLAgg::kSAMPLE:
            case SQLAgg::kMAX:
            case SQLAgg::kMIN: {
              // we just replace the agg_expr into a plain expr
              // i.e, avg(x1) --> x1
              auto agg_expr_ti = agg_expr->get_type_info();
              auto target_expr = agg_expr->get_own_arg();
              if (agg_expr_ti != target_expr->get_type_info()) {
                target_expr = target_expr->add_cast(agg_expr_ti);
              }
              new_target_exprs.push_back(target_expr.get());
              target_exprs_owned_.emplace_back(target_expr);
              rewritten = true;
              break;
            }
            default:
              break;
          }
        }
      }
    }
    if (!rewritten) {
      new_target_exprs.push_back(expr);
    }
  }

  RelAlgExecutionUnit rewritten_exe_unit{ra_exe_unit_in.input_descs,
                                         ra_exe_unit_in.input_col_descs,
                                         ra_exe_unit_in.simple_quals,
                                         ra_exe_unit_in.quals,
                                         ra_exe_unit_in.join_quals,
                                         ra_exe_unit_in.groupby_exprs,
                                         new_target_exprs,
                                         ra_exe_unit_in.estimator,
                                         ra_exe_unit_in.sort_info,
                                         ra_exe_unit_in.scan_limit,
                                         ra_exe_unit_in.query_hint,
                                         ra_exe_unit_in.query_plan_dag,
                                         ra_exe_unit_in.hash_table_build_plan_dag,
                                         ra_exe_unit_in.table_id_to_node_map,
                                         ra_exe_unit_in.use_bump_allocator,
                                         ra_exe_unit_in.union_all,
                                         ra_exe_unit_in.query_state};
  return rewritten_exe_unit;
}
