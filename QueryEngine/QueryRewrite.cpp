#include "QueryRewrite.h"
#include "ExpressionRange.h"

#include <glog/logging.h>

void QueryRewriter::rewrite() {
  if (!dynamic_cast<const Planner::AggPlan*>(plan_)) {
    return;
  }
  rewriteConstrainedByIn();
}

void QueryRewriter::rewriteConstrainedByIn() {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan_);
  CHECK(agg_plan);
  const auto& groupby_list = agg_plan->get_groupby_list();
  if (groupby_list.empty()) {
    return;
  }
  const auto scan_plan = static_cast<const Planner::Scan*>(agg_plan->get_child_plan());
  if (!scan_plan->get_simple_quals().empty()) {
    return;
  }
  const auto& quals = scan_plan->get_quals();
  if (quals.size() != 1 || !std::dynamic_pointer_cast<Analyzer::InValues>(quals.front())) {
    return;
  }
  const auto in_vals = std::static_pointer_cast<Analyzer::InValues>(quals.front());
  CHECK(!in_vals->get_value_list().empty());
  for (const auto in_val : in_vals->get_value_list()) {
    if (!std::dynamic_pointer_cast<Analyzer::Constant>(in_val)) {
      break;
    }
  }
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>> case_expr_list;
  if (dynamic_cast<const Analyzer::CaseExpr*>(in_vals->get_arg())) {
    return;
  }
  auto in_val_arg = in_vals->get_arg()->deep_copy();
  for (const auto in_val : in_vals->get_value_list()) {
    auto case_cond = makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, true), false, kEQ, kONE, in_val_arg, in_val);
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
  auto case_expr =
      makeExpr<Analyzer::CaseExpr>(case_expr_list.front().second->get_type_info(), false, case_expr_list, else_expr);
  std::list<std::shared_ptr<Analyzer::Expr>> new_groupby_list;
  bool rewrite{false};
  size_t groupby_idx{0};
  for (const auto group_expr : groupby_list) {
    ++groupby_idx;
    if (*group_expr == *in_vals->get_arg()) {
      const auto expr_range = getExpressionRange(group_expr.get(), query_infos_, executor_);
      if (expr_range.getType() != ExpressionRangeType::Integer) {
        continue;
      }
      const size_t use_constraint_thresh{10};
      const size_t range_sz = expr_range.getIntMax() - expr_range.getIntMin() + 1;
      if (range_sz <= in_vals->get_value_list().size() * use_constraint_thresh) {
        continue;
      }
      new_groupby_list.push_back(case_expr);
      for (auto target : agg_plan->get_targetlist()) {
        if (*target->get_expr() == *in_vals->get_arg()) {
          auto var_case_expr =
              makeExpr<Analyzer::Var>(case_expr->get_type_info(), Analyzer::Var::kGROUPBY, groupby_idx);
          target->set_expr(var_case_expr);
        }
      }
      rewrite = true;
    } else {
      new_groupby_list.push_back(group_expr);
    }
  }
  if (!rewrite) {
    return;
  }
  const_cast<Planner::AggPlan*>(agg_plan)->set_groupby_list(new_groupby_list);
}
