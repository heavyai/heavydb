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
 * @file		Planner.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Functions for query plan nodes
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cassert>
#include <iostream>
#include <stdexcept>
#include "../Analyzer/Analyzer.h"
#include "Planner.h"
#include "gen-cpp/MapD.h"

namespace Planner {

Scan::Scan(const Analyzer::RangeTblEntry& rte) : Plan() {
  table_id = rte.get_table_id();
  for (auto cd : rte.get_column_descs())
    col_list.push_back(cd->columnId);
}

RootPlan* Optimizer::optimize() {
  Plan* plan;
  SQLStmtType stmt_type = query.get_stmt_type();
  int result_table_id = 0;
  std::list<int> result_col_list;
  switch (stmt_type) {
    case kSELECT:
      // nothing to do for SELECT for now
      break;
    case kINSERT:
      result_table_id = query.get_result_table_id();
      result_col_list = query.get_result_col_list();
      break;
    case kUPDATE:
    case kDELETE:
      // should have been rejected by the Analyzer for now
      assert(false);
      break;
    default:
      assert(false);
  }
  plan = optimize_query();
  return new RootPlan(
      plan, stmt_type, result_table_id, result_col_list, catalog, query.get_limit(), query.get_offset());
}

Plan* Optimizer::optimize_query() {
  //@TODO add support for union queries
  if (query.get_next_query() != nullptr)
    throw std::runtime_error("UNION queries are not supported yet.");
  cur_query = &query;
  optimize_scans();
  cur_plan = nullptr;
  if (base_scans.size() > 2)
    throw std::runtime_error("More than two tables in a join not supported yet.");
  for (auto base_scan : base_scans) {
    if (cur_plan) {
      std::list<std::shared_ptr<Analyzer::Expr>> shared_join_predicates;
      for (auto join_pred : join_predicates) {
        shared_join_predicates.emplace_back(join_pred->deep_copy());
      }
      std::vector<std::shared_ptr<Analyzer::TargetEntry>> join_targetlist;
      for (const auto tle : cur_plan->get_targetlist()) {
        join_targetlist.emplace_back(
            new Analyzer::TargetEntry(tle->get_resname(), tle->get_expr()->deep_copy(), tle->get_unnest()));
      }
      cur_plan = new Join(join_targetlist, shared_join_predicates, 0, cur_plan, base_scan);
    } else {
      cur_plan = base_scan;
    }
  }
  optimize_current_query();
  if (query.get_order_by() != nullptr)
    optimize_orderby();
  return cur_plan;
}

void Optimizer::optimize_current_query() {
  if (cur_query->get_num_aggs() > 0 || cur_query->get_having_predicate() != nullptr ||
      !cur_query->get_group_by().empty())
    optimize_aggs();
  else {
    process_targetlist();
    if (!const_predicates.empty()) {
      // add result node to evaluate constant predicates
      std::vector<std::shared_ptr<Analyzer::TargetEntry>> tlist;
      int i = 1;
      for (auto tle : cur_query->get_targetlist()) {
        tlist.emplace_back(new Analyzer::TargetEntry(
            tle->get_resname(),
            makeExpr<Analyzer::Var>(tle->get_expr()->get_type_info(), Analyzer::Var::kINPUT_OUTER, i++),
            false));
      }
      std::list<std::shared_ptr<Analyzer::Expr>> const_quals;
      for (auto p : const_predicates)
        const_quals.push_back(p->deep_copy());
      cur_plan = new Result(tlist, {}, 0.0, cur_plan, const_quals);
    }
  }
}

void Optimizer::optimize_scans() {
  const std::vector<Analyzer::RangeTblEntry*>& rt = cur_query->get_rangetable();
  for (auto rte : rt) {
    base_scans.push_back(new Scan(*rte));
  }
  const Analyzer::Expr* where_pred = cur_query->get_where_predicate();
  std::list<const Analyzer::Expr*> scan_predicates;
  if (where_pred != nullptr)
    where_pred->group_predicates(scan_predicates, join_predicates, const_predicates);
  for (auto p : scan_predicates) {
    int rte_idx;
    auto simple_pred = p->normalize_simple_predicate(rte_idx);
    if (simple_pred != nullptr)
      base_scans[rte_idx]->add_simple_predicate(simple_pred);
    else {
      std::set<int> rte_idx_set;
      p->collect_rte_idx(rte_idx_set);
      for (auto x : rte_idx_set) {
        rte_idx = x;
        break;  // grab rte_idx out of the singleton set
      }
      base_scans[rte_idx]->add_predicate(p->deep_copy());
    }
  }
  const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& tlist = cur_query->get_targetlist();
  bool (*fn_pt)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*) = Analyzer::ColumnVar::colvar_comp;
  std::set<const Analyzer::ColumnVar*, bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)> colvar_set(
      fn_pt);
  for (auto tle : tlist)
    tle->get_expr()->collect_column_var(colvar_set, true);
  for (auto p : join_predicates)
    p->collect_column_var(colvar_set, true);
  const auto group_by = cur_query->get_group_by();
  if (!group_by.empty()) {
    for (auto e : group_by) {
      e->collect_column_var(colvar_set, true);
    }
  }
  const Analyzer::Expr* having_pred = cur_query->get_having_predicate();
  if (having_pred != nullptr)
    having_pred->collect_column_var(colvar_set, true);
  for (auto colvar : colvar_set) {
    auto tle = std::make_shared<Analyzer::TargetEntry>("", colvar->deep_copy(), false);
    base_scans[colvar->get_rte_idx()]->add_tle(tle);
  };
}

namespace {

const Planner::Scan* get_scan_child(const Planner::Plan* plan) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
  return agg_plan ? dynamic_cast<const Planner::Scan*>(plan->get_child_plan())
                  : dynamic_cast<const Planner::Scan*>(plan);
}

std::vector<std::shared_ptr<Analyzer::TargetEntry>> get_join_target_list(const Planner::Join* join_plan) {
  const auto outer_plan = get_scan_child(join_plan->get_outerplan());
  CHECK(outer_plan);
  auto join_target_list = outer_plan->get_targetlist();
  const auto inner_plan = get_scan_child(join_plan->get_innerplan());
  CHECK(inner_plan);
  const auto inner_target_list = inner_plan->get_targetlist();
  join_target_list.insert(join_target_list.end(), inner_target_list.begin(), inner_target_list.end());
  return join_target_list;
}

}  // namespace

void Optimizer::optimize_aggs() {
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> agg_tlist;
  const Analyzer::Expr* having_pred = cur_query->get_having_predicate();
  bool (*fn_pt)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*) = Analyzer::ColumnVar::colvar_comp;
  std::set<const Analyzer::ColumnVar*, bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)> colvar_set(
      fn_pt);
  auto is_agg = [](const Analyzer::Expr* e) -> bool { return typeid(*e) == typeid(Analyzer::AggExpr); };
  std::list<const Analyzer::Expr*> aggexpr_list;
  // need to determine if the final targetlist involves expressions over
  // the group by columns and/or aggregates, e.g., sum(x) + sum(y).
  // see no_expression to false if an expression is found.
  bool no_expression = true;
  for (auto tle : cur_query->get_targetlist()) {
    // collect all the aggregate functions from targetlist
    tle->get_expr()->find_expr(is_agg, aggexpr_list);
    if (!dynamic_cast<Analyzer::ColumnVar*>(tle->get_expr()) && !dynamic_cast<Analyzer::Var*>(tle->get_expr()) &&
        !dynamic_cast<Analyzer::AggExpr*>(tle->get_expr()))
      no_expression = false;
  }
  // collect all the group by columns in having clause
  if (having_pred != nullptr) {
    having_pred->find_expr(is_agg, aggexpr_list);
  }

  std::list<std::shared_ptr<Analyzer::Expr>> groupby_list;
  auto target_list = cur_plan->get_targetlist();
  if (dynamic_cast<const Planner::Join*>(cur_plan)) {
    target_list = get_join_target_list(static_cast<const Planner::Join*>(cur_plan));
  }
  if (!cur_query->get_group_by().empty()) {
    for (auto e : cur_query->get_group_by()) {
      groupby_list.push_back(e->rewrite_with_child_targetlist(target_list));
    }
  }

  // form the AggPlan targetlist with the group by columns followed by aggregates
  int varno = 1;
  for (auto e : groupby_list) {
    std::shared_ptr<Analyzer::TargetEntry> new_tle;
    std::shared_ptr<Analyzer::Expr> expr;
    auto c = std::dynamic_pointer_cast<Analyzer::ColumnVar>(e);
    if (c != nullptr) {
      expr = makeExpr<Analyzer::Var>(
          c->get_type_info(), c->get_table_id(), c->get_column_id(), c->get_rte_idx(), Analyzer::Var::kGROUPBY, varno);
    } else
      expr = makeExpr<Analyzer::Var>(e->get_type_info(), Analyzer::Var::kGROUPBY, varno);
    new_tle = std::make_shared<Analyzer::TargetEntry>("", expr, false);
    agg_tlist.push_back(new_tle);
    varno++;
  }
  for (auto e : aggexpr_list) {
    std::shared_ptr<Analyzer::TargetEntry> new_tle;
    new_tle = std::make_shared<Analyzer::TargetEntry>("", e->rewrite_with_child_targetlist(target_list), false);
    agg_tlist.push_back(new_tle);
  }

  std::list<std::shared_ptr<Analyzer::Expr>> having_quals;
  if (having_pred != nullptr) {
    std::list<const Analyzer::Expr*> preds;
    having_pred->group_predicates(preds, preds, preds);
    for (auto p : preds) {
      having_quals.push_back(p->rewrite_agg_to_var(agg_tlist));
    }
  }

  cur_plan = new AggPlan(agg_tlist, 0.0, cur_plan, groupby_list);
  if (no_expression && having_pred == nullptr)
    // in this case, no need to add a Result node on top
    process_targetlist();
  else {
    std::vector<std::shared_ptr<Analyzer::TargetEntry>> result_tlist;
    for (auto tle : cur_query->get_targetlist()) {
      result_tlist.emplace_back(new Analyzer::TargetEntry(
          tle->get_resname(), tle->get_expr()->rewrite_agg_to_var(agg_tlist), tle->get_unnest()));
    }
    std::list<std::shared_ptr<Analyzer::Expr>> const_quals;
    for (auto p : const_predicates)
      const_quals.push_back(p->deep_copy());
    cur_plan = new Result(result_tlist, having_quals, 0.0, cur_plan, const_quals);
  }
}

void Optimizer::optimize_orderby() {
  if (query.get_order_by() == nullptr)
    return;
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> tlist;
  int varno = 1;
  for (auto tle : cur_plan->get_targetlist()) {
    tlist.emplace_back(new Analyzer::TargetEntry(
        tle->get_resname(),
        makeExpr<Analyzer::Var>(tle->get_expr()->get_type_info(), Analyzer::Var::kINPUT_OUTER, varno),
        false));
    varno++;
  }
  cur_plan = new Sort(tlist, 0.0, cur_plan, *query.get_order_by(), query.get_is_distinct());
}

void Optimizer::process_targetlist() {
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> final_tlist;
  for (auto tle : query.get_targetlist()) {
    std::shared_ptr<Analyzer::TargetEntry> new_tle;
    if (cur_plan == nullptr)
      new_tle =
          std::make_shared<Analyzer::TargetEntry>(tle->get_resname(), tle->get_expr()->deep_copy(), tle->get_unnest());
    else {
      auto target_list = cur_plan->get_targetlist();
      if (dynamic_cast<const Planner::Join*>(cur_plan)) {
        target_list = get_join_target_list(static_cast<const Planner::Join*>(cur_plan));
      }
      new_tle = std::make_shared<Analyzer::TargetEntry>(
          tle->get_resname(), tle->get_expr()->rewrite_with_targetlist(target_list), tle->get_unnest());
    }
    final_tlist.push_back(new_tle);
  }
  if (cur_plan == nullptr)
    cur_plan = new ValuesScan(final_tlist);
  else {
    cur_plan->set_targetlist(final_tlist);
  }
}

void Plan::print() const {
  std::cout << "targetlist: ";
  for (auto t : targetlist)
    t->print();
  std::cout << std::endl;
  std::cout << "quals: ";
  for (auto p : quals)
    p->print();
  std::cout << std::endl;
}

void Result::print() const {
  std::cout << "(Result" << std::endl;
  Plan::print();
  child_plan->print();
  std::cout << "const_quals: ";
  for (auto p : const_quals)
    p->print();
  std::cout << ")" << std::endl;
}

void Scan::print() const {
  std::cout << "(Scan" << std::endl;
  Plan::print();
  std::cout << "simple_quals: ";
  for (auto p : simple_quals)
    p->print();
  std::cout << std::endl << "table: " << table_id;
  std::cout << " columns: ";
  for (auto i : col_list) {
    std::cout << i;
    std::cout << " ";
  }
  std::cout << ")" << std::endl;
}

void ValuesScan::print() const {
  std::cout << "(ValuesScan" << std::endl;
  Plan::print();
  std::cout << ")" << std::endl;
}

void Join::print() const {
  std::cout << "(Join" << std::endl;
  Plan::print();
  std::cout << "Outer Plan: ";
  get_outerplan()->print();
  std::cout << "Inner Plan: ";
  get_innerplan()->print();
  std::cout << ")" << std::endl;
}

void AggPlan::print() const {
  std::cout << "(Agg" << std::endl;
  Plan::print();
  child_plan->print();
  std::cout << "Group By: ";
  for (auto e : groupby_list)
    e->print();
  std::cout << ")" << std::endl;
}

void Append::print() const {
  std::cout << "(Append" << std::endl;
  for (const auto& p : plan_list)
    p->print();
  std::cout << ")" << std::endl;
}

void MergeAppend::print() const {
  std::cout << "(MergeAppend" << std::endl;
  for (const auto& p : mergeplan_list)
    p->print();
  std::cout << ")" << std::endl;
}

void Sort::print() const {
  std::cout << "(Sort" << std::endl;
  Plan::print();
  child_plan->print();
  std::cout << "Order By: ";
  for (auto o : order_entries)
    o.print();
  std::cout << ")" << std::endl;
}

void RootPlan::print() const {
  std::cout << "(RootPlan ";
  switch (stmt_type) {
    case kSELECT:
      std::cout << "SELECT" << std::endl;
      break;
    case kUPDATE:
      std::cout << "UPDATE "
                << "result table: " << result_table_id << " columns: ";
      for (auto i : result_col_list) {
        std::cout << i;
        std::cout << " ";
      }
      std::cout << std::endl;
      break;
    case kINSERT:
      std::cout << "INSERT "
                << "result table: " << result_table_id << " columns: ";
      for (auto i : result_col_list) {
        std::cout << i;
        std::cout << " ";
      }
      std::cout << std::endl;
      break;
    case kDELETE:
      std::cout << "DELETE "
                << "result table: " << result_table_id << std::endl;
      break;
    default:
      break;
  }
  plan->print();
  std::cout << "limit: " << limit;
  std::cout << " offset: " << offset << ")" << std::endl;
}
}
