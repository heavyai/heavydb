/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "JoinFilterPushDown.h"
#include "DeepCopyVisitor.h"
#include "RelAlgExecutor.h"

namespace {

class BindFilterToOutermostVisitor : public DeepCopyVisitor {
  std::shared_ptr<Analyzer::Expr> visitColumnVar(
      const Analyzer::ColumnVar* col_var) const override {
    return makeExpr<Analyzer::ColumnVar>(
        col_var->get_type_info(), col_var->get_table_id(), col_var->get_column_id(), 0);
  }
};

class CollectInputColumnsVisitor
    : public ScalarExprVisitor<std::unordered_set<InputColDescriptor>> {
  std::unordered_set<InputColDescriptor> visitColumnVar(
      const Analyzer::ColumnVar* col_var) const override {
    return {InputColDescriptor(col_var->get_column_id(), col_var->get_table_id(), 0)};
  }

 public:
  std::unordered_set<InputColDescriptor> aggregateResult(
      const std::unordered_set<InputColDescriptor>& aggregate,
      const std::unordered_set<InputColDescriptor>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

}  // namespace

/**
 * Given a set of filter expressions for a table, it launches a new COUNT query to
 * compute the number of passing rows, and then generates a set of statistics
 * related to those filters.
 * Later, these stats are used to decide whether
 * a filter should be pushed down or not.
 */
FilterSelectivity RelAlgExecutor::getFilterSelectivity(
    const std::vector<std::shared_ptr<Analyzer::Expr>>& filter_expressions,
    const CompilationOptions& co,
    const ExecutionOptions& eo) {
  CollectInputColumnsVisitor input_columns_visitor;
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  std::unordered_set<InputColDescriptor> input_column_descriptors;
  BindFilterToOutermostVisitor bind_filter_to_outermost;
  for (const auto& filter_expr : filter_expressions) {
    input_column_descriptors = input_columns_visitor.aggregateResult(
        input_column_descriptors, input_columns_visitor.visit(filter_expr.get()));
    quals.push_back(bind_filter_to_outermost.visit(filter_expr.get()));
  }
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  for (const auto& input_col_desc : input_column_descriptors) {
    if (input_descs.empty()) {
      input_descs.push_back(input_col_desc.getScanDesc());
    } else {
      CHECK(input_col_desc.getScanDesc() == input_descs.front());
    }
    input_col_descs.push_back(std::make_shared<const InputColDescriptor>(input_col_desc));
  }
  const auto count_expr =
      makeExpr<Analyzer::AggExpr>(SQLTypeInfo(g_bigint_count ? kBIGINT : kINT, false),
                                  kCOUNT,
                                  nullptr,
                                  false,
                                  nullptr);
  RelAlgExecutionUnit ra_exe_unit{input_descs,
                                  input_col_descs,
                                  {},
                                  quals,
                                  {},
                                  {},
                                  {count_expr.get()},
                                  nullptr,
                                  {{}, SortAlgorithm::Default, 0, 0},
                                  0};
  int32_t error_code{0};
  size_t one{1};
  ResultSetPtr filtered_result;
  const auto table_infos = get_table_infos(input_descs, executor_);
  CHECK_EQ(size_t(1), table_infos.size());
  const size_t total_rows_upper_bound = table_infos.front().info.getNumTuplesUpperBound();
  try {
    filtered_result = executor_->executeWorkUnit(&error_code,
                                                 one,
                                                 true,
                                                 table_infos,
                                                 ra_exe_unit,
                                                 co,
                                                 eo,
                                                 cat_,
                                                 executor_->getRowSetMemoryOwner(),
                                                 nullptr,
                                                 false);
  } catch (...) {
    return {false, 1.0, 0};
  }
  if (error_code) {
    return {false, 1.0, 0};
  }
  const auto count_row = filtered_result->getNextRow(false, false);
  CHECK_EQ(size_t(1), count_row.size());
  const auto& count_tv = count_row.front();
  const auto count_scalar_tv = boost::get<ScalarTargetValue>(&count_tv);
  CHECK(count_scalar_tv);
  const auto count_ptr = boost::get<int64_t>(count_scalar_tv);
  CHECK(count_ptr);
  const auto rows_passing = *count_ptr;
  const auto rows_total = std::max(total_rows_upper_bound, size_t(1));
  return {true, static_cast<float>(rows_passing) / rows_total, total_rows_upper_bound};
}

/**
 * Goes through all candidate filters and evaluate whether they pass the selectivity
 * criteria or not.
 */
std::vector<PushedDownFilterInfo> RelAlgExecutor::selectFiltersToBePushedDown(
    const RelAlgExecutor::WorkUnit& work_unit,
    const CompilationOptions& co,
    const ExecutionOptions& eo) {
  const auto all_push_down_candidates =
      find_push_down_filters(work_unit.exe_unit,
                             work_unit.input_permutation,
                             work_unit.left_deep_join_input_sizes);
  std::vector<PushedDownFilterInfo> selective_push_down_candidates;
  const auto ti = get_table_infos(work_unit.exe_unit.input_descs, executor_);
  if (to_gather_info_for_filter_selectivity(ti)) {
    for (const auto& candidate : all_push_down_candidates) {
      const auto selectivity = getFilterSelectivity(candidate.filter_expressions, co, eo);
      if (selectivity.is_valid && selectivity.isFilterSelectiveEnough()) {
        selective_push_down_candidates.push_back(candidate);
      }
    }
  }
  return selective_push_down_candidates;
}

ExecutionResult RelAlgExecutor::executeRelAlgQueryWithFilterPushDown(
    std::vector<RaExecutionDesc>& ed_list,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    RenderInfo* render_info,
    const int64_t queue_time_ms) {
  // we currently do not fully support filter push down with
  // multi-step execution and/or with subqueries
  // TODO(Saman): add proper caching to enable filter push down for all cases
  if (ed_list.size() > 1 || !subqueries_.empty()) {
    if (eo.just_calcite_explain) {
      return ExecutionResult(std::vector<PushedDownFilterInfo>{},
                             eo.find_push_down_candidates);
    }
    const ExecutionOptions eo_modified{eo.output_columnar_hint,
                                       eo.allow_multifrag,
                                       eo.just_explain,
                                       eo.allow_loop_joins,
                                       eo.with_watchdog,
                                       eo.jit_debug,
                                       eo.just_validate,
                                       eo.with_dynamic_watchdog,
                                       eo.dynamic_watchdog_time_limit,
                                       /*find_push_down_candidates=*/false,
                                       /*just_calcite_explain=*/false,
                                       eo.gpu_input_mem_limit_percent};

    // Dispatch the subqueries first
    for (auto subquery : subqueries_) {
      // Execute the subquery and cache the result.
      RelAlgExecutor ra_executor(executor_, cat_);
      auto result = ra_executor.executeRelAlgSubQuery(subquery.get(), co, eo_modified);
      subquery->setExecutionResult(std::make_shared<ExecutionResult>(result));
    }
    return executeRelAlgSeq(ed_list, co, eo_modified, render_info, queue_time_ms);
  } else {
    // Dispatch the subqueries first
    for (auto subquery : subqueries_) {
      // Execute the subquery and cache the result.
      RelAlgExecutor ra_executor(executor_, cat_);
      auto result = ra_executor.executeRelAlgSubQuery(subquery.get(), co, eo);
      subquery->setExecutionResult(std::make_shared<ExecutionResult>(result));
    }
    return executeRelAlgSeq(ed_list, co, eo, render_info, queue_time_ms);
  }
}
/**
 * The main purpose of this function is to prevent going through extra overhead of
 * computing required statistics for finding the right candidates and then the actual
 * push-down, unless the problem is large enough that such effort is potentially helpful.
 */
bool to_gather_info_for_filter_selectivity(
    const std::vector<InputTableInfo>& table_infos) {
  if (table_infos.size() < 2) {
    return false;
  }
  // we currently do not support filter push down when there is a self-join involved:
  // TODO(Saman): prevent Calcite from optimizing self-joins to remove this exclusion
  std::unordered_set<int> table_ids;
  for (auto ti : table_infos) {
    if (table_ids.find(ti.table_id) == table_ids.end()) {
      table_ids.insert(ti.table_id);
    } else {
      // a self-join is involved
      return false;
    }
  }
  // TODO(Saman): add some extra heuristics to avoid preflight count and push down if it
  // is not going to be helpful.
  return true;
}

/**
 * Go through all tables involved in the relational algebra plan, and select potential
 * candidates to be pushed down by calcite. For each filter we store a set of
 * intermediate indices (previous, current, and next table) based on the column
 * indices in their query string.
 */
std::vector<PushedDownFilterInfo> find_push_down_filters(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<size_t>& input_permutation,
    const std::vector<size_t>& left_deep_join_input_sizes) {
  std::vector<PushedDownFilterInfo> result;
  if (left_deep_join_input_sizes.empty()) {
    return result;
  }
  std::vector<size_t> input_size_prefix_sums(left_deep_join_input_sizes.size());
  std::partial_sum(left_deep_join_input_sizes.begin(),
                   left_deep_join_input_sizes.end(),
                   input_size_prefix_sums.begin());
  std::vector<int> to_original_rte_idx(ra_exe_unit.input_descs.size(),
                                       ra_exe_unit.input_descs.size());
  if (!input_permutation.empty()) {
    CHECK_EQ(to_original_rte_idx.size(), input_permutation.size());
    for (size_t i = 0; i < input_permutation.size(); ++i) {
      CHECK_LT(input_permutation[i], to_original_rte_idx.size());
      CHECK_EQ(to_original_rte_idx[input_permutation[i]], to_original_rte_idx.size());
      to_original_rte_idx[input_permutation[i]] = i;
    }
  } else {
    std::iota(to_original_rte_idx.begin(), to_original_rte_idx.end(), 0);
  }
  std::unordered_map<int, std::vector<std::shared_ptr<Analyzer::Expr>>>
      filters_per_nesting_level;
  for (const auto& level_conditions : ra_exe_unit.join_quals) {
    AllRangeTableIndexVisitor visitor;
    for (const auto& cond : level_conditions.quals) {
      const auto rte_indices = visitor.visit(cond.get());
      if (rte_indices.size() > 1) {
        continue;
      }
      const int rte_idx = (!rte_indices.empty()) ? *rte_indices.cbegin() : 0;
      if (!rte_idx) {
        continue;
      }
      CHECK_GE(rte_idx, 0);
      CHECK_LT(static_cast<size_t>(rte_idx), to_original_rte_idx.size());
      filters_per_nesting_level[to_original_rte_idx[rte_idx]].push_back(cond);
    }
  }
  for (const auto& kv : filters_per_nesting_level) {
    CHECK_GE(kv.first, 0);
    CHECK_LT(kv.first, input_size_prefix_sums.size());
    size_t input_prev = (kv.first > 1) ? input_size_prefix_sums[kv.first - 2] : 0;
    size_t input_start = kv.first ? input_size_prefix_sums[kv.first - 1] : 0;
    size_t input_next = input_size_prefix_sums[kv.first];
    result.emplace_back(
        PushedDownFilterInfo{kv.second, input_prev, input_start, input_next});
  }
  return result;
}
