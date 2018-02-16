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

#include "Execute.h"
#include "QueryRewrite.h"

#include "Shared/scope.h"

// The legacy way of executing queries. Don't change it, it's going away.

namespace {

int64_t get_scan_limit(const Planner::Plan* plan, const int64_t limit) {
  return (dynamic_cast<const Planner::Scan*>(plan) || dynamic_cast<const Planner::Join*>(plan)) ? limit : 0;
}

const Planner::Scan* get_scan_child(const Planner::Plan* plan) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
  return agg_plan ? dynamic_cast<const Planner::Scan*>(plan->get_child_plan())
                  : dynamic_cast<const Planner::Scan*>(plan);
}

const Planner::Join* get_join_child(const Planner::Plan* plan) {
  const auto join_plan = dynamic_cast<const Planner::Join*>(plan);
  return join_plan ? join_plan : dynamic_cast<const Planner::Join*>(plan->get_child_plan());
}

void collect_simple_quals(std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals, const Planner::Scan* scan_plan) {
  CHECK(scan_plan);
  const auto& more_simple_quals = scan_plan->get_simple_quals();
  simple_quals.insert(simple_quals.end(), more_simple_quals.begin(), more_simple_quals.end());
}

void collect_quals(std::list<std::shared_ptr<Analyzer::Expr>>& quals, const Planner::Scan* scan_plan) {
  CHECK(scan_plan);
  const auto& more_quals = scan_plan->get_quals();
  quals.insert(quals.end(), more_quals.begin(), more_quals.end());
}

void collect_quals_from_join(std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                             std::list<std::shared_ptr<Analyzer::Expr>>& quals,
                             const Planner::Join* join_plan) {
  const auto outer_plan = get_scan_child(join_plan->get_outerplan());
  CHECK(outer_plan);
  const auto inner_plan = get_scan_child(join_plan->get_innerplan());
  CHECK(inner_plan);
  collect_simple_quals(simple_quals, outer_plan);
  collect_simple_quals(simple_quals, inner_plan);
  collect_quals(quals, outer_plan);
  collect_quals(quals, inner_plan);
}

bool check_plan_sanity(const Planner::Plan* plan) {
  const auto join_plan = get_join_child(plan);
  const auto scan_plan = get_scan_child(plan);
  return static_cast<bool>(scan_plan) != static_cast<bool>(join_plan);
}

void collect_input_col_descs(std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs,
                             const Planner::Scan* scan_plan,
                             const Catalog_Namespace::Catalog& cat,
                             const bool is_join,
                             const size_t scan_idx) {
  CHECK(scan_idx == 0 || is_join);
  CHECK(scan_plan);
  const int table_id = scan_plan->get_table_id();
  for (const int scan_col_id : scan_plan->get_col_list()) {
    auto cd = get_column_descriptor(scan_col_id, table_id, cat);
    if (cd->isVirtualCol) {
      CHECK_EQ("rowid", cd->columnName);
    } else {
      input_col_descs.push_back(std::make_shared<const InputColDescriptor>(scan_col_id, table_id, scan_idx));
    }
  }
}

void collect_input_descs(std::vector<InputDescriptor>& input_descs,
                         std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs,
                         const Planner::Plan* plan,
                         const Catalog_Namespace::Catalog& cat) {
  const auto scan_plan = get_scan_child(plan);
  const auto join_plan = get_join_child(plan);
  const Planner::Scan* outer_plan{nullptr};
  const Planner::Scan* inner_plan{nullptr};
  if (join_plan) {
    outer_plan = get_scan_child(join_plan->get_outerplan());
    CHECK(outer_plan);
    inner_plan = get_scan_child(join_plan->get_innerplan());
    CHECK(inner_plan);
    input_descs.emplace_back(outer_plan->get_table_id(), 0);
    input_descs.emplace_back(inner_plan->get_table_id(), 1);
    collect_input_col_descs(input_col_descs, outer_plan, cat, true, 0);
    collect_input_col_descs(input_col_descs, inner_plan, cat, true, 1);
  } else {
    CHECK(scan_plan);
    input_descs.emplace_back(scan_plan->get_table_id(), 0);
    collect_input_col_descs(input_col_descs, scan_plan, cat, false, 0);
  }
}

}  // namespace

RowSetPtr Executor::executeSelectPlan(const Planner::Plan* plan,
                                      const int64_t limit,
                                      const int64_t offset,
                                      const bool hoist_literals,
                                      const ExecutorDeviceType device_type,
                                      const ExecutorOptLevel opt_level,
                                      const Catalog_Namespace::Catalog& cat,
                                      size_t& max_groups_buffer_entry_guess,
                                      int32_t* error_code,
                                      const Planner::Sort* sort_plan_in,
                                      const bool allow_multifrag,
                                      const bool just_explain,
                                      const bool allow_loop_joins,
                                      RenderInfo* render_info) {
  if (dynamic_cast<const Planner::Scan*>(plan) || dynamic_cast<const Planner::AggPlan*>(plan) ||
      dynamic_cast<const Planner::Join*>(plan)) {
    row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
    lit_str_dict_proxy_ = nullptr;
    const auto target_exprs = get_agg_target_exprs(plan);
    const auto scan_plan = get_scan_child(plan);
    auto simple_quals = scan_plan ? scan_plan->get_simple_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
    auto quals = scan_plan ? scan_plan->get_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
    const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
    auto groupby_exprs = agg_plan ? agg_plan->get_groupby_list() : std::list<std::shared_ptr<Analyzer::Expr>>{nullptr};
    std::vector<InputDescriptor> input_descs;
    std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
    collect_input_descs(input_descs, input_col_descs, plan, cat);
    const auto join_plan = get_join_child(plan);
    if (join_plan) {
      collect_quals_from_join(simple_quals, quals, join_plan);
    }
    const auto join_quals = join_plan ? join_plan->get_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
    CHECK(check_plan_sanity(plan));
    const bool is_agg = dynamic_cast<const Planner::AggPlan*>(plan);
    const auto order_entries = sort_plan_in ? sort_plan_in->get_order_entries() : std::list<Analyzer::OrderEntry>{};
    const auto query_infos = get_table_infos(input_descs, this);
    const size_t scan_limit = get_scan_limit(plan, limit);
    const size_t scan_total_limit = scan_limit ? get_scan_limit(plan, scan_limit + offset) : 0;
    const auto ra_exe_unit_in = RelAlgExecutionUnit{
        input_descs,
        {},
        input_col_descs,
        simple_quals,
        quals,
        JoinType::INVALID,
        {},
        {},
        join_quals,
        {},
        groupby_exprs,
        target_exprs,
        {},
        nullptr,
        {order_entries, SortAlgorithm::Default, static_cast<size_t>(limit), static_cast<size_t>(offset)},
        scan_total_limit};
    QueryRewriter query_rewriter(ra_exe_unit_in, query_infos, this, agg_plan);
    const auto ra_exe_unit = query_rewriter.rewrite();
    if (limit || offset) {
      size_t max_groups_buffer_entry_guess_limit{scan_total_limit ? scan_total_limit : max_groups_buffer_entry_guess};
      auto result = executeWorkUnit(error_code,
                                    max_groups_buffer_entry_guess_limit,
                                    is_agg,
                                    query_infos,
                                    ra_exe_unit,
                                    {device_type, hoist_literals, opt_level, g_enable_dynamic_watchdog},
                                    {false,
                                     allow_multifrag,
                                     just_explain,
                                     allow_loop_joins,
                                     g_enable_watchdog,
                                     false,
                                     false,
                                     g_enable_dynamic_watchdog,
                                     g_dynamic_watchdog_time_limit},
                                    cat,
                                    row_set_mem_owner_,
                                    render_info,
                                    true);
      auto& rows = boost::get<RowSetPtr>(result);
      max_groups_buffer_entry_guess = max_groups_buffer_entry_guess_limit;
      CHECK(rows);
      rows->dropFirstN(offset);
      if (limit) {
        rows->keepFirstN(limit);
      }
      return std::move(rows);
    }
    auto result = executeWorkUnit(error_code,
                                  max_groups_buffer_entry_guess,
                                  is_agg,
                                  query_infos,
                                  ra_exe_unit,
                                  {device_type, hoist_literals, opt_level, g_enable_dynamic_watchdog},
                                  {false,
                                   allow_multifrag,
                                   just_explain,
                                   allow_loop_joins,
                                   g_enable_watchdog,
                                   false,
                                   false,
                                   g_enable_dynamic_watchdog,
                                   g_dynamic_watchdog_time_limit},
                                  cat,
                                  row_set_mem_owner_,
                                  render_info,
                                  true);
    auto& rows = boost::get<RowSetPtr>(result);
    CHECK(rows);
    return std::move(rows);
  }
  const auto result_plan = dynamic_cast<const Planner::Result*>(plan);
  if (result_plan) {
    if (limit || offset) {
      auto rows = executeResultPlan(result_plan,
                                    hoist_literals,
                                    device_type,
                                    opt_level,
                                    cat,
                                    max_groups_buffer_entry_guess,
                                    error_code,
                                    sort_plan_in,
                                    allow_multifrag,
                                    just_explain,
                                    allow_loop_joins);
      CHECK(rows);
      rows->dropFirstN(offset);
      if (limit) {
        rows->keepFirstN(limit);
      }
      return rows;
    }
    return executeResultPlan(result_plan,
                             hoist_literals,
                             device_type,
                             opt_level,
                             cat,
                             max_groups_buffer_entry_guess,
                             error_code,
                             sort_plan_in,
                             allow_multifrag,
                             just_explain,
                             allow_loop_joins);
  }
  const auto sort_plan = dynamic_cast<const Planner::Sort*>(plan);
  if (sort_plan) {
    return executeSortPlan(sort_plan,
                           limit,
                           offset,
                           hoist_literals,
                           device_type,
                           opt_level,
                           cat,
                           max_groups_buffer_entry_guess,
                           error_code,
                           allow_multifrag,
                           just_explain,
                           allow_loop_joins);
  }
  abort();
}

RowSetPtr Executor::executeResultPlan(const Planner::Result* result_plan,
                                      const bool hoist_literals,
                                      const ExecutorDeviceType device_type,
                                      const ExecutorOptLevel opt_level,
                                      const Catalog_Namespace::Catalog& cat,
                                      size_t& max_groups_buffer_entry_guess,
                                      int32_t* error_code,
                                      const Planner::Sort* sort_plan,
                                      const bool allow_multifrag,
                                      const bool just_explain,
                                      const bool allow_loop_joins) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(result_plan->get_child_plan());
  if (!agg_plan) {  // TODO(alex)
    throw std::runtime_error("Query not supported yet, child plan needs to be an aggregate plan.");
  }
  row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  lit_str_dict_proxy_ = nullptr;
  const auto scan_plan = dynamic_cast<const Planner::Scan*>(agg_plan->get_child_plan());
  auto simple_quals = scan_plan ? scan_plan->get_simple_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
  auto quals = scan_plan ? scan_plan->get_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  collect_input_descs(input_descs, input_col_descs, agg_plan, cat);
  const auto join_plan = get_join_child(agg_plan);
  if (join_plan) {
    collect_quals_from_join(simple_quals, quals, join_plan);
  }
  const auto join_quals = join_plan ? join_plan->get_quals() : std::list<std::shared_ptr<Analyzer::Expr>>{};
  CHECK(check_plan_sanity(agg_plan));
  const auto query_infos = get_table_infos(input_descs, this);
  const auto ra_exe_unit_in = RelAlgExecutionUnit{input_descs,
                                                  {},
                                                  input_col_descs,
                                                  simple_quals,
                                                  quals,
                                                  JoinType::INVALID,
                                                  {},
                                                  {},
                                                  join_quals,
                                                  {},
                                                  agg_plan->get_groupby_list(),
                                                  get_agg_target_exprs(agg_plan),
                                                  {},
                                                  nullptr,
                                                  {{}, SortAlgorithm::Default, 0, 0},
                                                  0};
  QueryRewriter query_rewriter(ra_exe_unit_in, query_infos, this, result_plan);
  const auto ra_exe_unit = query_rewriter.rewrite();
  auto result = executeWorkUnit(error_code,
                                max_groups_buffer_entry_guess,
                                true,
                                query_infos,
                                ra_exe_unit,
                                {device_type, hoist_literals, opt_level, g_enable_dynamic_watchdog},
                                {false,
                                 allow_multifrag,
                                 just_explain,
                                 allow_loop_joins,
                                 g_enable_watchdog,
                                 false,
                                 false,
                                 g_enable_dynamic_watchdog,
                                 g_dynamic_watchdog_time_limit},
                                cat,
                                row_set_mem_owner_,
                                nullptr,
                                true);
  auto& rows = boost::get<RowSetPtr>(result);
  CHECK(rows);
  if (just_explain) {
    return std::move(rows);
  }

  const int in_col_count{static_cast<int>(agg_plan->get_targetlist().size())};
  std::list<std::shared_ptr<const InputColDescriptor>> pseudo_input_col_descs;
  for (int pseudo_col = 1; pseudo_col <= in_col_count; ++pseudo_col) {
    pseudo_input_col_descs.push_back(std::make_shared<const InputColDescriptor>(pseudo_col, 0, -1));
  }
  const auto order_entries = sort_plan ? sort_plan->get_order_entries() : std::list<Analyzer::OrderEntry>{};
  const RelAlgExecutionUnit res_ra_unit{{},
                                        {},
                                        pseudo_input_col_descs,
                                        result_plan->get_constquals(),
                                        result_plan->get_quals(),
                                        JoinType::INVALID,
                                        {},
                                        {},
                                        {},
                                        {},
                                        {nullptr},
                                        get_agg_target_exprs(result_plan),
                                        {},
                                        nullptr,
                                        {
                                            order_entries, SortAlgorithm::Default, 0, 0,
                                        },
                                        0};
  if (*error_code) {
    return std::make_shared<ResultSet>(
        std::vector<TargetInfo>{}, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, this);
  }
  const auto& targets = result_plan->get_targetlist();
  CHECK(!targets.empty());
  std::vector<AggInfo> agg_infos;
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    const auto target_entry = targets[target_idx];
    const auto target_type = target_entry->get_expr()->get_type_info().get_type();
    agg_infos.emplace_back((target_type == kFLOAT || target_type == kDOUBLE) ? "agg_id_double" : "agg_id",
                           target_entry->get_expr(),
                           0,
                           target_idx);
  }
  std::vector<SQLTypeInfo> target_types;
  for (auto in_col : agg_plan->get_targetlist()) {
    target_types.push_back(in_col->get_expr()->get_type_info());
  }
  CHECK(rows);
  ColumnarResults result_columns(row_set_mem_owner_, *rows, in_col_count, target_types);
  std::vector<llvm::Value*> col_heads;
  // Nested query, let the compiler know
  ResetIsNested reset_is_nested(this);
  is_nested_ = true;
  std::vector<Analyzer::Expr*> target_exprs;
  for (auto target_entry : targets) {
    target_exprs.emplace_back(target_entry->get_expr());
  }
  const auto row_count = rows->rowCount();
  if (!row_count) {
    return std::make_shared<ResultSet>(
        std::vector<TargetInfo>{}, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, this);
  }
  std::vector<ColWidths> agg_col_widths;
  for (auto wid : get_col_byte_widths(target_exprs, {})) {
    agg_col_widths.push_back(
        {wid, int8_t(compact_byte_width(wid, pick_target_compact_width(res_ra_unit, {}, get_min_byte_width())))});
  }
  QueryMemoryDescriptor query_mem_desc{this,
                                       allow_multifrag,
                                       GroupByColRangeType::Projection,
                                       false,
                                       false,
                                       -1,
                                       0,
                                       {sizeof(int64_t)},
#ifdef ENABLE_KEY_COMPACTION
                                       0,
#endif
                                       agg_col_widths,
                                       {},
                                       row_count,
                                       small_groups_buffer_entry_count_,
                                       0,
                                       0,
                                       0,
                                       false,
                                       GroupByMemSharing::Shared,
                                       CountDistinctDescriptors{},
                                       false,
                                       true,
                                       false,
                                       false,
                                       {},
                                       {},
                                       false};
  ColumnCacheMap column_cache;
  OOM_TRACE_PUSH();
  auto compilation_result =
      compileWorkUnit({},
                      res_ra_unit,
                      {ExecutorDeviceType::CPU, hoist_literals, opt_level, g_enable_dynamic_watchdog},
                      {false,
                       allow_multifrag,
                       just_explain,
                       allow_loop_joins,
                       g_enable_watchdog,
                       false,
                       false,
                       g_enable_dynamic_watchdog,
                       g_dynamic_watchdog_time_limit},
                      nullptr,
                      false,
                      row_set_mem_owner_,
                      row_count,
                      small_groups_buffer_entry_count_,
                      get_min_byte_width(),
                      JoinInfo(JoinImplType::Invalid, std::vector<std::shared_ptr<Analyzer::BinOper>>{}, {}, ""),
                      false,
                      column_cache);
  auto column_buffers = result_columns.getColumnBuffers();
  CHECK_EQ(column_buffers.size(), static_cast<size_t>(in_col_count));
  std::vector<int64_t> init_agg_vals(query_mem_desc.agg_col_widths.size());
  auto query_exe_context = query_mem_desc.getQueryExecutionContext(res_ra_unit,
                                                                   init_agg_vals,
                                                                   this,
                                                                   ExecutorDeviceType::CPU,
                                                                   0,
                                                                   {},
                                                                   {},
                                                                   {},
                                                                   row_set_mem_owner_,
                                                                   false,
                                                                   false,
                                                                   nullptr);
  const auto hoist_buf = serializeLiterals(compilation_result.literal_values, 0);
  *error_code = 0;
  std::vector<std::vector<const int8_t*>> multi_frag_col_buffers{column_buffers};
  query_exe_context->launchCpuCode(res_ra_unit,
                                   compilation_result.native_functions,
                                   hoist_literals,
                                   hoist_buf,
                                   multi_frag_col_buffers,
                                   {{static_cast<int64_t>(result_columns.size())}},
                                   {{0}},
                                   1u,
                                   0,
                                   init_agg_vals,
                                   error_code,
                                   1,
                                   {});
  CHECK_GE(*error_code, 0);
  return query_exe_context->groupBufferToResults(0, target_exprs, false);
}

RowSetPtr Executor::executeSortPlan(const Planner::Sort* sort_plan,
                                    const int64_t limit,
                                    const int64_t offset,
                                    const bool hoist_literals,
                                    const ExecutorDeviceType device_type,
                                    const ExecutorOptLevel opt_level,
                                    const Catalog_Namespace::Catalog& cat,
                                    size_t& max_groups_buffer_entry_guess,
                                    int32_t* error_code,
                                    const bool allow_multifrag,
                                    const bool just_explain,
                                    const bool allow_loop_joins) {
  *error_code = 0;
  auto rows_to_sort = executeSelectPlan(sort_plan->get_child_plan(),
                                        0,
                                        0,
                                        hoist_literals,
                                        device_type,
                                        opt_level,
                                        cat,
                                        max_groups_buffer_entry_guess,
                                        error_code,
                                        sort_plan,
                                        allow_multifrag,
                                        just_explain,
                                        allow_loop_joins,
                                        nullptr);
  CHECK(rows_to_sort);
  if (just_explain || *error_code == ERR_OUT_OF_GPU_MEM || *error_code == ERR_OUT_OF_TIME ||
      *error_code == ERR_INTERRUPTED) {
    return rows_to_sort;
  }
  rows_to_sort->sort(sort_plan->get_order_entries(), limit + offset);
  if (limit || offset) {
    rows_to_sort->dropFirstN(offset);
    if (limit) {
      rows_to_sort->keepFirstN(limit);
    }
  }
  return rows_to_sort;
}

/*
 * x64 benchmark: "SELECT COUNT(*) FROM test WHERE x > 41;"
 *                x = 42, 64-bit column, 1-byte encoding
 *                3B rows in 1.2s on a i7-4870HQ core
 *
 * TODO(alex): check we haven't introduced a regression with the new translator.
 */

std::shared_ptr<ResultSet> Executor::execute(const Planner::RootPlan* root_plan,
                                             const Catalog_Namespace::SessionInfo& session,
                                             const bool hoist_literals,
                                             const ExecutorDeviceType device_type,
                                             const ExecutorOptLevel opt_level,
                                             const bool allow_multifrag,
                                             const bool allow_loop_joins,
                                             RenderInfo* render_info) {
  catalog_ = &root_plan->get_catalog();
  const auto stmt_type = root_plan->get_stmt_type();
  // capture the lock acquistion time
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> lock(execute_mutex_);
  if (g_enable_dynamic_watchdog) {
    resetInterrupt();
  }
  ScopeGuard restore_metainfo_cache = [this] { clearMetaInfoCache(); };
  int64_t queue_time_ms = timer_stop(clock_begin);
  ScopeGuard row_set_holder = [this] { row_set_mem_owner_ = nullptr; };
  switch (stmt_type) {
    case kSELECT: {
      int32_t error_code{0};
      size_t max_groups_buffer_entry_guess{16384};

      std::unique_ptr<RenderInfo> render_info_ptr;
      if (root_plan->get_plan_dest() == Planner::RootPlan::kRENDER) {
        CHECK(render_info);

        render_info->setInSituDataIfUnset(device_type == ExecutorDeviceType::GPU);

        if (!render_manager_) {
          throw std::runtime_error("This build doesn't support backend rendering");
        }

        if (!render_info->render_allocator_map_ptr) {
          // make backwards compatible, can be removed when MapDHandler::render(...)
          // in MapDServer.cpp is removed
          render_info->render_allocator_map_ptr.reset(new RenderAllocatorMap(render_manager_, blockSize(), gridSize()));
        }
      }
      auto rows = executeSelectPlan(root_plan->get_plan(),
                                    root_plan->get_limit(),
                                    root_plan->get_offset(),
                                    hoist_literals,
                                    device_type,
                                    opt_level,
                                    root_plan->get_catalog(),
                                    max_groups_buffer_entry_guess,
                                    &error_code,
                                    nullptr,
                                    allow_multifrag,
                                    root_plan->get_plan_dest() == Planner::RootPlan::kEXPLAIN,
                                    allow_loop_joins,
                                    render_info);
      if (error_code == ERR_OUT_OF_RENDER_MEM) {
        CHECK_EQ(Planner::RootPlan::kRENDER, root_plan->get_plan_dest());
        throw std::runtime_error("Not enough OpenGL memory to render the query results");
      }

      if (render_info && render_info->hasInSituData() && render_info->render_allocator_map_ptr) {
        if (error_code == ERR_OUT_OF_GPU_MEM) {
          throw std::runtime_error("Not enough GPU memory to execute the query");
        }
        if (error_code && !root_plan->get_limit()) {
          CHECK_LT(error_code, 0);
          throw std::runtime_error("Ran out of slots in the output buffer");
        }
        auto clock_begin = timer_start();
        render_info->targets = root_plan->get_plan()->get_targetlist();
        auto rrows = renderRows(render_info);
        int64_t render_time_ms = timer_stop(clock_begin);
        return std::make_shared<ResultSet>(rrows, queue_time_ms, render_time_ms, row_set_mem_owner_);
      }

      if (error_code == ERR_OVERFLOW_OR_UNDERFLOW) {
        throw std::runtime_error("Overflow or underflow");
      }
      if (error_code == ERR_DIV_BY_ZERO) {
        throw std::runtime_error("Division by zero");
      }
      if (error_code == ERR_UNSUPPORTED_SELF_JOIN) {
        throw std::runtime_error("Self joins not supported yet");
      }
      if (error_code == ERR_OUT_OF_TIME) {
        if (!interrupted_)
          throw std::runtime_error("Query execution has exceeded the time limit");
        error_code = ERR_INTERRUPTED;
      }
      if (error_code == ERR_INTERRUPTED) {
        throw std::runtime_error("Query execution has been interrupted");
      }
      if (error_code == ERR_OUT_OF_CPU_MEM) {
        throw std::runtime_error("Not enough host memory to execute the query");
      }
      if (error_code == ERR_OUT_OF_GPU_MEM) {
        rows = executeSelectPlan(root_plan->get_plan(),
                                 root_plan->get_limit(),
                                 root_plan->get_offset(),
                                 hoist_literals,
                                 device_type,
                                 opt_level,
                                 root_plan->get_catalog(),
                                 max_groups_buffer_entry_guess,
                                 &error_code,
                                 nullptr,
                                 false,
                                 false,
                                 allow_loop_joins,
                                 nullptr);
      }
      if (error_code) {
        max_groups_buffer_entry_guess = 0;
        while (true) {
          rows = executeSelectPlan(root_plan->get_plan(),
                                   root_plan->get_limit(),
                                   root_plan->get_offset(),
                                   hoist_literals,
                                   ExecutorDeviceType::CPU,
                                   opt_level,
                                   root_plan->get_catalog(),
                                   max_groups_buffer_entry_guess,
                                   &error_code,
                                   nullptr,
                                   false,
                                   false,
                                   allow_loop_joins,
                                   nullptr);
          CHECK(rows);
          if (!error_code) {
            rows->setQueueTime(queue_time_ms);
            return rows;
          }
          // Even the conservative guess failed; it should only happen when we group
          // by a huge cardinality array. Maybe we should throw an exception instead?
          // Such a heavy query is entirely capable of exhausting all the host memory.
          CHECK(max_groups_buffer_entry_guess);
          max_groups_buffer_entry_guess *= 2;
        }
      }
      CHECK(rows);
      rows->setQueueTime(queue_time_ms);
      return rows;
    }
    case kINSERT: {
      if (root_plan->get_plan_dest() == Planner::RootPlan::kEXPLAIN) {
        auto explanation_rs = std::make_shared<ResultSet>("No explanation available.");
        explanation_rs->setQueueTime(queue_time_ms);
        return explanation_rs;
      }
      auto& cat = session.get_catalog();
      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      auto user_metadata = session.get_currentUser();
      const int table_id = root_plan->get_result_table_id();
      auto td = cat.getMetadataForTable(table_id);
      DBObject dbObject(td->tableName, TableDBObjectType);
      dbObject.loadKey(cat);
      dbObject.setPrivileges(AccessPrivileges::INSERT);
      std::vector<DBObject> privObjects;
      privObjects.push_back(dbObject);
      if (Catalog_Namespace::SysCatalog::instance().arePrivilegesOn() &&
          !sys_cat.checkPrivileges(user_metadata, privObjects)) {
        throw std::runtime_error("Violation of access privileges: user " + user_metadata.userName +
                                 " has no insert privileges for table " + td->tableName + ".");
        break;
      }
      executeSimpleInsert(root_plan);
      auto empty_rs = std::make_shared<ResultSet>(
          std::vector<TargetInfo>{}, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, this);
      empty_rs->setQueueTime(queue_time_ms);
      return empty_rs;
    }
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}
