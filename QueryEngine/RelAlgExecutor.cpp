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

#include "RelAlgExecutor.h"

#include <algorithm>
#include <boost/range/adaptor/reversed.hpp>
#include <numeric>

#include "Parser/ParserNode.h"
#include "QueryEngine/CalciteDeserializerUtils.h"
#include "QueryEngine/CardinalityEstimator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/ExternalExecutor.h"
#include "QueryEngine/FromTableReordering.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/JoinFilterPushDown.h"
#include "QueryEngine/QueryPhysicalInputsCollector.h"
#include "QueryEngine/RangeTableIndexVisitor.h"
#include "QueryEngine/RelAlgDagBuilder.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "QueryEngine/RexVisitor.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "QueryEngine/UsedColumnsVisitor.h"
#include "QueryEngine/WindowContext.h"
#include "Shared/TypedDataAccessors.h"
#include "Shared/measure.h"
#include "Shared/shard_key.h"

bool g_skip_intermediate_count{true};
extern bool g_enable_bump_allocator;
bool g_enable_interop{false};

namespace {

bool node_is_aggregate(const RelAlgNode* ra) {
  const auto compound = dynamic_cast<const RelCompound*>(ra);
  const auto aggregate = dynamic_cast<const RelAggregate*>(ra);
  return ((compound && compound->isAggregate()) || aggregate);
}

std::unordered_set<PhysicalInput> get_physical_inputs(
    const Catalog_Namespace::Catalog& cat,
    const RelAlgNode* ra) {
  auto phys_inputs = get_physical_inputs(ra);
  std::unordered_set<PhysicalInput> phys_inputs2;
  for (auto& phi : phys_inputs) {
    phys_inputs2.insert(
        PhysicalInput{cat.getColumnIdBySpi(phi.table_id, phi.col_id), phi.table_id});
  }
  return phys_inputs2;
}

}  // namespace

size_t RelAlgExecutor::getOuterFragmentCount(const CompilationOptions& co,
                                             const ExecutionOptions& eo) {
  if (eo.find_push_down_candidates) {
    return 0;
  }

  if (eo.just_explain) {
    return 0;
  }

  CHECK(query_dag_);

  query_dag_->resetQueryExecutionState();
  const auto& ra = query_dag_->getRootNode();

  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);
  ScopeGuard row_set_holder = [this] { cleanupPostExecution(); };
  const auto phys_inputs = get_physical_inputs(cat_, &ra);
  const auto phys_table_ids = get_physical_table_inputs(&ra);
  executor_->setCatalog(&cat_);
  executor_->setupCaching(phys_inputs, phys_table_ids);

  ScopeGuard restore_metainfo_cache = [this] { executor_->clearMetaInfoCache(); };
  auto ed_seq = RaExecutionSequence(&ra);

  if (!getSubqueries().empty()) {
    return 0;
  }

  CHECK(!ed_seq.empty());
  if (ed_seq.size() > 1) {
    return 0;
  }

  decltype(temporary_tables_)().swap(temporary_tables_);
  decltype(target_exprs_owned_)().swap(target_exprs_owned_);
  executor_->catalog_ = &cat_;
  executor_->temporary_tables_ = &temporary_tables_;

  WindowProjectNodeContext::reset();
  auto exec_desc_ptr = ed_seq.getDescriptor(0);
  CHECK(exec_desc_ptr);
  auto& exec_desc = *exec_desc_ptr;
  const auto body = exec_desc.getBody();
  if (body->isNop()) {
    return 0;
  }

  const auto project = dynamic_cast<const RelProject*>(body);
  if (project) {
    auto work_unit =
        createProjectWorkUnit(project, {{}, SortAlgorithm::Default, 0, 0}, eo);

    return get_frag_count_of_table(work_unit.exe_unit.input_descs[0].getTableId(),
                                   executor_);
  }

  const auto compound = dynamic_cast<const RelCompound*>(body);
  if (compound) {
    if (compound->isDeleteViaSelect()) {
      return 0;
    } else if (compound->isUpdateViaSelect()) {
      return 0;
    } else {
      if (compound->isAggregate()) {
        return 0;
      }

      const auto work_unit =
          createCompoundWorkUnit(compound, {{}, SortAlgorithm::Default, 0, 0}, eo);

      return get_frag_count_of_table(work_unit.exe_unit.input_descs[0].getTableId(),
                                     executor_);
    }
  }

  return 0;
}

ExecutionResult RelAlgExecutor::executeRelAlgQuery(const CompilationOptions& co,
                                                   const ExecutionOptions& eo,
                                                   const bool just_explain_plan,
                                                   RenderInfo* render_info) {
  CHECK(query_dag_);
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(executeRelAlgQuery);
  try {
    return executeRelAlgQueryNoRetry(co, eo, just_explain_plan, render_info);
  } catch (const QueryMustRunOnCpu&) {
    if (!g_allow_cpu_retry) {
      throw;
    }
  }
  LOG(INFO) << "Query unable to run in GPU mode, retrying on CPU";
  auto co_cpu = CompilationOptions::makeCpuOnly(co);

  if (render_info) {
    render_info->setForceNonInSituData();
  }
  return executeRelAlgQueryNoRetry(co_cpu, eo, just_explain_plan, render_info);
}

ExecutionResult RelAlgExecutor::executeRelAlgQueryNoRetry(const CompilationOptions& co,
                                                          const ExecutionOptions& eo,
                                                          const bool just_explain_plan,
                                                          RenderInfo* render_info) {
  INJECT_TIMER(executeRelAlgQueryNoRetry);
  auto timer = DEBUG_TIMER(__func__);
  auto timer_setup = DEBUG_TIMER("Query pre-execution steps");

  query_dag_->resetQueryExecutionState();
  const auto& ra = query_dag_->getRootNode();

  // capture the lock acquistion time
  auto clock_begin = timer_start();
  if (g_enable_dynamic_watchdog) {
    executor_->resetInterrupt();
  }

  std::string query_session = "";
  if (g_enable_runtime_query_interrupt) {
    // a request of query execution without session id can happen, i.e., test query
    // if so, we turn back to the original way: a runtime query interrupt
    // without per-session management (as similar to dynamic watchdog)
    if (query_state_ != nullptr && query_state_->getConstSessionInfo() != nullptr) {
      query_session = query_state_->getConstSessionInfo()->get_session_id();
    } else if (executor_->getCurrentQuerySession() != query_session) {
      query_session = executor_->getCurrentQuerySession();
    }
    if (query_session != "") {
      // if session is valid, then we allow per-session runtime query interrupt
      executor_->addToQuerySessionList(query_session);
      // hybrid spinlock.  if it fails to acquire a lock, then
      // it sleeps {g_runtime_query_interrupt_frequency} millisecond.
      while (executor_->execute_spin_lock_.test_and_set(std::memory_order_acquire)) {
        // failed to get the spinlock: check whether query is interrupted
        if (executor_->checkIsQuerySessionInterrupted(query_session)) {
          executor_->removeFromQuerySessionList(query_session);
          VLOG(1) << "Kill the Interrupted pending query.";
          throw std::runtime_error(
              "Query execution has been interrupted (pending query).");
        }
        // here it fails to acquire the lock
        std::this_thread::sleep_for(
            std::chrono::milliseconds(g_runtime_query_interrupt_frequency));
      };
    }
    // currently, atomic_flag does not provide a way to get its current status,
    // i.e., spinlock.is_locked(), so we additionally lock the execute_mutex_
    // right after acquiring spinlock to let other part of the code can know
    // whether there exists a running query on the executor
  }
  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);

  ScopeGuard clearRuntimeInterruptStatus = [this] {
    // reset the runtime query interrupt status
    if (g_enable_runtime_query_interrupt) {
      executor_->removeFromQuerySessionList(executor_->getCurrentQuerySession());
      executor_->invalidateQuerySession();
      executor_->resetInterrupt();
      executor_->execute_spin_lock_.clear(std::memory_order_acquire);
      VLOG(1) << "RESET runtime query interrupt status of Executor " << this;
    }
  };

  if (g_enable_runtime_query_interrupt) {
    // make sure to set the running session ID
    executor_->invalidateQuerySession();
    executor_->setCurrentQuerySession(query_session);
  }

  int64_t queue_time_ms = timer_stop(clock_begin);
  ScopeGuard row_set_holder = [this] { cleanupPostExecution(); };
  const auto phys_inputs = get_physical_inputs(cat_, &ra);
  const auto phys_table_ids = get_physical_table_inputs(&ra);
  executor_->setCatalog(&cat_);
  executor_->setupCaching(phys_inputs, phys_table_ids);

  ScopeGuard restore_metainfo_cache = [this] { executor_->clearMetaInfoCache(); };
  auto ed_seq = RaExecutionSequence(&ra);

  if (just_explain_plan) {
    std::stringstream ss;
    std::vector<const RelAlgNode*> nodes;
    for (size_t i = 0; i < ed_seq.size(); i++) {
      nodes.emplace_back(ed_seq.getDescriptor(i)->getBody());
    }
    size_t ctr = nodes.size();
    size_t tab_ctr = 0;
    for (auto& body : boost::adaptors::reverse(nodes)) {
      const auto index = ctr--;
      const auto tabs = std::string(tab_ctr++, '\t');
      CHECK(body);
      ss << tabs << std::to_string(index) << " : " << body->toString() << "\n";
      if (auto sort = dynamic_cast<const RelSort*>(body)) {
        ss << tabs << "  : " << sort->getInput(0)->toString() << "\n";
      }
    }
    auto rs = std::make_shared<ResultSet>(ss.str());
    return {rs, {}};
  }

  if (render_info) {
    // set render to be non-insitu in certain situations.
    if (!render_info->disallow_in_situ_only_if_final_ED_is_aggregate &&
        ed_seq.size() > 1) {
      // old logic
      // disallow if more than one ED
      render_info->setInSituDataIfUnset(false);
    }
  }

  if (eo.find_push_down_candidates) {
    // this extra logic is mainly due to current limitations on multi-step queries
    // and/or subqueries.
    return executeRelAlgQueryWithFilterPushDown(
        ed_seq, co, eo, render_info, queue_time_ms);
  }
  timer_setup.stop();

  // Dispatch the subqueries first
  for (auto subquery : getSubqueries()) {
    const auto subquery_ra = subquery->getRelAlg();
    CHECK(subquery_ra);
    if (subquery_ra->hasContextData()) {
      continue;
    }
    // Execute the subquery and cache the result.
    RelAlgExecutor ra_executor(executor_, cat_, query_state_);
    RaExecutionSequence subquery_seq(subquery_ra);
    auto result = ra_executor.executeRelAlgSeq(subquery_seq, co, eo, nullptr, 0);
    subquery->setExecutionResult(std::make_shared<ExecutionResult>(result));
  }
  return executeRelAlgSeq(ed_seq, co, eo, render_info, queue_time_ms);
}

AggregatedColRange RelAlgExecutor::computeColRangesCache() {
  AggregatedColRange agg_col_range_cache;
  const auto phys_inputs = get_physical_inputs(cat_, &getRootRelAlgNode());
  return executor_->computeColRangesCache(phys_inputs);
}

StringDictionaryGenerations RelAlgExecutor::computeStringDictionaryGenerations() {
  const auto phys_inputs = get_physical_inputs(cat_, &getRootRelAlgNode());
  return executor_->computeStringDictionaryGenerations(phys_inputs);
}

TableGenerations RelAlgExecutor::computeTableGenerations() {
  const auto phys_table_ids = get_physical_table_inputs(&getRootRelAlgNode());
  return executor_->computeTableGenerations(phys_table_ids);
}

Executor* RelAlgExecutor::getExecutor() const {
  return executor_;
}

void RelAlgExecutor::cleanupPostExecution() {
  CHECK(executor_);
  executor_->row_set_mem_owner_ = nullptr;
  executor_->lit_str_dict_proxy_ = nullptr;
}

namespace {

inline void check_sort_node_source_constraint(const RelSort* sort) {
  CHECK_EQ(size_t(1), sort->inputCount());
  const auto source = sort->getInput(0);
  if (dynamic_cast<const RelSort*>(source)) {
    throw std::runtime_error("Sort node not supported as input to another sort");
  }
}

}  // namespace

FirstStepExecutionResult RelAlgExecutor::executeRelAlgQuerySingleStep(
    const RaExecutionSequence& seq,
    const size_t step_idx,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    RenderInfo* render_info) {
  INJECT_TIMER(executeRelAlgQueryStep);
  auto exe_desc_ptr = seq.getDescriptor(step_idx);
  CHECK(exe_desc_ptr);
  const auto sort = dynamic_cast<const RelSort*>(exe_desc_ptr->getBody());

  size_t shard_count{0};
  auto merge_type = [&shard_count](const RelAlgNode* body) -> MergeType {
    return node_is_aggregate(body) && !shard_count ? MergeType::Reduce : MergeType::Union;
  };

  if (sort) {
    check_sort_node_source_constraint(sort);
    const auto source_work_unit = createSortInputWorkUnit(sort, eo);
    shard_count = GroupByAndAggregate::shard_count_for_top_groups(
        source_work_unit.exe_unit, *executor_->getCatalog());
    if (!shard_count) {
      // No point in sorting on the leaf, only execute the input to the sort node.
      CHECK_EQ(size_t(1), sort->inputCount());
      const auto source = sort->getInput(0);
      if (sort->collationCount() || node_is_aggregate(source)) {
        auto temp_seq = RaExecutionSequence(std::make_unique<RaExecutionDesc>(source));
        CHECK_EQ(temp_seq.size(), size_t(1));
        // Use subseq to avoid clearing existing temporary tables
        return {executeRelAlgSubSeq(temp_seq, std::make_pair(0, 1), co, eo, nullptr, 0),
                merge_type(source),
                source->getId(),
                false};
      }
    }
  }
  return {executeRelAlgSubSeq(seq,
                              std::make_pair(step_idx, step_idx + 1),
                              co,
                              eo,
                              render_info,
                              queue_time_ms_),
          merge_type(exe_desc_ptr->getBody()),
          exe_desc_ptr->getBody()->getId(),
          false};
}

void RelAlgExecutor::prepareLeafExecution(
    const AggregatedColRange& agg_col_range,
    const StringDictionaryGenerations& string_dictionary_generations,
    const TableGenerations& table_generations) {
  // capture the lock acquistion time
  auto clock_begin = timer_start();
  if (g_enable_dynamic_watchdog) {
    executor_->resetInterrupt();
  }
  queue_time_ms_ = timer_stop(clock_begin);
  executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  executor_->table_generations_ = table_generations;
  executor_->agg_col_range_cache_ = agg_col_range;
  executor_->string_dictionary_generations_ = string_dictionary_generations;
}

ExecutionResult RelAlgExecutor::executeRelAlgSeq(const RaExecutionSequence& seq,
                                                 const CompilationOptions& co,
                                                 const ExecutionOptions& eo,
                                                 RenderInfo* render_info,
                                                 const int64_t queue_time_ms,
                                                 const bool with_existing_temp_tables) {
  INJECT_TIMER(executeRelAlgSeq);
  auto timer = DEBUG_TIMER(__func__);
  if (!with_existing_temp_tables) {
    decltype(temporary_tables_)().swap(temporary_tables_);
  }
  decltype(target_exprs_owned_)().swap(target_exprs_owned_);
  executor_->catalog_ = &cat_;
  executor_->temporary_tables_ = &temporary_tables_;

  time(&now_);
  CHECK(!seq.empty());
  const auto exec_desc_count = eo.just_explain ? size_t(1) : seq.size();

  for (size_t i = 0; i < exec_desc_count; i++) {
    // only render on the last step
    try {
      executeRelAlgStep(seq,
                        i,
                        co,
                        eo,
                        (i == exec_desc_count - 1) ? render_info : nullptr,
                        queue_time_ms);
    } catch (const NativeExecutionError&) {
      if (!g_enable_interop) {
        throw;
      }
      auto eo_extern = eo;
      eo_extern.executor_type = ::ExecutorType::Extern;
      auto exec_desc_ptr = seq.getDescriptor(i);
      const auto body = exec_desc_ptr->getBody();
      const auto compound = dynamic_cast<const RelCompound*>(body);
      if (compound && (compound->getGroupByCount() || compound->isAggregate())) {
        LOG(INFO) << "Also failed to run the query using interoperability";
        throw;
      }
      executeRelAlgStep(seq,
                        i,
                        co,
                        eo_extern,
                        (i == exec_desc_count - 1) ? render_info : nullptr,
                        queue_time_ms);
    }
  }

  return seq.getDescriptor(exec_desc_count - 1)->getResult();
}

ExecutionResult RelAlgExecutor::executeRelAlgSubSeq(
    const RaExecutionSequence& seq,
    const std::pair<size_t, size_t> interval,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    RenderInfo* render_info,
    const int64_t queue_time_ms) {
  INJECT_TIMER(executeRelAlgSubSeq);
  executor_->catalog_ = &cat_;
  executor_->temporary_tables_ = &temporary_tables_;

  time(&now_);
  CHECK(!eo.just_explain);

  for (size_t i = interval.first; i < interval.second; i++) {
    // only render on the last step
    executeRelAlgStep(seq,
                      i,
                      co,
                      eo,
                      (i == interval.second - 1) ? render_info : nullptr,
                      queue_time_ms);
  }

  return seq.getDescriptor(interval.second - 1)->getResult();
}

void RelAlgExecutor::executeRelAlgStep(const RaExecutionSequence& seq,
                                       const size_t step_idx,
                                       const CompilationOptions& co,
                                       const ExecutionOptions& eo,
                                       RenderInfo* render_info,
                                       const int64_t queue_time_ms) {
  INJECT_TIMER(executeRelAlgStep);
  auto timer = DEBUG_TIMER(__func__);
  WindowProjectNodeContext::reset();
  auto exec_desc_ptr = seq.getDescriptor(step_idx);
  CHECK(exec_desc_ptr);
  auto& exec_desc = *exec_desc_ptr;
  const auto body = exec_desc.getBody();
  if (body->isNop()) {
    handleNop(exec_desc);
    return;
  }
  const ExecutionOptions eo_work_unit{
      eo.output_columnar_hint,
      eo.allow_multifrag,
      eo.just_explain,
      eo.allow_loop_joins,
      eo.with_watchdog && (step_idx == 0 || dynamic_cast<const RelProject*>(body)),
      eo.jit_debug,
      eo.just_validate,
      eo.with_dynamic_watchdog,
      eo.dynamic_watchdog_time_limit,
      eo.find_push_down_candidates,
      eo.just_calcite_explain,
      eo.gpu_input_mem_limit_percent,
      eo.allow_runtime_query_interrupt,
      eo.runtime_query_interrupt_frequency,
      eo.executor_type,
      step_idx == 0 ? eo.outer_fragment_indices : std::vector<size_t>()};

  const auto compound = dynamic_cast<const RelCompound*>(body);
  if (compound) {
    if (compound->isDeleteViaSelect()) {
      executeDelete(compound, co, eo_work_unit, queue_time_ms);
    } else if (compound->isUpdateViaSelect()) {
      executeUpdate(compound, co, eo_work_unit, queue_time_ms);
    } else {
      exec_desc.setResult(
          executeCompound(compound, co, eo_work_unit, render_info, queue_time_ms));
      if (exec_desc.getResult().isFilterPushDownEnabled()) {
        return;
      }
      addTemporaryTable(-compound->getId(), exec_desc.getResult().getDataPtr());
    }
    return;
  }
  const auto project = dynamic_cast<const RelProject*>(body);
  if (project) {
    if (project->isDeleteViaSelect()) {
      executeDelete(project, co, eo_work_unit, queue_time_ms);
    } else if (project->isUpdateViaSelect()) {
      executeUpdate(project, co, eo_work_unit, queue_time_ms);
    } else {
      ssize_t prev_count = -1;
      // Disabling the intermediate count optimization in distributed, as the previous
      // execution descriptor will likely not hold the aggregated result.
      if (g_skip_intermediate_count && step_idx > 0 && !g_cluster) {
        auto prev_exec_desc = seq.getDescriptor(step_idx - 1);
        CHECK(prev_exec_desc);
        // If the previous node produced a reliable count, skip the pre-flight count
        if (dynamic_cast<const RelCompound*>(prev_exec_desc->getBody()) ||
            dynamic_cast<const RelLogicalValues*>(prev_exec_desc->getBody())) {
          const auto& prev_exe_result = prev_exec_desc->getResult();
          const auto prev_result = prev_exe_result.getRows();
          if (prev_result) {
            prev_count = static_cast<ssize_t>(prev_result->rowCount());
          }
        }
      }
      exec_desc.setResult(executeProject(
          project, co, eo_work_unit, render_info, queue_time_ms, prev_count));
      if (exec_desc.getResult().isFilterPushDownEnabled()) {
        return;
      }
      addTemporaryTable(-project->getId(), exec_desc.getResult().getDataPtr());
    }
    return;
  }
  const auto aggregate = dynamic_cast<const RelAggregate*>(body);
  if (aggregate) {
    exec_desc.setResult(
        executeAggregate(aggregate, co, eo_work_unit, render_info, queue_time_ms));
    addTemporaryTable(-aggregate->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto filter = dynamic_cast<const RelFilter*>(body);
  if (filter) {
    exec_desc.setResult(
        executeFilter(filter, co, eo_work_unit, render_info, queue_time_ms));
    addTemporaryTable(-filter->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto sort = dynamic_cast<const RelSort*>(body);
  if (sort) {
    exec_desc.setResult(executeSort(sort, co, eo_work_unit, render_info, queue_time_ms));
    if (exec_desc.getResult().isFilterPushDownEnabled()) {
      return;
    }
    addTemporaryTable(-sort->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto logical_values = dynamic_cast<const RelLogicalValues*>(body);
  if (logical_values) {
    exec_desc.setResult(executeLogicalValues(logical_values, eo_work_unit));
    addTemporaryTable(-logical_values->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto modify = dynamic_cast<const RelModify*>(body);
  if (modify) {
    exec_desc.setResult(executeModify(modify, eo_work_unit));
    return;
  }
  const auto table_func = dynamic_cast<const RelTableFunction*>(body);
  if (table_func) {
    exec_desc.setResult(
        executeTableFunction(table_func, co, eo_work_unit, queue_time_ms));
    addTemporaryTable(-table_func->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  CHECK(false);
}

void RelAlgExecutor::handleNop(RaExecutionDesc& ed) {
  // just set the result of the previous node as the result of no op
  auto body = ed.getBody();
  CHECK(dynamic_cast<const RelAggregate*>(body));
  CHECK_EQ(size_t(1), body->inputCount());
  const auto input = body->getInput(0);
  body->setOutputMetainfo(input->getOutputMetainfo());
  const auto it = temporary_tables_.find(-input->getId());
  CHECK(it != temporary_tables_.end());
  // set up temp table as it could be used by the outer query or next step
  addTemporaryTable(-body->getId(), it->second);

  ed.setResult({it->second, input->getOutputMetainfo()});
}

namespace {

class RexUsedInputsVisitor : public RexVisitor<std::unordered_set<const RexInput*>> {
 public:
  RexUsedInputsVisitor(const Catalog_Namespace::Catalog& cat) : RexVisitor(), cat_(cat) {}

  const std::vector<std::shared_ptr<RexInput>>& get_inputs_owned() const {
    return synthesized_physical_inputs_owned;
  }

  std::unordered_set<const RexInput*> visitInput(
      const RexInput* rex_input) const override {
    const auto input_ra = rex_input->getSourceNode();
    CHECK(input_ra);
    const auto scan_ra = dynamic_cast<const RelScan*>(input_ra);
    if (scan_ra) {
      const auto td = scan_ra->getTableDescriptor();
      if (td) {
        const auto col_id = rex_input->getIndex();
        const auto cd = cat_.getMetadataForColumnBySpi(td->tableId, col_id + 1);
        if (cd && cd->columnType.get_physical_cols() > 0) {
          CHECK(IS_GEO(cd->columnType.get_type()));
          std::unordered_set<const RexInput*> synthesized_physical_inputs;
          for (auto i = 0; i < cd->columnType.get_physical_cols(); i++) {
            auto physical_input =
                new RexInput(scan_ra, SPIMAP_GEO_PHYSICAL_INPUT(col_id, i));
            synthesized_physical_inputs_owned.emplace_back(physical_input);
            synthesized_physical_inputs.insert(physical_input);
          }
          return synthesized_physical_inputs;
        }
      }
    }
    return {rex_input};
  }

 protected:
  std::unordered_set<const RexInput*> aggregateResult(
      const std::unordered_set<const RexInput*>& aggregate,
      const std::unordered_set<const RexInput*>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }

 private:
  mutable std::vector<std::shared_ptr<RexInput>> synthesized_physical_inputs_owned;
  const Catalog_Namespace::Catalog& cat_;
};

const RelAlgNode* get_data_sink(const RelAlgNode* ra_node) {
  if (auto join = dynamic_cast<const RelJoin*>(ra_node)) {
    CHECK_EQ(size_t(2), join->inputCount());
    return join;
  }
  CHECK_EQ(size_t(1), ra_node->inputCount());
  auto only_src = ra_node->getInput(0);
  const bool is_join = dynamic_cast<const RelJoin*>(only_src) ||
                       dynamic_cast<const RelLeftDeepInnerJoin*>(only_src);
  return is_join ? only_src : ra_node;
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>>
get_used_inputs(const RelCompound* compound, const Catalog_Namespace::Catalog& cat) {
  RexUsedInputsVisitor visitor(cat);
  const auto filter_expr = compound->getFilterExpr();
  std::unordered_set<const RexInput*> used_inputs =
      filter_expr ? visitor.visit(filter_expr) : std::unordered_set<const RexInput*>{};
  const auto sources_size = compound->getScalarSourcesSize();
  for (size_t i = 0; i < sources_size; ++i) {
    const auto source_inputs = visitor.visit(compound->getScalarSource(i));
    used_inputs.insert(source_inputs.begin(), source_inputs.end());
  }
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned(visitor.get_inputs_owned());
  return std::make_pair(used_inputs, used_inputs_owned);
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>>
get_used_inputs(const RelAggregate* aggregate, const Catalog_Namespace::Catalog& cat) {
  CHECK_EQ(size_t(1), aggregate->inputCount());
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto source = aggregate->getInput(0);
  const auto& in_metainfo = source->getOutputMetainfo();
  const auto group_count = aggregate->getGroupByCount();
  CHECK_GE(in_metainfo.size(), group_count);
  for (size_t i = 0; i < group_count; ++i) {
    auto synthesized_used_input = new RexInput(source, i);
    used_inputs_owned.emplace_back(synthesized_used_input);
    used_inputs.insert(synthesized_used_input);
  }
  for (const auto& agg_expr : aggregate->getAggExprs()) {
    for (size_t i = 0; i < agg_expr->size(); ++i) {
      const auto operand_idx = agg_expr->getOperand(i);
      CHECK_GE(in_metainfo.size(), static_cast<size_t>(operand_idx));
      auto synthesized_used_input = new RexInput(source, operand_idx);
      used_inputs_owned.emplace_back(synthesized_used_input);
      used_inputs.insert(synthesized_used_input);
    }
  }
  return std::make_pair(used_inputs, used_inputs_owned);
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>>
get_used_inputs(const RelProject* project, const Catalog_Namespace::Catalog& cat) {
  RexUsedInputsVisitor visitor(cat);
  std::unordered_set<const RexInput*> used_inputs;
  for (size_t i = 0; i < project->size(); ++i) {
    const auto proj_inputs = visitor.visit(project->getProjectAt(i));
    used_inputs.insert(proj_inputs.begin(), proj_inputs.end());
  }
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned(visitor.get_inputs_owned());
  return std::make_pair(used_inputs, used_inputs_owned);
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>>
get_used_inputs(const RelTableFunction* table_func,
                const Catalog_Namespace::Catalog& cat) {
  RexUsedInputsVisitor visitor(cat);
  std::unordered_set<const RexInput*> used_inputs;
  for (size_t i = 0; i < table_func->getTableFuncInputsSize(); ++i) {
    const auto table_func_inputs = visitor.visit(table_func->getTableFuncInputAt(i));
    used_inputs.insert(table_func_inputs.begin(), table_func_inputs.end());
  }
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned(visitor.get_inputs_owned());
  return std::make_pair(used_inputs, used_inputs_owned);
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>>
get_used_inputs(const RelFilter* filter, const Catalog_Namespace::Catalog& cat) {
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto data_sink_node = get_data_sink(filter);
  for (size_t nest_level = 0; nest_level < data_sink_node->inputCount(); ++nest_level) {
    const auto source = data_sink_node->getInput(nest_level);
    const auto scan_source = dynamic_cast<const RelScan*>(source);
    if (scan_source) {
      CHECK(source->getOutputMetainfo().empty());
      for (size_t i = 0; i < scan_source->size(); ++i) {
        auto synthesized_used_input = new RexInput(scan_source, i);
        used_inputs_owned.emplace_back(synthesized_used_input);
        used_inputs.insert(synthesized_used_input);
      }
    } else {
      const auto& partial_in_metadata = source->getOutputMetainfo();
      for (size_t i = 0; i < partial_in_metadata.size(); ++i) {
        auto synthesized_used_input = new RexInput(source, i);
        used_inputs_owned.emplace_back(synthesized_used_input);
        used_inputs.insert(synthesized_used_input);
      }
    }
  }
  return std::make_pair(used_inputs, used_inputs_owned);
}

int table_id_from_ra(const RelAlgNode* ra_node) {
  const auto scan_ra = dynamic_cast<const RelScan*>(ra_node);
  if (scan_ra) {
    const auto td = scan_ra->getTableDescriptor();
    CHECK(td);
    return td->tableId;
  }
  return -ra_node->getId();
}

std::unordered_map<const RelAlgNode*, int> get_input_nest_levels(
    const RelAlgNode* ra_node,
    const std::vector<size_t>& input_permutation) {
  const auto data_sink_node = get_data_sink(ra_node);
  std::unordered_map<const RelAlgNode*, int> input_to_nest_level;
  for (size_t input_idx = 0; input_idx < data_sink_node->inputCount(); ++input_idx) {
    const auto input_node_idx =
        input_permutation.empty() ? input_idx : input_permutation[input_idx];
    const auto input_ra = data_sink_node->getInput(input_node_idx);
    const auto it_ok = input_to_nest_level.emplace(input_ra, input_idx);
    CHECK(it_ok.second);
    LOG_IF(INFO, !input_permutation.empty())
        << "Assigned input " << input_ra->toString() << " to nest level " << input_idx;
  }
  return input_to_nest_level;
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>>
get_join_source_used_inputs(const RelAlgNode* ra_node,
                            const Catalog_Namespace::Catalog& cat) {
  const auto data_sink_node = get_data_sink(ra_node);
  if (auto join = dynamic_cast<const RelJoin*>(data_sink_node)) {
    CHECK_EQ(join->inputCount(), 2u);
    const auto condition = join->getCondition();
    RexUsedInputsVisitor visitor(cat);
    auto condition_inputs = visitor.visit(condition);
    std::vector<std::shared_ptr<RexInput>> condition_inputs_owned(
        visitor.get_inputs_owned());
    return std::make_pair(condition_inputs, condition_inputs_owned);
  }

  if (auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(data_sink_node)) {
    CHECK_GE(left_deep_join->inputCount(), 2u);
    const auto condition = left_deep_join->getInnerCondition();
    RexUsedInputsVisitor visitor(cat);
    auto result = visitor.visit(condition);
    for (size_t nesting_level = 1; nesting_level <= left_deep_join->inputCount() - 1;
         ++nesting_level) {
      const auto outer_condition = left_deep_join->getOuterCondition(nesting_level);
      if (outer_condition) {
        const auto outer_result = visitor.visit(outer_condition);
        result.insert(outer_result.begin(), outer_result.end());
      }
    }
    std::vector<std::shared_ptr<RexInput>> used_inputs_owned(visitor.get_inputs_owned());
    return std::make_pair(result, used_inputs_owned);
  }

  CHECK_EQ(ra_node->inputCount(), 1u);
  return std::make_pair(std::unordered_set<const RexInput*>{},
                        std::vector<std::shared_ptr<RexInput>>{});
}

std::vector<const RelAlgNode*> get_non_join_sequence(const RelAlgNode* ra) {
  std::vector<const RelAlgNode*> seq;
  for (auto join = dynamic_cast<const RelJoin*>(ra); join;
       join = static_cast<const RelJoin*>(join->getInput(0))) {
    CHECK_EQ(size_t(2), join->inputCount());
    seq.emplace_back(join->getInput(1));
    auto lhs = join->getInput(0);
    if (!dynamic_cast<const RelJoin*>(lhs)) {
      seq.emplace_back(lhs);
      break;
    }
  }
  std::reverse(seq.begin(), seq.end());
  return seq;
}

void collect_used_input_desc(
    std::vector<InputDescriptor>& input_descs,
    const Catalog_Namespace::Catalog& cat,
    std::unordered_set<std::shared_ptr<const InputColDescriptor>>& input_col_descs_unique,
    const RelAlgNode* ra_node,
    const std::unordered_set<const RexInput*>& source_used_inputs,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  std::unordered_set<InputDescriptor> input_descs_unique(input_descs.begin(),
                                                         input_descs.end());
  const auto non_join_src_seq = get_non_join_sequence(get_data_sink(ra_node));
  std::unordered_map<const RelAlgNode*, int> non_join_to_nest_level;
  for (const auto node : non_join_src_seq) {
    non_join_to_nest_level.insert(std::make_pair(node, non_join_to_nest_level.size()));
  }
  for (const auto used_input : source_used_inputs) {
    const auto input_ra = used_input->getSourceNode();
    const int table_id = table_id_from_ra(input_ra);
    const auto col_id = used_input->getIndex();
    auto it = input_to_nest_level.find(input_ra);
    if (it == input_to_nest_level.end()) {
      throw std::runtime_error("Bushy joins not supported");
    }
    const int input_desc = it->second;
    input_col_descs_unique.insert(std::make_shared<const InputColDescriptor>(
        dynamic_cast<const RelScan*>(input_ra)
            ? cat.getColumnIdBySpi(table_id, col_id + 1)
            : col_id,
        table_id,
        input_desc));
  }
}

template <class RA>
std::pair<std::vector<InputDescriptor>,
          std::list<std::shared_ptr<const InputColDescriptor>>>
get_input_desc_impl(const RA* ra_node,
                    const std::unordered_set<const RexInput*>& used_inputs,
                    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                    const std::vector<size_t>& input_permutation,
                    const Catalog_Namespace::Catalog& cat) {
  std::vector<InputDescriptor> input_descs;
  const auto data_sink_node = get_data_sink(ra_node);
  for (size_t input_idx = 0; input_idx < data_sink_node->inputCount(); ++input_idx) {
    const auto input_node_idx =
        input_permutation.empty() ? input_idx : input_permutation[input_idx];
    const auto input_ra = data_sink_node->getInput(input_node_idx);
    const int table_id = table_id_from_ra(input_ra);
    input_descs.emplace_back(table_id, input_idx);
  }
  std::sort(input_descs.begin(),
            input_descs.end(),
            [](const InputDescriptor& lhs, const InputDescriptor& rhs) {
              return lhs.getNestLevel() < rhs.getNestLevel();
            });
  std::unordered_set<std::shared_ptr<const InputColDescriptor>> input_col_descs_unique;
  collect_used_input_desc(input_descs,
                          cat,
                          input_col_descs_unique,
                          ra_node,
                          used_inputs,
                          input_to_nest_level);
  std::unordered_set<const RexInput*> join_source_used_inputs;
  std::vector<std::shared_ptr<RexInput>> join_source_used_inputs_owned;
  std::tie(join_source_used_inputs, join_source_used_inputs_owned) =
      get_join_source_used_inputs(ra_node, cat);
  collect_used_input_desc(input_descs,
                          cat,
                          input_col_descs_unique,
                          ra_node,
                          join_source_used_inputs,
                          input_to_nest_level);
  std::vector<std::shared_ptr<const InputColDescriptor>> input_col_descs(
      input_col_descs_unique.begin(), input_col_descs_unique.end());

  std::sort(
      input_col_descs.begin(),
      input_col_descs.end(),
      [](std::shared_ptr<const InputColDescriptor> const& lhs,
         std::shared_ptr<const InputColDescriptor> const& rhs) {
        if (lhs->getScanDesc().getNestLevel() == rhs->getScanDesc().getNestLevel()) {
          return lhs->getColId() < rhs->getColId();
        }
        return lhs->getScanDesc().getNestLevel() < rhs->getScanDesc().getNestLevel();
      });
  return {input_descs,
          std::list<std::shared_ptr<const InputColDescriptor>>(input_col_descs.begin(),
                                                               input_col_descs.end())};
}

template <class RA>
std::tuple<std::vector<InputDescriptor>,
           std::list<std::shared_ptr<const InputColDescriptor>>,
           std::vector<std::shared_ptr<RexInput>>>
get_input_desc(const RA* ra_node,
               const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
               const std::vector<size_t>& input_permutation,
               const Catalog_Namespace::Catalog& cat) {
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  std::tie(used_inputs, used_inputs_owned) = get_used_inputs(ra_node, cat);
  auto input_desc_pair = get_input_desc_impl(
      ra_node, used_inputs, input_to_nest_level, input_permutation, cat);
  return std::make_tuple(
      input_desc_pair.first, input_desc_pair.second, used_inputs_owned);
}

size_t get_scalar_sources_size(const RelCompound* compound) {
  return compound->getScalarSourcesSize();
}

size_t get_scalar_sources_size(const RelProject* project) {
  return project->size();
}

size_t get_scalar_sources_size(const RelTableFunction* table_func) {
  return table_func->getTableFuncInputsSize();
}

const RexScalar* scalar_at(const size_t i, const RelCompound* compound) {
  return compound->getScalarSource(i);
}

const RexScalar* scalar_at(const size_t i, const RelProject* project) {
  return project->getProjectAt(i);
}

const RexScalar* scalar_at(const size_t i, const RelTableFunction* table_func) {
  return table_func->getTableFuncInputAt(i);
}

std::shared_ptr<Analyzer::Expr> set_transient_dict(
    const std::shared_ptr<Analyzer::Expr> expr) {
  const auto& ti = expr->get_type_info();
  if (!ti.is_string() || ti.get_compression() != kENCODING_NONE) {
    return expr;
  }
  auto transient_dict_ti = ti;
  transient_dict_ti.set_compression(kENCODING_DICT);
  transient_dict_ti.set_comp_param(TRANSIENT_DICT_ID);
  transient_dict_ti.set_fixed_size();
  return expr->add_cast(transient_dict_ti);
}

void set_transient_dict_maybe(
    std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
    const std::shared_ptr<Analyzer::Expr>& expr) {
  try {
    scalar_sources.push_back(set_transient_dict(fold_expr(expr.get())));
  } catch (...) {
    scalar_sources.push_back(fold_expr(expr.get()));
  }
}

std::shared_ptr<Analyzer::Expr> cast_dict_to_none(
    const std::shared_ptr<Analyzer::Expr>& input) {
  const auto& input_ti = input->get_type_info();
  if (input_ti.is_string() && input_ti.get_compression() == kENCODING_DICT) {
    return input->add_cast(SQLTypeInfo(kTEXT, input_ti.get_notnull()));
  }
  return input;
}

template <class RA>
std::vector<std::shared_ptr<Analyzer::Expr>> translate_scalar_sources(
    const RA* ra_node,
    const RelAlgTranslator& translator,
    const ::ExecutorType executor_type) {
  std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources;
  for (size_t i = 0; i < get_scalar_sources_size(ra_node); ++i) {
    const auto scalar_rex = scalar_at(i, ra_node);
    if (dynamic_cast<const RexRef*>(scalar_rex)) {
      // RexRef are synthetic scalars we append at the end of the real ones
      // for the sake of taking memory ownership, no real work needed here.
      continue;
    }

    const auto scalar_expr =
        rewrite_array_elements(translator.translateScalarRex(scalar_rex).get());
    const auto rewritten_expr = rewrite_expr(scalar_expr.get());
    if (executor_type == ExecutorType::Native) {
      set_transient_dict_maybe(scalar_sources, rewritten_expr);
    } else {
      scalar_sources.push_back(cast_dict_to_none(fold_expr(rewritten_expr.get())));
    }
  }

  return scalar_sources;
}

template <class RA>
std::vector<std::shared_ptr<Analyzer::Expr>> translate_scalar_sources_for_update(
    const RA* ra_node,
    const RelAlgTranslator& translator,
    int32_t tableId,
    const Catalog_Namespace::Catalog& cat,
    const ColumnNameList& colNames,
    size_t starting_projection_column_idx) {
  std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources;
  for (size_t i = 0; i < get_scalar_sources_size(ra_node); ++i) {
    const auto scalar_rex = scalar_at(i, ra_node);
    if (dynamic_cast<const RexRef*>(scalar_rex)) {
      // RexRef are synthetic scalars we append at the end of the real ones
      // for the sake of taking memory ownership, no real work needed here.
      continue;
    }

    std::shared_ptr<Analyzer::Expr> translated_expr;
    if (i >= starting_projection_column_idx && i < get_scalar_sources_size(ra_node) - 1) {
      translated_expr = cast_to_column_type(translator.translateScalarRex(scalar_rex),
                                            tableId,
                                            cat,
                                            colNames[i - starting_projection_column_idx]);
    } else {
      translated_expr = translator.translateScalarRex(scalar_rex);
    }
    const auto scalar_expr = rewrite_array_elements(translated_expr.get());
    const auto rewritten_expr = rewrite_expr(scalar_expr.get());
    set_transient_dict_maybe(scalar_sources, rewritten_expr);
  }

  return scalar_sources;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_groupby_exprs(
    const RelCompound* compound,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  if (!compound->isAggregate()) {
    return {nullptr};
  }
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (size_t group_idx = 0; group_idx < compound->getGroupByCount(); ++group_idx) {
    groupby_exprs.push_back(set_transient_dict(scalar_sources[group_idx]));
  }
  return groupby_exprs;
}

std::list<std::shared_ptr<Analyzer::Expr>> translate_groupby_exprs(
    const RelAggregate* aggregate,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (size_t group_idx = 0; group_idx < aggregate->getGroupByCount(); ++group_idx) {
    groupby_exprs.push_back(set_transient_dict(scalar_sources[group_idx]));
  }
  return groupby_exprs;
}

QualsConjunctiveForm translate_quals(const RelCompound* compound,
                                     const RelAlgTranslator& translator) {
  const auto filter_rex = compound->getFilterExpr();
  const auto filter_expr =
      filter_rex ? translator.translateScalarRex(filter_rex) : nullptr;
  return filter_expr ? qual_to_conjunctive_form(fold_expr(filter_expr.get()))
                     : QualsConjunctiveForm{};
}

std::vector<Analyzer::Expr*> translate_targets(
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
    const RelCompound* compound,
    const RelAlgTranslator& translator,
    const ExecutorType executor_type) {
  std::vector<Analyzer::Expr*> target_exprs;
  for (size_t i = 0; i < compound->size(); ++i) {
    const auto target_rex = compound->getTargetExpr(i);
    const auto target_rex_agg = dynamic_cast<const RexAgg*>(target_rex);
    std::shared_ptr<Analyzer::Expr> target_expr;
    if (target_rex_agg) {
      target_expr =
          RelAlgTranslator::translateAggregateRex(target_rex_agg, scalar_sources);
    } else {
      const auto target_rex_scalar = dynamic_cast<const RexScalar*>(target_rex);
      const auto target_rex_ref = dynamic_cast<const RexRef*>(target_rex_scalar);
      if (target_rex_ref) {
        const auto ref_idx = target_rex_ref->getIndex();
        CHECK_GE(ref_idx, size_t(1));
        CHECK_LE(ref_idx, groupby_exprs.size());
        const auto groupby_expr = *std::next(groupby_exprs.begin(), ref_idx - 1);
        target_expr = var_ref(groupby_expr.get(), Analyzer::Var::kGROUPBY, ref_idx);
      } else {
        target_expr = translator.translateScalarRex(target_rex_scalar);
        auto rewritten_expr = rewrite_expr(target_expr.get());
        target_expr = fold_expr(rewritten_expr.get());
        if (executor_type == ExecutorType::Native) {
          try {
            target_expr = set_transient_dict(target_expr);
          } catch (...) {
            // noop
          }
        } else {
          target_expr = cast_dict_to_none(target_expr);
        }
      }
    }
    CHECK(target_expr);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}

std::vector<Analyzer::Expr*> translate_targets(
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
    const RelAggregate* aggregate,
    const RelAlgTranslator& translator) {
  std::vector<Analyzer::Expr*> target_exprs;
  size_t group_key_idx = 0;
  for (const auto& groupby_expr : groupby_exprs) {
    auto target_expr =
        var_ref(groupby_expr.get(), Analyzer::Var::kGROUPBY, group_key_idx++);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }

  for (const auto& target_rex_agg : aggregate->getAggExprs()) {
    auto target_expr =
        RelAlgTranslator::translateAggregateRex(target_rex_agg.get(), scalar_sources);
    CHECK(target_expr);
    target_expr = fold_expr(target_expr.get());
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}

bool is_count_distinct(const Analyzer::Expr* expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(expr);
  return agg_expr && agg_expr->get_is_distinct();
}

bool is_agg(const Analyzer::Expr* expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(expr);
  if (agg_expr && agg_expr->get_contains_agg()) {
    auto agg_type = agg_expr->get_aggtype();
    if (agg_type == SQLAgg::kMIN || agg_type == SQLAgg::kMAX ||
        agg_type == SQLAgg::kSUM || agg_type == SQLAgg::kAVG) {
      return true;
    }
  }
  return false;
}

inline SQLTypeInfo get_logical_type_for_expr(const Analyzer::Expr& expr) {
  if (is_count_distinct(&expr)) {
    return SQLTypeInfo(kBIGINT, false);
  } else if (is_agg(&expr)) {
    return get_nullable_logical_type_info(expr.get_type_info());
  }
  return get_logical_type_info(expr.get_type_info());
}

template <class RA>
std::vector<TargetMetaInfo> get_targets_meta(
    const RA* ra_node,
    const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<TargetMetaInfo> targets_meta;
  for (size_t i = 0; i < ra_node->size(); ++i) {
    CHECK(target_exprs[i]);
    // TODO(alex): remove the count distinct type fixup.
    targets_meta.emplace_back(ra_node->getFieldName(i),
                              get_logical_type_for_expr(*target_exprs[i]),
                              target_exprs[i]->get_type_info());
  }
  return targets_meta;
}

}  // namespace

void RelAlgExecutor::executeUpdate(const RelAlgNode* node,
                                   const CompilationOptions& co_in,
                                   const ExecutionOptions& eo_in,
                                   const int64_t queue_time_ms) {
  CHECK(node);
  auto timer = DEBUG_TIMER(__func__);

  UpdateTriggeredCacheInvalidator::invalidateCaches();

  auto co = co_in;
  co.hoist_literals = false;  // disable literal hoisting as it interferes with dict
                              // encoded string updates

  auto execute_update_for_node =
      [this, &co, &eo_in](const auto node, auto& work_unit, const bool is_aggregate) {
        UpdateTransactionParameters update_params(node->getModifiedTableDescriptor(),
                                                  node->getTargetColumns(),
                                                  node->getOutputMetainfo(),
                                                  node->isVarlenUpdateRequired());

        const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);

        auto execute_update_ra_exe_unit =
            [this, &co, &eo_in, &table_infos, &update_params](
                const RelAlgExecutionUnit& ra_exe_unit, const bool is_aggregate) {
              CompilationOptions co_project = CompilationOptions::makeCpuOnly(co);

              auto eo = eo_in;
              if (update_params.tableIsTemporary()) {
                eo.output_columnar_hint = true;
                co_project.allow_lazy_fetch = false;
              }

              auto update_callback = yieldUpdateCallback(update_params);
              executor_->executeUpdate(ra_exe_unit,
                                       table_infos,
                                       co_project,
                                       eo,
                                       cat_,
                                       executor_->row_set_mem_owner_,
                                       update_callback,
                                       is_aggregate);
              update_params.finalizeTransaction();
            };

        if (update_params.tableIsTemporary()) {
          // hold owned target exprs during execution if rewriting
          auto query_rewrite = std::make_unique<QueryRewriter>(table_infos, executor_);
          // rewrite temp table updates to generate the full column by moving the where
          // clause into a case if such a rewrite is not possible, bail on the update
          // operation build an expr for the update target
          const auto td = update_params.getTableDescriptor();
          CHECK(td);
          const auto update_column_names = update_params.getUpdateColumnNames();
          if (update_column_names.size() > 1) {
            throw std::runtime_error(
                "Multi-column update is not yet supported for temporary tables.");
          }

          auto cd = cat_.getMetadataForColumn(td->tableId, update_column_names.front());
          CHECK(cd);
          auto projected_column_to_update =
              makeExpr<Analyzer::ColumnVar>(cd->columnType, td->tableId, cd->columnId, 0);
          const auto rewritten_exe_unit = query_rewrite->rewriteColumnarUpdate(
              work_unit.exe_unit, projected_column_to_update);
          if (rewritten_exe_unit.target_exprs.front()->get_type_info().is_varlen()) {
            throw std::runtime_error(
                "Variable length updates not yet supported on temporary tables.");
          }
          execute_update_ra_exe_unit(rewritten_exe_unit, is_aggregate);
        } else {
          execute_update_ra_exe_unit(work_unit.exe_unit, is_aggregate);
        }
      };

  if (auto compound = dynamic_cast<const RelCompound*>(node)) {
    auto work_unit =
        createCompoundWorkUnit(compound, {{}, SortAlgorithm::Default, 0, 0}, eo_in);

    execute_update_for_node(compound, work_unit, compound->isAggregate());
  } else if (auto project = dynamic_cast<const RelProject*>(node)) {
    auto work_unit =
        createProjectWorkUnit(project, {{}, SortAlgorithm::Default, 0, 0}, eo_in);

    if (project->isSimple()) {
      CHECK_EQ(size_t(1), project->inputCount());
      const auto input_ra = project->getInput(0);
      if (dynamic_cast<const RelSort*>(input_ra)) {
        const auto& input_table =
            get_temporary_table(&temporary_tables_, -input_ra->getId());
        CHECK(input_table);
        work_unit.exe_unit.scan_limit = input_table->rowCount();
      }
    }

    execute_update_for_node(project, work_unit, false);
  } else {
    throw std::runtime_error("Unsupported parent node for update: " + node->toString());
  }
}

void RelAlgExecutor::executeDelete(const RelAlgNode* node,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo_in,
                                   const int64_t queue_time_ms) {
  CHECK(node);
  auto timer = DEBUG_TIMER(__func__);

  DeleteTriggeredCacheInvalidator::invalidateCaches();

  auto execute_delete_for_node = [this, &co, &eo_in](const auto node,
                                                     auto& work_unit,
                                                     const bool is_aggregate) {
    auto* table_descriptor = node->getModifiedTableDescriptor();
    CHECK(table_descriptor);
    if (!table_descriptor->hasDeletedCol) {
      throw std::runtime_error(
          "DELETE only supported on tables with the vacuum attribute set to 'delayed'");
    }

    const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);

    auto execute_delete_ra_exe_unit =
        [this, &table_infos, &table_descriptor, &eo_in, &co](const auto& exe_unit,
                                                             const bool is_aggregate) {
          DeleteTransactionParameters delete_params(table_is_temporary(table_descriptor));
          auto delete_callback = yieldDeleteCallback(delete_params);
          CompilationOptions co_delete = CompilationOptions::makeCpuOnly(co);

          auto eo = eo_in;
          if (delete_params.tableIsTemporary()) {
            eo.output_columnar_hint = true;
            co_delete.add_delete_column =
                false;  // project the entire delete column for columnar update
          }

          executor_->executeUpdate(exe_unit,
                                   table_infos,
                                   co_delete,
                                   eo,
                                   cat_,
                                   executor_->row_set_mem_owner_,
                                   delete_callback,
                                   is_aggregate);
          delete_params.finalizeTransaction();
        };

    if (table_is_temporary(table_descriptor)) {
      auto query_rewrite = std::make_unique<QueryRewriter>(table_infos, executor_);
      auto cd = cat_.getDeletedColumn(table_descriptor);
      CHECK(cd);
      auto delete_column_expr = makeExpr<Analyzer::ColumnVar>(
          cd->columnType, table_descriptor->tableId, cd->columnId, 0);
      const auto rewritten_exe_unit =
          query_rewrite->rewriteColumnarDelete(work_unit.exe_unit, delete_column_expr);
      execute_delete_ra_exe_unit(rewritten_exe_unit, is_aggregate);
    } else {
      execute_delete_ra_exe_unit(work_unit.exe_unit, is_aggregate);
    }
  };

  if (auto compound = dynamic_cast<const RelCompound*>(node)) {
    const auto work_unit =
        createCompoundWorkUnit(compound, {{}, SortAlgorithm::Default, 0, 0}, eo_in);
    execute_delete_for_node(compound, work_unit, compound->isAggregate());
  } else if (auto project = dynamic_cast<const RelProject*>(node)) {
    auto work_unit =
        createProjectWorkUnit(project, {{}, SortAlgorithm::Default, 0, 0}, eo_in);
    if (project->isSimple()) {
      CHECK_EQ(size_t(1), project->inputCount());
      const auto input_ra = project->getInput(0);
      if (dynamic_cast<const RelSort*>(input_ra)) {
        const auto& input_table =
            get_temporary_table(&temporary_tables_, -input_ra->getId());
        CHECK(input_table);
        work_unit.exe_unit.scan_limit = input_table->rowCount();
      }
    }
    execute_delete_for_node(project, work_unit, false);
  } else {
    throw std::runtime_error("Unsupported parent node for delete: " + node->toString());
  }
}

ExecutionResult RelAlgExecutor::executeCompound(const RelCompound* compound,
                                                const CompilationOptions& co,
                                                const ExecutionOptions& eo,
                                                RenderInfo* render_info,
                                                const int64_t queue_time_ms) {
  auto timer = DEBUG_TIMER(__func__);
  const auto work_unit =
      createCompoundWorkUnit(compound, {{}, SortAlgorithm::Default, 0, 0}, eo);
  CompilationOptions co_compound = co;
  return executeWorkUnit(work_unit,
                         compound->getOutputMetainfo(),
                         compound->isAggregate(),
                         co_compound,
                         eo,
                         render_info,
                         queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeAggregate(const RelAggregate* aggregate,
                                                 const CompilationOptions& co,
                                                 const ExecutionOptions& eo,
                                                 RenderInfo* render_info,
                                                 const int64_t queue_time_ms) {
  auto timer = DEBUG_TIMER(__func__);
  const auto work_unit = createAggregateWorkUnit(
      aggregate, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  return executeWorkUnit(work_unit,
                         aggregate->getOutputMetainfo(),
                         true,
                         co,
                         eo,
                         render_info,
                         queue_time_ms);
}

namespace {

// Returns true iff the execution unit contains window functions.
bool is_window_execution_unit(const RelAlgExecutionUnit& ra_exe_unit) {
  return std::any_of(ra_exe_unit.target_exprs.begin(),
                     ra_exe_unit.target_exprs.end(),
                     [](const Analyzer::Expr* expr) {
                       return dynamic_cast<const Analyzer::WindowFunction*>(expr);
                     });
}

}  // namespace

ExecutionResult RelAlgExecutor::executeProject(const RelProject* project,
                                               const CompilationOptions& co,
                                               const ExecutionOptions& eo,
                                               RenderInfo* render_info,
                                               const int64_t queue_time_ms,
                                               const ssize_t previous_count) {
  auto timer = DEBUG_TIMER(__func__);
  auto work_unit = createProjectWorkUnit(project, {{}, SortAlgorithm::Default, 0, 0}, eo);
  CompilationOptions co_project = co;
  if (project->isSimple()) {
    CHECK_EQ(size_t(1), project->inputCount());
    const auto input_ra = project->getInput(0);
    if (dynamic_cast<const RelSort*>(input_ra)) {
      co_project.device_type = ExecutorDeviceType::CPU;
      const auto& input_table =
          get_temporary_table(&temporary_tables_, -input_ra->getId());
      CHECK(input_table);
      work_unit.exe_unit.scan_limit =
          std::min(input_table->getLimit(), input_table->rowCount());
    }
  }
  return executeWorkUnit(work_unit,
                         project->getOutputMetainfo(),
                         false,
                         co_project,
                         eo,
                         render_info,
                         queue_time_ms,
                         previous_count);
}

ExecutionResult RelAlgExecutor::executeTableFunction(const RelTableFunction* table_func,
                                                     const CompilationOptions& co_in,
                                                     const ExecutionOptions& eo,
                                                     const int64_t queue_time_ms) {
  INJECT_TIMER(executeTableFunction);
  auto timer = DEBUG_TIMER(__func__);

  auto co = co_in;

  if (g_cluster) {
    throw std::runtime_error("Table functions not supported in distributed mode yet");
  }
  if (!g_enable_table_functions) {
    throw std::runtime_error("Table function support is disabled");
  }

  auto table_func_work_unit = createTableFunctionWorkUnit(table_func, eo.just_explain);
  const auto body = table_func_work_unit.body;
  CHECK(body);

  const auto table_infos =
      get_table_infos(table_func_work_unit.exe_unit.input_descs, executor_);

  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     co.device_type,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     executor_),
                         {}};

  try {
    result = {executor_->executeTableFunction(
                  table_func_work_unit.exe_unit, table_infos, co, eo, cat_),
              body->getOutputMetainfo()};
  } catch (const QueryExecutionError& e) {
    handlePersistentError(e.getErrorCode());
    CHECK(e.getErrorCode() == Executor::ERR_OUT_OF_GPU_MEM);
    throw std::runtime_error("Table function ran out of memory during execution");
  }
  result.setQueueTime(queue_time_ms);
  return result;
}

namespace {

// Creates a new expression which has the range table index set to 1. This is needed to
// reuse the hash join construction helpers to generate a hash table for the window
// function partition: create an equals expression with left and right sides identical
// except for the range table index.
std::shared_ptr<Analyzer::Expr> transform_to_inner(const Analyzer::Expr* expr) {
  const auto tuple = dynamic_cast<const Analyzer::ExpressionTuple*>(expr);
  if (tuple) {
    std::vector<std::shared_ptr<Analyzer::Expr>> transformed_tuple;
    for (const auto& element : tuple->getTuple()) {
      transformed_tuple.push_back(transform_to_inner(element.get()));
    }
    return makeExpr<Analyzer::ExpressionTuple>(transformed_tuple);
  }
  const auto col = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (!col) {
    throw std::runtime_error("Only columns supported in the window partition for now");
  }
  return makeExpr<Analyzer::ColumnVar>(
      col->get_type_info(), col->get_table_id(), col->get_column_id(), 1);
}

}  // namespace

void RelAlgExecutor::computeWindow(const RelAlgExecutionUnit& ra_exe_unit,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   ColumnCacheMap& column_cache_map,
                                   const int64_t queue_time_ms) {
  auto query_infos = get_table_infos(ra_exe_unit.input_descs, executor_);
  CHECK_EQ(query_infos.size(), size_t(1));
  if (query_infos.front().info.fragments.size() != 1) {
    throw std::runtime_error(
        "Only single fragment tables supported for window functions for now");
  }
  if (eo.executor_type == ::ExecutorType::Extern) {
    return;
  }
  query_infos.push_back(query_infos.front());
  auto window_project_node_context = WindowProjectNodeContext::create();
  for (size_t target_index = 0; target_index < ra_exe_unit.target_exprs.size();
       ++target_index) {
    const auto& target_expr = ra_exe_unit.target_exprs[target_index];
    const auto window_func = dynamic_cast<const Analyzer::WindowFunction*>(target_expr);
    if (!window_func) {
      continue;
    }
    // Always use baseline layout hash tables for now, make the expression a tuple.
    const auto& partition_keys = window_func->getPartitionKeys();
    std::shared_ptr<Analyzer::Expr> partition_key_tuple;
    if (partition_keys.size() > 1) {
      partition_key_tuple = makeExpr<Analyzer::ExpressionTuple>(partition_keys);
    } else {
      if (partition_keys.empty()) {
        throw std::runtime_error(
            "Empty window function partitions are not supported yet");
      }
      CHECK_EQ(partition_keys.size(), size_t(1));
      partition_key_tuple = partition_keys.front();
    }
    // Creates a tautology equality with the partition expression on both sides.
    const auto partition_key_cond =
        makeExpr<Analyzer::BinOper>(kBOOLEAN,
                                    kBW_EQ,
                                    kONE,
                                    partition_key_tuple,
                                    transform_to_inner(partition_key_tuple.get()));
    auto context = createWindowFunctionContext(
        window_func, partition_key_cond, ra_exe_unit, query_infos, co, column_cache_map);
    context->compute();
    window_project_node_context->addWindowFunctionContext(std::move(context),
                                                          target_index);
  }
}

std::unique_ptr<WindowFunctionContext> RelAlgExecutor::createWindowFunctionContext(
    const Analyzer::WindowFunction* window_func,
    const std::shared_ptr<Analyzer::BinOper>& partition_key_cond,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const CompilationOptions& co,
    ColumnCacheMap& column_cache_map) {
  const auto memory_level = co.device_type == ExecutorDeviceType::GPU
                                ? MemoryLevel::GPU_LEVEL
                                : MemoryLevel::CPU_LEVEL;
  const auto join_table_or_err =
      executor_->buildHashTableForQualifier(partition_key_cond,
                                            query_infos,
                                            memory_level,
                                            JoinHashTableInterface::HashType::OneToMany,
                                            column_cache_map);
  if (!join_table_or_err.fail_reason.empty()) {
    throw std::runtime_error(join_table_or_err.fail_reason);
  }
  CHECK(join_table_or_err.hash_table->getHashType() ==
        JoinHashTableInterface::HashType::OneToMany);
  const auto& order_keys = window_func->getOrderKeys();
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  const size_t elem_count = query_infos.front().info.fragments.front().getNumTuples();
  auto context = std::make_unique<WindowFunctionContext>(
      window_func, join_table_or_err.hash_table, elem_count, co.device_type);
  for (const auto& order_key : order_keys) {
    const auto order_col =
        std::dynamic_pointer_cast<const Analyzer::ColumnVar>(order_key);
    if (!order_col) {
      throw std::runtime_error("Only order by columns supported for now");
    }
    const int8_t* column;
    size_t join_col_elem_count;
    std::tie(column, join_col_elem_count) =
        ColumnFetcher::getOneColumnFragment(executor_,
                                            *order_col,
                                            query_infos.front().info.fragments.front(),
                                            memory_level,
                                            0,
                                            chunks_owner,
                                            column_cache_map);
    CHECK_EQ(join_col_elem_count, elem_count);
    context->addOrderColumn(column, order_col.get(), chunks_owner);
  }
  return context;
}

ExecutionResult RelAlgExecutor::executeFilter(const RelFilter* filter,
                                              const CompilationOptions& co,
                                              const ExecutionOptions& eo,
                                              RenderInfo* render_info,
                                              const int64_t queue_time_ms) {
  auto timer = DEBUG_TIMER(__func__);
  const auto work_unit =
      createFilterWorkUnit(filter, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  return executeWorkUnit(
      work_unit, filter->getOutputMetainfo(), false, co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeLogicalValues(
    const RelLogicalValues* logical_values,
    const ExecutionOptions& eo) {
  auto timer = DEBUG_TIMER(__func__);
  if (eo.just_explain) {
    throw std::runtime_error("EXPLAIN not supported for LogicalValues");
  }

  QueryMemoryDescriptor query_mem_desc(executor_,
                                       logical_values->getNumRows(),
                                       QueryDescriptionType::Projection,
                                       /*is_table_function=*/false);

  auto tuple_type = logical_values->getTupleType();
  for (size_t i = 0; i < tuple_type.size(); ++i) {
    auto& target_meta_info = tuple_type[i];
    if (target_meta_info.get_type_info().is_varlen()) {
      throw std::runtime_error("Variable length types not supported in VALUES yet.");
    }
    if (target_meta_info.get_type_info().get_type() == kNULLT) {
      // replace w/ bigint
      tuple_type[i] =
          TargetMetaInfo(target_meta_info.get_resname(), SQLTypeInfo(kBIGINT, false));
    }
    query_mem_desc.addColSlotInfo(
        {std::make_tuple(tuple_type[i].get_type_info().get_size(), 8)});
  }
  logical_values->setOutputMetainfo(tuple_type);

  std::vector<TargetInfo> target_infos;
  for (const auto& tuple_type_component : tuple_type) {
    target_infos.emplace_back(TargetInfo{false,
                                         kCOUNT,
                                         tuple_type_component.get_type_info(),
                                         SQLTypeInfo(kNULLT, false),
                                         false,
                                         false});
  }
  auto rs = std::make_shared<ResultSet>(target_infos,
                                        ExecutorDeviceType::CPU,
                                        query_mem_desc,
                                        executor_->getRowSetMemoryOwner(),
                                        executor_);

  if (logical_values->hasRows()) {
    CHECK_EQ(logical_values->getRowsSize(), logical_values->size());

    auto storage = rs->allocateStorage();
    auto buff = storage->getUnderlyingBuffer();

    for (size_t i = 0; i < logical_values->getNumRows(); i++) {
      std::vector<std::shared_ptr<Analyzer::Expr>> row_literals;
      int8_t* ptr = buff + i * query_mem_desc.getRowSize();

      for (size_t j = 0; j < logical_values->getRowsSize(); j++) {
        auto rex_literal =
            dynamic_cast<const RexLiteral*>(logical_values->getValueAt(i, j));
        CHECK(rex_literal);
        const auto expr = RelAlgTranslator::translateLiteral(rex_literal);
        const auto constant = std::dynamic_pointer_cast<Analyzer::Constant>(expr);
        CHECK(constant);

        if (constant->get_is_null()) {
          CHECK(!target_infos[j].sql_type.is_varlen());
          *reinterpret_cast<int64_t*>(ptr) =
              inline_int_null_val(target_infos[j].sql_type);
        } else {
          const auto ti = constant->get_type_info();
          const auto datum = constant->get_constval();

          // Initialize the entire 8-byte slot
          *reinterpret_cast<int64_t*>(ptr) = EMPTY_KEY_64;

          const auto sz = ti.get_size();
          CHECK_GE(sz, int(0));
          std::memcpy(ptr, &datum, sz);
        }
        ptr += 8;
      }
    }
  }
  return {rs, tuple_type};
}

namespace {

template <class T>
int64_t insert_one_dict_str(T* col_data,
                            const std::string& columnName,
                            const SQLTypeInfo& columnType,
                            const Analyzer::Constant* col_cv,
                            const Catalog_Namespace::Catalog& catalog) {
  if (col_cv->get_is_null()) {
    *col_data = inline_fixed_encoding_null_val(columnType);
  } else {
    const int dict_id = columnType.get_comp_param();
    const auto col_datum = col_cv->get_constval();
    const auto& str = *col_datum.stringval;
    const auto dd = catalog.getMetadataForDict(dict_id);
    CHECK(dd && dd->stringDict);
    int32_t str_id = dd->stringDict->getOrAdd(str);
    if (!dd->dictIsTemp) {
      const auto checkpoint_ok = dd->stringDict->checkpoint();
      if (!checkpoint_ok) {
        throw std::runtime_error("Failed to checkpoint dictionary for column " +
                                 columnName);
      }
    }
    const bool invalid = str_id > max_valid_int_value<T>();
    if (invalid || str_id == inline_int_null_value<int32_t>()) {
      if (invalid) {
        LOG(ERROR) << "Could not encode string: " << str
                   << ", the encoded value doesn't fit in " << sizeof(T) * 8
                   << " bits. Will store NULL instead.";
      }
      str_id = inline_fixed_encoding_null_val(columnType);
    }
    *col_data = str_id;
  }
  return *col_data;
}

template <class T>
int64_t insert_one_dict_str(T* col_data,
                            const ColumnDescriptor* cd,
                            const Analyzer::Constant* col_cv,
                            const Catalog_Namespace::Catalog& catalog) {
  return insert_one_dict_str(col_data, cd->columnName, cd->columnType, col_cv, catalog);
}

}  // namespace

ExecutionResult RelAlgExecutor::executeModify(const RelModify* modify,
                                              const ExecutionOptions& eo) {
  auto timer = DEBUG_TIMER(__func__);
  if (eo.just_explain) {
    throw std::runtime_error("EXPLAIN not supported for ModifyTable");
  }

  auto rs = std::make_shared<ResultSet>(TargetInfoList{},
                                        ExecutorDeviceType::CPU,
                                        QueryMemoryDescriptor(),
                                        executor_->getRowSetMemoryOwner(),
                                        executor_);

  std::vector<TargetMetaInfo> empty_targets;
  return {rs, empty_targets};
}

ExecutionResult RelAlgExecutor::executeSimpleInsert(const Analyzer::Query& query) {
  // Note: We currently obtain an executor for this method, but we do not need it.
  // Therefore, we skip the executor state setup in the regular execution path. In the
  // future, we will likely want to use the executor to evaluate expressions in the insert
  // statement.

  const auto& targets = query.get_targetlist();
  const int table_id = query.get_result_table_id();
  const auto& col_id_list = query.get_result_col_list();

  std::vector<const ColumnDescriptor*> col_descriptors;
  std::vector<int> col_ids;
  std::unordered_map<int, std::unique_ptr<uint8_t[]>> col_buffers;
  std::unordered_map<int, std::vector<std::string>> str_col_buffers;
  std::unordered_map<int, std::vector<ArrayDatum>> arr_col_buffers;

  const auto table_descriptor = cat_.getMetadataForTable(table_id);
  CHECK(table_descriptor);
  const auto shard_tables = cat_.getPhysicalTablesDescriptors(table_descriptor);
  const TableDescriptor* shard{nullptr};

  for (const int col_id : col_id_list) {
    const auto cd = get_column_descriptor(col_id, table_id, cat_);
    const auto col_enc = cd->columnType.get_compression();
    if (cd->columnType.is_string()) {
      switch (col_enc) {
        case kENCODING_NONE: {
          auto it_ok =
              str_col_buffers.insert(std::make_pair(col_id, std::vector<std::string>{}));
          CHECK(it_ok.second);
          break;
        }
        case kENCODING_DICT: {
          const auto dd = cat_.getMetadataForDict(cd->columnType.get_comp_param());
          CHECK(dd);
          const auto it_ok = col_buffers.emplace(
              col_id, std::make_unique<uint8_t[]>(cd->columnType.get_size()));
          CHECK(it_ok.second);
          break;
        }
        default:
          CHECK(false);
      }
    } else if (cd->columnType.is_geometry()) {
      auto it_ok =
          str_col_buffers.insert(std::make_pair(col_id, std::vector<std::string>{}));
      CHECK(it_ok.second);
    } else if (cd->columnType.is_array()) {
      auto it_ok =
          arr_col_buffers.insert(std::make_pair(col_id, std::vector<ArrayDatum>{}));
      CHECK(it_ok.second);
    } else {
      const auto it_ok = col_buffers.emplace(
          col_id,
          std::unique_ptr<uint8_t[]>(
              new uint8_t[cd->columnType.get_logical_size()]()));  // changed to zero-init
                                                                   // the buffer
      CHECK(it_ok.second);
    }
    col_descriptors.push_back(cd);
    col_ids.push_back(col_id);
  }
  size_t col_idx = 0;
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = cat_.getCurrentDB().dbId;
  insert_data.tableId = table_id;
  int64_t int_col_val{0};
  for (auto target_entry : targets) {
    auto col_cv = dynamic_cast<const Analyzer::Constant*>(target_entry->get_expr());
    if (!col_cv) {
      auto col_cast = dynamic_cast<const Analyzer::UOper*>(target_entry->get_expr());
      CHECK(col_cast);
      CHECK_EQ(kCAST, col_cast->get_optype());
      col_cv = dynamic_cast<const Analyzer::Constant*>(col_cast->get_operand());
    }
    CHECK(col_cv);
    const auto cd = col_descriptors[col_idx];
    auto col_datum = col_cv->get_constval();
    auto col_type = cd->columnType.get_type();
    uint8_t* col_data_bytes{nullptr};
    if (!cd->columnType.is_array() && !cd->columnType.is_geometry() &&
        (!cd->columnType.is_string() ||
         cd->columnType.get_compression() == kENCODING_DICT)) {
      const auto col_data_bytes_it = col_buffers.find(col_ids[col_idx]);
      CHECK(col_data_bytes_it != col_buffers.end());
      col_data_bytes = col_data_bytes_it->second.get();
    }
    switch (col_type) {
      case kBOOLEAN: {
        auto col_data = col_data_bytes;
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : (col_datum.boolval ? 1 : 0);
        break;
      }
      case kTINYINT: {
        auto col_data = reinterpret_cast<int8_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.tinyintval;
        int_col_val = col_datum.tinyintval;
        break;
      }
      case kSMALLINT: {
        auto col_data = reinterpret_cast<int16_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.smallintval;
        int_col_val = col_datum.smallintval;
        break;
      }
      case kINT: {
        auto col_data = reinterpret_cast<int32_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.intval;
        int_col_val = col_datum.intval;
        break;
      }
      case kBIGINT:
      case kDECIMAL:
      case kNUMERIC: {
        auto col_data = reinterpret_cast<int64_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.bigintval;
        int_col_val = col_datum.bigintval;
        break;
      }
      case kFLOAT: {
        auto col_data = reinterpret_cast<float*>(col_data_bytes);
        *col_data = col_datum.floatval;
        break;
      }
      case kDOUBLE: {
        auto col_data = reinterpret_cast<double*>(col_data_bytes);
        *col_data = col_datum.doubleval;
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        switch (cd->columnType.get_compression()) {
          case kENCODING_NONE:
            str_col_buffers[col_ids[col_idx]].push_back(
                col_datum.stringval ? *col_datum.stringval : "");
            break;
          case kENCODING_DICT: {
            switch (cd->columnType.get_size()) {
              case 1:
                int_col_val = insert_one_dict_str(
                    reinterpret_cast<uint8_t*>(col_data_bytes), cd, col_cv, cat_);
                break;
              case 2:
                int_col_val = insert_one_dict_str(
                    reinterpret_cast<uint16_t*>(col_data_bytes), cd, col_cv, cat_);
                break;
              case 4:
                int_col_val = insert_one_dict_str(
                    reinterpret_cast<int32_t*>(col_data_bytes), cd, col_cv, cat_);
                break;
              default:
                CHECK(false);
            }
            break;
          }
          default:
            CHECK(false);
        }
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        auto col_data = reinterpret_cast<int64_t*>(col_data_bytes);
        *col_data = col_cv->get_is_null() ? inline_fixed_encoding_null_val(cd->columnType)
                                          : col_datum.bigintval;
        break;
      }
      case kARRAY: {
        const auto is_null = col_cv->get_is_null();
        const auto size = cd->columnType.get_size();
        const SQLTypeInfo elem_ti = cd->columnType.get_elem_type();
        // POINT coords: [un]compressed coords always need to be encoded, even if NULL
        const auto is_point_coords = (cd->isGeoPhyCol && elem_ti.get_type() == kTINYINT);
        if (is_null && !is_point_coords) {
          if (size > 0) {
            // NULL fixlen array: NULL_ARRAY sentinel followed by NULL sentinels
            if (elem_ti.is_string() && elem_ti.get_compression() == kENCODING_DICT) {
              throw std::runtime_error("Column " + cd->columnName +
                                       " doesn't accept NULL values");
            }
            int8_t* buf = (int8_t*)checked_malloc(size);
            put_null_array(static_cast<void*>(buf), elem_ti, "");
            for (int8_t* p = buf + elem_ti.get_size(); (p - buf) < size;
                 p += elem_ti.get_size()) {
              put_null(static_cast<void*>(p), elem_ti, "");
            }
            arr_col_buffers[col_ids[col_idx]].emplace_back(size, buf, is_null);
          } else {
            arr_col_buffers[col_ids[col_idx]].emplace_back(0, nullptr, is_null);
          }
          break;
        }
        const auto l = col_cv->get_value_list();
        size_t len = l.size() * elem_ti.get_size();
        if (size > 0 && static_cast<size_t>(size) != len) {
          throw std::runtime_error("Array column " + cd->columnName + " expects " +
                                   std::to_string(size / elem_ti.get_size()) +
                                   " values, " + "received " + std::to_string(l.size()));
        }
        if (elem_ti.is_string()) {
          CHECK(kENCODING_DICT == elem_ti.get_compression());
          CHECK(4 == elem_ti.get_size());

          int8_t* buf = (int8_t*)checked_malloc(len);
          int32_t* p = reinterpret_cast<int32_t*>(buf);

          int elemIndex = 0;
          for (auto& e : l) {
            auto c = std::dynamic_pointer_cast<Analyzer::Constant>(e);
            CHECK(c);

            int_col_val = insert_one_dict_str(
                &p[elemIndex], cd->columnName, elem_ti, c.get(), cat_);

            elemIndex++;
          }
          arr_col_buffers[col_ids[col_idx]].push_back(ArrayDatum(len, buf, is_null));

        } else {
          int8_t* buf = (int8_t*)checked_malloc(len);
          int8_t* p = buf;
          for (auto& e : l) {
            auto c = std::dynamic_pointer_cast<Analyzer::Constant>(e);
            CHECK(c);
            p = appendDatum(p, c->get_constval(), elem_ti);
          }
          arr_col_buffers[col_ids[col_idx]].push_back(ArrayDatum(len, buf, is_null));
        }
        break;
      }
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
        str_col_buffers[col_ids[col_idx]].push_back(
            col_datum.stringval ? *col_datum.stringval : "");
        break;
      default:
        CHECK(false);
    }
    ++col_idx;
    if (col_idx == static_cast<size_t>(table_descriptor->shardedColumnId)) {
      const auto shard_count = shard_tables.size();
      const size_t shard_idx = SHARD_FOR_KEY(int_col_val, shard_count);
      shard = shard_tables[shard_idx];
    }
  }
  for (const auto& kv : col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.numbersPtr = reinterpret_cast<int8_t*>(kv.second.get());
    insert_data.data.push_back(p);
  }
  for (auto& kv : str_col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.stringsPtr = &kv.second;
    insert_data.data.push_back(p);
  }
  for (auto& kv : arr_col_buffers) {
    insert_data.columnIds.push_back(kv.first);
    DataBlockPtr p;
    p.arraysPtr = &kv.second;
    insert_data.data.push_back(p);
  }
  insert_data.numRows = 1;
  if (shard) {
    shard->fragmenter->insertData(insert_data);
  } else {
    table_descriptor->fragmenter->insertData(insert_data);
  }

  auto rs = std::make_shared<ResultSet>(TargetInfoList{},
                                        ExecutorDeviceType::CPU,
                                        QueryMemoryDescriptor(),
                                        executor_->getRowSetMemoryOwner(),
                                        executor_);
  std::vector<TargetMetaInfo> empty_targets;
  return {rs, empty_targets};
}

namespace {

// TODO(alex): Once we're fully migrated to the relational algebra model, change
// the executor interface to use the collation directly and remove this conversion.
std::list<Analyzer::OrderEntry> get_order_entries(const RelSort* sort) {
  std::list<Analyzer::OrderEntry> result;
  for (size_t i = 0; i < sort->collationCount(); ++i) {
    const auto sort_field = sort->getCollation(i);
    result.emplace_back(sort_field.getField() + 1,
                        sort_field.getSortDir() == SortDirection::Descending,
                        sort_field.getNullsPosition() == NullSortedPosition::First);
  }
  return result;
}

size_t get_scan_limit(const RelAlgNode* ra, const size_t limit) {
  const auto aggregate = dynamic_cast<const RelAggregate*>(ra);
  if (aggregate) {
    return 0;
  }
  const auto compound = dynamic_cast<const RelCompound*>(ra);
  return (compound && compound->isAggregate()) ? 0 : limit;
}

bool first_oe_is_desc(const std::list<Analyzer::OrderEntry>& order_entries) {
  return !order_entries.empty() && order_entries.front().is_desc;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeSort(const RelSort* sort,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo,
                                            RenderInfo* render_info,
                                            const int64_t queue_time_ms) {
  auto timer = DEBUG_TIMER(__func__);
  check_sort_node_source_constraint(sort);
  const auto source = sort->getInput(0);
  const bool is_aggregate = node_is_aggregate(source);
  auto it = leaf_results_.find(sort->getId());
  if (it != leaf_results_.end()) {
    // Add any transient string literals to the sdp on the agg
    const auto source_work_unit = createSortInputWorkUnit(sort, eo);
    GroupByAndAggregate::addTransientStringLiterals(
        source_work_unit.exe_unit, executor_, executor_->row_set_mem_owner_);
    // Handle push-down for LIMIT for multi-node
    auto& aggregated_result = it->second;
    auto& result_rows = aggregated_result.rs;
    const size_t limit = sort->getLimit();
    const size_t offset = sort->getOffset();
    const auto order_entries = get_order_entries(sort);
    if (limit || offset) {
      if (!order_entries.empty()) {
        result_rows->sort(order_entries, limit + offset);
      }
      result_rows->dropFirstN(offset);
      if (limit) {
        result_rows->keepFirstN(limit);
      }
    }
    ExecutionResult result(result_rows, aggregated_result.targets_meta);
    sort->setOutputMetainfo(aggregated_result.targets_meta);
    return result;
  }
  while (true) {
    std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
    bool is_desc{false};
    try {
      const auto source_work_unit = createSortInputWorkUnit(sort, eo);
      is_desc = first_oe_is_desc(source_work_unit.exe_unit.sort_info.order_entries);
      ExecutionOptions eo_copy = {
          eo.output_columnar_hint,
          eo.allow_multifrag,
          eo.just_explain,
          eo.allow_loop_joins,
          eo.with_watchdog,
          eo.jit_debug,
          eo.just_validate || sort->isEmptyResult(),
          eo.with_dynamic_watchdog,
          eo.dynamic_watchdog_time_limit,
          eo.find_push_down_candidates,
          eo.just_calcite_explain,
          eo.gpu_input_mem_limit_percent,
          eo.allow_runtime_query_interrupt,
          eo.runtime_query_interrupt_frequency,
          eo.executor_type,
      };

      groupby_exprs = source_work_unit.exe_unit.groupby_exprs;
      auto source_result = executeWorkUnit(source_work_unit,
                                           source->getOutputMetainfo(),
                                           is_aggregate,
                                           co,
                                           eo_copy,
                                           render_info,
                                           queue_time_ms);
      if (render_info && render_info->isPotentialInSituRender()) {
        return source_result;
      }
      if (source_result.isFilterPushDownEnabled()) {
        return source_result;
      }
      auto rows_to_sort = source_result.getRows();
      if (eo.just_explain) {
        return {rows_to_sort, {}};
      }
      const size_t limit = sort->getLimit();
      const size_t offset = sort->getOffset();
      if (sort->collationCount() != 0 && !rows_to_sort->definitelyHasNoRows() &&
          !use_speculative_top_n(source_work_unit.exe_unit,
                                 rows_to_sort->getQueryMemDesc())) {
        rows_to_sort->sort(source_work_unit.exe_unit.sort_info.order_entries,
                           limit + offset);
      }
      if (limit || offset) {
        if (g_cluster && sort->collationCount() == 0) {
          if (offset >= rows_to_sort->rowCount()) {
            rows_to_sort->dropFirstN(offset);
          } else {
            rows_to_sort->keepFirstN(limit + offset);
          }
        } else {
          rows_to_sort->dropFirstN(offset);
          if (limit) {
            rows_to_sort->keepFirstN(limit);
          }
        }
      }
      return {rows_to_sort, source_result.getTargetsMeta()};
    } catch (const SpeculativeTopNFailed&) {
      CHECK_EQ(size_t(1), groupby_exprs.size());
      speculative_topn_blacklist_.add(groupby_exprs.front(), is_desc);
    }
  }
  CHECK(false);
  return {std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                      co.device_type,
                                      QueryMemoryDescriptor(),
                                      nullptr,
                                      executor_),
          {}};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createSortInputWorkUnit(
    const RelSort* sort,
    const ExecutionOptions& eo) {
  const auto source = sort->getInput(0);
  const size_t limit = sort->getLimit();
  const size_t offset = sort->getOffset();
  const size_t scan_limit = sort->collationCount() ? 0 : get_scan_limit(source, limit);
  const size_t scan_total_limit =
      scan_limit ? get_scan_limit(source, scan_limit + offset) : 0;
  size_t max_groups_buffer_entry_guess{
      scan_total_limit ? scan_total_limit : max_groups_buffer_entry_default_guess};
  SortAlgorithm sort_algorithm{SortAlgorithm::SpeculativeTopN};
  const auto order_entries = get_order_entries(sort);
  SortInfo sort_info{order_entries, sort_algorithm, limit, offset};
  auto source_work_unit = createWorkUnit(source, sort_info, eo);
  const auto& source_exe_unit = source_work_unit.exe_unit;
  if (source_exe_unit.groupby_exprs.size() == 1) {
    if (!source_exe_unit.groupby_exprs.front()) {
      sort_algorithm = SortAlgorithm::StreamingTopN;
    } else {
      if (speculative_topn_blacklist_.contains(source_exe_unit.groupby_exprs.front(),
                                               first_oe_is_desc(order_entries))) {
        sort_algorithm = SortAlgorithm::Default;
      }
    }
  }

  sort->setOutputMetainfo(source->getOutputMetainfo());
  // NB: the `body` field of the returned `WorkUnit` needs to be the `source` node,
  // not the `sort`. The aggregator needs the pre-sorted result from leaves.
  return {RelAlgExecutionUnit{source_exe_unit.input_descs,
                              std::move(source_exe_unit.input_col_descs),
                              source_exe_unit.simple_quals,
                              source_exe_unit.quals,
                              source_exe_unit.join_quals,
                              source_exe_unit.groupby_exprs,
                              source_exe_unit.target_exprs,
                              nullptr,
                              {sort_info.order_entries, sort_algorithm, limit, offset},
                              scan_total_limit,
                              source_exe_unit.query_features,
                              source_exe_unit.use_bump_allocator,
                              source_exe_unit.query_state},
          source,
          max_groups_buffer_entry_guess,
          std::move(source_work_unit.query_rewriter),
          source_work_unit.input_permutation,
          source_work_unit.left_deep_join_input_sizes};
}

namespace {

/**
 *  Upper bound estimation for the number of groups. Not strictly correct and not tight,
 * but if the tables involved are really small we shouldn't waste time doing the NDV
 * estimation. We don't account for cross-joins and / or group by unnested array, which
 * is the reason this estimation isn't entirely reliable.
 */
size_t groups_approx_upper_bound(const std::vector<InputTableInfo>& table_infos) {
  CHECK(!table_infos.empty());
  const auto& first_table = table_infos.front();
  size_t max_num_groups = first_table.info.getNumTuplesUpperBound();
  for (const auto& table_info : table_infos) {
    if (table_info.info.getNumTuplesUpperBound() > max_num_groups) {
      max_num_groups = table_info.info.getNumTuplesUpperBound();
    }
  }
  return std::max(max_num_groups, size_t(1));
}

/**
 * Determines whether a query needs to compute the size of its output buffer. Returns
 * true for projection queries with no LIMIT or a LIMIT that exceeds the high scan limit
 * threshold (meaning it would be cheaper to compute the number of rows passing or use
 * the bump allocator than allocate the current scan limit per GPU)
 */
bool compute_output_buffer_size(const RelAlgExecutionUnit& ra_exe_unit) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return false;
    }
  }
  if (ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front() &&
      (!ra_exe_unit.scan_limit || ra_exe_unit.scan_limit > Executor::high_scan_limit)) {
    return true;
  }
  return false;
}

inline bool exe_unit_has_quals(const RelAlgExecutionUnit ra_exe_unit) {
  return !(ra_exe_unit.quals.empty() && ra_exe_unit.join_quals.empty() &&
           ra_exe_unit.simple_quals.empty());
}

RelAlgExecutionUnit decide_approx_count_distinct_implementation(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const std::vector<InputTableInfo>& table_infos,
    const Executor* executor,
    const ExecutorDeviceType device_type_in,
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned) {
  RelAlgExecutionUnit ra_exe_unit = ra_exe_unit_in;
  for (size_t i = 0; i < ra_exe_unit.target_exprs.size(); ++i) {
    const auto target_expr = ra_exe_unit.target_exprs[i];
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    if (agg_info.agg_kind != kAPPROX_COUNT_DISTINCT) {
      continue;
    }
    CHECK(dynamic_cast<const Analyzer::AggExpr*>(target_expr));
    const auto arg = static_cast<Analyzer::AggExpr*>(target_expr)->get_own_arg();
    CHECK(arg);
    const auto& arg_ti = arg->get_type_info();
    // Avoid calling getExpressionRange for variable length types (string and array),
    // it'd trigger an assertion since that API expects to be called only for types
    // for which the notion of range is well-defined. A bit of a kludge, but the
    // logic to reject these types anyway is at lower levels in the stack and not
    // really worth pulling into a separate function for now.
    if (!(arg_ti.is_number() || arg_ti.is_boolean() || arg_ti.is_time() ||
          (arg_ti.is_string() && arg_ti.get_compression() == kENCODING_DICT))) {
      continue;
    }
    const auto arg_range = getExpressionRange(arg.get(), table_infos, executor);
    if (arg_range.getType() != ExpressionRangeType::Integer) {
      continue;
    }
    // When running distributed, the threshold for using the precise implementation
    // must be consistent across all leaves, otherwise we could have a mix of precise
    // and approximate bitmaps and we cannot aggregate them.
    const auto device_type = g_cluster ? ExecutorDeviceType::GPU : device_type_in;
    const auto bitmap_sz_bits = arg_range.getIntMax() - arg_range.getIntMin() + 1;
    const auto sub_bitmap_count =
        get_count_distinct_sub_bitmap_count(bitmap_sz_bits, ra_exe_unit, device_type);
    int64_t approx_bitmap_sz_bits{0};
    const auto error_rate =
        static_cast<Analyzer::AggExpr*>(target_expr)->get_error_rate();
    if (error_rate) {
      CHECK(error_rate->get_type_info().get_type() == kINT);
      CHECK_GE(error_rate->get_constval().intval, 1);
      approx_bitmap_sz_bits = hll_size_for_rate(error_rate->get_constval().intval);
    } else {
      approx_bitmap_sz_bits = g_hll_precision_bits;
    }
    CountDistinctDescriptor approx_count_distinct_desc{CountDistinctImplType::Bitmap,
                                                       arg_range.getIntMin(),
                                                       approx_bitmap_sz_bits,
                                                       true,
                                                       device_type,
                                                       sub_bitmap_count};
    CountDistinctDescriptor precise_count_distinct_desc{CountDistinctImplType::Bitmap,
                                                        arg_range.getIntMin(),
                                                        bitmap_sz_bits,
                                                        false,
                                                        device_type,
                                                        sub_bitmap_count};
    if (approx_count_distinct_desc.bitmapPaddedSizeBytes() >=
        precise_count_distinct_desc.bitmapPaddedSizeBytes()) {
      auto precise_count_distinct = makeExpr<Analyzer::AggExpr>(
          get_agg_type(kCOUNT, arg.get()), kCOUNT, arg, true, nullptr);
      target_exprs_owned.push_back(precise_count_distinct);
      ra_exe_unit.target_exprs[i] = precise_count_distinct.get();
    }
  }
  return ra_exe_unit;
}

void build_render_targets(RenderInfo& render_info,
                          const std::vector<Analyzer::Expr*>& work_unit_target_exprs,
                          const std::vector<TargetMetaInfo>& targets_meta) {
  CHECK_EQ(work_unit_target_exprs.size(), targets_meta.size());
  render_info.targets.clear();
  for (size_t i = 0; i < targets_meta.size(); ++i) {
    render_info.targets.emplace_back(std::make_shared<Analyzer::TargetEntry>(
        targets_meta[i].get_resname(),
        work_unit_target_exprs[i]->get_shared_ptr(),
        false));
  }
}

inline bool can_use_bump_allocator(const RelAlgExecutionUnit& ra_exe_unit,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo) {
  return g_enable_bump_allocator && (co.device_type == ExecutorDeviceType::GPU) &&
         !eo.output_columnar_hint && ra_exe_unit.sort_info.order_entries.empty();
}

}  // namespace

ExecutionResult RelAlgExecutor::executeWorkUnit(
    const RelAlgExecutor::WorkUnit& work_unit,
    const std::vector<TargetMetaInfo>& targets_meta,
    const bool is_agg,
    const CompilationOptions& co_in,
    const ExecutionOptions& eo,
    RenderInfo* render_info,
    const int64_t queue_time_ms,
    const ssize_t previous_count) {
  INJECT_TIMER(executeWorkUnit);
  auto timer = DEBUG_TIMER(__func__);

  auto co = co_in;
  ColumnCacheMap column_cache;
  if (is_window_execution_unit(work_unit.exe_unit)) {
    if (!g_enable_window_functions) {
      throw std::runtime_error("Window functions support is disabled");
    }
    co.device_type = ExecutorDeviceType::CPU;
    computeWindow(work_unit.exe_unit, co, eo, column_cache, queue_time_ms);
  }
  if (!eo.just_explain && eo.find_push_down_candidates) {
    // find potential candidates:
    auto selected_filters = selectFiltersToBePushedDown(work_unit, co, eo);
    if (!selected_filters.empty() || eo.just_calcite_explain) {
      return ExecutionResult(selected_filters, eo.find_push_down_candidates);
    }
  }
  if (render_info && render_info->isPotentialInSituRender()) {
    co.allow_lazy_fetch = false;
  }
  const auto body = work_unit.body;
  CHECK(body);
  auto it = leaf_results_.find(body->getId());
  if (it != leaf_results_.end()) {
    GroupByAndAggregate::addTransientStringLiterals(
        work_unit.exe_unit, executor_, executor_->row_set_mem_owner_);
    auto& aggregated_result = it->second;
    auto& result_rows = aggregated_result.rs;
    ExecutionResult result(result_rows, aggregated_result.targets_meta);
    body->setOutputMetainfo(aggregated_result.targets_meta);
    if (render_info) {
      build_render_targets(*render_info, work_unit.exe_unit.target_exprs, targets_meta);
    }
    return result;
  }
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);

  auto ra_exe_unit = decide_approx_count_distinct_implementation(
      work_unit.exe_unit, table_infos, executor_, co.device_type, target_exprs_owned_);
  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  if (is_window_execution_unit(ra_exe_unit)) {
    CHECK_EQ(table_infos.size(), size_t(1));
    CHECK_EQ(table_infos.front().info.fragments.size(), size_t(1));
    max_groups_buffer_entry_guess =
        table_infos.front().info.fragments.front().getNumTuples();
    ra_exe_unit.scan_limit = max_groups_buffer_entry_guess;
  } else if (compute_output_buffer_size(ra_exe_unit) && !isRowidLookup(work_unit)) {
    if (previous_count > 0 && !exe_unit_has_quals(ra_exe_unit)) {
      ra_exe_unit.scan_limit = static_cast<size_t>(previous_count);
    } else {
      // TODO(adb): enable bump allocator path for render queries
      if (can_use_bump_allocator(ra_exe_unit, co, eo) && !render_info) {
        ra_exe_unit.scan_limit = 0;
        ra_exe_unit.use_bump_allocator = true;
      } else if (eo.executor_type == ::ExecutorType::Extern) {
        ra_exe_unit.scan_limit = 0;
      } else if (!eo.just_explain) {
        const auto filter_count_all = getFilteredCountAll(work_unit, true, co, eo);
        if (filter_count_all >= 0) {
          ra_exe_unit.scan_limit = std::max(filter_count_all, ssize_t(1));
        }
      }
    }
  }

  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     co.device_type,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     executor_),
                         {}};

  auto execute_and_handle_errors =
      [&](const auto max_groups_buffer_entry_guess_in,
          const bool has_cardinality_estimation) -> ExecutionResult {
    // Note that the groups buffer entry guess may be modified during query execution.
    // Create a local copy so we can track those changes if we need to attempt a retry
    // due to OOM
    auto local_groups_buffer_entry_guess = max_groups_buffer_entry_guess_in;
    try {
      return {executor_->executeWorkUnit(local_groups_buffer_entry_guess,
                                         is_agg,
                                         table_infos,
                                         ra_exe_unit,
                                         co,
                                         eo,
                                         cat_,
                                         executor_->row_set_mem_owner_,
                                         render_info,
                                         has_cardinality_estimation,
                                         column_cache),
              targets_meta};
    } catch (const QueryExecutionError& e) {
      handlePersistentError(e.getErrorCode());
      return handleOutOfMemoryRetry(
          {ra_exe_unit, work_unit.body, local_groups_buffer_entry_guess},
          targets_meta,
          is_agg,
          co,
          eo,
          render_info,
          e.wasMultifragKernelLaunch(),
          queue_time_ms);
    }
  };

  try {
    result = execute_and_handle_errors(
        max_groups_buffer_entry_guess,
        groups_approx_upper_bound(table_infos) <= g_big_group_threshold);
  } catch (const CardinalityEstimationRequired&) {
    const auto estimated_groups_buffer_entry_guess =
        2 * std::min(groups_approx_upper_bound(table_infos),
                     getNDVEstimation(work_unit, is_agg, co, eo));
    CHECK_GT(estimated_groups_buffer_entry_guess, size_t(0));
    result = execute_and_handle_errors(estimated_groups_buffer_entry_guess, true);
  }

  result.setQueueTime(queue_time_ms);
  if (render_info) {
    build_render_targets(*render_info, work_unit.exe_unit.target_exprs, targets_meta);
    if (render_info->isPotentialInSituRender()) {
      // return an empty result (with the same queue time, and zero render time)
      return {std::make_shared<ResultSet>(
                  queue_time_ms,
                  0,
                  executor_->row_set_mem_owner_
                      ? executor_->row_set_mem_owner_->cloneStrDictDataOnly()
                      : nullptr),
              {}};
    }
  }
  return result;
}

ssize_t RelAlgExecutor::getFilteredCountAll(const WorkUnit& work_unit,
                                            const bool is_agg,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo) {
  const auto count =
      makeExpr<Analyzer::AggExpr>(SQLTypeInfo(g_bigint_count ? kBIGINT : kINT, false),
                                  kCOUNT,
                                  nullptr,
                                  false,
                                  nullptr);
  const auto count_all_exe_unit =
      create_count_all_execution_unit(work_unit.exe_unit, count);
  size_t one{1};
  ResultSetPtr count_all_result;
  try {
    ColumnCacheMap column_cache;
    count_all_result =
        executor_->executeWorkUnit(one,
                                   is_agg,
                                   get_table_infos(work_unit.exe_unit, executor_),
                                   count_all_exe_unit,
                                   co,
                                   eo,
                                   cat_,
                                   executor_->row_set_mem_owner_,
                                   nullptr,
                                   false,
                                   column_cache);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to run pre-flight filtered count with error " << e.what();
    return -1;
  }
  const auto count_row = count_all_result->getNextRow(false, false);
  CHECK_EQ(size_t(1), count_row.size());
  const auto& count_tv = count_row.front();
  const auto count_scalar_tv = boost::get<ScalarTargetValue>(&count_tv);
  CHECK(count_scalar_tv);
  const auto count_ptr = boost::get<int64_t>(count_scalar_tv);
  CHECK(count_ptr);
  CHECK_GE(*count_ptr, 0);
  auto count_upper_bound = static_cast<size_t>(*count_ptr);
  return std::max(count_upper_bound, size_t(1));
}

bool RelAlgExecutor::isRowidLookup(const WorkUnit& work_unit) {
  const auto& ra_exe_unit = work_unit.exe_unit;
  if (ra_exe_unit.input_descs.size() != 1) {
    return false;
  }
  const auto& table_desc = ra_exe_unit.input_descs.front();
  if (table_desc.getSourceType() != InputSourceType::TABLE) {
    return false;
  }
  const int table_id = table_desc.getTableId();
  for (const auto simple_qual : ra_exe_unit.simple_quals) {
    const auto comp_expr =
        std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
    if (!comp_expr || comp_expr->get_optype() != kEQ) {
      return false;
    }
    const auto lhs = comp_expr->get_left_operand();
    const auto lhs_col = dynamic_cast<const Analyzer::ColumnVar*>(lhs);
    if (!lhs_col || !lhs_col->get_table_id() || lhs_col->get_rte_idx()) {
      return false;
    }
    const auto rhs = comp_expr->get_right_operand();
    const auto rhs_const = dynamic_cast<const Analyzer::Constant*>(rhs);
    if (!rhs_const) {
      return false;
    }
    auto cd = get_column_descriptor(lhs_col->get_column_id(), table_id, cat_);
    if (cd->isVirtualCol) {
      CHECK_EQ("rowid", cd->columnName);
      return true;
    }
  }
  return false;
}

ExecutionResult RelAlgExecutor::handleOutOfMemoryRetry(
    const RelAlgExecutor::WorkUnit& work_unit,
    const std::vector<TargetMetaInfo>& targets_meta,
    const bool is_agg,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    RenderInfo* render_info,
    const bool was_multifrag_kernel_launch,
    const int64_t queue_time_ms) {
  // Disable the bump allocator
  // Note that this will have basically the same affect as using the bump allocator for
  // the kernel per fragment path. Need to unify the max_groups_buffer_entry_guess = 0
  // path and the bump allocator path for kernel per fragment execution.
  auto ra_exe_unit_in = work_unit.exe_unit;
  ra_exe_unit_in.use_bump_allocator = false;

  auto result = ExecutionResult{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                            co.device_type,
                                                            QueryMemoryDescriptor(),
                                                            nullptr,
                                                            executor_),
                                {}};

  const auto table_infos = get_table_infos(ra_exe_unit_in, executor_);
  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  ExecutionOptions eo_no_multifrag{eo.output_columnar_hint,
                                   false,
                                   false,
                                   eo.allow_loop_joins,
                                   eo.with_watchdog,
                                   eo.jit_debug,
                                   false,
                                   eo.with_dynamic_watchdog,
                                   eo.dynamic_watchdog_time_limit,
                                   false,
                                   false,
                                   eo.gpu_input_mem_limit_percent,
                                   false};

  if (was_multifrag_kernel_launch) {
    try {
      // Attempt to retry using the kernel per fragment path. The smaller input size
      // required may allow the entire kernel to execute in GPU memory.
      LOG(WARNING) << "Multifrag query ran out of memory, retrying with multifragment "
                      "kernels disabled.";
      const auto ra_exe_unit = decide_approx_count_distinct_implementation(
          ra_exe_unit_in, table_infos, executor_, co.device_type, target_exprs_owned_);
      ColumnCacheMap column_cache;
      result = {executor_->executeWorkUnit(max_groups_buffer_entry_guess,
                                           is_agg,
                                           table_infos,
                                           ra_exe_unit,
                                           co,
                                           eo_no_multifrag,
                                           cat_,
                                           executor_->row_set_mem_owner_,
                                           nullptr,
                                           true,
                                           column_cache),
                targets_meta};
      result.setQueueTime(queue_time_ms);
    } catch (const QueryExecutionError& e) {
      handlePersistentError(e.getErrorCode());
      LOG(WARNING) << "Kernel per fragment query ran out of memory, retrying on CPU.";
    }
  }

  if (render_info) {
    render_info->setForceNonInSituData();
  }

  const auto co_cpu = CompilationOptions::makeCpuOnly(co);
  // Only reset the group buffer entry guess if we ran out of slots, which
  // suggests a
  // highly pathological input which prevented a good estimation of distinct tuple
  // count. For projection queries, this will force a per-fragment scan limit, which is
  // compatible with the CPU path
  VLOG(1) << "Resetting max groups buffer entry guess.";
  max_groups_buffer_entry_guess = 0;

  int iteration_ctr = -1;
  while (true) {
    iteration_ctr++;
    auto ra_exe_unit = decide_approx_count_distinct_implementation(
        ra_exe_unit_in, table_infos, executor_, co_cpu.device_type, target_exprs_owned_);
    ColumnCacheMap column_cache;
    try {
      result = {executor_->executeWorkUnit(max_groups_buffer_entry_guess,
                                           is_agg,
                                           table_infos,
                                           ra_exe_unit,
                                           co_cpu,
                                           eo_no_multifrag,
                                           cat_,
                                           executor_->row_set_mem_owner_,
                                           nullptr,
                                           true,
                                           column_cache),
                targets_meta};
    } catch (const QueryExecutionError& e) {
      // Ran out of slots
      if (e.getErrorCode() < 0) {
        // Even the conservative guess failed; it should only happen when we group
        // by a huge cardinality array. Maybe we should throw an exception instead?
        // Such a heavy query is entirely capable of exhausting all the host memory.
        CHECK(max_groups_buffer_entry_guess);
        // Only allow two iterations of increasingly large entry guesses up to a maximum
        // of 512MB per column per kernel
        if (g_enable_watchdog || iteration_ctr > 1) {
          throw std::runtime_error("Query ran out of output slots in the result");
        }
        max_groups_buffer_entry_guess *= 2;
        LOG(WARNING) << "Query ran out of slots in the output buffer, retrying with max "
                        "groups buffer entry "
                        "guess equal to "
                     << max_groups_buffer_entry_guess;
      } else {
        handlePersistentError(e.getErrorCode());
      }
      continue;
    }
    result.setQueueTime(queue_time_ms);
    return result;
  }
  return result;
}

void RelAlgExecutor::handlePersistentError(const int32_t error_code) {
  LOG(ERROR) << "Query execution failed with error "
             << getErrorMessageFromCode(error_code);
  if (error_code == Executor::ERR_SPECULATIVE_TOP_OOM) {
    throw SpeculativeTopNFailed();
  }
  if (error_code == Executor::ERR_OUT_OF_GPU_MEM) {
    // We ran out of GPU memory, this doesn't count as an error if the query is
    // allowed to continue on CPU because retry on CPU is explicitly allowed through
    // --allow-cpu-retry.
    LOG(INFO) << "Query ran out of GPU memory, attempting punt to CPU";
    if (!g_allow_cpu_retry) {
      throw std::runtime_error(
          "Query ran out of GPU memory, unable to automatically retry on CPU");
    }
    return;
  }
  throw std::runtime_error(getErrorMessageFromCode(error_code));
}

std::string RelAlgExecutor::getErrorMessageFromCode(const int32_t error_code) {
  if (error_code < 0) {
    return "Ran out of slots in the query output buffer";
  }
  switch (error_code) {
    case Executor::ERR_DIV_BY_ZERO:
      return "Division by zero";
    case Executor::ERR_OUT_OF_GPU_MEM:
      return "Query couldn't keep the entire working set of columns in GPU memory";
    case Executor::ERR_UNSUPPORTED_SELF_JOIN:
      return "Self joins not supported yet";
    case Executor::ERR_OUT_OF_CPU_MEM:
      return "Not enough host memory to execute the query";
    case Executor::ERR_OVERFLOW_OR_UNDERFLOW:
      return "Overflow or underflow";
    case Executor::ERR_OUT_OF_TIME:
      return "Query execution has exceeded the time limit";
    case Executor::ERR_INTERRUPTED:
      return "Query execution has been interrupted";
    case Executor::ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED:
      return "Columnar conversion not supported for variable length types";
    case Executor::ERR_TOO_MANY_LITERALS:
      return "Too many literals in the query";
    case Executor::ERR_STRING_CONST_IN_RESULTSET:
      return "NONE ENCODED String types are not supported as input result set.";
    case Executor::ERR_OUT_OF_RENDER_MEM:
      return "Not enough OpenGL memory to render the query results";
    case Executor::ERR_STREAMING_TOP_N_NOT_SUPPORTED_IN_RENDER_QUERY:
      return "Streaming-Top-N not supported in Render Query";
    case Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES:
      return "Multiple distinct values encountered";
  }
  return "Other error: code " + std::to_string(error_code);
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createWorkUnit(const RelAlgNode* node,
                                                        const SortInfo& sort_info,
                                                        const ExecutionOptions& eo) {
  const auto compound = dynamic_cast<const RelCompound*>(node);
  if (compound) {
    return createCompoundWorkUnit(compound, sort_info, eo);
  }
  const auto project = dynamic_cast<const RelProject*>(node);
  if (project) {
    return createProjectWorkUnit(project, sort_info, eo);
  }
  const auto aggregate = dynamic_cast<const RelAggregate*>(node);
  if (aggregate) {
    return createAggregateWorkUnit(aggregate, sort_info, eo.just_explain);
  }
  const auto filter = dynamic_cast<const RelFilter*>(node);
  CHECK(filter);
  return createFilterWorkUnit(filter, sort_info, eo.just_explain);
}

namespace {

JoinType get_join_type(const RelAlgNode* ra) {
  auto sink = get_data_sink(ra);
  if (auto join = dynamic_cast<const RelJoin*>(sink)) {
    return join->getJoinType();
  }
  if (dynamic_cast<const RelLeftDeepInnerJoin*>(sink)) {
    return JoinType::INNER;
  }

  return JoinType::INVALID;
}

std::unique_ptr<const RexOperator> get_bitwise_equals(const RexScalar* scalar) {
  const auto condition = dynamic_cast<const RexOperator*>(scalar);
  if (!condition || condition->getOperator() != kOR || condition->size() != 2) {
    return nullptr;
  }
  const auto equi_join_condition =
      dynamic_cast<const RexOperator*>(condition->getOperand(0));
  if (!equi_join_condition || equi_join_condition->getOperator() != kEQ) {
    return nullptr;
  }
  const auto both_are_null_condition =
      dynamic_cast<const RexOperator*>(condition->getOperand(1));
  if (!both_are_null_condition || both_are_null_condition->getOperator() != kAND ||
      both_are_null_condition->size() != 2) {
    return nullptr;
  }
  const auto lhs_is_null =
      dynamic_cast<const RexOperator*>(both_are_null_condition->getOperand(0));
  const auto rhs_is_null =
      dynamic_cast<const RexOperator*>(both_are_null_condition->getOperand(1));
  if (!lhs_is_null || !rhs_is_null || lhs_is_null->getOperator() != kISNULL ||
      rhs_is_null->getOperator() != kISNULL) {
    return nullptr;
  }
  CHECK_EQ(size_t(1), lhs_is_null->size());
  CHECK_EQ(size_t(1), rhs_is_null->size());
  CHECK_EQ(size_t(2), equi_join_condition->size());
  const auto eq_lhs = dynamic_cast<const RexInput*>(equi_join_condition->getOperand(0));
  const auto eq_rhs = dynamic_cast<const RexInput*>(equi_join_condition->getOperand(1));
  const auto is_null_lhs = dynamic_cast<const RexInput*>(lhs_is_null->getOperand(0));
  const auto is_null_rhs = dynamic_cast<const RexInput*>(rhs_is_null->getOperand(0));
  if (!eq_lhs || !eq_rhs || !is_null_lhs || !is_null_rhs) {
    return nullptr;
  }
  std::vector<std::unique_ptr<const RexScalar>> eq_operands;
  if (*eq_lhs == *is_null_lhs && *eq_rhs == *is_null_rhs) {
    RexDeepCopyVisitor deep_copy_visitor;
    auto lhs_op_copy = deep_copy_visitor.visit(equi_join_condition->getOperand(0));
    auto rhs_op_copy = deep_copy_visitor.visit(equi_join_condition->getOperand(1));
    eq_operands.emplace_back(lhs_op_copy.release());
    eq_operands.emplace_back(rhs_op_copy.release());
    return boost::make_unique<const RexOperator>(
        kBW_EQ, eq_operands, equi_join_condition->getType());
  }
  return nullptr;
}

std::unique_ptr<const RexOperator> get_bitwise_equals_conjunction(
    const RexScalar* scalar) {
  const auto condition = dynamic_cast<const RexOperator*>(scalar);
  if (condition && condition->getOperator() == kAND) {
    CHECK_GE(condition->size(), size_t(2));
    auto acc = get_bitwise_equals(condition->getOperand(0));
    if (!acc) {
      return nullptr;
    }
    for (size_t i = 1; i < condition->size(); ++i) {
      std::vector<std::unique_ptr<const RexScalar>> and_operands;
      and_operands.emplace_back(std::move(acc));
      and_operands.emplace_back(get_bitwise_equals_conjunction(condition->getOperand(i)));
      acc =
          boost::make_unique<const RexOperator>(kAND, and_operands, condition->getType());
    }
    return acc;
  }
  return get_bitwise_equals(scalar);
}

std::vector<JoinType> left_deep_join_types(const RelLeftDeepInnerJoin* left_deep_join) {
  CHECK_GE(left_deep_join->inputCount(), size_t(2));
  std::vector<JoinType> join_types(left_deep_join->inputCount() - 1, JoinType::INNER);
  for (size_t nesting_level = 1; nesting_level <= left_deep_join->inputCount() - 1;
       ++nesting_level) {
    if (left_deep_join->getOuterCondition(nesting_level)) {
      join_types[nesting_level - 1] = JoinType::LEFT;
    }
  }
  return join_types;
}

template <class RA>
std::vector<size_t> do_table_reordering(
    std::vector<InputDescriptor>& input_descs,
    std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs,
    const JoinQualsPerNestingLevel& left_deep_join_quals,
    std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const RA* node,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor) {
  if (g_cluster) {
    // Disable table reordering in distributed mode. The aggregator does not have enough
    // information to break ties
    return {};
  }
  const auto& cat = *executor->getCatalog();
  for (const auto& table_info : query_infos) {
    if (table_info.table_id < 0) {
      continue;
    }
    const auto td = cat.getMetadataForTable(table_info.table_id);
    CHECK(td);
    if (table_is_replicated(td)) {
      return {};
    }
  }
  const auto input_permutation =
      get_node_input_permutation(left_deep_join_quals, query_infos, executor);
  input_to_nest_level = get_input_nest_levels(node, input_permutation);
  std::tie(input_descs, input_col_descs, std::ignore) =
      get_input_desc(node, input_to_nest_level, input_permutation, cat);
  return input_permutation;
}

std::vector<size_t> get_left_deep_join_input_sizes(
    const RelLeftDeepInnerJoin* left_deep_join) {
  std::vector<size_t> input_sizes;
  for (size_t i = 0; i < left_deep_join->inputCount(); ++i) {
    const auto inputs = get_node_output(left_deep_join->getInput(i));
    input_sizes.push_back(inputs.size());
  }
  return input_sizes;
}

std::list<std::shared_ptr<Analyzer::Expr>> rewrite_quals(
    const std::list<std::shared_ptr<Analyzer::Expr>>& quals) {
  std::list<std::shared_ptr<Analyzer::Expr>> rewritten_quals;
  for (const auto& qual : quals) {
    const auto rewritten_qual = rewrite_expr(qual.get());
    rewritten_quals.push_back(rewritten_qual ? rewritten_qual : qual);
  }
  return rewritten_quals;
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createCompoundWorkUnit(
    const RelCompound* compound,
    const SortInfo& sort_info,
    const ExecutionOptions& eo) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  auto input_to_nest_level = get_input_nest_levels(compound, {});
  std::tie(input_descs, input_col_descs, std::ignore) =
      get_input_desc(compound, input_to_nest_level, {}, cat_);
  const auto query_infos = get_table_infos(input_descs, executor_);
  CHECK_EQ(size_t(1), compound->inputCount());
  const auto left_deep_join =
      dynamic_cast<const RelLeftDeepInnerJoin*>(compound->getInput(0));
  JoinQualsPerNestingLevel left_deep_join_quals;
  const auto join_types = left_deep_join ? left_deep_join_types(left_deep_join)
                                         : std::vector<JoinType>{get_join_type(compound)};
  std::vector<size_t> input_permutation;
  std::vector<size_t> left_deep_join_input_sizes;
  if (left_deep_join) {
    left_deep_join_input_sizes = get_left_deep_join_input_sizes(left_deep_join);
    left_deep_join_quals = translateLeftDeepJoinFilter(
        left_deep_join, input_descs, input_to_nest_level, eo.just_explain);
    if (g_from_table_reordering &&
        std::find(join_types.begin(), join_types.end(), JoinType::LEFT) ==
            join_types.end()) {
      input_permutation = do_table_reordering(input_descs,
                                              input_col_descs,
                                              left_deep_join_quals,
                                              input_to_nest_level,
                                              compound,
                                              query_infos,
                                              executor_);
      input_to_nest_level = get_input_nest_levels(compound, input_permutation);
      std::tie(input_descs, input_col_descs, std::ignore) =
          get_input_desc(compound, input_to_nest_level, input_permutation, cat_);
      left_deep_join_quals = translateLeftDeepJoinFilter(
          left_deep_join, input_descs, input_to_nest_level, eo.just_explain);
    }
  }
  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              eo.just_explain,
                              query_features);
  const auto scalar_sources =
      translate_scalar_sources(compound, translator, eo.executor_type);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals_cf = translate_quals(compound, translator);
  const auto target_exprs = translate_targets(target_exprs_owned_,
                                              scalar_sources,
                                              groupby_exprs,
                                              compound,
                                              translator,
                                              eo.executor_type);
  CHECK_EQ(compound->size(), target_exprs.size());
  const RelAlgExecutionUnit exe_unit = {input_descs,
                                        input_col_descs,
                                        quals_cf.simple_quals,
                                        rewrite_quals(quals_cf.quals),
                                        left_deep_join_quals,
                                        groupby_exprs,
                                        target_exprs,
                                        nullptr,
                                        sort_info,
                                        0,
                                        query_features,
                                        false,
                                        query_state_};
  auto query_rewriter = std::make_unique<QueryRewriter>(query_infos, executor_);
  const auto rewritten_exe_unit = query_rewriter->rewrite(exe_unit);
  const auto targets_meta = get_targets_meta(compound, rewritten_exe_unit.target_exprs);
  compound->setOutputMetainfo(targets_meta);
  return {rewritten_exe_unit,
          compound,
          max_groups_buffer_entry_default_guess,
          std::move(query_rewriter),
          input_permutation,
          left_deep_join_input_sizes};
}

namespace {

std::vector<const RexScalar*> rex_to_conjunctive_form(const RexScalar* qual_expr) {
  CHECK(qual_expr);
  const auto bin_oper = dynamic_cast<const RexOperator*>(qual_expr);
  if (!bin_oper || bin_oper->getOperator() != kAND) {
    return {qual_expr};
  }
  CHECK_GE(bin_oper->size(), size_t(2));
  auto lhs_cf = rex_to_conjunctive_form(bin_oper->getOperand(0));
  for (size_t i = 1; i < bin_oper->size(); ++i) {
    const auto rhs_cf = rex_to_conjunctive_form(bin_oper->getOperand(i));
    lhs_cf.insert(lhs_cf.end(), rhs_cf.begin(), rhs_cf.end());
  }
  return lhs_cf;
}

std::shared_ptr<Analyzer::Expr> build_logical_expression(
    const std::vector<std::shared_ptr<Analyzer::Expr>>& factors,
    const SQLOps sql_op) {
  CHECK(!factors.empty());
  auto acc = factors.front();
  for (size_t i = 1; i < factors.size(); ++i) {
    acc = Parser::OperExpr::normalize(sql_op, kONE, acc, factors[i]);
  }
  return acc;
}

template <class QualsList>
bool list_contains_expression(const QualsList& haystack,
                              const std::shared_ptr<Analyzer::Expr>& needle) {
  for (const auto& qual : haystack) {
    if (*qual == *needle) {
      return true;
    }
  }
  return false;
}

// Transform `(p AND q) OR (p AND r)` to `p AND (q OR r)`. Avoids redundant
// evaluations of `p` and allows use of the original form in joins if `p`
// can be used for hash joins.
std::shared_ptr<Analyzer::Expr> reverse_logical_distribution(
    const std::shared_ptr<Analyzer::Expr>& expr) {
  const auto expr_terms = qual_to_disjunctive_form(expr);
  CHECK_GE(expr_terms.size(), size_t(1));
  const auto& first_term = expr_terms.front();
  const auto first_term_factors = qual_to_conjunctive_form(first_term);
  std::vector<std::shared_ptr<Analyzer::Expr>> common_factors;
  // First, collect the conjunctive components common to all the disjunctive components.
  // Don't do it for simple qualifiers, we only care about expensive or join qualifiers.
  for (const auto& first_term_factor : first_term_factors.quals) {
    bool is_common =
        expr_terms.size() > 1;  // Only report common factors for disjunction.
    for (size_t i = 1; i < expr_terms.size(); ++i) {
      const auto crt_term_factors = qual_to_conjunctive_form(expr_terms[i]);
      if (!list_contains_expression(crt_term_factors.quals, first_term_factor)) {
        is_common = false;
        break;
      }
    }
    if (is_common) {
      common_factors.push_back(first_term_factor);
    }
  }
  if (common_factors.empty()) {
    return expr;
  }
  // Now that the common expressions are known, collect the remaining expressions.
  std::vector<std::shared_ptr<Analyzer::Expr>> remaining_terms;
  for (const auto& term : expr_terms) {
    const auto term_cf = qual_to_conjunctive_form(term);
    std::vector<std::shared_ptr<Analyzer::Expr>> remaining_quals(
        term_cf.simple_quals.begin(), term_cf.simple_quals.end());
    for (const auto& qual : term_cf.quals) {
      if (!list_contains_expression(common_factors, qual)) {
        remaining_quals.push_back(qual);
      }
    }
    if (!remaining_quals.empty()) {
      remaining_terms.push_back(build_logical_expression(remaining_quals, kAND));
    }
  }
  // Reconstruct the expression with the transformation applied.
  const auto common_expr = build_logical_expression(common_factors, kAND);
  if (remaining_terms.empty()) {
    return common_expr;
  }
  const auto remaining_expr = build_logical_expression(remaining_terms, kOR);
  return Parser::OperExpr::normalize(kAND, kONE, common_expr, remaining_expr);
}

}  // namespace

std::list<std::shared_ptr<Analyzer::Expr>> RelAlgExecutor::makeJoinQuals(
    const RexScalar* join_condition,
    const std::vector<JoinType>& join_types,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const bool just_explain) const {
  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              just_explain,
                              query_features);
  const auto rex_condition_cf = rex_to_conjunctive_form(join_condition);
  std::list<std::shared_ptr<Analyzer::Expr>> join_condition_quals;
  for (const auto rex_condition_component : rex_condition_cf) {
    const auto bw_equals = get_bitwise_equals_conjunction(rex_condition_component);
    const auto join_condition =
        reverse_logical_distribution(translator.translateScalarRex(
            bw_equals ? bw_equals.get() : rex_condition_component));
    auto join_condition_cf = qual_to_conjunctive_form(join_condition);
    join_condition_quals.insert(join_condition_quals.end(),
                                join_condition_cf.quals.begin(),
                                join_condition_cf.quals.end());
    join_condition_quals.insert(join_condition_quals.end(),
                                join_condition_cf.simple_quals.begin(),
                                join_condition_cf.simple_quals.end());
  }
  return combine_equi_join_conditions(join_condition_quals);
}

// Translate left deep join filter and separate the conjunctive form qualifiers
// per nesting level. The code generated for hash table lookups on each level
// must dominate its uses in deeper nesting levels.
JoinQualsPerNestingLevel RelAlgExecutor::translateLeftDeepJoinFilter(
    const RelLeftDeepInnerJoin* join,
    const std::vector<InputDescriptor>& input_descs,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const bool just_explain) {
  const auto join_types = left_deep_join_types(join);
  const auto join_condition_quals = makeJoinQuals(
      join->getInnerCondition(), join_types, input_to_nest_level, just_explain);
  MaxRangeTableIndexVisitor rte_idx_visitor;
  JoinQualsPerNestingLevel result(input_descs.size() - 1);
  std::unordered_set<std::shared_ptr<Analyzer::Expr>> visited_quals;
  for (size_t rte_idx = 1; rte_idx < input_descs.size(); ++rte_idx) {
    const auto outer_condition = join->getOuterCondition(rte_idx);
    if (outer_condition) {
      result[rte_idx - 1].quals =
          makeJoinQuals(outer_condition, join_types, input_to_nest_level, just_explain);
      CHECK_LE(rte_idx, join_types.size());
      CHECK(join_types[rte_idx - 1] == JoinType::LEFT);
      result[rte_idx - 1].type = JoinType::LEFT;
      continue;
    }
    for (const auto qual : join_condition_quals) {
      if (visited_quals.count(qual)) {
        continue;
      }
      const auto qual_rte_idx = rte_idx_visitor.visit(qual.get());
      if (static_cast<size_t>(qual_rte_idx) <= rte_idx) {
        const auto it_ok = visited_quals.emplace(qual);
        CHECK(it_ok.second);
        result[rte_idx - 1].quals.push_back(qual);
      }
    }
    CHECK_LE(rte_idx, join_types.size());
    CHECK(join_types[rte_idx - 1] == JoinType::INNER);
    result[rte_idx - 1].type = JoinType::INNER;
  }
  return result;
}

namespace {

std::vector<std::shared_ptr<Analyzer::Expr>> synthesize_inputs(
    const RelAlgNode* ra_node,
    const size_t nest_level,
    const std::vector<TargetMetaInfo>& in_metainfo,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  CHECK_LE(size_t(1), ra_node->inputCount());
  CHECK_GE(size_t(2), ra_node->inputCount());
  const auto input = ra_node->getInput(nest_level);
  const auto it_rte_idx = input_to_nest_level.find(input);
  CHECK(it_rte_idx != input_to_nest_level.end());
  const int rte_idx = it_rte_idx->second;
  const int table_id = table_id_from_ra(input);
  std::vector<std::shared_ptr<Analyzer::Expr>> inputs;
  const auto scan_ra = dynamic_cast<const RelScan*>(input);
  int input_idx = 0;
  for (const auto& input_meta : in_metainfo) {
    inputs.push_back(
        std::make_shared<Analyzer::ColumnVar>(input_meta.get_type_info(),
                                              table_id,
                                              scan_ra ? input_idx + 1 : input_idx,
                                              rte_idx));
    ++input_idx;
  }
  return inputs;
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createAggregateWorkUnit(
    const RelAggregate* aggregate,
    const SortInfo& sort_info,
    const bool just_explain) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto input_to_nest_level = get_input_nest_levels(aggregate, {});
  std::tie(input_descs, input_col_descs, used_inputs_owned) =
      get_input_desc(aggregate, input_to_nest_level, {}, cat_);
  const auto join_type = get_join_type(aggregate);
  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              {join_type},
                              now_,
                              just_explain,
                              query_features);
  CHECK_EQ(size_t(1), aggregate->inputCount());
  const auto source = aggregate->getInput(0);
  const auto& in_metainfo = source->getOutputMetainfo();
  const auto scalar_sources =
      synthesize_inputs(aggregate, size_t(0), in_metainfo, input_to_nest_level);
  const auto groupby_exprs = translate_groupby_exprs(aggregate, scalar_sources);
  const auto target_exprs = translate_targets(
      target_exprs_owned_, scalar_sources, groupby_exprs, aggregate, translator);
  const auto targets_meta = get_targets_meta(aggregate, target_exprs);
  aggregate->setOutputMetainfo(targets_meta);
  return {RelAlgExecutionUnit{input_descs,
                              input_col_descs,
                              {},
                              {},
                              {},
                              groupby_exprs,
                              target_exprs,
                              nullptr,
                              sort_info,
                              0,
                              query_features,
                              false,
                              query_state_},
          aggregate,
          max_groups_buffer_entry_default_guess,
          nullptr};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createProjectWorkUnit(
    const RelProject* project,
    const SortInfo& sort_info,
    const ExecutionOptions& eo) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  auto input_to_nest_level = get_input_nest_levels(project, {});
  std::tie(input_descs, input_col_descs, std::ignore) =
      get_input_desc(project, input_to_nest_level, {}, cat_);
  const auto query_infos = get_table_infos(input_descs, executor_);

  const auto left_deep_join =
      dynamic_cast<const RelLeftDeepInnerJoin*>(project->getInput(0));
  JoinQualsPerNestingLevel left_deep_join_quals;
  const auto join_types = left_deep_join ? left_deep_join_types(left_deep_join)
                                         : std::vector<JoinType>{get_join_type(project)};
  std::vector<size_t> input_permutation;
  std::vector<size_t> left_deep_join_input_sizes;
  if (left_deep_join) {
    left_deep_join_input_sizes = get_left_deep_join_input_sizes(left_deep_join);
    const auto query_infos = get_table_infos(input_descs, executor_);
    left_deep_join_quals = translateLeftDeepJoinFilter(
        left_deep_join, input_descs, input_to_nest_level, eo.just_explain);
    if (g_from_table_reordering) {
      input_permutation = do_table_reordering(input_descs,
                                              input_col_descs,
                                              left_deep_join_quals,
                                              input_to_nest_level,
                                              project,
                                              query_infos,
                                              executor_);
      input_to_nest_level = get_input_nest_levels(project, input_permutation);
      std::tie(input_descs, input_col_descs, std::ignore) =
          get_input_desc(project, input_to_nest_level, input_permutation, cat_);
      left_deep_join_quals = translateLeftDeepJoinFilter(
          left_deep_join, input_descs, input_to_nest_level, eo.just_explain);
    }
  }

  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              eo.just_explain,
                              query_features);
  const auto target_exprs_owned =
      translate_scalar_sources(project, translator, eo.executor_type);
  target_exprs_owned_.insert(
      target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  const RelAlgExecutionUnit exe_unit = {input_descs,
                                        input_col_descs,
                                        {},
                                        {},
                                        left_deep_join_quals,
                                        {nullptr},
                                        target_exprs,
                                        nullptr,
                                        sort_info,
                                        0,
                                        query_features,
                                        false,
                                        query_state_};
  auto query_rewriter = std::make_unique<QueryRewriter>(query_infos, executor_);
  const auto rewritten_exe_unit = query_rewriter->rewrite(exe_unit);
  const auto targets_meta = get_targets_meta(project, rewritten_exe_unit.target_exprs);
  project->setOutputMetainfo(targets_meta);
  return {rewritten_exe_unit,
          project,
          max_groups_buffer_entry_default_guess,
          std::move(query_rewriter),
          input_permutation,
          left_deep_join_input_sizes};
}

RelAlgExecutor::TableFunctionWorkUnit RelAlgExecutor::createTableFunctionWorkUnit(
    const RelTableFunction* table_func,
    const bool just_explain) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  auto input_to_nest_level = get_input_nest_levels(table_func, {});
  std::tie(input_descs, input_col_descs, std::ignore) =
      get_input_desc(table_func, input_to_nest_level, {}, cat_);
  const auto query_infos = get_table_infos(input_descs, executor_);
  CHECK_EQ(size_t(1), table_func->inputCount());

  QueryFeatureDescriptor query_features;  // TODO(adb): remove/make optional
  RelAlgTranslator translator(
      cat_, executor_, input_to_nest_level, {}, now_, just_explain, query_features);
  const auto input_exprs_owned =
      translate_scalar_sources(table_func, translator, ::ExecutorType::Native);
  target_exprs_owned_.insert(
      target_exprs_owned_.end(), input_exprs_owned.begin(), input_exprs_owned.end());
  const auto input_exprs = get_exprs_not_owned(input_exprs_owned);

  std::vector<Analyzer::ColumnVar*> input_col_exprs;
  for (auto input_expr : input_exprs) {
    if (auto col_var = dynamic_cast<Analyzer::ColumnVar*>(input_expr)) {
      input_col_exprs.push_back(col_var);
    }
  }
  CHECK_EQ(input_col_exprs.size(), table_func->getColInputsSize());

  const auto& table_function_impl =
      table_functions::TableFunctionsFactory::get(table_func->getFunctionName());

  std::vector<Analyzer::Expr*> table_func_outputs;
  for (size_t i = 0; i < table_function_impl.getOutputsSize(); i++) {
    const auto ti = table_function_impl.getOutputSQLType(i);
    target_exprs_owned_.push_back(std::make_shared<Analyzer::ColumnVar>(ti, 0, i, -1));
    table_func_outputs.push_back(target_exprs_owned_.back().get());
  }

  std::optional<size_t> output_row_multiplier;
  if (table_function_impl.hasUserSpecifiedOutputMultiplier()) {
    const auto parameter_index = table_function_impl.getOutputRowParameter();
    CHECK_GT(parameter_index, size_t(0));
    const auto parameter_expr = table_func->getTableFuncInputAt(parameter_index - 1);
    const auto parameter_expr_literal = dynamic_cast<const RexLiteral*>(parameter_expr);
    if (!parameter_expr_literal) {
      throw std::runtime_error(
          "Provided output buffer multiplier parameter is not a literal. Only literal "
          "values are supported with output buffer multiplier configured table "
          "functions.");
    }
    int64_t literal_val = parameter_expr_literal->getVal<int64_t>();
    if (literal_val < 0) {
      throw std::runtime_error("Provided output row multiplier " +
                               std::to_string(literal_val) +
                               " is not valid for table functions.");
    }
    output_row_multiplier = static_cast<size_t>(literal_val);
  }

  const TableFunctionExecutionUnit exe_unit = {
      input_descs,
      input_col_descs,
      input_exprs,            // table function inputs
      input_col_exprs,        // table function column inputs (duplicates w/ above)
      table_func_outputs,     // table function projected exprs
      output_row_multiplier,  // output buffer multiplier
      table_func->getFunctionName()};
  const auto targets_meta = get_targets_meta(table_func, exe_unit.target_exprs);
  table_func->setOutputMetainfo(targets_meta);
  return {exe_unit, table_func};
}

namespace {

std::pair<std::vector<TargetMetaInfo>, std::vector<std::shared_ptr<Analyzer::Expr>>>
get_inputs_meta(const RelFilter* filter,
                const RelAlgTranslator& translator,
                const std::vector<std::shared_ptr<RexInput>>& inputs_owned,
                const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  std::vector<TargetMetaInfo> in_metainfo;
  std::vector<std::shared_ptr<Analyzer::Expr>> exprs_owned;
  const auto data_sink_node = get_data_sink(filter);
  auto input_it = inputs_owned.begin();
  for (size_t nest_level = 0; nest_level < data_sink_node->inputCount(); ++nest_level) {
    const auto source = data_sink_node->getInput(nest_level);
    const auto scan_source = dynamic_cast<const RelScan*>(source);
    if (scan_source) {
      CHECK(source->getOutputMetainfo().empty());
      std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources_owned;
      for (size_t i = 0; i < scan_source->size(); ++i, ++input_it) {
        scalar_sources_owned.push_back(translator.translateScalarRex(input_it->get()));
      }
      const auto source_metadata =
          get_targets_meta(scan_source, get_exprs_not_owned(scalar_sources_owned));
      in_metainfo.insert(
          in_metainfo.end(), source_metadata.begin(), source_metadata.end());
      exprs_owned.insert(
          exprs_owned.end(), scalar_sources_owned.begin(), scalar_sources_owned.end());
    } else {
      const auto& source_metadata = source->getOutputMetainfo();
      input_it += source_metadata.size();
      in_metainfo.insert(
          in_metainfo.end(), source_metadata.begin(), source_metadata.end());
      const auto scalar_sources_owned = synthesize_inputs(
          data_sink_node, nest_level, source_metadata, input_to_nest_level);
      exprs_owned.insert(
          exprs_owned.end(), scalar_sources_owned.begin(), scalar_sources_owned.end());
    }
  }
  return std::make_pair(in_metainfo, exprs_owned);
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createFilterWorkUnit(const RelFilter* filter,
                                                              const SortInfo& sort_info,
                                                              const bool just_explain) {
  CHECK_EQ(size_t(1), filter->inputCount());
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::vector<TargetMetaInfo> in_metainfo;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;

  const auto input_to_nest_level = get_input_nest_levels(filter, {});
  std::tie(input_descs, input_col_descs, used_inputs_owned) =
      get_input_desc(filter, input_to_nest_level, {}, cat_);
  const auto join_type = get_join_type(filter);
  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              {join_type},
                              now_,
                              just_explain,
                              query_features);
  std::tie(in_metainfo, target_exprs_owned) =
      get_inputs_meta(filter, translator, used_inputs_owned, input_to_nest_level);
  const auto filter_expr = translator.translateScalarRex(filter->getCondition());
  const auto qual = fold_expr(filter_expr.get());
  target_exprs_owned_.insert(
      target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  filter->setOutputMetainfo(in_metainfo);
  const auto rewritten_qual = rewrite_expr(qual.get());
  return {{input_descs,
           input_col_descs,
           {},
           {rewritten_qual ? rewritten_qual : qual},
           {},
           {nullptr},
           target_exprs,
           nullptr,
           sort_info,
           0},
          filter,
          max_groups_buffer_entry_default_guess,
          nullptr};
}

SpeculativeTopNBlacklist RelAlgExecutor::speculative_topn_blacklist_;
