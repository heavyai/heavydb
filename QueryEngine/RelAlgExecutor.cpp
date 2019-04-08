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
#include "RelAlgTranslator.h"

#include "CalciteDeserializerUtils.h"
#include "CardinalityEstimator.h"
#include "EquiJoinCondition.h"
#include "ExpressionRewrite.h"
#include "FromTableReordering.h"
#include "InputMetadata.h"
#include "JoinFilterPushDown.h"
#include "QueryPhysicalInputsCollector.h"
#include "RangeTableIndexVisitor.h"
#include "RexVisitor.h"

#include "../Parser/ParserNode.h"
#include "../Shared/measure.h"

#include <algorithm>
#include <numeric>

namespace {

bool node_is_aggregate(const RelAlgNode* ra) {
  const auto compound = dynamic_cast<const RelCompound*>(ra);
  const auto aggregate = dynamic_cast<const RelAggregate*>(ra);
  return ((compound && compound->isAggregate()) || aggregate);
}

void scanForTablesAndAggsInRelAlgSeqForRender(std::vector<RaExecutionDesc>& exec_descs,
                                              RenderInfo* render_info) {
  CHECK(render_info);

  std::vector<std::string>& rtn_table_names = render_info->table_names;
  rtn_table_names.clear();
  if (exec_descs.empty()) {
    return;
  }

  std::unordered_set<std::string> table_names;
  std::unordered_set<const RelAlgNode*> visited;
  std::vector<const RelAlgNode*> work_set;
  for (const auto& exec_desc : exec_descs) {
    const auto body = exec_desc.getBody();
    if (visited.count(body)) {
      continue;
    }
    work_set.push_back(body);
    while (!work_set.empty()) {
      auto walker = work_set.back();
      work_set.pop_back();
      if (visited.count(walker)) {
        continue;
      }
      visited.insert(walker);
      if (walker->isNop()) {
        CHECK_EQ(size_t(1), walker->inputCount());
        work_set.push_back(walker->getInput(0));
        continue;
      }
      if (const auto scan = dynamic_cast<const RelScan*>(walker)) {
        auto td = scan->getTableDescriptor();
        CHECK(td);
        if (table_names.insert(td->tableName).second) {
          // keeping the traversed table names in order
          rtn_table_names.push_back(td->tableName);
        }
        continue;
      }
      if (node_is_aggregate(walker)) {
        // see new logic in executeRelAlgQueryNoRetry
        // if not using more relaxed logic, disallow if we find an aggregate *anywhere*
        if (!render_info->disallow_in_situ_only_if_final_ED_is_aggregate) {
          // set the render to be non in-situ if we have an
          // aggregate node
          render_info->setInSituDataIfUnset(false);
        }
      }
      const auto compound = dynamic_cast<const RelCompound*>(walker);
      const auto join = dynamic_cast<const RelJoin*>(walker);
      const auto project = dynamic_cast<const RelProject*>(walker);
      const auto aggregate = dynamic_cast<const RelAggregate*>(walker);
      const auto filter = dynamic_cast<const RelFilter*>(walker);
      const auto sort = dynamic_cast<const RelSort*>(walker);
      const auto leftdeepinnerjoin = dynamic_cast<const RelLeftDeepInnerJoin*>(walker);
      if (compound || join || project || aggregate || filter || sort ||
          leftdeepinnerjoin) {
        for (int i = walker->inputCount(); i > 0; --i) {
          work_set.push_back(walker->getInput(i - 1));
        }
        continue;
      }
      CHECK(false);
    }
  }
}

}  // namespace

ExecutionResult RelAlgExecutor::executeRelAlgQuery(const std::string& query_ra,
                                                   const CompilationOptions& co,
                                                   const ExecutionOptions& eo,
                                                   RenderInfo* render_info) {
  INJECT_TIMER(executeRelAlgQuery);
  try {
    return executeRelAlgQueryNoRetry(query_ra, co, eo, render_info);
  } catch (const QueryMustRunOnCpu&) {
    if (g_enable_watchdog && !g_allow_cpu_retry) {
      throw;
    }
  }
  LOG(INFO) << "Query unable to run in GPU mode, retrying on CPU";
  CompilationOptions co_cpu{ExecutorDeviceType::CPU,
                            co.hoist_literals_,
                            co.opt_level_,
                            co.with_dynamic_watchdog_};
  if (render_info) {
    render_info->setForceNonInSituData();
  }
  return executeRelAlgQueryNoRetry(query_ra, co_cpu, eo, render_info);
}

ExecutionResult RelAlgExecutor::executeRelAlgQueryNoRetry(const std::string& query_ra,
                                                          const CompilationOptions& co,
                                                          const ExecutionOptions& eo,
                                                          RenderInfo* render_info) {
  INJECT_TIMER(executeRelAlgQueryNoRetry);

  const auto ra = deserialize_ra_dag(query_ra, cat_, this);
  // capture the lock acquistion time
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);
  int64_t queue_time_ms = timer_stop(clock_begin);
  if (g_enable_dynamic_watchdog) {
    executor_->resetInterrupt();
  }
  ScopeGuard row_set_holder = [this, &render_info] {
    if (render_info) {
      // need to hold onto the RowSetMemOwner for potential
      // string id lookups during render vega validation
      render_info->row_set_mem_owner = executor_->row_set_mem_owner_;
    }
    cleanupPostExecution();
  };
  executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  executor_->catalog_ = &cat_;
  executor_->agg_col_range_cache_ = computeColRangesCache(ra.get());
  executor_->string_dictionary_generations_ =
      computeStringDictionaryGenerations(ra.get());
  executor_->table_generations_ = computeTableGenerations(ra.get());
  ScopeGuard restore_metainfo_cache = [this] { executor_->clearMetaInfoCache(); };
  auto ed_list = get_execution_descriptors(ra.get());
  if (render_info) {  // save the table names for render queries
    // set whether the render will be done in-situ (in_situ_data = true) or
    // set whether the query results will be transferred to the host and then
    // back to the device for rendering (in_situ_data = false)

    // if this is a potential in-situ poly render, we use more relaxed logic
    // only disallow in-situ if the *final* ED is an aggregate
    // this *should* be usable for point renders too, but not safe yet
    if (render_info->disallow_in_situ_only_if_final_ED_is_aggregate) {
      // new logic
      CHECK(ed_list.size() > 0);
      if (node_is_aggregate(ed_list.back().getBody())) {
        render_info->setInSituDataIfUnset(false);
      }
    } else {
      // old logic
      // disallow if more than one ED, and there's an aggregate *anywhere*
      if (ed_list.size() != 1) {
        render_info->setInSituDataIfUnset(false);
      }
    }
    scanForTablesAndAggsInRelAlgSeqForRender(ed_list, render_info);
  }
  if (eo.find_push_down_candidates) {
    // this extra logic is mainly due to current limitations on multi-step queries
    // and/or subqueries.
    return executeRelAlgQueryWithFilterPushDown(
        ed_list, co, eo, render_info, queue_time_ms);
  }

  // Dispatch the subqueries first
  for (auto subquery : subqueries_) {
    // Execute the subquery and cache the result.
    RelAlgExecutor ra_executor(executor_, cat_);
    auto result = ra_executor.executeRelAlgSubQuery(subquery.get(), co, eo);
    subquery->setExecutionResult(std::make_shared<ExecutionResult>(result));
  }
  return executeRelAlgSeq(ed_list, co, eo, render_info, queue_time_ms);
}

namespace {

std::unordered_set<int> get_physical_table_ids(
    const std::unordered_set<PhysicalInput>& phys_inputs) {
  std::unordered_set<int> physical_table_ids;
  for (const auto& phys_input : phys_inputs) {
    physical_table_ids.insert(phys_input.table_id);
  }
  return physical_table_ids;
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

AggregatedColRange RelAlgExecutor::computeColRangesCache(const RelAlgNode* ra) {
  AggregatedColRange agg_col_range_cache;
  const auto phys_inputs = get_physical_inputs(cat_, ra);
  const auto phys_table_ids = get_physical_table_ids(phys_inputs);
  std::vector<InputTableInfo> query_infos;
  executor_->catalog_ = &cat_;
  for (const int table_id : phys_table_ids) {
    query_infos.emplace_back(InputTableInfo{table_id, executor_->getTableInfo(table_id)});
  }
  for (const auto& phys_input : phys_inputs) {
    const auto cd = cat_.getMetadataForColumn(phys_input.table_id, phys_input.col_id);
    CHECK(cd);
    const auto& col_ti =
        cd->columnType.is_array() ? cd->columnType.get_elem_type() : cd->columnType;
    if (col_ti.is_number() || col_ti.is_boolean() || col_ti.is_time() ||
        (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT)) {
      const auto col_var = boost::make_unique<Analyzer::ColumnVar>(
          cd->columnType, phys_input.table_id, phys_input.col_id, 0);
      const auto col_range =
          getLeafColumnRange(col_var.get(), query_infos, executor_, false);
      agg_col_range_cache.setColRange(phys_input, col_range);
    }
  }
  return agg_col_range_cache;
}

StringDictionaryGenerations RelAlgExecutor::computeStringDictionaryGenerations(
    const RelAlgNode* ra) {
  StringDictionaryGenerations string_dictionary_generations;
  const auto phys_inputs = get_physical_inputs(cat_, ra);
  for (const auto& phys_input : phys_inputs) {
    const auto cd = cat_.getMetadataForColumn(phys_input.table_id, phys_input.col_id);
    CHECK(cd);
    const auto& col_ti =
        cd->columnType.is_array() ? cd->columnType.get_elem_type() : cd->columnType;
    if (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT) {
      const int dict_id = col_ti.get_comp_param();
      const auto dd = cat_.getMetadataForDict(dict_id);
      CHECK(dd && dd->stringDict);
      string_dictionary_generations.setGeneration(dict_id,
                                                  dd->stringDict->storageEntryCount());
    }
  }
  return string_dictionary_generations;
}

TableGenerations RelAlgExecutor::computeTableGenerations(const RelAlgNode* ra) {
  const auto phys_table_ids = get_physical_table_inputs(ra);
  TableGenerations table_generations;
  for (const int table_id : phys_table_ids) {
    const auto table_info = executor_->getTableInfo(table_id);
    table_generations.setGeneration(
        table_id, TableGeneration{table_info.getPhysicalNumTuples(), 0});
  }
  return table_generations;
}

Executor* RelAlgExecutor::getExecutor() const {
  return executor_;
}

void RelAlgExecutor::cleanupPostExecution() {
  CHECK(executor_);
  executor_->row_set_mem_owner_ = nullptr;
  executor_->lit_str_dict_proxy_ = nullptr;
}

FirstStepExecutionResult RelAlgExecutor::executeRelAlgQueryFirstStep(
    const RelAlgNode* ra,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    RenderInfo* render_info) {
  auto ed_list = get_execution_descriptors(ra);
  CHECK(!ed_list.empty());
  auto first_exec_desc = ed_list.front();
  const auto sort = dynamic_cast<const RelSort*>(first_exec_desc.getBody());
  size_t shard_count{0};
  if (sort) {
    const auto source_work_unit = createSortInputWorkUnit(sort, eo.just_explain);
    shard_count = GroupByAndAggregate::shard_count_for_top_groups(
        source_work_unit.exe_unit, *executor_->getCatalog());
    if (!shard_count) {
      // No point in sorting on the leaf, only execute the input to the sort node.
      CHECK_EQ(size_t(1), sort->inputCount());
      const auto source = sort->getInput(0);
      if (sort->collationCount() || node_is_aggregate(source)) {
        first_exec_desc = RaExecutionDesc(source);
      }
    }
  }
  std::vector<RaExecutionDesc> first_exec_desc_singleton_list{first_exec_desc};
  const auto merge_type = (node_is_aggregate(first_exec_desc.getBody()) && !shard_count)
                              ? MergeType::Reduce
                              : MergeType::Union;
  return {executeRelAlgSeq(
              first_exec_desc_singleton_list, co, eo, render_info, queue_time_ms_),
          merge_type,
          first_exec_desc.getBody()->getId(),
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

ExecutionResult RelAlgExecutor::executeRelAlgSubQuery(const RexSubQuery* subquery,
                                                      const CompilationOptions& co,
                                                      const ExecutionOptions& eo) {
  INJECT_TIMER(executeRelAlgSubQuery);
  auto ed_list = get_execution_descriptors(subquery->getRelAlg());
  return executeRelAlgSeq(ed_list, co, eo, nullptr, 0);
}

ExecutionResult RelAlgExecutor::executeRelAlgSeq(std::vector<RaExecutionDesc>& exec_descs,
                                                 const CompilationOptions& co,
                                                 const ExecutionOptions& eo,
                                                 RenderInfo* render_info,
                                                 const int64_t queue_time_ms) {
  INJECT_TIMER(executeRelAlgSeq);
  decltype(temporary_tables_)().swap(temporary_tables_);
  decltype(target_exprs_owned_)().swap(target_exprs_owned_);
  executor_->catalog_ = &cat_;
  executor_->temporary_tables_ = &temporary_tables_;

  // we will have to make sure the temp tables generated as a result of execution of inner
  // subqueries are available throughout the execution of the sequence.
  for (auto subquery : subqueries_) {
    auto temp_table = subquery->getExecutionResult();
    if (temp_table.get()) {
      addTemporaryTable(-(subquery->getRelAlg()->getId()), temp_table->getDataPtr());
    }
  }
  time(&now_);
  CHECK(!exec_descs.empty());
  const auto exec_desc_count = eo.just_explain ? size_t(1) : exec_descs.size();
  for (size_t i = 0; i < exec_desc_count; ++i) {
    // only render on the last step
    executeRelAlgStep(i,
                      exec_descs,
                      co,
                      eo,
                      (i == exec_desc_count - 1 ? render_info : nullptr),
                      queue_time_ms);
  }

  return exec_descs[exec_desc_count - 1].getResult();
}

void RelAlgExecutor::executeRelAlgStep(const size_t i,
                                       std::vector<RaExecutionDesc>& exec_descs,
                                       const CompilationOptions& co,
                                       const ExecutionOptions& eo,
                                       RenderInfo* render_info,
                                       const int64_t queue_time_ms) {
  INJECT_TIMER(executeRelAlgStep);
  auto& exec_desc = exec_descs[i];
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
      eo.with_watchdog && (i == 0 || dynamic_cast<const RelProject*>(body)),
      eo.jit_debug,
      eo.just_validate,
      eo.with_dynamic_watchdog,
      eo.dynamic_watchdog_time_limit,
      eo.find_push_down_candidates,
      eo.just_calcite_explain,
      eo.gpu_input_mem_limit_percent};

  if (render_info && !render_info->table_names.size() && leaf_results_.size()) {
    // Save the table names for render queries for distributed aggregation queries.
    // Doing this here as the aggregator calls executeRelAlgSeq() directly rather
    // than going thru the executeRelAlg() path.
    // TODO(croot): should we propagate these table names from the leaves instead
    // of populating this here, or expose this api for the aggregator to call directly?
    scanForTablesAndAggsInRelAlgSeqForRender(exec_descs, render_info);
  }

  const auto compound = dynamic_cast<const RelCompound*>(body);
  if (compound) {
  	std::cout<<"!!!------executeRelAlgStep:compound------!!!"<<std::endl;
    if (compound->isDeleteViaSelect()) {
      executeDeleteViaCompound(compound, co, eo_work_unit, render_info, queue_time_ms);
    } else if (compound->isUpdateViaSelect()) {
      executeUpdateViaCompound(compound, co, eo_work_unit, render_info, queue_time_ms);
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
  	std::cout<<"!!!------executeRelAlgStep:project-------!!!"<<std::endl;
    if (project->isDeleteViaSelect()) {
      executeDeleteViaProject(project, co, eo_work_unit, render_info, queue_time_ms);
    } else if (project->isUpdateViaSelect()) {
      executeUpdateViaProject(project, co, eo_work_unit, render_info, queue_time_ms);
    } else {
      exec_desc.setResult(
          executeProject(project, co, eo_work_unit, render_info, queue_time_ms));
      if (exec_desc.getResult().isFilterPushDownEnabled()) {
        return;
      }
      addTemporaryTable(-project->getId(), exec_desc.getResult().getDataPtr());
    }
    return;
  }
  const auto aggregate = dynamic_cast<const RelAggregate*>(body);
  if (aggregate) {
  	std::cout<<"!!!------executeRelAlgStep:aggregate-----!!!"<<std::endl;
    exec_desc.setResult(
        executeAggregate(aggregate, co, eo_work_unit, render_info, queue_time_ms));
    addTemporaryTable(-aggregate->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto filter = dynamic_cast<const RelFilter*>(body);
  if (filter) {
  	std::cout<<"!!!------executeRelAlgStep:filter--------!!!"<<std::endl;
    exec_desc.setResult(
        executeFilter(filter, co, eo_work_unit, render_info, queue_time_ms));
    addTemporaryTable(-filter->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto sort = dynamic_cast<const RelSort*>(body);
  if (sort) {
  	std::cout<<"!!!------executeRelAlgStep:sort----------!!!"<<std::endl;
    exec_desc.setResult(executeSort(sort, co, eo_work_unit, render_info, queue_time_ms));
    if (exec_desc.getResult().isFilterPushDownEnabled()) {
      return;
    }
    addTemporaryTable(-sort->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto logical_values = dynamic_cast<const RelLogicalValues*>(body);
  if (logical_values) {
  	std::cout<<"!!!------executeRelAlgStep:logical_values!!!"<<std::endl;
    exec_desc.setResult(executeLogicalValues(logical_values, eo_work_unit));
    addTemporaryTable(-logical_values->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto modify = dynamic_cast<const RelModify*>(body);
  if (modify) {
  	std::cout<<"!!!------executeRelAlgStep:modify--------!!!"<<std::endl;
    exec_desc.setResult(executeModify(modify, eo_work_unit));
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
    CHECK_EQ(join->inputCount(), 2);
    const auto condition = join->getCondition();
    RexUsedInputsVisitor visitor(cat);
    auto condition_inputs = visitor.visit(condition);
    std::vector<std::shared_ptr<RexInput>> condition_inputs_owned(
        visitor.get_inputs_owned());
    return std::make_pair(condition_inputs, condition_inputs_owned);
  }

  if (auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(data_sink_node)) {
    CHECK_GE(left_deep_join->inputCount(), 2);
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

  CHECK_EQ(ra_node->inputCount(), 1);
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

const RexScalar* scalar_at(const size_t i, const RelCompound* compound) {
  return compound->getScalarSource(i);
}

const RexScalar* scalar_at(const size_t i, const RelProject* project) {
  return project->getProjectAt(i);
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

template <class RA>
std::vector<std::shared_ptr<Analyzer::Expr>> translate_scalar_sources(
    const RA* ra_node,
    const RelAlgTranslator& translator) {
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
    try {
      scalar_sources.push_back(set_transient_dict(fold_expr(rewritten_expr.get())));
    } catch (...) {
      scalar_sources.push_back(fold_expr(rewritten_expr.get()));
    }
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
    const RelAlgTranslator& translator) {
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

std::vector<TargetMetaInfo> get_modify_manipulated_targets_meta(
    ModifyManipulationTarget const* manip_node,
    const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<TargetMetaInfo> targets_meta;

  for (int i = 0; i < (manip_node->getTargetColumnCount()); ++i) {
    CHECK(target_exprs[i]);
    // TODO(alex): remove the count distinct type fixup.
    targets_meta.emplace_back(manip_node->getTargetColumns()[i],
                              is_count_distinct(target_exprs[i])
                                  ? SQLTypeInfo(kBIGINT, false)
                                  : target_exprs[i]->get_type_info());
  }

  return targets_meta;
}

template <class RA>
std::vector<TargetMetaInfo> get_targets_meta(
    const RA* ra_node,
    const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<TargetMetaInfo> targets_meta;
  for (size_t i = 0; i < ra_node->size(); ++i) {
    CHECK(target_exprs[i]);
    // TODO(alex): remove the count distinct type fixup.
    targets_meta.emplace_back(
        ra_node->getFieldName(i),
        is_count_distinct(target_exprs[i])
            ? SQLTypeInfo(kBIGINT, false)
            : get_logical_type_info(target_exprs[i]->get_type_info()),
        target_exprs[i]->get_type_info());
  }
  return targets_meta;
}

}  // namespace

void RelAlgExecutor::executeUpdateViaCompound(const RelCompound* compound,
                                              const CompilationOptions& co,
                                              const ExecutionOptions& eo,
                                              RenderInfo* render_info,
                                              const int64_t queue_time_ms) {
  if (!compound->validateTargetColumns(
          yieldColumnValidator(compound->getModifiedTableDescriptor()))) {
    throw std::runtime_error(
        "Unsupported update operation encountered.  (None-encoded string column updates "
        "are not supported.)");
  }

  const auto work_unit = createModifyCompoundWorkUnit(
      compound, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);
  CompilationOptions co_project = co;
  co_project.device_type_ = ExecutorDeviceType::CPU;

  try {
    UpdateTriggeredCacheInvalidator::invalidateCaches();

    UpdateTransactionParameters update_params(compound->getModifiedTableDescriptor(),
                                              compound->getTargetColumns(),
                                              compound->getOutputMetainfo(),
                                              compound->isVarlenUpdateRequired());
    auto update_callback = yieldUpdateCallback(update_params);
    executor_->executeUpdate(work_unit.exe_unit,
                             table_infos.front(),
                             co_project,
                             eo,
                             cat_,
                             executor_->row_set_mem_owner_,
                             update_callback);
    update_params.finalizeTransaction();
  } catch (...) {
    LOG(INFO) << "Update operation failed.";
    throw;
  }
}

void RelAlgExecutor::executeUpdateViaProject(const RelProject* project,
                                             const CompilationOptions& co,
                                             const ExecutionOptions& eo,
                                             RenderInfo* render_info,
                                             const int64_t queue_time_ms) {
  if (!project->validateTargetColumns(
          yieldColumnValidator(project->getModifiedTableDescriptor()))) {
    throw std::runtime_error(
        "Unsupported update operation encountered.  (None-encoded string column updates "
        "are not supported.)");
  }

  auto work_unit = createModifyProjectWorkUnit(
      project, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);
  CompilationOptions co_project = co;
  co_project.device_type_ = ExecutorDeviceType::CPU;

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

  try {
    UpdateTriggeredCacheInvalidator::invalidateCaches();

    UpdateTransactionParameters update_params(project->getModifiedTableDescriptor(),
                                              project->getTargetColumns(),
                                              project->getOutputMetainfo(),
                                              project->isVarlenUpdateRequired());
    auto update_callback = yieldUpdateCallback(update_params);
    executor_->executeUpdate(work_unit.exe_unit,
                             table_infos.front(),
                             co_project,
                             eo,
                             cat_,
                             executor_->row_set_mem_owner_,
                             update_callback);
    update_params.finalizeTransaction();
  } catch (...) {
    LOG(INFO) << "Update operation failed.";
    throw;
  }
}

void RelAlgExecutor::executeDeleteViaCompound(const RelCompound* compound,
                                              const CompilationOptions& co,
                                              const ExecutionOptions& eo,
                                              RenderInfo* render_info,
                                              const int64_t queue_time_ms) {
  auto* table_descriptor = compound->getModifiedTableDescriptor();
  if (!table_descriptor->hasDeletedCol) {
    throw std::runtime_error(
        "DELETE only supported on tables with the vacuum attribute set to 'delayed'");
  }

  const auto work_unit = createModifyCompoundWorkUnit(
      compound, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);
  CompilationOptions co_project = co;
  co_project.device_type_ = ExecutorDeviceType::CPU;

  try {
    DeleteTriggeredCacheInvalidator::invalidateCaches();

    DeleteTransactionParameters delete_params;
    auto delete_callback = yieldDeleteCallback(delete_params);

    executor_->executeUpdate(work_unit.exe_unit,
                             table_infos.front(),
                             co_project,
                             eo,
                             cat_,
                             executor_->row_set_mem_owner_,
                             delete_callback);
    delete_params.finalizeTransaction();
  } catch (...) {
    LOG(INFO) << "Delete operation failed.";
    throw;
  }
}

void RelAlgExecutor::executeDeleteViaProject(const RelProject* project,
                                             const CompilationOptions& co,
                                             const ExecutionOptions& eo,
                                             RenderInfo* render_info,
                                             const int64_t queue_time_ms) {
  auto* table_descriptor = project->getModifiedTableDescriptor();
  if (!table_descriptor->hasDeletedCol) {
    throw std::runtime_error(
        "DELETE only supported on tables with the vacuum attribute set to 'delayed'");
  }

  auto work_unit = createModifyProjectWorkUnit(
      project, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);
  CompilationOptions co_project = co;
  co_project.device_type_ = ExecutorDeviceType::CPU;

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

  try {
    DeleteTriggeredCacheInvalidator::invalidateCaches();

    DeleteTransactionParameters delete_params;
    auto delete_callback = yieldDeleteCallback(delete_params);

    executor_->executeUpdate(work_unit.exe_unit,
                             table_infos.front(),
                             co_project,
                             eo,
                             cat_,
                             executor_->row_set_mem_owner_,
                             delete_callback);
    delete_params.finalizeTransaction();
  } catch (...) {
    LOG(INFO) << "Delete operation failed.";
    throw;
  }
}

ExecutionResult RelAlgExecutor::executeCompound(const RelCompound* compound,
                                                const CompilationOptions& co,
                                                const ExecutionOptions& eo,
                                                RenderInfo* render_info,
                                                const int64_t queue_time_ms) {
  const auto work_unit = createCompoundWorkUnit(
      compound, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  CompilationOptions co_compound = co;
  if (work_unit.exe_unit.query_features.isCPUOnlyExecutionRequired()) {
    co_compound.device_type_ = ExecutorDeviceType::CPU;
  }
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

ExecutionResult RelAlgExecutor::executeProject(const RelProject* project,
                                               const CompilationOptions& co,
                                               const ExecutionOptions& eo,
                                               RenderInfo* render_info,
                                               const int64_t queue_time_ms) {
  auto work_unit =
      createProjectWorkUnit(project, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  CompilationOptions co_project = co;
  if (work_unit.exe_unit.query_features.isCPUOnlyExecutionRequired()) {
    co_project.device_type_ = ExecutorDeviceType::CPU;
  }

  if (project->isSimple()) {
    CHECK_EQ(size_t(1), project->inputCount());
    const auto input_ra = project->getInput(0);
    if (dynamic_cast<const RelSort*>(input_ra)) {
      co_project.device_type_ = ExecutorDeviceType::CPU;
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
                         queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeFilter(const RelFilter* filter,
                                              const CompilationOptions& co,
                                              const ExecutionOptions& eo,
                                              RenderInfo* render_info,
                                              const int64_t queue_time_ms) {
  const auto work_unit =
      createFilterWorkUnit(filter, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  return executeWorkUnit(
      work_unit, filter->getOutputMetainfo(), false, co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeModify(const RelModify* modify,
                                              const ExecutionOptions& eo) {
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

ExecutionResult RelAlgExecutor::executeLogicalValues(
    const RelLogicalValues* logical_values,
    const ExecutionOptions& eo) {
  if (eo.just_explain) {
    throw std::runtime_error("EXPLAIN not supported for LogicalValues");
  }
  QueryMemoryDescriptor query_mem_desc(
      executor_, 1, QueryDescriptionType::NonGroupedAggregate);

  const auto& tuple_type = logical_values->getTupleType();
  for (size_t i = 0; i < tuple_type.size(); ++i) {
    query_mem_desc.addAggColWidth(ColWidths{8, 8});
  }
  logical_values->setOutputMetainfo(tuple_type);
  std::vector<std::unique_ptr<Analyzer::ColumnVar>> owned_column_expressions;
  std::vector<Analyzer::Expr*> target_expressions;
  for (const auto& tuple_component : tuple_type) {
    const auto column_var =
        new Analyzer::ColumnVar(tuple_component.get_type_info(), 0, 0, 0);
    target_expressions.push_back(column_var);
    owned_column_expressions.emplace_back(column_var);
  }
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
  return {rs, tuple_type};
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
  CHECK_EQ(size_t(1), sort->inputCount());
  const auto source = sort->getInput(0);
  if (dynamic_cast<const RelSort*>(source)) {
    throw std::runtime_error("Sort node not supported as input to another sort");
  }
  const bool is_aggregate = node_is_aggregate(source);
  auto it = leaf_results_.find(sort->getId());
  if (it != leaf_results_.end()) {
    // Add any transient string literals to the sdp on the agg
    const auto source_work_unit = createSortInputWorkUnit(sort, eo.just_explain);
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
      const auto source_work_unit = createSortInputWorkUnit(sort, eo.just_explain);
      is_desc = first_oe_is_desc(source_work_unit.exe_unit.sort_info.order_entries);
      groupby_exprs = source_work_unit.exe_unit.groupby_exprs;
      auto source_result = executeWorkUnit(source_work_unit,
                                           source->getOutputMetainfo(),
                                           is_aggregate,
                                           co,
                                           eo,
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
                                      co.device_type_,
                                      QueryMemoryDescriptor(),
                                      nullptr,
                                      executor_),
          {}};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createSortInputWorkUnit(
    const RelSort* sort,
    const bool just_explain) {
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
  auto source_work_unit = createWorkUnit(source, sort_info, just_explain);
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
  return {{source_exe_unit.input_descs,
           std::move(source_exe_unit.input_col_descs),
           source_exe_unit.simple_quals,
           source_exe_unit.quals,
           source_exe_unit.join_quals,
           source_exe_unit.groupby_exprs,
           source_exe_unit.target_exprs,
           nullptr,
           {sort_info.order_entries, sort_algorithm, limit, offset},
           scan_total_limit},
          source,
          max_groups_buffer_entry_guess,
          std::move(source_work_unit.query_rewriter),
          source_work_unit.input_permutation,
          source_work_unit.left_deep_join_input_sizes};
}

namespace {

// Upper bound estimation for the number of groups. Not strictly correct and not
// tight, but if the tables involved are really small we shouldn't waste time doing
// the NDV estimation. We don't account for cross-joins and / or group by unnested
// array, which is the reason this estimation isn't entirely reliable.
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

bool can_use_scan_limit(const RelAlgExecutionUnit& ra_exe_unit) {
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

RelAlgExecutionUnit decide_approx_count_distinct_implementation(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const std::vector<InputTableInfo>& table_infos,
    const Executor* executor,
    const ExecutorDeviceType device_type_in,
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned) {
  RelAlgExecutionUnit ra_exe_unit = ra_exe_unit_in;
  for (size_t i = 0; i < ra_exe_unit.target_exprs.size(); ++i) {
    const auto target_expr = ra_exe_unit.target_exprs[i];
    const auto agg_info = target_info(target_expr);
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
      CHECK(error_rate->get_type_info().get_type() == kSMALLINT);
      CHECK_GE(error_rate->get_constval().smallintval, 1);
      approx_bitmap_sz_bits = hll_size_for_rate(error_rate->get_constval().smallintval);
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

}  // namespace

ExecutionResult RelAlgExecutor::executeWorkUnit(
    const RelAlgExecutor::WorkUnit& work_unit,
    const std::vector<TargetMetaInfo>& targets_meta,
    const bool is_agg,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    RenderInfo* render_info,
    const int64_t queue_time_ms) {
  INJECT_TIMER(executeWorkUnit);
  if (!eo.just_explain && eo.find_push_down_candidates) {
    // find potential candidates:
    auto selected_filters = selectFiltersToBePushedDown(work_unit, co, eo);
    if (!selected_filters.empty() || eo.just_calcite_explain) {
      return ExecutionResult(selected_filters, eo.find_push_down_candidates);
    }
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
    return result;
  }
  int32_t error_code{0};

  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);

  auto ra_exe_unit = decide_approx_count_distinct_implementation(
      work_unit.exe_unit, table_infos, executor_, co.device_type_, target_exprs_owned_);
  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;

  if (!eo.just_explain && can_use_scan_limit(ra_exe_unit) && !isRowidLookup(work_unit)) {
    const auto filter_count_all = getFilteredCountAll(work_unit, true, co, eo);
    if (filter_count_all >= 0) {
      ra_exe_unit.scan_limit = std::max(filter_count_all, ssize_t(1));
    }
  }

  static const size_t big_group_threshold{20000};

  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     co.device_type_,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     executor_),
                         {}};

  try {
    result = {executor_->executeWorkUnit(
                  &error_code,
                  max_groups_buffer_entry_guess,
                  is_agg,
                  table_infos,
                  ra_exe_unit,
                  co,
                  eo,
                  cat_,
                  executor_->row_set_mem_owner_,
                  render_info,
                  groups_approx_upper_bound(table_infos) <= big_group_threshold),
              targets_meta};
  } catch (const CardinalityEstimationRequired&) {
    max_groups_buffer_entry_guess =
        2 * std::min(groups_approx_upper_bound(table_infos),
                     getNDVEstimation(work_unit, is_agg, co, eo));
    CHECK_GT(max_groups_buffer_entry_guess, size_t(0));
    result = {executor_->executeWorkUnit(&error_code,
                                         max_groups_buffer_entry_guess,
                                         is_agg,
                                         table_infos,
                                         ra_exe_unit,
                                         co,
                                         eo,
                                         cat_,
                                         executor_->row_set_mem_owner_,
                                         render_info,
                                         true),
              targets_meta};
  }

  result.setQueueTime(queue_time_ms);
  if (render_info) {
    const auto& target_exprs = work_unit.exe_unit.target_exprs;
    CHECK_EQ(target_exprs.size(), targets_meta.size());
    render_info->targets.clear();
    for (size_t i = 0; i < targets_meta.size(); ++i) {
      // TODO(croot): find a better way to iterate through these or a better data
      // structure for faster lookup to avoid the double for-loop. These vectors should be
      // small tho and have no impact on overall performance.
      size_t j{0};
      for (j = 0; j < target_exprs_owned_.size(); ++j) {
        if (target_exprs_owned_[j].get() == target_exprs[i]) {
          break;
        }
      }
      CHECK_LT(j, target_exprs_owned_.size());

      const auto& meta_ti = targets_meta[i].get_physical_type_info();
      const auto& expr_ti = target_exprs_owned_[j]->get_type_info();
      CHECK(meta_ti == expr_ti) << targets_meta[i].get_resname() << " " << i << "," << j
                                << ", targets meta: " << meta_ti.get_type_name() << "("
                                << meta_ti.get_compression_name()
                                << "), target_expr: " << expr_ti.get_type_name() << "("
                                << expr_ti.get_compression_name() << ")";
      render_info->targets.emplace_back(std::make_shared<Analyzer::TargetEntry>(
          targets_meta[i].get_resname(), target_exprs_owned_[j], false));
    }
    if (render_info->isPotentialInSituRender()) {
      // return an empty result (with the same queue time, and zero render time)
      return {
          std::make_shared<ResultSet>(queue_time_ms, 0, executor_->row_set_mem_owner_),
          {}};
    }
  }
  if (!error_code) {
    return result;
  }
  handlePersistentError(error_code);
  return handleRetry(error_code,
                     {ra_exe_unit, work_unit.body, max_groups_buffer_entry_guess},
                     targets_meta,
                     is_agg,
                     co,
                     eo,
                     queue_time_ms);
}

ssize_t RelAlgExecutor::getFilteredCountAll(const WorkUnit& work_unit,
                                            const bool is_agg,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo) {
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);
  if (table_infos.size() == 1 &&
      table_infos.front().info.getNumTuplesUpperBound() <= 50000) {
    return table_infos.front().info.getNumTuplesUpperBound();
  }
  const auto count =
      makeExpr<Analyzer::AggExpr>(SQLTypeInfo(g_bigint_count ? kBIGINT : kINT, false),
                                  kCOUNT,
                                  nullptr,
                                  false,
                                  nullptr);
  const auto count_all_exe_unit =
      create_count_all_execution_unit(work_unit.exe_unit, count);
  int32_t error_code{0};
  size_t one{1};
  ResultSetPtr count_all_result;
  try {
    count_all_result =
        executor_->executeWorkUnit(&error_code,
                                   one,
                                   is_agg,
                                   get_table_infos(work_unit.exe_unit, executor_),
                                   count_all_exe_unit,
                                   co,
                                   eo,
                                   cat_,
                                   executor_->row_set_mem_owner_,
                                   nullptr,
                                   false);
  } catch (...) {
    return -1;
  }
  if (error_code) {
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
  if (table_infos.size() == 1) {
    count_upper_bound = std::min(
        table_infos.front().info.getFragmentNumTuplesUpperBound(), count_upper_bound);
  }
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

ExecutionResult RelAlgExecutor::handleRetry(
    const int32_t error_code_in,
    const RelAlgExecutor::WorkUnit& work_unit,
    const std::vector<TargetMetaInfo>& targets_meta,
    const bool is_agg,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const int64_t queue_time_ms) {
  auto error_code = error_code_in;
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
                                   eo.gpu_input_mem_limit_percent};
  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     co.device_type_,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     executor_),
                         {}};
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);
  if (error_code == Executor::ERR_OUT_OF_GPU_MEM) {
    if (g_enable_watchdog && !g_allow_cpu_retry) {
      throw std::runtime_error(getErrorMessageFromCode(error_code));
    }
    const auto ra_exe_unit = decide_approx_count_distinct_implementation(
        work_unit.exe_unit, table_infos, executor_, co.device_type_, target_exprs_owned_);
    result = {executor_->executeWorkUnit(&error_code,
                                         max_groups_buffer_entry_guess,
                                         is_agg,
                                         table_infos,
                                         ra_exe_unit,
                                         co,
                                         eo_no_multifrag,
                                         cat_,
                                         executor_->row_set_mem_owner_,
                                         nullptr,
                                         true),
              targets_meta};
    result.setQueueTime(queue_time_ms);
    if (!error_code) {
      return result;
    }
  }
  handlePersistentError(error_code);
  if (co.device_type_ == ExecutorDeviceType::GPU) {
    std::string out_of_memory{"Query ran out of GPU memory, punt to CPU"};
    LOG(INFO) << out_of_memory;
    if (g_enable_watchdog && !g_allow_cpu_retry) {
      throw std::runtime_error(out_of_memory);
    }
  }
  CompilationOptions co_cpu{ExecutorDeviceType::CPU,
                            co.hoist_literals_,
                            co.opt_level_,
                            co.with_dynamic_watchdog_};
  if (error_code) {
    max_groups_buffer_entry_guess = 0;
    while (true) {
      const auto ra_exe_unit =
          decide_approx_count_distinct_implementation(work_unit.exe_unit,
                                                      table_infos,
                                                      executor_,
                                                      co_cpu.device_type_,
                                                      target_exprs_owned_);
      result = {executor_->executeWorkUnit(&error_code,
                                           max_groups_buffer_entry_guess,
                                           is_agg,
                                           table_infos,
                                           ra_exe_unit,
                                           co_cpu,
                                           eo_no_multifrag,
                                           cat_,
                                           executor_->row_set_mem_owner_,
                                           nullptr,
                                           true),
                targets_meta};
      result.setQueueTime(queue_time_ms);
      if (!error_code) {
        return result;
      }
      handlePersistentError(error_code);
      // Even the conservative guess failed; it should only happen when we group
      // by a huge cardinality array. Maybe we should throw an exception instead?
      // Such a heavy query is entirely capable of exhausting all the host memory.
      CHECK(max_groups_buffer_entry_guess);
      if (g_enable_watchdog) {
        throw std::runtime_error("Query ran out of output slots in the result");
      }
      max_groups_buffer_entry_guess *= 2;
    }
  }
  return result;
}

void RelAlgExecutor::handlePersistentError(const int32_t error_code) {
  if (error_code == Executor::ERR_SPECULATIVE_TOP_OOM) {
    throw SpeculativeTopNFailed();
  }
  if (error_code == Executor::ERR_OUT_OF_GPU_MEM &&
      (!g_enable_watchdog || g_allow_cpu_retry)) {
    // We ran out of GPU memory, this doesn't count as an error if the query is
    // allowed to continue on CPU because either the watchdog is disabled or
    // retry on CPU is explicitly allowed through --allow-cpu-retry.
    return;
  }
  throw std::runtime_error(getErrorMessageFromCode(error_code));
}

std::string RelAlgExecutor::getErrorMessageFromCode(const int32_t error_code) {
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
  }
  return "Other error: code " + std::to_string(error_code);
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createWorkUnit(const RelAlgNode* node,
                                                        const SortInfo& sort_info,
                                                        const bool just_explain) {
  const auto compound = dynamic_cast<const RelCompound*>(node);
  if (compound) {
    return createCompoundWorkUnit(compound, sort_info, just_explain);
  }
  const auto project = dynamic_cast<const RelProject*>(node);
  if (project) {
    return createProjectWorkUnit(project, sort_info, just_explain);
  }
  const auto aggregate = dynamic_cast<const RelAggregate*>(node);
  if (aggregate) {
    return createAggregateWorkUnit(aggregate, sort_info, just_explain);
  }
  const auto filter = dynamic_cast<const RelFilter*>(node);
  CHECK(filter);
  return createFilterWorkUnit(filter, sort_info, just_explain);
}

namespace {

template <typename SET_TYPE>
class UsedColumnsVisitor : public ScalarExprVisitor<SET_TYPE> {
 public:
  using ColumnIdSet = SET_TYPE;

 protected:
  virtual ColumnIdSet visitColumnVar(const Analyzer::ColumnVar* col_var) const override {
    return {col_var->get_column_id()};
  }

  virtual std::unordered_set<int> aggregateResult(
      const std::unordered_set<int>& aggregate,
      const std::unordered_set<int>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

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

RelAlgExecutor::WorkUnit RelAlgExecutor::createModifyCompoundWorkUnit(
    const RelCompound* compound,
    const SortInfo& sort_info,
    const bool just_explain) {
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
  if (left_deep_join) {
    left_deep_join_quals = translateLeftDeepJoinFilter(
        left_deep_join, input_descs, input_to_nest_level, just_explain);
  }
  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              just_explain,
                              query_features);
  const auto scalar_sources = translate_scalar_sources(compound, translator);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals_cf = translate_quals(compound, translator);
  const auto target_exprs = translate_targets(
      target_exprs_owned_, scalar_sources, groupby_exprs, compound, translator);
  CHECK_EQ(compound->size(), target_exprs.size());

  // Filter col descs and drop unneeded col_descs
  CHECK((target_exprs.size() - compound->getTargetColumnCount() - 1) > 0);
  const auto update_expr_iter = std::next(
      target_exprs.cbegin(), target_exprs.size() - compound->getTargetColumnCount() - 1);

  using ColumnIdSet = std::unordered_set<int>;
  UsedColumnsVisitor<ColumnIdSet> used_columns_visitor;
  ColumnIdSet id_accumulator;

  decltype(target_exprs) filtered_target_exprs(update_expr_iter, target_exprs.end());
  for (auto const& expr :
       boost::make_iterator_range(update_expr_iter, target_exprs.end())) {
    auto used_column_ids = used_columns_visitor.visit(expr);
    id_accumulator.insert(used_column_ids.begin(), used_column_ids.end());
  }
  for (auto const& expr : quals_cf.simple_quals) {
    auto simple_quals_used_column_ids = used_columns_visitor.visit(expr.get());
    id_accumulator.insert(simple_quals_used_column_ids.begin(),
                          simple_quals_used_column_ids.end());
  }
  for (auto const& expr : quals_cf.quals) {
    auto quals_used_column_ids = used_columns_visitor.visit(expr.get());
    id_accumulator.insert(quals_used_column_ids.begin(), quals_used_column_ids.end());
  }

  decltype(input_col_descs) filtered_input_col_descs;
  for (auto col_desc : input_col_descs) {
    if (id_accumulator.find(col_desc->getColId()) != id_accumulator.end()) {
      filtered_input_col_descs.push_back(col_desc);
    }
  }

  const RelAlgExecutionUnit exe_unit = {input_descs,
                                        filtered_input_col_descs,
                                        quals_cf.simple_quals,
                                        rewrite_quals(quals_cf.quals),
                                        left_deep_join_quals,
                                        groupby_exprs,
                                        filtered_target_exprs,
                                        nullptr,
                                        sort_info,
                                        0,
                                        query_features};
  auto query_rewriter = std::make_unique<QueryRewriter>(query_infos, executor_);
  const auto rewritten_exe_unit = query_rewriter->rewrite(exe_unit);
  const auto targets_meta =
      get_modify_manipulated_targets_meta(compound, rewritten_exe_unit.target_exprs);
  compound->setOutputMetainfo(targets_meta);
  return {rewritten_exe_unit,
          compound,
          max_groups_buffer_entry_default_guess,
          std::move(query_rewriter)};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createCompoundWorkUnit(
    const RelCompound* compound,
    const SortInfo& sort_info,
    const bool just_explain) {
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
        left_deep_join, input_descs, input_to_nest_level, just_explain);
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
          left_deep_join, input_descs, input_to_nest_level, just_explain);
    }
  }
  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              just_explain,
                              query_features);
  const auto scalar_sources = translate_scalar_sources(compound, translator);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals_cf = translate_quals(compound, translator);
  const auto target_exprs = translate_targets(
      target_exprs_owned_, scalar_sources, groupby_exprs, compound, translator);
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
                                        query_features};
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
  return {{input_descs,
           input_col_descs,
           {},
           {},
           {},
           groupby_exprs,
           target_exprs,
           nullptr,
           sort_info,
           0,
           query_features},
          aggregate,
          max_groups_buffer_entry_default_guess,
          nullptr};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createModifyProjectWorkUnit(
    const RelProject* project,
    const SortInfo& sort_info,
    const bool just_explain) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  auto input_to_nest_level = get_input_nest_levels(project, {});
  std::tie(input_descs, input_col_descs, std::ignore) =
      get_input_desc(project, input_to_nest_level, {}, cat_);
  const auto left_deep_join =
      dynamic_cast<const RelLeftDeepInnerJoin*>(project->getInput(0));
  JoinQualsPerNestingLevel left_deep_join_quals;
  const auto join_types = left_deep_join ? left_deep_join_types(left_deep_join)
                                         : std::vector<JoinType>{get_join_type(project)};
  if (left_deep_join) {
    left_deep_join_quals = translateLeftDeepJoinFilter(
        left_deep_join, input_descs, input_to_nest_level, just_explain);
  }
  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              just_explain,
                              query_features);
  const auto target_exprs_owned = translate_scalar_sources(project, translator);
  target_exprs_owned_.insert(
      target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);

  CHECK((target_exprs.size() - project->getTargetColumnCount() - 1) > 0);
  const auto update_expr_iter = std::next(
      target_exprs.cbegin(), target_exprs.size() - project->getTargetColumnCount() - 1);
  decltype(target_exprs) filtered_target_exprs(update_expr_iter, target_exprs.end());

  using ColumnIdSet = std::unordered_set<int>;
  UsedColumnsVisitor<ColumnIdSet> used_columns_visitor;
  ColumnIdSet id_accumulator;

  for (auto const& expr :
       boost::make_iterator_range(update_expr_iter, target_exprs.end())) {
    auto used_column_ids = used_columns_visitor.visit(expr);
    id_accumulator.insert(used_column_ids.begin(), used_column_ids.end());
  }

  decltype(input_col_descs) filtered_input_col_descs;
  for (auto col_desc : input_col_descs) {
    if (id_accumulator.find(col_desc->getColId()) != id_accumulator.end()) {
      filtered_input_col_descs.push_back(col_desc);
    }
  }

  const auto targets_meta =
      get_modify_manipulated_targets_meta(project, filtered_target_exprs);
  project->setOutputMetainfo(targets_meta);
  return {{input_descs,
           filtered_input_col_descs,
           {},
           {},
           left_deep_join_quals,
           {nullptr},
           filtered_target_exprs,
           nullptr,
           sort_info,
           0,
           query_features},
          project,
          max_groups_buffer_entry_default_guess,
          nullptr};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createProjectWorkUnit(const RelProject* project,
                                                               const SortInfo& sort_info,
                                                               const bool just_explain) {
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
        left_deep_join, input_descs, input_to_nest_level, just_explain);
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
          left_deep_join, input_descs, input_to_nest_level, just_explain);
    }
  }

  QueryFeatureDescriptor query_features;
  RelAlgTranslator translator(cat_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              just_explain,
                              query_features);
  const auto target_exprs_owned = translate_scalar_sources(project, translator);
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
                                        query_features};
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
