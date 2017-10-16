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
#include "ExecutionException.h"
#include "ExpressionRewrite.h"
#include "InputMetadata.h"
#include "QueryPhysicalInputsCollector.h"
#include "RangeTableIndexVisitor.h"
#include "RexVisitor.h"

#include "../Shared/measure.h"

#include <algorithm>
#include <numeric>

ExecutionResult RelAlgExecutor::executeRelAlgQuery(const std::string& query_ra,
                                                   const CompilationOptions& co,
                                                   const ExecutionOptions& eo,
                                                   RenderInfo* render_info) {
  try {
    return executeRelAlgQueryNoRetry(query_ra, co, eo, render_info);
  } catch (const QueryMustRunOnCpu&) {
    if (g_enable_watchdog && !g_allow_cpu_retry) {
      throw;
    }
  }
  CompilationOptions co_cpu{ExecutorDeviceType::CPU, co.hoist_literals_, co.opt_level_, co.with_dynamic_watchdog_};
  return executeRelAlgQueryNoRetry(query_ra, co_cpu, eo, render_info);
}

ExecutionResult RelAlgExecutor::executeRelAlgQueryNoRetry(const std::string& query_ra,
                                                          const CompilationOptions& co,
                                                          const ExecutionOptions& eo,
                                                          RenderInfo* render_info) {
  const auto ra = deserialize_ra_dag(query_ra, cat_, this);
  // capture the lock acquistion time
  auto clock_begin = timer_start();
  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);
  int64_t queue_time_ms = timer_stop(clock_begin);
  if (g_enable_dynamic_watchdog) {
    executor_->resetInterrupt();
  }
  ScopeGuard row_set_holder = [this] {
    executor_->row_set_mem_owner_ = nullptr;
    executor_->lit_str_dict_proxy_ = nullptr;
  };
  executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
  executor_->catalog_ = &cat_;
  executor_->agg_col_range_cache_ = computeColRangesCache(ra.get());
  executor_->string_dictionary_generations_ = computeStringDictionaryGenerations(ra.get());
  executor_->table_generations_ = computeTableGenerations(ra.get());
  ScopeGuard restore_metainfo_cache = [this] { executor_->clearMetaInfoCache(); };
  auto ed_list = get_execution_descriptors(ra.get());
  if (render_info && render_info->do_render && ed_list.size() > 1) {
    throw std::runtime_error("Cannot render queries which need more than one execution step");
  }
  if (render_info) {  // save the table names for render queries
    table_names_ = getScanTableNamesInRelAlgSeq(ed_list);
  }
  // Dispatch the subqueries first
  for (auto subquery : subqueries_) {
    // Execute the subquery and cache the result.
    RelAlgExecutor ra_executor(executor_, cat_);
    auto result = ra_executor.executeRelAlgSubQuery(subquery, co, eo);
    subquery->setExecutionResult(std::make_shared<ExecutionResult>(result));
  }
  return executeRelAlgSeq(ed_list, co, eo, render_info, queue_time_ms);
}

namespace {

std::unordered_set<int> get_physical_table_ids(const std::unordered_set<PhysicalInput>& phys_inputs) {
  std::unordered_set<int> physical_table_ids;
  for (const auto& phys_input : phys_inputs) {
    physical_table_ids.insert(phys_input.table_id);
  }
  return physical_table_ids;
}

}  // namespace

AggregatedColRange RelAlgExecutor::computeColRangesCache(const RelAlgNode* ra) {
  AggregatedColRange agg_col_range_cache;
  const auto phys_inputs = get_physical_inputs(ra);
  const auto phys_table_ids = get_physical_table_ids(phys_inputs);
  std::vector<InputTableInfo> query_infos;
  executor_->catalog_ = &cat_;
  for (const int table_id : phys_table_ids) {
    query_infos.emplace_back(InputTableInfo{table_id, executor_->getTableInfo(table_id)});
  }
  for (const auto& phys_input : phys_inputs) {
    const auto cd = cat_.getMetadataForColumn(phys_input.table_id, phys_input.col_id);
    CHECK(cd);
    const auto& col_ti = cd->columnType.is_array() ? cd->columnType.get_elem_type() : cd->columnType;
    if (col_ti.is_number() || col_ti.is_boolean() || col_ti.is_time() ||
        (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT)) {
      const auto col_var =
          boost::make_unique<Analyzer::ColumnVar>(cd->columnType, phys_input.table_id, phys_input.col_id, 0);
      const auto col_range = getLeafColumnRange(col_var.get(), query_infos, executor_, false);
      agg_col_range_cache.setColRange(phys_input, col_range);
    }
  }
  return agg_col_range_cache;
}

StringDictionaryGenerations RelAlgExecutor::computeStringDictionaryGenerations(const RelAlgNode* ra) {
  StringDictionaryGenerations string_dictionary_generations;
  const auto phys_inputs = get_physical_inputs(ra);
  for (const auto& phys_input : phys_inputs) {
    const auto cd = cat_.getMetadataForColumn(phys_input.table_id, phys_input.col_id);
    CHECK(cd);
    const auto& col_ti = cd->columnType.is_array() ? cd->columnType.get_elem_type() : cd->columnType;
    if (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT) {
      const int dict_id = col_ti.get_comp_param();
      const auto dd = cat_.getMetadataForDict(dict_id);
      CHECK(dd && dd->stringDict);
      string_dictionary_generations.setGeneration(dict_id, dd->stringDict->storageEntryCount());
    }
  }
  return string_dictionary_generations;
}

TableGenerations RelAlgExecutor::computeTableGenerations(const RelAlgNode* ra) {
  const auto phys_table_ids = get_physical_table_inputs(ra);
  TableGenerations table_generations;
  for (const int table_id : phys_table_ids) {
    const auto table_info = executor_->getTableInfo(table_id);
    table_generations.setGeneration(table_id, TableGeneration{table_info.getPhysicalNumTuples(), 0});
  }
  return table_generations;
}

Executor* RelAlgExecutor::getExecutor() const {
  return executor_;
}

namespace {

bool node_is_aggregate(const RelAlgNode* ra) {
  const auto compound = dynamic_cast<const RelCompound*>(ra);
  const auto aggregate = dynamic_cast<const RelAggregate*>(ra);
  return ((compound && compound->isAggregate()) || aggregate);
}

}  // namespace

FirstStepExecutionResult RelAlgExecutor::executeRelAlgQueryFirstStep(const RelAlgNode* ra,
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
    shard_count = shard_count_for_top_groups(source_work_unit.exe_unit, *executor_->getCatalog());
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
  const auto merge_type =
      (node_is_aggregate(first_exec_desc.getBody()) && !shard_count) ? MergeType::Reduce : MergeType::Union;
  return {executeRelAlgSeq(first_exec_desc_singleton_list, co, eo, render_info, queue_time_ms_),
          merge_type,
          first_exec_desc.getBody()->getId(),
          false};
}

void RelAlgExecutor::prepareLeafExecution(const AggregatedColRange& agg_col_range,
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
  auto ed_list = get_execution_descriptors(subquery->getRelAlg());
  return executeRelAlgSeq(ed_list, co, eo, nullptr, 0);
}

ExecutionResult RelAlgExecutor::executeRelAlgSeq(std::vector<RaExecutionDesc>& exec_descs,
                                                 const CompilationOptions& co,
                                                 const ExecutionOptions& eo,
                                                 RenderInfo* render_info,
                                                 const int64_t queue_time_ms) {
  decltype(temporary_tables_)().swap(temporary_tables_);
  decltype(target_exprs_owned_)().swap(target_exprs_owned_);
  executor_->catalog_ = &cat_;
  executor_->temporary_tables_ = &temporary_tables_;
  time(&now_);
  CHECK(!exec_descs.empty());
  const auto exec_desc_count = eo.just_explain ? size_t(1) : exec_descs.size();
  for (size_t i = 0; i < exec_desc_count; ++i) {
    executeRelAlgStep(i, exec_descs, co, eo, render_info, queue_time_ms);
  }
  return exec_descs[exec_desc_count - 1].getResult();
}

namespace {

class RexInputRedirector : public RexDeepCopyVisitor {
 public:
  RexInputRedirector(const RelJoin* head, const RelJoin* tail) {
    CHECK(head && tail);
    lhs_join_ = dynamic_cast<const RelJoin*>(tail->getInput(0));
    CHECK(lhs_join_);
    for (auto walker = lhs_join_; walker; walker = dynamic_cast<const RelJoin*>(walker->getInput(0))) {
      node_to_col_base_.insert(std::make_pair(walker->getInput(1), walker->getInput(0)->size()));
      if (walker == head) {
        node_to_col_base_.insert(std::make_pair(walker->getInput(0), size_t(0)));
        break;
      }
    }
  }

  RetType visitInput(const RexInput* input) const override {
    auto source = input->getSourceNode();
    auto new_base_it = node_to_col_base_.find(source);
    if (new_base_it == node_to_col_base_.end()) {
      return input->deepCopy();
    }

    return boost::make_unique<RexInput>(lhs_join_, input->getIndex() + new_base_it->second);
  }

  void visitNode(RelAlgNode* node) const {
    if (dynamic_cast<RelAggregate*>(node) || dynamic_cast<RelSort*>(node)) {
      return;
    }
    if (auto compound = dynamic_cast<RelCompound*>(node)) {
      if (auto filter_expr = compound->getFilterExpr()) {
        auto new_filter = visit(filter_expr);
        compound->setFilterExpr(new_filter);
      }
      std::vector<std::unique_ptr<const RexScalar>> new_source_exprs;
      for (size_t i = 0; i < compound->getScalarSourcesSize(); ++i) {
        new_source_exprs.push_back(visit(compound->getScalarSource(i)));
      }
      compound->setScalarSources(new_source_exprs);
      return;
    }
    if (auto project = dynamic_cast<RelProject*>(node)) {
      std::vector<std::unique_ptr<const RexScalar>> new_exprs;
      for (size_t i = 0; i < project->size(); ++i) {
        new_exprs.push_back(visit(project->getProjectAt(i)));
      }
      project->setExpressions(new_exprs);
      return;
    }
    if (auto filter = dynamic_cast<RelFilter*>(node)) {
      auto new_condition = visit(filter->getCondition());
      filter->setCondition(new_condition);
      return;
    }
    CHECK(false);
  }

 private:
  std::unordered_map<const RelAlgNode*, size_t> node_to_col_base_;
  const RelJoin* lhs_join_;
};

}  // namespace

void RelAlgExecutor::executeUnfoldedMultiJoin(const RelAlgNode* user,
                                              RaExecutionDesc& exec_desc,
                                              const CompilationOptions& co,
                                              const ExecutionOptions& eo,
                                              const int64_t queue_time_ms) {
  CHECK(!dynamic_cast<const RelJoin*>(user));
  CHECK_EQ(size_t(1), user->inputCount());
  auto sort = dynamic_cast<const RelSort*>(user);
  CHECK(!sort || sort->getInput(0)->inputCount() == 1);
  auto multi_join = sort ? std::dynamic_pointer_cast<const RelMultiJoin>(sort->getInput(0)->getAndOwnInput(0))
                         : std::dynamic_pointer_cast<const RelMultiJoin>(user->getAndOwnInput(0));
  CHECK(multi_join);
  const auto join_count = multi_join->joinCount();
  CHECK_LE(size_t(2), join_count);
  auto tail = multi_join->getAndOwnJoinAt(join_count - 1);
  std::vector<const RelAlgNode*> retry_sequence;
  if (!temporary_tables_.count(-tail->getId())) {
    for (size_t i = 0; i < join_count - 1; ++i) {
      retry_sequence.push_back(multi_join->getJoinAt(i));
    }
  }
  std::shared_ptr<RelAlgNode> copy;
  RexInputRedirector patcher(multi_join->getJoinAt(0), tail.get());
  if (sort) {
    auto source_copy = sort->getInput(0)->deepCopy();
    patcher.visitNode(source_copy.get());
    source_copy->RelAlgNode::replaceInput(multi_join, tail);
    copy = sort->deepCopy();
    copy->replaceInput(sort->getAndOwnInput(0), source_copy);
    retry_sequence.push_back(copy.get());
  } else {
    copy = user->deepCopy();
    patcher.visitNode(copy.get());
    copy->RelAlgNode::replaceInput(multi_join, tail);
    retry_sequence.push_back(copy.get());
  }
  auto retry_descs = get_execution_descriptors(retry_sequence);
  CHECK(!eo.just_explain);
  for (size_t i = 0; i < retry_descs.size(); ++i) {
    executeRelAlgStep(i, retry_descs, co, eo, nullptr, queue_time_ms);
  }

  CHECK(!retry_descs.empty());
  exec_desc.setResult(retry_descs.back().getResult());
  addTemporaryTable(-user->getId(), exec_desc.getResult().getDataPtr());
}

void RelAlgExecutor::executeRelAlgStep(const size_t i,
                                       std::vector<RaExecutionDesc>& exec_descs,
                                       const CompilationOptions& co,
                                       const ExecutionOptions& eo,
                                       RenderInfo* render_info,
                                       const int64_t queue_time_ms) {
  auto& exec_desc = exec_descs[i];
  const auto body = exec_desc.getBody();
  if (body->isNop()) {
    handleNop(body);
    return;
  }
  const ExecutionOptions eo_work_unit{eo.output_columnar_hint,
                                      eo.allow_multifrag,
                                      eo.just_explain,
                                      eo.allow_loop_joins,
                                      eo.with_watchdog && (i == 0 || dynamic_cast<const RelProject*>(body)),
                                      eo.jit_debug,
                                      eo.just_validate,
                                      eo.with_dynamic_watchdog,
                                      eo.dynamic_watchdog_time_limit};
  try {
    const auto compound = dynamic_cast<const RelCompound*>(body);
    if (compound) {
      exec_desc.setResult(executeCompound(compound, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-compound->getId(), exec_desc.getResult().getDataPtr());
      return;
    }
    const auto project = dynamic_cast<const RelProject*>(body);
    if (project) {
      exec_desc.setResult(executeProject(project, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-project->getId(), exec_desc.getResult().getDataPtr());
      return;
    }
    const auto aggregate = dynamic_cast<const RelAggregate*>(body);
    if (aggregate) {
      exec_desc.setResult(executeAggregate(aggregate, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-aggregate->getId(), exec_desc.getResult().getDataPtr());
      return;
    }
    const auto filter = dynamic_cast<const RelFilter*>(body);
    if (filter) {
      exec_desc.setResult(executeFilter(filter, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-filter->getId(), exec_desc.getResult().getDataPtr());
      return;
    }
    const auto sort = dynamic_cast<const RelSort*>(body);
    if (sort) {
      exec_desc.setResult(executeSort(sort, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-sort->getId(), exec_desc.getResult().getDataPtr());
      return;
    }
#ifdef ENABLE_JOIN_EXEC
    const auto join = dynamic_cast<const RelJoin*>(body);
    if (join) {
      exec_desc.setResult(executeJoin(join, co, eo_work_unit, render_info, queue_time_ms));
      addTemporaryTable(-join->getId(), exec_desc.getResult().getDataPtr());
      return;
    }
#endif
    CHECK(false);
  } catch (const UnfoldedMultiJoinRequired&) {
    executeUnfoldedMultiJoin(body, exec_desc, co, eo, queue_time_ms);
  }
}

const std::vector<std::string>& RelAlgExecutor::getScanTableNamesInRelAlgSeq() const {
  return table_names_;
}

std::vector<std::string> RelAlgExecutor::getScanTableNamesInRelAlgSeq(std::vector<RaExecutionDesc>& exec_descs) {
  if (exec_descs.empty()) {
    return {};
  }
  std::unordered_set<std::string> table_names;
  std::vector<std::string> rtn_table_names;
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
      const auto compound = dynamic_cast<const RelCompound*>(walker);
      const auto join = dynamic_cast<const RelJoin*>(walker);
      const auto project = dynamic_cast<const RelProject*>(walker);
      const auto aggregate = dynamic_cast<const RelAggregate*>(walker);
      const auto filter = dynamic_cast<const RelFilter*>(walker);
      const auto sort = dynamic_cast<const RelSort*>(walker);
      if (compound || join || project || aggregate || filter || sort) {
        for (int i = walker->inputCount(); i > 0; --i) {
          work_set.push_back(walker->getInput(i - 1));
        }
        continue;
      }
      CHECK(false);
    }
  }
  return rtn_table_names;
}

void RelAlgExecutor::handleNop(const RelAlgNode* body) {
  CHECK(dynamic_cast<const RelAggregate*>(body));
  CHECK_EQ(size_t(1), body->inputCount());
  const auto input = body->getInput(0);
  body->setOutputMetainfo(input->getOutputMetainfo());
  const auto it = temporary_tables_.find(-input->getId());
  CHECK(it != temporary_tables_.end());
  addTemporaryTable(-body->getId(), it->second);
}

namespace {

class RexUsedInputsVisitor : public RexVisitor<std::unordered_set<const RexInput*>> {
 public:
  std::unordered_set<const RexInput*> visitInput(const RexInput* rex_input) const override { return {rex_input}; }

 protected:
  std::unordered_set<const RexInput*> aggregateResult(
      const std::unordered_set<const RexInput*>& aggregate,
      const std::unordered_set<const RexInput*>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

const RelAlgNode* get_data_sink(const RelAlgNode* ra_node) {
  if (auto join = dynamic_cast<const RelJoin*>(ra_node)) {
    CHECK_EQ(size_t(2), join->inputCount());
    return join;
  }
  if (auto multi_join = dynamic_cast<const RelMultiJoin*>(ra_node)) {
    CHECK_LT(2, multi_join->inputCount());
    return multi_join;
  }
  CHECK_EQ(size_t(1), ra_node->inputCount());
  auto only_src = ra_node->getInput(0);
  const bool is_join = dynamic_cast<const RelJoin*>(only_src) || dynamic_cast<const RelMultiJoin*>(only_src) ||
                       dynamic_cast<const RelLeftDeepInnerJoin*>(only_src);
  return is_join ? only_src : ra_node;
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelCompound* compound) {
  RexUsedInputsVisitor visitor;
  const auto filter_expr = compound->getFilterExpr();
  std::unordered_set<const RexInput*> used_inputs =
      filter_expr ? visitor.visit(filter_expr) : std::unordered_set<const RexInput*>{};
  const auto sources_size = compound->getScalarSourcesSize();
  for (size_t i = 0; i < sources_size; ++i) {
    const auto source_inputs = visitor.visit(compound->getScalarSource(i));
    used_inputs.insert(source_inputs.begin(), source_inputs.end());
  }
  return std::make_pair(used_inputs, std::vector<std::shared_ptr<RexInput>>{});
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelAggregate* aggregate) {
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
    const auto operand_idx = agg_expr->getOperand();
    const bool takes_arg{operand_idx >= 0};
    if (takes_arg) {
      CHECK_GE(in_metainfo.size(), static_cast<size_t>(operand_idx));
      auto synthesized_used_input = new RexInput(source, operand_idx);
      used_inputs_owned.emplace_back(synthesized_used_input);
      used_inputs.insert(synthesized_used_input);
    }
  }
  return std::make_pair(used_inputs, used_inputs_owned);
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelProject* project) {
  RexUsedInputsVisitor visitor;
  std::unordered_set<const RexInput*> used_inputs;
  for (size_t i = 0; i < project->size(); ++i) {
    const auto proj_inputs = visitor.visit(project->getProjectAt(i));
    used_inputs.insert(proj_inputs.begin(), proj_inputs.end());
  }
  return std::make_pair(used_inputs, std::vector<std::shared_ptr<RexInput>>{});
}

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelFilter* filter) {
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

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>> get_used_inputs(
    const RelJoin* join) {
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto lhs = join->getInput(0);
  if (dynamic_cast<const RelJoin*>(lhs)) {
    auto synthesized_used_input = new RexInput(lhs, 0);
    used_inputs_owned.emplace_back(synthesized_used_input);
    used_inputs.insert(synthesized_used_input);
    for (auto previous_join = static_cast<const RelJoin*>(lhs); previous_join;
         previous_join = dynamic_cast<const RelJoin*>(previous_join->getInput(0))) {
      synthesized_used_input = new RexInput(lhs, 0);
      used_inputs_owned.emplace_back(synthesized_used_input);
      used_inputs.insert(synthesized_used_input);
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

std::unordered_map<const RelAlgNode*, int> get_input_nest_levels(const RelAlgNode* ra_node,
                                                                 const std::vector<size_t>& input_permutation) {
  const auto data_sink_node = get_data_sink(ra_node);
  std::unordered_map<const RelAlgNode*, int> input_to_nest_level;
  for (size_t input_idx = 0; input_idx < data_sink_node->inputCount(); ++input_idx) {
    const auto input_ra = data_sink_node->getInput(input_idx);
    size_t nest_level = input_permutation.empty() ? input_idx : input_permutation[input_idx];
    const auto it_ok = input_to_nest_level.emplace(input_ra, nest_level);
    CHECK(it_ok.second);
  }
  return input_to_nest_level;
}

std::unordered_set<const RexInput*> get_join_source_used_inputs(const RelAlgNode* ra_node) {
  const auto data_sink_node = get_data_sink(ra_node);
  if (auto join = dynamic_cast<const RelJoin*>(data_sink_node)) {
    CHECK_EQ(join->inputCount(), 2);
    const auto condition = join->getCondition();
    RexUsedInputsVisitor visitor;
    return visitor.visit(condition);
  }

  if (auto multi_join = dynamic_cast<const RelMultiJoin*>(data_sink_node)) {
    CHECK_GT(multi_join->inputCount(), 2);
    RexUsedInputsVisitor visitor;
    std::unordered_set<const RexInput*> input_set;
    for (auto& condition : multi_join->getConditions()) {
      auto inputs = visitor.visit(condition.get());
      input_set.insert(inputs.begin(), inputs.end());
    }
    return input_set;
  }

  if (auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(data_sink_node)) {
    CHECK_GE(left_deep_join->inputCount(), 2);
    const auto condition = left_deep_join->getCondition();
    RexUsedInputsVisitor visitor;
    return visitor.visit(condition);
  }

  CHECK_EQ(ra_node->inputCount(), 1);
  return std::unordered_set<const RexInput*>{};
}

size_t get_target_list_size(const RelAlgNode* ra_node) {
  const auto scan = dynamic_cast<const RelScan*>(ra_node);
  if (scan) {
    return scan->getFieldNames().size();
  }
  const auto join = dynamic_cast<const RelJoin*>(ra_node);
  if (join) {
    return get_target_list_size(join->getInput(0)) + get_target_list_size(join->getInput(1));
  }
  const auto aggregate = dynamic_cast<const RelAggregate*>(ra_node);
  if (aggregate) {
    return aggregate->getFields().size();
  }
  const auto compound = dynamic_cast<const RelCompound*>(ra_node);
  if (compound) {
    return compound->getFields().size();
  }
  const auto filter = dynamic_cast<const RelFilter*>(ra_node);
  if (filter) {
    return get_target_list_size(filter->getInput(0));
  }
  const auto project = dynamic_cast<const RelProject*>(ra_node);
  if (project) {
    return project->getFields().size();
  }
  const auto sort = dynamic_cast<const RelSort*>(ra_node);
  if (sort) {
    return get_target_list_size(sort->getInput(0));
  }
  CHECK(false);
  return 0;
}

std::vector<const RelAlgNode*> get_non_join_sequence(const RelAlgNode* ra) {
  std::vector<const RelAlgNode*> seq;
  if (auto multi_join = dynamic_cast<const RelMultiJoin*>(ra)) {
    CHECK_GT(multi_join->joinCount(), size_t(1));
    seq = get_non_join_sequence(multi_join->getJoinAt(0)->getInput(0));
    for (size_t i = 0; i < multi_join->joinCount(); ++i) {
      seq.push_back(multi_join->getJoinAt(i)->getInput(1));
    }
    return seq;
  }
  for (auto join = dynamic_cast<const RelJoin*>(ra); join; join = static_cast<const RelJoin*>(join->getInput(0))) {
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

std::pair<const RelAlgNode*, int> get_non_join_source_node(const RelAlgNode* crt_source, const int col_id) {
  CHECK_LE(0, col_id);
  const auto join = dynamic_cast<const RelJoin*>(crt_source);
  if (!join) {
    return std::make_pair(crt_source, col_id);
  }
  const auto lhs = join->getInput(0);
  const auto rhs = join->getInput(1);
  const size_t left_source_size = get_target_list_size(lhs);
  if (size_t(col_id) >= left_source_size) {
    return std::make_pair(rhs, col_id - int(left_source_size));
  }
  if (dynamic_cast<const RelJoin*>(lhs)) {
    return get_non_join_source_node(static_cast<const RelJoin*>(lhs), col_id);
  }
  return std::make_pair(lhs, col_id);
}

void collect_used_input_desc(std::vector<InputDescriptor>& input_descs,
                             std::unordered_set<std::shared_ptr<const InputColDescriptor>>& input_col_descs_unique,
                             const RelAlgNode* ra_node,
                             const std::unordered_set<const RexInput*>& source_used_inputs,
                             const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  std::unordered_set<InputDescriptor> input_descs_unique(input_descs.begin(), input_descs.end());
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
      throw std::runtime_error("Multi-way join not supported");
    }
    const int input_desc = it->second;

    const RelAlgNode* indirect_input_ra{nullptr};
    int indirect_col_id{-1};
    std::tie(indirect_input_ra, indirect_col_id) = get_non_join_source_node(input_ra, col_id);
    if (indirect_input_ra == input_ra) {
      CHECK_EQ(indirect_col_id, static_cast<ssize_t>(col_id));
      input_col_descs_unique.insert(std::make_shared<const InputColDescriptor>(
          dynamic_cast<const RelScan*>(input_ra) ? col_id + 1 : col_id, table_id, input_desc));
      continue;
    }

    // A column from indirect source indexed by an iterator
    const int indirect_table_id = table_id_from_ra(indirect_input_ra);
    CHECK(!input_to_nest_level.count(indirect_input_ra));
    it = non_join_to_nest_level.find(indirect_input_ra);
    CHECK(it != non_join_to_nest_level.end());
    const int nest_level = it->second;
    if (!input_descs_unique.count(InputDescriptor(indirect_table_id, -1))) {
      input_descs_unique.emplace(indirect_table_id, -1);
      input_descs.emplace_back(indirect_table_id, -1);
    }
    CHECK(!dynamic_cast<const RelScan*>(input_ra));
    CHECK_EQ(size_t(0), static_cast<size_t>(input_desc));
    // Physical columns from a scan node are numbered from 1 in our system.
    input_col_descs_unique.insert(std::make_shared<const IndirectInputColDescriptor>(
        col_id,
        table_id,
        input_desc,
        nest_level,
        table_id,
        input_desc,
        dynamic_cast<const RelScan*>(indirect_input_ra) ? indirect_col_id + 1 : indirect_col_id,
        indirect_table_id,
        nest_level));
  }
}

template <class RA>
std::pair<std::vector<InputDescriptor>, std::list<std::shared_ptr<const InputColDescriptor>>> get_input_desc_impl(
    const RA* ra_node,
    const std::unordered_set<const RexInput*>& used_inputs,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const std::vector<size_t>& input_permutation) {
  std::vector<InputDescriptor> input_descs;
  const auto data_sink_node = get_data_sink(ra_node);
  for (size_t input_idx = 0; input_idx < data_sink_node->inputCount(); ++input_idx) {
    const auto input_ra = data_sink_node->getInput(input_idx);
    const int table_id = table_id_from_ra(input_ra);
    const auto nest_level = input_permutation.empty() ? input_idx : input_permutation[input_idx];
    input_descs.emplace_back(table_id, nest_level);
  }
  std::sort(input_descs.begin(), input_descs.end(), [](const InputDescriptor& lhs, const InputDescriptor& rhs) {
    return lhs.getNestLevel() < rhs.getNestLevel();
  });
  std::unordered_set<std::shared_ptr<const InputColDescriptor>> input_col_descs_unique;
  collect_used_input_desc(input_descs, input_col_descs_unique, ra_node, used_inputs, input_to_nest_level);
  collect_used_input_desc(
      input_descs, input_col_descs_unique, ra_node, get_join_source_used_inputs(ra_node), input_to_nest_level);
  std::vector<std::shared_ptr<const InputColDescriptor>> input_col_descs(input_col_descs_unique.begin(),
                                                                         input_col_descs_unique.end());

  std::sort(
      input_col_descs.begin(),
      input_col_descs.end(),
      [](std::shared_ptr<const InputColDescriptor> const& lhs, std::shared_ptr<const InputColDescriptor> const& rhs) {
        if (lhs->getScanDesc().getNestLevel() == rhs->getScanDesc().getNestLevel()) {
          return lhs->getColId() < rhs->getColId();
        }
        return lhs->getScanDesc().getNestLevel() < rhs->getScanDesc().getNestLevel();
      });
  return {input_descs,
          std::list<std::shared_ptr<const InputColDescriptor>>(input_col_descs.begin(), input_col_descs.end())};
}

template <class RA>
std::tuple<std::vector<InputDescriptor>,
           std::list<std::shared_ptr<const InputColDescriptor>>,
           std::vector<std::shared_ptr<RexInput>>>
get_input_desc(const RA* ra_node,
               const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
               const std::vector<size_t>& input_permutation) {
  std::unordered_set<const RexInput*> used_inputs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  std::tie(used_inputs, used_inputs_owned) = get_used_inputs(ra_node);
  auto input_desc_pair = get_input_desc_impl(ra_node, used_inputs, input_to_nest_level, input_permutation);
  return std::make_tuple(input_desc_pair.first, input_desc_pair.second, used_inputs_owned);
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

std::shared_ptr<Analyzer::Expr> set_transient_dict(const std::shared_ptr<Analyzer::Expr> expr) {
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
std::vector<std::shared_ptr<Analyzer::Expr>> translate_scalar_sources(const RA* ra_node,
                                                                      const RelAlgTranslator& translator) {
  std::vector<std::shared_ptr<Analyzer::Expr>> scalar_sources;
  for (size_t i = 0; i < get_scalar_sources_size(ra_node); ++i) {
    const auto scalar_rex = scalar_at(i, ra_node);
    if (dynamic_cast<const RexRef*>(scalar_rex)) {
      // RexRef are synthetic scalars we append at the end of the real ones
      // for the sake of taking memory ownership, no real work needed here.
      continue;
    }
    const auto scalar_expr = translator.translateScalarRex(scalar_rex);
    const auto folded_scalar_expr = fold_expr(scalar_expr.get());
    scalar_sources.push_back(folded_scalar_expr);
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

QualsConjunctiveForm translate_quals(const RelCompound* compound, const RelAlgTranslator& translator) {
  const auto filter_rex = compound->getFilterExpr();
  const auto filter_expr = filter_rex ? translator.translateScalarRex(filter_rex) : nullptr;
  return filter_expr ? qual_to_conjunctive_form(fold_expr(filter_expr.get())) : QualsConjunctiveForm{};
}

std::vector<Analyzer::Expr*> translate_targets(std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
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
      target_expr = RelAlgTranslator::translateAggregateRex(target_rex_agg, scalar_sources);
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
        target_expr = fold_expr(target_expr.get());
      }
    }
    CHECK(target_expr);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}

std::vector<Analyzer::Expr*> translate_targets(std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
                                               const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
                                               const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
                                               const RelAggregate* aggregate,
                                               const RelAlgTranslator& translator) {
  std::vector<Analyzer::Expr*> target_exprs;
  size_t group_key_idx = 0;
  for (const auto& groupby_expr : groupby_exprs) {
    auto target_expr = var_ref(groupby_expr.get(), Analyzer::Var::kGROUPBY, group_key_idx++);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }

  for (const auto& target_rex_agg : aggregate->getAggExprs()) {
    auto target_expr = RelAlgTranslator::translateAggregateRex(target_rex_agg.get(), scalar_sources);
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

template <class RA>
std::vector<TargetMetaInfo> get_targets_meta(const RA* ra_node, const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<TargetMetaInfo> targets_meta;
  for (size_t i = 0; i < ra_node->size(); ++i) {
    CHECK(target_exprs[i]);
    // TODO(alex): remove the count distinct type fixup.
    targets_meta.emplace_back(
        ra_node->getFieldName(i),
        is_count_distinct(target_exprs[i]) ? SQLTypeInfo(kBIGINT, false) : target_exprs[i]->get_type_info());
  }
  return targets_meta;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeCompound(const RelCompound* compound,
                                                const CompilationOptions& co,
                                                const ExecutionOptions& eo,
                                                RenderInfo* render_info,
                                                const int64_t queue_time_ms) {
  const auto work_unit = createCompoundWorkUnit(compound, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  return executeWorkUnit(
      work_unit, compound->getOutputMetainfo(), compound->isAggregate(), co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeAggregate(const RelAggregate* aggregate,
                                                 const CompilationOptions& co,
                                                 const ExecutionOptions& eo,
                                                 RenderInfo* render_info,
                                                 const int64_t queue_time_ms) {
  const auto work_unit = createAggregateWorkUnit(aggregate, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  return executeWorkUnit(work_unit, aggregate->getOutputMetainfo(), true, co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeProject(const RelProject* project,
                                               const CompilationOptions& co,
                                               const ExecutionOptions& eo,
                                               RenderInfo* render_info,
                                               const int64_t queue_time_ms) {
  auto work_unit = createProjectWorkUnit(project, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  CompilationOptions co_project = co;
  if (project->isSimple()) {
    CHECK_EQ(size_t(1), project->inputCount());
    const auto input_ra = project->getInput(0);
    if (dynamic_cast<const RelSort*>(input_ra)) {
      co_project.device_type_ = ExecutorDeviceType::CPU;
      const auto& input_table = get_temporary_table(&temporary_tables_, -input_ra->getId());
      const auto input_rows = boost::get<RowSetPtr>(&input_table);
      CHECK(input_rows && *input_rows);
      work_unit.exe_unit.scan_limit = (*input_rows)->rowCount();
    }
  }
  return executeWorkUnit(work_unit, project->getOutputMetainfo(), false, co_project, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeFilter(const RelFilter* filter,
                                              const CompilationOptions& co,
                                              const ExecutionOptions& eo,
                                              RenderInfo* render_info,
                                              const int64_t queue_time_ms) {
  const auto work_unit = createFilterWorkUnit(filter, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  return executeWorkUnit(work_unit, filter->getOutputMetainfo(), false, co, eo, render_info, queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeJoin(const RelJoin* join,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo,
                                            RenderInfo* render_info,
                                            const int64_t queue_time_ms) {
  const auto work_unit = createJoinWorkUnit(join, {{}, SortAlgorithm::Default, 0, 0}, eo.just_explain);
  return executeWorkUnit(work_unit, join->getOutputMetainfo(), false, co, eo, render_info, queue_time_ms);
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
  while (true) {
    std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
    bool is_desc{false};
    try {
      const auto source_work_unit = createSortInputWorkUnit(sort, eo.just_explain);
      is_desc = first_oe_is_desc(source_work_unit.exe_unit.sort_info.order_entries);
      groupby_exprs = source_work_unit.exe_unit.groupby_exprs;
      auto source_result = executeWorkUnit(
          source_work_unit, source->getOutputMetainfo(), is_aggregate, co, eo, render_info, queue_time_ms);
      if (render_info && render_info->do_render) {
        return source_result;
      }
      auto rows_to_sort = source_result.getRows();
      if (eo.just_explain) {
        return {rows_to_sort, {}};
      }
      const size_t limit = sort->getLimit();
      const size_t offset = sort->getOffset();
      if (sort->collationCount() != 0 && !rows_to_sort->definitelyHasNoRows() &&
          !use_speculative_top_n(source_work_unit.exe_unit, rows_to_sort->getQueryMemDesc())) {
        rows_to_sort->sort(source_work_unit.exe_unit.sort_info.order_entries, limit + offset);
      }
      if (limit || offset) {
        if (g_cluster && sort->collationCount() == 0) {
          rows_to_sort->keepFirstN(limit + offset);
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
  return {std::make_shared<ResultSet>(
              std::vector<TargetInfo>{}, co.device_type_, QueryMemoryDescriptor{}, nullptr, executor_),
          {}};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createSortInputWorkUnit(const RelSort* sort, const bool just_explain) {
  const auto source = sort->getInput(0);
  const size_t limit = sort->getLimit();
  const size_t offset = sort->getOffset();
  const size_t scan_limit = sort->collationCount() ? 0 : get_scan_limit(source, limit);
  const size_t scan_total_limit = scan_limit ? get_scan_limit(source, scan_limit + offset) : 0;
  size_t max_groups_buffer_entry_guess{scan_total_limit ? scan_total_limit : max_groups_buffer_entry_default_guess};
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
           source_exe_unit.extra_input_descs,
           std::move(source_exe_unit.input_col_descs),
           source_exe_unit.simple_quals,
           source_exe_unit.quals,
           source_exe_unit.join_type,
           source_exe_unit.inner_joins,
           source_exe_unit.join_dimensions,
           source_exe_unit.inner_join_quals,
           source_exe_unit.outer_join_quals,
           source_exe_unit.groupby_exprs,
           source_exe_unit.target_exprs,
           source_exe_unit.orig_target_exprs,
           nullptr,
           {sort_info.order_entries, sort_algorithm, limit, offset},
           scan_total_limit},
          source,
          max_groups_buffer_entry_guess,
          std::move(source_work_unit.query_rewriter)};
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
    const auto arg_range =
        getExpressionRange(redirect_expr(arg.get(), ra_exe_unit.input_col_descs).get(), table_infos, executor);
    if (arg_range.getType() != ExpressionRangeType::Integer) {
      continue;
    }
    // When running distributed, the threshold for using the precise implementation
    // must be consistent across all leaves, otherwise we could have a mix of precise
    // and approximate bitmaps and we cannot aggregate them.
    const auto device_type = g_cluster ? ExecutorDeviceType::GPU : device_type_in;
    const auto bitmap_sz_bits = arg_range.getIntMax() - arg_range.getIntMin() + 1;
    const auto sub_bitmap_count = get_count_distinct_sub_bitmap_count(bitmap_sz_bits, ra_exe_unit, device_type);
    CountDistinctDescriptor approx_count_distinct_desc{CountDistinctImplType::Bitmap,
                                                       arg_range.getIntMin(),
                                                       g_hll_precision_bits,
                                                       true,
                                                       device_type,
                                                       sub_bitmap_count};
    CountDistinctDescriptor precise_count_distinct_desc{
        CountDistinctImplType::Bitmap, arg_range.getIntMin(), bitmap_sz_bits, false, device_type, sub_bitmap_count};
    if (approx_count_distinct_desc.bitmapPaddedSizeBytes() >= precise_count_distinct_desc.bitmapPaddedSizeBytes()) {
      auto precise_count_distinct = makeExpr<Analyzer::AggExpr>(get_agg_type(kCOUNT, arg.get()), kCOUNT, arg, true);
      target_exprs_owned.push_back(precise_count_distinct);
      ra_exe_unit.target_exprs[i] = precise_count_distinct.get();
    }
  }
  return ra_exe_unit;
}

}  // namespace

ExecutionResult RelAlgExecutor::executeWorkUnit(const RelAlgExecutor::WorkUnit& work_unit,
                                                const std::vector<TargetMetaInfo>& targets_meta,
                                                const bool is_agg,
                                                const CompilationOptions& co,
                                                const ExecutionOptions& eo,
                                                RenderInfo* render_info,
                                                const int64_t queue_time_ms) {
  const auto body = work_unit.body;
  CHECK(body);
  int32_t error_code{0};
  if (render_info && render_info->do_render) {
    if (co.device_type_ != ExecutorDeviceType::GPU) {
      throw std::runtime_error("Backend rendering is only supported on GPU");
    }
    if (!executor_->render_manager_) {
      throw std::runtime_error("This build doesn't support backend rendering");
    }

    if (!render_info->render_allocator_map_ptr) {
      // for backwards compatibility, can be removed when MapDHandler::render(...)
      // in MapDServer.cpp is removed
      render_info->render_allocator_map_ptr.reset(new RenderAllocatorMap(
          cat_.get_dataMgr().cudaMgr_, executor_->render_manager_, executor_->blockSize(), executor_->gridSize()));
    }
  }

  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);

  auto ra_exe_unit = decide_approx_count_distinct_implementation(
      work_unit.exe_unit, table_infos, executor_, co.device_type_, target_exprs_owned_);
  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;

  if (!eo.just_explain && can_use_scan_limit(ra_exe_unit) && !isRowidLookup(work_unit)) {
    const auto filter_count_all = getFilteredCountAll(work_unit, true, co, eo);
    if (filter_count_all >= 0) {
      ra_exe_unit.scan_limit = filter_count_all;
    }
  }

  static const size_t big_group_threshold{20000};

  ExecutionResult result{std::make_shared<ResultSet>(
                             std::vector<TargetInfo>{}, co.device_type_, QueryMemoryDescriptor{}, nullptr, executor_),
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
                  (render_info && render_info->do_render ? render_info->render_allocator_map_ptr.get() : nullptr),
                  groups_approx_upper_bound(table_infos) <= big_group_threshold),
              targets_meta};
  } catch (const CardinalityEstimationRequired&) {
    max_groups_buffer_entry_guess =
        2 * std::min(groups_approx_upper_bound(table_infos), getNDVEstimation(work_unit, is_agg, co, eo));
    CHECK_GT(max_groups_buffer_entry_guess, size_t(0));
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
                  (render_info && render_info->do_render ? render_info->render_allocator_map_ptr.get() : nullptr),
                  true),
              targets_meta};
  }

  result.setQueueTime(queue_time_ms);
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

size_t RelAlgExecutor::getNDVEstimation(const WorkUnit& work_unit,
                                        const bool is_agg,
                                        const CompilationOptions& co,
                                        const ExecutionOptions& eo) {
  const auto estimator_exe_unit = create_ndv_execution_unit(work_unit.exe_unit);
  int32_t error_code{0};
  size_t one{1};
  const auto estimator_result = executor_->executeWorkUnit(&error_code,
                                                           one,
                                                           is_agg,
                                                           get_table_infos(work_unit.exe_unit, executor_),
                                                           estimator_exe_unit,
                                                           co,
                                                           eo,
                                                           cat_,
                                                           executor_->row_set_mem_owner_,
                                                           nullptr,
                                                           false);
  if (error_code == Executor::ERR_OUT_OF_TIME) {
    throw std::runtime_error("Cardinality estimation query ran out of time");
  }
  if (error_code == Executor::ERR_INTERRUPTED) {
    throw std::runtime_error("Cardinality estimation query has been interrupted");
  }
  if (error_code) {
    throw std::runtime_error("Failed to run the cardinality estimation query: " + getErrorMessageFromCode(error_code));
  }
  const auto& estimator_result_rows = boost::get<RowSetPtr>(estimator_result);
  if (!estimator_result_rows) {
    return 1;
  }
  return std::max(estimator_result_rows->getNDVEstimator(), size_t(1));
}

ssize_t RelAlgExecutor::getFilteredCountAll(const WorkUnit& work_unit,
                                            const bool is_agg,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo) {
  const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);
  if (table_infos.size() == 1 && table_infos.front().info.getNumTuplesUpperBound() <= 50000) {
    return table_infos.front().info.getNumTuplesUpperBound();
  }
  const auto count =
      makeExpr<Analyzer::AggExpr>(SQLTypeInfo(g_bigint_count ? kBIGINT : kINT, false), kCOUNT, nullptr, false);
  const auto count_all_exe_unit = create_count_all_execution_unit(work_unit.exe_unit, count);
  int32_t error_code{0};
  size_t one{1};
  ResultPtr count_all_result;
  try {
    count_all_result = executor_->executeWorkUnit(&error_code,
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
  const auto& count_all_result_rows = boost::get<RowSetPtr>(count_all_result);
  CHECK(count_all_result_rows);
  const auto count_row = count_all_result_rows->getNextRow(false, false);
  CHECK_EQ(size_t(1), count_row.size());
  const auto& count_tv = count_row.front();
  const auto count_scalar_tv = boost::get<ScalarTargetValue>(&count_tv);
  CHECK(count_scalar_tv);
  const auto count_ptr = boost::get<int64_t>(count_scalar_tv);
  CHECK(count_ptr);
  return std::max(*count_ptr, int64_t(1));
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
    const auto comp_expr = std::dynamic_pointer_cast<const Analyzer::BinOper>(simple_qual);
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

ExecutionResult RelAlgExecutor::handleRetry(const int32_t error_code_in,
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
                                   eo.dynamic_watchdog_time_limit};
  ExecutionResult result{std::make_shared<ResultSet>(
                             std::vector<TargetInfo>{}, co.device_type_, QueryMemoryDescriptor{}, nullptr, executor_),
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
  CompilationOptions co_cpu{ExecutorDeviceType::CPU, co.hoist_literals_, co.opt_level_, co.with_dynamic_watchdog_};
  if (error_code) {
    max_groups_buffer_entry_guess = 0;
    while (true) {
      const auto ra_exe_unit = decide_approx_count_distinct_implementation(
          work_unit.exe_unit, table_infos, executor_, co_cpu.device_type_, target_exprs_owned_);
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
  if (error_code == Executor::ERR_OUT_OF_GPU_MEM && (!g_enable_watchdog || g_allow_cpu_retry)) {
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

class UsedTablesVisitor : public ScalarExprVisitor<std::unordered_set<int>> {
 protected:
  virtual std::unordered_set<int> visitColumnVar(const Analyzer::ColumnVar* column) const override {
    return {column->get_table_id()};
  }

  virtual std::unordered_set<int> aggregateResult(const std::unordered_set<int>& aggregate,
                                                  const std::unordered_set<int>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

struct SeparatedQuals {
  const std::list<std::shared_ptr<Analyzer::Expr>> regular_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
};

SeparatedQuals separate_join_quals(const std::list<std::shared_ptr<Analyzer::Expr>>& all_quals) {
  std::list<std::shared_ptr<Analyzer::Expr>> regular_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
  UsedTablesVisitor qual_visitor;
  for (auto qual_candidate : all_quals) {
    const auto used_table_ids = qual_visitor.visit(qual_candidate.get());
    if (used_table_ids.size() > 1) {
      CHECK_EQ(size_t(2), used_table_ids.size());
      join_quals.push_back(qual_candidate);
    } else {
      const auto rewritten_qual_candidate = rewrite_expr(qual_candidate.get());
      regular_quals.push_back(rewritten_qual_candidate ? rewritten_qual_candidate : qual_candidate);
    }
  }
  return {regular_quals, join_quals};
}

JoinType get_join_type(const RelAlgNode* ra) {
  auto sink = get_data_sink(ra);
  if (auto join = dynamic_cast<const RelJoin*>(sink)) {
    return join->getJoinType();
  }
  if (auto multi_join = dynamic_cast<const RelMultiJoin*>(sink)) {
    return multi_join->getJoinType();
  }
  if (dynamic_cast<const RelLeftDeepInnerJoin*>(sink)) {
    return JoinType::INNER;
  }

  return JoinType::INVALID;
}

bool is_literal_true(const RexScalar* condition) {
  CHECK(condition);
  const auto literal = dynamic_cast<const RexLiteral*>(condition);
  return literal && literal->getType() == kBOOLEAN && literal->getVal<bool>();
}

std::list<std::shared_ptr<Analyzer::Expr>> get_outer_join_quals(const RelAlgNode* ra,
                                                                const RelAlgTranslator& translator) {
  const auto join = dynamic_cast<const RelJoin*>(ra) ? static_cast<const RelJoin*>(ra)
                                                     : dynamic_cast<const RelJoin*>(ra->getInput(0));
  if (join && join->getCondition() && !is_literal_true(join->getCondition()) && join->getJoinType() == JoinType::LEFT) {
    const auto join_cond_cf = qual_to_conjunctive_form(translator.translateScalarRex(join->getCondition()));
    if (join_cond_cf.simple_quals.empty()) {
      return join_cond_cf.quals;
    }
    std::list<std::shared_ptr<Analyzer::Expr>> all_quals = join_cond_cf.simple_quals;
    all_quals.insert(all_quals.end(), join_cond_cf.quals.begin(), join_cond_cf.quals.end());
    return all_quals;
  }
  return {};
}

std::unique_ptr<const RexOperator> get_bitwise_equals(const RexScalar* scalar) {
  const auto condition = dynamic_cast<const RexOperator*>(scalar);
  if (!condition || condition->getOperator() != kOR || condition->size() != 2) {
    return nullptr;
  }
  const auto equi_join_condition = dynamic_cast<const RexOperator*>(condition->getOperand(0));
  if (!equi_join_condition || equi_join_condition->getOperator() != kEQ) {
    return nullptr;
  }
  const auto both_are_null_condition = dynamic_cast<const RexOperator*>(condition->getOperand(1));
  if (!both_are_null_condition || both_are_null_condition->getOperator() != kAND ||
      both_are_null_condition->size() != 2) {
    return nullptr;
  }
  const auto lhs_is_null = dynamic_cast<const RexOperator*>(both_are_null_condition->getOperand(0));
  const auto rhs_is_null = dynamic_cast<const RexOperator*>(both_are_null_condition->getOperand(1));
  if (!lhs_is_null || !rhs_is_null || lhs_is_null->getOperator() != kISNULL || rhs_is_null->getOperator() != kISNULL) {
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
    eq_operands.emplace_back(equi_join_condition->getOperandAndRelease(0));
    eq_operands.emplace_back(equi_join_condition->getOperandAndRelease(1));
    return boost::make_unique<const RexOperator>(kBW_EQ, eq_operands, equi_join_condition->getType());
  }
  return nullptr;
}

std::unique_ptr<const RexOperator> get_bitwise_equals_conjunction(const RexScalar* scalar) {
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
      acc = boost::make_unique<const RexOperator>(kAND, and_operands, condition->getType());
    }
    return acc;
  }
  return get_bitwise_equals(scalar);
}

std::list<std::shared_ptr<Analyzer::Expr>> get_inner_join_quals(const RelAlgNode* ra,
                                                                const RelAlgTranslator& translator) {
  std::vector<const RexScalar*> work_set;
  if (auto join = dynamic_cast<const RelJoin*>(ra)) {
    if (join->getJoinType() == JoinType::INNER) {
      work_set.push_back(join->getCondition());
    }
  } else {
    CHECK_EQ(size_t(1), ra->inputCount());
    auto only_src = ra->getInput(0);
    if (auto join = dynamic_cast<const RelJoin*>(only_src)) {
      if (join->getJoinType() == JoinType::INNER) {
        work_set.push_back(join->getCondition());
      }
    } else if (auto multi_join = dynamic_cast<const RelMultiJoin*>(only_src)) {
      CHECK_EQ(multi_join->joinCount(), multi_join->getConditions().size());
      for (size_t i = 0; i < multi_join->joinCount(); ++i) {
        if (multi_join->getJoinAt(i)->getJoinType() == JoinType::INNER) {
          work_set.push_back(multi_join->getConditions()[i].get());
        }
      }
    }
  }
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  for (auto condition : work_set) {
    if (condition && !is_literal_true(condition)) {
      const auto bw_equals = get_bitwise_equals_conjunction(condition);
      const auto eq_condition = bw_equals ? bw_equals.get() : condition;
      const auto join_cond_cf = qual_to_conjunctive_form(translator.translateScalarRex(eq_condition));
      quals.insert(quals.end(), join_cond_cf.simple_quals.begin(), join_cond_cf.simple_quals.end());
      quals.insert(quals.end(), join_cond_cf.quals.begin(), join_cond_cf.quals.end());
    }
  }
  return combine_equi_join_conditions(quals);
}

std::vector<std::pair<int, size_t>> get_join_dimensions(const RelAlgNode* ra, Executor* executor) {
  std::vector<std::pair<int, size_t>> dims;
  if (auto mj = dynamic_cast<const RelMultiJoin*>(ra)) {
    ra = mj->getJoinAt(mj->joinCount() - 1);
  }
  for (auto join = dynamic_cast<const RelJoin*>(ra); join; join = static_cast<const RelJoin*>(join->getInput(0))) {
    CHECK_EQ(size_t(2), join->inputCount());
    const auto id = table_id_from_ra(join->getInput(1));
    dims.emplace_back(id, get_frag_count_of_table(id, executor));
    auto lhs = join->getInput(0);
    if (!dynamic_cast<const RelJoin*>(lhs)) {
      const auto id = table_id_from_ra(lhs);
      dims.emplace_back(id, get_frag_count_of_table(id, executor));
      break;
    }
  }
  std::reverse(dims.begin(), dims.end());
  return dims;
}

std::vector<InputDescriptor> separate_extra_input_descs(std::vector<InputDescriptor>& input_descs) {
  std::vector<InputDescriptor> new_input_descs;
  std::vector<InputDescriptor> extra_input_descs;

  for (const auto& input_desc : input_descs) {
    if (input_desc.getNestLevel() < 0) {
      extra_input_descs.push_back(input_desc);
    } else {
      new_input_descs.push_back(input_desc);
    }
  }

  input_descs.swap(new_input_descs);
  return extra_input_descs;
}

std::vector<size_t> get_node_input_permutation(const std::vector<InputTableInfo>& table_infos) {
  std::vector<size_t> input_permutation(table_infos.size());
  std::iota(input_permutation.begin(), input_permutation.end(), 0);
  std::sort(input_permutation.begin(),
            input_permutation.end(),
            [&table_infos](const size_t lhs_index, const size_t rhs_index) {
              return table_infos[lhs_index].info.getNumTuples() > table_infos[rhs_index].info.getNumTuples();
            });
  return input_permutation;
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createCompoundWorkUnit(const RelCompound* compound,
                                                                const SortInfo& sort_info,
                                                                const bool just_explain) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  auto input_to_nest_level = get_input_nest_levels(compound, {});
  std::tie(input_descs, input_col_descs, std::ignore) = get_input_desc(compound, input_to_nest_level, {});
  const auto query_infos = get_table_infos(input_descs, executor_);
  CHECK_EQ(size_t(1), compound->inputCount());
  const auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(compound->getInput(0));
  JoinQualsPerNestingLevel left_deep_inner_joins;
  if (left_deep_join) {
    const auto input_permutation = get_node_input_permutation(query_infos);
    input_to_nest_level = get_input_nest_levels(compound, input_permutation);
    std::tie(input_descs, input_col_descs, std::ignore) =
        get_input_desc(compound, input_to_nest_level, input_permutation);
    left_deep_inner_joins = translateLeftDeepJoinFilter(left_deep_join, input_descs, input_to_nest_level, just_explain);
  }
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = left_deep_join ? JoinType::INVALID : get_join_type(compound);
  RelAlgTranslator translator(cat_, executor_, input_to_nest_level, join_type, now_, just_explain);
  const auto scalar_sources = translate_scalar_sources(compound, translator);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals_cf = translate_quals(compound, translator);
  const auto separated_quals =
      join_type == JoinType::LEFT ? SeparatedQuals{quals_cf.quals, {}} : separate_join_quals(quals_cf.quals);
  const auto simple_separated_quals = separate_join_quals(quals_cf.simple_quals);
  CHECK(simple_separated_quals.join_quals.empty());
  const auto target_exprs = translate_targets(target_exprs_owned_, scalar_sources, groupby_exprs, compound, translator);
  CHECK_EQ(compound->size(), target_exprs.size());
  auto inner_join_quals = get_inner_join_quals(compound, translator);
  inner_join_quals.insert(inner_join_quals.end(), separated_quals.join_quals.begin(), separated_quals.join_quals.end());
  const RelAlgExecutionUnit exe_unit = {input_descs,
                                        extra_input_descs,
                                        input_col_descs,
                                        quals_cf.simple_quals,
                                        separated_quals.regular_quals,
                                        join_type,
                                        left_deep_inner_joins,
                                        get_join_dimensions(get_data_sink(compound), executor_),
                                        inner_join_quals,
                                        get_outer_join_quals(compound, translator),
                                        groupby_exprs,
                                        target_exprs,
                                        {},
                                        nullptr,
                                        sort_info,
                                        0};
  QueryRewriter* query_rewriter = new QueryRewriter(exe_unit, query_infos, executor_, nullptr);
  const auto rewritten_exe_unit = query_rewriter->rewrite();
  const auto targets_meta = get_targets_meta(compound, rewritten_exe_unit.target_exprs);
  compound->setOutputMetainfo(targets_meta);
  return {rewritten_exe_unit,
          compound,
          max_groups_buffer_entry_default_guess,
          std::unique_ptr<QueryRewriter>(query_rewriter)};
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

}  // namespace

// Translate left deep join filter and separate the conjunctive form qualifiers
// per nesting level. The code generated for hash table lookups on each level
// must dominate its uses in deeper nesting levels.
JoinQualsPerNestingLevel RelAlgExecutor::translateLeftDeepJoinFilter(
    const RelLeftDeepInnerJoin* join,
    const std::vector<InputDescriptor>& input_descs,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const bool just_explain) {
  RelAlgTranslator translator(cat_, executor_, input_to_nest_level, JoinType::INNER, now_, just_explain);
  const auto rex_condition_cf = rex_to_conjunctive_form(join->getCondition());
  std::list<std::shared_ptr<Analyzer::Expr>> join_condition_quals;
  for (const auto rex_condition_component : rex_condition_cf) {
    const auto bw_equals = get_bitwise_equals_conjunction(rex_condition_component);
    const auto join_condition = translator.translateScalarRex(bw_equals ? bw_equals.get() : rex_condition_component);
    auto join_condition_cf = qual_to_conjunctive_form(join_condition);
    join_condition_quals.insert(
        join_condition_quals.end(), join_condition_cf.quals.begin(), join_condition_cf.quals.end());
    join_condition_quals.insert(
        join_condition_quals.end(), join_condition_cf.simple_quals.begin(), join_condition_cf.simple_quals.end());
  }
  join_condition_quals = combine_equi_join_conditions(join_condition_quals);
  RangeTableIndexVisitor rte_idx_visitor;
  JoinQualsPerNestingLevel result(input_descs.size() - 1);
  std::unordered_set<std::shared_ptr<Analyzer::Expr>> visited_quals;
  for (size_t rte_idx = 1; rte_idx < input_descs.size(); ++rte_idx) {
    for (const auto qual : join_condition_quals) {
      if (visited_quals.count(qual)) {
        continue;
      }
      const auto qual_rte_idx = rte_idx_visitor.visit(qual.get());
      if (static_cast<size_t>(qual_rte_idx) <= rte_idx) {
        const auto it_ok = visited_quals.emplace(qual);
        CHECK(it_ok.second);
        result[rte_idx - 1].push_back(qual);
      }
    }
  }
  return result;
}

namespace {

std::vector<TargetMetaInfo> get_inputs_meta(const RelScan* scan, const Catalog_Namespace::Catalog& cat) {
  std::vector<TargetMetaInfo> in_metainfo;
  for (const auto& col_name : scan->getFieldNames()) {
    const auto table_desc = scan->getTableDescriptor();
    const auto cd = cat.getMetadataForColumn(table_desc->tableId, col_name);
    CHECK(cd);
    auto col_ti = cd->columnType;
    in_metainfo.emplace_back(col_name, col_ti);
  }
  return in_metainfo;
}

std::vector<std::shared_ptr<Analyzer::Expr>> get_input_exprs(const RelJoin* join, const bool need_original) {
  const auto join_type = join->getJoinType();
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;
  const auto lhs = join->getInput(0);
  if (need_original && dynamic_cast<const RelJoin*>(lhs)) {
    const auto previous_join = static_cast<const RelJoin*>(lhs);
    auto source_exprs_owned = get_input_exprs(previous_join, true);
    for (size_t i = 0; i < source_exprs_owned.size(); ++i) {
      const auto iter_ti = source_exprs_owned[i]->get_type_info();
      auto iter_expr = std::make_shared<Analyzer::IterExpr>(iter_ti, table_id_from_ra(lhs), 0);
      target_exprs_owned.push_back(iter_expr);
    }
  } else {
    const auto iter_ti = SQLTypeInfo(kBIGINT, true);
    auto iter_expr = std::make_shared<Analyzer::IterExpr>(iter_ti, table_id_from_ra(lhs), 0);
    target_exprs_owned.push_back(iter_expr);
  }

  const auto rhs = join->getInput(1);
  CHECK(!dynamic_cast<const RelJoin*>(rhs));
  const auto iter_ti = SQLTypeInfo(kBIGINT, join_type == JoinType::INNER);
  auto iter_expr = std::make_shared<Analyzer::IterExpr>(iter_ti, table_id_from_ra(rhs), 1);
  target_exprs_owned.push_back(iter_expr);

  return target_exprs_owned;
}

std::pair<std::vector<TargetMetaInfo>, std::vector<std::shared_ptr<Analyzer::Expr>>> get_inputs_meta(
    const RelJoin* join,
    const Catalog_Namespace::Catalog& cat) {
  std::vector<TargetMetaInfo> targets_meta;
  const auto lhs = join->getInput(0);
  if (auto scan = dynamic_cast<const RelScan*>(lhs)) {
    const auto lhs_in_meta = get_inputs_meta(scan, cat);
    targets_meta.insert(targets_meta.end(), lhs_in_meta.begin(), lhs_in_meta.end());
  } else {
    const auto& lhs_in_meta = lhs->getOutputMetainfo();
    targets_meta.insert(targets_meta.end(), lhs_in_meta.begin(), lhs_in_meta.end());
  }
  const auto rhs = join->getInput(1);
  CHECK(!dynamic_cast<const RelJoin*>(rhs));
  if (auto scan = dynamic_cast<const RelScan*>(rhs)) {
    const auto rhs_in_meta = get_inputs_meta(scan, cat);
    targets_meta.insert(targets_meta.end(), rhs_in_meta.begin(), rhs_in_meta.end());
  } else {
    const auto& rhs_in_meta = rhs->getOutputMetainfo();
    targets_meta.insert(targets_meta.end(), rhs_in_meta.begin(), rhs_in_meta.end());
  }
  return std::make_pair(targets_meta, get_input_exprs(join, false));
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createJoinWorkUnit(const RelJoin* join,
                                                            const SortInfo& sort_info,
                                                            const bool just_explain) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  const auto input_to_nest_level = get_input_nest_levels(join, {});
  std::tie(input_descs, input_col_descs, std::ignore) = get_input_desc(join, input_to_nest_level, {});
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = join->getJoinType();
  RelAlgTranslator translator(cat_, executor_, input_to_nest_level, join_type, now_, just_explain);
  auto inner_join_quals = get_inner_join_quals(join, translator);
  auto outer_join_quals = get_outer_join_quals(join, translator);
  CHECK((join_type == JoinType::INNER && outer_join_quals.empty()) ||
        (join_type == JoinType::LEFT && inner_join_quals.empty()));
  std::vector<TargetMetaInfo> targets_meta;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned;
  std::tie(targets_meta, target_exprs_owned) = get_inputs_meta(join, cat_);
  target_exprs_owned_.insert(target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  auto orig_target_exprs_owned = get_input_exprs(join, true);
  target_exprs_owned_.insert(target_exprs_owned_.end(), orig_target_exprs_owned.begin(), orig_target_exprs_owned.end());
  join->setOutputMetainfo(targets_meta);
  return {{input_descs,
           extra_input_descs,
           input_col_descs,
           {},
           {},
           join_type,
           {},
           get_join_dimensions(join, executor_),
           inner_join_quals,
           outer_join_quals,
           {nullptr},
           get_exprs_not_owned(target_exprs_owned),
           get_exprs_not_owned(orig_target_exprs_owned),
           nullptr,
           sort_info,
           0},
          join,
          max_groups_buffer_entry_default_guess,
          nullptr};
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
    inputs.push_back(std::make_shared<Analyzer::ColumnVar>(
        input_meta.get_type_info(), table_id, scan_ra ? input_idx + 1 : input_idx, rte_idx));
    ++input_idx;
  }
  return inputs;
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createAggregateWorkUnit(const RelAggregate* aggregate,
                                                                 const SortInfo& sort_info,
                                                                 const bool just_explain) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  const auto input_to_nest_level = get_input_nest_levels(aggregate, {});
  std::tie(input_descs, input_col_descs, used_inputs_owned) = get_input_desc(aggregate, input_to_nest_level, {});
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = get_join_type(aggregate);
  RelAlgTranslator translator(cat_, executor_, input_to_nest_level, join_type, now_, just_explain);
  CHECK_EQ(size_t(1), aggregate->inputCount());
  const auto source = aggregate->getInput(0);
  const auto& in_metainfo = source->getOutputMetainfo();
  const auto scalar_sources = synthesize_inputs(aggregate, size_t(0), in_metainfo, input_to_nest_level);
  const auto groupby_exprs = translate_groupby_exprs(aggregate, scalar_sources);
  const auto target_exprs =
      translate_targets(target_exprs_owned_, scalar_sources, groupby_exprs, aggregate, translator);
  const auto targets_meta = get_targets_meta(aggregate, target_exprs);
  aggregate->setOutputMetainfo(targets_meta);
  return {{input_descs,
           extra_input_descs,
           input_col_descs,
           {},
           {},
           join_type,
           {},
           get_join_dimensions(get_data_sink(aggregate), executor_),
           get_inner_join_quals(aggregate, translator),
           get_outer_join_quals(aggregate, translator),
           groupby_exprs,
           target_exprs,
           {},
           nullptr,
           sort_info,
           0},
          aggregate,
          max_groups_buffer_entry_default_guess,
          nullptr};
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createProjectWorkUnit(const RelProject* project,
                                                               const SortInfo& sort_info,
                                                               const bool just_explain) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  auto input_to_nest_level = get_input_nest_levels(project, {});
  std::tie(input_descs, input_col_descs, std::ignore) = get_input_desc(project, input_to_nest_level, {});
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(project->getInput(0));
  JoinQualsPerNestingLevel left_deep_inner_joins;
  if (left_deep_join) {
    const auto query_infos = get_table_infos(input_descs, executor_);
    const auto input_permutation = get_node_input_permutation(query_infos);
    input_to_nest_level = get_input_nest_levels(project, input_permutation);
    std::tie(input_descs, input_col_descs, std::ignore) =
        get_input_desc(project, input_to_nest_level, input_permutation);
    left_deep_inner_joins = translateLeftDeepJoinFilter(left_deep_join, input_descs, input_to_nest_level, just_explain);
  }
  const auto join_type = left_deep_join ? JoinType::INVALID : get_join_type(project);
  RelAlgTranslator translator(cat_, executor_, input_to_nest_level, join_type, now_, just_explain);
  const auto target_exprs_owned = translate_scalar_sources(project, translator);
  target_exprs_owned_.insert(target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  const auto targets_meta = get_targets_meta(project, target_exprs);
  project->setOutputMetainfo(targets_meta);
  return {{input_descs,
           extra_input_descs,
           input_col_descs,
           {},
           {},
           join_type,
           left_deep_inner_joins,
           get_join_dimensions(get_data_sink(project), executor_),
           get_inner_join_quals(project, translator),
           get_outer_join_quals(project, translator),
           {nullptr},
           target_exprs,
           {},
           nullptr,
           sort_info,
           0},
          project,
          max_groups_buffer_entry_default_guess,
          nullptr};
}

namespace {

std::pair<std::vector<TargetMetaInfo>, std::vector<std::shared_ptr<Analyzer::Expr>>> get_inputs_meta(
    const RelFilter* filter,
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
      const auto source_metadata = get_targets_meta(scan_source, get_exprs_not_owned(scalar_sources_owned));
      in_metainfo.insert(in_metainfo.end(), source_metadata.begin(), source_metadata.end());
      exprs_owned.insert(exprs_owned.end(), scalar_sources_owned.begin(), scalar_sources_owned.end());
    } else {
      const auto& source_metadata = source->getOutputMetainfo();
      input_it += source_metadata.size();
      in_metainfo.insert(in_metainfo.end(), source_metadata.begin(), source_metadata.end());
      const auto scalar_sources_owned =
          synthesize_inputs(data_sink_node, nest_level, source_metadata, input_to_nest_level);
      exprs_owned.insert(exprs_owned.end(), scalar_sources_owned.begin(), scalar_sources_owned.end());
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
  std::tie(input_descs, input_col_descs, used_inputs_owned) = get_input_desc(filter, input_to_nest_level, {});
  const auto extra_input_descs = separate_extra_input_descs(input_descs);
  const auto join_type = get_join_type(filter);
  RelAlgTranslator translator(cat_, executor_, input_to_nest_level, join_type, now_, just_explain);
  std::tie(in_metainfo, target_exprs_owned) =
      get_inputs_meta(filter, translator, used_inputs_owned, input_to_nest_level);
  const auto filter_expr = translator.translateScalarRex(filter->getCondition());
  const auto qual = fold_expr(filter_expr.get());
  std::list<std::shared_ptr<Analyzer::Expr>> quals{qual};
  const auto separated_quals = join_type == JoinType::LEFT ? SeparatedQuals{quals, {}} : separate_join_quals(quals);
  target_exprs_owned_.insert(target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_exprs_not_owned(target_exprs_owned);
  filter->setOutputMetainfo(in_metainfo);
  return {{input_descs,
           extra_input_descs,
           input_col_descs,
           {},
           separated_quals.regular_quals,
           join_type,
           {},
           get_join_dimensions(get_data_sink(filter), executor_),
           separated_quals.join_quals,
           get_outer_join_quals(filter, translator),
           {nullptr},
           target_exprs,
           {},
           nullptr,
           sort_info,
           0},
          filter,
          max_groups_buffer_entry_default_guess,
          nullptr};
}

SpeculativeTopNBlacklist RelAlgExecutor::speculative_topn_blacklist_;
