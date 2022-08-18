/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "Fragmenter/InsertDataLoader.h"
#include "Parser/ParserNode.h"
#include "QueryEngine/CalciteDeserializerUtils.h"
#include "QueryEngine/CardinalityEstimator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/ExtensionFunctionsBinding.h"
#include "QueryEngine/ExternalExecutor.h"
#include "QueryEngine/FromTableReordering.h"
#include "QueryEngine/QueryPhysicalInputsCollector.h"
#include "QueryEngine/QueryPlanDagExtractor.h"
#include "QueryEngine/RangeTableIndexVisitor.h"
#include "QueryEngine/RelAlgDag.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "QueryEngine/RelAlgVisitor.h"
#include "QueryEngine/ResultSetBuilder.h"
#include "QueryEngine/RexVisitor.h"
#include "QueryEngine/TableOptimizer.h"
#include "QueryEngine/WindowContext.h"
#include "Shared/TypedDataAccessors.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/shard_key.h"

#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include <algorithm>
#include <functional>
#include <numeric>

bool g_skip_intermediate_count{true};
bool g_enable_interop{false};
bool g_enable_union{true};  // DEPRECATED
size_t g_estimator_failure_max_groupby_size{256000000};
bool g_columnar_large_projections{true};
size_t g_columnar_large_projections_threshold{1000000};

extern bool g_enable_watchdog;
extern size_t g_watchdog_none_encoded_string_translation_limit;
extern bool g_enable_bump_allocator;
extern size_t g_default_max_groups_buffer_entry_guess;
extern bool g_enable_system_tables;

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

void set_parallelism_hints(const RelAlgNode& ra_node,
                           const Catalog_Namespace::Catalog& catalog) {
  std::map<ChunkKey, std::set<foreign_storage::ForeignStorageMgr::ParallelismHint>>
      parallelism_hints_per_table;
  for (const auto& physical_input : get_physical_inputs(&ra_node)) {
    int table_id = physical_input.table_id;
    auto table = catalog.getMetadataForTable(table_id, false);
    if (table && table->storageType == StorageType::FOREIGN_TABLE &&
        !table->is_in_memory_system_table) {
      int col_id = catalog.getColumnIdBySpi(table_id, physical_input.col_id);
      const auto col_desc = catalog.getMetadataForColumn(table_id, col_id);
      auto foreign_table = catalog.getForeignTable(table_id);
      for (const auto& fragment :
           foreign_table->fragmenter->getFragmentsForQuery().fragments) {
        Chunk_NS::Chunk chunk{col_desc};
        ChunkKey chunk_key = {
            catalog.getDatabaseId(), table_id, col_id, fragment.fragmentId};

        // Parallelism hints should not include fragments that are not mapped to the
        // current node, otherwise we will try to prefetch them and run into trouble.
        if (foreign_storage::key_does_not_shard_to_leaf(chunk_key)) {
          continue;
        }

        // do not include chunk hints that are in CPU memory
        if (!chunk.isChunkOnDevice(
                &catalog.getDataMgr(), chunk_key, Data_Namespace::CPU_LEVEL, 0)) {
          parallelism_hints_per_table[{catalog.getDatabaseId(), table_id}].insert(
              foreign_storage::ForeignStorageMgr::ParallelismHint{col_id,
                                                                  fragment.fragmentId});
        }
      }
    }
  }
  if (!parallelism_hints_per_table.empty()) {
    auto foreign_storage_mgr =
        catalog.getDataMgr().getPersistentStorageMgr()->getForeignStorageMgr();
    CHECK(foreign_storage_mgr);
    foreign_storage_mgr->setParallelismHints(parallelism_hints_per_table);
  }
}

void prepare_string_dictionaries(const RelAlgNode& ra_node,
                                 const Catalog_Namespace::Catalog& catalog) {
  for (const auto [col_id, table_id] : get_physical_inputs(&ra_node)) {
    auto table = catalog.getMetadataForTable(table_id, false);
    if (table && table->storageType == StorageType::FOREIGN_TABLE) {
      auto spi_col_id = catalog.getColumnIdBySpi(table_id, col_id);
      foreign_storage::populate_string_dictionary(table_id, spi_col_id, catalog);
    }
  }
}

void prepare_foreign_table_for_execution(const RelAlgNode& ra_node,
                                         const Catalog_Namespace::Catalog& catalog) {
  // Iterate through ra_node inputs for types that need to be loaded pre-execution
  // If they do not have valid metadata, load them into CPU memory to generate
  // the metadata and leave them ready to be used by the query
  set_parallelism_hints(ra_node, catalog);
  prepare_string_dictionaries(ra_node, catalog);
}

void prepare_for_system_table_execution(const RelAlgNode& ra_node,
                                        const Catalog_Namespace::Catalog& catalog,
                                        const CompilationOptions& co) {
  if (g_enable_system_tables) {
    std::map<int32_t, std::vector<int32_t>> system_table_columns_by_table_id;
    for (const auto& physical_input : get_physical_inputs(&ra_node)) {
      int table_id = physical_input.table_id;
      auto table = catalog.getMetadataForTable(table_id, false);
      if (table && table->is_in_memory_system_table) {
        auto column_id = catalog.getColumnIdBySpi(table_id, physical_input.col_id);
        system_table_columns_by_table_id[table_id].emplace_back(column_id);
      }
    }
    // Execute on CPU for queries involving system tables
    if (!system_table_columns_by_table_id.empty() &&
        co.device_type != ExecutorDeviceType::CPU) {
      throw QueryMustRunOnCpu();
    }

    for (const auto& [table_id, column_ids] : system_table_columns_by_table_id) {
      // Clear any previously cached data, since system tables depend on point in
      // time data snapshots.
      catalog.getDataMgr().deleteChunksWithPrefix(
          ChunkKey{catalog.getDatabaseId(), table_id}, Data_Namespace::CPU_LEVEL);

      // TODO(Misiu): This prefetching can be removed if we can add support for
      // ExpressionRanges to reduce invalid with valid ranges (right now prefetching
      // causes us to fetch the chunks twice).  Right now if we do not prefetch (i.e. if
      // we remove the code below) some nodes will return valid ranges and others will
      // return unknown because they only use placeholder metadata and the LeafAggregator
      // has no idea how to reduce the two.
      auto td = catalog.getMetadataForTable(table_id);
      CHECK(td);
      CHECK(td->fragmenter);
      auto fragment_count = td->fragmenter->getFragmentsForQuery().fragments.size();
      CHECK_LE(fragment_count, static_cast<size_t>(1))
          << "In-memory system tables are expected to have a single fragment.";
      if (fragment_count > 0) {
        for (auto column_id : column_ids) {
          // Prefetch system table chunks in order to force chunk statistics metadata
          // computation.
          auto cd = catalog.getMetadataForColumn(table_id, column_id);
          ChunkKey chunk_key{catalog.getDatabaseId(), table_id, column_id, 0};
          Chunk_NS::Chunk::getChunk(
              cd, &(catalog.getDataMgr()), chunk_key, Data_Namespace::CPU_LEVEL, 0, 0, 0);
        }
      }
    }
  }
}

bool has_valid_query_plan_dag(const RelAlgNode* node) {
  return node->getQueryPlanDagHash() != EMPTY_HASHED_PLAN_DAG_KEY;
}

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

bool is_validate_or_explain_query(const ExecutionOptions& eo) {
  return eo.just_validate || eo.just_explain || eo.just_calcite_explain;
}

class RelLeftDeepTreeIdsCollector : public RelAlgVisitor<std::vector<unsigned>> {
 public:
  std::vector<unsigned> visitLeftDeepInnerJoin(
      const RelLeftDeepInnerJoin* left_deep_join_tree) const override {
    return {left_deep_join_tree->getId()};
  }

 protected:
  std::vector<unsigned> aggregateResult(
      const std::vector<unsigned>& aggregate,
      const std::vector<unsigned>& next_result) const override {
    auto result = aggregate;
    std::copy(next_result.begin(), next_result.end(), std::back_inserter(result));
    return result;
  }
};

struct TextEncodingCastCounts {
  size_t text_decoding_casts;
  size_t text_encoding_casts;
  TextEncodingCastCounts() : text_decoding_casts(0UL), text_encoding_casts(0UL) {}
  TextEncodingCastCounts(const size_t text_decoding_casts,
                         const size_t text_encoding_casts)
      : text_decoding_casts(text_decoding_casts)
      , text_encoding_casts(text_encoding_casts) {}
};

class TextEncodingCastCountVisitor : public ScalarExprVisitor<TextEncodingCastCounts> {
 public:
  TextEncodingCastCountVisitor(const bool default_disregard_casts_to_none_encoding)
      : default_disregard_casts_to_none_encoding_(
            default_disregard_casts_to_none_encoding) {}

 protected:
  TextEncodingCastCounts visitUOper(const Analyzer::UOper* u_oper) const override {
    TextEncodingCastCounts result = defaultResult();
    // Save state of input disregard_casts_to_none_encoding_ as child node traversal
    // will reset it
    const bool disregard_casts_to_none_encoding = disregard_casts_to_none_encoding_;
    result = aggregateResult(result, visit(u_oper->get_operand()));
    if (u_oper->get_optype() != kCAST) {
      return result;
    }
    const auto& operand_ti = u_oper->get_operand()->get_type_info();
    const auto& casted_ti = u_oper->get_type_info();
    if (!operand_ti.is_string() && casted_ti.is_dict_encoded_string()) {
      return aggregateResult(result, TextEncodingCastCounts(0UL, 1UL));
    }
    if (!casted_ti.is_string()) {
      return result;
    }
    const bool literals_only = u_oper->get_operand()->get_num_column_vars(true) == 0UL;
    if (literals_only) {
      return result;
    }
    if (operand_ti.is_none_encoded_string() && casted_ti.is_dict_encoded_string()) {
      return aggregateResult(result, TextEncodingCastCounts(0UL, 1UL));
    }
    if (operand_ti.is_dict_encoded_string() && casted_ti.is_none_encoded_string()) {
      if (!disregard_casts_to_none_encoding) {
        return aggregateResult(result, TextEncodingCastCounts(1UL, 0UL));
      } else {
        return result;
      }
    }
    return result;
  }

  TextEncodingCastCounts visitStringOper(
      const Analyzer::StringOper* string_oper) const override {
    TextEncodingCastCounts result = defaultResult();
    if (string_oper->getArity() > 0) {
      result = aggregateResult(result, visit(string_oper->getArg(0)));
    }
    if (string_op_returns_string(string_oper->get_kind()) &&
        string_oper->hasNoneEncodedTextArg()) {
      result = aggregateResult(result, TextEncodingCastCounts(0UL, 1UL));
    }
    return result;
  }

  TextEncodingCastCounts visitBinOper(const Analyzer::BinOper* bin_oper) const override {
    TextEncodingCastCounts result = defaultResult();
    // Currently the join framework handles casts between string types, and
    // casts to none-encoded strings should be considered spurious, except
    // when the join predicate is not a =/<> operator, in which case
    // for both join predicates and all other instances we have to decode
    // to a none-encoded string to do the comparison. The logic below essentially
    // overrides the logic such as to always count none-encoded casts on strings
    // that are children of binary operators other than =/<>
    if (bin_oper->get_optype() != kEQ && bin_oper->get_optype() != kNE) {
      // Override the global override so that join opers don't skip
      // the check when there is an actual cast to none-encoded string
      const auto prev_disregard_casts_to_none_encoding_state =
          disregard_casts_to_none_encoding_;
      const auto left_u_oper =
          dynamic_cast<const Analyzer::UOper*>(bin_oper->get_left_operand());
      if (left_u_oper && left_u_oper->get_optype() == kCAST) {
        disregard_casts_to_none_encoding_ = false;
        result = aggregateResult(result, visitUOper(left_u_oper));
      } else {
        disregard_casts_to_none_encoding_ = prev_disregard_casts_to_none_encoding_state;
        result = aggregateResult(result, visit(bin_oper->get_left_operand()));
      }

      const auto right_u_oper =
          dynamic_cast<const Analyzer::UOper*>(bin_oper->get_left_operand());
      if (right_u_oper && right_u_oper->get_optype() == kCAST) {
        disregard_casts_to_none_encoding_ = false;
        result = aggregateResult(result, visitUOper(right_u_oper));
      } else {
        disregard_casts_to_none_encoding_ = prev_disregard_casts_to_none_encoding_state;
        result = aggregateResult(result, visit(bin_oper->get_right_operand()));
      }
      disregard_casts_to_none_encoding_ = prev_disregard_casts_to_none_encoding_state;
    } else {
      result = aggregateResult(result, visit(bin_oper->get_left_operand()));
      result = aggregateResult(result, visit(bin_oper->get_right_operand()));
    }
    return result;
  }

  TextEncodingCastCounts visitLikeExpr(const Analyzer::LikeExpr* like) const override {
    TextEncodingCastCounts result = defaultResult();
    const auto u_oper = dynamic_cast<const Analyzer::UOper*>(like->get_arg());
    const auto prev_disregard_casts_to_none_encoding_state =
        disregard_casts_to_none_encoding_;
    if (u_oper && u_oper->get_optype() == kCAST) {
      disregard_casts_to_none_encoding_ = true;
      result = aggregateResult(result, visitUOper(u_oper));
      disregard_casts_to_none_encoding_ = prev_disregard_casts_to_none_encoding_state;
    } else {
      result = aggregateResult(result, visit(like->get_arg()));
    }
    result = aggregateResult(result, visit(like->get_like_expr()));
    if (like->get_escape_expr()) {
      result = aggregateResult(result, visit(like->get_escape_expr()));
    }
    return result;
  }

  TextEncodingCastCounts aggregateResult(
      const TextEncodingCastCounts& aggregate,
      const TextEncodingCastCounts& next_result) const override {
    auto result = aggregate;
    result.text_decoding_casts += next_result.text_decoding_casts;
    result.text_encoding_casts += next_result.text_encoding_casts;
    return result;
  }

  void visitBegin() const override {
    disregard_casts_to_none_encoding_ = default_disregard_casts_to_none_encoding_;
  }

  TextEncodingCastCounts defaultResult() const override {
    return TextEncodingCastCounts();
  }

 private:
  mutable bool disregard_casts_to_none_encoding_ = false;
  const bool default_disregard_casts_to_none_encoding_;
};

TextEncodingCastCounts get_text_cast_counts(const RelAlgExecutionUnit& ra_exe_unit) {
  TextEncodingCastCounts cast_counts;

  auto check_node_for_text_casts = [&cast_counts](
                                       const Analyzer::Expr* expr,
                                       const bool disregard_casts_to_none_encoding) {
    if (!expr) {
      return;
    }
    TextEncodingCastCountVisitor visitor(disregard_casts_to_none_encoding);
    const auto this_node_cast_counts = visitor.visit(expr);
    cast_counts.text_encoding_casts += this_node_cast_counts.text_encoding_casts;
    cast_counts.text_decoding_casts += this_node_cast_counts.text_decoding_casts;
  };

  for (const auto& qual : ra_exe_unit.quals) {
    check_node_for_text_casts(qual.get(), false);
  }
  for (const auto& simple_qual : ra_exe_unit.simple_quals) {
    check_node_for_text_casts(simple_qual.get(), false);
  }
  for (const auto& groupby_expr : ra_exe_unit.groupby_exprs) {
    check_node_for_text_casts(groupby_expr.get(), false);
  }
  for (const auto& target_expr : ra_exe_unit.target_exprs) {
    check_node_for_text_casts(target_expr, false);
  }
  for (const auto& join_condition : ra_exe_unit.join_quals) {
    for (const auto& join_qual : join_condition.quals) {
      // We currently need to not count casts to none-encoded strings for join quals,
      // as analyzer will generate these but our join framework disregards them.
      // Some investigation was done on having analyzer issue the correct inter-string
      // dictionary casts, but this actually causes them to get executed and so the same
      // work gets done twice.
      check_node_for_text_casts(join_qual.get(),
                                true /* disregard_casts_to_none_encoding */);
    }
  }
  return cast_counts;
}

void check_none_encoded_string_cast_tuple_limit(
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit) {
  if (!g_enable_watchdog) {
    return;
  }
  auto const tuples_upper_bound =
      std::accumulate(query_infos.cbegin(),
                      query_infos.cend(),
                      size_t(0),
                      [](auto max, auto const& query_info) {
                        return std::max(max, query_info.info.getNumTuples());
                      });
  if (tuples_upper_bound <= g_watchdog_none_encoded_string_translation_limit) {
    return;
  }

  const auto& text_cast_counts = get_text_cast_counts(ra_exe_unit);
  const bool has_text_casts =
      text_cast_counts.text_decoding_casts + text_cast_counts.text_encoding_casts > 0UL;

  if (!has_text_casts) {
    return;
  }
  std::ostringstream oss;
  oss << "Query requires one or more casts between none-encoded and dictionary-encoded "
      << "strings, and the estimated table size (" << tuples_upper_bound << " rows) "
      << "exceeds the configured watchdog none-encoded string translation limit of "
      << g_watchdog_none_encoded_string_translation_limit << " rows.";
  throw std::runtime_error(oss.str());
}

}  // namespace

bool RelAlgExecutor::canUseResultsetCache(const ExecutionOptions& eo,
                                          RenderInfo* render_info) const {
  auto validate_or_explain_query = is_validate_or_explain_query(eo);
  auto query_for_partial_outer_frag = !eo.outer_fragment_indices.empty();
  return g_enable_data_recycler && g_use_query_resultset_cache && !g_cluster &&
         !validate_or_explain_query && !hasStepForUnion() &&
         !query_for_partial_outer_frag &&
         (!render_info || (render_info && !render_info->isInSitu()));
}

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

  auto lock = executor_->acquireExecuteMutex();
  ScopeGuard row_set_holder = [this] { cleanupPostExecution(); };
  setupCaching(&ra);

  ScopeGuard restore_metainfo_cache = [this] { executor_->clearMetaInfoCache(); };
  auto ed_seq = RaExecutionSequence(&ra, executor_);

  if (!getSubqueries().empty()) {
    return 0;
  }

  CHECK(!ed_seq.empty());
  if (ed_seq.size() > 1) {
    return 0;
  }

  decltype(temporary_tables_)().swap(temporary_tables_);
  decltype(target_exprs_owned_)().swap(target_exprs_owned_);
  executor_->setCatalog(&cat_);
  executor_->temporary_tables_ = &temporary_tables_;

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
  CHECK(query_dag_->getBuildState() == RelAlgDag::BuildState::kBuiltOptimized)
      << static_cast<int>(query_dag_->getBuildState());

  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(executeRelAlgQuery);

  auto run_query = [&](const CompilationOptions& co_in) {
    auto execution_result =
        executeRelAlgQueryNoRetry(co_in, eo, just_explain_plan, render_info);

    constexpr bool vlog_result_set_summary{false};
    if constexpr (vlog_result_set_summary) {
      VLOG(1) << execution_result.getRows()->summaryToString();
    }

    if (post_execution_callback_) {
      VLOG(1) << "Running post execution callback.";
      (*post_execution_callback_)();
    }
    return execution_result;
  };

  try {
    return run_query(co);
  } catch (const QueryMustRunOnCpu&) {
    if (!g_allow_cpu_retry) {
      throw;
    }
  }
  LOG(INFO) << "Query unable to run in GPU mode, retrying on CPU";
  auto co_cpu = CompilationOptions::makeCpuOnly(co);

  if (render_info) {
    render_info->forceNonInSitu();
  }
  return run_query(co_cpu);
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
  std::string query_session{""};
  std::string query_str{"N/A"};
  std::string query_submitted_time{""};
  // gather necessary query's info
  if (query_state_ != nullptr && query_state_->getConstSessionInfo() != nullptr) {
    query_session = query_state_->getConstSessionInfo()->get_session_id();
    query_str = query_state_->getQueryStr();
    query_submitted_time = query_state_->getQuerySubmittedTime();
  }

  auto validate_or_explain_query =
      just_explain_plan || eo.just_validate || eo.just_explain || eo.just_calcite_explain;
  auto interruptable = !render_info && !query_session.empty() &&
                       eo.allow_runtime_query_interrupt && !validate_or_explain_query;
  if (interruptable) {
    // if we reach here, the current query which was waiting an idle executor
    // within the dispatch queue is now scheduled to the specific executor
    // (not UNITARY_EXECUTOR)
    // so we update the query session's status with the executor that takes this query
    std::tie(query_session, query_str) = executor_->attachExecutorToQuerySession(
        query_session, query_str, query_submitted_time);

    // now the query is going to be executed, so update the status as
    // "RUNNING_QUERY_KERNEL"
    executor_->updateQuerySessionStatus(
        query_session,
        query_submitted_time,
        QuerySessionStatus::QueryStatus::RUNNING_QUERY_KERNEL);
  }

  // so it should do cleanup session info after finishing its execution
  ScopeGuard clearQuerySessionInfo =
      [this, &query_session, &interruptable, &query_submitted_time] {
        // reset the runtime query interrupt status after the end of query execution
        if (interruptable) {
          // cleanup running session's info
          executor_->clearQuerySessionStatus(query_session, query_submitted_time);
        }
      };

  auto acquire_execute_mutex = [](Executor * executor) -> auto {
    auto ret = executor->acquireExecuteMutex();
    return ret;
  };
  // now we acquire executor lock in here to make sure that this executor holds
  // all necessary resources and at the same time protect them against other executor
  auto lock = acquire_execute_mutex(executor_);

  if (interruptable) {
    // check whether this query session is "already" interrupted
    // this case occurs when there is very short gap between being interrupted and
    // taking the execute lock
    // if so we have to remove "all" queries initiated by this session and we do in here
    // without running the query
    try {
      executor_->checkPendingQueryStatus(query_session);
    } catch (QueryExecutionError& e) {
      if (e.getErrorCode() == Executor::ERR_INTERRUPTED) {
        throw std::runtime_error("Query execution has been interrupted (pending query)");
      }
      throw e;
    } catch (...) {
      throw std::runtime_error("Checking pending query status failed: unknown error");
    }
  }
  int64_t queue_time_ms = timer_stop(clock_begin);

  prepare_for_system_table_execution(ra, cat_, co);

  // Notify foreign tables to load prior to caching
  prepare_foreign_table_for_execution(ra, cat_);

  ScopeGuard row_set_holder = [this] { cleanupPostExecution(); };
  setupCaching(&ra);

  ScopeGuard restore_metainfo_cache = [this] { executor_->clearMetaInfoCache(); };
  auto ed_seq = RaExecutionSequence(&ra, executor_);

  if (just_explain_plan) {
    std::stringstream ss;
    std::vector<const RelAlgNode*> nodes;
    for (size_t i = 0; i < ed_seq.size(); i++) {
      nodes.emplace_back(ed_seq.getDescriptor(i)->getBody());
    }
    size_t ctr_node_id_in_plan = nodes.size();
    for (auto& body : boost::adaptors::reverse(nodes)) {
      // we set each node's id in the query plan in advance before calling toString
      // method to properly represent the query plan
      auto node_id_in_plan_tree = ctr_node_id_in_plan--;
      body->setIdInPlanTree(node_id_in_plan_tree);
    }
    size_t ctr = nodes.size();
    size_t tab_ctr = 0;
    RelRexToStringConfig config;
    config.skip_input_nodes = true;
    for (auto& body : boost::adaptors::reverse(nodes)) {
      const auto index = ctr--;
      const auto tabs = std::string(tab_ctr++, '\t');
      CHECK(body);
      ss << tabs << std::to_string(index) << " : " << body->toString(config) << "\n";
      if (auto sort = dynamic_cast<const RelSort*>(body)) {
        ss << tabs << "  : " << sort->getInput(0)->toString(config) << "\n";
      }
      if (dynamic_cast<const RelProject*>(body) ||
          dynamic_cast<const RelCompound*>(body)) {
        if (auto join = dynamic_cast<const RelLeftDeepInnerJoin*>(body->getInput(0))) {
          ss << tabs << "  : " << join->toString(config) << "\n";
        }
      }
    }
    const auto& subqueries = getSubqueries();
    if (!subqueries.empty()) {
      ss << "Subqueries: "
         << "\n";
      for (const auto& subquery : subqueries) {
        const auto ra = subquery->getRelAlg();
        ss << "\t" << ra->toString(config) << "\n";
      }
    }
    auto rs = std::make_shared<ResultSet>(ss.str());
    return {rs, {}};
  }

  if (eo.find_push_down_candidates) {
    // this extra logic is mainly due to current limitations on multi-step queries
    // and/or subqueries.
    return executeRelAlgQueryWithFilterPushDown(
        ed_seq, co, eo, render_info, queue_time_ms);
  }
  timer_setup.stop();

  // Dispatch the subqueries first
  const auto global_hints = getGlobalQueryHint();
  for (auto subquery : getSubqueries()) {
    const auto subquery_ra = subquery->getRelAlg();
    CHECK(subquery_ra);
    if (subquery_ra->hasContextData()) {
      continue;
    }
    // Execute the subquery and cache the result.
    RelAlgExecutor subquery_executor(executor_, cat_, query_state_);
    // Propagate global and local query hint if necessary
    const auto local_hints = getParsedQueryHint(subquery_ra);
    if (global_hints || local_hints) {
      const auto subquery_rel_alg_dag = subquery_executor.getRelAlgDag();
      if (global_hints) {
        subquery_rel_alg_dag->setGlobalQueryHints(*global_hints);
      }
      if (local_hints) {
        subquery_rel_alg_dag->registerQueryHint(subquery_ra, *local_hints);
      }
    }
    RaExecutionSequence subquery_seq(subquery_ra, executor_);
    auto result = subquery_executor.executeRelAlgSeq(subquery_seq, co, eo, nullptr, 0);
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
}

std::pair<std::vector<unsigned>, std::unordered_map<unsigned, JoinQualsPerNestingLevel>>
RelAlgExecutor::getJoinInfo(const RelAlgNode* root_node) {
  auto sort_node = dynamic_cast<const RelSort*>(root_node);
  if (sort_node) {
    // we assume that test query that needs join info does not contain any sort node
    return {};
  }
  auto work_unit = createWorkUnit(root_node, {}, ExecutionOptions::defaults());
  RelLeftDeepTreeIdsCollector visitor;
  auto left_deep_tree_ids = visitor.visit(root_node);
  return {left_deep_tree_ids, getLeftDeepJoinTreesInfo()};
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

QueryStepExecutionResult RelAlgExecutor::executeRelAlgQuerySingleStep(
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
    auto order_entries = get_order_entries(sort);
    const auto source_work_unit = createSortInputWorkUnit(sort, order_entries, eo);
    shard_count = GroupByAndAggregate::shard_count_for_top_groups(
        source_work_unit.exe_unit, *executor_->getCatalog());
    if (!shard_count) {
      // No point in sorting on the leaf, only execute the input to the sort node.
      CHECK_EQ(size_t(1), sort->inputCount());
      const auto source = sort->getInput(0);
      if (sort->collationCount() || node_is_aggregate(source)) {
        auto temp_seq = RaExecutionSequence(std::make_unique<RaExecutionDesc>(source));
        CHECK_EQ(temp_seq.size(), size_t(1));
        ExecutionOptions eo_copy = {
            eo.output_columnar_hint,
            eo.keep_result,
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
            eo.running_query_interrupt_freq,
            eo.pending_query_interrupt_freq,
            eo.executor_type,
        };
        // Use subseq to avoid clearing existing temporary tables
        return {
            executeRelAlgSubSeq(temp_seq, std::make_pair(0, 1), co, eo_copy, nullptr, 0),
            merge_type(source),
            source->getId(),
            false};
      }
    }
  }
  QueryStepExecutionResult result{
      executeRelAlgSubSeq(seq,
                          std::make_pair(step_idx, step_idx + 1),
                          co,
                          eo,
                          render_info,
                          queue_time_ms_),
      merge_type(exe_desc_ptr->getBody()),
      exe_desc_ptr->getBody()->getId(),
      false};
  if (post_execution_callback_) {
    VLOG(1) << "Running post execution callback.";
    (*post_execution_callback_)();
  }
  return result;
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
  executor_->row_set_mem_owner_ =
      std::make_shared<RowSetMemoryOwner>(Executor::getArenaBlockSize(), cpu_threads());
  executor_->row_set_mem_owner_->setDictionaryGenerations(string_dictionary_generations);
  executor_->table_generations_ = table_generations;
  executor_->agg_col_range_cache_ = agg_col_range;
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
  decltype(left_deep_join_info_)().swap(left_deep_join_info_);
  executor_->setCatalog(&cat_);
  executor_->temporary_tables_ = &temporary_tables_;

  time(&now_);
  CHECK(!seq.empty());

  auto get_descriptor_count = [&seq, &eo]() -> size_t {
    if (eo.just_explain) {
      if (dynamic_cast<const RelLogicalValues*>(seq.getDescriptor(0)->getBody())) {
        // run the logical values descriptor to generate the result set, then the next
        // descriptor to generate the explain
        CHECK_GE(seq.size(), size_t(2));
        return 2;
      } else {
        return 1;
      }
    } else {
      return seq.size();
    }
  };

  const auto exec_desc_count = get_descriptor_count();
  auto eo_copied = eo;
  if (seq.hasQueryStepForUnion()) {
    // we currently do not support resultset recycling when an input query
    // contains union (all) operation
    eo_copied.keep_result = false;
  }

  // we have to register resultset(s) of the skipped query step(s) as temporary table
  // before executing the remaining query steps
  // since they may be required during the query processing
  // i.e., get metadata of the target expression from the skipped query step
  if (g_allow_query_step_skipping) {
    for (const auto& kv : seq.getSkippedQueryStepCacheKeys()) {
      const auto cached_res =
          executor_->getRecultSetRecyclerHolder().getCachedQueryResultSet(kv.second);
      CHECK(cached_res);
      addTemporaryTable(kv.first, cached_res);
    }
  }

  const auto num_steps = exec_desc_count - 1;
  for (size_t i = 0; i < exec_desc_count; i++) {
    VLOG(1) << "Executing query step " << i << " / " << num_steps;
    try {
      executeRelAlgStep(
          seq, i, co, eo_copied, (i == num_steps) ? render_info : nullptr, queue_time_ms);
    } catch (const QueryMustRunOnCpu&) {
      // Do not allow per-step retry if flag is off or in distributed mode
      // TODO(todd): Determine if and when we can relax this restriction
      // for distributed
      CHECK(co.device_type == ExecutorDeviceType::GPU);
      if (!g_allow_query_step_cpu_retry || g_cluster) {
        throw;
      }
      LOG(INFO) << "Retrying current query step " << i << " / " << num_steps << " on CPU";
      const auto co_cpu = CompilationOptions::makeCpuOnly(co);
      if (render_info && i == num_steps) {
        // only render on the last step
        render_info->forceNonInSitu();
      }
      executeRelAlgStep(seq,
                        i,
                        co_cpu,
                        eo_copied,
                        (i == num_steps) ? render_info : nullptr,
                        queue_time_ms);
    } catch (const NativeExecutionError&) {
      if (!g_enable_interop) {
        throw;
      }
      auto eo_extern = eo_copied;
      eo_extern.executor_type = ::ExecutorType::Extern;
      auto exec_desc_ptr = seq.getDescriptor(i);
      const auto body = exec_desc_ptr->getBody();
      const auto compound = dynamic_cast<const RelCompound*>(body);
      if (compound && (compound->getGroupByCount() || compound->isAggregate())) {
        LOG(INFO) << "Also failed to run the query using interoperability";
        throw;
      }
      executeRelAlgStep(
          seq, i, co, eo_extern, (i == num_steps) ? render_info : nullptr, queue_time_ms);
    }
  }

  return seq.getDescriptor(num_steps)->getResult();
}

ExecutionResult RelAlgExecutor::executeRelAlgSubSeq(
    const RaExecutionSequence& seq,
    const std::pair<size_t, size_t> interval,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    RenderInfo* render_info,
    const int64_t queue_time_ms) {
  INJECT_TIMER(executeRelAlgSubSeq);
  executor_->setCatalog(&cat_);
  executor_->temporary_tables_ = &temporary_tables_;
  decltype(left_deep_join_info_)().swap(left_deep_join_info_);
  time(&now_);
  for (size_t i = interval.first; i < interval.second; i++) {
    // only render on the last step
    try {
      executeRelAlgStep(seq,
                        i,
                        co,
                        eo,
                        (i == interval.second - 1) ? render_info : nullptr,
                        queue_time_ms);
    } catch (const QueryMustRunOnCpu&) {
      // Do not allow per-step retry if flag is off or in distributed mode
      // TODO(todd): Determine if and when we can relax this restriction
      // for distributed
      CHECK(co.device_type == ExecutorDeviceType::GPU);
      if (!g_allow_query_step_cpu_retry || g_cluster) {
        throw;
      }
      LOG(INFO) << "Retrying current query step " << i << " on CPU";
      const auto co_cpu = CompilationOptions::makeCpuOnly(co);
      if (render_info && i == interval.second - 1) {
        render_info->forceNonInSitu();
      }
      executeRelAlgStep(seq,
                        i,
                        co_cpu,
                        eo,
                        (i == interval.second - 1) ? render_info : nullptr,
                        queue_time_ms);
    }
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
      eo.keep_result,
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
      eo.running_query_interrupt_freq,
      eo.pending_query_interrupt_freq,
      eo.executor_type,
      step_idx == 0 ? eo.outer_fragment_indices : std::vector<size_t>()};

  auto handle_hint = [co,
                      eo_work_unit,
                      body,
                      this]() -> std::pair<CompilationOptions, ExecutionOptions> {
    ExecutionOptions eo_hint_applied = eo_work_unit;
    CompilationOptions co_hint_applied = co;
    auto target_node = body;
    if (auto sort_body = dynamic_cast<const RelSort*>(body)) {
      target_node = sort_body->getInput(0);
    }
    auto query_hints = getParsedQueryHint(target_node);
    auto columnar_output_hint_enabled = false;
    auto rowwise_output_hint_enabled = false;
    if (query_hints) {
      if (query_hints->isHintRegistered(QueryHint::kCpuMode)) {
        VLOG(1) << "A user forces to run the query on the CPU execution mode";
        co_hint_applied.device_type = ExecutorDeviceType::CPU;
      }
      if (query_hints->isHintRegistered(QueryHint::kKeepResult)) {
        if (!g_enable_data_recycler) {
          VLOG(1) << "A user enables keeping query resultset but is skipped since data "
                     "recycler is disabled";
        }
        if (!g_use_query_resultset_cache) {
          VLOG(1) << "A user enables keeping query resultset but is skipped since query "
                     "resultset recycler is disabled";
        } else {
          VLOG(1) << "A user enables keeping query resultset";
          eo_hint_applied.keep_result = true;
        }
      }
      if (query_hints->isHintRegistered(QueryHint::kKeepTableFuncResult)) {
        // we use this hint within the function 'executeTableFunction`
        if (!g_enable_data_recycler) {
          VLOG(1) << "A user enables keeping table function's resultset but is skipped "
                     "since data recycler is disabled";
        }
        if (!g_use_query_resultset_cache) {
          VLOG(1) << "A user enables keeping table function's resultset but is skipped "
                     "since query resultset recycler is disabled";
        } else {
          VLOG(1) << "A user enables keeping table function's resultset";
          eo_hint_applied.keep_result = true;
        }
      }
      if (query_hints->isHintRegistered(QueryHint::kWatchdog)) {
        if (!eo_hint_applied.with_watchdog) {
          VLOG(1) << "A user enables watchdog for this query";
          eo_hint_applied.with_watchdog = true;
        }
      }
      if (query_hints->isHintRegistered(QueryHint::kWatchdogOff)) {
        if (eo_hint_applied.with_watchdog) {
          VLOG(1) << "A user disables watchdog for this query";
          eo_hint_applied.with_watchdog = false;
        }
      }
      if (query_hints->isHintRegistered(QueryHint::kDynamicWatchdog)) {
        if (!eo_hint_applied.with_dynamic_watchdog) {
          VLOG(1) << "A user enables dynamic watchdog for this query";
          eo_hint_applied.with_watchdog = true;
        }
      }
      if (query_hints->isHintRegistered(QueryHint::kDynamicWatchdogOff)) {
        if (eo_hint_applied.with_dynamic_watchdog) {
          VLOG(1) << "A user disables dynamic watchdog for this query";
          eo_hint_applied.with_watchdog = false;
        }
      }
      if (query_hints->isHintRegistered(QueryHint::kQueryTimeLimit)) {
        std::ostringstream oss;
        oss << "A user sets query time limit to " << query_hints->query_time_limit
            << " ms";
        eo_hint_applied.dynamic_watchdog_time_limit = query_hints->query_time_limit;
        if (!eo_hint_applied.with_dynamic_watchdog) {
          eo_hint_applied.with_dynamic_watchdog = true;
          oss << " (and system automatically enables dynamic watchdog to activate the "
                 "given \"query_time_limit\" hint)";
        }
        VLOG(1) << oss.str();
      }
      if (query_hints->isHintRegistered(QueryHint::kColumnarOutput)) {
        VLOG(1) << "A user forces the query to run with columnar output";
        columnar_output_hint_enabled = true;
      } else if (query_hints->isHintRegistered(QueryHint::kRowwiseOutput)) {
        VLOG(1) << "A user forces the query to run with rowwise output";
        rowwise_output_hint_enabled = true;
      }
    }
    auto columnar_output_enabled = eo_work_unit.output_columnar_hint
                                       ? !rowwise_output_hint_enabled
                                       : columnar_output_hint_enabled;
    if (g_cluster && (columnar_output_hint_enabled || rowwise_output_hint_enabled)) {
      LOG(INFO) << "Currently, we do not support applying query hint to change query "
                   "output layout in distributed mode.";
    }
    eo_hint_applied.output_columnar_hint = columnar_output_enabled;
    return std::make_pair(co_hint_applied, eo_hint_applied);
  };

  auto hint_applied = handle_hint();
  setHasStepForUnion(seq.hasQueryStepForUnion());

  if (canUseResultsetCache(eo, render_info) && has_valid_query_plan_dag(body)) {
    if (auto cached_resultset =
            executor_->getRecultSetRecyclerHolder().getCachedQueryResultSet(
                body->getQueryPlanDagHash())) {
      VLOG(1) << "recycle resultset of the root node " << body->getRelNodeDagId()
              << " from resultset cache";
      body->setOutputMetainfo(cached_resultset->getTargetMetaInfo());
      if (render_info) {
        std::vector<std::shared_ptr<Analyzer::Expr>>& cached_target_exprs =
            executor_->getRecultSetRecyclerHolder().getTargetExprs(
                body->getQueryPlanDagHash());
        std::vector<Analyzer::Expr*> copied_target_exprs;
        for (const auto& expr : cached_target_exprs) {
          copied_target_exprs.push_back(expr.get());
        }
        build_render_targets(
            *render_info, copied_target_exprs, cached_resultset->getTargetMetaInfo());
      }
      exec_desc.setResult({cached_resultset, cached_resultset->getTargetMetaInfo()});
      addTemporaryTable(-body->getId(), exec_desc.getResult().getDataPtr());
      return;
    }
  }

  const auto compound = dynamic_cast<const RelCompound*>(body);
  if (compound) {
    if (compound->isDeleteViaSelect()) {
      executeDelete(compound, hint_applied.first, hint_applied.second, queue_time_ms);
    } else if (compound->isUpdateViaSelect()) {
      executeUpdate(compound, hint_applied.first, hint_applied.second, queue_time_ms);
    } else {
      exec_desc.setResult(executeCompound(
          compound, hint_applied.first, hint_applied.second, render_info, queue_time_ms));
      VLOG(3) << "Returned from executeCompound(), addTemporaryTable("
              << static_cast<int>(-compound->getId()) << ", ...)"
              << " exec_desc.getResult().getDataPtr()->rowCount()="
              << exec_desc.getResult().getDataPtr()->rowCount();
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
      executeDelete(project, hint_applied.first, hint_applied.second, queue_time_ms);
    } else if (project->isUpdateViaSelect()) {
      executeUpdate(project, hint_applied.first, hint_applied.second, queue_time_ms);
    } else {
      std::optional<size_t> prev_count;
      // Disabling the intermediate count optimization in distributed, as the previous
      // execution descriptor will likely not hold the aggregated result.
      if (g_skip_intermediate_count && step_idx > 0 && !g_cluster) {
        // If the previous node produced a reliable count, skip the pre-flight count.
        RelAlgNode const* const prev_body = project->getInput(0);
        if (shared::dynamic_castable_to_any<RelCompound, RelLogicalValues>(prev_body)) {
          if (RaExecutionDesc const* const prev_exec_desc =
                  prev_body->hasContextData()
                      ? prev_body->getContextData()
                      : seq.getDescriptorByBodyId(prev_body->getId(), step_idx - 1)) {
            const auto& prev_exe_result = prev_exec_desc->getResult();
            const auto prev_result = prev_exe_result.getRows();
            if (prev_result) {
              prev_count = prev_result->rowCount();
              VLOG(3) << "Setting output row count for projection node to previous node ("
                      << prev_exec_desc->getBody()->toString(
                             RelRexToStringConfig::defaults())
                      << ") to " << *prev_count;
            }
          }
        }
      }
      exec_desc.setResult(executeProject(project,
                                         hint_applied.first,
                                         hint_applied.second,
                                         render_info,
                                         queue_time_ms,
                                         prev_count));
      VLOG(3) << "Returned from executeProject(), addTemporaryTable("
              << static_cast<int>(-project->getId()) << ", ...)"
              << " exec_desc.getResult().getDataPtr()->rowCount()="
              << exec_desc.getResult().getDataPtr()->rowCount();
      if (exec_desc.getResult().isFilterPushDownEnabled()) {
        return;
      }
      addTemporaryTable(-project->getId(), exec_desc.getResult().getDataPtr());
    }
    return;
  }
  const auto aggregate = dynamic_cast<const RelAggregate*>(body);
  if (aggregate) {
    exec_desc.setResult(executeAggregate(
        aggregate, hint_applied.first, hint_applied.second, render_info, queue_time_ms));
    addTemporaryTable(-aggregate->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto filter = dynamic_cast<const RelFilter*>(body);
  if (filter) {
    exec_desc.setResult(executeFilter(
        filter, hint_applied.first, hint_applied.second, render_info, queue_time_ms));
    addTemporaryTable(-filter->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto sort = dynamic_cast<const RelSort*>(body);
  if (sort) {
    exec_desc.setResult(executeSort(
        sort, hint_applied.first, hint_applied.second, render_info, queue_time_ms));
    if (exec_desc.getResult().isFilterPushDownEnabled()) {
      return;
    }
    addTemporaryTable(-sort->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto logical_values = dynamic_cast<const RelLogicalValues*>(body);
  if (logical_values) {
    exec_desc.setResult(executeLogicalValues(logical_values, hint_applied.second));
    addTemporaryTable(-logical_values->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto modify = dynamic_cast<const RelModify*>(body);
  if (modify) {
    exec_desc.setResult(executeModify(modify, hint_applied.second));
    return;
  }
  const auto logical_union = dynamic_cast<const RelLogicalUnion*>(body);
  if (logical_union) {
    exec_desc.setResult(executeUnion(logical_union,
                                     seq,
                                     hint_applied.first,
                                     hint_applied.second,
                                     render_info,
                                     queue_time_ms));
    addTemporaryTable(-logical_union->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  const auto table_func = dynamic_cast<const RelTableFunction*>(body);
  if (table_func) {
    exec_desc.setResult(executeTableFunction(
        table_func, hint_applied.first, hint_applied.second, queue_time_ms));
    addTemporaryTable(-table_func->getId(), exec_desc.getResult().getDataPtr());
    return;
  }
  LOG(FATAL) << "Unhandled body type: "
             << body->toString(RelRexToStringConfig::defaults());
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
  if (auto table_func = dynamic_cast<const RelTableFunction*>(ra_node)) {
    return table_func;
  }
  if (auto join = dynamic_cast<const RelJoin*>(ra_node)) {
    CHECK_EQ(size_t(2), join->inputCount());
    return join;
  }
  if (!dynamic_cast<const RelLogicalUnion*>(ra_node)) {
    CHECK_EQ(size_t(1), ra_node->inputCount());
  }
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

std::pair<std::unordered_set<const RexInput*>, std::vector<std::shared_ptr<RexInput>>>
get_used_inputs(const RelLogicalUnion* logical_union, const Catalog_Namespace::Catalog&) {
  std::unordered_set<const RexInput*> used_inputs(logical_union->inputCount());
  std::vector<std::shared_ptr<RexInput>> used_inputs_owned;
  used_inputs_owned.reserve(logical_union->inputCount());
  VLOG(3) << "logical_union->inputCount()=" << logical_union->inputCount();
  auto const n_inputs = logical_union->inputCount();
  for (size_t nest_level = 0; nest_level < n_inputs; ++nest_level) {
    auto input = logical_union->getInput(nest_level);
    for (size_t i = 0; i < input->size(); ++i) {
      used_inputs_owned.emplace_back(std::make_shared<RexInput>(input, i));
      used_inputs.insert(used_inputs_owned.back().get());
    }
  }
  return std::make_pair(std::move(used_inputs), std::move(used_inputs_owned));
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
    // Having a non-zero mapped value (input_idx) results in the query being interpretted
    // as a JOIN within CodeGenerator::codegenColVar() due to rte_idx being set to the
    // mapped value (input_idx) which originates here. This would be incorrect for UNION.
    size_t const idx = dynamic_cast<const RelLogicalUnion*>(ra_node) ? 0 : input_idx;
    const auto it_ok = input_to_nest_level.emplace(input_ra, idx);
    CHECK(it_ok.second);
    LOG_IF(INFO, !input_permutation.empty())
        << "Assigned input " << input_ra->toString(RelRexToStringConfig::defaults())
        << " to nest level " << input_idx;
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

  if (dynamic_cast<const RelLogicalUnion*>(ra_node)) {
    CHECK_GT(ra_node->inputCount(), 1u)
        << ra_node->toString(RelRexToStringConfig::defaults());
  } else if (dynamic_cast<const RelTableFunction*>(ra_node)) {
    // no-op
    CHECK_GE(ra_node->inputCount(), 0u)
        << ra_node->toString(RelRexToStringConfig::defaults());
  } else {
    CHECK_EQ(ra_node->inputCount(), 1u)
        << ra_node->toString(RelRexToStringConfig::defaults());
  }
  return std::make_pair(std::unordered_set<const RexInput*>{},
                        std::vector<std::shared_ptr<RexInput>>{});
}

void collect_used_input_desc(
    std::vector<InputDescriptor>& input_descs,
    const Catalog_Namespace::Catalog& cat,
    std::unordered_set<std::shared_ptr<const InputColDescriptor>>& input_col_descs_unique,
    const RelAlgNode* ra_node,
    const std::unordered_set<const RexInput*>& source_used_inputs,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level) {
  VLOG(3) << "ra_node=" << ra_node->toString(RelRexToStringConfig::defaults())
          << " input_col_descs_unique.size()=" << input_col_descs_unique.size()
          << " source_used_inputs.size()=" << source_used_inputs.size();
  for (const auto used_input : source_used_inputs) {
    const auto input_ra = used_input->getSourceNode();
    const int table_id = table_id_from_ra(input_ra);
    const auto col_id = used_input->getIndex();
    auto it = input_to_nest_level.find(input_ra);
    if (it != input_to_nest_level.end()) {
      const int input_desc = it->second;
      input_col_descs_unique.insert(std::make_shared<const InputColDescriptor>(
          dynamic_cast<const RelScan*>(input_ra)
              ? cat.getColumnIdBySpi(table_id, col_id + 1)
              : col_id,
          table_id,
          input_desc));
    } else if (!dynamic_cast<const RelLogicalUnion*>(ra_node)) {
      throw std::runtime_error("Bushy joins not supported");
    }
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
    auto input_ra = data_sink_node->getInput(input_node_idx);
    const int table_id = table_id_from_ra(input_ra);
    input_descs.emplace_back(table_id, input_idx);
  }
  std::unordered_set<std::shared_ptr<const InputColDescriptor>> input_col_descs_unique;
  collect_used_input_desc(input_descs,
                          cat,
                          input_col_descs_unique,  // modified
                          ra_node,
                          used_inputs,
                          input_to_nest_level);
  std::unordered_set<const RexInput*> join_source_used_inputs;
  std::vector<std::shared_ptr<RexInput>> join_source_used_inputs_owned;
  std::tie(join_source_used_inputs, join_source_used_inputs_owned) =
      get_join_source_used_inputs(ra_node, cat);
  collect_used_input_desc(input_descs,
                          cat,
                          input_col_descs_unique,  // modified
                          ra_node,
                          join_source_used_inputs,
                          input_to_nest_level);
  std::vector<std::shared_ptr<const InputColDescriptor>> input_col_descs(
      input_col_descs_unique.begin(), input_col_descs_unique.end());

  std::sort(input_col_descs.begin(),
            input_col_descs.end(),
            [](std::shared_ptr<const InputColDescriptor> const& lhs,
               std::shared_ptr<const InputColDescriptor> const& rhs) {
              return std::make_tuple(lhs->getScanDesc().getNestLevel(),
                                     lhs->getColId(),
                                     lhs->getScanDesc().getTableId()) <
                     std::make_tuple(rhs->getScanDesc().getNestLevel(),
                                     rhs->getColId(),
                                     rhs->getScanDesc().getTableId());
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
  VLOG(3) << "used_inputs.size() = " << used_inputs.size();
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
  const size_t scalar_sources_size = get_scalar_sources_size(ra_node);
  VLOG(3) << "get_scalar_sources_size("
          << ra_node->toString(RelRexToStringConfig::defaults())
          << ") = " << scalar_sources_size;
  for (size_t i = 0; i < scalar_sources_size; ++i) {
    const auto scalar_rex = scalar_at(i, ra_node);
    if (dynamic_cast<const RexRef*>(scalar_rex)) {
      // RexRef are synthetic scalars we append at the end of the real ones
      // for the sake of taking memory ownership, no real work needed here.
      continue;
    }

    const auto scalar_expr =
        rewrite_array_elements(translator.translate(scalar_rex).get());
    const auto rewritten_expr = rewrite_expr(scalar_expr.get());
    if (executor_type == ExecutorType::Native) {
      set_transient_dict_maybe(scalar_sources, rewritten_expr);
    } else if (executor_type == ExecutorType::TableFunctions) {
      scalar_sources.push_back(fold_expr(rewritten_expr.get()));
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
      translated_expr = cast_to_column_type(translator.translate(scalar_rex),
                                            tableId,
                                            cat,
                                            colNames[i - starting_projection_column_idx]);
    } else {
      translated_expr = translator.translate(scalar_rex);
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
  const auto filter_expr = filter_rex ? translator.translate(filter_rex) : nullptr;
  return filter_expr ? qual_to_conjunctive_form(fold_expr(filter_expr.get()))
                     : QualsConjunctiveForm{};
}

namespace {
// If an encoded type is used in the context of COUNT(DISTINCT ...) then don't
// bother decoding it. This is done by changing the sql type to an integer.
void conditionally_change_arg_to_int_type(
    size_t target_expr_idx,
    std::shared_ptr<Analyzer::Expr>& target_expr,
    std::unordered_map<size_t, SQLTypeInfo>& target_exprs_type_infos) {
  auto* agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr.get());
  CHECK(agg_expr);
  if (agg_expr->get_is_distinct()) {
    SQLTypeInfo const& ti = agg_expr->get_arg()->get_type_info();
    if (ti.get_type() != kARRAY && ti.get_compression() == kENCODING_DATE_IN_DAYS) {
      target_exprs_type_infos.emplace(target_expr_idx, ti);
      target_expr = target_expr->deep_copy();
      auto* arg = dynamic_cast<Analyzer::AggExpr*>(target_expr.get())->get_arg();
      arg->set_type_info({get_int_type_by_size(ti.get_size()), ti.get_notnull()});
      return;
    }
  }
  target_exprs_type_infos.emplace(target_expr_idx, target_expr->get_type_info());
}
}  // namespace

std::vector<Analyzer::Expr*> translate_targets(
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
    std::unordered_map<size_t, SQLTypeInfo>& target_exprs_type_infos,
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
      conditionally_change_arg_to_int_type(i, target_expr, target_exprs_type_infos);
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
        target_expr = translator.translate(target_rex_scalar);
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
      target_exprs_type_infos.emplace(i, target_expr->get_type_info());
    }
    CHECK(target_expr);
    target_exprs_owned.push_back(target_expr);
    target_exprs.push_back(target_expr.get());
  }
  return target_exprs;
}

std::vector<Analyzer::Expr*> translate_targets(
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs_owned,
    std::unordered_map<size_t, SQLTypeInfo>& target_exprs_type_infos,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources,
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
    const RelAggregate* aggregate,
    const RelAlgTranslator& translator) {
  std::vector<Analyzer::Expr*> target_exprs;
  size_t group_key_idx = 1;
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
  CHECK_EQ(ra_node->size(), target_exprs.size());
  for (size_t i = 0; i < ra_node->size(); ++i) {
    CHECK(target_exprs[i]);
    // TODO(alex): remove the count distinct type fixup.
    targets_meta.emplace_back(ra_node->getFieldName(i),
                              get_logical_type_for_expr(*target_exprs[i]),
                              target_exprs[i]->get_type_info());
  }
  return targets_meta;
}

template <>
std::vector<TargetMetaInfo> get_targets_meta(
    const RelFilter* filter,
    const std::vector<Analyzer::Expr*>& target_exprs) {
  RelAlgNode const* input0 = filter->getInput(0);
  if (auto const* input = dynamic_cast<RelCompound const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  } else if (auto const* input = dynamic_cast<RelProject const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  } else if (auto const* input = dynamic_cast<RelLogicalUnion const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  } else if (auto const* input = dynamic_cast<RelAggregate const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  } else if (auto const* input = dynamic_cast<RelScan const*>(input0)) {
    return get_targets_meta(input, target_exprs);
  }
  UNREACHABLE() << "Unhandled node type: "
                << input0->toString(RelRexToStringConfig::defaults());
  return {};
}

}  // namespace

void RelAlgExecutor::executeUpdate(const RelAlgNode* node,
                                   const CompilationOptions& co_in,
                                   const ExecutionOptions& eo_in,
                                   const int64_t queue_time_ms) {
  CHECK(node);
  auto timer = DEBUG_TIMER(__func__);

  auto co = co_in;
  co.hoist_literals = false;  // disable literal hoisting as it interferes with dict
                              // encoded string updates

  auto execute_update_for_node = [this, &co, &eo_in](const auto node,
                                                     auto& work_unit,
                                                     const bool is_aggregate) {
    auto table_descriptor = node->getModifiedTableDescriptor();
    CHECK(table_descriptor);
    if (node->isVarlenUpdateRequired() && !table_descriptor->hasDeletedCol) {
      throw std::runtime_error(
          "UPDATE queries involving variable length columns are only supported on tables "
          "with the vacuum attribute set to 'delayed'");
    }

    Executor::clearExternalCaches(true, table_descriptor, cat_.getDatabaseId());

    dml_transaction_parameters_ =
        std::make_unique<UpdateTransactionParameters>(table_descriptor,
                                                      node->getTargetColumns(),
                                                      node->getOutputMetainfo(),
                                                      node->isVarlenUpdateRequired());

    const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);

    auto execute_update_ra_exe_unit =
        [this, &co, &eo_in, &table_infos, &table_descriptor, &node](
            const RelAlgExecutionUnit& ra_exe_unit, const bool is_aggregate) {
          CompilationOptions co_project = CompilationOptions::makeCpuOnly(co);

          auto eo = eo_in;
          if (dml_transaction_parameters_->tableIsTemporary()) {
            eo.output_columnar_hint = true;
            co_project.allow_lazy_fetch = false;
            co_project.filter_on_deleted_column =
                false;  // project the entire delete column for columnar update
          }

          auto update_transaction_parameters = dynamic_cast<UpdateTransactionParameters*>(
              dml_transaction_parameters_.get());
          update_transaction_parameters->setInputSourceNode(node);
          CHECK(update_transaction_parameters);
          auto update_callback = yieldUpdateCallback(*update_transaction_parameters);
          try {
            auto table_update_metadata =
                executor_->executeUpdate(ra_exe_unit,
                                         table_infos,
                                         table_descriptor,
                                         co_project,
                                         eo,
                                         cat_,
                                         executor_->row_set_mem_owner_,
                                         update_callback,
                                         is_aggregate);
            post_execution_callback_ = [table_update_metadata, this]() {
              dml_transaction_parameters_->finalizeTransaction(cat_);
              TableOptimizer table_optimizer{
                  dml_transaction_parameters_->getTableDescriptor(), executor_, cat_};
              table_optimizer.vacuumFragmentsAboveMinSelectivity(table_update_metadata);
            };
          } catch (const QueryExecutionError& e) {
            throw std::runtime_error(getErrorMessageFromCode(e.getErrorCode()));
          }
        };

    if (dml_transaction_parameters_->tableIsTemporary()) {
      // hold owned target exprs during execution if rewriting
      auto query_rewrite = std::make_unique<QueryRewriter>(table_infos, executor_);
      // rewrite temp table updates to generate the full column by moving the where
      // clause into a case if such a rewrite is not possible, bail on the update
      // operation build an expr for the update target
      auto update_transaction_params =
          dynamic_cast<UpdateTransactionParameters*>(dml_transaction_parameters_.get());
      CHECK(update_transaction_params);
      const auto td = update_transaction_params->getTableDescriptor();
      CHECK(td);
      const auto update_column_names = update_transaction_params->getUpdateColumnNames();
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
    if (project->hasWindowFunctionExpr() || project->hasPushedDownWindowExpr()) {
      // the first condition means this project node has at least one window function
      // and the second condition indicates that this project node falls into
      // one of the following cases:
      // 1) window function expression on a multi-fragmented table
      // 2) window function expression is too complex to evaluate without codegen:
      // i.e., sum(x+y+z) instead of sum(x) -> we currently do not support codegen to
      // evaluate such a complex window function expression
      // 3) nested window function expression
      // but currently we do not support update on a multi-fragmented table having
      // window function, so the second condition only refers to non-fragmented table with
      // cases 2) or 3)
      // if at least one of two conditions satisfy, we must compute corresponding window
      // context before entering `execute_update_for_node` to properly update the table
      if (!leaf_results_.empty()) {
        throw std::runtime_error(
            "Update query having window function is not yet supported in distributed "
            "mode.");
      }
      ColumnCacheMap column_cache;
      co.device_type = ExecutorDeviceType::CPU;
      computeWindow(work_unit, co, eo_in, column_cache, queue_time_ms);
    }
    execute_update_for_node(project, work_unit, false);
  } else {
    throw std::runtime_error("Unsupported parent node for update: " +
                             node->toString(RelRexToStringConfig::defaults()));
  }
}

void RelAlgExecutor::executeDelete(const RelAlgNode* node,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo_in,
                                   const int64_t queue_time_ms) {
  CHECK(node);
  auto timer = DEBUG_TIMER(__func__);

  auto execute_delete_for_node = [this, &co, &eo_in](const auto node,
                                                     auto& work_unit,
                                                     const bool is_aggregate) {
    auto* table_descriptor = node->getModifiedTableDescriptor();
    CHECK(table_descriptor);
    if (!table_descriptor->hasDeletedCol) {
      throw std::runtime_error(
          "DELETE queries are only supported on tables with the vacuum attribute set to "
          "'delayed'");
    }

    Executor::clearExternalCaches(false, table_descriptor, cat_.getDatabaseId());

    const auto table_infos = get_table_infos(work_unit.exe_unit, executor_);

    auto execute_delete_ra_exe_unit =
        [this, &table_infos, &table_descriptor, &eo_in, &co](const auto& exe_unit,
                                                             const bool is_aggregate) {
          dml_transaction_parameters_ =
              std::make_unique<DeleteTransactionParameters>(table_descriptor);
          auto delete_params = dynamic_cast<DeleteTransactionParameters*>(
              dml_transaction_parameters_.get());
          CHECK(delete_params);
          auto delete_callback = yieldDeleteCallback(*delete_params);
          CompilationOptions co_delete = CompilationOptions::makeCpuOnly(co);

          auto eo = eo_in;
          if (dml_transaction_parameters_->tableIsTemporary()) {
            eo.output_columnar_hint = true;
            co_delete.filter_on_deleted_column =
                false;  // project the entire delete column for columnar update
          } else {
            CHECK_EQ(exe_unit.target_exprs.size(), size_t(1));
          }

          try {
            auto table_update_metadata =
                executor_->executeUpdate(exe_unit,
                                         table_infos,
                                         table_descriptor,
                                         co_delete,
                                         eo,
                                         cat_,
                                         executor_->row_set_mem_owner_,
                                         delete_callback,
                                         is_aggregate);
            post_execution_callback_ = [table_update_metadata, this]() {
              dml_transaction_parameters_->finalizeTransaction(cat_);
              TableOptimizer table_optimizer{
                  dml_transaction_parameters_->getTableDescriptor(), executor_, cat_};
              table_optimizer.vacuumFragmentsAboveMinSelectivity(table_update_metadata);
            };
          } catch (const QueryExecutionError& e) {
            throw std::runtime_error(getErrorMessageFromCode(e.getErrorCode()));
          }
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
    throw std::runtime_error("Unsupported parent node for delete: " +
                             node->toString(RelRexToStringConfig::defaults()));
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

ExecutionResult RelAlgExecutor::executeProject(
    const RelProject* project,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    RenderInfo* render_info,
    const int64_t queue_time_ms,
    const std::optional<size_t> previous_count) {
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
  auto table_func_work_unit = createTableFunctionWorkUnit(
      table_func,
      eo.just_explain,
      /*is_gpu = */ co.device_type == ExecutorDeviceType::GPU);
  const auto body = table_func_work_unit.body;
  CHECK(body);

  const auto table_infos =
      get_table_infos(table_func_work_unit.exe_unit.input_descs, executor_);

  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     co.device_type,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     executor_->getCatalog(),
                                                     executor_->blockSize(),
                                                     executor_->gridSize()),
                         {}};

  auto global_hint = getGlobalQueryHint();
  auto use_resultset_recycler = canUseResultsetCache(eo, nullptr);
  if (use_resultset_recycler && has_valid_query_plan_dag(table_func)) {
    auto cached_resultset =
        executor_->getRecultSetRecyclerHolder().getCachedQueryResultSet(
            table_func->getQueryPlanDagHash());
    if (cached_resultset) {
      VLOG(1) << "recycle table function's resultset of the root node "
              << table_func->getRelNodeDagId() << " from resultset cache";
      result = {cached_resultset, cached_resultset->getTargetMetaInfo()};
      addTemporaryTable(-body->getId(), result.getDataPtr());
      return result;
    }
  }

  auto query_exec_time_begin = timer_start();
  try {
    result = {executor_->executeTableFunction(
                  table_func_work_unit.exe_unit, table_infos, co, eo, cat_),
              body->getOutputMetainfo()};
  } catch (const QueryExecutionError& e) {
    handlePersistentError(e.getErrorCode());
    CHECK(e.getErrorCode() == Executor::ERR_OUT_OF_GPU_MEM);
    throw std::runtime_error("Table function ran out of memory during execution");
  }
  auto query_exec_time = timer_stop(query_exec_time_begin);
  result.setQueueTime(queue_time_ms);
  auto resultset_ptr = result.getDataPtr();
  auto allow_auto_caching_resultset = resultset_ptr && resultset_ptr->hasValidBuffer() &&
                                      g_allow_auto_resultset_caching &&
                                      resultset_ptr->getBufferSizeBytes(co.device_type) <=
                                          g_auto_resultset_caching_threshold;
  bool keep_result = global_hint->isHintRegistered(QueryHint::kKeepTableFuncResult);
  if (use_resultset_recycler && (keep_result || allow_auto_caching_resultset) &&
      !hasStepForUnion()) {
    resultset_ptr->setExecTime(query_exec_time);
    resultset_ptr->setQueryPlanHash(table_func_work_unit.exe_unit.query_plan_dag_hash);
    resultset_ptr->setTargetMetaInfo(body->getOutputMetainfo());
    auto input_table_keys = ScanNodeTableKeyCollector::getScanNodeTableKey(body);
    resultset_ptr->setInputTableKeys(std::move(input_table_keys));
    if (allow_auto_caching_resultset) {
      VLOG(1) << "Automatically keep table function's query resultset to recycler";
    }
    executor_->getRecultSetRecyclerHolder().putQueryResultSetToCache(
        table_func_work_unit.exe_unit.query_plan_dag_hash,
        resultset_ptr->getInputTableKeys(),
        resultset_ptr,
        resultset_ptr->getBufferSizeBytes(co.device_type),
        target_exprs_owned_);
  } else {
    if (eo.keep_result) {
      if (g_cluster) {
        VLOG(1) << "Query hint \'keep_table_function_result\' is ignored since we do not "
                   "support resultset recycling on distributed mode";
      } else if (hasStepForUnion()) {
        VLOG(1) << "Query hint \'keep_table_function_result\' is ignored since a query "
                   "has union-(all) operator";
      } else if (is_validate_or_explain_query(eo)) {
        VLOG(1) << "Query hint \'keep_table_function_result\' is ignored since a query "
                   "is either validate or explain query";
      } else {
        VLOG(1) << "Query hint \'keep_table_function_result\' is ignored";
      }
    }
  }

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

void RelAlgExecutor::computeWindow(const WorkUnit& work_unit,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   ColumnCacheMap& column_cache_map,
                                   const int64_t queue_time_ms) {
  auto query_infos = get_table_infos(work_unit.exe_unit.input_descs, executor_);
  CHECK_EQ(query_infos.size(), size_t(1));
  if (query_infos.front().info.fragments.size() != 1) {
    throw std::runtime_error(
        "Only single fragment tables supported for window functions for now");
  }
  if (eo.executor_type == ::ExecutorType::Extern) {
    return;
  }
  query_infos.push_back(query_infos.front());
  auto window_project_node_context = WindowProjectNodeContext::create(executor_);
  // a query may hold multiple window functions having the same partition by condition
  // then after building the first hash partition we can reuse it for the rest of
  // the window functions
  // here, a cached partition can be shared via multiple window function contexts as is
  // but sorted partition should be copied to reuse since we use it for (intermediate)
  // output buffer
  // todo (yoonmin) : support recycler for window function computation?
  std::unordered_map<QueryPlanHash, std::shared_ptr<HashJoin>> partition_cache;
  std::unordered_map<QueryPlanHash, std::shared_ptr<std::vector<int64_t>>>
      sorted_partition_cache;
  std::unordered_map<QueryPlanHash, size_t> sorted_partition_key_ref_count_map;
  std::unordered_map<size_t, std::unique_ptr<WindowFunctionContext>>
      window_function_context_map;
  for (size_t target_index = 0; target_index < work_unit.exe_unit.target_exprs.size();
       ++target_index) {
    const auto& target_expr = work_unit.exe_unit.target_exprs[target_index];
    const auto window_func = dynamic_cast<const Analyzer::WindowFunction*>(target_expr);
    if (!window_func) {
      continue;
    }
    // Always use baseline layout hash tables for now, make the expression a tuple.
    const auto& partition_keys = window_func->getPartitionKeys();
    std::shared_ptr<Analyzer::BinOper> partition_key_cond;
    if (partition_keys.size() >= 1) {
      std::shared_ptr<Analyzer::Expr> partition_key_tuple;
      if (partition_keys.size() > 1) {
        partition_key_tuple = makeExpr<Analyzer::ExpressionTuple>(partition_keys);
      } else {
        CHECK_EQ(partition_keys.size(), size_t(1));
        partition_key_tuple = partition_keys.front();
      }
      // Creates a tautology equality with the partition expression on both sides.
      partition_key_cond =
          makeExpr<Analyzer::BinOper>(kBOOLEAN,
                                      kBW_EQ,
                                      kONE,
                                      partition_key_tuple,
                                      transform_to_inner(partition_key_tuple.get()));
    }
    auto context =
        createWindowFunctionContext(window_func,
                                    partition_key_cond /*nullptr if no partition key*/,
                                    partition_cache,
                                    sorted_partition_key_ref_count_map,
                                    work_unit,
                                    query_infos,
                                    co,
                                    column_cache_map,
                                    executor_->getRowSetMemoryOwner());
    CHECK(window_function_context_map.emplace(target_index, std::move(context)).second);
  }

  for (auto& kv : window_function_context_map) {
    kv.second->compute(sorted_partition_key_ref_count_map, sorted_partition_cache);
    window_project_node_context->addWindowFunctionContext(std::move(kv.second), kv.first);
  }
}

std::unique_ptr<WindowFunctionContext> RelAlgExecutor::createWindowFunctionContext(
    const Analyzer::WindowFunction* window_func,
    const std::shared_ptr<Analyzer::BinOper>& partition_key_cond,
    std::unordered_map<QueryPlanHash, std::shared_ptr<HashJoin>>& partition_cache,
    std::unordered_map<QueryPlanHash, size_t>& sorted_partition_key_ref_count_map,
    const WorkUnit& work_unit,
    const std::vector<InputTableInfo>& query_infos,
    const CompilationOptions& co,
    ColumnCacheMap& column_cache_map,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  const size_t elem_count = query_infos.front().info.fragments.front().getNumTuples();
  const auto memory_level = co.device_type == ExecutorDeviceType::GPU
                                ? MemoryLevel::GPU_LEVEL
                                : MemoryLevel::CPU_LEVEL;
  std::unique_ptr<WindowFunctionContext> context;
  auto partition_cache_key = work_unit.body->getQueryPlanDagHash();
  if (partition_key_cond) {
    auto partition_cond_str = partition_key_cond->toString();
    auto partition_key_hash = boost::hash_value(partition_cond_str);
    boost::hash_combine(partition_cache_key, partition_key_hash);
    std::shared_ptr<HashJoin> partition_ptr;
    auto cached_hash_table_it = partition_cache.find(partition_cache_key);
    if (cached_hash_table_it != partition_cache.end()) {
      partition_ptr = cached_hash_table_it->second;
      VLOG(1) << "Reuse a hash table to compute window function context (key: "
              << partition_cache_key << ", partition condition: " << partition_cond_str
              << ")";
    } else {
      const auto hash_table_or_err = executor_->buildHashTableForQualifier(
          partition_key_cond,
          query_infos,
          memory_level,
          JoinType::INVALID,  // for window function
          HashType::OneToMany,
          column_cache_map,
          work_unit.exe_unit.hash_table_build_plan_dag,
          work_unit.exe_unit.query_hint,
          work_unit.exe_unit.table_id_to_node_map);
      if (!hash_table_or_err.fail_reason.empty()) {
        throw std::runtime_error(hash_table_or_err.fail_reason);
      }
      CHECK(hash_table_or_err.hash_table->getHashType() == HashType::OneToMany);
      partition_ptr = hash_table_or_err.hash_table;
      CHECK(partition_cache.insert(std::make_pair(partition_cache_key, partition_ptr))
                .second);
      VLOG(1) << "Put a generated hash table for computing window function context to "
                 "cache (key: "
              << partition_cache_key << ", partition condition: " << partition_cond_str
              << ")";
    }
    CHECK(partition_ptr);
    auto aggregate_tree_fanout = g_window_function_aggregation_tree_fanout;
    if (work_unit.exe_unit.query_hint.aggregate_tree_fanout != aggregate_tree_fanout) {
      aggregate_tree_fanout = work_unit.exe_unit.query_hint.aggregate_tree_fanout;
      VLOG(1) << "Aggregate tree's fanout is set to " << aggregate_tree_fanout;
    }
    context = std::make_unique<WindowFunctionContext>(window_func,
                                                      partition_cache_key,
                                                      partition_ptr,
                                                      elem_count,
                                                      co.device_type,
                                                      row_set_mem_owner,
                                                      aggregate_tree_fanout);
  } else {
    context = std::make_unique<WindowFunctionContext>(
        window_func, elem_count, co.device_type, row_set_mem_owner);
  }
  const auto& order_keys = window_func->getOrderKeys();
  if (!order_keys.empty()) {
    auto sorted_partition_cache_key = partition_cache_key;
    for (auto& order_key : order_keys) {
      boost::hash_combine(sorted_partition_cache_key, order_key->toString());
    }
    for (auto& collation : window_func->getCollation()) {
      boost::hash_combine(sorted_partition_cache_key, collation.toString());
    }
    context->setSortedPartitionCacheKey(sorted_partition_cache_key);
    auto cache_key_cnt_it =
        sorted_partition_key_ref_count_map.try_emplace(sorted_partition_cache_key, 1);
    if (!cache_key_cnt_it.second) {
      sorted_partition_key_ref_count_map[sorted_partition_cache_key] =
          cache_key_cnt_it.first->second + 1;
    }

    std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
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
                                              nullptr,
                                              /*thread_idx=*/0,
                                              chunks_owner,
                                              column_cache_map);

      CHECK_EQ(join_col_elem_count, elem_count);
      context->addOrderColumn(column, order_col->get_type_info(), chunks_owner);
    }
  }
  if (context->getWindowFunction()->hasFraming()) {
    // todo (yoonmin) : if we try to support generic window function expression without
    // extra project node, we need to revisit here b/c the current logic assumes that
    // window function expression has a single input source
    auto& window_function_expression_args = window_func->getArgs();
    std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
    for (auto& expr : window_function_expression_args) {
      if (const auto arg_col_var =
              std::dynamic_pointer_cast<const Analyzer::ColumnVar>(expr)) {
        auto const [column, join_col_elem_count] = ColumnFetcher::getOneColumnFragment(
            executor_,
            *arg_col_var,
            query_infos.front().info.fragments.front(),
            memory_level,
            0,
            nullptr,
            /*thread_idx=*/0,
            chunks_owner,
            column_cache_map);

        CHECK_EQ(join_col_elem_count, elem_count);
        context->addColumnBufferForWindowFunctionExpression(column, chunks_owner);
      }
    }
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

bool sameTypeInfo(std::vector<TargetMetaInfo> const& lhs,
                  std::vector<TargetMetaInfo> const& rhs) {
  if (lhs.size() == rhs.size()) {
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i].get_type_info() != rhs[i].get_type_info()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool isGeometry(TargetMetaInfo const& target_meta_info) {
  return target_meta_info.get_type_info().is_geometry();
}

ExecutionResult RelAlgExecutor::executeUnion(const RelLogicalUnion* logical_union,
                                             const RaExecutionSequence& seq,
                                             const CompilationOptions& co,
                                             const ExecutionOptions& eo,
                                             RenderInfo* render_info,
                                             const int64_t queue_time_ms) {
  auto timer = DEBUG_TIMER(__func__);
  if (!logical_union->isAll()) {
    throw std::runtime_error("UNION without ALL is not supported yet.");
  }
  // Will throw a std::runtime_error if types don't match.
  logical_union->checkForMatchingMetaInfoTypes();
  logical_union->setOutputMetainfo(logical_union->getInput(0)->getOutputMetainfo());
  if (boost::algorithm::any_of(logical_union->getOutputMetainfo(), isGeometry)) {
    throw std::runtime_error("UNION does not support subqueries with geo-columns.");
  }
  auto work_unit =
      createUnionWorkUnit(logical_union, {{}, SortAlgorithm::Default, 0, 0}, eo);
  return executeWorkUnit(work_unit,
                         logical_union->getOutputMetainfo(),
                         false,
                         CompilationOptions::makeCpuOnly(co),
                         eo,
                         render_info,
                         queue_time_ms);
}

ExecutionResult RelAlgExecutor::executeLogicalValues(
    const RelLogicalValues* logical_values,
    const ExecutionOptions& eo) {
  auto timer = DEBUG_TIMER(__func__);
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
                                         false,
                                         /*is_varlen_projection=*/false});
  }

  std::shared_ptr<ResultSet> rs{
      ResultSetLogicalValuesBuilder{logical_values,
                                    target_infos,
                                    ExecutorDeviceType::CPU,
                                    query_mem_desc,
                                    executor_->getRowSetMemoryOwner(),
                                    executor_}
          .build()};

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
                                        executor_->getCatalog(),
                                        executor_->blockSize(),
                                        executor_->gridSize());

  std::vector<TargetMetaInfo> empty_targets;
  return {rs, empty_targets};
}

ExecutionResult RelAlgExecutor::executeSimpleInsert(
    const Analyzer::Query& query,
    Fragmenter_Namespace::InsertDataLoader& inserter,
    const Catalog_Namespace::SessionInfo& session) {
  // Note: We currently obtain an executor for this method, but we do not need it.
  // Therefore, we skip the executor state setup in the regular execution path. In the
  // future, we will likely want to use the executor to evaluate expressions in the insert
  // statement.

  const auto& values_lists = query.get_values_lists();
  const int table_id = query.get_result_table_id();
  const auto& col_id_list = query.get_result_col_list();
  size_t rows_number = values_lists.size();
  size_t leaf_count = inserter.getLeafCount();
  const auto td = cat_.getMetadataForTable(table_id);
  CHECK(td);
  size_t rows_per_leaf = rows_number;
  if (td->nShards == 0) {
    rows_per_leaf =
        ceil(static_cast<double>(rows_number) / static_cast<double>(leaf_count));
  }
  auto max_number_of_rows_per_package =
      std::max(size_t(1), std::min(rows_per_leaf, size_t(64 * 1024)));

  std::vector<const ColumnDescriptor*> col_descriptors;
  std::vector<int> col_ids;
  std::unordered_map<int, std::unique_ptr<uint8_t[]>> col_buffers;
  std::unordered_map<int, std::vector<std::string>> str_col_buffers;
  std::unordered_map<int, std::vector<ArrayDatum>> arr_col_buffers;
  std::unordered_map<int, int> sequential_ids;

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
              col_id,
              std::make_unique<uint8_t[]>(cd->columnType.get_size() *
                                          max_number_of_rows_per_package));
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
          std::unique_ptr<uint8_t[]>(new uint8_t[cd->columnType.get_logical_size() *
                                                 max_number_of_rows_per_package]()));
      CHECK(it_ok.second);
    }
    col_descriptors.push_back(cd);
    sequential_ids[col_id] = col_ids.size();
    col_ids.push_back(col_id);
  }

  // mark the target table's cached item as dirty
  std::vector<int> table_chunk_key_prefix{cat_.getCurrentDB().dbId, table_id};
  auto table_key = boost::hash_value(table_chunk_key_prefix);
  ResultSetCacheInvalidator::invalidateCachesByTable(table_key);
  UpdateTriggeredCacheInvalidator::invalidateCachesByTable(table_key);

  size_t start_row = 0;
  size_t rows_left = rows_number;
  while (rows_left != 0) {
    // clear the buffers
    for (const auto& kv : col_buffers) {
      memset(kv.second.get(), 0, max_number_of_rows_per_package);
    }
    for (auto& kv : str_col_buffers) {
      kv.second.clear();
    }
    for (auto& kv : arr_col_buffers) {
      kv.second.clear();
    }

    auto package_size = std::min(rows_left, max_number_of_rows_per_package);
    // Note: if there will be use cases with batch inserts with lots of rows, it might be
    // more efficient to do the loops below column by column instead of row by row.
    // But for now I consider such a refactoring not worth investigating, as we have more
    // efficient ways to insert many rows anyway.
    for (size_t row_idx = 0; row_idx < package_size; ++row_idx) {
      const auto& values_list = values_lists[row_idx + start_row];
      for (size_t col_idx = 0; col_idx < col_descriptors.size(); ++col_idx) {
        CHECK(values_list.size() == col_descriptors.size());
        auto col_cv =
            dynamic_cast<const Analyzer::Constant*>(values_list[col_idx]->get_expr());
        if (!col_cv) {
          auto col_cast =
              dynamic_cast<const Analyzer::UOper*>(values_list[col_idx]->get_expr());
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
            auto col_data = reinterpret_cast<int8_t*>(col_data_bytes);
            auto null_bool_val =
                col_datum.boolval == inline_fixed_encoding_null_val(cd->columnType);
            col_data[row_idx] = col_cv->get_is_null() || null_bool_val
                                    ? inline_fixed_encoding_null_val(cd->columnType)
                                    : (col_datum.boolval ? 1 : 0);
            break;
          }
          case kTINYINT: {
            auto col_data = reinterpret_cast<int8_t*>(col_data_bytes);
            col_data[row_idx] = col_cv->get_is_null()
                                    ? inline_fixed_encoding_null_val(cd->columnType)
                                    : col_datum.tinyintval;
            break;
          }
          case kSMALLINT: {
            auto col_data = reinterpret_cast<int16_t*>(col_data_bytes);
            col_data[row_idx] = col_cv->get_is_null()
                                    ? inline_fixed_encoding_null_val(cd->columnType)
                                    : col_datum.smallintval;
            break;
          }
          case kINT: {
            auto col_data = reinterpret_cast<int32_t*>(col_data_bytes);
            col_data[row_idx] = col_cv->get_is_null()
                                    ? inline_fixed_encoding_null_val(cd->columnType)
                                    : col_datum.intval;
            break;
          }
          case kBIGINT:
          case kDECIMAL:
          case kNUMERIC: {
            auto col_data = reinterpret_cast<int64_t*>(col_data_bytes);
            col_data[row_idx] = col_cv->get_is_null()
                                    ? inline_fixed_encoding_null_val(cd->columnType)
                                    : col_datum.bigintval;
            break;
          }
          case kFLOAT: {
            auto col_data = reinterpret_cast<float*>(col_data_bytes);
            col_data[row_idx] = col_datum.floatval;
            break;
          }
          case kDOUBLE: {
            auto col_data = reinterpret_cast<double*>(col_data_bytes);
            col_data[row_idx] = col_datum.doubleval;
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
                    insert_one_dict_str(
                        &reinterpret_cast<uint8_t*>(col_data_bytes)[row_idx],
                        cd,
                        col_cv,
                        cat_);
                    break;
                  case 2:
                    insert_one_dict_str(
                        &reinterpret_cast<uint16_t*>(col_data_bytes)[row_idx],
                        cd,
                        col_cv,
                        cat_);
                    break;
                  case 4:
                    insert_one_dict_str(
                        &reinterpret_cast<int32_t*>(col_data_bytes)[row_idx],
                        cd,
                        col_cv,
                        cat_);
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
            col_data[row_idx] = col_cv->get_is_null()
                                    ? inline_fixed_encoding_null_val(cd->columnType)
                                    : col_datum.bigintval;
            break;
          }
          case kARRAY: {
            const auto is_null = col_cv->get_is_null();
            const auto size = cd->columnType.get_size();
            const SQLTypeInfo elem_ti = cd->columnType.get_elem_type();
            // POINT coords: [un]compressed coords always need to be encoded, even if NULL
            const auto is_point_coords =
                (cd->isGeoPhyCol && elem_ti.get_type() == kTINYINT);
            if (is_null && !is_point_coords) {
              if (size > 0) {
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
                                       " values, " + "received " +
                                       std::to_string(l.size()));
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
                insert_one_dict_str(
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
                p = append_datum(p, c->get_constval(), elem_ti);
                CHECK(p);
              }
              arr_col_buffers[col_ids[col_idx]].push_back(ArrayDatum(len, buf, is_null));
            }
            break;
          }
          case kPOINT:
          case kMULTIPOINT:
          case kLINESTRING:
          case kMULTILINESTRING:
          case kPOLYGON:
          case kMULTIPOLYGON:
            str_col_buffers[col_ids[col_idx]].push_back(
                col_datum.stringval ? *col_datum.stringval : "");
            break;
          default:
            CHECK(false);
        }
      }
    }
    start_row += package_size;
    rows_left -= package_size;

    Fragmenter_Namespace::InsertData insert_data;
    insert_data.databaseId = cat_.getCurrentDB().dbId;
    insert_data.tableId = table_id;
    insert_data.data.resize(col_ids.size());
    insert_data.columnIds = col_ids;
    for (const auto& kv : col_buffers) {
      DataBlockPtr p;
      p.numbersPtr = reinterpret_cast<int8_t*>(kv.second.get());
      insert_data.data[sequential_ids[kv.first]] = p;
    }
    for (auto& kv : str_col_buffers) {
      DataBlockPtr p;
      p.stringsPtr = &kv.second;
      insert_data.data[sequential_ids[kv.first]] = p;
    }
    for (auto& kv : arr_col_buffers) {
      DataBlockPtr p;
      p.arraysPtr = &kv.second;
      insert_data.data[sequential_ids[kv.first]] = p;
    }
    insert_data.numRows = package_size;
    auto data_memory_holder = import_export::fill_missing_columns(&cat_, insert_data);
    inserter.insertData(session, insert_data);
  }

  auto rs = std::make_shared<ResultSet>(TargetInfoList{},
                                        ExecutorDeviceType::CPU,
                                        QueryMemoryDescriptor(),
                                        executor_->getRowSetMemoryOwner(),
                                        nullptr,
                                        0,
                                        0);
  std::vector<TargetMetaInfo> empty_targets;
  return {rs, empty_targets};
}

namespace {

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
  auto order_entries = get_order_entries(sort);
  if (it != leaf_results_.end()) {
    // Add any transient string literals to the sdp on the agg
    const auto source_work_unit = createSortInputWorkUnit(sort, order_entries, eo);
    executor_->addTransientStringLiterals(source_work_unit.exe_unit,
                                          executor_->row_set_mem_owner_);
    // Handle push-down for LIMIT for multi-node
    auto& aggregated_result = it->second;
    auto& result_rows = aggregated_result.rs;
    const size_t limit = sort->getLimit();
    const size_t offset = sort->getOffset();
    if (limit || offset) {
      if (!order_entries.empty()) {
        result_rows->sort(order_entries, limit + offset, executor_);
      }
      result_rows->dropFirstN(offset);
      if (limit) {
        result_rows->keepFirstN(limit);
      }
    }

    if (render_info) {
      // We've hit a sort step that is the very last step
      // in a distributed render query. We'll fill in the render targets
      // since we have all that data needed to do so. This is normally
      // done in executeWorkUnit, but that is bypassed in this case.
      build_render_targets(*render_info,
                           source_work_unit.exe_unit.target_exprs,
                           aggregated_result.targets_meta);
    }

    ExecutionResult result(result_rows, aggregated_result.targets_meta);
    sort->setOutputMetainfo(aggregated_result.targets_meta);

    return result;
  }

  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  bool is_desc{false};
  bool use_speculative_top_n_sort{false};

  auto execute_sort_query = [this,
                             sort,
                             &source,
                             &is_aggregate,
                             &eo,
                             &co,
                             render_info,
                             queue_time_ms,
                             &groupby_exprs,
                             &is_desc,
                             &order_entries,
                             &use_speculative_top_n_sort]() -> ExecutionResult {
    const size_t limit = sort->getLimit();
    const size_t offset = sort->getOffset();
    // check whether sort's input is cached
    auto source_node = sort->getInput(0);
    CHECK(source_node);
    ExecutionResult source_result{nullptr, {}};
    auto source_query_plan_dag = source_node->getQueryPlanDagHash();
    bool enable_resultset_recycler = canUseResultsetCache(eo, render_info);
    if (enable_resultset_recycler && has_valid_query_plan_dag(source_node) &&
        !sort->isEmptyResult()) {
      if (auto cached_resultset =
              executor_->getRecultSetRecyclerHolder().getCachedQueryResultSet(
                  source_query_plan_dag)) {
        CHECK(cached_resultset->canUseSpeculativeTopNSort());
        VLOG(1) << "recycle resultset of the root node " << source_node->getRelNodeDagId()
                << " from resultset cache";
        source_result =
            ExecutionResult{cached_resultset, cached_resultset->getTargetMetaInfo()};
        if (temporary_tables_.find(-source_node->getId()) == temporary_tables_.end()) {
          addTemporaryTable(-source_node->getId(), cached_resultset);
        }
        use_speculative_top_n_sort = *cached_resultset->canUseSpeculativeTopNSort() &&
                                     co.device_type == ExecutorDeviceType::GPU;
        source_node->setOutputMetainfo(cached_resultset->getTargetMetaInfo());
        sort->setOutputMetainfo(source_node->getOutputMetainfo());
      }
    }
    if (!source_result.getDataPtr()) {
      const auto source_work_unit = createSortInputWorkUnit(sort, order_entries, eo);
      is_desc = first_oe_is_desc(order_entries);
      ExecutionOptions eo_copy = {
          eo.output_columnar_hint,
          eo.keep_result,
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
          eo.running_query_interrupt_freq,
          eo.pending_query_interrupt_freq,
          eo.executor_type,
      };

      groupby_exprs = source_work_unit.exe_unit.groupby_exprs;
      source_result = executeWorkUnit(source_work_unit,
                                      source->getOutputMetainfo(),
                                      is_aggregate,
                                      co,
                                      eo_copy,
                                      render_info,
                                      queue_time_ms);
      use_speculative_top_n_sort =
          source_result.getDataPtr() && source_result.getRows()->hasValidBuffer() &&
          use_speculative_top_n(source_work_unit.exe_unit,
                                source_result.getRows()->getQueryMemDesc());
    }
    if (render_info && render_info->isInSitu()) {
      return source_result;
    }
    if (source_result.isFilterPushDownEnabled()) {
      return source_result;
    }
    auto rows_to_sort = source_result.getRows();
    if (eo.just_explain) {
      return {rows_to_sort, {}};
    }
    if (sort->collationCount() != 0 && !rows_to_sort->definitelyHasNoRows() &&
        !use_speculative_top_n_sort) {
      const size_t top_n = limit == 0 ? 0 : limit + offset;
      rows_to_sort->sort(order_entries, top_n, executor_);
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
  };

  try {
    return execute_sort_query();
  } catch (const SpeculativeTopNFailed& e) {
    CHECK_EQ(size_t(1), groupby_exprs.size());
    CHECK(groupby_exprs.front());
    speculative_topn_blacklist_.add(groupby_exprs.front(), is_desc);
    return execute_sort_query();
  }
}

RelAlgExecutor::WorkUnit RelAlgExecutor::createSortInputWorkUnit(
    const RelSort* sort,
    std::list<Analyzer::OrderEntry>& order_entries,
    const ExecutionOptions& eo) {
  const auto source = sort->getInput(0);
  const size_t limit = sort->getLimit();
  const size_t offset = sort->getOffset();
  const size_t scan_limit = sort->collationCount() ? 0 : get_scan_limit(source, limit);
  const size_t scan_total_limit =
      scan_limit ? get_scan_limit(source, scan_limit + offset) : 0;
  size_t max_groups_buffer_entry_guess{
      scan_total_limit ? scan_total_limit : g_default_max_groups_buffer_entry_guess};
  SortAlgorithm sort_algorithm{SortAlgorithm::SpeculativeTopN};
  SortInfo sort_info{
      order_entries, sort_algorithm, limit, offset, sort->isLimitDelivered()};
  auto source_work_unit = createWorkUnit(source, sort_info, eo);
  const auto& source_exe_unit = source_work_unit.exe_unit;

  // we do not allow sorting geometry or array types
  for (auto order_entry : order_entries) {
    CHECK_GT(order_entry.tle_no, 0);  // tle_no is a 1-base index
    const auto& te = source_exe_unit.target_exprs[order_entry.tle_no - 1];
    const auto& ti = get_target_info(te, false);
    if (ti.sql_type.is_geometry() || ti.sql_type.is_array()) {
      throw std::runtime_error(
          "Columns with geometry or array types cannot be used in an ORDER BY clause.");
    }
  }

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
                              source_exe_unit.target_exprs_original_type_infos,
                              nullptr,
                              {sort_info.order_entries,
                               sort_algorithm,
                               limit,
                               offset,
                               sort_info.limit_delivered},
                              scan_total_limit,
                              source_exe_unit.query_hint,
                              source_exe_unit.query_plan_dag_hash,
                              source_exe_unit.hash_table_build_plan_dag,
                              source_exe_unit.table_id_to_node_map,
                              source_exe_unit.use_bump_allocator,
                              source_exe_unit.union_all,
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

bool is_projection(const RelAlgExecutionUnit& ra_exe_unit) {
  return ra_exe_unit.groupby_exprs.size() == 1 && !ra_exe_unit.groupby_exprs.front();
}

bool can_output_columnar(const RelAlgExecutionUnit& ra_exe_unit,
                         const RenderInfo* render_info,
                         const RelAlgNode* body) {
  if (!is_projection(ra_exe_unit)) {
    return false;
  }
  if (render_info && render_info->isInSitu()) {
    return false;
  }
  if (!ra_exe_unit.sort_info.order_entries.empty()) {
    // disable output columnar when we have top-sort node query
    return false;
  }
  for (const auto& target_expr : ra_exe_unit.target_exprs) {
    // We don't currently support varlen columnar projections, so
    // return false if we find one
    if (target_expr->get_type_info().is_varlen()) {
      return false;
    }
  }
  if (auto top_project = dynamic_cast<const RelProject*>(body)) {
    if (top_project->isRowwiseOutputForced()) {
      return false;
    }
  }
  return true;
}

bool should_output_columnar(const RelAlgExecutionUnit& ra_exe_unit) {
  return g_columnar_large_projections &&
         ra_exe_unit.scan_limit >= g_columnar_large_projections_threshold;
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
    const auto error_rate = static_cast<Analyzer::AggExpr*>(target_expr)->get_arg1();
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
    const ExecutionOptions& eo_in,
    RenderInfo* render_info,
    const int64_t queue_time_ms,
    const std::optional<size_t> previous_count) {
  INJECT_TIMER(executeWorkUnit);
  auto timer = DEBUG_TIMER(__func__);
  auto query_exec_time_begin = timer_start();

  const auto query_infos = get_table_infos(work_unit.exe_unit.input_descs, executor_);
  check_none_encoded_string_cast_tuple_limit(query_infos, work_unit.exe_unit);

  auto co = co_in;
  auto eo = eo_in;
  ColumnCacheMap column_cache;
  ScopeGuard clearWindowContextIfNecessary = [&]() {
    if (is_window_execution_unit(work_unit.exe_unit)) {
      WindowProjectNodeContext::reset(executor_);
    }
  };
  if (is_window_execution_unit(work_unit.exe_unit)) {
    if (!g_enable_window_functions) {
      throw std::runtime_error("Window functions support is disabled");
    }
    co.device_type = ExecutorDeviceType::CPU;
    co.allow_lazy_fetch = false;
    computeWindow(work_unit, co, eo, column_cache, queue_time_ms);
  }
  if (!eo.just_explain && eo.find_push_down_candidates) {
    // find potential candidates:
    auto selected_filters = selectFiltersToBePushedDown(work_unit, co, eo);
    if (!selected_filters.empty() || eo.just_calcite_explain) {
      return ExecutionResult(selected_filters, eo.find_push_down_candidates);
    }
  }
  if (render_info && render_info->isInSitu()) {
    co.allow_lazy_fetch = false;
  }
  const auto body = work_unit.body;
  CHECK(body);
  auto it = leaf_results_.find(body->getId());
  VLOG(3) << "body->getId()=" << body->getId()
          << " body->toString()=" << body->toString(RelRexToStringConfig::defaults())
          << " it==leaf_results_.end()=" << (it == leaf_results_.end());
  if (it != leaf_results_.end()) {
    executor_->addTransientStringLiterals(work_unit.exe_unit,
                                          executor_->row_set_mem_owner_);
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

  // register query hint if query_dag_ is valid
  ra_exe_unit.query_hint = RegisteredQueryHint::defaults();
  if (query_dag_) {
    auto candidate = query_dag_->getQueryHint(body);
    if (candidate) {
      ra_exe_unit.query_hint = *candidate;
    }
  }

  const auto& query_hints = ra_exe_unit.query_hint;
  ScopeGuard reset_cuda_block_grid_sizes = [&,
                                            orig_block_size = executor_->blockSize(),
                                            orig_grid_size = executor_->gridSize()]() {
    if (cat_.getDataMgr().getCudaMgr()) {
      if (query_hints.isHintRegistered(QueryHint::kCudaBlockSize)) {
        if (orig_block_size) {
          executor_->setBlockSize(orig_block_size);
        } else {
          executor_->resetBlockSize();
        }
      }
      if (query_hints.isHintRegistered(QueryHint::kCudaGridSize)) {
        if (orig_grid_size) {
          executor_->setGridSize(orig_grid_size);
        } else {
          executor_->resetGridSize();
        }
      }
    }
  };

  if (co.device_type == ExecutorDeviceType::GPU) {
    if (query_hints.isHintRegistered(QueryHint::kCudaGridSize)) {
      if (!cat_.getDataMgr().getCudaMgr()) {
        VLOG(1) << "Skip CUDA grid size query hint: cannot detect CUDA device";
      } else {
        const auto num_sms = executor_->cudaMgr()->getMinNumMPsForAllDevices();
        const auto new_grid_size =
            std::round(num_sms * query_hints.cuda_grid_size_multiplier);
        const auto default_grid_size = executor_->gridSize();
        if (new_grid_size != default_grid_size) {
          VLOG(1) << "Change CUDA grid size: " << default_grid_size
                  << " (default_grid_size) -> " << new_grid_size << " (# SMs * "
                  << query_hints.cuda_grid_size_multiplier << ")";
          // todo (yoonmin): do we need to check a hard limit?
          executor_->setGridSize(new_grid_size);
        } else {
          VLOG(1) << "Skip CUDA grid size query hint: invalid grid size";
        }
      }
    }
    if (query_hints.isHintRegistered(QueryHint::kCudaBlockSize)) {
      if (!cat_.getDataMgr().getCudaMgr()) {
        VLOG(1) << "Skip CUDA block size query hint: cannot detect CUDA device";
      } else {
        int cuda_block_size = query_hints.cuda_block_size;
        int warp_size = executor_->warpSize();
        if (cuda_block_size >= warp_size) {
          cuda_block_size = (cuda_block_size + warp_size - 1) / warp_size * warp_size;
          VLOG(1) << "Change CUDA block size w.r.t warp size (" << warp_size
                  << "): " << executor_->blockSize() << " -> " << cuda_block_size;
        } else {
          VLOG(1) << "Change CUDA block size: " << executor_->blockSize() << " -> "
                  << cuda_block_size;
        }
        executor_->setBlockSize(cuda_block_size);
      }
    }
  }

  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  if (is_window_execution_unit(ra_exe_unit)) {
    CHECK_EQ(table_infos.size(), size_t(1));
    CHECK_EQ(table_infos.front().info.fragments.size(), size_t(1));
    max_groups_buffer_entry_guess =
        table_infos.front().info.fragments.front().getNumTuples();
    ra_exe_unit.scan_limit = max_groups_buffer_entry_guess;
  } else if (compute_output_buffer_size(ra_exe_unit) && !isRowidLookup(work_unit)) {
    if (previous_count && !exe_unit_has_quals(ra_exe_unit)) {
      ra_exe_unit.scan_limit = *previous_count;
    } else {
      // TODO(adb): enable bump allocator path for render queries
      if (can_use_bump_allocator(ra_exe_unit, co, eo) && !render_info) {
        ra_exe_unit.scan_limit = 0;
        ra_exe_unit.use_bump_allocator = true;
      } else if (eo.executor_type == ::ExecutorType::Extern) {
        ra_exe_unit.scan_limit = 0;
      } else if (!eo.just_explain) {
        const auto filter_count_all = getFilteredCountAll(work_unit, true, co, eo);
        if (filter_count_all) {
          ra_exe_unit.scan_limit = std::max(*filter_count_all, size_t(1));
        }
      }
    }
  }

  // when output_columnar_hint is true here, it means either 1) columnar output
  // configuration is on or 2) a user hint is given but we have to disable it if some
  // requirements are not satisfied
  if (can_output_columnar(ra_exe_unit, render_info, body)) {
    if (!eo.output_columnar_hint && should_output_columnar(ra_exe_unit)) {
      VLOG(1) << "Using columnar layout for projection as output size of "
              << ra_exe_unit.scan_limit << " rows exceeds threshold of "
              << g_columnar_large_projections_threshold << ".";
      eo.output_columnar_hint = true;
    }
  } else {
    eo.output_columnar_hint = false;
  }

  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     co.device_type,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     executor_->getCatalog(),
                                                     executor_->blockSize(),
                                                     executor_->gridSize()),
                         {}};

  auto execute_and_handle_errors = [&](const auto max_groups_buffer_entry_guess_in,
                                       const bool has_cardinality_estimation,
                                       const bool has_ndv_estimation) -> ExecutionResult {
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
                                         render_info,
                                         has_cardinality_estimation,
                                         column_cache),
              targets_meta};
    } catch (const QueryExecutionError& e) {
      if (!has_ndv_estimation && e.getErrorCode() < 0) {
        throw CardinalityEstimationRequired(/*range=*/0);
      }
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

  auto use_resultset_cache = canUseResultsetCache(eo, render_info);
  for (const auto& table_info : table_infos) {
    const auto td = cat_.getMetadataForTable(table_info.table_id);
    if (td && (td->isTemporaryTable() || td->isView)) {
      use_resultset_cache = false;
      if (eo.keep_result) {
        VLOG(1) << "Query hint \'keep_result\' is ignored since a query has either "
                   "temporary table or view";
      }
    }
  }

  auto cache_key = ra_exec_unit_desc_for_caching(ra_exe_unit);
  try {
    auto cached_cardinality = executor_->getCachedCardinality(cache_key);
    auto card = cached_cardinality.second;
    if (cached_cardinality.first && card >= 0) {
      result = execute_and_handle_errors(
          card, /*has_cardinality_estimation=*/true, /*has_ndv_estimation=*/false);
    } else {
      result = execute_and_handle_errors(
          max_groups_buffer_entry_guess,
          groups_approx_upper_bound(table_infos) <= g_big_group_threshold,
          /*has_ndv_estimation=*/false);
    }
  } catch (const CardinalityEstimationRequired& e) {
    // check the cardinality cache
    auto cached_cardinality = executor_->getCachedCardinality(cache_key);
    auto card = cached_cardinality.second;
    if (cached_cardinality.first && card >= 0) {
      result = execute_and_handle_errors(card, true, /*has_ndv_estimation=*/true);
    } else {
      const auto ndv_groups_estimation =
          getNDVEstimation(work_unit, e.range(), is_agg, co, eo);
      const auto estimated_groups_buffer_entry_guess =
          ndv_groups_estimation > 0 ? 2 * ndv_groups_estimation
                                    : std::min(groups_approx_upper_bound(table_infos),
                                               g_estimator_failure_max_groupby_size);
      CHECK_GT(estimated_groups_buffer_entry_guess, size_t(0));
      result = execute_and_handle_errors(
          estimated_groups_buffer_entry_guess, true, /*has_ndv_estimation=*/true);
      if (!(eo.just_validate || eo.just_explain)) {
        executor_->addToCardinalityCache(cache_key, estimated_groups_buffer_entry_guess);
      }
    }
  }

  result.setQueueTime(queue_time_ms);
  if (render_info) {
    build_render_targets(*render_info, work_unit.exe_unit.target_exprs, targets_meta);
    if (render_info->isInSitu()) {
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

  for (auto& target_info : result.getTargetsMeta()) {
    if (target_info.get_type_info().is_string() &&
        !target_info.get_type_info().is_dict_encoded_string()) {
      // currently, we do not support resultset caching if non-encoded string is projected
      use_resultset_cache = false;
      if (eo.keep_result) {
        VLOG(1) << "Query hint \'keep_result\' is ignored since a query has non-encoded "
                   "string column projection";
      }
    }
  }

  const auto res = result.getDataPtr();
  auto allow_auto_caching_resultset =
      res && res->hasValidBuffer() && g_allow_auto_resultset_caching &&
      res->getBufferSizeBytes(co.device_type) <= g_auto_resultset_caching_threshold;
  if (use_resultset_cache && (eo.keep_result || allow_auto_caching_resultset) &&
      !work_unit.exe_unit.sort_info.limit_delivered) {
    auto query_exec_time = timer_stop(query_exec_time_begin);
    res->setExecTime(query_exec_time);
    res->setQueryPlanHash(ra_exe_unit.query_plan_dag_hash);
    res->setTargetMetaInfo(body->getOutputMetainfo());
    auto input_table_keys = ScanNodeTableKeyCollector::getScanNodeTableKey(body);
    res->setInputTableKeys(std::move(input_table_keys));
    if (allow_auto_caching_resultset) {
      VLOG(1) << "Automatically keep query resultset to recycler";
    }
    res->setUseSpeculativeTopNSort(
        use_speculative_top_n(ra_exe_unit, res->getQueryMemDesc()));
    executor_->getRecultSetRecyclerHolder().putQueryResultSetToCache(
        ra_exe_unit.query_plan_dag_hash,
        res->getInputTableKeys(),
        res,
        res->getBufferSizeBytes(co.device_type),
        target_exprs_owned_);
  } else {
    if (eo.keep_result) {
      if (g_cluster) {
        VLOG(1) << "Query hint \'keep_result\' is ignored since we do not support "
                   "resultset recycling on distributed mode";
      } else if (hasStepForUnion()) {
        VLOG(1) << "Query hint \'keep_result\' is ignored since a query has union-(all) "
                   "operator";
      } else if (render_info && render_info->isInSitu()) {
        VLOG(1) << "Query hint \'keep_result\' is ignored since a query is classified as "
                   "a in-situ rendering query";
      } else if (is_validate_or_explain_query(eo)) {
        VLOG(1) << "Query hint \'keep_result\' is ignored since a query is either "
                   "validate or explain query";
      } else {
        VLOG(1) << "Query hint \'keep_result\' is ignored";
      }
    }
  }

  return result;
}

std::optional<size_t> RelAlgExecutor::getFilteredCountAll(const WorkUnit& work_unit,
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
      work_unit.exe_unit.createCountAllExecutionUnit(count.get());
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
                                   nullptr,
                                   false,
                                   column_cache);
  } catch (const foreign_storage::ForeignStorageException& error) {
    throw error;
  } catch (const QueryMustRunOnCpu&) {
    // force a retry of the top level query on CPU
    throw;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to run pre-flight filtered count with error " << e.what();
    return std::nullopt;
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
  for (const auto& simple_qual : ra_exe_unit.simple_quals) {
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
                                                            executor_->getCatalog(),
                                                            executor_->blockSize(),
                                                            executor_->gridSize()),
                                {}};

  const auto table_infos = get_table_infos(ra_exe_unit_in, executor_);
  auto max_groups_buffer_entry_guess = work_unit.max_groups_buffer_entry_guess;
  ExecutionOptions eo_no_multifrag{eo.output_columnar_hint,
                                   eo.keep_result,
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
                                   eo.allow_runtime_query_interrupt,
                                   eo.running_query_interrupt_freq,
                                   eo.pending_query_interrupt_freq,
                                   eo.executor_type,
                                   eo.outer_fragment_indices};

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
    render_info->forceNonInSitu();
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

namespace {
struct ErrorInfo {
  const char* code{nullptr};
  const char* description{nullptr};
};
ErrorInfo getErrorDescription(const int32_t error_code) {
  // 'designated initializers' don't compile on Windows for std 17
  // They require /std:c++20.  They been removed for the windows port.
  switch (error_code) {
    case Executor::ERR_DIV_BY_ZERO:
      return {"ERR_DIV_BY_ZERO", "Division by zero"};
    case Executor::ERR_OUT_OF_GPU_MEM:
      return {"ERR_OUT_OF_GPU_MEM",

              "Query couldn't keep the entire working set of columns in GPU memory"};
    case Executor::ERR_UNSUPPORTED_SELF_JOIN:
      return {"ERR_UNSUPPORTED_SELF_JOIN", "Self joins not supported yet"};
    case Executor::ERR_OUT_OF_CPU_MEM:
      return {"ERR_OUT_OF_CPU_MEM", "Not enough host memory to execute the query"};
    case Executor::ERR_OVERFLOW_OR_UNDERFLOW:
      return {"ERR_OVERFLOW_OR_UNDERFLOW", "Overflow or underflow"};
    case Executor::ERR_OUT_OF_TIME:
      return {"ERR_OUT_OF_TIME", "Query execution has exceeded the time limit"};
    case Executor::ERR_INTERRUPTED:
      return {"ERR_INTERRUPTED", "Query execution has been interrupted"};
    case Executor::ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED:
      return {"ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED",
              "Columnar conversion not supported for variable length types"};
    case Executor::ERR_TOO_MANY_LITERALS:
      return {"ERR_TOO_MANY_LITERALS", "Too many literals in the query"};
    case Executor::ERR_STRING_CONST_IN_RESULTSET:
      return {"ERR_STRING_CONST_IN_RESULTSET",

              "NONE ENCODED String types are not supported as input result set."};
    case Executor::ERR_OUT_OF_RENDER_MEM:
      return {"ERR_OUT_OF_RENDER_MEM",

              "Insufficient GPU memory for query results in render output buffer "
              "sized by render-mem-bytes"};
    case Executor::ERR_STREAMING_TOP_N_NOT_SUPPORTED_IN_RENDER_QUERY:
      return {"ERR_STREAMING_TOP_N_NOT_SUPPORTED_IN_RENDER_QUERY",
              "Streaming-Top-N not supported in Render Query"};
    case Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES:
      return {"ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES",
              "Multiple distinct values encountered"};
    case Executor::ERR_GEOS:
      return {"ERR_GEOS", "ERR_GEOS"};
    case Executor::ERR_WIDTH_BUCKET_INVALID_ARGUMENT:
      return {"ERR_WIDTH_BUCKET_INVALID_ARGUMENT",

              "Arguments of WIDTH_BUCKET function does not satisfy the condition"};
    default:
      return {nullptr, nullptr};
  }
}

}  // namespace

std::string RelAlgExecutor::getErrorMessageFromCode(const int32_t error_code) {
  if (error_code < 0) {
    return "Ran out of slots in the query output buffer";
  }
  const auto errorInfo = getErrorDescription(error_code);

  if (errorInfo.code) {
    return errorInfo.code + ": "s + errorInfo.description;
  } else {
    return "Other error: code "s + std::to_string(error_code);
  }
}

void RelAlgExecutor::executePostExecutionCallback() {
  if (post_execution_callback_) {
    VLOG(1) << "Running post execution callback.";
    (*post_execution_callback_)();
  }
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
  if (filter) {
    return createFilterWorkUnit(filter, sort_info, eo.just_explain);
  }
  LOG(FATAL) << "Unhandled node type: "
             << node->toString(RelRexToStringConfig::defaults());
  return {};
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
    auto cur_level_join_type = left_deep_join->getJoinType(nesting_level);
    if (cur_level_join_type == JoinType::SEMI || cur_level_join_type == JoinType::ANTI) {
      join_types[nesting_level - 1] = cur_level_join_type;
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
  VLOG(3) << "input_descs=" << shared::printContainer(input_descs);
  const auto query_infos = get_table_infos(input_descs, executor_);
  CHECK_EQ(size_t(1), compound->inputCount());
  const auto left_deep_join =
      dynamic_cast<const RelLeftDeepInnerJoin*>(compound->getInput(0));
  JoinQualsPerNestingLevel left_deep_join_quals;
  const auto join_types = left_deep_join ? left_deep_join_types(left_deep_join)
                                         : std::vector<JoinType>{get_join_type(compound)};
  std::vector<size_t> input_permutation;
  std::vector<size_t> left_deep_join_input_sizes;
  std::optional<unsigned> left_deep_tree_id;
  if (left_deep_join) {
    left_deep_tree_id = left_deep_join->getId();
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
  RelAlgTranslator translator(cat_,
                              query_state_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              eo.just_explain);
  const auto scalar_sources =
      translate_scalar_sources(compound, translator, eo.executor_type);
  const auto groupby_exprs = translate_groupby_exprs(compound, scalar_sources);
  const auto quals_cf = translate_quals(compound, translator);
  std::unordered_map<size_t, SQLTypeInfo> target_exprs_type_infos;
  const auto target_exprs = translate_targets(target_exprs_owned_,
                                              target_exprs_type_infos,
                                              scalar_sources,
                                              groupby_exprs,
                                              compound,
                                              translator,
                                              eo.executor_type);

  auto query_hint = RegisteredQueryHint::defaults();
  if (query_dag_) {
    auto candidate = query_dag_->getQueryHint(compound);
    if (candidate) {
      query_hint = *candidate;
    }
  }
  CHECK_EQ(compound->size(), target_exprs.size());
  const RelAlgExecutionUnit exe_unit = {input_descs,
                                        input_col_descs,
                                        quals_cf.simple_quals,
                                        rewrite_quals(quals_cf.quals),
                                        left_deep_join_quals,
                                        groupby_exprs,
                                        target_exprs,
                                        target_exprs_type_infos,
                                        nullptr,
                                        sort_info,
                                        0,
                                        query_hint,
                                        compound->getQueryPlanDagHash(),
                                        {},
                                        {},
                                        false,
                                        std::nullopt,
                                        query_state_};
  auto query_rewriter = std::make_unique<QueryRewriter>(query_infos, executor_);
  auto rewritten_exe_unit = query_rewriter->rewrite(exe_unit);
  const auto targets_meta = get_targets_meta(compound, rewritten_exe_unit.target_exprs);
  compound->setOutputMetainfo(targets_meta);
  auto& left_deep_trees_info = getLeftDeepJoinTreesInfo();
  if (left_deep_tree_id && left_deep_tree_id.has_value()) {
    left_deep_trees_info.emplace(left_deep_tree_id.value(),
                                 rewritten_exe_unit.join_quals);
  }
  if (has_valid_query_plan_dag(compound)) {
    auto join_info = QueryPlanDagExtractor::extractJoinInfo(
        compound, left_deep_tree_id, left_deep_trees_info, executor_);
    rewritten_exe_unit.hash_table_build_plan_dag = join_info.hash_table_plan_dag;
    rewritten_exe_unit.table_id_to_node_map = join_info.table_id_to_node_map;
  }
  return {rewritten_exe_unit,
          compound,
          g_default_max_groups_buffer_entry_guess,
          std::move(query_rewriter),
          input_permutation,
          left_deep_join_input_sizes};
}

std::shared_ptr<RelAlgTranslator> RelAlgExecutor::getRelAlgTranslator(
    const RelAlgNode* node) {
  auto input_to_nest_level = get_input_nest_levels(node, {});
  const auto left_deep_join =
      dynamic_cast<const RelLeftDeepInnerJoin*>(node->getInput(0));
  const auto join_types = left_deep_join ? left_deep_join_types(left_deep_join)
                                         : std::vector<JoinType>{get_join_type(node)};
  return std::make_shared<RelAlgTranslator>(
      cat_, query_state_, executor_, input_to_nest_level, join_types, now_, false);
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
  RelAlgTranslator translator(
      cat_, query_state_, executor_, input_to_nest_level, join_types, now_, just_explain);
  const auto rex_condition_cf = rex_to_conjunctive_form(join_condition);
  std::list<std::shared_ptr<Analyzer::Expr>> join_condition_quals;
  for (const auto rex_condition_component : rex_condition_cf) {
    const auto bw_equals = get_bitwise_equals_conjunction(rex_condition_component);
    const auto join_condition = reverse_logical_distribution(
        translator.translate(bw_equals ? bw_equals.get() : rex_condition_component));
    auto join_condition_cf = qual_to_conjunctive_form(join_condition);

    auto append_folded_cf_quals = [&join_condition_quals](const auto& cf_quals) {
      for (const auto& cf_qual : cf_quals) {
        join_condition_quals.emplace_back(fold_expr(cf_qual.get()));
      }
    };

    append_folded_cf_quals(join_condition_cf.quals);
    append_folded_cf_quals(join_condition_cf.simple_quals);
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
    for (const auto& qual : join_condition_quals) {
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
    CHECK(join_types[rte_idx - 1] == JoinType::INNER ||
          join_types[rte_idx - 1] == JoinType::SEMI ||
          join_types[rte_idx - 1] == JoinType::ANTI);
    result[rte_idx - 1].type = join_types[rte_idx - 1];
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

std::vector<Analyzer::Expr*> get_raw_pointers(
    std::vector<std::shared_ptr<Analyzer::Expr>> const& input) {
  std::vector<Analyzer::Expr*> output(input.size());
  auto const raw_ptr = [](auto& shared_ptr) { return shared_ptr.get(); };
  std::transform(input.cbegin(), input.cend(), output.begin(), raw_ptr);
  return output;
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

  RelAlgTranslator translator(cat_,
                              query_state_,
                              executor_,
                              input_to_nest_level,
                              {join_type},
                              now_,
                              just_explain);
  CHECK_EQ(size_t(1), aggregate->inputCount());
  const auto source = aggregate->getInput(0);
  const auto& in_metainfo = source->getOutputMetainfo();
  const auto scalar_sources =
      synthesize_inputs(aggregate, size_t(0), in_metainfo, input_to_nest_level);
  const auto groupby_exprs = translate_groupby_exprs(aggregate, scalar_sources);
  std::unordered_map<size_t, SQLTypeInfo> target_exprs_type_infos;
  const auto target_exprs = translate_targets(target_exprs_owned_,
                                              target_exprs_type_infos,
                                              scalar_sources,
                                              groupby_exprs,
                                              aggregate,
                                              translator);

  const auto query_infos = get_table_infos(input_descs, executor_);

  const auto targets_meta = get_targets_meta(aggregate, target_exprs);
  aggregate->setOutputMetainfo(targets_meta);
  auto query_hint = RegisteredQueryHint::defaults();
  if (query_dag_) {
    auto candidate = query_dag_->getQueryHint(aggregate);
    if (candidate) {
      query_hint = *candidate;
    }
  }
  auto join_info = QueryPlanDagExtractor::extractJoinInfo(
      aggregate, std::nullopt, getLeftDeepJoinTreesInfo(), executor_);
  return {RelAlgExecutionUnit{input_descs,
                              input_col_descs,
                              {},
                              {},
                              {},
                              groupby_exprs,
                              target_exprs,
                              target_exprs_type_infos,
                              nullptr,
                              sort_info,
                              0,
                              query_hint,
                              aggregate->getQueryPlanDagHash(),
                              join_info.hash_table_plan_dag,
                              join_info.table_id_to_node_map,
                              false,
                              std::nullopt,
                              query_state_},
          aggregate,
          g_default_max_groups_buffer_entry_guess,
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
  std::optional<unsigned> left_deep_tree_id;
  if (left_deep_join) {
    left_deep_tree_id = left_deep_join->getId();
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

  RelAlgTranslator translator(cat_,
                              query_state_,
                              executor_,
                              input_to_nest_level,
                              join_types,
                              now_,
                              eo.just_explain);
  const auto target_exprs_owned =
      translate_scalar_sources(project, translator, eo.executor_type);

  target_exprs_owned_.insert(
      target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());
  const auto target_exprs = get_raw_pointers(target_exprs_owned);
  auto query_hint = RegisteredQueryHint::defaults();
  if (query_dag_) {
    auto candidate = query_dag_->getQueryHint(project);
    if (candidate) {
      query_hint = *candidate;
    }
  }
  const RelAlgExecutionUnit exe_unit = {input_descs,
                                        input_col_descs,
                                        {},
                                        {},
                                        left_deep_join_quals,
                                        {nullptr},
                                        target_exprs,
                                        {},
                                        nullptr,
                                        sort_info,
                                        0,
                                        query_hint,
                                        project->getQueryPlanDagHash(),
                                        {},
                                        {},
                                        false,
                                        std::nullopt,
                                        query_state_};
  auto query_rewriter = std::make_unique<QueryRewriter>(query_infos, executor_);
  auto rewritten_exe_unit = query_rewriter->rewrite(exe_unit);
  const auto targets_meta = get_targets_meta(project, rewritten_exe_unit.target_exprs);
  project->setOutputMetainfo(targets_meta);
  auto& left_deep_trees_info = getLeftDeepJoinTreesInfo();
  if (left_deep_tree_id && left_deep_tree_id.has_value()) {
    left_deep_trees_info.emplace(left_deep_tree_id.value(),
                                 rewritten_exe_unit.join_quals);
  }
  if (has_valid_query_plan_dag(project)) {
    auto join_info = QueryPlanDagExtractor::extractJoinInfo(
        project, left_deep_tree_id, left_deep_trees_info, executor_);
    rewritten_exe_unit.hash_table_build_plan_dag = join_info.hash_table_plan_dag;
    rewritten_exe_unit.table_id_to_node_map = join_info.table_id_to_node_map;
  }
  return {rewritten_exe_unit,
          project,
          g_default_max_groups_buffer_entry_guess,
          std::move(query_rewriter),
          input_permutation,
          left_deep_join_input_sizes};
}

namespace {

std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_for_union(
    RelAlgNode const* input_node) {
  std::vector<TargetMetaInfo> const& tmis = input_node->getOutputMetainfo();
  VLOG(3) << "input_node->getOutputMetainfo()=" << shared::printContainer(tmis);
  const int negative_node_id = -input_node->getId();
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs;
  target_exprs.reserve(tmis.size());
  for (size_t i = 0; i < tmis.size(); ++i) {
    target_exprs.push_back(std::make_shared<Analyzer::ColumnVar>(
        tmis[i].get_type_info(), negative_node_id, i, 0));
  }
  return target_exprs;
}

}  // namespace

RelAlgExecutor::WorkUnit RelAlgExecutor::createUnionWorkUnit(
    const RelLogicalUnion* logical_union,
    const SortInfo& sort_info,
    const ExecutionOptions& eo) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  // Map ra input ptr to index (0, 1).
  auto input_to_nest_level = get_input_nest_levels(logical_union, {});
  std::tie(input_descs, input_col_descs, std::ignore) =
      get_input_desc(logical_union, input_to_nest_level, {}, cat_);
  const auto query_infos = get_table_infos(input_descs, executor_);
  auto const max_num_tuples =
      std::accumulate(query_infos.cbegin(),
                      query_infos.cend(),
                      size_t(0),
                      [](auto max, auto const& query_info) {
                        return std::max(max, query_info.info.getNumTuples());
                      });

  VLOG(3) << "input_to_nest_level.size()=" << input_to_nest_level.size() << " Pairs are:";
  for (auto& pair : input_to_nest_level) {
    VLOG(3) << "  (" << pair.first->toString(RelRexToStringConfig::defaults()) << ", "
            << pair.second << ')';
  }

  // For UNION queries, we need to keep the target_exprs from both subqueries since they
  // may differ on StringDictionaries.
  std::vector<Analyzer::Expr*> target_exprs_pair[2];
  for (unsigned i = 0; i < 2; ++i) {
    auto input_exprs_owned = target_exprs_for_union(logical_union->getInput(i));
    CHECK(!input_exprs_owned.empty())
        << "No metainfo found for input node(" << i << ") "
        << logical_union->getInput(i)->toString(RelRexToStringConfig::defaults());
    VLOG(3) << "i(" << i << ") input_exprs_owned.size()=" << input_exprs_owned.size();
    for (auto& input_expr : input_exprs_owned) {
      VLOG(3) << "  " << input_expr->toString();
    }
    target_exprs_pair[i] = get_raw_pointers(input_exprs_owned);
    shared::append_move(target_exprs_owned_, std::move(input_exprs_owned));
  }

  VLOG(3) << "input_descs=" << shared::printContainer(input_descs)
          << " input_col_descs=" << shared::printContainer(input_col_descs)
          << " target_exprs.size()=" << target_exprs_pair[0].size()
          << " max_num_tuples=" << max_num_tuples;

  const RelAlgExecutionUnit exe_unit = {input_descs,
                                        input_col_descs,
                                        {},  // quals_cf.simple_quals,
                                        {},  // rewrite_quals(quals_cf.quals),
                                        {},
                                        {nullptr},
                                        target_exprs_pair[0],
                                        {},
                                        nullptr,
                                        sort_info,
                                        max_num_tuples,
                                        RegisteredQueryHint::defaults(),
                                        EMPTY_HASHED_PLAN_DAG_KEY,
                                        {},
                                        {},
                                        false,
                                        logical_union->isAll(),
                                        query_state_,
                                        target_exprs_pair[1]};
  auto query_rewriter = std::make_unique<QueryRewriter>(query_infos, executor_);
  const auto rewritten_exe_unit = query_rewriter->rewrite(exe_unit);

  RelAlgNode const* input0 = logical_union->getInput(0);
  if (auto const* node = dynamic_cast<const RelCompound*>(input0)) {
    logical_union->setOutputMetainfo(
        get_targets_meta(node, rewritten_exe_unit.target_exprs));
  } else if (auto const* node = dynamic_cast<const RelProject*>(input0)) {
    logical_union->setOutputMetainfo(
        get_targets_meta(node, rewritten_exe_unit.target_exprs));
  } else if (auto const* node = dynamic_cast<const RelLogicalUnion*>(input0)) {
    logical_union->setOutputMetainfo(
        get_targets_meta(node, rewritten_exe_unit.target_exprs));
  } else if (auto const* node = dynamic_cast<const RelAggregate*>(input0)) {
    logical_union->setOutputMetainfo(
        get_targets_meta(node, rewritten_exe_unit.target_exprs));
  } else if (auto const* node = dynamic_cast<const RelScan*>(input0)) {
    logical_union->setOutputMetainfo(
        get_targets_meta(node, rewritten_exe_unit.target_exprs));
  } else if (auto const* node = dynamic_cast<const RelFilter*>(input0)) {
    logical_union->setOutputMetainfo(
        get_targets_meta(node, rewritten_exe_unit.target_exprs));
  } else if (dynamic_cast<const RelSort*>(input0)) {
    throw QueryNotSupported("LIMIT and OFFSET are not currently supported with UNION.");
  } else {
    throw QueryNotSupported("Unsupported input type: " +
                            input0->toString(RelRexToStringConfig::defaults()));
  }
  VLOG(3) << "logical_union->getOutputMetainfo()="
          << shared::printContainer(logical_union->getOutputMetainfo())
          << " rewritten_exe_unit.input_col_descs.front()->getScanDesc().getTableId()="
          << rewritten_exe_unit.input_col_descs.front()->getScanDesc().getTableId();

  return {rewritten_exe_unit,
          logical_union,
          g_default_max_groups_buffer_entry_guess,
          std::move(query_rewriter)};
}

RelAlgExecutor::TableFunctionWorkUnit RelAlgExecutor::createTableFunctionWorkUnit(
    const RelTableFunction* rel_table_func,
    const bool just_explain,
    const bool is_gpu) {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  auto input_to_nest_level = get_input_nest_levels(rel_table_func, {});
  std::tie(input_descs, input_col_descs, std::ignore) =
      get_input_desc(rel_table_func, input_to_nest_level, {}, cat_);
  const auto query_infos = get_table_infos(input_descs, executor_);
  RelAlgTranslator translator(
      cat_, query_state_, executor_, input_to_nest_level, {}, now_, just_explain);
  auto input_exprs_owned = translate_scalar_sources(
      rel_table_func, translator, ::ExecutorType::TableFunctions);
  target_exprs_owned_.insert(
      target_exprs_owned_.end(), input_exprs_owned.begin(), input_exprs_owned.end());

  const auto table_function_impl_and_type_infos = [=]() {
    if (is_gpu) {
      try {
        return bind_table_function(
            rel_table_func->getFunctionName(), input_exprs_owned, is_gpu);
      } catch (ExtensionFunctionBindingError& e) {
        LOG(WARNING) << "createTableFunctionWorkUnit[GPU]: " << e.what()
                     << " Redirecting " << rel_table_func->getFunctionName()
                     << " step to run on CPU.";
        throw QueryMustRunOnCpu();
      }
    } else {
      try {
        return bind_table_function(
            rel_table_func->getFunctionName(), input_exprs_owned, is_gpu);
      } catch (ExtensionFunctionBindingError& e) {
        LOG(WARNING) << "createTableFunctionWorkUnit[CPU]: " << e.what();
        throw;
      }
    }
  }();
  const auto& table_function_impl = std::get<0>(table_function_impl_and_type_infos);
  const auto& table_function_type_infos = std::get<1>(table_function_impl_and_type_infos);
  size_t output_row_sizing_param = 0;
  if (table_function_impl
          .hasUserSpecifiedOutputSizeParameter()) {  // constant and row multiplier
    const auto parameter_index =
        table_function_impl.getOutputRowSizeParameter(table_function_type_infos);
    CHECK_GT(parameter_index, size_t(0));
    if (rel_table_func->countRexLiteralArgs() == table_function_impl.countScalarArgs()) {
      const auto parameter_expr =
          rel_table_func->getTableFuncInputAt(parameter_index - 1);
      const auto parameter_expr_literal = dynamic_cast<const RexLiteral*>(parameter_expr);
      if (!parameter_expr_literal) {
        throw std::runtime_error(
            "Provided output buffer sizing parameter is not a literal. Only literal "
            "values are supported with output buffer sizing configured table "
            "functions.");
      }
      int64_t literal_val = parameter_expr_literal->getVal<int64_t>();
      if (literal_val < 0) {
        throw std::runtime_error("Provided output sizing parameter " +
                                 std::to_string(literal_val) +
                                 " must be positive integer.");
      }
      output_row_sizing_param = static_cast<size_t>(literal_val);
    } else {
      // RowMultiplier not specified in the SQL query. Set it to 1
      output_row_sizing_param = 1;  // default value for RowMultiplier
      static Datum d = {DEFAULT_ROW_MULTIPLIER_VALUE};
      static Analyzer::ExpressionPtr DEFAULT_ROW_MULTIPLIER_EXPR =
          makeExpr<Analyzer::Constant>(kINT, false, d);
      // Push the constant 1 to input_exprs
      input_exprs_owned.insert(input_exprs_owned.begin() + parameter_index - 1,
                               DEFAULT_ROW_MULTIPLIER_EXPR);
    }
  } else if (table_function_impl.hasNonUserSpecifiedOutputSize()) {
    output_row_sizing_param = table_function_impl.getOutputRowSizeParameter();
  } else {
    UNREACHABLE();
  }

  std::vector<Analyzer::ColumnVar*> input_col_exprs;
  size_t input_index = 0;
  size_t arg_index = 0;
  const auto table_func_args = table_function_impl.getInputArgs();
  CHECK_EQ(table_func_args.size(), table_function_type_infos.size());
  for (const auto& ti : table_function_type_infos) {
    if (ti.is_column_list()) {
      for (int i = 0; i < ti.get_dimension(); i++) {
        auto& input_expr = input_exprs_owned[input_index];
        auto col_var = dynamic_cast<Analyzer::ColumnVar*>(input_expr.get());
        CHECK(col_var);

        // avoid setting type info to ti here since ti doesn't have all the
        // properties correctly set
        auto type_info = input_expr->get_type_info();
        if (ti.is_column_array()) {
          type_info.set_compression(kENCODING_ARRAY);
          type_info.set_subtype(type_info.get_subtype());  // set type to be subtype
        } else {
          type_info.set_subtype(type_info.get_type());  // set type to be subtype
        }
        type_info.set_type(ti.get_type());  // set type to column list
        type_info.set_dimension(ti.get_dimension());
        input_expr->set_type_info(type_info);

        input_col_exprs.push_back(col_var);
        input_index++;
      }
    } else if (ti.is_column()) {
      auto& input_expr = input_exprs_owned[input_index];
      auto col_var = dynamic_cast<Analyzer::ColumnVar*>(input_expr.get());
      CHECK(col_var);
      // same here! avoid setting type info to ti since it doesn't have all the
      // properties correctly set
      auto type_info = input_expr->get_type_info();
      if (ti.is_column_array()) {
        type_info.set_compression(kENCODING_ARRAY);
        type_info.set_subtype(type_info.get_subtype());  // set type to be subtype
      } else {
        type_info.set_subtype(type_info.get_type());  // set type to be subtype
      }
      type_info.set_type(ti.get_type());  // set type to column
      input_expr->set_type_info(type_info);
      input_col_exprs.push_back(col_var);
      input_index++;
    } else {
      auto input_expr = input_exprs_owned[input_index];
      auto ext_func_arg_ti = ext_arg_type_to_type_info(table_func_args[arg_index]);
      if (ext_func_arg_ti != input_expr->get_type_info()) {
        input_exprs_owned[input_index] = input_expr->add_cast(ext_func_arg_ti);
      }
      input_index++;
    }
    arg_index++;
  }
  CHECK_EQ(input_col_exprs.size(), rel_table_func->getColInputsSize());
  std::vector<Analyzer::Expr*> table_func_outputs;
  constexpr int32_t transient_pos{-1};
  for (size_t i = 0; i < table_function_impl.getOutputsSize(); i++) {
    auto ti = table_function_impl.getOutputSQLType(i);
    if (ti.is_dict_encoded_string() || ti.is_text_encoding_dict_array()) {
      auto p = table_function_impl.getInputID(i);

      int32_t input_pos = p.first;
      if (input_pos == transient_pos) {
        ti.set_comp_param(TRANSIENT_DICT_ID);
      } else {
        // Iterate over the list of arguments to compute the offset. Use this offset to
        // get the corresponding input
        int32_t offset = 0;
        for (int j = 0; j < input_pos; j++) {
          const auto ti = table_function_type_infos[j];
          offset += ti.is_column_list() ? ti.get_dimension() : 1;
        }
        input_pos = offset + p.second;

        CHECK_LT(input_pos, input_exprs_owned.size());
        int32_t comp_param =
            input_exprs_owned[input_pos]->get_type_info().get_comp_param();
        ti.set_comp_param(comp_param);
      }
    }
    target_exprs_owned_.push_back(std::make_shared<Analyzer::ColumnVar>(ti, 0, i, -1));
    table_func_outputs.push_back(target_exprs_owned_.back().get());
  }
  auto input_exprs = get_raw_pointers(input_exprs_owned);
  const TableFunctionExecutionUnit exe_unit = {
      input_descs,
      input_col_descs,
      input_exprs,              // table function inputs
      input_col_exprs,          // table function column inputs (duplicates w/ above)
      table_func_outputs,       // table function projected exprs
      output_row_sizing_param,  // output buffer sizing param
      table_function_impl};
  const auto targets_meta = get_targets_meta(rel_table_func, exe_unit.target_exprs);
  rel_table_func->setOutputMetainfo(targets_meta);
  return {exe_unit, rel_table_func};
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
        scalar_sources_owned.push_back(translator.translate(input_it->get()));
      }
      const auto source_metadata =
          get_targets_meta(scan_source, get_raw_pointers(scalar_sources_owned));
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
  RelAlgTranslator translator(cat_,
                              query_state_,
                              executor_,
                              input_to_nest_level,
                              {join_type},
                              now_,
                              just_explain);
  std::tie(in_metainfo, target_exprs_owned) =
      get_inputs_meta(filter, translator, used_inputs_owned, input_to_nest_level);
  const auto filter_expr = translator.translate(filter->getCondition());
  const auto query_infos = get_table_infos(input_descs, executor_);

  const auto qual = fold_expr(filter_expr.get());
  target_exprs_owned_.insert(
      target_exprs_owned_.end(), target_exprs_owned.begin(), target_exprs_owned.end());

  const auto target_exprs = get_raw_pointers(target_exprs_owned);
  filter->setOutputMetainfo(in_metainfo);
  const auto rewritten_qual = rewrite_expr(qual.get());
  auto query_hint = RegisteredQueryHint::defaults();
  if (query_dag_) {
    auto candidate = query_dag_->getQueryHint(filter);
    if (candidate) {
      query_hint = *candidate;
    }
  }
  auto join_info = QueryPlanDagExtractor::extractJoinInfo(
      filter, std::nullopt, getLeftDeepJoinTreesInfo(), executor_);
  return {{input_descs,
           input_col_descs,
           {},
           {rewritten_qual ? rewritten_qual : qual},
           {},
           {nullptr},
           target_exprs,
           {},
           nullptr,
           sort_info,
           0,
           query_hint,
           filter->getQueryPlanDagHash(),
           join_info.hash_table_plan_dag,
           join_info.table_id_to_node_map},
          filter,
          g_default_max_groups_buffer_entry_guess,
          nullptr};
}

SpeculativeTopNBlacklist RelAlgExecutor::speculative_topn_blacklist_;

void RelAlgExecutor::initializeParallelismHints() {
  if (auto foreign_storage_mgr =
          cat_.getDataMgr().getPersistentStorageMgr()->getForeignStorageMgr()) {
    // Parallelism hints need to be reset to empty so that we don't accidentally re-use
    // them.  This can cause attempts to fetch strings that do not shard to the correct
    // node in distributed mode.
    foreign_storage_mgr->setParallelismHints({});
  }
}

void RelAlgExecutor::setupCaching(const RelAlgNode* ra) {
  CHECK(executor_);
  const auto phys_inputs = get_physical_inputs(cat_, ra);
  const auto phys_table_ids = get_physical_table_inputs(ra);
  executor_->setCatalog(&cat_);
  executor_->setupCaching(phys_inputs, phys_table_ids);
}

void RelAlgExecutor::prepareForeignTables() {
  const auto& ra = query_dag_->getRootNode();
  prepare_foreign_table_for_execution(ra, cat_);
}

std::unordered_set<int> RelAlgExecutor::getPhysicalTableIds() const {
  return get_physical_table_inputs(&getRootRelAlgNode());
}

void RelAlgExecutor::prepareForSystemTableExecution(const CompilationOptions& co) const {
  prepare_for_system_table_execution(getRootRelAlgNode(), cat_, co);
}
