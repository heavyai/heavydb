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

#ifndef QUERYENGINE_RELALGEXECUTOR_H
#define QUERYENGINE_RELALGEXECUTOR_H

#include "InputMetadata.h"
#include "Execute.h"
#include "QueryRewrite.h"
#include "RelAlgExecutionDescriptor.h"
#include "SpeculativeTopN.h"
#include "StreamingTopN.h"
#include "../Shared/scope.h"
#include "Distributed/AggregatedResult.h"

#include <ctime>

enum class MergeType { Union, Reduce };

struct FirstStepExecutionResult {
  ExecutionResult result;
  const MergeType merge_type;
  const unsigned node_id;
  bool is_outermost_query;
};

class RelAlgExecutor {
 public:
  RelAlgExecutor(Executor* executor, const Catalog_Namespace::Catalog& cat)
      : executor_(executor), cat_(cat), now_(0), queue_time_ms_(0) {}

  ExecutionResult executeRelAlgQuery(const std::string& query_ra,
                                     const CompilationOptions& co,
                                     const ExecutionOptions& eo,
                                     RenderInfo* render_info);

  FirstStepExecutionResult executeRelAlgQueryFirstStep(const RelAlgNode* ra,
                                                       const CompilationOptions& co,
                                                       const ExecutionOptions& eo,
                                                       RenderInfo* render_info);

  void prepareLeafExecution(const AggregatedColRange& agg_col_range,
                            const StringDictionaryGenerations& string_dictionary_generations,
                            const TableGenerations& table_generations);

  ExecutionResult executeRelAlgSubQuery(const RexSubQuery* subquery,
                                        const CompilationOptions& co,
                                        const ExecutionOptions& eo);

  ExecutionResult executeRelAlgSeq(std::vector<RaExecutionDesc>& ed_list,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   RenderInfo* render_info,
                                   const int64_t queue_time_ms);

  void addLeafResult(const unsigned id, const AggregatedResult& result) {
    const auto it_ok = leaf_results_.emplace(id, result);
    CHECK(it_ok.second);
  }

  void registerSubquery(RexSubQuery* subquery) noexcept { subqueries_.push_back(subquery); }

  const std::vector<RexSubQuery*>& getSubqueries() const noexcept { return subqueries_; };

  AggregatedColRange computeColRangesCache(const RelAlgNode* ra);

  StringDictionaryGenerations computeStringDictionaryGenerations(const RelAlgNode* ra);

  TableGenerations computeTableGenerations(const RelAlgNode* ra);

  Executor* getExecutor() const;

 private:
  ExecutionResult executeRelAlgQueryNoRetry(const std::string& query_ra,
                                            const CompilationOptions& co,
                                            const ExecutionOptions& eo,
                                            RenderInfo* render_info);

  void executeRelAlgStep(const size_t step_idx,
                         std::vector<RaExecutionDesc>&,
                         const CompilationOptions&,
                         const ExecutionOptions&,
                         RenderInfo*,
                         const int64_t queue_time_ms);

  ExecutionResult executeCompound(const RelCompound*,
                                  const CompilationOptions&,
                                  const ExecutionOptions&,
                                  RenderInfo*,
                                  const int64_t queue_time_ms);

  ExecutionResult executeAggregate(const RelAggregate* aggregate,
                                   const CompilationOptions& co,
                                   const ExecutionOptions& eo,
                                   RenderInfo* render_info,
                                   const int64_t queue_time_ms);

  ExecutionResult executeProject(const RelProject*,
                                 const CompilationOptions&,
                                 const ExecutionOptions&,
                                 RenderInfo*,
                                 const int64_t queue_time_ms);

  ExecutionResult executeFilter(const RelFilter*,
                                const CompilationOptions&,
                                const ExecutionOptions&,
                                RenderInfo*,
                                const int64_t queue_time_ms);

  ExecutionResult executeSort(const RelSort*,
                              const CompilationOptions&,
                              const ExecutionOptions&,
                              RenderInfo*,
                              const int64_t queue_time_ms);

  ExecutionResult executeJoin(const RelJoin*,
                              const CompilationOptions&,
                              const ExecutionOptions&,
                              RenderInfo*,
                              const int64_t queue_time_ms);

  ExecutionResult executeLogicalValues(const RelLogicalValues*, const ExecutionOptions&);

  // TODO(alex): just move max_groups_buffer_entry_guess to RelAlgExecutionUnit once
  //             we deprecate the plan-based executor paths and remove WorkUnit
  struct WorkUnit {
    RelAlgExecutionUnit exe_unit;
    const RelAlgNode* body;
    const size_t max_groups_buffer_entry_guess;
    std::unique_ptr<QueryRewriter> query_rewriter;
  };

  WorkUnit createSortInputWorkUnit(const RelSort*, const bool just_explain);

  ExecutionResult executeWorkUnit(const WorkUnit& work_unit,
                                  const std::vector<TargetMetaInfo>& targets_meta,
                                  const bool is_agg,
                                  const CompilationOptions& co,
                                  const ExecutionOptions& eo,
                                  RenderInfo*,
                                  const int64_t queue_time_ms);

  size_t getNDVEstimation(const WorkUnit& work_unit,
                          const bool is_agg,
                          const CompilationOptions& co,
                          const ExecutionOptions& eo);

  ssize_t getFilteredCountAll(const WorkUnit& work_unit,
                              const bool is_agg,
                              const CompilationOptions& co,
                              const ExecutionOptions& eo);

  bool isRowidLookup(const WorkUnit& work_unit);

  ExecutionResult renderWorkUnit(const RelAlgExecutor::WorkUnit& work_unit,
                                 const std::vector<TargetMetaInfo>& targets_meta,
                                 RenderInfo* render_info,
                                 const int32_t error_code,
                                 const int64_t queue_time_ms);

  void executeUnfoldedMultiJoin(const RelAlgNode* user,
                                RaExecutionDesc& exec_desc,
                                const CompilationOptions& co,
                                const ExecutionOptions& eo,
                                const int64_t queue_time_ms);

  ExecutionResult handleRetry(const int32_t error_code_in,
                              const RelAlgExecutor::WorkUnit& work_unit,
                              const std::vector<TargetMetaInfo>& targets_meta,
                              const bool is_agg,
                              const CompilationOptions& co,
                              const ExecutionOptions& eo,
                              const int64_t queue_time_ms);

  static void handlePersistentError(const int32_t error_code);

  static std::string getErrorMessageFromCode(const int32_t error_code);

  WorkUnit createWorkUnit(const RelAlgNode*, const SortInfo&, const bool just_explain);

  WorkUnit createCompoundWorkUnit(const RelCompound*, const SortInfo&, const bool just_explain);

  WorkUnit createAggregateWorkUnit(const RelAggregate*, const SortInfo&, const bool just_explain);

  WorkUnit createProjectWorkUnit(const RelProject*, const SortInfo&, const bool just_explain);

  WorkUnit createFilterWorkUnit(const RelFilter*, const SortInfo&, const bool just_explain);

  WorkUnit createJoinWorkUnit(const RelJoin*, const SortInfo&, const bool just_explain);

  void addTemporaryTable(const int table_id, const ResultPtr& result) {
    auto row_set = boost::get<RowSetPtr>(&result);
    if (row_set) {
      CHECK_LT(size_t(0), (*row_set)->colCount());
    }
    CHECK_LT(table_id, 0);
    const auto it_ok = temporary_tables_.emplace(table_id, result);
    CHECK(it_ok.second);
  }

  void handleNop(const RelAlgNode*);

  JoinQualsPerNestingLevel translateLeftDeepJoinFilter(
      const RelLeftDeepInnerJoin* join,
      const std::vector<InputDescriptor>& input_descs,
      const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
      const bool just_explain);

  // Transform the provided `join_condition` to conjunctive form, find composite
  // key opportunities and finally translate it to an Analyzer expression.
  std::list<std::shared_ptr<Analyzer::Expr>> makeJoinQuals(
      const RexScalar* join_condition,
      const JoinType join_type,
      const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
      const bool just_explain) const;

  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;
  TemporaryTables temporary_tables_;
  time_t now_;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs_owned_;  // TODO(alex): remove
  std::vector<RexSubQuery*> subqueries_;
  std::unordered_map<unsigned, AggregatedResult> leaf_results_;
  int64_t queue_time_ms_;
  static SpeculativeTopNBlacklist speculative_topn_blacklist_;
  static const size_t max_groups_buffer_entry_default_guess{16384};
};

#endif  // QUERYENGINE_RELALGEXECUTOR_H
