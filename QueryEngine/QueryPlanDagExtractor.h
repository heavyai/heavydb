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

#pragma once

#include "QueryPlanDagCache.h"
#include "RelAlgExecutionUnit.h"
#include "RelAlgExecutor.h"
#include "RelAlgTranslator.h"

#include <unordered_map>
#include <vector>

// a vector of explained join info
using TranslatedJoinInfo = std::vector<std::shared_ptr<RelTranslatedJoin>>;

struct ExtractedQueryPlanDag {
  // extracted DAG starting from the root_node
  const QueryPlanDAG extracted_dag;
  // show whether this query plan is not available to extract a DAG
  bool contain_not_supported_rel_node;
};

struct ExtractedJoinInfo {
  // a map btw. join qual to an access path (as a query plan DAG) to corresponding
  // hashtable
  const HashTableBuildDagMap hash_table_plan_dag;
  // a map btw. inner join col's input table id to corresponding rel node ptr
  const TableIdToNodeMap table_id_to_node_map;
  // show whether this query plan is not available to extract a DAG
};

// set the bool flag be true when the InnerOuter qual is for loop join
struct InnerOuterOrLoopQual {
  std::pair<const Analyzer::Expr*, const Analyzer::Expr*> inner_outer;
  bool loop_join_qual{false};
};

class QueryPlanDagExtractor {
 public:
  QueryPlanDagExtractor(
      QueryPlanDagCache& global_dag,
      std::unordered_map<unsigned, JoinQualsPerNestingLevel> left_deep_tree_infos,
      Executor* executor,
      bool analyze_join_ops)
      : global_dag_(global_dag)
      , contain_not_supported_rel_node_(false)
      , left_deep_tree_infos_(left_deep_tree_infos)
      , executor_(executor)
      , analyze_join_ops_(analyze_join_ops) {
    if (analyze_join_ops_) {
      translated_join_info_ = std::make_shared<TranslatedJoinInfo>();
    }
  }

  static ExtractedJoinInfo extractJoinInfo(
      const RelAlgNode* top_node,
      std::optional<unsigned> left_deep_tree_id,
      std::unordered_map<unsigned, JoinQualsPerNestingLevel> left_deep_tree_infos,
      Executor* executor);

  static ExtractedQueryPlanDag extractQueryPlanDag(const RelAlgNode* top_node,
                                                   Executor* executor);

  HashTableBuildDagMap getHashTableBuildDag() { return hash_table_query_plan_dag_; }

  const JoinQualsPerNestingLevel* getPerNestingJoinQualInfo(
      std::optional<unsigned> left_deep_join_tree_id) {
    if (left_deep_tree_infos_.empty() || !left_deep_join_tree_id) {
      return nullptr;
    }
    CHECK(left_deep_join_tree_id.has_value());
    auto it = left_deep_tree_infos_.find(left_deep_join_tree_id.value());
    if (it == left_deep_tree_infos_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  bool isDagExtractionAvailable() { return contain_not_supported_rel_node_; }

  std::string getExtractedQueryPlanDagStr(size_t start_pos = 0);

  std::vector<InnerOuterOrLoopQual> normalizeColumnsPair(
      const Analyzer::BinOper* condition);

  bool isEmptyQueryPlanDag(const std::string& dag) { return dag.compare("N/A") == 0; }

  TableIdToNodeMap getTableIdToNodeMap() { return table_id_to_node_map_; }

  void addTableIdToNodeLink(const shared::TableKey& table_id, const RelAlgNode* node) {
    auto it = table_id_to_node_map_.find(table_id);
    if (it == table_id_to_node_map_.end()) {
      table_id_to_node_map_.emplace(table_id, node);
    }
  }

  void clearInternalStatus() {
    contain_not_supported_rel_node_ = true;
    extracted_dag_.clear();
    table_id_to_node_map_.clear();
  }

  static size_t applyLimitClauseToCacheKey(size_t cache_key, SortInfo const& sort_info);

 private:
  void visit(const RelAlgNode*, const RelAlgNode*);
  std::vector<Analyzer::ColumnVar const*> getColVar(const Analyzer::Expr* col_info);
  void register_and_visit(const RelAlgNode* parent_node, const RelAlgNode* child_node);
  void handleLeftDeepJoinTree(const RelAlgNode*, const RelLeftDeepInnerJoin*);
  void handleTranslatedJoin(const RelAlgNode*, const RelTranslatedJoin*);
  bool validateNodeId(const RelAlgNode* node, std::optional<RelNodeId> retrieved_node_id);
  bool registerNodeToDagCache(const RelAlgNode* parent_node,
                              const RelAlgNode* child_node,
                              std::optional<RelNodeId> retrieved_node_id);
  static void extractQueryPlanDagImpl(const RelAlgNode* top_npde,
                                      QueryPlanDagExtractor& dag_extractor);

  QueryPlanDagCache& global_dag_;
  bool contain_not_supported_rel_node_;
  std::unordered_map<unsigned, JoinQualsPerNestingLevel> left_deep_tree_infos_;
  Executor* executor_;
  bool analyze_join_ops_;
  std::shared_ptr<TranslatedJoinInfo> translated_join_info_;
  HashTableBuildDagMap hash_table_query_plan_dag_;
  TableIdToNodeMap table_id_to_node_map_;
  std::vector<std::string> extracted_dag_;
};
