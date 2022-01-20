/*
 * Copyright 2021 OmniSci, Inc.
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

struct ExtractedPlanDag {
  // a root node of the query plan
  const RelAlgNode* root_node;
  // extracted DAG starting from the root_node
  const QueryPlan extracted_dag;
  // join qual's info collected while converting calcite AST to logical query plan tree
  const std::shared_ptr<TranslatedJoinInfo> translated_join_info;
  // join condition info collected during converting calcite AST to logical query plan
  // tree
  const JoinQualsPerNestingLevel* per_nesting_level_join_quals_info;
  // a map btw. join qual to an access path (as a query plan DAG) to corresponding
  // hashtable
  const HashTableBuildDagMap hash_table_plan_dag;
  // a map btw. inner join col's input table id to corresponding rel node ptr
  const TableIdToNodeMap table_id_to_node_map;
  // show whether this query plan is not available to extract a DAG
  bool contain_not_supported_rel_node;
};

// set the bool flag be true when the InnerOuter qual is for loop join
struct InnerOuterOrLoopQual {
  std::pair<const Analyzer::Expr*, const Analyzer::Expr*> inner_outer;
  bool loop_join_qual{false};
};

class QueryPlanDagExtractor {
 public:
  // TODO: remove executor from constructor
  QueryPlanDagExtractor(
      QueryPlanDagCache& global_dag,
      SchemaProviderPtr schema_provider,
      std::unordered_map<unsigned, JoinQualsPerNestingLevel>& left_deep_tree_infos,
      const TemporaryTables& temporary_tables,
      Executor* executor)
      : global_dag_(global_dag)
      , schema_provider_(schema_provider)
      , contain_not_supported_rel_node_(false)
      , left_deep_tree_infos_(left_deep_tree_infos)
      , temporary_tables_(temporary_tables) {
    translated_join_info_ = std::make_shared<TranslatedJoinInfo>();
  }

  // a function that try to extract query plan DAG
  static ExtractedPlanDag extractQueryPlanDag(
      const RelAlgNode* node,
      SchemaProviderPtr schema_provider,
      std::optional<unsigned> left_deep_tree_id,
      std::unordered_map<unsigned, JoinQualsPerNestingLevel>& left_deep_tree_infos,
      const TemporaryTables& temporary_tables,
      Executor* executor,
      const RelAlgTranslator& rel_alg_translator);

  HashTableBuildDagMap& getHashTableBuildDag() { return hash_table_query_plan_dag_; }

  std::shared_ptr<TranslatedJoinInfo> getTranslatedJoinInfo() {
    return translated_join_info_;
  }

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

  std::string getExtractedQueryPlanDagStr();

  std::vector<InnerOuterOrLoopQual> normalizeColumnsPair(
      const Analyzer::BinOper* condition);

  bool isEmptyQueryPlanDag(const std::string& dag) { return dag.compare("N/A") == 0; }

  TableIdToNodeMap& getTableIdToNodeMap() { return table_id_to_node_map_; }

  void addTableIdToNodeLink(const int table_id, const RelAlgNode* node) {
    auto it = table_id_to_node_map_.find(table_id);
    if (it == table_id_to_node_map_.end()) {
      table_id_to_node_map_.emplace(table_id, node);
    }
  }

  void clearInternaStatus() {
    contain_not_supported_rel_node_ = true;
    extracted_dag_.clear();
    table_id_to_node_map_.clear();
  }

 private:
  void visit(const RelAlgNode*, const RelAlgNode*);
  Analyzer::ColumnVar const* getColVar(const Analyzer::Expr* col_info);
  void handleLeftDeepJoinTree(const RelAlgNode*, const RelLeftDeepInnerJoin*);
  void handleTranslatedJoin(const RelAlgNode*, const RelTranslatedJoin*);
  bool validateNodeId(const RelAlgNode* node, std::optional<RelNodeId> retrieved_node_id);
  bool registerNodeToDagCache(const RelAlgNode* parent_node,
                              const RelAlgNode* child_node,
                              std::optional<RelNodeId> retrieved_node_id);
  static ExtractedPlanDag extractQueryPlanDagImpl(
      const RelAlgNode* node,
      SchemaProviderPtr schema_provider,
      std::optional<unsigned> left_deep_tree_id,
      std::unordered_map<unsigned, JoinQualsPerNestingLevel>& left_deep_tree_infos,
      const TemporaryTables& temporary_tables,
      Executor* executor);

  QueryPlanDagCache& global_dag_;
  SchemaProviderPtr schema_provider_;
  bool contain_not_supported_rel_node_;
  std::unordered_map<unsigned, JoinQualsPerNestingLevel>& left_deep_tree_infos_;
  const TemporaryTables& temporary_tables_;
  std::shared_ptr<TranslatedJoinInfo> translated_join_info_;
  HashTableBuildDagMap hash_table_query_plan_dag_;
  TableIdToNodeMap table_id_to_node_map_;
  std::vector<size_t> extracted_dag_;
};
