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

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/labeled_graph.hpp>

#include <iostream>
#include <memory>
#include <vector>

#include "RelAlgDagBuilder.h"
#include "RelAlgExecutionUnit.h"
#include "ScalarExprVisitor.h"

// we manage the uniqueness of node ID by its explained contents that each rel node has
using RelNodeMap = std::unordered_map<RelNodeExplainedHash, RelNodeId>;
// we also maintain labeled graph to manage extracted query plan DAG
// this can be used in a future to support advanced features such as partial resultset
// reuse and compiled kernel reuse by exploiting graph-centric computation like subgraph
// matching and graph isomorphism
using QueryPlanDag = boost::labeled_graph<AdjacentList, RelNodeId, boost::hash_mapS>;

class ColumnVarsVisitor
    : public ScalarExprVisitor<std::vector<const Analyzer::ColumnVar*>> {
 protected:
  std::vector<const Analyzer::ColumnVar*> visitColumnVar(
      const Analyzer::ColumnVar* column) const override {
    return {column};
  }

  std::vector<const Analyzer::ColumnVar*> visitColumnVarTuple(
      const Analyzer::ExpressionTuple* expr_tuple) const override {
    ColumnVarsVisitor visitor;
    std::vector<const Analyzer::ColumnVar*> result;
    for (size_t i = 0; i < expr_tuple->getTuple().size(); ++i) {
      const auto col_vars = visitor.visit(expr_tuple->getTuple()[i].get());
      for (const auto col_var : col_vars) {
        result.push_back(col_var);
      }
    }
    return result;
  }

  std::vector<const Analyzer::ColumnVar*> aggregateResult(
      const std::vector<const Analyzer::ColumnVar*>& aggregate,
      const std::vector<const Analyzer::ColumnVar*>& next_result) const override {
    auto result = aggregate;
    for (const auto col_var : next_result) {
      result.push_back(col_var);
    }
    return result;
  }
};

// This is one of main data structure for data recycling which manages a query plan shape
// as a DAG representation
// A query plan DAG is a sequence of unique node ID, and it means that we can assign the
// same node ID to a node iff we already saw that node in a different query plan that we
// extracted to retrieve a query plan DAG we visit each rel node of an input query plan
// starting from the root to the bottom (left-to-right child visiting), and check whether
// it is valid for DAG extraction and its usage (we do not allow dag extraction if a query
// plan has not supported rel node for data recycling such as logical value) and if that
// visited node is valid then we check its uniqueness against DAG cache and assign the
// unique ID once it is unique one (otherwise we reuse node id) after visiting a query
// plan we have a sequence of node IDs and return it as an extracted query plan DAG
class QueryPlanDagCache {
 public:
  QueryPlanDagCache(size_t max_node_cache_size = 1e9)
      : max_node_map_size_(max_node_cache_size) {}

  QueryPlanDagCache(QueryPlanDagCache&& other) = delete;
  QueryPlanDagCache& operator=(QueryPlanDagCache&& other) = delete;
  QueryPlanDagCache(const QueryPlanDagCache&) = delete;
  QueryPlanDagCache& operator=(const QueryPlanDagCache&) = delete;

  std::optional<RelNodeId> addNodeIfAbsent(const RelAlgNode*);

  void connectNodes(const RelNodeId parent_id, const RelNodeId child_id);

  std::vector<const Analyzer::ColumnVar*> collectColVars(const Analyzer::Expr* target);

  size_t getCurrentNodeMapSize() const;

  void setNodeMapMaxSize(const size_t map_size);

  size_t getCurrentNodeMapCardinality() const;

  JoinColumnsInfo getJoinColumnsInfoString(const Analyzer::Expr* join_expr,
                                           JoinColumnSide target_side,
                                           bool extract_only_col_id);

  JoinColumnsInfo translateColVarsToInfoString(
      std::vector<const Analyzer::ColumnVar*>& col_vars,
      bool col_id_only) const;

  void clearQueryPlanCache();

  void printDag();

 private:
  // a map btw. rel node and its unique node id
  RelNodeMap node_map_;
  // a graph structure that represents relationships among extracted query plan DAGs
  QueryPlanDag cached_query_plan_dag_;
  // a limitation of the maximum size of DAG cache (to prevent unlimited usage of memory
  // for DAG maintanence)
  size_t max_node_map_size_;
  // a lock to protect contentions while accessing internal data structure of DAG cache
  std::mutex cache_lock_;
  ColumnVarsVisitor col_var_visitor_;
};
