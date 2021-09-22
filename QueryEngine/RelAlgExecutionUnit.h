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

/**
 * @file    RelAlgExecutionUnit.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Execution unit for relational algebra. It's a low-level description
 *          of any relational algebra operation in a format understood by our VM.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_RELALGEXECUTIONUNIT_H
#define QUERYENGINE_RELALGEXECUTIONUNIT_H

#include "Descriptors/InputDescriptors.h"
#include "QueryHint.h"
#include "RelAlgDagBuilder.h"
#include "Shared/sqldefs.h"
#include "Shared/toString.h"
#include "TableFunctions/TableFunctionOutputBufferSizeType.h"
#include "TableFunctions/TableFunctionsFactory.h"
#include "ThriftHandler/QueryState.h"

#include <boost/graph/adjacency_list.hpp>

#include <list>
#include <memory>
#include <optional>
#include <vector>

using AdjacentList = boost::adjacency_list<boost::setS, boost::vecS, boost::directedS>;
// node ID used when extracting query plan DAG
// note this ID is different from RelNode's id since query plan DAG extractor assigns an
// unique node ID only to a rel node which is included in extracted DAG (if we cannot
// extract a DAG from the query plan DAG extractor skips to assign unique IDs to rel nodes
// in that query plan
using RelNodeId = size_t;
// toString content of each extracted rel node
using RelNodeExplained = std::string;
// hash value of explained rel node
using RelNodeExplainedHash = size_t;
// a string representation of a query plan that is collected by visiting query plan DAG
// starting from root to leaf and concatenate each rel node's id
// where two adjacent rel nodes in a QueryPlan are connected via '|' delimiter
// i.e., 1|2|3|4|
using QueryPlan = std::string;
// join column's column id info
using JoinColumnsInfo = std::string;
// hashed value of QueryPlanNodeIds
using QueryPlanHash = size_t;
// a map btw. a join column and its access path, i.e., a query plan DAG to project B.b
// here this join column is used to build a hashtable
using HashTableBuildDag = std::pair<JoinColumnsInfo, QueryPlan>;
// A map btw. join qual's column info and its corresponding hashtable access path as query
// plan DAG i.e., A.a = B.b and build hashtable on B.b? <(A.a = B.b) --> query plan DAG of
// projecting B.b> here, this two-level mapping (join qual -> inner join col -> hashtable
// access plan DAG) is required since we have to extract query plan before deciding which
// join col becomes inner since rel alg related metadata is required to extract query
// plan, and the actual decision happens at the time of building hashtable
using HashTableBuildDagMap = std::unordered_map<JoinColumnsInfo, HashTableBuildDag>;
// A map btw. join column's input table id to its corresponding rel node
// for each hash join operation, we can determine whether its input source
// has inconsistency in its source data, e.g., row ordering
// by seeing a type of input node, e.g., RelSort
// note that disabling DAG extraction when we find sort node from join's input
// is too restrict when a query becomes complex (and so have multiple joins)
// since it eliminates a change of data recycling
using TableIdToNodeMap = std::unordered_map<int, const RelAlgNode*>;

enum JoinColumnSide {
  kInner,
  kOuter,
  kQual,   // INNER + OUTER
  kDirect  // set target directly (i.e., put Analyzer::Expr* instead of
           // Analyzer::BinOper*)
};
constexpr char const* EMPTY_QUERY_PLAN = "";
constexpr QueryPlanHash EMPTY_HASHED_PLAN_DAG_KEY = 0;

enum class SortAlgorithm { Default, SpeculativeTopN, StreamingTopN };

namespace Analyzer {
class Expr;
class ColumnVar;
class Estimator;
struct OrderEntry;

}  // namespace Analyzer

struct SortInfo {
  const std::list<Analyzer::OrderEntry> order_entries;
  const SortAlgorithm algorithm;
  const size_t limit;
  const size_t offset;
};

struct JoinCondition {
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  JoinType type;
};

using JoinQualsPerNestingLevel = std::vector<JoinCondition>;

struct RelAlgExecutionUnit {
  std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  const JoinQualsPerNestingLevel join_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  std::vector<Analyzer::Expr*> target_exprs;
  const std::shared_ptr<Analyzer::Estimator> estimator;
  const SortInfo sort_info;
  size_t scan_limit;
  RegisteredQueryHint query_hint;
  QueryPlan query_plan_dag{EMPTY_QUERY_PLAN};
  HashTableBuildDagMap hash_table_build_plan_dag{};
  TableIdToNodeMap table_id_to_node_map{};
  bool use_bump_allocator{false};
  // empty if not a UNION, true if UNION ALL, false if regular UNION
  const std::optional<bool> union_all;
  std::shared_ptr<const query_state::QueryState> query_state;
};

std::ostream& operator<<(std::ostream& os, const RelAlgExecutionUnit& ra_exe_unit);
std::string ra_exec_unit_desc_for_caching(const RelAlgExecutionUnit& ra_exe_unit);

struct TableFunctionExecutionUnit {
  const std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::vector<Analyzer::Expr*> input_exprs;
  std::vector<Analyzer::ColumnVar*> table_func_inputs;
  std::vector<Analyzer::Expr*> target_exprs;
  const size_t output_buffer_size_param;
  const table_functions::TableFunction table_func;

 public:
  std::string toString() const {
    return typeName(this) + "(" + "input_exprs=" + ::toString(input_exprs) +
           ", table_func_inputs=" + ::toString(table_func_inputs) +
           ", target_exprs=" + ::toString(target_exprs) +
           ", output_buffer_size_param=" + ::toString(output_buffer_size_param) +
           ", table_func=" + ::toString(table_func) + ")";
  }
};

class ResultSet;
using ResultSetPtr = std::shared_ptr<ResultSet>;

#endif  // QUERYENGINE_RELALGEXECUTIONUNIT_H
