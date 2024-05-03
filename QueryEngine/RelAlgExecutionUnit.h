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

/**
 * @file    RelAlgExecutionUnit.h
 * @brief   Execution unit for relational algebra. It's a low-level description
 *          of any relational algebra operation in a format understood by our VM.
 *
 */

#pragma once

#include "Descriptors/InputDescriptors.h"
#include "QueryHint.h"
#include "RelAlgDag.h"
#include "Shared/DbObjectKeys.h"
#include "Shared/sqldefs.h"
#include "Shared/toString.h"
#include "TableFunctions/TableFunctionOutputBufferSizeType.h"
#include "TableFunctions/TableFunctionsFactory.h"
#include "ThriftHandler/QueryState.h"

#include <boost/graph/adjacency_list.hpp>

#include <functional>
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
// hash value of explained rel node
using RelNodeExplainedHash = size_t;
// a string representation of a query plan that is collected by visiting query plan DAG
// starting from root to leaf and concatenate each rel node's id
// where two adjacent rel nodes in a QueryPlanDAG are connected via '|' delimiter
// i.e., 1|2|3|4|
using QueryPlanDAG = std::string;
// hashed value of QueryPlanNodeIds
using QueryPlanHash = size_t;
// hold query plan dag and column info of join columns
// used to detect a correct cached hashtable
struct HashTableBuildDag {
 public:
  HashTableBuildDag(size_t in_inner_cols_info,
                    size_t in_outer_cols_info,
                    QueryPlanHash in_inner_cols_access_path,
                    QueryPlanHash in_outer_cols_access_path,
                    std::unordered_set<size_t>&& inputTableKeys)
      : inner_cols_info(in_inner_cols_info)
      , outer_cols_info(in_outer_cols_info)
      , inner_cols_access_path(in_inner_cols_access_path)
      , outer_cols_access_path(in_outer_cols_access_path)
      , inputTableKeys(std::move(inputTableKeys)) {}
  size_t inner_cols_info;
  size_t outer_cols_info;
  QueryPlanHash inner_cols_access_path;
  QueryPlanHash outer_cols_access_path;
  std::unordered_set<size_t>
      inputTableKeys;  // table keys of input(s), e.g., scan node or subquery's DAG
};
// A map btw. join qual's column info and its corresponding hashtable access path as query
// plan DAG i.e., A.a = B.b and build hashtable on B.b? <(A.a = B.b) --> query plan DAG of
// projecting B.b> here, this two-level mapping (join qual -> inner join col -> hashtable
// access plan DAG) is required since we have to extract query plan before deciding which
// join col becomes inner since rel alg related metadata is required to extract query
// plan, and the actual decision happens at the time of building hashtable
using HashTableBuildDagMap = std::unordered_map<size_t, HashTableBuildDag>;
// A map btw. join column's input table id to its corresponding rel node
// for each hash join operation, we can determine whether its input source
// has inconsistency in its source data, e.g., row ordering
// by seeing a type of input node, e.g., RelSort
// note that disabling DAG extraction when we find sort node from join's input
// is too restrict when a query becomes complex (and so have multiple joins)
// since it eliminates a change of data recycling
using TableIdToNodeMap = std::unordered_map<shared::TableKey, const RelAlgNode*>;

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
  SortInfo()
      : order_entries({})
      , algorithm(SortAlgorithm::Default)
      , limit(std::nullopt)
      , offset(0) {}

  SortInfo(const std::list<Analyzer::OrderEntry>& oe,
           const SortAlgorithm sa,
           std::optional<size_t> l,
           size_t o)
      : order_entries(oe), algorithm(sa), limit(l), offset(o) {}

  SortInfo& operator=(const SortInfo& other) {
    order_entries = other.order_entries;
    algorithm = other.algorithm;
    limit = other.limit;
    offset = other.offset;
    return *this;
  }

  static SortInfo createFromSortNode(const RelSort* sort_node) {
    return {sort_node->getOrderEntries(),
            SortAlgorithm::SpeculativeTopN,
            sort_node->getLimit(),
            sort_node->getOffset()};
  }

  size_t hashLimit() const {
    size_t hash{0};
    boost::hash_combine(hash, limit.has_value());
    boost::hash_combine(hash, limit.value_or(0));
    return hash;
  }

  std::list<Analyzer::OrderEntry> order_entries;
  SortAlgorithm algorithm;
  std::optional<size_t> limit;
  size_t offset;
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
  QueryPlanHash query_plan_dag_hash{EMPTY_HASHED_PLAN_DAG_KEY};
  HashTableBuildDagMap hash_table_build_plan_dag{};
  TableIdToNodeMap table_id_to_node_map{};
  bool use_bump_allocator{false};
  // empty if not a UNION, true if UNION ALL, false if regular UNION
  const std::optional<bool> union_all;
  std::shared_ptr<const query_state::QueryState> query_state;
  std::vector<Analyzer::Expr*> target_exprs_union;  // targets in second subquery of UNION
  mutable std::vector<std::pair<std::vector<size_t>, size_t>> per_device_cardinality;

  RelAlgExecutionUnit createNdvExecutionUnit(const int64_t range) const;
  RelAlgExecutionUnit createCountAllExecutionUnit(
      Analyzer::Expr* replacement_target) const;

  // Call lambda() for each aggregate target_expr of SQLAgg type AggType.
  template <SQLAgg AggType>
  void eachAggTarget(
      std::function<void(Analyzer::AggExpr const*, size_t target_idx)> lambda) const {
    for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
      Analyzer::Expr const* target_expr = target_exprs[target_idx];
      if (auto const* agg_expr = dynamic_cast<Analyzer::AggExpr const*>(target_expr)) {
        if (agg_expr->get_aggtype() == AggType) {
          lambda(agg_expr, target_idx);
        }
      }
    }
  }
};

std::ostream& operator<<(std::ostream& os, const RelAlgExecutionUnit& ra_exe_unit);

struct TableFunctionExecutionUnit {
  const std::vector<InputDescriptor> input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::vector<Analyzer::Expr*> input_exprs;
  std::vector<Analyzer::ColumnVar*> table_func_inputs;
  std::vector<Analyzer::Expr*> target_exprs;
  mutable size_t output_buffer_size_param;
  const table_functions::TableFunction table_func;
  QueryPlanHash query_plan_dag_hash;

 public:
  std::string toString() const {
    return typeName(this) + "(" + "input_exprs=" + ::toString(input_exprs) +
           ", table_func_inputs=" + ::toString(table_func_inputs) +
           ", target_exprs=" + ::toString(target_exprs) +
           ", output_buffer_size_param=" + ::toString(output_buffer_size_param) +
           ", table_func=" + ::toString(table_func) +
           ", query_plan_dag=" + ::toString(query_plan_dag_hash) + ")";
  }
};

class ResultSet;
using ResultSetPtr = std::shared_ptr<ResultSet>;
