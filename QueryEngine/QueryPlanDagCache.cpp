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

#include "QueryPlanDagCache.h"
#include "RexVisitor.h"

#include <unordered_set>

// an approximation of each rel node's size: 2 * sizeof(size_t): unique ID for node_map_
// and for DAG graph
constexpr size_t elem_size_ = 2 * sizeof(size_t);

std::optional<RelNodeId> QueryPlanDagCache::addNodeIfAbsent(const RelAlgNode* node) {
  std::lock_guard<std::mutex> cache_lock(cache_lock_);
  auto key = node->toHash();
  auto const result = node_map_.emplace(key, getCurrentNodeMapCardinality());
  if (result.second) {
    // key did not already exist in node_map_. Check max size wasn't hit.
    if (getCurrentNodeMapSize() > max_node_map_size_ ||
        getCurrentNodeMapCardinality() == SIZE_MAX) {
      // unfortunately our DAG cache becomes full if we add this node to it
      // so we skip to cache this plan DAG and clear cache map
      // b/c this can be happen in a middle of dag extraction
      node_map_.clear();
      cached_query_plan_dag_.graph().clear();
      // assume we cannot keep 'InvalidQueryPlanHash' nodes for our DAG cache
      return std::nullopt;
    }
  }
  return result.first->second;
}

void QueryPlanDagCache::connectNodes(const RelNodeId parent_id,
                                     const RelNodeId child_id) {
  std::lock_guard<std::mutex> cache_lock(cache_lock_);
  boost::add_vertex(parent_id, cached_query_plan_dag_);
  boost::add_vertex(child_id, cached_query_plan_dag_);
  add_edge_by_label(parent_id, child_id, cached_query_plan_dag_);
}

void QueryPlanDagCache::setNodeMapMaxSize(const size_t map_size) {
  std::lock_guard<std::mutex> cache_lock(cache_lock_);
  max_node_map_size_ = map_size;
}

JoinColumnsInfo QueryPlanDagCache::translateColVarsToInfoString(
    std::vector<const Analyzer::ColumnVar*>& col_vars,
    bool col_id_only) const {
  // we need to sort col ids to prevent missing data reuse case in multi column qual
  // scenarios like a) A.a = B.b and A.c = B.c and b) A.c = B.c and A.a = B.a
  std::sort(col_vars.begin(),
            col_vars.end(),
            [](const Analyzer::ColumnVar* lhs, const Analyzer::ColumnVar* rhs) {
              return lhs->get_column_id() < rhs->get_column_id();
            });
  if (col_id_only) {
    std::vector<int> sorted_col_ids;
    for (auto cv : col_vars) {
      sorted_col_ids.push_back(cv->get_column_id());
    }
    return ::toString(sorted_col_ids);
  } else {
    return ::toString(col_vars);
  }
}

JoinColumnsInfo QueryPlanDagCache::getJoinColumnsInfoString(
    const Analyzer::Expr* join_expr,
    JoinColumnSide target_side,
    bool extract_only_col_id) {
  // this function returns qual_bin_oper's info depending on the requested context
  // such as extracted col_id of inner join cols
  // (target_side = JoinColumnSide::kInner, extract_only_col_id = true)
  // and extract all infos of an entire join quals
  // (target_side = JoinColumnSide::kQual, extract_only_col_id = false)
  // todo (yoonmin): we may need to use a whole "EXPR" contents in a future
  // to support a join qual with more general expression like A.a + 1 = (B.b * 2) / 2
  if (!join_expr) {
    return "";
  }
  auto get_sorted_col_info = [&](const Analyzer::Expr* join_cols) -> JoinColumnsInfo {
    auto join_col_vars = collectColVars(join_cols);
    if (join_col_vars.empty()) {
      return "";
    }
    return translateColVarsToInfoString(join_col_vars, extract_only_col_id);
  };

  if (target_side == JoinColumnSide::kQual) {
    auto qual_bin_oper = reinterpret_cast<const Analyzer::BinOper*>(join_expr);
    CHECK(qual_bin_oper);
    auto inner_join_col_info = get_sorted_col_info(qual_bin_oper->get_left_operand());
    auto outer_join_col_info = get_sorted_col_info(qual_bin_oper->get_right_operand());
    return outer_join_col_info + "|" + inner_join_col_info;
  } else if (target_side == JoinColumnSide::kInner) {
    auto qual_bin_oper = reinterpret_cast<const Analyzer::BinOper*>(join_expr);
    CHECK(qual_bin_oper);
    return get_sorted_col_info(qual_bin_oper->get_left_operand());
  } else if (target_side == JoinColumnSide::kOuter) {
    auto qual_bin_oper = reinterpret_cast<const Analyzer::BinOper*>(join_expr);
    CHECK(qual_bin_oper);
    return get_sorted_col_info(qual_bin_oper->get_right_operand());
  } else {
    CHECK(target_side == JoinColumnSide::kDirect);
    return get_sorted_col_info(join_expr);
  }
}

void QueryPlanDagCache::printDag() {
  std::cout << "Edge list:" << std::endl;
  boost::print_graph(cached_query_plan_dag_.graph());
  std::ostringstream os;
  os << "\n\nNodeMap:\n";
  for (auto& kv : node_map_) {
    os << "[" << kv.second << "] " << kv.first << "\n";
  }
  std::cout << os.str() << std::endl;
}

size_t QueryPlanDagCache::getCurrentNodeMapSize() const {
  return node_map_.size() * elem_size_;
}

size_t QueryPlanDagCache::getCurrentNodeMapCardinality() const {
  return node_map_.size();
}

void QueryPlanDagCache::clearQueryPlanCache() {
  std::lock_guard<std::mutex> cache_lock(cache_lock_);
  node_map_.clear();
  cached_query_plan_dag_.graph().clear();
}

std::vector<const Analyzer::ColumnVar*> QueryPlanDagCache::collectColVars(
    const Analyzer::Expr* target) {
  if (target) {
    return col_var_visitor_.visit(target);
  }
  return {};
}
