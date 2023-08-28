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

#include "QueryPlanDagExtractor.h"

#include <boost/algorithm/cxx11/any_of.hpp>

#include "RexVisitor.h"
#include "Shared/DbObjectKeys.h"
#include "Visitors/QueryPlanDagChecker.h"

extern bool g_is_test_env;

namespace {
struct IsEquivBinOp {
  bool operator()(std::shared_ptr<Analyzer::Expr> const& qual) {
    if (auto oper = std::dynamic_pointer_cast<const Analyzer::BinOper>(qual)) {
      return IS_EQUIVALENCE(oper->get_optype());
    }
    return false;
  }
};
}  // namespace

std::vector<InnerOuterOrLoopQual> QueryPlanDagExtractor::normalizeColumnsPair(
    const Analyzer::BinOper* condition) {
  std::vector<InnerOuterOrLoopQual> result;
  const auto lhs_tuple_expr =
      dynamic_cast<const Analyzer::ExpressionTuple*>(condition->get_left_operand());
  const auto rhs_tuple_expr =
      dynamic_cast<const Analyzer::ExpressionTuple*>(condition->get_right_operand());

  CHECK_EQ(static_cast<bool>(lhs_tuple_expr), static_cast<bool>(rhs_tuple_expr));
  auto do_normalize_inner_outer_pair = [&result, &condition](
                                           const Analyzer::Expr* lhs,
                                           const Analyzer::Expr* rhs,
                                           const TemporaryTables* temporary_table) {
    try {
      auto inner_outer_pair =
          HashJoin::normalizeColumnPair(
              lhs, rhs, temporary_table, condition->is_overlaps_oper())
              .first;
      InnerOuterOrLoopQual valid_qual{
          std::make_pair(inner_outer_pair.first, inner_outer_pair.second), false};
      result.push_back(valid_qual);
    } catch (HashJoinFail& e) {
      InnerOuterOrLoopQual invalid_qual{std::make_pair(lhs, rhs), true};
      result.push_back(invalid_qual);
    }
  };
  if (lhs_tuple_expr) {
    const auto& lhs_tuple = lhs_tuple_expr->getTuple();
    const auto& rhs_tuple = rhs_tuple_expr->getTuple();
    CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
    for (size_t i = 0; i < lhs_tuple.size(); ++i) {
      do_normalize_inner_outer_pair(
          lhs_tuple[i].get(), rhs_tuple[i].get(), executor_->getTemporaryTables());
    }
  } else {
    do_normalize_inner_outer_pair(condition->get_left_operand(),
                                  condition->get_right_operand(),
                                  executor_->getTemporaryTables());
  }
  return result;
}

// To extract query plan DAG, we call this function with root node of the query plan
// and some objects required while extracting DAG
// We consider a DAG representation of a query plan as a series of "unique" rel node ids
// We decide each rel node's node id by searching the cached plan DAG first,
// and assign a new id iff there exists no duplicated rel node that can reuse
ExtractedQueryPlanDag QueryPlanDagExtractor::extractQueryPlanDag(
    const RelAlgNode* top_node,
    Executor* executor) {
  auto dag_checker_res = QueryPlanDagChecker::hasNonSupportedNodeInDag(top_node);
  if (dag_checker_res.first) {
    VLOG(1) << "Stop DAG extraction (" << dag_checker_res.second << ")";
    return {EMPTY_QUERY_PLAN, true};
  }
  heavyai::unique_lock<heavyai::shared_mutex> lock(executor->getDataRecyclerLock());
  auto& cached_dag = executor->getQueryPlanDagCache();
  QueryPlanDagExtractor dag_extractor(cached_dag, {}, executor, false);
  extractQueryPlanDagImpl(top_node, dag_extractor);
  auto extracted_query_plan_dag = dag_extractor.getExtractedQueryPlanDagStr();
  top_node->setQueryPlanDag(extracted_query_plan_dag);
  if (auto sort_node = dynamic_cast<RelSort const*>(top_node)) {
    // we evaluate sort node based on the resultset of its child node
    // so we need to mark the extracted query plan of the child node
    // for the resultset recycling
    auto child_dag = dag_extractor.getExtractedQueryPlanDagStr(1);
    sort_node->getInput(0)->setQueryPlanDag(child_dag);
  }
  return {extracted_query_plan_dag, dag_extractor.isDagExtractionAvailable()};
}

ExtractedJoinInfo QueryPlanDagExtractor::extractJoinInfo(
    const RelAlgNode* top_node,
    std::optional<unsigned> left_deep_tree_id,
    std::unordered_map<unsigned, JoinQualsPerNestingLevel> left_deep_tree_infos,
    Executor* executor) {
  // we already extract a query plan dag for the input query which is stored at top_node
  if (top_node->getQueryPlanDagHash() == EMPTY_HASHED_PLAN_DAG_KEY) {
    return {};
  }
  heavyai::unique_lock<heavyai::shared_mutex> lock(executor->getDataRecyclerLock());
  auto& cached_dag = executor->getQueryPlanDagCache();
  QueryPlanDagExtractor dag_extractor(cached_dag, left_deep_tree_infos, executor, true);
  extractQueryPlanDagImpl(top_node, dag_extractor);
  return {dag_extractor.getHashTableBuildDag(), dag_extractor.getTableIdToNodeMap()};
}

void QueryPlanDagExtractor::extractQueryPlanDagImpl(
    const RelAlgNode* top_node,
    QueryPlanDagExtractor& dag_extractor) {
  // add the root node of this query plan DAG
  auto res = dag_extractor.global_dag_.addNodeIfAbsent(top_node);
  if (!res) {
    VLOG(1) << "Stop DAG extraction (Query plan dag cache reaches the maximum capacity)";
    dag_extractor.contain_not_supported_rel_node_ = true;
    return;
  }
  CHECK(res.has_value());
  top_node->setRelNodeDagId(res.value());
  dag_extractor.extracted_dag_.push_back(::toString(res.value()));

  // visit child node if necessary
  if (auto table_func_node = dynamic_cast<const RelTableFunction*>(top_node)) {
    for (size_t i = 0; i < table_func_node->inputCount(); ++i) {
      dag_extractor.visit(table_func_node, table_func_node->getInput(i));
    }
  } else {
    auto num_child_node = top_node->inputCount();
    switch (num_child_node) {
      case 1:  // unary op
        dag_extractor.visit(top_node, top_node->getInput(0));
        break;
      case 2:  // binary op
        if (auto trans_join_node = dynamic_cast<const RelTranslatedJoin*>(top_node)) {
          dag_extractor.visit(trans_join_node, trans_join_node->getLHS());
          dag_extractor.visit(trans_join_node, trans_join_node->getRHS());
          break;
        }
        VLOG(1) << "Visit an invalid rel node while extracting query plan DAG: "
                << ::toString(top_node);
        return;
      case 0:  // leaf node
        break;
      default:
        // since we replace RelLeftDeepJoin as a set of RelTranslatedJoin
        // which is a binary op, # child nodes for every rel node should be <= 2
        UNREACHABLE();
    }
  }

  // check whether extracted DAG is available to use
  if (dag_extractor.extracted_dag_.empty() || dag_extractor.isDagExtractionAvailable()) {
    dag_extractor.contain_not_supported_rel_node_ = true;
    return;
  }

  if (g_is_test_env) {
    dag_extractor.executor_->registerExtractedQueryPlanDag(
        dag_extractor.getExtractedQueryPlanDagStr());
  }
  return;
}

std::string QueryPlanDagExtractor::getExtractedQueryPlanDagStr(size_t start_pos) {
  std::ostringstream oss;
  size_t cnt = 0;
  if (start_pos > extracted_dag_.size()) {
    return EMPTY_QUERY_PLAN;
  }
  for (auto& dag_node_id : extracted_dag_) {
    if (cnt >= start_pos) {
      oss << dag_node_id << "|";
    }
    ++cnt;
  }
  return oss.str();
}

bool QueryPlanDagExtractor::validateNodeId(const RelAlgNode* node,
                                           std::optional<RelNodeId> retrieved_node_id) {
  if (!retrieved_node_id) {
    VLOG(1) << "Stop DAG extraction (Detect an invalid dag id)";
    clearInternalStatus();
    return false;
  }
  CHECK(retrieved_node_id.has_value());
  node->setRelNodeDagId(retrieved_node_id.value());
  return true;
}

bool QueryPlanDagExtractor::registerNodeToDagCache(
    const RelAlgNode* parent_node,
    const RelAlgNode* child_node,
    std::optional<RelNodeId> retrieved_node_id) {
  CHECK(parent_node);
  CHECK(child_node);
  CHECK(retrieved_node_id.has_value());
  auto parent_node_id = parent_node->getRelNodeDagId();
  global_dag_.connectNodes(parent_node_id, retrieved_node_id.value());
  extracted_dag_.push_back(::toString(retrieved_node_id.value()));
  return true;
}

void QueryPlanDagExtractor::register_and_visit(const RelAlgNode* parent_node,
                                               const RelAlgNode* child_node) {
  // This function takes a responsibility for all rel nodes
  // except 1) RelLeftDeepJoinTree and 2) RelTranslatedJoin
  auto res = global_dag_.addNodeIfAbsent(child_node);
  if (validateNodeId(child_node, res) &&
      registerNodeToDagCache(parent_node, child_node, res)) {
    for (size_t i = 0; i < child_node->inputCount(); i++) {
      visit(child_node, child_node->getInput(i));
    }
  }
}

// we recursively visit each rel node starting from the root
// and collect assigned rel node ids and return them as query plan DAG
// for join operations we additionally generate additional information
// to recycle each hashtable that needs to process a given query
void QueryPlanDagExtractor::visit(const RelAlgNode* parent_node,
                                  const RelAlgNode* child_node) {
  if (!child_node || contain_not_supported_rel_node_) {
    return;
  }
  bool child_visited = false;
  if (analyze_join_ops_) {
    if (auto left_deep_joins = dynamic_cast<const RelLeftDeepInnerJoin*>(child_node)) {
      if (left_deep_tree_infos_.empty()) {
        // we should have left_deep_tree_info for input left deep tree node
        VLOG(1) << "Stop DAG extraction (Detect non-supported join pattern)";
        clearInternalStatus();
        return;
      }
      auto true_parent_node = parent_node;
      std::shared_ptr<RelFilter> dummy_filter_node{nullptr};
      const auto inner_cond = left_deep_joins->getInnerCondition();
      // we analyze left-deep join tree as per-join qual level, so
      // when visiting RelLeftDeepInnerJoin we decompose it into individual join node
      // (RelTranslatedJoin).
      // Thus, this RelLeftDeepInnerJoin object itself is useless when recycling data
      // but sometimes it has inner condition that has to consider so we add an extra
      // RelFilter node containing the condition to keep query semantic correctly
      if (auto cond = dynamic_cast<const RexOperator*>(inner_cond)) {
        RexDeepCopyVisitor copier;
        auto copied_inner_cond = copier.visit(cond);
        dummy_filter_node = std::make_shared<RelFilter>(copied_inner_cond);
        register_and_visit(parent_node, dummy_filter_node.get());
        true_parent_node = dummy_filter_node.get();
      }
      handleLeftDeepJoinTree(true_parent_node, left_deep_joins);
      child_visited = true;
    } else if (auto translated_join_node =
                   dynamic_cast<const RelTranslatedJoin*>(child_node)) {
      handleTranslatedJoin(parent_node, translated_join_node);
      child_visited = true;
    }
  }
  if (!child_visited) {
    register_and_visit(parent_node, child_node);
  }
}

void QueryPlanDagExtractor::handleTranslatedJoin(
    const RelAlgNode* parent_node,
    const RelTranslatedJoin* rel_trans_join) {
  // when left-deep tree has multiple joins this rel_trans_join can be revisited
  // but we need to mark the child query plan to accurately catch the query plan dag
  // here we do not create new dag id since all rel nodes are visited already
  CHECK(parent_node);
  CHECK(rel_trans_join);

  auto res = global_dag_.addNodeIfAbsent(rel_trans_join);
  if (!validateNodeId(rel_trans_join, res) ||
      !registerNodeToDagCache(parent_node, rel_trans_join, res)) {
    return;
  }

  // To extract an access path (query plan DAG) for hashtable is to use a difference of
  // two query plan DAGs 1) query plan DAG after visiting RHS node and 2) query plan DAG
  // after visiting LHS node so by comparing 1) and 2) we can extract which query plan DAG
  // is necessary to project join cols that are used to build a hashtable and we use it as
  // hashtable access path

  // this lambda function deals with the case of recycled query resultset
  // specifically, we can avoid unnecessary visiting of child tree(s) when we already have
  // the extracted query plan DAG for the given child rel node
  // instead, we just fill the node id vector (a sequence of node ids we visited) by using
  // the dag of the child node
  auto fill_node_ids_to_dag_vec = [&](const std::string& node_ids) {
    auto node_ids_vec = split(node_ids, "|");
    // the last elem is an empty one
    std::for_each(node_ids_vec.begin(),
                  std::prev(node_ids_vec.end()),
                  [&](const std::string& node_id) { extracted_dag_.push_back(node_id); });
  };
  QueryPlanDAG current_plan_dag, after_rhs_visited, after_lhs_visited;
  current_plan_dag = getExtractedQueryPlanDagStr();
  auto rhs_node = rel_trans_join->getRHS();
  std::unordered_set<size_t> rhs_input_keys, lhs_input_keys;
  if (rhs_node) {
    if (rhs_node->getQueryPlanDagHash() == EMPTY_HASHED_PLAN_DAG_KEY) {
      visit(rel_trans_join, rhs_node);
    } else {
      fill_node_ids_to_dag_vec(rhs_node->getQueryPlanDag());
    }
    after_rhs_visited = getExtractedQueryPlanDagStr();
    addTableIdToNodeLink({0, int32_t(rhs_node->getId())}, rhs_node);
    rhs_input_keys = ScanNodeTableKeyCollector::getScanNodeTableKey(rhs_node);
  }
  auto lhs_node = rel_trans_join->getLHS();
  if (rel_trans_join->getLHS()) {
    if (lhs_node->getQueryPlanDagHash() == EMPTY_HASHED_PLAN_DAG_KEY) {
      visit(rel_trans_join, lhs_node);
    } else {
      fill_node_ids_to_dag_vec(lhs_node->getQueryPlanDag());
    }
    after_lhs_visited = getExtractedQueryPlanDagStr();
    addTableIdToNodeLink({0, int32_t(lhs_node->getId())}, lhs_node);
    lhs_input_keys = ScanNodeTableKeyCollector::getScanNodeTableKey(lhs_node);
  }
  if (isEmptyQueryPlanDag(after_lhs_visited) || isEmptyQueryPlanDag(after_rhs_visited)) {
    VLOG(1) << "Stop DAG extraction (Detect invalid query plan dag of join col(s))";
    clearInternalStatus();
    return;
  }
  // after visiting new node, we have added node id(s) which can be used as an access path
  // so, we extract that node id(s) by splitting the new plan dag by the current plan dag
  auto outer_table_identifier = split(after_rhs_visited, current_plan_dag)[1];
  auto hash_table_identfier = split(after_lhs_visited, after_rhs_visited)[1];
  auto join_qual_info = EMPTY_HASHED_PLAN_DAG_KEY;
  if (!rel_trans_join->isNestedLoopQual()) {
    auto inner_join_cols = rel_trans_join->getJoinCols(true);
    auto inner_join_col_info =
        global_dag_.translateColVarsToInfoHash(inner_join_cols, false);
    boost::hash_combine(join_qual_info, inner_join_col_info);
    auto outer_join_cols = rel_trans_join->getJoinCols(false);
    auto outer_join_col_info =
        global_dag_.translateColVarsToInfoHash(outer_join_cols, false);
    boost::hash_combine(join_qual_info, outer_join_col_info);
    // collect table keys from both rhs and lhs side
    std::unordered_set<size_t> collected_table_keys;
    collected_table_keys.insert(lhs_input_keys.begin(), lhs_input_keys.end());
    if (!inner_join_cols.empty() &&
        inner_join_cols[0]->get_type_info().is_dict_encoded_type()) {
      collected_table_keys.insert(rhs_input_keys.begin(), rhs_input_keys.end());
    }

    auto it = hash_table_query_plan_dag_.find(join_qual_info);
    if (it == hash_table_query_plan_dag_.end()) {
      VLOG(2) << "Add hashtable access path"
              << ", inner join col info: " << inner_join_col_info
              << " (access path: " << hash_table_identfier << ")"
              << ", outer join col info: " << outer_join_col_info
              << " (access path: " << outer_table_identifier << ")";
      hash_table_query_plan_dag_.emplace(
          join_qual_info,
          HashTableBuildDag(inner_join_col_info,
                            outer_join_col_info,
                            boost::hash_value(hash_table_identfier),
                            boost::hash_value(outer_table_identifier),
                            std::move(collected_table_keys)));
    }
  } else {
    VLOG(2) << "Add loop join access path, for LHS: " << outer_table_identifier
            << ", for RHS: " << hash_table_identfier << "\n";
  }
}

namespace {
struct OpInfo {
  std::string type_;
  std::string qualifier_;
  std::string typeinfo_;
};

// Return the input index whose tableId matches the given tbl_id.
// If none then return -1.
int get_input_idx(const RelLeftDeepInnerJoin* rel_left_deep_join,
                  const shared::TableKey& table_key) {
  for (size_t input_idx = 0; input_idx < rel_left_deep_join->inputCount(); ++input_idx) {
    auto const input_node = rel_left_deep_join->getInput(input_idx);
    auto const scan_node = dynamic_cast<const RelScan*>(input_node);
    shared::TableKey target_table_key{0, 0};
    if (scan_node) {
      target_table_key.db_id = scan_node->getCatalog().getDatabaseId();
      target_table_key.table_id = scan_node->getTableDescriptor()->tableId;
    } else {
      target_table_key.table_id = -1 * input_node->getId();  // temporary table
    }
    if (target_table_key == table_key) {
      return input_idx;
    }
  }
  return -1;
}
}  // namespace

std::vector<Analyzer::ColumnVar const*> QueryPlanDagExtractor::getColVar(
    Analyzer::Expr const* col_info) {
  if (auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(col_info)) {
    return {col_var};
  } else {
    return global_dag_.collectColVars(col_info);
  }
}

// we coalesce join quals and related filter conditions into a single RelLeftDeepInnerJoin
// node when converting calcite AST to logical query plan, but to recycle hashtable(s) we
// need to know access path of each hashtable, so we disassemble it into a set of join
// qual and collect hashtable info from there
void QueryPlanDagExtractor::handleLeftDeepJoinTree(
    const RelAlgNode* parent_node,
    const RelLeftDeepInnerJoin* rel_left_deep_join) {
  CHECK(parent_node);
  CHECK(rel_left_deep_join);

  // RelLeftDeepInnerJoin node does not need to be added to DAG since
  // RelLeftDeepInnerJoin is a logical node and
  // we add all join nodes of this `RelLeftDeepInnerJoin`
  // thus, the below `left_deep_tree_id` is not the same as its DAG id
  // (we do not have a DAG node id for this `RelLeftDeepInnerJoin`)
  auto left_deep_tree_id = rel_left_deep_join->getId();
  auto left_deep_join_info = getPerNestingJoinQualInfo(left_deep_tree_id);
  if (!left_deep_join_info) {
    // we should have left_deep_tree_info for input left deep tree node
    VLOG(1) << "Stop DAG extraction (Detect Non-supported join pattern)";
    clearInternalStatus();
    return;
  }

  // gathering per-join qual level info to correctly recycle each hashtable (and necessary
  // info) that we created
  // Here we visit each join qual in bottom-up manner to distinct DAGs among join quals
  // Let say we have three joins- #1: R.a = S.a / #2: R.b = T.b / #3. R.c = X.c
  // When we start to visit #1, we cannot determine outer col's dag clearly
  // since we need to visit #2 and #3 due to the current visitor's behavior
  // In contrast, when starting from #3, we clearly retrieve both inputs' dag
  // by skipping already visited nodes
  // So when we visit #2 after visiting #3, we can skip to consider nodes beloning to
  // qual #3 so we clearly retrieve DAG only corresponding to #2's
  for (size_t level_idx = 0; level_idx < left_deep_join_info->size(); ++level_idx) {
    const auto& current_level_join_conditions = left_deep_join_info->at(level_idx);
    std::vector<const Analyzer::ColumnVar*> inner_join_cols;
    std::vector<const Analyzer::ColumnVar*> outer_join_cols;
    std::vector<std::shared_ptr<const Analyzer::Expr>> filter_ops;
    int inner_input_idx{-1};
    int outer_input_idx{-1};
    OpInfo op_info{"UNDEFINED", "UNDEFINED", "UNDEFINED"};
    std::unordered_set<std::string> visited_filter_ops;

    // we first check whether this qual needs nested loop
    const bool found_eq_join_qual =
        current_level_join_conditions.type != JoinType::INVALID &&
        boost::algorithm::any_of(current_level_join_conditions.quals, IsEquivBinOp{});
    const bool nested_loop = !found_eq_join_qual;
    const bool is_left_join = current_level_join_conditions.type == JoinType::LEFT;
    RexScalar const* const outer_join_cond =
        is_left_join ? rel_left_deep_join->getOuterCondition(level_idx + 1) : nullptr;

    // collect join col, filter ops, and detailed info of join operation, i.e., op_type,
    // qualifier, ...
    // when we have more than one quals, i.e., current_level_join_conditions.quals.size()
    // > 1, we consider the first qual is used as hashtable building
    bool is_geo_join{false};
    for (const auto& join_qual : current_level_join_conditions.quals) {
      auto qual_bin_oper = std::dynamic_pointer_cast<const Analyzer::BinOper>(join_qual);
      auto join_qual_str = ::toString(join_qual);
      if (qual_bin_oper) {
        is_geo_join = qual_bin_oper->is_overlaps_oper();
        if (join_qual == current_level_join_conditions.quals.front()) {
          // set op_info based on the first qual
          op_info = OpInfo{::toString(qual_bin_oper->get_optype()),
                           ::toString(qual_bin_oper->get_qualifier()),
                           qual_bin_oper->get_type_info().to_string()};
        }
        for (auto& col_pair_info : normalizeColumnsPair(qual_bin_oper.get())) {
          // even though we fall back to loop join when left outer join has
          // non-eq comparator so we additionally check it here to classify the qual
          // properly todo(yoonmin): relax left join case once we have an improved logic
          // for this
          if (!found_eq_join_qual && (is_left_join || col_pair_info.loop_join_qual)) {
            // we only consider that cur level's join is loop join if we have no
            // equi-join qual and both lhs and rhs are not col_var,
            // i.e., lhs: col_var / rhs: constant / bin_op: kGE
            // also, currently we fallback to loop-join when left outer join
            // has non-eq join qual
            if (visited_filter_ops.insert(join_qual_str).second) {
              filter_ops.push_back(join_qual);
            }
          } else {
            // a qual_bin_oper becomes an inner join qual iff both lhs and rhs are col_var
            // otherwise it becomes a filter qual
            bool found_valid_col_vars = false;
            std::vector<const Analyzer::ColumnVar*> lhs_cvs, rhs_cvs;
            if (col_pair_info.inner_outer.first && col_pair_info.inner_outer.second) {
              // we need to modify lhs and rhs if range_oper is detected
              if (auto range_oper = dynamic_cast<const Analyzer::RangeOper*>(
                      col_pair_info.inner_outer.second)) {
                lhs_cvs = getColVar(range_oper->get_left_operand());
                rhs_cvs = getColVar(col_pair_info.inner_outer.first);
                is_geo_join = true;
              } else {
                // this qual is valid and used for typical hash join op
                lhs_cvs = getColVar(col_pair_info.inner_outer.first);
                rhs_cvs = getColVar(col_pair_info.inner_outer.second);
              }
              if (!lhs_cvs.empty() && !rhs_cvs.empty()) {
                found_valid_col_vars = true;
                if (inner_input_idx == -1) {
                  inner_input_idx =
                      get_input_idx(rel_left_deep_join, lhs_cvs.front()->getTableKey());
                }
                if (outer_input_idx == -1) {
                  outer_input_idx =
                      get_input_idx(rel_left_deep_join, rhs_cvs.front()->getTableKey());
                }
                std::copy(
                    lhs_cvs.begin(), lhs_cvs.end(), std::back_inserter(inner_join_cols));
                std::copy(
                    rhs_cvs.begin(), rhs_cvs.end(), std::back_inserter(outer_join_cols));
              }
            }
            if (!found_valid_col_vars &&
                visited_filter_ops.insert(join_qual_str).second) {
              filter_ops.push_back(join_qual);
            }
          }
        }
      } else {
        if (visited_filter_ops.insert(join_qual_str).second) {
          filter_ops.push_back(join_qual);
        }
      }
    }
    if (!is_geo_join && (inner_join_cols.size() != outer_join_cols.size())) {
      VLOG(1) << "Stop DAG extraction (Detect inner/outer col mismatch)";
      clearInternalStatus();
      return;
    }

    // create RelTranslatedJoin based on the collected info from the join qual
    // there are total seven types of join query pattern
    //  1. INNER HASH ONLY
    //  2. INNER LOOP ONLY (!)
    //  3. LEFT LOOP ONLY
    //  4. INNER HASH + INNER LOOP (!)
    //  5. LEFT LOOP + INNER HASH
    //  6. LEFT LOOP + INNER LOOP (!)
    //  7. LEFT LOOP + INNER HASH + INNER LOOP (!)
    // here, if a query contains INNER LOOP join, its qual has nothing
    // so, some patterns do not have bin_oper at the specific join nest level
    // if we find INNER LOOP, corresponding RelTranslatedJoin has nulled LHS and RHS
    // to mark it as loop join
    const RelAlgNode* lhs;
    const RelAlgNode* rhs;
    if (inner_input_idx != -1 && outer_input_idx != -1) {
      lhs = rel_left_deep_join->getInput(inner_input_idx);
      rhs = rel_left_deep_join->getInput(outer_input_idx);
    } else {
      if (level_idx == 0) {
        lhs = rel_left_deep_join->getInput(0);
        rhs = rel_left_deep_join->getInput(1);
      } else {
        lhs = translated_join_info_->rbegin()->get();
        rhs = rel_left_deep_join->getInput(level_idx + 1);
      }
    }
    CHECK(lhs);
    CHECK(rhs);
    auto cur_translated_join_node =
        std::make_shared<RelTranslatedJoin>(lhs,
                                            rhs,
                                            inner_join_cols,
                                            outer_join_cols,
                                            filter_ops,
                                            outer_join_cond,
                                            nested_loop,
                                            current_level_join_conditions.type,
                                            op_info.type_,
                                            op_info.qualifier_,
                                            op_info.typeinfo_);
    CHECK(cur_translated_join_node);
    handleTranslatedJoin(parent_node, cur_translated_join_node.get());
    translated_join_info_->push_back(std::move(cur_translated_join_node));
  }
}

size_t QueryPlanDagExtractor::applyLimitClauseToCacheKey(size_t cache_key,
                                                         SortInfo const& sort_info) {
  boost::hash_combine(cache_key, sort_info.hashLimit());
  return cache_key;
}
