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

#include "QueryPlanDagExtractor.h"
#include "RexVisitor.h"
#include "Visitors/QueryPlanDagChecker.h"

#include <boost/algorithm/cxx11/any_of.hpp>

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
  auto do_normalize_inner_outer_pair = [this, &result](
                                           const Analyzer::Expr* lhs,
                                           const Analyzer::Expr* rhs,
                                           const TemporaryTables* temporary_table) {
    try {
      auto inner_outer_pair =
          HashJoin::normalizeColumnPair(lhs, rhs, schema_provider_, temporary_table);
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
          lhs_tuple[i].get(), rhs_tuple[i].get(), &temporary_tables_);
    }
  } else {
    do_normalize_inner_outer_pair(condition->get_left_operand(),
                                  condition->get_right_operand(),
                                  &temporary_tables_);
  }
  return result;
}

// To extract query plan DAG, we call this function with root node of the query plan
// and some objects required while extracting DAG
// We consider a DAG representation of a query plan as a series of "unique" rel node ids
// We decide each rel node's node id by searching the cached plan DAG first,
// and assign a new id iff there exists no duplicated rel node that can reuse
ExtractedPlanDag QueryPlanDagExtractor::extractQueryPlanDag(
    const RelAlgNode*
        node, /* the root node of the query plan tree we want to extract its DAG */
    SchemaProviderPtr schema_provider,
    std::optional<unsigned> left_deep_tree_id,
    std::unordered_map<unsigned, JoinQualsPerNestingLevel>& left_deep_tree_infos,
    const TemporaryTables& temporary_tables,
    Executor* executor,
    const RelAlgTranslator& rel_alg_translator) {
  // check if this plan tree has not supported pattern for DAG extraction
  if (QueryPlanDagChecker::isNotSupportedDag(node, rel_alg_translator)) {
    VLOG(1) << "Stop DAG extraction due to not supproed node: " << node->toString();
    return {node, EMPTY_QUERY_PLAN, nullptr, nullptr, {}, {}, true};
  }

  return extractQueryPlanDagImpl(node,
                                 schema_provider,
                                 left_deep_tree_id,
                                 left_deep_tree_infos,
                                 temporary_tables,
                                 executor);
}

ExtractedPlanDag QueryPlanDagExtractor::extractQueryPlanDagImpl(
    const RelAlgNode*
        node, /* the root node of the query plan tree we want to extract its DAG */
    SchemaProviderPtr schema_provider,
    std::optional<unsigned> left_deep_tree_id,
    std::unordered_map<unsigned, JoinQualsPerNestingLevel>& left_deep_tree_infos,
    const TemporaryTables& temporary_tables,
    Executor* executor) {
  mapd_unique_lock<mapd_shared_mutex> lock(executor->getDataRecyclerLock());

  auto& cached_dag = executor->getQueryPlanDagCache();
  QueryPlanDagExtractor dag_extractor(
      cached_dag, schema_provider, left_deep_tree_infos, temporary_tables, executor);

  // add the root node of this query plan DAG
  auto res = cached_dag.addNodeIfAbsent(node);
  if (!res) {
    VLOG(1) << "Stop DAG extraction while adding node to the DAG node cache: "
            << node->toString();
    return {node, EMPTY_QUERY_PLAN, nullptr, nullptr, {}, {}, true};
  }
  CHECK(res.has_value());
  node->setRelNodeDagId(res.value());
  dag_extractor.extracted_dag_.push_back(res.value());

  // visit child node if necessary
  auto num_child_node = node->inputCount();
  switch (num_child_node) {
    case 1:  // unary op
      dag_extractor.visit(node, node->getInput(0));
      break;
    case 2:  // binary op
      if (auto trans_join_node = dynamic_cast<const RelTranslatedJoin*>(node)) {
        dag_extractor.visit(trans_join_node, trans_join_node->getLHS());
        dag_extractor.visit(trans_join_node, trans_join_node->getRHS());
        break;
      }
      VLOG(1) << "Visit an invalid rel node while extracting query plan DAG: "
              << ::toString(node);
      return {node, EMPTY_QUERY_PLAN, nullptr, nullptr, {}, {}, true};
    case 0:  // leaf node
      break;
    default:
      // since we replace RelLeftDeepJoin as a set of RelTranslatedJoin
      // which is a binary op, # child nodes for every rel node should be <= 2
      UNREACHABLE();
  }

  // check whether extracted DAG is available to use
  if (dag_extractor.extracted_dag_.empty() || dag_extractor.isDagExtractionAvailable()) {
    return {node, EMPTY_QUERY_PLAN, nullptr, nullptr, {}, {}, true};
  }

  return {node,
          dag_extractor.getExtractedQueryPlanDagStr(),
          dag_extractor.getTranslatedJoinInfo(),
          dag_extractor.getPerNestingJoinQualInfo(left_deep_tree_id),
          dag_extractor.getHashTableBuildDag(),
          dag_extractor.getTableIdToNodeMap(),
          false};
}

std::string QueryPlanDagExtractor::getExtractedQueryPlanDagStr() {
  std::ostringstream oss;
  if (extracted_dag_.empty() || contain_not_supported_rel_node_) {
    oss << "N/A";
  } else {
    for (auto& dag_node_id : extracted_dag_) {
      oss << dag_node_id << "|";
    }
  }
  return oss.str();
}

bool QueryPlanDagExtractor::validateNodeId(const RelAlgNode* node,
                                           std::optional<RelNodeId> retrieved_node_id) {
  if (!retrieved_node_id) {
    VLOG(1) << "Stop DAG extraction while adding node to the DAG node cache: "
            << node->toString();
    clearInternaStatus();
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
  extracted_dag_.push_back(retrieved_node_id.value());
  return true;
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
  auto register_and_visit = [this](const RelAlgNode* parent_node,
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
  };
  if (auto left_deep_joins = dynamic_cast<const RelLeftDeepInnerJoin*>(child_node)) {
    if (left_deep_tree_infos_.empty()) {
      // we should have left_deep_tree_info for input left deep tree node
      VLOG(1) << "Stop DAG extraction due to not supported join pattern";
      clearInternaStatus();
      return;
    }
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
      auto dummy_filter = std::make_shared<RelFilter>(copied_inner_cond);
      register_and_visit(parent_node, dummy_filter.get());
      handleLeftDeepJoinTree(dummy_filter.get(), left_deep_joins);
    } else {
      handleLeftDeepJoinTree(parent_node, left_deep_joins);
    }
  } else if (auto translated_join_node =
                 dynamic_cast<const RelTranslatedJoin*>(child_node)) {
    handleTranslatedJoin(parent_node, translated_join_node);
  } else {
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
  QueryPlan current_plan_dag, after_rhs_visited, after_lhs_visited;
  current_plan_dag = getExtractedQueryPlanDagStr();
  auto rhs_node = rel_trans_join->getRHS();
  if (rhs_node) {
    visit(rel_trans_join, rhs_node);
    after_rhs_visited = getExtractedQueryPlanDagStr();
    addTableIdToNodeLink(rhs_node->getId(), rhs_node);
  }
  auto lhs_node = rel_trans_join->getLHS();
  if (rel_trans_join->getLHS()) {
    visit(rel_trans_join, lhs_node);
    after_lhs_visited = getExtractedQueryPlanDagStr();
    addTableIdToNodeLink(lhs_node->getId(), lhs_node);
  }
  if (isEmptyQueryPlanDag(after_lhs_visited) || isEmptyQueryPlanDag(after_rhs_visited)) {
    VLOG(1) << "Stop DAG extraction while extracting query plan DAG for join qual";
    clearInternaStatus();
    return;
  }
  // after visiting new node, we have added node id(s) which can be used as an access path
  // so, we extract that node id(s) by splitting the new plan dag by the current plan dag
  auto outer_table_identifier = split(after_rhs_visited, current_plan_dag)[1];
  auto hash_table_identfier = split(after_lhs_visited, after_rhs_visited)[1];

  if (!rel_trans_join->isNestedLoopQual()) {
    std::ostringstream oss;
    auto inner_join_cols = rel_trans_join->getJoinCols(true);
    oss << global_dag_.translateColVarsToInfoString(inner_join_cols, false);
    auto hash_table_cols_info = oss.str();
    oss << "|";
    auto outer_join_cols = rel_trans_join->getJoinCols(false);
    oss << global_dag_.translateColVarsToInfoString(outer_join_cols, false);
    auto join_qual_info = oss.str();
    // hash table join cols info | hash table build plan dag (hashtable identifier or
    // hashtable access path)
    auto it = hash_table_query_plan_dag_.find(join_qual_info);
    if (it == hash_table_query_plan_dag_.end()) {
      VLOG(2) << "Add hashtable access path"
              << ", join col info: " << hash_table_cols_info
              << ", access path: " << hash_table_identfier << "\n";
      hash_table_query_plan_dag_.emplace(
          join_qual_info, std::make_pair(hash_table_cols_info, hash_table_identfier));
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
int get_input_idx(const RelLeftDeepInnerJoin* rel_left_deep_join, int const tbl_id) {
  for (size_t input_idx = 0; input_idx < rel_left_deep_join->inputCount(); ++input_idx) {
    auto const input_node = rel_left_deep_join->getInput(input_idx);
    auto const scan_node = dynamic_cast<const RelScan*>(input_node);
    int const target_table_id = scan_node ? scan_node->getTableId()
                                          : -1 * input_node->getId();  // temporary table
    if (target_table_id == tbl_id) {
      return input_idx;
    }
  }
  return -1;
}
}  // namespace

Analyzer::ColumnVar const* QueryPlanDagExtractor::getColVar(
    Analyzer::Expr const* col_info) {
  auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(col_info);
  if (!col_var) {
    auto visited_cols = global_dag_.collectColVars(col_info);
    if (visited_cols.size() == 1) {
      col_var = dynamic_cast<const Analyzer::ColumnVar*>(visited_cols[0]);
    }
  }
  return col_var;
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
    VLOG(1) << "Stop DAG extraction due to not supported join pattern";
    clearInternaStatus();
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

    RexScalar const* const outer_join_cond =
        current_level_join_conditions.type == JoinType::LEFT
            ? rel_left_deep_join->getOuterCondition(level_idx + 1)
            : nullptr;

    // collect join col, filter ops, and detailed info of join operation, i.e., op_type,
    // qualifier, ...
    // when we have more than one quals, i.e., current_level_join_conditions.quals.size()
    // > 1, we consider the first qual is used as hashtable building
    for (const auto& join_qual : current_level_join_conditions.quals) {
      auto qual_bin_oper = std::dynamic_pointer_cast<const Analyzer::BinOper>(join_qual);
      auto join_qual_str = ::toString(join_qual);
      if (qual_bin_oper) {
        if (join_qual == current_level_join_conditions.quals.front()) {
          // set op_info based on the first qual
          op_info = OpInfo{::toString(qual_bin_oper->get_optype()),
                           ::toString(qual_bin_oper->get_qualifier()),
                           qual_bin_oper->get_type_info().to_string()};
        }
        for (auto& col_pair_info : normalizeColumnsPair(qual_bin_oper.get())) {
          if (col_pair_info.loop_join_qual && !found_eq_join_qual) {
            // we only consider that cur level's join is loop join if we have no
            // equi-join qual and both lhs and rhs are not col_var,
            // i.e., lhs: col_var / rhs: constant / bin_op: kGE
            if (visited_filter_ops.emplace(std::move(join_qual_str)).second) {
              filter_ops.push_back(join_qual);
            }
          } else {
            // a qual_bin_oper becomes an inner join qual iff both lhs and rhs are col_var
            // otherwise it becomes a filter qual
            bool found_valid_col_vars = false;
            if (col_pair_info.inner_outer.first && col_pair_info.inner_outer.second) {
              auto const* lhs_col_var = getColVar(col_pair_info.inner_outer.first);
              auto const* rhs_col_var = getColVar(col_pair_info.inner_outer.second);
              // this qual is valid and used for join op
              if (lhs_col_var && rhs_col_var) {
                found_valid_col_vars = true;
                if (inner_input_idx == -1) {
                  inner_input_idx =
                      get_input_idx(rel_left_deep_join, lhs_col_var->get_table_id());
                }
                if (outer_input_idx == -1) {
                  outer_input_idx =
                      get_input_idx(rel_left_deep_join, rhs_col_var->get_table_id());
                }
                inner_join_cols.push_back(lhs_col_var);
                outer_join_cols.push_back(rhs_col_var);
              }
            }
            if (!found_valid_col_vars &&
                visited_filter_ops.emplace(std::move(join_qual_str)).second) {
              filter_ops.push_back(join_qual);
            }
          }
        }
      } else {
        if (visited_filter_ops.emplace(std::move(join_qual_str)).second) {
          filter_ops.push_back(join_qual);
        }
      }
    }
    if (inner_join_cols.size() != outer_join_cols.size()) {
      VLOG(1) << "Stop DAG extraction due to inner/outer col mismatch";
      clearInternaStatus();
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
