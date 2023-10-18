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

#include "FromTableReordering.h"
#include "Execute.h"
#include "ExpressionRewrite.h"
#include "RangeTableIndexVisitor.h"
#include "Visitors/GeospatialFunctionFinder.h"

#include <numeric>
#include <queue>
#include <regex>

namespace {

using cost_t = unsigned;
using node_t = size_t;

const Analyzer::ColumnVar* get_geo_cv(
    std::vector<const Analyzer::ColumnVar*> const& geo_args,
    shared::TableKey const& table_key) {
  auto it = std::find_if(
      geo_args.begin(), geo_args.end(), [&table_key](const Analyzer::ColumnVar* cv) {
        return cv->getTableKey() == table_key;
      });
  return it == geo_args.end() ? nullptr : *it;
}

static std::unordered_map<SQLTypes, cost_t> GEO_TYPE_COSTS{{kPOINT, 60},
                                                           {kARRAY, 60},
                                                           {kLINESTRING, 70},
                                                           {kPOLYGON, 80},
                                                           {kMULTIPOLYGON, 90}};

static bool force_table_reordering_st_contain_func(std::string_view target_func_name) {
  return std::any_of(BoundingBoxIntersectJoinSupportedFunction::
                         ST_CONTAIN_FORCE_TABLE_REORDERING_TARGET_FUNC.begin(),
                     BoundingBoxIntersectJoinSupportedFunction::
                         ST_CONTAIN_FORCE_TABLE_REORDERING_TARGET_FUNC.end(),
                     [target_func_name](std::string_view func_name) {
                       return target_func_name == func_name;
                     });
}

static bool force_table_reordering_st_intersects_func(std::string_view target_func_name) {
  return std::any_of(BoundingBoxIntersectJoinSupportedFunction::
                         ST_INTERSECTS_FORCE_TABLE_REORDERING_TARGET_FUNC.begin(),
                     BoundingBoxIntersectJoinSupportedFunction::
                         ST_INTERSECTS_FORCE_TABLE_REORDERING_TARGET_FUNC.end(),
                     [target_func_name](std::string_view func_name) {
                       return target_func_name == func_name;
                     });
}

bool should_force_table_reordering(shared::TableKey const& inner_arg_key,
                                   SQLTypes const inner_type,
                                   shared::TableKey const& outer_arg_key,
                                   SQLTypes const outer_type,
                                   std::string const& geo_func_name,
                                   const std::vector<InputTableInfo>& table_infos) {
  // if, |R| > |S|
  // case-1: SELECT ... FROM R, S WHERE ST_...(R.c, S.c);
  // case-2: SELECT ... FROM R, S WHERE ST_...(S.c, R.c);
  // case-3: SELECT ... FROM S, R WHERE ST_...(R.c, S.c);
  // case-4: SELECT ... FROM S, R WHERE ST_...(S.c, R.c);
  auto const inner_poly_outer_pt_pair =
      shared::is_any<kPOLYGON, kMULTIPOLYGON>(inner_type) && outer_type == kPOINT;
  auto const outer_poly_inner_pt_pair =
      shared::is_any<kPOLYGON, kMULTIPOLYGON>(outer_type) && inner_type == kPOINT;
  auto const force_swap_st_contains =
      force_table_reordering_st_contain_func(geo_func_name);
  auto const force_swap_st_intersects =
      force_table_reordering_st_intersects_func(geo_func_name);
  size_t inner_idx = 0;
  size_t outer_idx = 0;
  for (size_t i = 0; i < table_infos.size(); i++) {
    if (table_infos[i].table_key == inner_arg_key) {
      inner_idx = i;
    } else if (table_infos[i].table_key == outer_arg_key) {
      outer_idx = i;
    }
  }
  size_t first_listed_idx = std::min(inner_idx, outer_idx);
  size_t first_listed_card = table_infos[first_listed_idx].info.getNumTuples();
  size_t last_listed_idx = std::max(inner_idx, outer_idx);
  size_t last_listed_card = table_infos[last_listed_idx].info.getNumTuples();
  if (first_listed_card > last_listed_card) {
    if (inner_arg_key == table_infos[first_listed_idx].table_key) {
      // case 1
      return inner_poly_outer_pt_pair &&
             (force_swap_st_contains || force_swap_st_intersects);
    } else {
      // case 2
      CHECK_EQ(outer_arg_key, table_infos[first_listed_idx].table_key);
      return outer_poly_inner_pt_pair && force_swap_st_intersects;
    }
  }
  return false;
}

// Returns a lhs/rhs cost for the given qualifier. Must be strictly greater than 0.
// todo (yoonmin): compute the cost of inner join edge and outer join edge
// Currently, we set 100 for inner join and 200 for loop join
// for geometries, we use types of geometries as its cost factor
std::tuple<cost_t, cost_t, InnerQualDecision> get_join_qual_cost(
    const Analyzer::Expr* qual,
    const std::vector<InputTableInfo>& table_infos,
    const Executor* executor) {
  if (executor) {
    GeospatialFunctionFinder geo_func_finder;
    geo_func_finder.visit(qual);
    if (auto table_key_pair = geo_func_finder.getJoinTableKeyPair()) {
      auto const inner_table_key = (*table_key_pair).inner_table_key;
      auto const outer_table_key = (*table_key_pair).outer_table_key;
      // try to find a chance to swap tables in the binary join
      // note that self-join does not need to be swapped
      CHECK_NE(inner_table_key, outer_table_key);
      const auto& target_geo_func_name = geo_func_finder.getGeoFunctionName();
      auto const inner_table_cardinality =
          get_table_cardinality(inner_table_key, executor);
      auto const outer_table_cardinality =
          get_table_cardinality(outer_table_key, executor);
      auto inner_qual_decision = inner_table_cardinality > outer_table_cardinality
                                     ? InnerQualDecision::LHS
                                     : InnerQualDecision::RHS;
      // detect the case when table reordering by cardinality incurs unexpected overhead
      // i.e., SELECT ... FROM R, S where ST_Interesects(S.poly, R.pt) where |R| > |S|
      // but |R| is not that larger than |S|, i.e., |R| / |S| < 10.0
      // in this case, it might be better if keeping the existing ordering
      // to exploit bounding-box intersection w/ hash join framework instead of loop join
      const auto inner_cv = get_geo_cv(geo_func_finder.getGeoArgCvs(), inner_table_key);
      CHECK(inner_cv);
      bool needs_table_reordering = inner_table_cardinality < outer_table_cardinality;
      const auto outer_inner_card_ratio =
          outer_table_cardinality / static_cast<double>(inner_table_cardinality);
      if (BoundingBoxIntersectJoinSupportedFunction::is_point_poly_rewrite_target_func(
              target_geo_func_name) ||
          BoundingBoxIntersectJoinSupportedFunction::is_poly_point_rewrite_target_func(
              target_geo_func_name)) {
        // the goal of this is to maximize the chance of using bounding-box intersection
        // (aka. bbox-intersection) to filter out unnecessary pair of geometries before
        // computing geo function to achieve this, point column has zero for its rte_idx
        // (so we build a hash table based on poly column which has rte_idx = 1) but if
        // it's cardinality is smaller than that of polygon table more than 10x we try to
        // fall back to loop join to avoid too expensive bbox-intersection
        if (inner_cv->get_rte_idx() == 0 &&
            (inner_cv->get_type_info().get_type() == kPOINT)) {
          // outer is poly, and we can use hash join w/ bbox-intersection
          if (needs_table_reordering && outer_inner_card_ratio > 10.0 &&
              inner_table_cardinality < 10000) {
            // but current pt table is small enough and hash table is larger than
            // the pt table at least 10 times, then we fall back to loop join
            // to avoid too expensive hash join
            // so let's try to set inner table as poly table to invalidate
            // rte index requirement
            VLOG(1) << "Force loop-join to avoid unexpected overhead of building large "
                       "hash table";
            return {200, 200, InnerQualDecision::RHS};
          } else {
            // otherwise, try to keep the existing ordering
            return {180, 190, InnerQualDecision::IGNORE};
          }
        } else {
          // poly is the inner table, so we need to reorder tables to use
          // bbox-intersection
          auto const geo_func_name = geo_func_finder.getGeoFunctionName();
          const auto outer_cv =
              get_geo_cv(geo_func_finder.getGeoArgCvs(), outer_table_key);
          CHECK(outer_cv);
          auto const inner_type = inner_cv->get_type_info().get_type();
          auto const outer_type = outer_cv->get_type_info().get_type();
          if (!needs_table_reordering && should_force_table_reordering(inner_table_key,
                                                                       inner_type,
                                                                       outer_table_key,
                                                                       outer_type,
                                                                       geo_func_name,
                                                                       table_infos)) {
            VLOG(1) << "Force reordering tables to enable a hash join for "
                    << geo_func_name;
            // let's reorder them regardless of table cardinality to build a hash table on
            // polygon side which can exploit bounding-box intersection
            return {190, 180, InnerQualDecision::RHS};
          }
          if (needs_table_reordering) {
            // outer point table is larger than inner poly table, so let's reorder them
            // by table cardinality
            VLOG(1) << "Try to reorder tables based on table cardinality";
            return {200, 200, InnerQualDecision::RHS};
          } else {
            // otherwise, try to keep the existing ordering
            return {180, 190, InnerQualDecision::IGNORE};
          }
        }
      }
      // rest of bbox-intersection-available and unavailable geo functions
      // can reach here, and they are reordered by table cardinality
      // specifically, bbox-intersection-available geo join functions are satisfied one of
      // followings: ST_INTERSECTSBOX_sv and is_poly_mpoly_rewrite_target_func we can use
      // bbox-intersection hash join for those functions regardless of table ordering see
      // rewrite_bbox_intersection function in ExpressionRewrite.cpp
      VLOG(2) << "Detect geo join operator, initial_inner_table(db_id: "
              << inner_table_key.db_id << ", table_id: " << inner_table_key.table_id
              << "), cardinality: " << inner_table_cardinality
              << "), initial_outer_table(db_id: " << outer_table_key.db_id
              << ", table_id: " << outer_table_key.table_id
              << "), cardinality: " << outer_table_cardinality
              << "), inner_qual_decision: " << inner_qual_decision;
      return {200, 200, inner_qual_decision};

    } else {
      // let's fall back to the old strategy by ordering tables by types of geometries
      const auto func_oper = dynamic_cast<const Analyzer::FunctionOper*>(qual);
      if (func_oper) {
        std::vector<SQLTypes> geo_types_for_func;
        for (size_t i = 0; i < func_oper->getArity(); i++) {
          const auto arg_expr = func_oper->getArg(i);
          const auto& ti = arg_expr->get_type_info();
          if (ti.is_geometry() || is_constructed_point(arg_expr)) {
            geo_types_for_func.push_back(ti.get_type());
          }
        }
        std::regex geo_func_regex("ST_[\\w]*");
        std::smatch geo_func_match;
        const auto& func_name = func_oper->getName();
        if (geo_types_for_func.size() == 2 &&
            std::regex_match(func_name, geo_func_match, geo_func_regex)) {
          const auto rhs_cost = GEO_TYPE_COSTS[geo_types_for_func[0]];
          const auto lhs_cost = GEO_TYPE_COSTS[geo_types_for_func[1]];
          return {lhs_cost, rhs_cost, InnerQualDecision::IGNORE};
        }
        return {200, 200, InnerQualDecision::IGNORE};
      }
    }
  }

  const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(qual);
  if (!bin_oper || !IS_EQUIVALENCE(bin_oper->get_optype())) {
    return {200, 200, InnerQualDecision::IGNORE};
  }
  InnerQualDecision inner_qual_decision = InnerQualDecision::UNKNOWN;
  if (executor) {
    try {
      const auto normalized_bin_oper =
          HashJoin::normalizeColumnPairs(bin_oper, executor->getTemporaryTables());
      const auto& inner_outer = normalized_bin_oper.first;
      // normalization success, so we need to figure out which cv becomes an inner
      auto lhs = bin_oper->get_left_operand();
      if (auto lhs_tuple = dynamic_cast<const Analyzer::ExpressionTuple*>(
              bin_oper->get_left_operand())) {
        lhs = lhs_tuple->getTuple().front().get();
      }
      CHECK(lhs);
      if (lhs == inner_outer.front().first) {
        inner_qual_decision = InnerQualDecision::LHS;
      } else if (lhs == inner_outer.front().second) {
        inner_qual_decision = InnerQualDecision::RHS;
      }
    } catch (const HashJoinFail& e) {
      return {200, 200, e.inner_qual_decision};
    } catch (...) {
      return {200, 200, inner_qual_decision};
    }
  }
  return {100, 100, inner_qual_decision};
}

// Builds a graph with nesting levels as nodes and join condition costs as edges.
std::vector<std::map<node_t, cost_t>> build_join_cost_graph(
    const JoinQualsPerNestingLevel& left_deep_join_quals,
    const std::vector<InputTableInfo>& table_infos,
    const Executor* executor,
    std::vector<std::map<node_t, InnerQualDecision>>& qual_detection_res) {
  CHECK_EQ(left_deep_join_quals.size() + 1, table_infos.size());
  std::vector<std::map<node_t, cost_t>> join_cost_graph(table_infos.size());
  AllRangeTableIndexVisitor visitor;
  // Build the constraints graph: nodes are nest levels, edges are the existence of
  // qualifiers between levels.
  for (const auto& current_level_join_conditions : left_deep_join_quals) {
    for (const auto& qual : current_level_join_conditions.quals) {
      std::set<int> qual_nest_levels = visitor.visit(qual.get());
      if (qual_nest_levels.size() != 2) {
        continue;
      }
      int lhs_nest_level = *qual_nest_levels.begin();
      CHECK_GE(lhs_nest_level, 0);
      qual_nest_levels.erase(qual_nest_levels.begin());
      int rhs_nest_level = *qual_nest_levels.begin();
      CHECK_GE(rhs_nest_level, 0);
      // Get the {lhs, rhs} cost for the qual
      const auto qual_costing = get_join_qual_cost(qual.get(), table_infos, executor);
      qual_detection_res[lhs_nest_level][rhs_nest_level] = std::get<2>(qual_costing);
      qual_detection_res[rhs_nest_level][lhs_nest_level] = std::get<2>(qual_costing);
      const auto edge_it = join_cost_graph[lhs_nest_level].find(rhs_nest_level);
      auto rhs_cost = std::get<1>(qual_costing);
      if (edge_it == join_cost_graph[lhs_nest_level].end() ||
          edge_it->second > rhs_cost) {
        auto lhs_cost = std::get<0>(qual_costing);
        join_cost_graph[lhs_nest_level][rhs_nest_level] = rhs_cost;
        join_cost_graph[rhs_nest_level][lhs_nest_level] = lhs_cost;
      }
    }
  }
  return join_cost_graph;
}

// Tracks dependencies between nodes.
class SchedulingDependencyTracking {
 public:
  SchedulingDependencyTracking(const size_t node_count) : inbound_(node_count) {}

  // Add a from -> to dependency.
  void addEdge(const node_t from, const node_t to) { inbound_[to].insert(from); }

  // Removes from's outbound dependencies.
  void removeNode(const node_t from) {
    for (auto& inbound_for_node : inbound_) {
      inbound_for_node.erase(from);
    }
  }

  // Returns the set of all nodes without dependencies.
  std::unordered_set<node_t> getRoots() const {
    std::unordered_set<node_t> roots;
    for (node_t candidate = 0; candidate < inbound_.size(); ++candidate) {
      if (inbound_[candidate].empty()) {
        roots.insert(candidate);
      }
    }
    return roots;
  }

 private:
  std::vector<std::unordered_set<node_t>> inbound_;
};

// The tree edge for traversal of the cost graph.
struct TraversalEdge {
  node_t nest_level;
  cost_t join_cost;
};

// Builds dependency tracking based on left joins
SchedulingDependencyTracking build_dependency_tracking(
    const JoinQualsPerNestingLevel& left_deep_join_quals,
    const std::vector<std::map<node_t, cost_t>>& join_cost_graph) {
  SchedulingDependencyTracking dependency_tracking(left_deep_join_quals.size() + 1);
  // Add directed graph edges for left join dependencies.
  // See also start_it inside traverse_join_cost_graph(). These
  // edges prevent start_it from pointing to a table with a
  // left join dependency on another table.
  for (size_t level_idx = 0; level_idx < left_deep_join_quals.size(); ++level_idx) {
    if (left_deep_join_quals[level_idx].type == JoinType::LEFT) {
      dependency_tracking.addEdge(level_idx, level_idx + 1);
    }
  }
  return dependency_tracking;
}

// Do a breadth-first traversal of the cost graph. This avoids scheduling a nest level
// before the ones which constraint it are scheduled and it favors equi joins over loop
// joins.
std::vector<node_t> traverse_join_cost_graph(
    const std::vector<std::map<node_t, cost_t>>& join_cost_graph,
    const std::vector<InputTableInfo>& table_infos,
    const std::function<bool(const node_t lhs_nest_level, const node_t rhs_nest_level)>&
        compare_node,
    const std::function<bool(const TraversalEdge&, const TraversalEdge&)>& compare_edge,
    const JoinQualsPerNestingLevel& left_deep_join_quals,
    std::vector<std::map<node_t, InnerQualDecision>>& qual_normalization_res) {
  std::vector<node_t> all_nest_levels(table_infos.size());
  std::iota(all_nest_levels.begin(), all_nest_levels.end(), 0);
  std::vector<node_t> input_permutation;
  std::unordered_set<node_t> visited;
  auto dependency_tracking =
      build_dependency_tracking(left_deep_join_quals, join_cost_graph);
  auto schedulable_node = [&dependency_tracking, &visited](const node_t node) {
    const auto nodes_ready = dependency_tracking.getRoots();
    return nodes_ready.find(node) != nodes_ready.end() &&
           visited.find(node) == visited.end();
  };
  while (visited.size() < table_infos.size()) {
    // Filter out nest levels which are already visited or have pending dependencies.
    std::vector<node_t> remaining_nest_levels;
    std::copy_if(all_nest_levels.begin(),
                 all_nest_levels.end(),
                 std::back_inserter(remaining_nest_levels),
                 schedulable_node);
    CHECK(!remaining_nest_levels.empty());
    // Start with the table with most tuples.
    const auto start_it = std::max_element(
        remaining_nest_levels.begin(), remaining_nest_levels.end(), compare_node);
    CHECK(start_it != remaining_nest_levels.end());
    std::priority_queue<TraversalEdge, std::vector<TraversalEdge>, decltype(compare_edge)>
        worklist(compare_edge);
    //  look at all edges, compare the
    //  cost of our edge vs theirs, and pick the best start edge
    node_t start = *start_it;
    // we adaptively switch the inner and outer when we have a chance to exploit
    // hash join framework for a query with a single binary join
    TraversalEdge start_edge{start, 0};

    // when we have a single binary join in the query, we can analyze the qual and apply
    // more smart table reordering logic that maximizes the chance of exploiting hash join
    // todo (yoonmin) : generalize this for an arbitrary join pipeline
    if (remaining_nest_levels.size() == 2 && qual_normalization_res[start].size() == 1) {
      auto inner_qual_decision = qual_normalization_res[start].begin()->second;
      auto join_qual = left_deep_join_quals.begin()->quals;
      using ColvarSet =
          std::set<const Analyzer::ColumnVar*,
                   bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)>;

      auto set_new_rte_idx = [](ColvarSet& cv_set, int new_rte) {
        std::for_each(
            cv_set.begin(), cv_set.end(), [new_rte](const Analyzer::ColumnVar* cv) {
              const_cast<Analyzer::ColumnVar*>(cv)->set_rte_idx(new_rte);
            });
      };

      // IGNORE: use the existing table reordering logic
      // KEEP: return the existing table permutation and related cvs (column variables)
      // SWAP: change the starting table of the table reordering logic and relevant
      // columns' rte index
      enum class Decision { IGNORE, KEEP, SWAP };

      auto analyze_join_qual = [&start,
                                &remaining_nest_levels,
                                &inner_qual_decision,
                                &table_infos,
                                compare_node](const std::shared_ptr<Analyzer::Expr>& lhs,
                                              ColvarSet& lhs_colvar_set,
                                              const std::shared_ptr<Analyzer::Expr>& rhs,
                                              ColvarSet& rhs_colvar_set) {
        if (!lhs || !rhs || lhs_colvar_set.empty() || rhs_colvar_set.empty()) {
          return std::make_pair(Decision::IGNORE, start);
        }

        auto alternative_it = std::find_if(
            remaining_nest_levels.begin(),
            remaining_nest_levels.end(),
            [start](const size_t nest_level) { return start != nest_level; });
        CHECK(alternative_it != remaining_nest_levels.end());
        auto alternative_rte = *alternative_it;

        Decision decision = Decision::IGNORE;
        // inner col's rte should be larger than outer col
        int inner_rte = -1;
        int outer_rte = -1;
        bool is_outer_col_valid = false;
        auto check_expr_is_valid_col = [&is_outer_col_valid](const Analyzer::Expr* expr) {
          if (auto expr_tuple = dynamic_cast<const Analyzer::ExpressionTuple*>(expr)) {
            for (auto& inner_expr : expr_tuple->getTuple()) {
              auto cv_from_expr =
                  HashJoin::getHashJoinColumn<Analyzer::ColumnVar>(inner_expr.get());
              if (!cv_from_expr) {
                is_outer_col_valid = false;
                return;
              }
            }
          } else {
            auto cv_from_expr = HashJoin::getHashJoinColumn<Analyzer::ColumnVar>(expr);
            if (!cv_from_expr) {
              is_outer_col_valid = false;
              return;
            }
          }
          is_outer_col_valid = true;
        };
        if (inner_qual_decision == InnerQualDecision::LHS) {
          inner_rte = (*lhs_colvar_set.begin())->get_rte_idx();
          outer_rte = (*rhs_colvar_set.begin())->get_rte_idx();
          check_expr_is_valid_col(rhs.get());
        } else if (inner_qual_decision == InnerQualDecision::RHS) {
          inner_rte = (*rhs_colvar_set.begin())->get_rte_idx();
          outer_rte = (*lhs_colvar_set.begin())->get_rte_idx();
          check_expr_is_valid_col(lhs.get());
        }
        if (inner_rte >= 0 && outer_rte >= 0) {
          const auto inner_cardinality =
              table_infos[inner_rte].info.getNumTuplesUpperBound();
          const auto outer_cardinality =
              table_infos[outer_rte].info.getNumTuplesUpperBound();
          if (inner_cardinality > g_trivial_loop_join_threshold) {
            if (inner_rte == static_cast<int>(start)) {
              // inner is driving the join loop but also has a valid join column
              // which is available for building a hash table
              // but ignore swapping when inner's cardinality is larger than that of
              // outer's / otherwise swap inner and outer (to use the valid inner)
              decision = is_outer_col_valid && inner_cardinality > outer_cardinality
                             ? Decision::IGNORE
                             : Decision::SWAP;
            } else {
              CHECK_EQ(inner_rte, static_cast<int>(alternative_rte));
              // now, a valid inner column is outer table
              if (compare_node(inner_rte, start)) {
                // but outer table is larger than the current inner
                // so we can exploit the existing table reordering logic
                decision = Decision::IGNORE;
              } else {
                // and outer table is smaller than the current inner
                // so we do not need to reorder the table starting from the inner
                decision = Decision::KEEP;
              }
            }
          }
        }

        if (decision == Decision::KEEP) {
          return std::make_pair(decision, start);
        } else if (decision == Decision::SWAP) {
          return std::make_pair(decision, alternative_rte);
        }
        return std::make_pair(Decision::IGNORE, start);
      };

      auto collect_colvars = [](const std::shared_ptr<Analyzer::Expr> expr,
                                ColvarSet& cv_set) {
        expr->collect_column_var(cv_set, false);
      };

      auto adjust_reordering_logic = [&start, &start_edge, &start_it, set_new_rte_idx](
                                         Decision decision,
                                         int alternative_rte,
                                         ColvarSet& lhs_colvar_set,
                                         ColvarSet& rhs_colvar_set) {
        CHECK(decision == Decision::SWAP);
        start = alternative_rte;
        set_new_rte_idx(lhs_colvar_set, alternative_rte);
        set_new_rte_idx(rhs_colvar_set, *start_it);
        start_edge.join_cost = 0;
        start_edge.nest_level = start;
      };

      auto bin_op = dynamic_cast<Analyzer::BinOper*>(join_qual.begin()->get());
      if (bin_op) {
        auto lhs = bin_op->get_own_left_operand();
        auto rhs = bin_op->get_own_right_operand();
        if (auto lhs_exp = dynamic_cast<Analyzer::ExpressionTuple*>(lhs.get())) {
          // retrieve the decision and info for adjusting reordering by referring the
          // first cv and apply them to the rest of cvs
          auto rhs_exp = dynamic_cast<Analyzer::ExpressionTuple*>(rhs.get());
          CHECK(rhs_exp);
          auto& lhs_exprs = lhs_exp->getTuple();
          auto& rhs_exprs = rhs_exp->getTuple();
          CHECK_EQ(lhs_exprs.size(), rhs_exprs.size());
          for (size_t i = 0; i < lhs_exprs.size(); ++i) {
            Decision decision{Decision::IGNORE};
            int alternative_rte_idx = -1;
            ColvarSet lhs_colvar_set(Analyzer::ColumnVar::colvar_comp);
            ColvarSet rhs_colvar_set(Analyzer::ColumnVar::colvar_comp);
            collect_colvars(lhs_exprs.at(i), lhs_colvar_set);
            collect_colvars(rhs_exprs.at(i), rhs_colvar_set);
            if (i == 0) {
              auto investigation_res =
                  analyze_join_qual(lhs, lhs_colvar_set, rhs, rhs_colvar_set);
              decision = investigation_res.first;
              if (decision == Decision::KEEP) {
                return remaining_nest_levels;
              }
              alternative_rte_idx = investigation_res.second;
            }
            if (decision == Decision::SWAP) {
              adjust_reordering_logic(
                  decision, alternative_rte_idx, lhs_colvar_set, rhs_colvar_set);
            }
          }
        } else {
          ColvarSet lhs_colvar_set(Analyzer::ColumnVar::colvar_comp);
          ColvarSet rhs_colvar_set(Analyzer::ColumnVar::colvar_comp);
          collect_colvars(lhs, lhs_colvar_set);
          collect_colvars(rhs, rhs_colvar_set);
          auto investigation_res =
              analyze_join_qual(lhs, lhs_colvar_set, rhs, rhs_colvar_set);
          if (investigation_res.first == Decision::KEEP) {
            return remaining_nest_levels;
          } else if (investigation_res.first == Decision::SWAP) {
            adjust_reordering_logic(investigation_res.first,
                                    investigation_res.second,
                                    lhs_colvar_set,
                                    rhs_colvar_set);
          }
        }
      }
    }

    VLOG(2) << "Table reordering starting with nest level " << start;
    for (const auto& graph_edge : join_cost_graph[*start_it]) {
      const node_t succ = graph_edge.first;
      if (!schedulable_node(succ)) {
        continue;
      }
      const TraversalEdge succ_edge{succ, graph_edge.second};
      for (const auto& successor_edge : join_cost_graph[succ]) {
        if (successor_edge.first == start) {
          start_edge.join_cost = successor_edge.second;
          // lhs cost / num tuples less than rhs cost if compare edge is true, swap nest
          // levels
          if (compare_edge(start_edge, succ_edge)) {
            VLOG(2) << "Table reordering changing start nest level from " << start
                    << " to " << succ;
            start = succ;
            start_edge = succ_edge;
          }
        }
      }
    }
    VLOG(2) << "Table reordering picked start nest level " << start << " with cost "
            << start_edge.join_cost;
    CHECK_EQ(start, start_edge.nest_level);
    worklist.push(start_edge);
    const auto it_ok = visited.insert(start);
    CHECK(it_ok.second);
    while (!worklist.empty()) {
      // Extract a node and add it to the permutation.
      TraversalEdge crt = worklist.top();
      worklist.pop();
      VLOG(1) << "Insert input permutation, idx: " << input_permutation.size()
              << ", nest_level: " << crt.nest_level;
      input_permutation.push_back(crt.nest_level);
      dependency_tracking.removeNode(crt.nest_level);
      // Add successors which are ready and not visited yet to the queue.
      for (const auto& graph_edge : join_cost_graph[crt.nest_level]) {
        const node_t succ = graph_edge.first;
        if (!schedulable_node(succ)) {
          continue;
        }
        worklist.push(TraversalEdge{succ, graph_edge.second});
        const auto it_ok = visited.insert(succ);
        CHECK(it_ok.second);
      }
    }
  }
  return input_permutation;
}

}  // namespace

std::vector<node_t> get_node_input_permutation(
    const JoinQualsPerNestingLevel& left_deep_join_quals,
    const std::vector<InputTableInfo>& table_infos,
    const Executor* executor) {
  std::vector<std::map<node_t, InnerQualDecision>> qual_normalization_res(
      table_infos.size());
  const auto join_cost_graph = build_join_cost_graph(
      left_deep_join_quals, table_infos, executor, qual_normalization_res);
  // Use the number of tuples in each table to break ties in BFS.
  const auto compare_node = [&table_infos](const node_t lhs_nest_level,
                                           const node_t rhs_nest_level) {
    return table_infos[lhs_nest_level].info.getNumTuplesUpperBound() <
           table_infos[rhs_nest_level].info.getNumTuplesUpperBound();
  };
  const auto compare_edge = [&compare_node](const TraversalEdge& lhs_edge,
                                            const TraversalEdge& rhs_edge) {
    // Only use the number of tuples as a tie-breaker, if costs are equal.
    if (lhs_edge.join_cost == rhs_edge.join_cost) {
      return compare_node(lhs_edge.nest_level, rhs_edge.nest_level);
    }
    return lhs_edge.join_cost > rhs_edge.join_cost;
  };
  return traverse_join_cost_graph(join_cost_graph,
                                  table_infos,
                                  compare_node,
                                  compare_edge,
                                  left_deep_join_quals,
                                  qual_normalization_res);
}
