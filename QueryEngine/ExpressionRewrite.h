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

#include <llvm/IR/Value.h>
#include <boost/optional.hpp>
#include <list>
#include <memory>
#include <vector>

#include "Analyzer/Analyzer.h"
#include "RelAlgExecutionUnit.h"

namespace Analyzer {

class Expr;

class InValues;

}  // namespace Analyzer

class InputColDescriptor;

// Rewrites an OR tree where leaves are equality compare against literals.
Analyzer::ExpressionPtr rewrite_expr(const Analyzer::Expr*);

// Rewrites array elements that are strings to be dict encoded transient literals
Analyzer::ExpressionPtr rewrite_array_elements(const Analyzer::Expr*);

// Rewrite a FunctionOper to an AND between a BinOper and the FunctionOper if the
// FunctionOper is supported for bounding box intersect hash joins
struct BoundingBoxIntersectJoinConjunction {
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  std::list<std::shared_ptr<Analyzer::Expr>> join_quals;
};

struct BoundingBoxIntersectJoinTranslationInfo {
  JoinQualsPerNestingLevel join_quals;
  bool has_bbox_intersect_join{false};
  bool is_reordered{false};
};

struct BoundingBoxIntersectJoinTranslationResult {
  bool swap_arguments{false};
  std::optional<BoundingBoxIntersectJoinConjunction> converted_bbox_intersect_join_info{
      std::nullopt};

  static BoundingBoxIntersectJoinTranslationResult createEmptyResult() {
    return {false, std::nullopt};
  }
};

enum class BoundingBoxIntersectJoinRewriteType {
  BBOX_INTERSECT_JOIN,
  RANGE_JOIN,
  UNKNOWN
};

struct BoundingBoxIntersectJoinSupportedFunction {
  static constexpr std::string_view ST_CONTAINS_POLYGON_POINT_sv{
      "ST_Contains_Polygon_Point"};
  static constexpr std::string_view ST_CONTAINS_POLYGON_POLYGON_sv{
      "ST_Contains_Polygon_Polygon"};
  static constexpr std::string_view ST_CONTAINS_MULTIPOLYGON_MULTIPOLYGON_sv{
      "ST_Contains_MultiPolygon_MultiPolygon"};
  static constexpr std::string_view ST_CONTAINS_MULTIPOLYGON_POINT_sv{
      "ST_Contains_MultiPolygon_Point"};
  static constexpr std::string_view ST_INTERSECTS_POINT_POLYGON_sv{
      "ST_Intersects_Point_Polygon"};
  static constexpr std::string_view ST_INTERSECTS_POINT_MULTIPOLYGON_sv{
      "ST_Intersects_Point_MultiPolygon"};
  static constexpr std::string_view ST_INTERSECTS_POLYGON_POINT_sv{
      "ST_Intersects_Polygon_Point"};
  static constexpr std::string_view ST_INTERSECTS_POLYGON_POLYGON_sv{
      "ST_Intersects_Polygon_Polygon"};
  static constexpr std::string_view ST_INTERSECTS_POLYGON_MULTIPOLYGON_sv{
      "ST_Intersects_Polygon_MultiPolygon"};
  static constexpr std::string_view ST_INTERSECTS_MULTIPOLYGON_MULTIPOLYGON_sv{
      "ST_Intersects_MultiPolygon_MultiPolygon"};
  static constexpr std::string_view ST_INTERSECTS_MULTIPOLYGON_POLYGON_sv{
      "ST_Intersects_MultiPolygon_Polygon"};
  static constexpr std::string_view ST_INTERSECTS_MULTIPOLYGON_POINT_sv{
      "ST_Intersects_MultiPolygon_Point"};
  static constexpr std::string_view ST_APPROX_OVERLAPS_MULTIPOLYGON_POINT_sv{
      "ST_Approx_Overlaps_MultiPolygon_Point"};
  static constexpr std::string_view ST_INTERSECTSBOX_sv{"ST_IntersectsBox"};
  static constexpr std::string_view ST_DISTANCE_sv{"ST_Distance"};
  // compressed coords version
  static constexpr std::string_view ST_CCONTAINS_MULTIPOLYGON_POINT_sv{
      "ST_cContains_MultiPolygon_Point"};
  static constexpr std::string_view ST_CCONTAINS_POLYGON_POINT_sv{
      "ST_cContains_Polygon_Point"};
  static constexpr std::string_view ST_CINTERSECTS_POLYGON_POINT_sv{
      "ST_cIntersects_Polygon_Point"};
  static constexpr std::string_view ST_CINTERSECTS_MULTIPOLYGON_POINT_sv{
      "ST_cIntersects_MultiPolygon_Point"};
  static constexpr std::string_view ST_DWITHIN_POINT_POINT_sv{"ST_DWithin_Point_Point"};

  static constexpr std::array<std::string_view, 19> BBOX_INTERSECT_SUPPORTED_FUNC{
      ST_CONTAINS_POLYGON_POINT_sv,
      ST_CONTAINS_MULTIPOLYGON_POINT_sv,
      ST_CONTAINS_POLYGON_POLYGON_sv,
      ST_CONTAINS_MULTIPOLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_POINT_POLYGON_sv,
      ST_INTERSECTS_POINT_MULTIPOLYGON_sv,
      ST_INTERSECTS_POLYGON_POINT_sv,
      ST_INTERSECTS_POLYGON_POLYGON_sv,
      ST_INTERSECTS_POLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_MULTIPOLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_MULTIPOLYGON_POLYGON_sv,
      ST_INTERSECTS_MULTIPOLYGON_POINT_sv,
      ST_APPROX_OVERLAPS_MULTIPOLYGON_POINT_sv,
      ST_CCONTAINS_POLYGON_POINT_sv,
      ST_CCONTAINS_MULTIPOLYGON_POINT_sv,
      ST_CINTERSECTS_POLYGON_POINT_sv,
      ST_CINTERSECTS_MULTIPOLYGON_POINT_sv,
      ST_INTERSECTSBOX_sv,
      ST_DWITHIN_POINT_POINT_sv};

  static constexpr std::array<std::string_view, 6> MANY_TO_MANY_BBOX_INTERSECT_FUNC{
      ST_CONTAINS_POLYGON_POLYGON_sv,
      ST_CONTAINS_MULTIPOLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_POLYGON_POLYGON_sv,
      ST_INTERSECTS_POLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_MULTIPOLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_MULTIPOLYGON_POLYGON_sv};

  static constexpr std::array<std::string_view, 6> POLY_MPOLY_REWRITE_TARGET_FUNC{
      ST_CONTAINS_POLYGON_POLYGON_sv,
      ST_CONTAINS_MULTIPOLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_POLYGON_POLYGON_sv,
      ST_INTERSECTS_POLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_MULTIPOLYGON_MULTIPOLYGON_sv,
      ST_INTERSECTS_MULTIPOLYGON_POLYGON_sv};

  static constexpr std::array<std::string_view, 9> POLY_POINT_REWRITE_TARGET_FUNC{
      ST_CONTAINS_POLYGON_POINT_sv,
      ST_CONTAINS_MULTIPOLYGON_POINT_sv,
      ST_INTERSECTS_POLYGON_POINT_sv,
      ST_INTERSECTS_MULTIPOLYGON_POINT_sv,
      ST_APPROX_OVERLAPS_MULTIPOLYGON_POINT_sv,
      ST_CCONTAINS_POLYGON_POINT_sv,
      ST_CCONTAINS_MULTIPOLYGON_POINT_sv,
      ST_CINTERSECTS_POLYGON_POINT_sv,
      ST_CINTERSECTS_MULTIPOLYGON_POINT_sv};

  static constexpr std::array<std::string_view, 2> POINT_POLY_REWRITE_TARGET_FUNC{
      ST_INTERSECTS_POINT_POLYGON_sv,
      ST_INTERSECTS_POINT_MULTIPOLYGON_sv};

  static constexpr std::array<std::string_view, 2> RANGE_JOIN_REWRITE_TARGET_FUNC{
      ST_DISTANCE_sv,
      ST_DWITHIN_POINT_POINT_sv};

  static constexpr std::array<std::string_view, 4>
      ST_CONTAIN_FORCE_TABLE_REORDERING_TARGET_FUNC{ST_CONTAINS_POLYGON_POINT_sv,
                                                    ST_CONTAINS_MULTIPOLYGON_POINT_sv,
                                                    ST_CCONTAINS_POLYGON_POINT_sv,
                                                    ST_CCONTAINS_MULTIPOLYGON_POINT_sv};

  static constexpr std::array<std::string_view, 6>
      ST_INTERSECTS_FORCE_TABLE_REORDERING_TARGET_FUNC{
          ST_INTERSECTS_POINT_POLYGON_sv,
          ST_INTERSECTS_POINT_MULTIPOLYGON_sv,
          ST_INTERSECTS_POLYGON_POINT_sv,
          ST_INTERSECTS_MULTIPOLYGON_POINT_sv,
          ST_CINTERSECTS_POLYGON_POINT_sv,
          ST_CINTERSECTS_MULTIPOLYGON_POINT_sv};

  static bool is_bbox_intersect_supported_func(std::string_view target_func_name) {
    return std::any_of(
        BoundingBoxIntersectJoinSupportedFunction::BBOX_INTERSECT_SUPPORTED_FUNC.begin(),
        BoundingBoxIntersectJoinSupportedFunction::BBOX_INTERSECT_SUPPORTED_FUNC.end(),
        [target_func_name](std::string_view func_name) {
          return target_func_name == func_name;
        });
  }

  static bool is_many_to_many_func(std::string_view target_func_name) {
    return std::any_of(
        BoundingBoxIntersectJoinSupportedFunction::MANY_TO_MANY_BBOX_INTERSECT_FUNC
            .begin(),
        BoundingBoxIntersectJoinSupportedFunction::MANY_TO_MANY_BBOX_INTERSECT_FUNC.end(),
        [target_func_name](std::string_view func_name) {
          return target_func_name == func_name;
        });
  }

  static bool is_poly_mpoly_rewrite_target_func(std::string_view target_func_name) {
    return std::any_of(
        BoundingBoxIntersectJoinSupportedFunction::POLY_MPOLY_REWRITE_TARGET_FUNC.begin(),
        BoundingBoxIntersectJoinSupportedFunction::POLY_MPOLY_REWRITE_TARGET_FUNC.end(),
        [target_func_name](std::string_view func_name) {
          return target_func_name == func_name;
        });
  }

  static bool is_point_poly_rewrite_target_func(std::string_view target_func_name) {
    return std::any_of(
        BoundingBoxIntersectJoinSupportedFunction::POINT_POLY_REWRITE_TARGET_FUNC.begin(),
        BoundingBoxIntersectJoinSupportedFunction::POINT_POLY_REWRITE_TARGET_FUNC.end(),
        [target_func_name](std::string_view func_name) {
          return target_func_name == func_name;
        });
  }

  static bool is_poly_point_rewrite_target_func(std::string_view target_func_name) {
    return std::any_of(
        BoundingBoxIntersectJoinSupportedFunction::POLY_POINT_REWRITE_TARGET_FUNC.begin(),
        BoundingBoxIntersectJoinSupportedFunction::POLY_POINT_REWRITE_TARGET_FUNC.end(),
        [target_func_name](std::string_view func_name) {
          return target_func_name == func_name;
        });
  }

  static bool is_range_join_rewrite_target_func(std::string_view target_func_name) {
    return std::any_of(
        BoundingBoxIntersectJoinSupportedFunction::RANGE_JOIN_REWRITE_TARGET_FUNC.begin(),
        BoundingBoxIntersectJoinSupportedFunction::RANGE_JOIN_REWRITE_TARGET_FUNC.end(),
        [target_func_name](std::string_view func_name) {
          return target_func_name == func_name;
        });
  }
};

BoundingBoxIntersectJoinTranslationResult translate_bbox_intersect_with_reordering(
    const std::shared_ptr<Analyzer::Expr> expr,
    std::vector<InputDescriptor> const& input_descs,
    std::unordered_map<const RelAlgNode*, int> const& input_to_nest_level,
    std::vector<size_t> const& input_permutation,
    std::list<std::shared_ptr<const InputColDescriptor>>& input_col_desc,
    const BoundingBoxIntersectJoinRewriteType rewrite_type,
    Executor const* executor);

BoundingBoxIntersectJoinTranslationInfo convert_bbox_intersect_join(
    JoinQualsPerNestingLevel const& join_quals,
    std::vector<InputDescriptor>& input_descs,
    std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    std::vector<size_t>& input_permutation,
    std::list<std::shared_ptr<const InputColDescriptor>>& input_col_desc,
    Executor const* executor);

std::list<std::shared_ptr<Analyzer::Expr>> strip_join_covered_filter_quals(
    const std::list<std::shared_ptr<Analyzer::Expr>>& quals,
    const JoinQualsPerNestingLevel& join_quals);

std::shared_ptr<Analyzer::Expr> fold_expr(const Analyzer::Expr*);

bool self_join_not_covered_by_left_deep_tree(const Analyzer::ColumnVar* lhs,
                                             const Analyzer::ColumnVar* rhs,
                                             const int max_rte_covered);

const int get_max_rte_scan_table(
    std::unordered_map<int, llvm::Value*>& scan_idx_to_hash_pos);

size_t get_table_cardinality(shared::TableKey const& table_key, Executor const* executor);
