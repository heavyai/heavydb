/*
 * Copyright 2022 Intel Corporation.
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

#include "QueryEngine/ColRangeInfo.h"

int64_t ColRangeInfo::getBucketedCardinality() const {
  checked_int64_t crt_col_cardinality = checked_int64_t(max) - checked_int64_t(min);
  if (bucket) {
    crt_col_cardinality /= bucket;
  }
  return static_cast<int64_t>(crt_col_cardinality + (1 + (has_nulls ? 1 : 0)));
}

ColRangeInfo get_expr_range_info(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<InputTableInfo>& query_infos,
                                 const hdk::ir::Expr* expr,
                                 Executor* executor) {
  if (!expr) {
    return {QueryDescriptionType::Projection, 0, 0, 0, false};
  }

  const auto expr_range = getExpressionRange(
      expr, query_infos, executor, boost::make_optional(ra_exe_unit.simple_quals));
  switch (expr_range.getType()) {
    case ExpressionRangeType::Integer: {
      if (expr_range.getIntMin() > expr_range.getIntMax()) {
        return {
            QueryDescriptionType::GroupByBaselineHash, 0, -1, 0, expr_range.hasNulls()};
      }
      return {QueryDescriptionType::GroupByPerfectHash,
              expr_range.getIntMin(),
              expr_range.getIntMax(),
              expr_range.getBucket(),
              expr_range.hasNulls()};
    }
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double: {
      if (expr_range.getFpMin() > expr_range.getFpMax()) {
        return {
            QueryDescriptionType::GroupByBaselineHash, 0, -1, 0, expr_range.hasNulls()};
      }
      return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
    }
    case ExpressionRangeType::Invalid:
      return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
    default:
      CHECK(false);
  }
  CHECK(false);
  return {QueryDescriptionType::NonGroupedAggregate, 0, 0, 0, false};
}
