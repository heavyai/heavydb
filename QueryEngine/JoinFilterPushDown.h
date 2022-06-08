/*
 * Copyright 2019 OmniSci, Inc.
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

#include <cstddef>
#include <numeric>

#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/RangeTableIndexVisitor.h"
#include "Shared/Config.h"

/**
 * The main purpose of this struct is to help identify the selected filters
 * in Calcite just by looking at the query.
 * input_prev, intput_start, and input_next represent the beginning column
 * index for the previous, current and next tables in the query string respectively.
 * TODO(Saman): should add some encoding based on the structure of the whole query
 * to be able to uniquely identify fitlers in multi-step queries and/or subqueries.
 */
struct PushedDownFilterInfo {
  std::vector<std::shared_ptr<Analyzer::Expr>> filter_expressions;
  size_t input_prev;
  size_t input_start;
  size_t input_next;
};

/**
 * Statistics stored for filters with respect to a table, so that
 * selective filters can be pushed down in join operations.
 */
struct FilterSelectivity {
  const bool is_valid;
  const float fraction_passing;
  const size_t total_rows_upper_bound;

  size_t getRowsPassingUpperBound() const {
    return fraction_passing * total_rows_upper_bound;
  }
  bool isFilterSelectiveEnough(const FilterPushdownConfig& config) const {
    auto low_frac_threshold = (config.low_frac >= 0.0) ? std::min(config.low_frac, 1.0f)
                                                       : kFractionPassingLowThreshold;
    auto high_frac_threshold = (config.high_frac >= 0.0)
                                   ? std::min(config.high_frac, 1.0f)
                                   : kFractionPassingHighThreshold;
    auto rows_ubound_threshold = (config.passing_row_ubound > 0)
                                     ? config.passing_row_ubound
                                     : kRowsPassingUpperBoundThreshold;
    return (fraction_passing < low_frac_threshold ||
            (fraction_passing < high_frac_threshold &&
             getRowsPassingUpperBound() < rows_ubound_threshold));
  }

  static constexpr float kFractionPassingLowThreshold = 0.1;
  static constexpr float kFractionPassingHighThreshold = 0.5;
  static constexpr size_t kRowsPassingUpperBoundThreshold = 4000000;
};

bool to_gather_info_for_filter_selectivity(
    const std::vector<InputTableInfo>& table_infos);

std::vector<PushedDownFilterInfo> find_push_down_filters(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<size_t>& input_permutation,
    const std::vector<size_t>& left_deep_join_input_sizes);
