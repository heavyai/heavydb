/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <numeric>

#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/RangeTableIndexVisitor.h"

extern bool g_enable_filter_push_down;
extern float g_filter_push_down_max_selectivity;
extern size_t g_filter_push_down_selectivity_override_max_passing_num_rows;

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
  const float selectivity;
  const size_t num_rows_passed;

  bool isFilterSelectiveEnough() const {
    return selectivity < g_filter_push_down_max_selectivity ||
           num_rows_passed < g_filter_push_down_selectivity_override_max_passing_num_rows;
  }
};

bool to_gather_info_for_filter_selectivity(
    const std::vector<InputTableInfo>& table_infos);

std::vector<PushedDownFilterInfo> find_push_down_filters(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<size_t>& input_permutation,
    const std::vector<size_t>& left_deep_join_input_sizes);
