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

#include "../SystemFunctions/os/Shared/TableFunctionsCommon.hpp"
#include "TableFunctionsTesting.h"

/*
  This file contains testing UDTFs related to filter pushdown optimization.

  It is extremely long due to the combinatorial number of template functions
  that need to be instantiated. We hope to implement automatic casting of
  column inputs for UDTFs in the near future. When that is available, the
  number of template functions can be reduced dramatically.

  NOTE: This file currently has no GPU UDTFs. If any GPU UDTFs are
  added, it should be added to CUDA_TABLE_FUNCTION_FILES in CMakeLists.txt
 */

enum TFAggType { MIN, MAX };

template <typename T>
TEMPLATE_INLINE T get_min_or_max(const Column<T>& col, const TFAggType min_or_max) {
  const auto input_min_max = get_column_min_max(col);
  if (min_or_max == TFAggType::MIN) {
    return input_min_max.first;
  }
  return input_min_max.second;
}

template <typename T>
TEMPLATE_INLINE T get_min_or_max_union(const Column<T>& col1,
                                       const Column<T>& col2,
                                       const TFAggType min_or_max) {
  const auto input1_min_max = get_column_min_max(col1);
  const auto input2_min_max = get_column_min_max(col2);
  if (min_or_max == TFAggType::MIN) {
    return input1_min_max.first < input2_min_max.first ? input1_min_max.first
                                                       : input2_min_max.first;
  }
  return input1_min_max.second > input2_min_max.second ? input1_min_max.second
                                                       : input2_min_max.second;
}

#ifndef __CUDACC__

template <typename K, typename T, typename Z>
NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<K>& input_id,
                                const Column<T>& input_x,
                                const Column<T>& input_y,
                                const Column<Z>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<K>& output_id,
                                Column<T>& output_x,
                                Column<T>& output_y,
                                Column<Z>& output_z) {
  const std::string agg_type_str = agg_type.getString();
  const TFAggType min_or_max = agg_type_str == "MIN" ? TFAggType::MIN : TFAggType::MAX;
  mgr.set_output_row_size(1);
  output_row_count[0] = input_id.size();
  output_id[0] = get_min_or_max(input_id, min_or_max);
  output_x[0] = get_min_or_max(input_x, min_or_max);
  output_y[0] = get_min_or_max(input_y, min_or_max);
  output_z[0] = get_min_or_max(input_z, min_or_max);
  return 1;
}

// explicit instantiations

// 000
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<int32_t>& output_z);

// 001
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<int64_t>& output_z);

// 002
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<float>& output_z);

// 003
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<double>& output_z);

// 010
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<int32_t>& output_z);

// 011
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<int64_t>& output_z);

// 012
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<float>& output_z);

// 013
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<double>& output_z);

// 020
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<int32_t>& output_z);

// 021
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<int64_t>& output_z);

// 022
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<float>& output_z);

// 023
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<double>& output_z);

// 030
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<int32_t>& output_z);

// 031
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<int64_t>& output_z);

// 032
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<float>& output_z);

// 033
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int32_t>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int32_t>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<double>& output_z);

// 100
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<int32_t>& output_z);

// 101
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<int64_t>& output_z);

// 102
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<float>& output_z);

// 103
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<double>& output_z);

// 110
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<int32_t>& output_z);

// 111
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<int64_t>& output_z);

// 112
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<float>& output_z);

// 113
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<double>& output_z);

// 120
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<int32_t>& output_z);

// 121
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<int64_t>& output_z);

// 122
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<float>& output_z);

// 123
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<double>& output_z);

// 130
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<int32_t>& output_z);

// 131
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<int64_t>& output_z);

// 132
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<float>& output_z);

// 133
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<int64_t>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<int64_t>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<double>& output_z);

// 200
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<int32_t>& output_z);

// 201
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<int64_t>& output_z);

// 202
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<float>& output_z);

// 203
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<int32_t>& input_x,
                                const Column<int32_t>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<int32_t>& output_x,
                                Column<int32_t>& output_y,
                                Column<double>& output_z);

// 210
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<int32_t>& output_z);

// 211
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<int64_t>& output_z);

// 212
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<float>& output_z);

// 213
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<int64_t>& input_x,
                                const Column<int64_t>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<int64_t>& output_x,
                                Column<int64_t>& output_y,
                                Column<double>& output_z);

// 220
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<int32_t>& output_z);

// 221
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<int64_t>& output_z);

// 222
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<float>& output_z);

// 223
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<float>& input_x,
                                const Column<float>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<float>& output_x,
                                Column<float>& output_y,
                                Column<double>& output_z);

// 230
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<int32_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<int32_t>& output_z);

// 231
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<int64_t>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<int64_t>& output_z);

// 232
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<float>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<float>& output_z);

// 233
template NEVER_INLINE HOST int32_t
ct_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                const TextEncodingNone& agg_type,
                                const Column<TextEncodingDict>& input_id,
                                const Column<double>& input_x,
                                const Column<double>& input_y,
                                const Column<double>& input_z,
                                Column<int32_t>& output_row_count,
                                Column<TextEncodingDict>& output_id,
                                Column<double>& output_x,
                                Column<double>& output_y,
                                Column<double>& output_z);

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename K, typename T, typename Z>
NEVER_INLINE HOST int32_t ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                                               const Column<K>& input_id,
                                                               const Column<T>& input_x,
                                                               const Column<T>& input_y,
                                                               const Column<Z>& input_z,
                                                               Column<K>& output_id,
                                                               Column<T>& output_x,
                                                               Column<T>& output_y,
                                                               Column<Z>& output_z) {
  const int64_t input_size = input_id.size();
  mgr.set_output_row_size(input_size);
  for (int64_t input_idx = 0; input_idx < input_size; ++input_idx) {
    output_id[input_idx] = input_id[input_idx];
    output_x[input_idx] = input_x[input_idx];
    output_y[input_idx] = input_y[input_idx];
    output_z[input_idx] = input_z[input_idx];
  }
  return input_size;
}

// explicit instantiations

// 000
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<int32_t>& output_z);

// 001
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<int64_t>& output_z);

// 002
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<float>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<float>& output_z);

// 003
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<double>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<double>& output_z);

// 010
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<int32_t>& output_z);

// 011
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<int64_t>& output_z);

// 012
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<float>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<float>& output_z);

// 013
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<double>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<double>& output_z);

// 020
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<int32_t>& output_z);

// 021
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<int64_t>& output_z);

// 022
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<float>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<float>& output_z);

// 023
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<double>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<double>& output_z);

// 030
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<int32_t>& output_z);

// 031
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<int64_t>& output_z);

// 032
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<float>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<float>& output_z);

// 033
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int32_t>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<double>& input_z,
                                     Column<int32_t>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<double>& output_z);

// 100
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<int32_t>& output_z);

// 101
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<int64_t>& output_z);

// 102
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<float>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<float>& output_z);

// 103
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<double>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<double>& output_z);

// 110
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<int32_t>& output_z);

// 111
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<int64_t>& output_z);

// 112
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<float>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<float>& output_z);

// 113
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<double>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<double>& output_z);

// 120
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<int32_t>& output_z);

// 121
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<int64_t>& output_z);

// 122
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<float>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<float>& output_z);

// 123
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<double>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<double>& output_z);

// 130
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<int32_t>& output_z);

// 131
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<int64_t>& output_z);

// 132
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<float>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<float>& output_z);

// 133
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<int64_t>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<double>& input_z,
                                     Column<int64_t>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<double>& output_z);

// 200
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<int32_t>& output_z);

// 201
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<int64_t>& output_z);

// 202
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<float>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<float>& output_z);

// 203
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<int32_t>& input_x,
                                     const Column<int32_t>& input_y,
                                     const Column<double>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<int32_t>& output_x,
                                     Column<int32_t>& output_y,
                                     Column<double>& output_z);

// 210
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<int32_t>& output_z);

// 211
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<int64_t>& output_z);

// 212
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<float>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<float>& output_z);

// 213
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<int64_t>& input_x,
                                     const Column<int64_t>& input_y,
                                     const Column<double>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<int64_t>& output_x,
                                     Column<int64_t>& output_y,
                                     Column<double>& output_z);

// 220
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<int32_t>& output_z);

// 221
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<int64_t>& output_z);

// 222
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<float>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<float>& output_z);

// 223
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<float>& input_x,
                                     const Column<float>& input_y,
                                     const Column<double>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<float>& output_x,
                                     Column<float>& output_y,
                                     Column<double>& output_z);

// 230
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<int32_t>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<int32_t>& output_z);

// 231
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<int64_t>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<int64_t>& output_z);

// 232
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<float>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<float>& output_z);

// 233
template NEVER_INLINE HOST int32_t
ct_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                     const Column<TextEncodingDict>& input_id,
                                     const Column<double>& input_x,
                                     const Column<double>& input_y,
                                     const Column<double>& input_z,
                                     Column<TextEncodingDict>& output_id,
                                     Column<double>& output_x,
                                     Column<double>& output_y,
                                     Column<double>& output_z);

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename K, typename T, typename Z>
NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<K>& input1_id,
                                      const Column<T>& input1_x,
                                      const Column<T>& input1_y,
                                      const Column<Z>& input1_z,
                                      const Column<K>& input2_id,
                                      const Column<T>& input2_x,
                                      const Column<T>& input2_y,
                                      const Column<Z>& input2_z,
                                      const Column<T>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<K>& output_id,
                                      Column<T>& output_x,
                                      Column<T>& output_y,
                                      Column<Z>& output_z,
                                      Column<T>& output_w) {
  mgr.set_output_row_size(1);
  const std::string agg_type_str = agg_type.getString();
  const TFAggType min_or_max = agg_type_str == "MIN" ? TFAggType::MIN : TFAggType::MAX;
  output_row_count[0] = input1_id.size() + input2_id.size();
  output_id[0] = get_min_or_max_union(input1_id, input2_id, min_or_max);
  output_x[0] = get_min_or_max_union(input1_x, input2_x, min_or_max);
  output_y[0] = get_min_or_max_union(input1_y, input2_y, min_or_max);
  output_z[0] = get_min_or_max_union(input1_z, input2_z, min_or_max);
  if (input2_w.size() > 0) {
    const auto w_min_max = get_column_min_max(input2_w);
    output_w[0] = agg_type_str == "MIN" ? w_min_max.first : w_min_max.second;
  } else {
    output_w.setNull(0);
  }
  return 1;
}

// explicit instantiations

// 000
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<int32_t>& output_w);

// 001
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<int32_t>& output_w);

// 002
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<float>& output_z,
                                      Column<int32_t>& output_w);

// 003
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<double>& output_z,
                                      Column<int32_t>& output_w);

// 010
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<int64_t>& output_w);

// 011
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<int64_t>& output_w);

// 012
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<float>& output_z,
                                      Column<int64_t>& output_w);

// 013
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<double>& output_z,
                                      Column<int64_t>& output_w);

// 020
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<float>& output_w);

// 021
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<float>& output_w);

// 022
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<float>& output_z,
                                      Column<float>& output_w);

// 023
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<double>& output_z,
                                      Column<float>& output_w);

// 030
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<double>& output_w);

// 031
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<double>& output_w);

// 032
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<float>& output_z,
                                      Column<double>& output_w);

// 033
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int32_t>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<int32_t>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int32_t>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<double>& output_z,
                                      Column<double>& output_w);

// 100
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<int32_t>& output_w);

// 101
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<int32_t>& output_w);

// 102
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<float>& output_z,
                                      Column<int32_t>& output_w);

// 103
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<double>& output_z,
                                      Column<int32_t>& output_w);

// 110
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<int64_t>& output_w);

// 111
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<int64_t>& output_w);

// 112
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<float>& output_z,
                                      Column<int64_t>& output_w);

// 113
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<double>& output_z,
                                      Column<int64_t>& output_w);

// 120
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<float>& output_w);

// 121
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<float>& output_w);

// 122
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<float>& output_z,
                                      Column<float>& output_w);

// 123
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<double>& output_z,
                                      Column<float>& output_w);

// 130
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<double>& output_w);

// 131
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<double>& output_w);

// 132
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<float>& output_z,
                                      Column<double>& output_w);

// 133
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<int64_t>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<int64_t>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<int64_t>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<double>& output_z,
                                      Column<double>& output_w);

// 200
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<int32_t>& output_w);

// 201
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<int32_t>& output_w);

// 202
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<float>& output_z,
                                      Column<int32_t>& output_w);

// 203
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<int32_t>& input1_x,
                                      const Column<int32_t>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<int32_t>& input2_x,
                                      const Column<int32_t>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<int32_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<int32_t>& output_x,
                                      Column<int32_t>& output_y,
                                      Column<double>& output_z,
                                      Column<int32_t>& output_w);

// 210
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<int64_t>& output_w);

// 211
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<int64_t>& output_w);

// 212
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<float>& output_z,
                                      Column<int64_t>& output_w);

// 213
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<int64_t>& input1_x,
                                      const Column<int64_t>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<int64_t>& input2_x,
                                      const Column<int64_t>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<int64_t>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<int64_t>& output_x,
                                      Column<int64_t>& output_y,
                                      Column<double>& output_z,
                                      Column<int64_t>& output_w);

// 220
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<float>& output_w);

// 221
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<float>& output_w);

// 222
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<float>& output_z,
                                      Column<float>& output_w);

// 223
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<float>& input1_x,
                                      const Column<float>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<float>& input2_x,
                                      const Column<float>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<float>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<float>& output_x,
                                      Column<float>& output_y,
                                      Column<double>& output_z,
                                      Column<float>& output_w);

// 230
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<int32_t>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<int32_t>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<int32_t>& output_z,
                                      Column<double>& output_w);

// 231
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<int64_t>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<int64_t>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<int64_t>& output_z,
                                      Column<double>& output_w);

// 232
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<float>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<float>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<float>& output_z,
                                      Column<double>& output_w);

// 233
template NEVER_INLINE HOST int32_t
ct_union_pushdown_stats__cpu_template(TableFunctionManager& mgr,
                                      const TextEncodingNone& agg_type,
                                      const Column<TextEncodingDict>& input1_id,
                                      const Column<double>& input1_x,
                                      const Column<double>& input1_y,
                                      const Column<double>& input1_z,
                                      const Column<TextEncodingDict>& input2_id,
                                      const Column<double>& input2_x,
                                      const Column<double>& input2_y,
                                      const Column<double>& input2_z,
                                      const Column<double>& input2_w,
                                      Column<int32_t>& output_row_count,
                                      Column<TextEncodingDict>& output_id,
                                      Column<double>& output_x,
                                      Column<double>& output_y,
                                      Column<double>& output_z,
                                      Column<double>& output_w);

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename K, typename T, typename Z>
NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<K>& input1_id,
                                           const Column<T>& input1_x,
                                           const Column<T>& input1_y,
                                           const Column<Z>& input1_z,
                                           const Column<K>& input2_id,
                                           const Column<T>& input2_x,
                                           const Column<T>& input2_y,
                                           const Column<Z>& input2_z,
                                           const Column<T>& input2_w,
                                           Column<K>& output_id,
                                           Column<T>& output_x,
                                           Column<T>& output_y,
                                           Column<Z>& output_z,
                                           Column<T>& output_w) {
  const int64_t input1_size = input1_id.size();
  const int64_t input2_size = input2_id.size();
  int64_t output_size = input1_size + input2_size;
  mgr.set_output_row_size(output_size);
  for (int64_t input1_idx = 0; input1_idx < input1_size; ++input1_idx) {
    output_id[input1_idx] = input1_id[input1_idx];
    output_x[input1_idx] = input1_x[input1_idx];
    output_y[input1_idx] = input1_y[input1_idx];
    output_z[input1_idx] = input1_z[input1_idx];
    output_w.setNull(input1_idx);
  }
  for (int64_t input2_idx = 0; input2_idx < input2_size; ++input2_idx) {
    output_id[input1_size + input2_idx] = input2_id[input2_idx];
    output_x[input1_size + input2_idx] = input2_x[input2_idx];
    output_y[input1_size + input2_idx] = input2_y[input2_idx];
    output_z[input1_size + input2_idx] = input2_z[input2_idx];
    output_w[input1_size + input2_idx] = input2_w[input2_idx];
  }
  return output_size;
}

// explicit instantiations

// 000
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<int32_t>& output_w);

// 001
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<int32_t>& output_w);

// 002
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<float>& output_z,
                                           Column<int32_t>& output_w);

// 003
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<double>& output_z,
                                           Column<int32_t>& output_w);

// 010
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<int64_t>& output_w);

// 011
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<int64_t>& output_w);

// 012
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<float>& output_z,
                                           Column<int64_t>& output_w);

// 013
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<double>& output_z,
                                           Column<int64_t>& output_w);

// 020
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<float>& output_w);

// 021
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<float>& output_w);

// 022
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<float>& output_z,
                                           Column<float>& output_w);

// 023
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<double>& output_z,
                                           Column<float>& output_w);

// 030
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<double>& output_w);

// 031
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<double>& output_w);

// 032
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<float>& output_z,
                                           Column<double>& output_w);

// 033
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int32_t>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<int32_t>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<int32_t>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<double>& output_z,
                                           Column<double>& output_w);

// 100
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<int32_t>& output_w);

// 101
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<int32_t>& output_w);

// 102
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<float>& output_z,
                                           Column<int32_t>& output_w);

// 103
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<double>& output_z,
                                           Column<int32_t>& output_w);

// 110
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<int64_t>& output_w);

// 111
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<int64_t>& output_w);

// 112
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<float>& output_z,
                                           Column<int64_t>& output_w);

// 113
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<double>& output_z,
                                           Column<int64_t>& output_w);

// 120
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<float>& output_w);

// 121
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<float>& output_w);

// 122
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<float>& output_z,
                                           Column<float>& output_w);

// 123
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<double>& output_z,
                                           Column<float>& output_w);

// 130
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<double>& output_w);

// 131
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<double>& output_w);

// 132
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<float>& output_z,
                                           Column<double>& output_w);

// 133
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<int64_t>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<int64_t>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<int64_t>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<double>& output_z,
                                           Column<double>& output_w);

// 200
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<int32_t>& output_w);

// 201
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<int32_t>& output_w);

// 202
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<float>& output_z,
                                           Column<int32_t>& output_w);

// 203
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<int32_t>& input1_x,
                                           const Column<int32_t>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<int32_t>& input2_x,
                                           const Column<int32_t>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<int32_t>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<int32_t>& output_x,
                                           Column<int32_t>& output_y,
                                           Column<double>& output_z,
                                           Column<int32_t>& output_w);

// 210
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<int64_t>& output_w);

// 211
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<int64_t>& output_w);

// 212
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<float>& output_z,
                                           Column<int64_t>& output_w);

// 213
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<int64_t>& input1_x,
                                           const Column<int64_t>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<int64_t>& input2_x,
                                           const Column<int64_t>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<int64_t>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<int64_t>& output_x,
                                           Column<int64_t>& output_y,
                                           Column<double>& output_z,
                                           Column<int64_t>& output_w);

// 220
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<float>& output_w);

// 221
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<float>& output_w);

// 222
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<float>& output_z,
                                           Column<float>& output_w);

// 223
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<float>& input1_x,
                                           const Column<float>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<float>& input2_x,
                                           const Column<float>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<float>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<float>& output_x,
                                           Column<float>& output_y,
                                           Column<double>& output_z,
                                           Column<float>& output_w);

// 230
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<int32_t>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<int32_t>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<int32_t>& output_z,
                                           Column<double>& output_w);

// 231
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<int64_t>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<int64_t>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<int64_t>& output_z,
                                           Column<double>& output_w);

// 232
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<float>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<float>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<float>& output_z,
                                           Column<double>& output_w);

// 233
template NEVER_INLINE HOST int32_t
ct_union_pushdown_projection__cpu_template(TableFunctionManager& mgr,
                                           const Column<TextEncodingDict>& input1_id,
                                           const Column<double>& input1_x,
                                           const Column<double>& input1_y,
                                           const Column<double>& input1_z,
                                           const Column<TextEncodingDict>& input2_id,
                                           const Column<double>& input2_x,
                                           const Column<double>& input2_y,
                                           const Column<double>& input2_z,
                                           const Column<double>& input2_w,
                                           Column<TextEncodingDict>& output_id,
                                           Column<double>& output_x,
                                           Column<double>& output_y,
                                           Column<double>& output_z,
                                           Column<double>& output_w);

#endif