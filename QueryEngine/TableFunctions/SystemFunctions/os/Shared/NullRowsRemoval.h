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

#ifndef __CUDACC__

#include "QueryEngine/heavydbTypes.h"

namespace TableFunctions_Namespace {

constexpr int32_t NULL_ROW_IDX{-1};

template <typename T>
struct MaskedData {
  std::vector<T*> data;
  int32_t unmasked_num_rows;
  int32_t masked_num_rows;
  std::vector<int32_t> index_map;
  std::vector<int32_t> reverse_index_map;

  std::vector<std::vector<T>> data_allocations;
};

template <typename T>
struct InputData {
  std::vector<T*> col_ptrs;
  int32_t num_rows;
  T null_val;
};

template <typename T>
InputData<T> strip_column_metadata(const ColumnList<T>& input_features);

template <typename T>
InputData<T> strip_column_metadata(const Column<T>& input_labels,
                                   const ColumnList<T>& input_features);

template <typename T>
InputData<T> get_input_ptrs(const ColumnList<T>& input_features);

template <typename T>
InputData<T> get_input_ptrs(const MaskedData<T>& masked_input_features);

template <typename T>
MaskedData<T> remove_null_rows(const InputData<T>& input_data);

template <typename T>
void unmask_data(const T* masked_input,
                 const std::vector<int32_t>& reverse_index_map,
                 T* unmasked_output,
                 const int64_t num_unmasked_rows,
                 const T null_val);

template <typename T>
MaskedData<T> denull_data(const ColumnList<T>& features) {
  auto input_data = strip_column_metadata(features);
  return remove_null_rows(input_data);
}

template <typename T>
MaskedData<T> denull_data(const Column<T>& labels, const ColumnList<T>& features) {
  auto input_data = strip_column_metadata(labels, features);
  return remove_null_rows(input_data);
}

}  // namespace TableFunctions_Namespace

#endif  // __CUDACC__
