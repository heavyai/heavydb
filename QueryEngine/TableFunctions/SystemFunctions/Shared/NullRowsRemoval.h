/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
