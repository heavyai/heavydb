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

#ifndef __CUDACC__

#include "NullRowsRemoval.h"
#include <tbb/parallel_for.h>
#include "Shared/ThreadInfo.h"

namespace TableFunctions_Namespace {

template <typename T>
InputData<T> strip_column_metadata(const ColumnList<T>& input_features) {
  InputData<T> input_data;
  input_data.num_rows = input_features.size();
  input_data.null_val = inline_null_value<T>();
  for (int64_t c = 0; c < input_features.numCols(); ++c) {
    input_data.col_ptrs.push_back(reinterpret_cast<T*>(input_features.ptrs_[c]));
  }
  return input_data;
}

template InputData<float> strip_column_metadata(const ColumnList<float>& input_features);
template InputData<double> strip_column_metadata(
    const ColumnList<double>& input_features);

template <typename T>
InputData<T> strip_column_metadata(const Column<T>& input_labels,
                                   const ColumnList<T>& input_features) {
  InputData<T> input_data;
  input_data.num_rows = input_features.size();
  CHECK_EQ(input_data.num_rows, input_labels.size());
  input_data.null_val = inline_null_value<T>();
  input_data.col_ptrs.push_back(input_labels.ptr_);
  for (int64_t c = 0; c < input_features.numCols(); ++c) {
    input_data.col_ptrs.push_back(reinterpret_cast<T*>(input_features.ptrs_[c]));
  }
  return input_data;
}

template InputData<float> strip_column_metadata(const Column<float>& input_labels,
                                                const ColumnList<float>& input_features);

template InputData<double> strip_column_metadata(
    const Column<double>& input_labels,
    const ColumnList<double>& input_features);

template <typename T>
MaskedData<T> remove_null_rows(const InputData<T>& input_data) {
  MaskedData<T> masked_data;
  const auto input_num_rows = input_data.num_rows;
  masked_data.unmasked_num_rows = input_num_rows;
  masked_data.index_map.resize(input_num_rows);
  auto& index_map = masked_data.index_map;
  const int32_t num_cols = input_data.col_ptrs.size();
  int32_t valid_row_count = 0;
  masked_data.reverse_index_map.reserve(masked_data.unmasked_num_rows);
  auto& reverse_index_map = masked_data.reverse_index_map;
  const auto null_val = input_data.null_val;
  constexpr int64_t target_rows_per_thread{20000};
  const ThreadInfo thread_info(
      std::thread::hardware_concurrency(), input_num_rows, target_rows_per_thread);
  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);
  std::vector<std::vector<int32_t>> per_thread_reverse_index_maps(
      thread_info.num_threads);
  tbb::task_arena limited_arena(thread_info.num_threads);
  limited_arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(
            0, input_num_rows, static_cast<int32_t>(thread_info.num_elems_per_thread)),
        [&](const tbb::blocked_range<int64_t>& r) {
          size_t thread_idx = tbb::this_task_arena::current_thread_index();
          auto& local_reverse_index_map = per_thread_reverse_index_maps[thread_idx];
          int32_t local_valid_row_count = 0;
          const int32_t start_idx = r.begin();
          const int32_t end_idx = r.end();
          for (int32_t row_idx = start_idx; row_idx < end_idx; ++row_idx) {
            int32_t col_idx = 0;
            for (; col_idx < num_cols; ++col_idx) {
              if (input_data.col_ptrs[col_idx][row_idx] == null_val) {
                index_map[row_idx] = NULL_ROW_IDX;
                break;
              }
            }
            if (col_idx == num_cols) {
              local_reverse_index_map.emplace_back(row_idx);
              index_map[row_idx] = local_valid_row_count++;
            }
          }
        },
        tbb::simple_partitioner());
  });

  for (const auto& per_thread_reverse_index_map : per_thread_reverse_index_maps) {
    valid_row_count += per_thread_reverse_index_map.size();
  }
  masked_data.masked_num_rows = valid_row_count;

  masked_data.data.resize(num_cols, nullptr);
  // Exit early if there are no nulls to avoid unneeded computation
  if (masked_data.masked_num_rows == masked_data.unmasked_num_rows) {
    for (int32_t col_idx = 0; col_idx < num_cols; ++col_idx) {
      masked_data.data[col_idx] = input_data.col_ptrs[col_idx];
    }
    return masked_data;
  }

  masked_data.reverse_index_map.resize(valid_row_count);

  int32_t start_reverse_index_offset = 0;
  std::vector<std::future<void>> worker_threads;
  for (const auto& per_thread_reverse_index_map : per_thread_reverse_index_maps) {
    const int32_t local_reverse_index_map_size = per_thread_reverse_index_map.size();
    worker_threads.emplace_back(std::async(
        std::launch::async,
        [&reverse_index_map](const auto& local_map,
                             const int32_t local_map_offset,
                             const int32_t local_map_size) {
          for (int32_t map_idx = 0; map_idx < local_map_size; ++map_idx) {
            reverse_index_map[map_idx + local_map_offset] = local_map[map_idx];
          }
        },
        per_thread_reverse_index_map,
        start_reverse_index_offset,
        local_reverse_index_map_size));
    start_reverse_index_offset += local_reverse_index_map_size;
  }
  for (auto& worker_thread : worker_threads) {
    worker_thread.wait();
  }
  masked_data.data_allocations.resize(num_cols);
  for (int32_t col_idx = 0; col_idx < num_cols; ++col_idx) {
    masked_data.data_allocations[col_idx].resize(valid_row_count);
    masked_data.data[col_idx] = masked_data.data_allocations[col_idx].data();
  }

  auto& denulled_data = masked_data.data;

  tbb::parallel_for(tbb::blocked_range<int32_t>(0, valid_row_count),
                    [&](const tbb::blocked_range<int32_t>& r) {
                      const int32_t start_idx = r.begin();
                      const int32_t end_idx = r.end();
                      for (int32_t row_idx = start_idx; row_idx < end_idx; ++row_idx) {
                        const auto input_row_idx = reverse_index_map[row_idx];
                        for (int32_t col_idx = 0; col_idx < num_cols; ++col_idx) {
                          denulled_data[col_idx][row_idx] =
                              input_data.col_ptrs[col_idx][input_row_idx];
                        }
                      }
                    });
  return masked_data;
}

template MaskedData<float> remove_null_rows(const InputData<float>& input_data);
template MaskedData<double> remove_null_rows(const InputData<double>& input_data);

template <typename T>
void unmask_data(const T* masked_input,
                 const std::vector<int32_t>& reverse_index_map,
                 T* unmasked_output,
                 const int64_t num_unmasked_rows,
                 const T null_val) {
  // First fill data with nulls (as its currently initialized to 0)
  // Todo(todd): Look at allowing override of default 0-initialization of output columns
  // to avoid overhead of double initialization
  tbb::parallel_for(tbb::blocked_range<size_t>(0, static_cast<size_t>(num_unmasked_rows)),
                    [&](const tbb::blocked_range<size_t>& r) {
                      const int32_t start_idx = r.begin();
                      const int32_t end_idx = r.end();
                      for (int32_t row_idx = start_idx; row_idx < end_idx; ++row_idx) {
                        unmasked_output[row_idx] = null_val;
                      }
                    });

  const auto num_masked_rows = reverse_index_map.size();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_masked_rows),
                    [&](const tbb::blocked_range<size_t>& r) {
                      const int32_t start_idx = r.begin();
                      const int32_t end_idx = r.end();
                      for (int32_t row_idx = start_idx; row_idx < end_idx; ++row_idx) {
                        unmasked_output[reverse_index_map[row_idx]] =
                            masked_input[row_idx];
                      }
                    });
}

template void unmask_data(const int32_t* masked_input,
                          const std::vector<int32_t>& reverse_index_map,
                          int32_t* unmasked_output,
                          const int64_t num_unmasked_rows,
                          const int32_t null_val);

template void unmask_data(const float* masked_input,
                          const std::vector<int32_t>& reverse_index_map,
                          float* unmasked_output,
                          const int64_t num_unmasked_rows,
                          const float null_val);

template void unmask_data(const double* masked_input,
                          const std::vector<int32_t>& reverse_index_map,
                          double* unmasked_output,
                          const int64_t num_unmasked_rows,
                          const double null_val);

}  // namespace TableFunctions_Namespace

#endif  // __CUDACC__