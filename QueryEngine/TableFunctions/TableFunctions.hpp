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

#include "../../QueryEngine/OmniSciTypes.h"
#include "../../Shared/funcannotations.h"

// clang-format off
/*
  UDTF: row_copier(Column<double>, RowMultiplier) -> Column<double>
  UDTF: row_copier_text(Column<TextEncodingDict>, RowMultiplier) -> Column<TextEncodingDict> | input_id=args<0>
*/
// clang-format on
EXTENSION_NOINLINE int32_t row_copier(const Column<double>& input_col,
                                      int copy_multiplier,
                                      Column<double>& output_col) {
  int32_t output_row_count = copy_multiplier * input_col.size();
  if (output_row_count > 100) {
    // Test failure propagation.
    return -1;
  }
  if (output_col.size() != output_row_count) {
    return -1;
  }

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col.size());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col.size();
  auto step = 1;
#endif

  for (auto i = start; i < stop; i += step) {
    for (int c = 0; c < copy_multiplier; c++) {
      output_col[i + (c * input_col.size())] = input_col[i];
    }
  }

  return output_row_count;
}

EXTENSION_NOINLINE int32_t row_copier_text(const Column<TextEncodingDict>& input_col,
                                           int copy_multiplier,
                                           Column<TextEncodingDict>& output_col) {
  int32_t output_row_count = copy_multiplier * input_col.size();
  if (output_row_count > 100) {
    // Test failure propagation.
    return -1;
  }
  if (output_col.size() != output_row_count) {
    return -2;
  }

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col.size());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col.size();
  auto step = 1;
#endif

  for (auto i = start; i < stop; i += step) {
    for (int c = 0; c < copy_multiplier; c++) {
      output_col[i + (c * input_col.size())] = input_col[i];
    }
  }

  return output_row_count;
}

/*
  UDTF: row_adder(RowMultiplier<1>, Cursor<ColumnDouble, ColumnDouble>) -> ColumnDouble
*/
EXTENSION_NOINLINE int32_t row_adder(const int copy_multiplier,
                                     const Column<double>& input_col1,
                                     const Column<double>& input_col2,
                                     Column<double>& output_col) {
  int32_t output_row_count = copy_multiplier * input_col1.size();
  if (output_row_count > 100) {
    // Test failure propagation.
    return -1;
  }
  if (output_col.size() != output_row_count) {
    return -1;
  }

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col1.size());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col1.size();
  auto step = 1;
#endif
  auto stride = input_col1.size();
  for (auto i = start; i < stop; i += step) {
    for (int c = 0; c < copy_multiplier; c++) {
      if (input_col1.isNull(i) || input_col2.isNull(i)) {
        output_col.setNull(i + (c * stride));
      } else {
        output_col[i + (c * stride)] = input_col1[i] + input_col2[i];
      }
    }
  }

  return output_row_count;
}

// clang-format off
/*
  UDTF: row_addsub(RowMultiplier, Cursor<double, double>) -> Column<double>, Column<double>
*/
// clang-format on
EXTENSION_NOINLINE int32_t row_addsub(const int copy_multiplier,
                                      const Column<double>& input_col1,
                                      const Column<double>& input_col2,
                                      Column<double>& output_col1,
                                      Column<double>& output_col2) {
  int32_t output_row_count = copy_multiplier * input_col1.size();
  if (output_row_count > 100) {
    // Test failure propagation.
    return -1;
  }
  if ((output_col1.size() != output_row_count) ||
      (output_col2.size() != output_row_count)) {
    return -1;
  }

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col1.size());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col1.size();
  auto step = 1;
#endif
  auto stride = input_col1.size();
  for (auto i = start; i < stop; i += step) {
    for (int c = 0; c < copy_multiplier; c++) {
      output_col1[i + (c * stride)] = input_col1[i] + input_col2[i];
      output_col2[i + (c * stride)] = input_col1[i] - input_col2[i];
    }
  }
  return output_row_count;
}

// clang-format off
/*
  UDTF: get_max_with_row_offset(Cursor<int>, Constant<1>) -> Column<int>, Column<int>
*/
// clang-format on
EXTENSION_NOINLINE int32_t get_max_with_row_offset(const Column<int>& input_col,
                                                   Column<int>& output_max_col,
                                                   Column<int>& output_max_row_col) {
  if ((output_max_col.size() != 1) || output_max_row_col.size() != 1) {
    return -1;
  }
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col.size());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col.size();
  auto step = 1;
#endif

  int curr_max = -2147483648;
  int curr_max_row = -1;
  for (auto i = start; i < stop; i += step) {
    if (input_col[i] > curr_max) {
      curr_max = input_col[i];
      curr_max_row = i;
    }
  }
  output_max_col[0] = curr_max;
  output_max_row_col[0] = curr_max_row;
  return 1;
}

#include "TableFunctionsTesting.hpp"

// clang-format off
/*
  UDTF: column_list_get__cpu_(ColumnList<double>, int, RowMultiplier) -> Column<double>
*/
// clang-format on
EXTENSION_NOINLINE int32_t column_list_get__cpu_(const ColumnList<double>& col_list,
                                                 const int index,
                                                 const int m,
                                                 Column<double>& col) {
  col = col_list[index];  // copy the data of col_list item to output column
  return col.size();
}

// clang-format off
/*
  UDTF: column_list_first_last(ColumnList<double>, RowMultiplier) -> Column<double>,
  Column<double>
*/
// clang-format on
EXTENSION_NOINLINE int32_t column_list_first_last(const ColumnList<double>& col_list,
                                                  const int m,
                                                  Column<double>& col1,
                                                  Column<double>& col2) {
  col1 = col_list[0];
  col2 = col_list[col_list.numCols() - 1];
  return col1.size();
}

// clang-format off
/*
  UDTF: column_list_row_sum__cpu_(Cursor<ColumnList<int32_t>>) -> Column<int32_t>
 */
// clang-format on

EXTENSION_NOINLINE int32_t column_list_row_sum__cpu_(const ColumnList<int32_t>& input,
                                                     Column<int32_t>& out) {
  int32_t output_num_rows = input.numCols();
  set_output_row_size(output_num_rows);
  for (int i = 0; i < output_num_rows; i++) {
    auto col = input[i];
    int32_t s = 0;
    for (int j = 0; j < col.size(); j++) {
      s += col[j];
    }
    out[i] = s;
  }
  return output_num_rows;
}

#include "MLFunctions.hpp"
