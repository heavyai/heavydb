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

#include "TableFunctionsTesting.h"

/*
  This file contains signature-related compile-time test UDTFs.
  These functions test features related to UDTF signatures, such
  as input/output sizing, templating, pre-flight features, overloading, etc.
 */

#ifndef __CUDACC__

template <typename T, typename U, typename K>
NEVER_INLINE HOST int32_t ct_binding_column2__cpu_template(const Column<T>& input1,
                                                           const Column<U>& input2,
                                                           Column<K>& out) {
  if constexpr (std::is_same<T, TextEncodingDict>::value &&
                std::is_same<U, TextEncodingDict>::value) {
    set_output_row_size(input1.size());
    for (int64_t i = 0; i < input1.size(); i++) {
      out[i] = input1[i];
    }
    return input1.size();
  }

  set_output_row_size(1);
  if constexpr (std::is_same<T, int32_t>::value && std::is_same<U, double>::value) {
    out[0] = 10;
  } else if constexpr (std::is_same<T, double>::value && std::is_same<U, double>::value) {
    out[0] = 20;
  } else if constexpr (std::is_same<T, int32_t>::value &&
                       std::is_same<U, int32_t>::value) {
    out[0] = 30;
  } else if constexpr (std::is_same<T, double>::value &&
                       std::is_same<U, int32_t>::value) {
    out[0] = 40;
  }
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_binding_column2__cpu_template(const Column<int32_t>& input1,
                                 const Column<int32_t>& input2,
                                 Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_column2__cpu_template(const Column<int32_t>& input1,
                                 const Column<double>& input2,
                                 Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_column2__cpu_template(const Column<double>& input1,
                                 const Column<int32_t>& input2,
                                 Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_column2__cpu_template(const Column<double>& input1,
                                 const Column<double>& input2,
                                 Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_column2__cpu_template(const Column<TextEncodingDict>& input1,
                                 const Column<TextEncodingDict>& input2,
                                 Column<TextEncodingDict>& out);

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_named_output__cpu_template(const Column<T>& input,
                                                        Column<T>& out) {
  set_output_row_size(1);
  T acc = 0;
  for (int64_t i = 0; i < input.size(); i++) {
    acc += input[i];
  }
  out[0] = acc;
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_named_output__cpu_template(const Column<int32_t>& input, Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_named_output__cpu_template(const Column<double>& input, Column<double>& out);

template <typename T>
NEVER_INLINE HOST int32_t ct_named_const_output__cpu_template(const Column<T>& input,
                                                              Column<T>& out) {
  T acc1 = 0, acc2 = 0;
  for (int64_t i = 0; i < input.size(); i++) {
    if (i % 2 == 0) {
      acc1 += input[i];
    } else {
      acc2 += input[i];
    }
  }
  out[0] = acc1;
  out[1] = acc2;
  return 2;
}

template NEVER_INLINE HOST int32_t
ct_named_const_output__cpu_template(const Column<int32_t>& input, Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_named_const_output__cpu_template(const Column<double>& input, Column<double>& out);

template <typename T>
NEVER_INLINE HOST int32_t ct_named_user_const_output__cpu_template(const Column<T>& input,
                                                                   int32_t c,
                                                                   Column<T>& out) {
  for (int64_t i = 0; i < c; i++) {
    out[i] = 0;
  }
  for (int64_t i = 0; i < input.size(); i++) {
    out[i % c] += input[i];
  }
  return c;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_named_user_const_output__cpu_template(const Column<int32_t>& input,
                                         int32_t c,
                                         Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_named_user_const_output__cpu_template(const Column<double>& input,
                                         int32_t c,
                                         Column<double>& out);

template <typename T>
NEVER_INLINE HOST int32_t ct_named_rowmul_output__cpu_template(const Column<T>& input,
                                                               int32_t m,
                                                               Column<T>& out) {
  for (int64_t j = 0; j < m; j++) {
    for (int64_t i = 0; i < input.size(); i++) {
      out[j * input.size() + i] += input[i];
    }
  }
  return m * input.size();
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_named_rowmul_output__cpu_template(const Column<int32_t>& input,
                                     int32_t m,
                                     Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_named_rowmul_output__cpu_template(const Column<double>& input,
                                     int32_t m,
                                     Column<double>& out);

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_no_arg_runtime_sizing__cpu_template(Column<T>& answer) {
  set_output_row_size(1);
  answer[0] = 42;
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_no_arg_runtime_sizing__cpu_template(Column<int32_t>& answer);

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE int32_t ct_no_arg_constant_sizing(Column<int32_t>& answer) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(42);
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = 42;
  auto step = 1;
#endif
  for (auto i = start; i < stop; i += step) {
    answer[i] = 42 * i;
  }
  return 42;
}

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t
ct_scalar_1_arg_runtime_sizing__cpu_template(const T num, Column<T>& answer) {
  T quotient = num;
  set_output_row_size(30);
  int32_t counter{0};
  while (quotient >= 1) {
    answer[counter++] = quotient;
    quotient /= 10;
  }
  return counter;
}

template NEVER_INLINE HOST int32_t
ct_scalar_1_arg_runtime_sizing__cpu_template(const float num, Column<float>& answer);
template NEVER_INLINE HOST int32_t
ct_scalar_1_arg_runtime_sizing__cpu_template(const double num, Column<double>& answer);
template NEVER_INLINE HOST int32_t
ct_scalar_1_arg_runtime_sizing__cpu_template(const int32_t num, Column<int32_t>& answer);
template NEVER_INLINE HOST int32_t
ct_scalar_1_arg_runtime_sizing__cpu_template(const int64_t num, Column<int64_t>& answer);

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE int32_t ct_scalar_2_args_constant_sizing(const int64_t num1,
                                                            const int64_t num2,
                                                            Column<int64_t>& answer1,
                                                            Column<int64_t>& answer2) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(5);
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = 5;
  auto step = 1;
#endif
  for (auto i = start; i < stop; i += step) {
    answer1[i] = num1 + i * num2;
    answer2[i] = num1 - i * num2;
  }
  return 5;
}

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
ct_no_cursor_user_constant_sizer__cpu_(const int32_t input_num,
                                       int32_t c,
                                       Column<int32_t>& output) {
  for (int32_t i = 0; i < c; i++) {
    output[i] = input_num;
  }
  return c;
}

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t
ct_templated_no_cursor_user_constant_sizer__cpu_template(const T input_num,
                                                         int32_t c,
                                                         Column<T>& output) {
  for (int32_t i = 0; i < c; i++) {
    output[i] = input_num;
  }
  return c;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_templated_no_cursor_user_constant_sizer__cpu_template(const int32_t input_num,
                                                         int32_t c,
                                                         Column<int32_t>& output);
template NEVER_INLINE HOST int32_t
ct_templated_no_cursor_user_constant_sizer__cpu_template(const float input_num,
                                                         int32_t c,
                                                         Column<float>& output);

#endif  // #ifndef __CUDACC__

#ifdef __CUDACC__

EXTENSION_NOINLINE int32_t
ct_no_cursor_user_constant_sizer__gpu_(const int32_t input_num,
                                       int32_t c,
                                       Column<int32_t>& output) {
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;

  for (int32_t i = start; i < c; i += step) {
    output[i] = input_num;
  }
  return c;
}

template <typename T>
TEMPLATE_NOINLINE int32_t
ct_templated_no_cursor_user_constant_sizer__gpu_template(const T input_num,
                                                         int32_t c,
                                                         Column<T>& output) {
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;

  for (int32_t i = start; i < c; i += step) {
    output[i] = input_num;
  }
  return c;
}

// explicit instantiations
template TEMPLATE_NOINLINE int32_t
ct_templated_no_cursor_user_constant_sizer__gpu_template(const int32_t input_num,
                                                         int32_t c,
                                                         Column<int32_t>& output);
template TEMPLATE_NOINLINE int32_t
ct_templated_no_cursor_user_constant_sizer__gpu_template(const float input_num,
                                                         int32_t c,
                                                         Column<float>& output);

#endif  //__CUDACC__

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t ct_require__cpu_(const Column<int32_t>& input1,
                                                 const int32_t i,
                                                 Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 3;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_str__cpu_(const Column<int32_t>& input1,
                                                     const TextEncodingNone& s,
                                                     Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 3;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_mgr(TableFunctionManager& mgr,
                                               const Column<int32_t>& input1,
                                               const int32_t i,
                                               Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 4;
  return 1;
}

template <typename T, typename K>
NEVER_INLINE HOST int32_t ct_require_templating__cpu_template(const Column<T>& input1,
                                                              const int32_t i,
                                                              Column<K>& out) {
  set_output_row_size(1);
  if constexpr (std::is_same<T, int32_t>::value) {
    out[0] = 5;
  } else if constexpr (std::is_same<T, double>::value) {
    out[0] = 6.0;
  }
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_require_templating__cpu_template(const Column<int>& input1,
                                    const int32_t i,
                                    Column<int>& out);
template NEVER_INLINE HOST int32_t
ct_require_templating__cpu_template(const Column<double>& input1,
                                    const int32_t i,
                                    Column<int>& out);

EXTENSION_NOINLINE_HOST int32_t ct_require_and__cpu_(const Column<int32_t>& input1,
                                                     const int32_t i,
                                                     Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 7;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_or_str__cpu_(const Column<int32_t>& input1,
                                                        const TextEncodingNone& i,
                                                        Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 8;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_str_diff__cpu_(const Column<int32_t>& input1,
                                                          const TextEncodingNone& i,
                                                          Column<int32_t>& out) {
  set_output_row_size(1);
  out[0] = 9;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t
ct_require_text_enc_dict__cpu_(const Column<TextEncodingDict>& input,
                               const int64_t x,
                               Column<int64_t>& out) {
  set_output_row_size(1);
  out[0] = 10;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t
ct_require_text_collist_enc_dict__cpu_(const ColumnList<TextEncodingDict>& input,
                                       const int64_t x,
                                       Column<int64_t>& out) {
  set_output_row_size(1);
  out[0] = 11;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_require_cursor__cpu_(const Column<int64_t>& input,
                                                        int64_t y,
                                                        Column<int64_t>& out) {
  set_output_row_size(1);
  out[0] = 12;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_test_allocator(TableFunctionManager& mgr,
                                                  const Column<int32_t>& input,
                                                  const TextEncodingNone& t,
                                                  Column<int32_t>& out) {
  mgr.enable_output_allocations();
  mgr.set_output_row_size(1);
  TextEncodingNone t2(mgr, t.getString());
  out[0] = 11;
  return 1;
}

#endif  // #ifndef __CUDACC__

#ifdef __CUDACC__

EXTENSION_NOINLINE int32_t ct_require_device_cuda__gpu_(const Column<int32_t>& input1,
                                                        const int32_t i,
                                                        Column<int32_t>& out) {
  out[0] = (i > 0 ? 12345 : 54321);
  return 1;
}

#endif  //__CUDACC__

EXTENSION_NOINLINE int32_t ct_test_preflight_sizer(const Column<int32_t>& input,
                                                   const int32_t i,
                                                   const int32_t j,
                                                   Column<int32_t>& out) {
  out[0] = 123;
  out[1] = 456;
  return out.size();
}

EXTENSION_NOINLINE int32_t ct_test_preflight_sizer_const(const Column<int32_t>& input,
                                                         Column<int32_t>& out) {
  out[0] = 789;
  out[1] = 321;
  return out.size();
}

#ifndef __CUDACC__

EXTENSION_NOINLINE int32_t
ct_test_preflight_singlecursor_qe227__cpu_(TableFunctionManager& mgr,
                                           const Column<int32_t>& col,
                                           const ColumnList<int32_t>& lst,
                                           const int x,
                                           const int y,
                                           Column<int32_t>& out) {
  mgr.set_output_row_size(lst.numCols() + 1);
  out[0] = col[0];
  for (int i = 0; i < lst.numCols(); i++) {
    out[i + 1] = lst[i][0];
  }
  return out.size();
}

EXTENSION_NOINLINE int32_t
ct_test_preflight_multicursor_qe227__cpu_(TableFunctionManager& mgr,
                                          const Column<int32_t>& col,
                                          const ColumnList<int32_t>& lst,
                                          const int x,
                                          const int y,
                                          Column<int32_t>& out) {
  mgr.set_output_row_size(lst.numCols() + 1);
  out[0] = col[1];
  for (int i = 0; i < lst.numCols(); i++) {
    out[i + 1] = lst[i][1];
  }
  return out.size();
}

EXTENSION_NOINLINE_HOST int32_t ct_scalar_named_args__cpu_(TableFunctionManager& mgr,
                                                           const int32_t arg1,
                                                           const int32_t arg2,
                                                           Column<int32_t>& out1,
                                                           Column<int32_t>& out2) {
  mgr.set_output_row_size(1);
  out1[0] = arg1;
  out2[0] = arg2;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t
ct_cursor_named_args__cpu_(TableFunctionManager& mgr,
                           const Column<int32_t>& input_arg1,
                           const Column<int32_t>& input_arg2,
                           const int32_t arg1,
                           const int32_t arg2,
                           Column<int32_t>& out1,
                           Column<int32_t>& out2) {
  const int32_t num_rows = input_arg1.size();
  mgr.set_output_row_size(num_rows);
  for (int32_t r = 0; r < num_rows; ++r) {
    out1[r] = input_arg1[r] + arg1;
    out2[r] = input_arg2[r] + arg2;
  }
  return num_rows;
}

// Test table functions overloaded on scalar types
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T>
NEVER_INLINE HOST int32_t ct_overload_scalar_test__cpu_template(const T scalar,
                                                                Column<T>& out) {
  set_output_row_size(1);
  if constexpr (std::is_same<T, int64_t>::value) {
    out[0] = scalar;
  } else if constexpr (std::is_same<T, Timestamp>::value) {
    out[0] = scalar.time;
  }
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_overload_scalar_test__cpu_template(const Timestamp scalar, Column<Timestamp>& out);

template NEVER_INLINE HOST int32_t
ct_overload_scalar_test__cpu_template(const int64_t scalar, Column<int64_t>& out);

// Test table functions overloaded on column types
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T>
NEVER_INLINE HOST int32_t ct_overload_column_test__cpu_template(const Column<T>& input,
                                                                Column<T>& out) {
  int64_t size = input.size();
  set_output_row_size(size);
  for (int64_t i = 0; i < size; ++i) {
    if (input.isNull(i)) {
      out.setNull(i);
    } else {
      out[i] = input[i];
    }
  }
  return size;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_overload_column_test__cpu_template(const Column<Timestamp>& input,
                                      Column<Timestamp>& out);

template NEVER_INLINE HOST int32_t
ct_overload_column_test__cpu_template(const Column<TextEncodingDict>& input,
                                      Column<TextEncodingDict>& out);

template NEVER_INLINE HOST int32_t
ct_overload_column_test__cpu_template(const Column<int64_t>& input, Column<int64_t>& out);

template NEVER_INLINE HOST int32_t
ct_overload_column_test__cpu_template(const Column<Array<TextEncodingDict>>& input,
                                      Column<Array<TextEncodingDict>>& out);

template NEVER_INLINE HOST int32_t
ct_overload_column_test__cpu_template(const Column<Array<int64_t>>& input,
                                      Column<Array<int64_t>>& out);

// Test Calcite overload resolution for table functions with ColumnList arguments
// Calcite should pick the proper overloaded operator for each templated table function
template <typename T, typename K>
NEVER_INLINE HOST int32_t
ct_overload_column_list_test__cpu_template(const Column<K>& first_col,
                                           const ColumnList<T>& col_list,
                                           const Column<K>& last_col,
                                           Column<K>& out) {
  set_output_row_size(1);
  int64_t num_cols = col_list.numCols();
  T sum = 0;
  for (int64_t i = 0; i < num_cols; ++i) {
    const Column<T>& col = col_list[i];
    for (int64_t j = 0; j < col.size(); ++j) {
      sum += col[j];
    }
  }
  if (sum > 0) {
    out[0] = first_col[0];
  } else {
    out[0] = last_col[0];
  }
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_overload_column_list_test__cpu_template(const Column<int64_t>& first_col,
                                           const ColumnList<int64_t>& col_list,
                                           const Column<int64_t>& last_col,
                                           Column<int64_t>& out);

template NEVER_INLINE HOST int32_t
ct_overload_column_list_test__cpu_template(const Column<int64_t>& first_col,
                                           const ColumnList<double>& col_list,
                                           const Column<int64_t>& last_col,
                                           Column<int64_t>& out);

// Test Calcite overload resolution for table functions with multiple ColumnList arguments
template <typename T, typename K>
NEVER_INLINE HOST int32_t
ct_overload_column_list_test2__cpu_template(const Column<K>& first_col,
                                            const ColumnList<K>& col_list1,
                                            const ColumnList<T>& col_list2,
                                            const Column<T>& last_col,
                                            Column<K>& out) {
  set_output_row_size(1);
  int64_t num_cols = col_list1.numCols();
  K sum = 0;
  for (int64_t i = 0; i < num_cols; ++i) {
    const Column<K>& col = col_list1[i];
    for (int64_t j = 0; j < col.size(); ++j) {
      sum += col[j];
    }
  }

  int64_t num_cols2 = col_list2.numCols();
  T sum2 = 0;
  for (int64_t i = 0; i < num_cols2; ++i) {
    const Column<T>& col = col_list2[i];
    for (int64_t j = 0; j < col.size(); ++j) {
      sum2 += col[j];
    }
  }

  if (sum + sum2 > 0) {
    out[0] = first_col[0];
  } else {
    out[0] = last_col[0];
  }
  return 1;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_overload_column_list_test2__cpu_template(const Column<int64_t>& first_col,
                                            const ColumnList<int64_t>& col_list1,
                                            const ColumnList<double>& col_list2,
                                            const Column<double>& last_col,
                                            Column<int64_t>& out);

template NEVER_INLINE HOST int32_t
ct_overload_column_list_test2__cpu_template(const Column<int64_t>& first_col,
                                            const ColumnList<int64_t>& col_list1,
                                            const ColumnList<int64_t>& col_list2,
                                            const Column<int64_t>& last_col,
                                            Column<int64_t>& out);

#endif  // ifndef __CUDACC__

EXTENSION_NOINLINE int32_t ct_require_range__cpu_(const Column<int32_t>& input1,
                                                  const int32_t x,
                                                  const int32_t multiplier,
                                                  Column<int32_t>& out) {
  out[0] = 0;
  return 1;
}

template <typename T>
TEMPLATE_NOINLINE int32_t ct_test_int_default_arg__template(const Column<T>& inp,
                                                            const T x,
                                                            const int32_t multiplier,
                                                            Column<T>& out) {
  int32_t size = inp.size();
  for (int i = 0; i < size; ++i) {
    out[i] = inp[i] * x;
  }

  return size;
}

// explicit instantiations
template TEMPLATE_NOINLINE int32_t
ct_test_int_default_arg__template(const Column<int8_t>& inp,
                                  const int8_t x,
                                  const int32_t multiplier,
                                  Column<int8_t>& out);

template TEMPLATE_NOINLINE int32_t
ct_test_int_default_arg__template(const Column<int16_t>& inp,
                                  const int16_t x,
                                  const int32_t multiplier,
                                  Column<int16_t>& out);

template TEMPLATE_NOINLINE int32_t
ct_test_int_default_arg__template(const Column<int32_t>& inp,
                                  const int32_t x,
                                  const int32_t multiplier,
                                  Column<int32_t>& out);

template TEMPLATE_NOINLINE int32_t
ct_test_int_default_arg__template(const Column<int64_t>& inp,
                                  const int64_t x,
                                  const int32_t multiplier,
                                  Column<int64_t>& out);

template <typename T>
TEMPLATE_NOINLINE int32_t ct_test_float_default_arg__template(const Column<T>& inp,
                                                              const T x,
                                                              const int32_t multiplier,
                                                              Column<T>& out) {
  int32_t size = inp.size();
  for (int i = 0; i < size; ++i) {
    out[i] = inp[i] / x;
  }

  return size;
}

// explicit instantiations
template TEMPLATE_NOINLINE int32_t
ct_test_float_default_arg__template(const Column<float>& inp,
                                    const float x,
                                    const int32_t multiplier,
                                    Column<float>& out);

template TEMPLATE_NOINLINE int32_t
ct_test_float_default_arg__template(const Column<double>& inp,
                                    const double x,
                                    const int32_t multiplier,
                                    Column<double>& out);

#ifndef __CUDACC__

EXTENSION_NOINLINE int32_t
ct_test_string_default_arg__cpu_(TableFunctionManager& mgr,
                                 const Column<TextEncodingDict>& inp,
                                 const TextEncodingNone& suffix,
                                 Column<TextEncodingDict>& out) {
  int32_t size = inp.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    std::string output_string = inp.getString(i);
    output_string += suffix.getString();
    const TextEncodingDict concatted_id = out.getOrAddTransient(output_string);
    out[i] = concatted_id;
  }

  return size;
}

EXTENSION_NOINLINE int32_t
ct_test_calcite_casting_char__cpu_(TableFunctionManager& mgr,
                                   const Column<TextEncodingDict>& input1,
                                   Column<TextEncodingDict>& out) {
  int32_t size = input1.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    out[i] = input1[i];
  }
  return size;
}

EXTENSION_NOINLINE int32_t
ct_test_calcite_casting_bigint__cpu_(TableFunctionManager& mgr,
                                     const Column<int64_t>& input1,
                                     Column<int64_t>& out) {
  int32_t size = input1.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    out[i] = input1[i];
  }
  return size;
}

EXTENSION_NOINLINE int32_t
ct_test_calcite_casting_double__cpu_(TableFunctionManager& mgr,
                                     const Column<double>& input1,
                                     Column<double>& out) {
  int32_t size = input1.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    out[i] = input1[i];
  }
  return size;
}

EXTENSION_NOINLINE int32_t
ct_test_calcite_casting_timestamp__cpu_(TableFunctionManager& mgr,
                                        const Column<Timestamp>& input1,
                                        Column<Timestamp>& out) {
  int32_t size = input1.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    out[i] = input1[i];
  }
  return size;
}

template <typename T>
TEMPLATE_NOINLINE int32_t
ct_test_calcite_casting_columnlist__template_cpu_(TableFunctionManager& mgr,
                                                  const Column<T>& first,
                                                  const ColumnList<T>& list,
                                                  Column<T>& out) {
  mgr.set_output_row_size(list.numCols() + 1);
  out[0] = first[0];
  for (int i = 0; i < list.numCols(); i++) {
    out[i + 1] = list[i][0];
  }
  return out.size();
}

// explicit instantiations
template TEMPLATE_NOINLINE int32_t
ct_test_calcite_casting_columnlist__template_cpu_(TableFunctionManager& mgr,
                                                  const Column<float>& first,
                                                  const ColumnList<float>& list,
                                                  Column<float>& out);

template TEMPLATE_NOINLINE int32_t
ct_test_calcite_casting_columnlist__template_cpu_(TableFunctionManager& mgr,
                                                  const Column<double>& first,
                                                  const ColumnList<double>& list,
                                                  Column<double>& out);

#endif