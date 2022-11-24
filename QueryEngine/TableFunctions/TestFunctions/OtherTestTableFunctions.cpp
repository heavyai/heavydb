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
  This file contains generic testing compile-time UDTFs. These are
  table functions that do not belong to other specific classes.
  Usually these are performance or specific feature tests, such as
  tests for specific data types or query plan rules.
 */

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t ct_binding_scalar_multiply__cpu_template(const Column<T>& input,
                                                                   const T multiplier,
                                                                   Column<T>& out) {
  const int64_t num_rows = input.size();
  set_output_row_size(num_rows);
  for (int64_t r = 0; r < num_rows; ++r) {
    if (!input.isNull(r)) {
      out[r] = input[r] * multiplier;
    } else {
      out.setNull(r);
    }
  }
  return num_rows;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_binding_scalar_multiply__cpu_template(const Column<float>& input,
                                         const float multiplier,
                                         Column<float>& out);
template NEVER_INLINE HOST int32_t
ct_binding_scalar_multiply__cpu_template(const Column<double>& input,
                                         const double multiplier,
                                         Column<double>& out);
template NEVER_INLINE HOST int32_t
ct_binding_scalar_multiply__cpu_template(const Column<int32_t>& input,
                                         const int32_t multiplier,
                                         Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
ct_binding_scalar_multiply__cpu_template(const Column<int64_t>& input,
                                         const int64_t multiplier,
                                         Column<int64_t>& out);

#ifndef __CUDACC__

#include <algorithm>

template <typename T>
struct SortAsc {
  SortAsc(const bool nulls_last)
      : null_value_(std::numeric_limits<T>::lowest())
      , null_value_mapped_(map_null_value(nulls_last)) {}
  static T map_null_value(const bool nulls_last) {
    return nulls_last ? std::numeric_limits<T>::max() : std::numeric_limits<T>::lowest();
  }
  inline T mapValue(const T& val) {
    return val == null_value_ ? null_value_mapped_ : val;
  }
  bool operator()(const T& a, const T& b) { return mapValue(a) < mapValue(b); }
  const T null_value_;
  const T null_value_mapped_;
};

template <typename T>
struct SortDesc {
  SortDesc(const bool nulls_last)
      : null_value_(std::numeric_limits<T>::lowest())
      , null_value_mapped_(map_null_value(nulls_last)) {}
  static T map_null_value(const bool nulls_last) {
    return nulls_last ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
  }

  inline T mapValue(const T& val) {
    return val == null_value_ ? null_value_mapped_ : val;
  }

  bool operator()(const T& a, const T& b) { return mapValue(a) > mapValue(b); }
  const T null_value_;
  const T null_value_mapped_;
};

template <typename T>
NEVER_INLINE HOST int32_t sort_column_limit__cpu_template(const Column<T>& input,
                                                          const int32_t limit,
                                                          const bool sort_ascending,
                                                          const bool nulls_last,
                                                          Column<T>& output) {
  const int64_t num_rows = input.size();
  set_output_row_size(num_rows);
  output = input;
  if (sort_ascending) {
    std::sort(output.ptr_, output.ptr_ + num_rows, SortAsc<T>(nulls_last));
  } else {
    std::sort(output.ptr_, output.ptr_ + num_rows, SortDesc<T>(nulls_last));
  }
  if (limit < 0 || limit > num_rows) {
    return num_rows;
  }
  return limit;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
sort_column_limit__cpu_template(const Column<int8_t>& input,
                                const int32_t limit,
                                const bool sort_ascending,
                                const bool nulls_last,
                                Column<int8_t>& output);
template NEVER_INLINE HOST int32_t
sort_column_limit__cpu_template(const Column<int16_t>& input,
                                const int32_t limit,
                                const bool sort_ascending,
                                const bool nulls_last,
                                Column<int16_t>& output);
template NEVER_INLINE HOST int32_t
sort_column_limit__cpu_template(const Column<int32_t>& input,
                                const int32_t limit,
                                const bool sort_ascending,
                                const bool nulls_last,
                                Column<int32_t>& output);
template NEVER_INLINE HOST int32_t
sort_column_limit__cpu_template(const Column<int64_t>& input,
                                const int32_t limit,
                                const bool sort_ascending,
                                const bool nulls_last,
                                Column<int64_t>& output);
template NEVER_INLINE HOST int32_t
sort_column_limit__cpu_template(const Column<float>& input,
                                const int32_t limit,
                                const bool sort_ascending,
                                const bool nulls_last,
                                Column<float>& output);
template NEVER_INLINE HOST int32_t
sort_column_limit__cpu_template(const Column<double>& input,
                                const int32_t limit,
                                const bool sort_ascending,
                                const bool nulls_last,
                                Column<double>& output);

#endif

template <typename T>
T safe_addition(T x, T y) {
  if (x >= 0) {
    if (y > (std::numeric_limits<T>::max() - x)) {
      throw std::overflow_error("Addition overflow detected");
    }
  } else {
    if (y < (std::numeric_limits<T>::min() - x)) {
      throw std::underflow_error("Addition underflow detected");
    }
  }
  return x + y;
}

#ifndef __CUDACC__

template <typename T>
NEVER_INLINE HOST int32_t
column_list_safe_row_sum__cpu_template(const ColumnList<T>& input, Column<T>& out) {
  int32_t output_num_rows = input.numCols();
  set_output_row_size(output_num_rows);
  for (int i = 0; i < output_num_rows; i++) {
    auto col = input[i];
    T s = 0;
    for (int j = 0; j < col.size(); j++) {
      try {
        s = safe_addition(s, col[j]);
      } catch (const std::exception& e) {
        return TABLE_FUNCTION_ERROR(e.what());
      } catch (...) {
        return TABLE_FUNCTION_ERROR("Unknown error");
      }
    }
    out[i] = s;
  }
  return output_num_rows;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
column_list_safe_row_sum__cpu_template(const ColumnList<int32_t>& input,
                                       Column<int32_t>& out);
template NEVER_INLINE HOST int32_t
column_list_safe_row_sum__cpu_template(const ColumnList<int64_t>& input,
                                       Column<int64_t>& out);
template NEVER_INLINE HOST int32_t
column_list_safe_row_sum__cpu_template(const ColumnList<float>& input,
                                       Column<float>& out);
template NEVER_INLINE HOST int32_t
column_list_safe_row_sum__cpu_template(const ColumnList<double>& input,
                                       Column<double>& out);

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE int32_t ct_sleep_worker(int32_t seconds, Column<int32_t>& output) {
  // save entering time
  output[0] = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count() &
              0xffffff;
  // store thread id info
  output[2] = std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xffff;
  // do "computations" for given seconds
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  // save leaving time
  output[1] = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count() &
              0xffffff;
  return 3;
}

EXTENSION_NOINLINE_HOST int32_t ct_sleep1__cpu_(int32_t seconds,
                                                int32_t mode,
                                                Column<int32_t>& output) {
  switch (mode) {
    case 0: {
      set_output_row_size(3);  // uses global singleton of TableFunctionManager
      break;
    }
    case 1: {
      auto* mgr = TableFunctionManager::get_singleton();
      mgr->set_output_row_size(3);
      break;
    }
    case 2:
    case 3: {
      break;
    }
    default:
      return TABLE_FUNCTION_ERROR("unexpected mode");
  }
  if (output.size() == 0) {
    return TABLE_FUNCTION_ERROR("unspecified output columns row size");
  }
  return ct_sleep_worker(seconds, output);
}

EXTENSION_NOINLINE_HOST int32_t ct_sleep2(TableFunctionManager& mgr,
                                          int32_t seconds,
                                          int32_t mode,
                                          Column<int32_t>& output) {
  switch (mode) {
    case 0:
    case 1: {
      mgr.set_output_row_size(3);  // uses thread-safe TableFunctionManager instance
      break;
    }
    case 2: {
      break;
    }
    case 3: {
      try {
        auto* mgr0 = TableFunctionManager::get_singleton();  // it may throw "singleton is
                                                             // not initialized"
        mgr0->set_output_row_size(3);
      } catch (std::exception& e) {
        return mgr.ERROR_MESSAGE(e.what());
      }
      break;
    }
    default:
      return mgr.ERROR_MESSAGE("unexpected mode");
  }
  if (output.size() == 0) {
    return mgr.ERROR_MESSAGE("unspecified output columns row size");
  }
  return ct_sleep_worker(seconds, output);
}

template <typename T>
NEVER_INLINE HOST int32_t ct_throw_if_gt_100__cpu_template(TableFunctionManager& mgr,
                                                           const Column<T>& input,
                                                           Column<T>& output) {
  int64_t num_rows = input.size();
  mgr.set_output_row_size(num_rows);
  for (int64_t r = 0; r < num_rows; ++r) {
    if (input[r] > 100) {
      return mgr.ERROR_MESSAGE("Values greater than 100 not allowed");
    }
    output[r] = input[r];
  }
  return num_rows;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_throw_if_gt_100__cpu_template(TableFunctionManager& mgr,
                                 const Column<float>& input,
                                 Column<float>& output);
template NEVER_INLINE HOST int32_t
ct_throw_if_gt_100__cpu_template(TableFunctionManager& mgr,
                                 const Column<double>& input,
                                 Column<double>& output);

EXTENSION_NOINLINE_HOST int32_t ct_copy_and_add_size(TableFunctionManager& mgr,
                                                     const Column<int32_t>& input,
                                                     Column<int32_t>& output) {
  mgr.set_output_row_size(input.size());
  for (int32_t i = 0; i < input.size(); i++) {
    output[i] = input[i] + input.size();
  }
  return output.size();
}

EXTENSION_NOINLINE_HOST int32_t ct_add_size_and_mul_alpha(TableFunctionManager& mgr,
                                                          const Column<int32_t>& input1,
                                                          const Column<int32_t>& input2,
                                                          int32_t alpha,
                                                          Column<int32_t>& output1,
                                                          Column<int32_t>& output2) {
  auto size = input1.size();
  mgr.set_output_row_size(size);
  for (int32_t i = 0; i < size; i++) {
    output1[i] = input1[i] + size;
    output2[i] = input2[i] * alpha;
  }
  return size;
}

/*
  Add two sparse graphs given by pairs of coordinates and the
  corresponding values and multiply with the size of output
  columns. Unspecified points are assumed to have the specified fill
  value.
*/

EXTENSION_NOINLINE_HOST int32_t ct_sparse_add(TableFunctionManager& mgr,
                                              const Column<int32_t>& x1,
                                              const Column<int32_t>& d1,
                                              int32_t f1,
                                              const Column<int32_t>& x2,
                                              const Column<int32_t>& d2,
                                              int32_t f2,
                                              Column<int32_t>& x,
                                              Column<int32_t>& d) {
  // sorted set of common coordinates:
  std::set<int32_t, std::less<int32_t>> x12;
  // inverse map of coordinates and indices, keys are sorted:
  std::map<int32_t, int32_t, std::less<int32_t>> i1, i2;

  for (int32_t i = 0; i < x1.size(); i++) {
    i1[x1[i]] = i;
    x12.insert(x1[i]);
  }
  for (int32_t i = 0; i < x2.size(); i++) {
    i2[x2[i]] = i;
    x12.insert(x2[i]);
  }
  auto size = x12.size();

  mgr.set_output_row_size(size);
  int32_t k = 0;
  for (auto x_ : x12) {
    x[k] = x_;
    auto i1_ = i1.find(x_);
    auto i2_ = i2.find(x_);
    if (i1_ != i1.end()) {
      if (i2_ != i2.end()) {
        d[k] = d1[i1_->second] + d2[i2_->second];
      } else {
        d[k] = d1[i1_->second] + f2;
      }
    } else if (i2_ != i2.end()) {
      d[k] = f1 + d2[i2_->second];
    } else {
      d[k] = f1 + f2;
    }
    d[k] *= size;
    k++;
  }
  return size;
}

#endif  // #ifndef __CUDACC__

#ifdef __CUDACC__

EXTENSION_NOINLINE int32_t
ct_cuda_enumerate_threads__gpu_(const int32_t output_size,
                                Column<int32_t>& out_local_thread_id,
                                Column<int32_t>& out_block_id,
                                Column<int32_t>& out_global_thread_id) {
  int32_t local_thread_id = threadIdx.x;
  int32_t block_id = blockIdx.x;
  int32_t global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  out_local_thread_id[global_thread_id] = local_thread_id;
  out_block_id[global_thread_id] = block_id;
  out_global_thread_id[global_thread_id] = global_thread_id;
  return output_size;
}

#endif  //__CUDACC__

EXTENSION_NOINLINE int32_t ct_test_nullable(const Column<int32_t>& input,
                                            const int32_t i,
                                            Column<int32_t>& out) {
  for (int i = 0; i < input.size(); i++) {
    if (i % 2 == 0) {
      out.setNull(i);
    } else {
      out[i] = input[i];
    }
  }
  return input.size();
}

#ifndef __CUDACC__

// Test table functions with Timestamp Column inputs
// and Timestamp type helper functions
EXTENSION_NOINLINE_HOST int32_t ct_timestamp_extract(TableFunctionManager& mgr,
                                                     const Column<Timestamp>& input,
                                                     Column<int64_t>& ns,
                                                     Column<int64_t>& us,
                                                     Column<int64_t>& ms,
                                                     Column<int64_t>& s,
                                                     Column<int64_t>& m,
                                                     Column<int64_t>& h,
                                                     Column<int64_t>& d,
                                                     Column<int64_t>& mo,
                                                     Column<int64_t>& y) {
  int size = input.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    if (input.isNull(i)) {
      ns.setNull(i);
      us.setNull(i);
      ms.setNull(i);
      s.setNull(i);
      m.setNull(i);
      h.setNull(i);
      d.setNull(i);
      mo.setNull(i);
      y.setNull(i);
    } else {
      ns[i] = input[i].getNanoseconds();
      us[i] = input[i].getMicroseconds();
      ms[i] = input[i].getMilliseconds();
      s[i] = input[i].getSeconds();
      m[i] = input[i].getMinutes();
      h[i] = input[i].getHours();
      d[i] = input[i].getDay();
      mo[i] = input[i].getMonth();
      y[i] = input[i].getYear();
    }
  }
  return size;
}

// Test table functions with scalar Timestamp inputs
EXTENSION_NOINLINE_HOST int32_t ct_timestamp_add_offset(TableFunctionManager& mgr,
                                                        const Column<Timestamp>& input,
                                                        const Timestamp offset,
                                                        Column<Timestamp>& out) {
  int size = input.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    if (input.isNull(i)) {
      out.setNull(i);
    } else {
      out[i] = input[i] + offset;
    }
  }
  return size;
}

// Test table function with sizer argument, and mix of scalar/column inputs.
EXTENSION_NOINLINE int32_t
ct_timestamp_test_columns_and_scalars__cpu(const Column<Timestamp>& input,
                                           const int64_t dummy,
                                           const int32_t multiplier,
                                           const Column<Timestamp>& input2,
                                           Column<Timestamp>& out) {
  int size = input.size();
  for (int i = 0; i < size; ++i) {
    if (input.isNull(i)) {
      out.setNull(i);
    } else {
      out[i] = input[i] + input2[i] + Timestamp(dummy);
    }
  }
  return size;
}

// Dummy test for ColumnList inputs + Column Timestamp input
EXTENSION_NOINLINE_HOST int32_t
ct_timestamp_column_list_input(TableFunctionManager& mgr,
                               const ColumnList<int64_t>& input,
                               const Column<Timestamp>& input2,
                               Column<int64_t>& out) {
  mgr.set_output_row_size(1);
  out[0] = 1;
  return 1;
}

EXTENSION_NOINLINE_HOST int32_t ct_timestamp_truncate(TableFunctionManager& mgr,
                                                      const Column<Timestamp>& input,
                                                      Column<Timestamp>& y,
                                                      Column<Timestamp>& mo,
                                                      Column<Timestamp>& d,
                                                      Column<Timestamp>& h,
                                                      Column<Timestamp>& m,
                                                      Column<Timestamp>& s,
                                                      Column<Timestamp>& ms,
                                                      Column<Timestamp>& us) {
  int size = input.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    y[i] = input[i].truncateToYear();
    mo[i] = input[i].truncateToMonth();
    d[i] = input[i].truncateToDay();
    h[i] = input[i].truncateToHours();
    m[i] = input[i].truncateToMinutes();
    s[i] = input[i].truncateToSeconds();
    ms[i] = input[i].truncateToMilliseconds();
    us[i] = input[i].truncateToMicroseconds();
  }

  return size;
}

template <typename T>
NEVER_INLINE HOST int32_t
ct_timestamp_add_interval__template(TableFunctionManager& mgr,
                                    const Column<Timestamp>& input,
                                    const T inter,
                                    Column<Timestamp>& out) {
  int size = input.size();
  mgr.set_output_row_size(size);
  for (int i = 0; i < size; ++i) {
    out[i] = inter + input[i];
  }
  return size;
}

// explicit instantiations
template NEVER_INLINE HOST int32_t
ct_timestamp_add_interval__template(TableFunctionManager& mgr,
                                    const Column<Timestamp>& input,
                                    const DayTimeInterval inter,
                                    Column<Timestamp>& out);

template NEVER_INLINE HOST int32_t
ct_timestamp_add_interval__template(TableFunctionManager& mgr,
                                    const Column<Timestamp>& input,
                                    const YearMonthTimeInterval inter,
                                    Column<Timestamp>& out);

#endif  // ifndef __CUDACC__

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

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t row_copier2__cpu__(const Column<double>& input_col,
                                                   int copy_multiplier,
                                                   Column<double>& output_col,
                                                   Column<double>& output_col2) {
  if (copy_multiplier == -1) {
    // Test UDTF return without allocating output columns, expect
    // empty output columns.
    return 0;
  }
  if (copy_multiplier == -2) {
    // Test UDTF return without allocating output columns but
    // returning positive row size, expect runtime error.
    return 1;
  }
#ifndef __CUDACC__
  if (copy_multiplier == -3) {
    // Test UDTF throw before allocating output columns, expect
    // runtime error.
    throw std::runtime_error("row_copier2: throw before calling set_output_row_size");
  }
  if (copy_multiplier == -4) {
    // Test UDTF throw after allocating output columns, expect
    // runtime error.
    set_output_row_size(1);
    throw std::runtime_error("row_copier2: throw after calling set_output_row_size");
  }
#endif
  if (copy_multiplier == -5) {
    // Test UDTF setting negative row size, expect runtime error.
    set_output_row_size(-1);
  }
  int32_t output_row_count = copy_multiplier * input_col.size();
  /* set_output_row_size can be used only when an UDTF is executed on CPU */
  set_output_row_size(output_row_count);
  auto result = row_copier(input_col, copy_multiplier, output_col);
  if (result >= 0) {
    result = row_copier(input_col, copy_multiplier, output_col2);
  }
  return result;
}

#endif  // #ifndef __CUDACC__

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

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
get_max_with_row_offset__cpu_(const Column<int>& input_col,
                              Column<int>& output_max_col,
                              Column<int>& output_max_row_col) {
  if ((output_max_col.size() != 1) || output_max_row_col.size() != 1) {
    return -1;
  }
  auto start = 0;
  auto stop = input_col.size();
  auto step = 1;

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

#endif  // #ifndef __CUDACC__

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t column_list_get__cpu_(const ColumnList<double>& col_list,
                                                      const int index,
                                                      const int m,
                                                      Column<double>& col) {
  col = col_list[index];  // copy the data of col_list item to output column
  return col.size();
}

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE int32_t column_list_first_last(const ColumnList<double>& col_list,
                                                  const int m,
                                                  Column<double>& col1,
                                                  Column<double>& col2) {
  col1 = col_list[0];
  col2 = col_list[col_list.numCols() - 1];
  return col1.size();
}

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST int32_t
column_list_row_sum__cpu_(const ColumnList<int32_t>& input, Column<int32_t>& out) {
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

NEVER_INLINE HOST int32_t tf_metadata_setter__cpu_template(TableFunctionManager& mgr,
                                                           Column<bool>& success) {
  // set one of each type
  mgr.set_metadata("test_int8_t", int8_t(1));
  mgr.set_metadata("test_int16_t", int16_t(2));
  mgr.set_metadata("test_int32_t", int32_t(3));
  mgr.set_metadata("test_int64_t", int64_t(4));
  mgr.set_metadata("test_float", 5.0f);
  mgr.set_metadata("test_double", 6.0);
  mgr.set_metadata("test_bool", true);

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

NEVER_INLINE HOST int32_t
tf_metadata_setter_repeated__cpu_template(TableFunctionManager& mgr,
                                          Column<bool>& success) {
  // set the same name twice
  mgr.set_metadata("test_int8_t", int8_t(1));
  mgr.set_metadata("test_int8_t", int8_t(2));

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

NEVER_INLINE HOST int32_t
tf_metadata_setter_size_mismatch__cpu_template(TableFunctionManager& mgr,
                                               Column<bool>& success) {
  // set the same name twice
  mgr.set_metadata("test_int8_t", int8_t(1));
  mgr.set_metadata("test_int8_t", int16_t(2));

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

NEVER_INLINE HOST int32_t tf_metadata_getter__cpu_template(TableFunctionManager& mgr,
                                                           const Column<bool>& input,
                                                           Column<bool>& success) {
  // get them all back and check values
  int8_t i8{};
  int16_t i16{};
  int32_t i32{};
  int64_t i64{};
  float f{};
  double d{};
  bool b{};

  try {
    mgr.get_metadata("test_int8_t", i8);
    mgr.get_metadata("test_int16_t", i16);
    mgr.get_metadata("test_int32_t", i32);
    mgr.get_metadata("test_int64_t", i64);
    mgr.get_metadata("test_float", f);
    mgr.get_metadata("test_double", d);
    mgr.get_metadata("test_bool", b);
  } catch (const std::runtime_error& ex) {
    return mgr.ERROR_MESSAGE(ex.what());
  }

  // return value indicates values were correct
  // types are implicitly correct by this point, or the above would have thrown
  bool result = (i8 == 1) && (i16 == 2) && (i32 == 3) && (i64 == 4) && (f == 5.0f) &&
                (d == 6.0) && b;
  if (!result) {
    return mgr.ERROR_MESSAGE("Metadata return values are incorrect");
  }

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

NEVER_INLINE HOST int32_t tf_metadata_getter_bad__cpu_template(TableFunctionManager& mgr,
                                                               const Column<bool>& input,
                                                               Column<bool>& success) {
  // get one back as the wrong type
  // this should throw
  float f{};
  try {
    mgr.get_metadata("test_double", f);
  } catch (const std::runtime_error& ex) {
    return mgr.ERROR_MESSAGE(ex.what());
  }

  mgr.set_output_row_size(1);
  success[0] = true;
  return 1;
}

#endif  // #ifndef __CUDACC__

EXTENSION_NOINLINE int32_t ct_gpu_default_init__gpu_(Column<int32_t>& output_buffer) {
  // output_buffer[0] should always be 0 due to default initialization for GPU
  return 1;
}