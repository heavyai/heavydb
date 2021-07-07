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

// #include <chrono>
#include <type_traits>
#include "../../Logger/Logger.h"
#include "../../QueryEngine/OmniSciTypes.h"
#include "../../Shared/funcannotations.h"
#include "daal.h"

/*
  UDTF: row_copier(Column<double>, RowMultiplier) -> Column<double>
*/
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

// clang-format off
/*
  UDTF: k_means(Cursor<Column<int>, ColumnList<float>>, int, int, RowMultiplier) -> Column<int>, Column<int>
 */
// clang-format on

EXTENSION_NOINLINE int32_t k_means(const Column<int>& input_ids,
                                   const ColumnList<float>& input,
                                   const int num_clusters,
                                   const int num_iterations,
                                   const int output_multiplier,
                                   Column<int>& output_ids,
                                   Column<int>& output_cluster) {
  using namespace daal::algorithms;
  using namespace daal::data_management;

  // using Clock = std::chrono::steady_clock;

  // const auto t0 = Clock::now();

  // Float data type
  using float_type =
      std::remove_cv_t<std::remove_reference_t<decltype(input)>>::value_type;

  // Assignments data type
  using assignments_type =
      std::remove_cv_t<std::remove_reference_t<decltype(output_cluster)>>::value_type;

  // Data dimensions
  const size_t num_rows = input_ids.size();
  const size_t num_columns = input.numCols();

  // // Arrays to handle parameters in the easy way
  // const Column<float>* const inputs[num_columns] = {
  //     &input_col0,  &input_col1,  &input_col2,  &input_col3,  &input_col4,
  //     &input_col5,  &input_col6,  &input_col7,  &input_col8,  &input_col9,
  //     &input_col10, &input_col11, &input_col12, &input_col13, &input_col14,
  //     &input_col15, &input_col16, &input_col17, &input_col18, &input_col19};
  // const Column<float>* const outputs[num_columns] = {
  //     &output_col0,  &output_col1,  &output_col2,  &output_col3,  &output_col4,
  //     &output_col5,  &output_col6,  &output_col7,  &output_col8,  &output_col9,
  //     &output_col10, &output_col11, &output_col12, &output_col13, &output_col14,
  //     &output_col15, &output_col16, &output_col17, &output_col18, &output_col19};

  // Prepare input data as structure of arrays (SOA) as columnar format (zero-copy)
  const auto dataTable = SOANumericTable::create(num_columns, num_rows);
  for (size_t i = 0; i < num_columns; ++i) {
    dataTable->setArray<float_type>(input[i].ptr_, i);
  }

  // const auto t1 = Clock::now();

  // Initialization phase of K-Means
  kmeans::init::Batch<float_type, kmeans::init::randomDense> init(num_clusters);
  init.input.set(kmeans::init::data, dataTable);
  init.compute();
  const NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

  // Prepare output data as homogeneous numeric table to allow zero-copy for assignments
  const auto assignmentsTable =
      HomogenNumericTable<assignments_type>::create(output_cluster.ptr_, 1, num_rows);
  const kmeans::ResultPtr result(new kmeans::Result);
  result->set(kmeans::assignments, assignmentsTable);
  result->set(kmeans::objectiveFunction,
              HomogenNumericTable<float>::create(1, 1, NumericTable::doAllocate));
  result->set(kmeans::nIterations,
              HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate));

  // Clustering phase of K-Means
  kmeans::Batch<> algorithm(num_clusters, num_iterations);
  algorithm.input.set(kmeans::data, dataTable);
  algorithm.input.set(kmeans::inputCentroids, centroids);
  algorithm.parameter().resultsToEvaluate = kmeans::computeAssignments;
  algorithm.setResult(result);
  algorithm.compute();

  // const auto t2 = Clock::now();

  // // Assign output columns
  // for (size_t i = 0; i < num_rows; ++i) {
  //   for (size_t j = 0; j < num_columns; ++j) {
  //     (*(outputs[j]))[i] = (*(inputs[j]))[i];
  //   }
  // }

  // Copying from input_ids to output_ids
  output_ids = input_ids;

  // const auto t3 = Clock::now();

  // LOG(INFO) << "k_means UDTF finished in "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count()
  //           << " ms";
  // LOG(INFO) << "   k_means inputs preparation took "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
  //           << " ms";
  // LOG(INFO) << "   k_means algorithm took "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  //           << " ms";
  // LOG(INFO) << "   k_means output columns assignment took "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
  //           << " ms";

  return num_rows;
}

// EXTENSION_NOINLINE int32_t db_scan(const Column<double>& input_col0,
//                                   const Column<double>& input_col1,
//                                   const float epsilon,
//                                   const int output_multiplier,
//                                   const Column<double>& output_col0,
//                                   const Column<double>& output_col1,
//                                   const Column<int>& output_cluster) {
//   size_t num_rows = input_col0.getSize();
//   arma::Mat<double> input_data (num_rows, 2);
//   memcpy(input_data.colptr(0), input_col0.ptr, sizeof(double) * num_rows);
//   memcpy(input_data.colptr(1), input_col1.ptr, sizeof(double) * num_rows);
//   arma::Mat<double> input_data_transposed = input_data.t();
//   mlpack::dbscan::DBSCAN<> dbscan(epsilon, 2);
//   arma::Row<size_t> assignments;
//   const size_t clusters = dbscan.Cluster(input_data_transposed, assignments);
//   for (size_t i = 0; i != num_rows; ++i) {
//     output_col0[i] = input_col0[i];
//     output_col1[i] = input_col1[i];
//     output_cluster[i] = assignments[i] == SIZE_MAX ? -1 : assignments[i];
//   }
//   return num_rows;
// }

#include "MLFunctions.hpp"
