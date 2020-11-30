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
#include "daal.h"

#define EXTENSION_INLINE extern "C" ALWAYS_INLINE DEVICE
#define EXTENSION_NOINLINE extern "C" NEVER_INLINE DEVICE

EXTENSION_NOINLINE int32_t row_copier(const Column<double>& input_col,
                                      int copy_multiplier,
                                      Column<double>& output_col) {
  int32_t output_row_count = copy_multiplier * input_col.getSize();
  if (output_row_count > 100) {
    // Test failure propagation.
    return -1;
  }
  if (output_col.getSize() != output_row_count) {
    return -1;
  }

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col.getSize());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col.getSize();
  auto step = 1;
#endif

  for (auto i = start; i < stop; i += step) {
    for (int c = 0; c < copy_multiplier; c++) {
      output_col[i + (c * input_col.getSize())] = input_col[i];
    }
  }

  return output_row_count;
}

EXTENSION_NOINLINE int32_t row_adder(const int copy_multiplier,
                                     const Column<double>& input_col1,
                                     const Column<double>& input_col2,
                                     Column<double>& output_col) {
  int32_t output_row_count = copy_multiplier * input_col1.getSize();
  if (output_row_count > 100) {
    // Test failure propagation.
    return -1;
  }
  if (output_col.getSize() != output_row_count) {
    return -1;
  }

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col1.getSize());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col1.getSize();
  auto step = 1;
#endif
  auto stride = input_col1.getSize();
  for (auto i = start; i < stop; i += step) {
    for (int c = 0; c < copy_multiplier; c++) {
      output_col[i + (c * stride)] = input_col1[i] + input_col2[i];
    }
  }

  return output_row_count;
}

EXTENSION_NOINLINE int32_t row_addsub(const int copy_multiplier,
                                      const Column<double>& input_col1,
                                      const Column<double>& input_col2,
                                      Column<double>& output_col1,
                                      Column<double>& output_col2) {
  int32_t output_row_count = copy_multiplier * input_col1.getSize();
  if (output_row_count > 100) {
    // Test failure propagation.
    return -1;
  }
  if ((output_col1.getSize() != output_row_count) ||
      (output_col2.getSize() != output_row_count)) {
    return -1;
  }

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col1.getSize());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col1.getSize();
  auto step = 1;
#endif
  auto stride = input_col1.getSize();
  for (auto i = start; i < stop; i += step) {
    for (int c = 0; c < copy_multiplier; c++) {
      output_col1[i + (c * stride)] = input_col1[i] + input_col2[i];
      output_col2[i + (c * stride)] = input_col1[i] - input_col2[i];
    }
  }
  return output_row_count;
}

EXTENSION_NOINLINE int32_t get_max_with_row_offset(const Column<int>& input_col,
                                                   Column<int>& output_max_col,
                                                   Column<int>& output_max_row_col) {
  if ((output_max_col.getSize() != 1) || output_max_row_col.getSize() != 1) {
    return -1;
  }
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col.getSize());
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col.getSize();
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

EXTENSION_NOINLINE int32_t k_means(const Column<float>& input_col0,
                                   const Column<float>& input_col1,
                                   const Column<float>& input_col2,
                                   const Column<float>& input_col3,
                                   const Column<float>& input_col4,
                                   const Column<float>& input_col5,
                                   const Column<float>& input_col6,
                                   const Column<float>& input_col7,
                                   const Column<float>& input_col8,
                                   const Column<float>& input_col9,
                                   const Column<float>& input_col10,
                                   const Column<float>& input_col11,
                                   const Column<float>& input_col12,
                                   const Column<float>& input_col13,
                                   const Column<float>& input_col14,
                                   const Column<float>& input_col15,
                                   const Column<float>& input_col16,
                                   const Column<float>& input_col17,
                                   const Column<float>& input_col18,
                                   const Column<float>& input_col19,
                                   const int num_clusters,
                                   const int num_iterations,
                                   const int output_multiplier,
                                   const Column<float>& output_col0,
                                   const Column<float>& output_col1,
                                   const Column<float>& output_col2,
                                   const Column<float>& output_col3,
                                   const Column<float>& output_col4,
                                   const Column<float>& output_col5,
                                   const Column<float>& output_col6,
                                   const Column<float>& output_col7,
                                   const Column<float>& output_col8,
                                   const Column<float>& output_col9,
                                   const Column<float>& output_col10,
                                   const Column<float>& output_col11,
                                   const Column<float>& output_col12,
                                   const Column<float>& output_col13,
                                   const Column<float>& output_col14,
                                   const Column<float>& output_col15,
                                   const Column<float>& output_col16,
                                   const Column<float>& output_col17,
                                   const Column<float>& output_col18,
                                   const Column<float>& output_col19,
                                   const Column<int>& output_cluster) {
  using namespace daal;
  using namespace daal::data_management;

  // Data dimensions
  const size_t num_rows = input_col0.getSize();
  constexpr size_t num_columns = 20;

  // Arrays to handle parameters in the easy way
  const Column<float>& inputs[num_columns] = {
      input_col0,  input_col1,  input_col2,  input_col3,  input_col4,
      input_col5,  input_col6,  input_col7,  input_col8,  input_col9,
      input_col10, input_col11, input_col12, input_col13, input_col14,
      input_col15, input_col16, input_col17, input_col18, input_col19};
  Column<float>& outputs[num_columns] = {
      output_col0,  output_col1,  output_col2,  output_col3,  output_col4,
      output_col5,  output_col6,  output_col7,  output_col8,  output_col9,
      output_col10, output_col11, output_col12, output_col13, output_col14,
      output_col15, output_col16, output_col17, output_col18, output_col19};

  // Prepare input data as structure of arrays (SOA) as columnar format
  SOANumericTablePtr dataTable = SOANumericTable::create(num_columns, num_rows);
  for (size_t i = 0; i < num_columns; ++i) {
    dataTable->setArray<float>(inputs[i].ptr, i);
  }

  // Initialization phase of K-Means
  kmeans::init::Batch<float, kmeans::init::randomDense> init(num_clusters);
  init.input.set(kmeans::init::data, dataTable);
  init.compute();
  const NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

  // Clustering phase of K-Means
  kmeans::Batch<> algorithm(num_clusters, num_iterations);
  algorithm.input.set(kmeans::data, dataTable);
  algorithm.input.set(kmeans::inputCentroids, centroids);
  algorithm.parameter().resultsToEvaluate = kmeans::computeAssignments;
  algorithm.compute();
  const NumericTablePtr assignments = algorithm.getResult()->get(kmeans::assignments);

  // Assign output columns and cluster assignments
  BlockDescriptor<int> assignmentsBlock;
  assignments->getBlockOfColumnValues(0, 0, num_rows, readOnly, assignmentsBlock);
  const auto assignmentsPtr = assignmentsBlock.getBlockPtr();
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      outputs[j] = inputs[j];
    }
    output_cluster[i] = assignmentsPtr[i];
  }
  assignments->releaseBlockOfColumnValues(assignmentsBlock);

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
