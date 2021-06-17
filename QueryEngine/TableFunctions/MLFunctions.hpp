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

#ifdef HAVE_MLPACK

#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

// clang-format off
/*
  UDTF: dbscan__cpu_(Cursor<Column<int>, ColumnList<double>>, float, int, RowMultiplier) -> Column<int>, Column<int>
*/
// clang-format on
EXTENSION_NOINLINE int32_t dbscan__cpu_(const Column<int>& input_ids,
                                        const ColumnList<double>& cluster_features,
                                        const float epsilon,
                                        const int min_num_points,
                                        const int output_size_multiplier,
                                        Column<int>& output_ids,
                                        Column<int>& output_clusters) {
  if (epsilon <= 0.0) {
    throw std::runtime_error("DBSCAN: epsilon must be positive");
  }
  if (min_num_points < 1) {
    throw std::runtime_error("DBSCAN: min_num_points must be >= 1");
  }
  const int64_t num_rows = input_ids.getSize();
#ifndef __CUDACC__
  const int64_t num_cluster_features = cluster_features.getCols();
  arma::Mat<double> cluster_features_matrix(num_rows, num_cluster_features);
  for (int64_t c = 0; c < num_cluster_features; ++c) {
    memcpy(cluster_features_matrix.colptr(c),
           cluster_features(c).ptr,
           sizeof(double) * num_rows);
  }
  arma::Mat<double> cluster_features_matrix_transposed = cluster_features_matrix.t();
  mlpack::dbscan::DBSCAN<> dbscan(epsilon, min_num_points);
  arma::Row<size_t> cluster_assignments;
  dbscan.Cluster(cluster_features_matrix_transposed, cluster_assignments);

  for (int64_t r = 0; r < num_rows; ++r) {
    output_ids[r] = input_ids[r];
    output_clusters[r] = cluster_assignments[r] == SIZE_MAX ? -1 : cluster_assignments[r];
  }
#endif
  return num_rows;
}

// clang-format off
/*
  UDTF: kmeans__cpu_(Cursor<Column<int>, ColumnList<double>>, int, RowMultiplier) -> Column<int>, Column<int>
*/
// clang-format on
EXTENSION_NOINLINE int32_t kmeans__cpu_(const Column<int>& input_ids,
                                        const ColumnList<double>& cluster_features,
                                        const int num_clusters,
                                        const int output_size_multiplier,
                                        Column<int>& output_ids,
                                        Column<int>& output_clusters) {
  if (num_clusters <= 0) {
    throw std::runtime_error("KMEANS: num_clusters must be positive integer");
  }
  const int64_t num_rows = input_ids.getSize();
#ifndef __CUDACC__
  const int64_t num_cluster_features = cluster_features.getCols();
  arma::Mat<double> cluster_features_matrix(num_rows, num_cluster_features);
  for (int64_t c = 0; c < num_cluster_features; ++c) {
    memcpy(cluster_features_matrix.colptr(c),
           cluster_features(c).ptr,
           sizeof(double) * num_rows);
  }
  arma::Mat<double> cluster_features_matrix_transposed = cluster_features_matrix.t();
  mlpack::kmeans::KMeans<> kmeans;
  arma::Row<size_t> cluster_assignments;
  kmeans.Cluster(cluster_features_matrix_transposed,
                 static_cast<size_t>(num_clusters),
                 cluster_assignments);

  for (int64_t r = 0; r < num_rows; ++r) {
    output_ids[r] = input_ids[r];
    output_clusters[r] = cluster_assignments[r] == SIZE_MAX ? -1 : cluster_assignments[r];
  }
#endif
  return num_rows;
}

#endif