/*
 * Copyright 2022 HEAVY.AI, Inc., Inc.
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
#ifdef HAVE_MLPACK
#ifdef HAVE_TBB

#include <tbb/parallel_for.h>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLTableFunctionsCommon.h"
#include "QueryEngine/heavydbTypes.h"

using MatrixT = arma::Mat<double>;

arma::Mat<double> create_input_matrix(
    const std::vector<std::vector<float>>& input_features,
    const int64_t num_rows) {
  const int64_t num_features = input_features.size();
  arma::Mat<double> features_matrix(num_rows, num_features);
  for (int64_t c = 0; c < num_features; ++c) {
    double* matrix_col_ptr = features_matrix.colptr(c);
    const float* input_features_col = input_features[c].data();
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_rows),
                      [&](const tbb::blocked_range<int64_t>& r) {
                        const int64_t start_idx = r.begin();
                        const int64_t end_idx = r.end();
                        for (int64_t r = start_idx; r < end_idx; ++r) {
                          matrix_col_ptr[r] = input_features_col[r];
                        }
                      });
  }
  return features_matrix;
}

arma::Mat<double> create_input_matrix(
    const std::vector<std::vector<double>>& input_features,
    const int64_t num_rows) {
  const int64_t num_features = input_features.size();
  arma::Mat<double> features_matrix(num_rows, num_features);
  for (int64_t c = 0; c < num_features; ++c) {
    memcpy(
        features_matrix.colptr(c), input_features[c].data(), sizeof(double) * num_rows);
  }
  return features_matrix;
}

void rewrite_cluster_id_nulls(const arma::Row<size_t>& cluster_assignments,
                              int32_t* output_clusters,
                              const int64_t num_rows) {
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_rows),
                    [&](const tbb::blocked_range<int64_t>& r) {
                      const int64_t start_idx = r.begin();
                      const int64_t end_idx = r.end();
                      for (int64_t r = start_idx; r < end_idx; ++r) {
                        output_clusters[r] = cluster_assignments[r] == SIZE_MAX
                                                 ? -1
                                                 : cluster_assignments[r];
                      }
                    });
}

template <typename InitStrategy>
void run_kmeans(const MatrixT& input_features_matrix_transposed,
                arma::Row<size_t>& cluster_assignments,
                const int32_t num_clusters) {
  mlpack::kmeans::KMeans<mlpack::metric::EuclideanDistance,
                         InitStrategy,
                         mlpack::kmeans::MaxVarianceNewCluster,
                         mlpack::kmeans::NaiveKMeans,
                         MatrixT>
      kmeans;
  kmeans.Cluster(input_features_matrix_transposed,
                 static_cast<size_t>(num_clusters),
                 cluster_assignments);
}

template <typename T>
NEVER_INLINE HOST int32_t
mlpack_kmeans_impl(const std::vector<std::vector<T>>& input_features,
                   int32_t* output_clusters,
                   const int64_t num_rows,
                   const int32_t num_clusters,
                   const int32_t num_iterations,
                   const KMeansInitStrategy init_type) {
  try {
    const auto input_features_matrix = create_input_matrix(input_features, num_rows);
    const MatrixT input_features_matrix_transposed = input_features_matrix.t();
    arma::Row<size_t> cluster_assignments;
    switch (init_type) {
      case KMeansInitStrategy::RANDOM:
      case KMeansInitStrategy::DEFAULT: {
        run_kmeans<mlpack::kmeans::SampleInitialization>(
            input_features_matrix_transposed, cluster_assignments, num_clusters);
        break;
      }
      // Note that PlusPlus strategy is unimplemented for version of MLPack pulled via apt
      // in Ubuntu 20.04, but landed at the beginning of 2021 so will be available if we
      // package it
      // case KMeansInitStrategy::PLUS_PLUS: {
      //  //run_kmeans<mlpack::kmeans::KMeansPlusPlusInitialization>(input_features_matrix_transposed,
      //  cluster_assignments, num_clusters);
      //  //break;
      //}
      default: {
        throw std::runtime_error(
            "MLPack KMeans initialization not implemented for given strategy.");
        break;
      }
    }

    rewrite_cluster_id_nulls(cluster_assignments, output_clusters, num_rows);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
  return num_rows;

  return num_rows;
}

template <typename T>
NEVER_INLINE HOST int32_t
mlpack_dbscan_impl(const std::vector<std::vector<T>>& input_features,
                   int32_t* output_clusters,
                   const int64_t num_rows,
                   const double epsilon,
                   const int32_t min_observations) {
  try {
    const auto input_features_matrix = create_input_matrix(input_features, num_rows);
    const MatrixT input_features_matrix_transposed = input_features_matrix.t();

    mlpack::dbscan::DBSCAN<> dbscan(epsilon, min_observations);
    arma::Row<size_t> cluster_assignments;
    dbscan.Cluster(input_features_matrix_transposed, cluster_assignments);

    rewrite_cluster_id_nulls(cluster_assignments, output_clusters, num_rows);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
  return num_rows;
}

#endif  // #ifdef HAVE_TBB
#endif  // #ifdef HAVE_MLPACK
#endif  // #ifdef __CUDACC__