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

inline arma::Mat<double> create_input_matrix(
    const std::vector<const float*>& input_features,
    const int64_t num_rows) {
  const int64_t num_features = input_features.size();
  arma::Mat<double> features_matrix(num_rows, num_features);
  for (int64_t c = 0; c < num_features; ++c) {
    double* matrix_col_ptr = features_matrix.colptr(c);
    const float* input_features_col = input_features[c];
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

inline arma::Mat<double> create_input_matrix(
    const std::vector<const double*>& input_features,
    const int64_t num_rows) {
  const int64_t num_features = input_features.size();
  arma::Mat<double> features_matrix(num_rows, num_features);
  for (int64_t c = 0; c < num_features; ++c) {
    memcpy(features_matrix.colptr(c), input_features[c], sizeof(double) * num_rows);
  }
  return features_matrix;
}

inline void rewrite_cluster_id_nulls(const arma::Row<size_t>& cluster_assignments,
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
NEVER_INLINE HOST int32_t mlpack_kmeans_impl(const std::vector<const T*>& input_features,
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
}

template <typename T>
NEVER_INLINE HOST int32_t mlpack_dbscan_impl(const std::vector<const T*>& input_features,
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

template <typename T>
NEVER_INLINE HOST int32_t
mlpack_linear_reg_fit_impl(const T* input_labels,
                           const std::vector<const T*>& input_features,
                           int64_t* output_coef_idxs,
                           double* output_coefs,
                           const int64_t num_rows) {
  try {
    // Implement simple linear regression entirely in Armadillo
    // to avoid overhead of mlpack copies and extra matrix inversion
    arma::Mat<T> X(num_rows, input_features.size() + 1);
    // Intercept
    X.unsafe_col(0).fill(1);
    // Now copy feature column pointers
    const int64_t num_features = input_features.size();
    for (int64_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
      memcpy(
          X.colptr(feature_idx + 1), input_features[feature_idx], sizeof(T) * num_rows);
    }
    const arma::Mat<T> Xt = X.t();
    const arma::Mat<T> XtX = Xt * X;  // XtX aka "Gram Matrix"
    const arma::Col<T> Y(input_labels, num_rows);
    const arma::Col<T> B_est = arma::solve(XtX, Xt * Y);
    for (int64_t coef_idx = 0; coef_idx < num_features + 1; ++coef_idx) {
      output_coef_idxs[coef_idx] = coef_idx;
      output_coefs[coef_idx] = B_est[coef_idx];
    }
    return num_features + 1;
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
NEVER_INLINE HOST int32_t
mlpack_linear_reg_predict_impl(const std::shared_ptr<LinearRegressionModel>& model,
                               const std::vector<const T*>& input_features,
                               T* output_predictions,
                               const int64_t num_rows) {
  CHECK(model->getModelType() == MLModelType::LINEAR_REG);
  try {
    if (model->getNumFeatures() != static_cast<int64_t>(input_features.size())) {
      throw std::runtime_error(
          "Number of model coefficients does not match number of input features.");
    }
    // Implement simple linear regression entirely in Armadillo
    // to avoid overhead of mlpack copies and extra matrix inversion
    const int64_t num_features = input_features.size();
    const int64_t num_coefs = num_features + 1;
    arma::Mat<T> X(num_rows, num_coefs);
    X.unsafe_col(0).fill(1);
    for (int64_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
      memcpy(
          X.colptr(feature_idx + 1), input_features[feature_idx], sizeof(T) * num_rows);
    }
    std::vector<T> casted_coefs(num_coefs);
    const auto& coefs = model->getCoefs();
    for (int64_t coef_idx = 0; coef_idx < num_coefs; ++coef_idx) {
      casted_coefs[coef_idx] = coefs[coef_idx];
    }
    const arma::Col<T> B(casted_coefs.data(), num_coefs);
    const arma::Col<T> Y_hat = X * B;
    memcpy(output_predictions, Y_hat.colptr(0), sizeof(T) * num_rows);
    return num_rows;
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

#endif  // #ifdef HAVE_TBB
#endif  // #ifdef HAVE_MLPACK
#endif  // #ifdef __CUDACC__
