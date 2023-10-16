/*
 * Copyright 2023 HEAVY.AI, Inc.
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
#ifdef HAVE_ONEDAL

#include <cstring>

#include "MLModel.h"
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLTableFunctionsCommon.h"
#include "QueryEngine/heavydbTypes.h"

#include "oneapi/dal/algo/dbscan.hpp"
#include "oneapi/dal/algo/decision_forest.hpp"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/algo/linear_regression.hpp"
#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/array.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include <iostream>

namespace dal = oneapi::dal;

inline std::ostream& operator<<(std::ostream& stream, const dal::table& table) {
  if (!table.has_data())
    return stream;

  auto arr = dal::row_accessor<const float>(table).pull();
  const auto x = arr.get_data();
  const std::int32_t precision =
      dal::detail::is_floating_point(table.get_metadata().get_data_type(0)) ? 3 : 0;

  if (table.get_row_count() <= 10) {
    for (std::int64_t i = 0; i < table.get_row_count(); i++) {
      for (std::int64_t j = 0; j < table.get_column_count(); j++) {
        stream << std::setw(10) << std::setiosflags(std::ios::fixed)
               << std::setprecision(precision) << x[i * table.get_column_count() + j];
      }
      stream << std::endl;
    }
  } else {
    for (std::int64_t i = 0; i < 5; i++) {
      for (std::int64_t j = 0; j < table.get_column_count(); j++) {
        stream << std::setw(10) << std::setiosflags(std::ios::fixed)
               << std::setprecision(precision) << x[i * table.get_column_count() + j];
      }
      stream << std::endl;
    }
    stream << "..." << (table.get_row_count() - 10) << " lines skipped..." << std::endl;
    for (std::int64_t i = table.get_row_count() - 5; i < table.get_row_count(); i++) {
      for (std::int64_t j = 0; j < table.get_column_count(); j++) {
        stream << std::setw(10) << std::setiosflags(std::ios::fixed)
               << std::setprecision(precision) << x[i * table.get_column_count() + j];
      }
      stream << std::endl;
    }
  }
  return stream;
}

template <typename T>
const dal::table prepare_oneapi_data_table(const T* data, const int64_t num_rows) {
  auto data_arr = dal::array<T>::empty(num_rows);
  std::copy(data, data + num_rows, data_arr.get_mutable_data());
  const auto data_table =
      dal::homogen_table::wrap(data_arr, num_rows, 1, dal::data_layout::column_major);
  return data_table;
}

template <typename T>
const dal::table prepare_oneapi_data_table(const std::vector<const T*>& data,
                                           const int64_t num_rows) {
  const size_t num_columns = data.size();
  auto data_arr = dal::array<T>::empty(num_rows * num_columns);
  T* raw_ptr = data_arr.get_mutable_data();
  for (size_t i = 0; i < num_columns; ++i) {
    const T* column_ptr = data[i];
    for (int64_t j = 0; j < num_rows; ++j) {
      raw_ptr[j * num_columns + i] = column_ptr[j];
    }
  }
  return dal::homogen_table::wrap(data_arr, num_rows, num_columns);
}

template <typename T>
const dal::table prepare_oneapi_pivoted_data_table(const T* data,
                                                   const int64_t num_elems) {
  auto data_arr = dal::array<T>::empty(num_elems);
  std::copy(data, data + num_elems, data_arr.get_mutable_data());
  return dal::homogen_table::wrap(data_arr, 1, num_elems);
}

template <typename T>
auto init_centroids_oneapi(const KMeansInitStrategy init_type,
                           const int num_clusters,
                           const dal::table features_table) {
  switch (init_type) {
    case KMeansInitStrategy::DEFAULT:
    case KMeansInitStrategy::DETERMINISTIC: {
      const auto kmeans_init_desc =
          dal::kmeans_init::descriptor<T, dal::kmeans_init::method::dense>()
              .set_cluster_count(num_clusters);
      return dal::compute(kmeans_init_desc, features_table);
    }
    case KMeansInitStrategy::RANDOM: {
      const auto kmeans_init_desc =
          dal::kmeans_init::descriptor<T, dal::kmeans_init::method::random_dense>()
              .set_cluster_count(num_clusters);
      return dal::compute(kmeans_init_desc, features_table);
    }
    case KMeansInitStrategy::PLUS_PLUS: {
      const auto kmeans_init_desc =
          dal::kmeans_init::descriptor<T, dal::kmeans_init::method::parallel_plus_dense>()
              .set_cluster_count(num_clusters);
      return dal::compute(kmeans_init_desc, features_table);
    }
    default: {
      throw std::runtime_error(
          "Invalid Kmeans cluster centroid init type. Was expecting one of "
          "DETERMINISTIC, RANDOM, PLUS_PLUS.");
    }
  }
}

template <typename T>
NEVER_INLINE HOST int32_t
onedal_oneapi_kmeans_impl(const std::vector<const T*>& input_features,
                          int32_t* output_clusters,
                          const int64_t num_rows,
                          const int num_clusters,
                          const int num_iterations,
                          const KMeansInitStrategy kmeans_init_type) {
  try {
    const auto features_table = prepare_oneapi_data_table(input_features, num_rows);
    const auto result_init =
        init_centroids_oneapi<T>(kmeans_init_type, num_clusters, features_table);

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(num_clusters)
                                 .set_max_iteration_count(num_iterations)
                                 .set_accuracy_threshold(0.001);
    const dal::kmeans::train_result result_train =
        dal::train(kmeans_desc, features_table, result_init.get_centroids());
    auto arr = dal::row_accessor<const int32_t>(result_train.get_responses()).pull();
    const auto x = arr.get_data();
    std::memcpy(output_clusters, x, num_rows * sizeof(int32_t));
  } catch (const std::exception& e) {
    throw std::runtime_error(e.what());
  }

  return num_rows;
}

template <typename T>
NEVER_INLINE HOST int32_t
onedal_oneapi_dbscan_impl(const std::vector<const T*>& input_features,
                          int32_t* output_clusters,
                          const int64_t num_rows,
                          const double epsilon,
                          const int32_t min_observations) {
  try {
    const auto features_table = prepare_oneapi_data_table(input_features, num_rows);
    auto dbscan_desc = dal::dbscan::descriptor<>(epsilon, min_observations);
    dbscan_desc.set_result_options(dal::dbscan::result_options::responses);
    const auto result_compute = dal::compute(dbscan_desc, features_table);

    auto arr = dal::row_accessor<const int32_t>(result_compute.get_responses()).pull();
    const auto x = arr.get_data();
    std::memcpy(output_clusters, x, num_rows * sizeof(int32_t));
  } catch (const std::exception& e) {
    throw std::runtime_error(e.what());
  }
  return num_rows;
}

template <typename T>
NEVER_INLINE HOST std::pair<std::vector<std::vector<T>>, std::vector<T>>
onedal_oneapi_pca_impl(const std::vector<const T*>& input_features,
                       const int64_t num_rows) {
  try {
    // TODO: Do we want to parameterize PCA to allow using SVD other than default COV?
    const auto pca_desc =
        dal::pca::descriptor<T, dal::pca::method::cov>().set_deterministic(true);
    const auto features_table = prepare_oneapi_data_table(input_features, num_rows);

    const auto result_train = dal::train(pca_desc, features_table);

    auto eigenvectors_table_asarray =
        dal::row_accessor<const T>(result_train.get_eigenvectors()).pull();
    const auto eigenvectors_data = eigenvectors_table_asarray.get_data();
    const int64_t num_dims = result_train.get_eigenvectors().get_row_count();
    std::vector<std::vector<T>> eigenvectors(num_dims, std::vector<T>(num_dims));
    for (std::int64_t i = 0; i < num_dims; i++) {
      for (std::int64_t j = 0; j < num_dims; j++) {
        eigenvectors[i][j] = eigenvectors_data[i * num_dims + j];
      }
    }

    auto eigenvalues_table_asarray =
        dal::row_accessor<const T>(result_train.get_eigenvalues()).pull();
    const auto eigenvalues_data = eigenvalues_table_asarray.get_data();
    std::vector<T> eigenvalues(eigenvalues_data, eigenvalues_data + num_dims);

    return std::make_pair(eigenvectors, eigenvalues);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
int32_t extract_model_coefs(const dal::table& coefs_table,
                            int64_t* coef_idxs,
                            double* coefs) {
  const int64_t num_coefs = coefs_table.get_column_count();

  auto coefs_table_data = dal::row_accessor<const float>(coefs_table).pull().get_data();
  for (int64_t coef_idx = 0; coef_idx < num_coefs; ++coef_idx) {
    coef_idxs[coef_idx] = coef_idx;
    coefs[coef_idx] = coefs_table_data[coef_idx];
  }

  return num_coefs;
}

template <typename T>
NEVER_INLINE HOST int32_t
onedal_oneapi_linear_reg_fit_impl(const T* input_labels,
                                  const std::vector<const T*>& input_features,
                                  int64_t* output_coef_idxs,
                                  double* output_coefs,
                                  const int64_t num_rows) {
  try {
    const auto labels_table = prepare_oneapi_data_table(input_labels, num_rows);
    const auto features_table = prepare_oneapi_data_table(input_features, num_rows);

    const auto lr_descriptor = dal::linear_regression::descriptor<>().set_result_options(
        dal::linear_regression::result_options::coefficients |
        dal::linear_regression::result_options::intercept);
    const auto train_result = dal::train(lr_descriptor, features_table, labels_table);

    return extract_model_coefs<T>(train_result.get_model().get_packed_coefficients(),
                                  output_coef_idxs,
                                  output_coefs);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
NEVER_INLINE HOST int32_t
onedal_oneapi_linear_reg_predict_impl(const std::shared_ptr<LinearRegressionModel>& model,
                                      const std::vector<const T*>& input_features,
                                      T* output_predictions,
                                      const int64_t num_rows) {
  CHECK(model->getModelType() == MLModelType::LINEAR_REG);
  try {
    if (model->getNumFeatures() != static_cast<int64_t>(input_features.size())) {
      throw std::runtime_error(
          "Number of model coefficients does not match number of input features.");
    }

    const auto model_coefs = prepare_oneapi_pivoted_data_table(model->getCoefs().data(),
                                                               input_features.size() + 1);
    auto lr_model = dal::linear_regression::model();
    lr_model.set_packed_coefficients(model_coefs);

    const auto features_table = prepare_oneapi_data_table(input_features, num_rows);
    const auto lr_descriptor = dal::linear_regression::descriptor<>().set_result_options(
        dal::linear_regression::result_options::coefficients |
        dal::linear_regression::result_options::intercept);
    const auto test_result = dal::infer(lr_descriptor, features_table, lr_model);

    // For some reason if we construct the dal::row_accessor separately to then copy the
    // memory later, the underlying array's destructor gets called and its memory is
    // freed, so we construct it in-place instead.
    std::memcpy(output_predictions,
                dal::row_accessor<const T>(test_result.get_responses()).pull().get_data(),
                num_rows * sizeof(T));
    return num_rows;
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

inline dal::decision_forest::variable_importance_mode
get_oneapi_var_importance_metric_type(const VarImportanceMetric var_importance_metric) {
  switch (var_importance_metric) {
    case VarImportanceMetric::NONE:
      return dal::decision_forest::variable_importance_mode::none;
    case VarImportanceMetric::DEFAULT:
    case VarImportanceMetric::MDI:
      return dal::decision_forest::variable_importance_mode::mdi;
    case VarImportanceMetric::MDA:
      return dal::decision_forest::variable_importance_mode::mda_raw;
    case VarImportanceMetric::MDA_SCALED:
      return dal::decision_forest::variable_importance_mode::mda_scaled;
    default: {
      std::ostringstream oss;
      oss << "Invalid variable importance mode type. "
          << "Was expecting one of DEFAULT, NONE, MDI, MDA, or MDA_SCALED.";
      throw std::runtime_error(oss.str());
    }
  }
}

template <typename T, typename Method>
NEVER_INLINE HOST void onedal_oneapi_random_forest_reg_fit_impl(
    const std::string& model_name,
    const T* input_labels,
    const std::vector<const T*>& input_features,
    const std::string& model_metadata,
    const std::vector<std::vector<std::string>>& cat_feature_keys,
    const int64_t num_rows,
    const int64_t num_trees,
    const double obs_per_tree_fraction,
    const int64_t max_tree_depth,
    const int64_t features_per_node,
    const double impurity_threshold,
    const bool bootstrap,
    const int64_t min_obs_per_leaf_node,
    const int64_t min_obs_per_split_node,
    const double min_weight_fraction_in_leaf_node,
    const double min_impurity_decrease_in_split_node,
    const int64_t max_leaf_nodes,
    const VarImportanceMetric var_importance_metric) {
  constexpr bool compute_out_of_bag_error{false};
  try {
    const auto features_table = prepare_oneapi_data_table(input_features, num_rows);
    const auto labels_table = prepare_oneapi_data_table(input_labels, num_rows);

    const auto error_metric =
        compute_out_of_bag_error
            ? dal::decision_forest::error_metric_mode::out_of_bag_error
            : dal::decision_forest::error_metric_mode::none;

    const auto importance_metric =
        get_oneapi_var_importance_metric_type(var_importance_metric);

    auto df_desc =
        dal::decision_forest::descriptor<T,
                                         Method,
                                         dal::decision_forest::task::regression>{}
            .set_tree_count(num_trees)
            .set_observations_per_tree_fraction(obs_per_tree_fraction)
            .set_max_tree_depth(max_tree_depth)
            .set_features_per_node(features_per_node)
            .set_impurity_threshold(impurity_threshold)
            .set_bootstrap(bootstrap)
            .set_min_observations_in_leaf_node(min_obs_per_leaf_node)
            .set_min_observations_in_split_node(min_obs_per_split_node)
            .set_min_weight_fraction_in_leaf_node(min_weight_fraction_in_leaf_node)
            .set_min_impurity_decrease_in_split_node(min_impurity_decrease_in_split_node)
            .set_max_leaf_nodes(max_leaf_nodes)
            .set_error_metric_mode(error_metric)
            .set_variable_importance_mode(importance_metric);

    const auto result_train = dal::train(df_desc, features_table, labels_table);

    const size_t num_features = input_features.size();
    std::vector<double> variable_importance(
        var_importance_metric != VarImportanceMetric::NONE ? num_features : 0);
    if (var_importance_metric != VarImportanceMetric::NONE) {
      auto var_importance_data =
          dal::row_accessor<const T>(result_train.get_var_importance()).pull().get_data();
      for (size_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        variable_importance[feature_idx] = var_importance_data[feature_idx];
      }
    }

    double out_of_bag_error{0};
    if (compute_out_of_bag_error) {
      auto oob_error_data =
          dal::row_accessor<const T>(result_train.get_oob_err()).pull().get_data();
      out_of_bag_error = oob_error_data[0];
    }

    auto abstract_model = std::make_shared<OneAPIRandomForestRegressionModel>(
        std::make_shared<df::model<df::task::regression>>(result_train.get_model()),
        model_metadata,
        cat_feature_keys,
        variable_importance,
        out_of_bag_error,
        num_features);
    g_ml_models.addModel(model_name, abstract_model);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
NEVER_INLINE HOST int32_t onedal_oneapi_random_forest_reg_predict_impl(
    const std::shared_ptr<OneAPIRandomForestRegressionModel>& model,
    const std::vector<const T*>& input_features,
    T* output_predictions,
    const int64_t num_rows) {
  CHECK(model->getModelType() == MLModelType::RANDOM_FOREST_REG);
  try {
    if (model->getNumFeatures() != static_cast<int64_t>(input_features.size())) {
      throw std::runtime_error("Number of provided features does not match model.");
    }
    const auto features_table = prepare_oneapi_data_table(input_features, num_rows);

    // oneAPI's ::infer method expects a decision_forest::descriptor argument as input.
    // The descriptor seems to have no effect on how the pre-trained model is executed
    // though, so we pass a dummy descriptor rather than storing the descriptor originally
    // used to train the model unnecessarily
    auto dummy_desc =
        dal::decision_forest::descriptor<T,
                                         dal::decision_forest::method::hist,
                                         dal::decision_forest::task::regression>{};

    const auto result_infer =
        dal::infer(dummy_desc, *(model->getModel()), features_table);

    auto result_table_data =
        dal::row_accessor<const T>(result_infer.get_responses()).pull().get_data();
    std::memcpy(output_predictions, result_table_data, num_rows * sizeof(T));

    return num_rows;
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

#endif  // #ifdef HAVE_ONEDAL
#endif  // #ifdef __CUDACC__
