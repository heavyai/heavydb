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
#ifdef HAVE_ONEDAL

#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLTableFunctionsCommon.h"
#include "QueryEngine/heavydbTypes.h"
#include "daal.h"

using namespace daal::algorithms;
using namespace daal::data_management;

template <typename T>
const NumericTablePtr prepare_data_table(const T* data,

                                         const int64_t num_rows) {
  // Prepare input data as structure of arrays (SOA) as columnar format (zero-copy)
  const auto data_table = SOANumericTable::create(1 /* num_columns */, num_rows);
  data_table->setArray<T>(const_cast<T*>(data), 0);
  return data_table;
}

template <typename T>
// const NumericTablePtr prepare_data_table(const std::vector<std::vector<T>>& data,
const NumericTablePtr prepare_data_table(const std::vector<const T*>& data,
                                         const int64_t num_rows) {
  // Data dimensions
  const size_t num_columns = data.size();

  // Prepare input data as structure of arrays (SOA) as columnar format (zero-copy)
  const auto data_table = SOANumericTable::create(num_columns, num_rows);
  for (size_t i = 0; i < num_columns; ++i) {
    data_table->setArray<T>(const_cast<T*>(data[i]), i);
  }
  return data_table;
}

template <typename T>
const NumericTablePtr prepare_pivoted_data_table(const T* data, const int64_t num_elems) {
  // Data dimensions
  // Prepare input data as structure of arrays (SOA) as columnar format (zero-copy)
  const auto data_table = SOANumericTable::create(num_elems, 1);
  for (size_t c = 0; c < static_cast<size_t>(num_elems); ++c) {
    data_table->setArray<T>(const_cast<T*>(data) + c, c);
  }
  return data_table;
}

inline kmeans::init::Method get_kmeans_init_type(const KMeansInitStrategy init_type) {
  const static std::map<KMeansInitStrategy, kmeans::init::Method> kmeans_init_type_map = {
      {KMeansInitStrategy::DEFAULT, kmeans::init::Method::deterministicDense},
      {KMeansInitStrategy::DETERMINISTIC, kmeans::init::Method::deterministicDense},
      {KMeansInitStrategy::RANDOM, kmeans::init::Method::randomDense},
      {KMeansInitStrategy::PLUS_PLUS, kmeans::init::Method::parallelPlusDense}};

  const auto itr = kmeans_init_type_map.find(init_type);
  if (itr == kmeans_init_type_map.end()) {
    std::ostringstream oss;
    oss << "Invalid Kmeans cluster centroid initialization type. "
        << "Was expecting one of DETERMINISTIC, RANDOM, or PLUS_PLUS.";
    throw std::runtime_error(oss.str());
  }
  return itr->second;
}

template <typename T, kmeans::init::Method M>
const NumericTablePtr init_centroids_for_type(const NumericTablePtr& input_features_table,
                                              const int32_t num_clusters) {
  kmeans::init::Batch<T, M> init(num_clusters);
  init.input.set(kmeans::init::data, input_features_table);
  init.compute();
  return init.getResult()->get(kmeans::init::centroids);
}

template <typename T>
const NumericTablePtr init_centroids(const NumericTablePtr& input_features_table,
                                     const kmeans::init::Method& init_type,
                                     const int32_t num_clusters) {
  switch (init_type) {
    case kmeans::init::Method::deterministicDense:
      return init_centroids_for_type<T, kmeans::init::Method::deterministicDense>(
          input_features_table, num_clusters);
    case kmeans::init::Method::randomDense:
      return init_centroids_for_type<T, kmeans::init::Method::randomDense>(
          input_features_table, num_clusters);
    case kmeans::init::Method::plusPlusDense:
      return init_centroids_for_type<T, kmeans::init::Method::plusPlusDense>(
          input_features_table, num_clusters);
    case kmeans::init::Method::parallelPlusDense:
      return init_centroids_for_type<T, kmeans::init::Method::parallelPlusDense>(
          input_features_table, num_clusters);
    default: {
      UNREACHABLE();
      return init_centroids_for_type<T, kmeans::init::Method::deterministicDense>(
          input_features_table, num_clusters);
    }
  }
}

template <typename T>
NEVER_INLINE HOST int32_t onedal_kmeans_impl(const std::vector<const T*>& input_features,
                                             int32_t* output_clusters,
                                             const int64_t num_rows,
                                             const int num_clusters,
                                             const int num_iterations,
                                             const KMeansInitStrategy kmeans_init_type) {
  try {
    const auto features_table = prepare_data_table(input_features, num_rows);
    const auto onedal_kmeans_init_type = get_kmeans_init_type(kmeans_init_type);
    const auto centroids =
        init_centroids<T>(features_table, onedal_kmeans_init_type, num_clusters);
    const auto assignments_table =
        HomogenNumericTable<int32_t>::create(output_clusters, 1, num_rows);
    const kmeans::ResultPtr result(new kmeans::Result);
    result->set(kmeans::assignments, assignments_table);
    result->set(kmeans::objectiveFunction,
                HomogenNumericTable<T>::create(1, 1, NumericTable::doAllocate));
    result->set(kmeans::nIterations,
                HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate));
    kmeans::Batch<> algorithm(num_clusters, num_iterations);
    algorithm.input.set(kmeans::data, features_table);
    algorithm.input.set(kmeans::inputCentroids, centroids);
    algorithm.parameter().resultsToEvaluate = kmeans::computeAssignments;
    algorithm.setResult(result);
    algorithm.compute();
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
  return num_rows;
}

template <typename T>
NEVER_INLINE HOST int32_t onedal_dbscan_impl(const std::vector<const T*>& input_features,
                                             int32_t* output_clusters,
                                             const int64_t num_rows,
                                             const double epsilon,
                                             const int32_t min_observations) {
  try {
    const auto features_table = prepare_data_table(input_features, num_rows);
    const auto assignments_table =
        HomogenNumericTable<int32_t>::create(output_clusters, 1, num_rows);
    const dbscan::ResultPtr result(new dbscan::Result);
    result->set(dbscan::assignments, assignments_table);
    result->set(dbscan::nClusters,
                HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate));
    dbscan::Batch<> algorithm(epsilon, min_observations);
    algorithm.input.set(dbscan::data, features_table);
    algorithm.parameter().resultsToCompute = dbscan::assignments;
    algorithm.setResult(result);
    algorithm.compute();
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
  return num_rows;
}

template <typename T>
int32_t extract_model_coefs(const NumericTablePtr& coefs_table,
                            int32_t* coef_idxs,
                            T* coefs) {
  const int64_t num_coefs = coefs_table->getNumberOfColumns();
  for (int64_t coef_idx = 0; coef_idx < num_coefs; ++coef_idx) {
    coef_idxs[coef_idx] = coef_idx;
    coefs[coef_idx] =
        coefs_table->NumericTable::getValue<T>(coef_idx, static_cast<size_t>(0));
  }
  return num_coefs;
}

template <typename T>
NEVER_INLINE HOST int32_t
onedal_linear_reg_fit_impl(const T* input_labels,
                           const std::vector<const T*>& input_features,
                           int32_t* output_coef_idxs,
                           T* output_coefs,
                           const int64_t num_rows) {
  try {
    const auto labels_table = prepare_data_table(input_labels, num_rows);
    const auto features_table = prepare_data_table(input_features, num_rows);

    linear_regression::training::Batch<T, linear_regression::training::Method::qrDense>
        algorithm;

    algorithm.input.set(linear_regression::training::data, features_table);
    algorithm.input.set(linear_regression::training::dependentVariables, labels_table);

    algorithm.compute();
    const auto training_result = algorithm.getResult();
    const auto coefs_table =
        training_result->get(linear_regression::training::model)->getBeta();
    return extract_model_coefs<T>(coefs_table, output_coef_idxs, output_coefs);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
NEVER_INLINE HOST linear_regression::ModelPtr build_linear_reg_model(
    const T* model_coefs,
    const int64_t num_coefs) {
  // See comment at end of onedal_lin_reg_fit_impl
  // We need to unpivot the model data back to the native
  // format oneDal expects, with 1 column per beta
  const auto betas_table = prepare_pivoted_data_table(model_coefs, num_coefs);
  CHECK_EQ(betas_table->getNumberOfColumns(), num_coefs);

  // Create model builder with true intercept flag
  linear_regression::ModelBuilder<T> model_builder(num_coefs - 1,
                                                   1 /* num_dependent_variables */);

  // Retrive pointer to the begining of betas_table
  BlockDescriptor<T> block_result;

  // Use generic code for getting start and end iterators for betas table, even though we
  // currently only support case of one dependent variable (i.e. 1 row in the betas table)
  betas_table->getBlockOfRows(0, betas_table->getNumberOfRows(), readOnly, block_result);
  size_t num_betas =
      (betas_table->getNumberOfRows()) * (betas_table->getNumberOfColumns());

  // Initialize iterators for beta array with itrecepts
  T* first_itr = block_result.getBlockPtr();
  T* last_itr = first_itr + num_betas;
  model_builder.setBeta(first_itr, last_itr);
  betas_table->releaseBlockOfRows(block_result);

  return model_builder.getModel();
}

template <typename T>
NEVER_INLINE HOST int32_t
onedal_linear_reg_predict_impl(const std::vector<const T*>& input_features,
                               T* output_predictions,
                               const int64_t num_rows,
                               const T* coefs) {
  try {
    const auto features_table = prepare_data_table(input_features, num_rows);
    const auto model_ptr = build_linear_reg_model(coefs, input_features.size() + 1);

    linear_regression::prediction::Batch<> algorithm;
    algorithm.input.set(linear_regression::prediction::data, features_table);
    algorithm.input.set(linear_regression::prediction::model, model_ptr);

    const auto predictions_table =
        HomogenNumericTable<T>::create(output_predictions, 1, num_rows);

    const linear_regression::prediction::ResultPtr result(
        new linear_regression::prediction::Result);
    result->set(linear_regression::prediction::prediction, predictions_table);
    algorithm.setResult(result);
    algorithm.compute();
    return num_rows;
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

#endif  // #ifdef HAVE_ONEDAL
#endif  // #ifdef __CUDACC__
