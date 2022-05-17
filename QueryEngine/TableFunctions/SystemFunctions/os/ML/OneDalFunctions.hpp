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
const NumericTablePtr prepare_data_table(const std::vector<std::vector<T>>& data,
                                         const int64_t num_rows) {
  using data_type =
      typename std::remove_cv_t<std::remove_reference_t<decltype(data)>>::value_type;

  // Data dimensions
  const size_t num_columns = data.size();

  // Prepare input data as structure of arrays (SOA) as columnar format (zero-copy)
  const auto data_table = SOANumericTable::create(num_columns, num_rows);
  for (size_t i = 0; i < num_columns; ++i) {
    data_table->setArray<T>(const_cast<T*>(data[i].data()), i);
  }
  return data_table;
}

kmeans::init::Method get_kmeans_init_type(const KMeansInitStrategy init_type) {
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
NEVER_INLINE HOST int32_t
onedal_kmeans_impl(const std::vector<std::vector<T>>& input_features,
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
NEVER_INLINE HOST int32_t
onedal_dbscan_impl(const std::vector<std::vector<T>>& input_features,
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

#endif  // #ifdef HAVE_ONEDAL
#endif  // #ifdef __CUDACC__
