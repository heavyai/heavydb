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

#include <cstring>

#include "MLModel.h"
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLTableFunctionsCommon.h"
#include "QueryEngine/heavydbTypes.h"
#include "daal.h"

#include <iomanip>
#include <iostream>

using namespace daal::algorithms;
using namespace daal::data_management;

inline void printAprioriItemsets(
    daal::data_management::NumericTablePtr largeItemsetsTable,
    daal::data_management::NumericTablePtr largeItemsetsSupportTable,
    size_t nItemsetToPrint = 20) {
  using namespace daal::data_management;

  size_t largeItemsetCount = largeItemsetsSupportTable->getNumberOfRows();
  size_t nItemsInLargeItemsets = largeItemsetsTable->getNumberOfRows();

  BlockDescriptor<int> block1;
  largeItemsetsTable->getBlockOfRows(0, nItemsInLargeItemsets, readOnly, block1);
  int* largeItemsets = block1.getBlockPtr();

  BlockDescriptor<int> block2;
  largeItemsetsSupportTable->getBlockOfRows(0, largeItemsetCount, readOnly, block2);
  int* largeItemsetsSupportData = block2.getBlockPtr();

  std::vector<std::vector<size_t>> largeItemsetsVector;
  largeItemsetsVector.resize(largeItemsetCount);

  for (size_t i = 0; i < nItemsInLargeItemsets; i++) {
    largeItemsetsVector[largeItemsets[2 * i]].push_back(largeItemsets[2 * i + 1]);
  }

  std::vector<size_t> supportVector;
  supportVector.resize(largeItemsetCount);

  for (size_t i = 0; i < largeItemsetCount; i++) {
    supportVector[largeItemsetsSupportData[2 * i]] = largeItemsetsSupportData[2 * i + 1];
  }

  std::cout << std::endl << "Apriori example program results" << std::endl;

  std::cout << std::endl
            << "Last " << nItemsetToPrint << " large itemsets: " << std::endl;
  std::cout << std::endl
            << "Itemset"
            << "\t\t\tSupport" << std::endl;

  size_t iMin = (((largeItemsetCount > nItemsetToPrint) && (nItemsetToPrint != 0))
                     ? largeItemsetCount - nItemsetToPrint
                     : 0);
  for (size_t i = iMin; i < largeItemsetCount; i++) {
    std::cout << "{";
    for (size_t l = 0; l < largeItemsetsVector[i].size() - 1; l++) {
      std::cout << largeItemsetsVector[i][l] << ", ";
    }
    std::cout << largeItemsetsVector[i][largeItemsetsVector[i].size() - 1] << "}\t\t";

    std::cout << supportVector[i] << std::endl;
  }

  largeItemsetsTable->releaseBlockOfRows(block1);
  largeItemsetsSupportTable->releaseBlockOfRows(block2);
}

inline void printAprioriRules(daal::data_management::NumericTablePtr leftItemsTable,
                              daal::data_management::NumericTablePtr rightItemsTable,
                              daal::data_management::NumericTablePtr confidenceTable,
                              size_t nRulesToPrint = 20) {
  using namespace daal::data_management;

  size_t nRules = confidenceTable->getNumberOfRows();
  size_t nLeftItems = leftItemsTable->getNumberOfRows();
  size_t nRightItems = rightItemsTable->getNumberOfRows();

  BlockDescriptor<int> block1;
  leftItemsTable->getBlockOfRows(0, nLeftItems, readOnly, block1);
  int* leftItems = block1.getBlockPtr();

  BlockDescriptor<int> block2;
  rightItemsTable->getBlockOfRows(0, nRightItems, readOnly, block2);
  int* rightItems = block2.getBlockPtr();

  BlockDescriptor<DAAL_DATA_TYPE> block3;
  confidenceTable->getBlockOfRows(0, nRules, readOnly, block3);
  DAAL_DATA_TYPE* confidence = block3.getBlockPtr();

  std::vector<std::vector<size_t>> leftItemsVector;
  leftItemsVector.resize(nRules);

  if (nRules == 0) {
    std::cout << std::endl << "No association rules were found " << std::endl;
    return;
  }

  for (size_t i = 0; i < nLeftItems; i++) {
    leftItemsVector[leftItems[2 * i]].push_back(leftItems[2 * i + 1]);
  }

  std::vector<std::vector<size_t>> rightItemsVector;
  rightItemsVector.resize(nRules);

  for (size_t i = 0; i < nRightItems; i++) {
    rightItemsVector[rightItems[2 * i]].push_back(rightItems[2 * i + 1]);
  }

  std::vector<DAAL_DATA_TYPE> confidenceVector;
  confidenceVector.resize(nRules);

  for (size_t i = 0; i < nRules; i++) {
    confidenceVector[i] = confidence[i];
  }

  std::cout << std::endl
            << "Last " << nRulesToPrint << " association rules: " << std::endl;
  std::cout << std::endl
            << "Rule"
            << "\t\t\t\tConfidence" << std::endl;
  size_t iMin =
      (((nRules > nRulesToPrint) && (nRulesToPrint != 0)) ? (nRules - nRulesToPrint) : 0);

  for (size_t i = iMin; i < nRules; i++) {
    std::cout << "{";
    for (size_t l = 0; l < leftItemsVector[i].size() - 1; l++) {
      std::cout << leftItemsVector[i][l] << ", ";
    }
    std::cout << leftItemsVector[i][leftItemsVector[i].size() - 1] << "} => {";

    for (size_t l = 0; l < rightItemsVector[i].size() - 1; l++) {
      std::cout << rightItemsVector[i][l] << ", ";
    }
    std::cout << rightItemsVector[i][rightItemsVector[i].size() - 1] << "}\t\t";

    std::cout << confidenceVector[i] << std::endl;
  }

  leftItemsTable->releaseBlockOfRows(block1);
  rightItemsTable->releaseBlockOfRows(block2);
  confidenceTable->releaseBlockOfRows(block3);
}

inline bool isFull(daal::data_management::NumericTableIface::StorageLayout layout) {
  int layoutInt = (int)layout;
  if (daal::data_management::packed_mask & layoutInt) {
    return false;
  }
  return true;
}

inline bool isUpper(daal::data_management::NumericTableIface::StorageLayout layout) {
  using daal::data_management::NumericTableIface;

  if (layout == NumericTableIface::upperPackedSymmetricMatrix ||
      layout == NumericTableIface::upperPackedTriangularMatrix) {
    return true;
  }
  return false;
}

inline bool isLower(daal::data_management::NumericTableIface::StorageLayout layout) {
  using daal::data_management::NumericTableIface;

  if (layout == NumericTableIface::lowerPackedSymmetricMatrix ||
      layout == NumericTableIface::lowerPackedTriangularMatrix) {
    return true;
  }
  return false;
}

template <typename T>
inline void printArray(T* array,
                       const size_t nPrintedCols,
                       const size_t nPrintedRows,
                       const size_t nCols,
                       std::string message,
                       size_t interval = 10) {
  std::cout << std::setiosflags(std::ios::left);
  std::cout << message << std::endl;
  for (size_t i = 0; i < nPrintedRows; i++) {
    for (size_t j = 0; j < nPrintedCols; j++) {
      std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                << std::setprecision(3);
      std::cout << array[i * nCols + j];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
inline void printArray(T* array,
                       const size_t nCols,
                       const size_t nRows,
                       std::string message,
                       size_t interval = 10) {
  printArray(array, nCols, nRows, nCols, message, interval);
}

template <typename T>
inline void printLowerArray(T* array,
                            const size_t nPrintedRows,
                            std::string message,
                            size_t interval = 10) {
  std::cout << std::setiosflags(std::ios::left);
  std::cout << message << std::endl;
  int ind = 0;
  for (size_t i = 0; i < nPrintedRows; i++) {
    for (size_t j = 0; j <= i; j++) {
      std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                << std::setprecision(3);
      std::cout << array[ind++];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
inline void printUpperArray(T* array,
                            const size_t nPrintedCols,
                            const size_t nPrintedRows,
                            const size_t nCols,
                            std::string message,
                            size_t interval = 10) {
  std::cout << std::setiosflags(std::ios::left);
  std::cout << message << std::endl;
  int ind = 0;
  for (size_t i = 0; i < nPrintedRows; i++) {
    for (size_t j = 0; j < i; j++) {
      std::cout << "          ";
    }
    for (size_t j = i; j < nPrintedCols; j++) {
      std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                << std::setprecision(3);
      std::cout << array[ind++];
    }
    for (size_t j = nPrintedCols; j < nCols; j++) {
      ind++;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

inline void printNumericTable(daal::data_management::NumericTable* dataTable,
                              const char* message = "",
                              size_t nPrintedRows = 0,
                              size_t nPrintedCols = 0,
                              size_t interval = 10) {
  using namespace daal::data_management;

  size_t nRows = dataTable->getNumberOfRows();
  size_t nCols = dataTable->getNumberOfColumns();
  NumericTableIface::StorageLayout layout = dataTable->getDataLayout();

  if (nPrintedRows != 0) {
    nPrintedRows = std::min(nRows, nPrintedRows);
  } else {
    nPrintedRows = nRows;
  }

  if (nPrintedCols != 0) {
    nPrintedCols = std::min(nCols, nPrintedCols);
  } else {
    nPrintedCols = nCols;
  }

  BlockDescriptor<DAAL_DATA_TYPE> block;
  if (isFull(layout) || layout == NumericTableIface::csrArray) {
    dataTable->getBlockOfRows(0, nRows, readOnly, block);
    printArray<DAAL_DATA_TYPE>(
        block.getBlockPtr(), nPrintedCols, nPrintedRows, nCols, message, interval);
    dataTable->releaseBlockOfRows(block);
  } else {
    PackedArrayNumericTableIface* packedTable =
        dynamic_cast<PackedArrayNumericTableIface*>(dataTable);
    packedTable->getPackedArray(readOnly, block);
    if (isLower(layout)) {
      printLowerArray<DAAL_DATA_TYPE>(
          block.getBlockPtr(), nPrintedRows, message, interval);
    } else if (isUpper(layout)) {
      printUpperArray<DAAL_DATA_TYPE>(
          block.getBlockPtr(), nPrintedCols, nPrintedRows, nCols, message, interval);
    }
    packedTable->releasePackedArray(block);
  }
}

inline void printNumericTable(daal::data_management::NumericTable& dataTable,
                              const char* message = "",
                              size_t nPrintedRows = 0,
                              size_t nPrintedCols = 0,
                              size_t interval = 10) {
  printNumericTable(&dataTable, message, nPrintedRows, nPrintedCols, interval);
}

inline void printNumericTable(const daal::data_management::NumericTablePtr& dataTable,
                              const char* message = "",
                              size_t nPrintedRows = 0,
                              size_t nPrintedCols = 0,
                              size_t interval = 10) {
  printNumericTable(dataTable.get(), message, nPrintedRows, nPrintedCols, interval);
}

template <typename T>
const NumericTablePtr prepare_data_table(const T* data, const int64_t num_rows) {
  // Prepare input data as structure of arrays (SOA) as columnar format (zero-copy)
  const auto data_table = SOANumericTable::create(1 /* num_columns */, num_rows);
  data_table->setArray<T>(const_cast<T*>(data), 0);

  return data_table;
}

template <typename T>
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
NEVER_INLINE HOST std::pair<std::vector<std::vector<T>>, std::vector<T>> onedal_pca_impl(
    const std::vector<const T*>& input_features,
    const int64_t num_rows) {
  try {
    const auto features_table = prepare_data_table(input_features, num_rows);
    pca::Batch<> algorithm;
    algorithm.input.set(pca::data, features_table);
    algorithm.parameter.resultsToCompute = pca::mean | pca::variance | pca::eigenvalue;
    algorithm.parameter.isDeterministic = true;

    algorithm.compute();
    pca::ResultPtr result = algorithm.getResult();
    const auto eigenvectors_table = result->get(pca::eigenvectors);
    const int64_t num_dims = eigenvectors_table->getNumberOfRows();
    CHECK_EQ(num_dims, static_cast<int64_t>(eigenvectors_table->getNumberOfColumns()));
    std::vector<std::vector<T>> eigenvectors(num_dims, std::vector<T>(num_dims));
    for (int64_t row_idx = 0; row_idx < num_dims; ++row_idx) {
      for (int64_t col_idx = 0; col_idx < num_dims; ++col_idx) {
        // eigenvectors_table is column major, so need to flip the lookup indicies
        eigenvectors[row_idx][col_idx] =
            eigenvectors_table->getValue<T>(col_idx, row_idx);
      }
    }
    const auto eigenvalues_table = result->get(pca::eigenvalues);
    std::vector<T> eigenvalues(num_dims);
    for (int64_t dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
      eigenvalues[dim_idx] = eigenvalues_table->getValue<T>(dim_idx, 0);
    }
    return std::make_pair(eigenvectors, eigenvalues);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
int32_t extract_model_coefs(const NumericTablePtr& coefs_table,
                            int64_t* coef_idxs,
                            double* coefs) {
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
                           int64_t* output_coef_idxs,
                           double* output_coefs,
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
    const double* model_coefs,
    const int64_t num_coefs) {
  // See comment at end of onedal_lin_reg_fit_impl
  // We need to unpivot the model data back to the native
  // format oneDal expects, with 1 column per beta
  std::vector<T> casted_model_coefs(num_coefs);
  for (int64_t coef_idx = 0; coef_idx < num_coefs; ++coef_idx) {
    casted_model_coefs[coef_idx] = model_coefs[coef_idx];
  }
  const auto betas_table =
      prepare_pivoted_data_table(casted_model_coefs.data(), num_coefs);

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
onedal_linear_reg_predict_impl(const std::shared_ptr<LinearRegressionModel>& model,
                               const std::vector<const T*>& input_features,
                               T* output_predictions,
                               const int64_t num_rows) {
  CHECK(model->getModelType() == MLModelType::LINEAR_REG);
  try {
    if (model->getNumFeatures() != static_cast<int64_t>(input_features.size())) {
      throw std::runtime_error(
          "Number of model coefficients does not match number of input features.");
    }
    const auto features_table = prepare_data_table(input_features, num_rows);
    const auto model_ptr =
        build_linear_reg_model<T>(model->getCoefs().data(), input_features.size() + 1);

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

template <typename T>
NEVER_INLINE HOST void onedal_decision_tree_reg_fit_impl(
    const std::string& model_name,
    const T* input_labels,
    const std::vector<const T*>& input_features,
    const std::string& model_metadata,
    const std::vector<std::vector<std::string>>& cat_feature_keys,
    const int64_t num_rows,
    const int64_t max_tree_depth,
    const int64_t min_observations_per_leaf_node) {
  try {
    const auto labels_table = prepare_data_table(input_labels, num_rows);
    const auto features_table = prepare_data_table(input_features, num_rows);
    decision_tree::regression::training::Batch<T> algorithm;
    algorithm.input.set(decision_tree::regression::training::data, features_table);
    algorithm.input.set(decision_tree::regression::training::dependentVariables,
                        labels_table);

    algorithm.parameter.pruning = decision_tree::Pruning::none;
    algorithm.parameter.maxTreeDepth = max_tree_depth;
    algorithm.parameter.minObservationsInLeafNodes = min_observations_per_leaf_node;
    algorithm.compute();
    /* Retrieve the algorithm results */
    decision_tree::regression::training::ResultPtr training_result =
        algorithm.getResult();

    auto model_ptr = training_result->get(decision_tree::regression::training::model);
    auto model = std::make_shared<DecisionTreeRegressionModel>(
        model_ptr, model_metadata, cat_feature_keys);
    g_ml_models.addModel(model_name, model);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
NEVER_INLINE HOST void onedal_gbt_reg_fit_impl(
    const std::string& model_name,
    const T* input_labels,
    const std::vector<const T*>& input_features,
    const std::string& model_metadata,
    const std::vector<std::vector<std::string>>& cat_feature_keys,
    const int64_t num_rows,
    const int64_t max_iterations,
    const int64_t max_tree_depth,
    const double shrinkage,
    const double min_split_loss,
    const double lambda,
    const double obs_per_tree_fraction,
    const int64_t features_per_node,
    const int64_t min_observations_per_leaf_node,
    const int64_t max_bins,
    const int64_t min_bin_size) {
  try {
    const auto labels_table = prepare_data_table(input_labels, num_rows);
    const auto features_table = prepare_data_table(input_features, num_rows);
    gbt::regression::training::Batch<T> algorithm;
    algorithm.input.set(gbt::regression::training::data, features_table);
    algorithm.input.set(gbt::regression::training::dependentVariable, labels_table);

    algorithm.parameter().maxIterations = max_iterations;
    algorithm.parameter().maxTreeDepth = max_tree_depth;
    algorithm.parameter().shrinkage = shrinkage;
    algorithm.parameter().minSplitLoss = min_split_loss;
    algorithm.parameter().lambda = lambda;
    algorithm.parameter().observationsPerTreeFraction = obs_per_tree_fraction;
    algorithm.parameter().featuresPerNode = features_per_node;
    algorithm.parameter().minObservationsInLeafNode = min_observations_per_leaf_node;
    algorithm.parameter().maxBins = max_bins;
    algorithm.parameter().minBinSize = min_bin_size;
    algorithm.compute();
    /* Retrieve the algorithm results */
    gbt::regression::training::ResultPtr training_result = algorithm.getResult();

    auto model_ptr = training_result->get(gbt::regression::training::model);
    auto model =
        std::make_shared<GbtRegressionModel>(model_ptr, model_metadata, cat_feature_keys);
    g_ml_models.addModel(model_name, model);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

inline decision_forest::training::VariableImportanceMode get_var_importance_metric_type(
    const VarImportanceMetric var_importance_metric) {
  const static std::map<VarImportanceMetric,
                        decision_forest::training::VariableImportanceMode>
      var_importance_mode_type_map = {
          {VarImportanceMetric::DEFAULT,
           decision_forest::training::VariableImportanceMode::MDI},
          {VarImportanceMetric::NONE,
           decision_forest::training::VariableImportanceMode::none},
          {VarImportanceMetric::MDI,
           decision_forest::training::VariableImportanceMode::MDI},
          {VarImportanceMetric::MDA,
           decision_forest::training::VariableImportanceMode::MDA_Raw},
          {VarImportanceMetric::MDA_SCALED,
           decision_forest::training::VariableImportanceMode::MDA_Scaled}};

  const auto itr = var_importance_mode_type_map.find(var_importance_metric);
  if (itr == var_importance_mode_type_map.end()) {
    std::ostringstream oss;
    oss << "Invalid variable importance mode type. "
        << "Was expecting one of DEFAULT, NONE, MDI, MDA, or MDA_SCALED.";
    throw std::runtime_error(oss.str());
  }
  return itr->second;
}

template <typename T, decision_forest::regression::training::Method M>
NEVER_INLINE HOST void onedal_random_forest_reg_fit_impl(
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
    const auto labels_table = prepare_data_table(input_labels, num_rows);
    const auto features_table = prepare_data_table(input_features, num_rows);
    decision_forest::regression::training::Batch<T, M> algorithm;
    algorithm.input.set(decision_forest::regression::training::data, features_table);
    algorithm.input.set(decision_forest::regression::training::dependentVariable,
                        labels_table);

    algorithm.parameter().nTrees = num_trees;
    algorithm.parameter().observationsPerTreeFraction = obs_per_tree_fraction;
    algorithm.parameter().maxTreeDepth = max_tree_depth;
    algorithm.parameter().featuresPerNode = features_per_node;
    algorithm.parameter().impurityThreshold = impurity_threshold;
    algorithm.parameter().bootstrap = bootstrap;
    algorithm.parameter().minObservationsInLeafNode = min_obs_per_leaf_node;
    algorithm.parameter().minObservationsInSplitNode = min_obs_per_split_node;
    algorithm.parameter().minWeightFractionInLeafNode = min_weight_fraction_in_leaf_node;
    algorithm.parameter().minImpurityDecreaseInSplitNode =
        min_impurity_decrease_in_split_node;
    algorithm.parameter().varImportance =
        get_var_importance_metric_type(var_importance_metric);
    algorithm.parameter().resultsToCompute =
        compute_out_of_bag_error ? decision_forest::training::computeOutOfBagError : 0;
    algorithm.compute();
    /* Retrieve the algorithm results */
    decision_forest::regression::training::ResultPtr training_result =
        algorithm.getResult();

    auto model_ptr = training_result->get(decision_forest::regression::training::model);
    auto variable_importance_table =
        training_result->get(decision_forest::regression::training::variableImportance);
    const size_t num_features = input_features.size();
    std::vector<double> variable_importance(
        var_importance_metric != VarImportanceMetric::NONE ? num_features : 0);
    if (var_importance_metric != VarImportanceMetric::NONE) {
      for (size_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        variable_importance[feature_idx] =
            variable_importance_table->NumericTable::getValue<T>(feature_idx, size_t(0));
      }
    }
    double out_of_bag_error{0};
    if (compute_out_of_bag_error) {
      auto out_of_bag_error_table =
          training_result->get(decision_forest::regression::training::outOfBagError);
      out_of_bag_error =
          out_of_bag_error_table->NumericTable::getValue<T>(0, static_cast<size_t>(0));
    }
    auto model = std::make_shared<RandomForestRegressionModel>(model_ptr,
                                                               model_metadata,
                                                               cat_feature_keys,
                                                               variable_importance,
                                                               out_of_bag_error);
    g_ml_models.addModel(model_name, model);
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
NEVER_INLINE HOST int32_t onedal_decision_tree_reg_predict_impl(
    const std::shared_ptr<DecisionTreeRegressionModel>& model,
    const std::vector<const T*>& input_features,
    T* output_predictions,
    const int64_t num_rows) {
  CHECK(model->getModelType() == MLModelType::DECISION_TREE_REG);
  try {
    if (model->getNumFeatures() != static_cast<int64_t>(input_features.size())) {
      throw std::runtime_error("Number of provided features does not match model.");
    }
    const auto features_table = prepare_data_table(input_features, num_rows);
    decision_tree::regression::prediction::Batch<T> algorithm;
    algorithm.input.set(decision_tree::regression::prediction::data, features_table);
    algorithm.input.set(decision_tree::regression::prediction::model,
                        model->getModelPtr());

    const auto predictions_table =
        HomogenNumericTable<T>::create(output_predictions, 1, num_rows);

    const decision_tree::regression::prediction::ResultPtr result(
        new decision_tree::regression::prediction::Result);
    result->set(decision_tree::regression::prediction::prediction, predictions_table);
    algorithm.setResult(result);
    algorithm.compute();
    return num_rows;
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
NEVER_INLINE HOST int32_t
onedal_gbt_reg_predict_impl(const std::shared_ptr<GbtRegressionModel>& model,
                            const std::vector<const T*>& input_features,
                            T* output_predictions,
                            const int64_t num_rows) {
  CHECK(model->getModelType() == MLModelType::GBT_REG);
  try {
    if (model->getNumFeatures() != static_cast<int64_t>(input_features.size())) {
      throw std::runtime_error("Number of provided features does not match model.");
    }
    const auto features_table = prepare_data_table(input_features, num_rows);
    gbt::regression::prediction::Batch<T> algorithm;
    algorithm.input.set(gbt::regression::prediction::data, features_table);
    algorithm.input.set(gbt::regression::prediction::model, model->getModelPtr());

    const auto predictions_table =
        HomogenNumericTable<T>::create(output_predictions, 1, num_rows);

    const gbt::regression::prediction::ResultPtr result(
        new gbt::regression::prediction::Result);
    result->set(gbt::regression::prediction::prediction, predictions_table);
    algorithm.setResult(result);
    algorithm.compute();
    return num_rows;
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

template <typename T>
NEVER_INLINE HOST int32_t onedal_random_forest_reg_predict_impl(
    const std::shared_ptr<RandomForestRegressionModel>& model,
    const std::vector<const T*>& input_features,
    T* output_predictions,
    const int64_t num_rows) {
  CHECK(model->getModelType() == MLModelType::RANDOM_FOREST_REG);
  try {
    if (model->getNumFeatures() != static_cast<int64_t>(input_features.size())) {
      throw std::runtime_error("Number of provided features does not match model.");
    }
    const auto features_table = prepare_data_table(input_features, num_rows);
    decision_forest::regression::prediction::Batch<T> algorithm;
    algorithm.input.set(decision_forest::regression::prediction::data, features_table);
    algorithm.input.set(decision_forest::regression::prediction::model,
                        model->getModelPtr());

    const auto predictions_table =
        HomogenNumericTable<T>::create(output_predictions, 1, num_rows);

    const decision_forest::regression::prediction::ResultPtr result(
        new decision_forest::regression::prediction::Result);
    result->set(decision_forest::regression::prediction::prediction, predictions_table);
    algorithm.setResult(result);
    algorithm.compute();

    return num_rows;
  } catch (std::exception& e) {
    throw std::runtime_error(e.what());
  }
}

#endif  // #ifdef HAVE_ONEDAL
#endif  // #ifdef __CUDACC__
