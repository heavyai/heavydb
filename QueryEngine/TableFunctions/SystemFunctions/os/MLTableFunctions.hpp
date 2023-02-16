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

#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLTableFunctionsCommon.h"
#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/NullRowsRemoval.h"
#include "QueryEngine/heavydbTypes.h"

#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLModel.h"

#ifdef HAVE_ONEDAL
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/OneDalFunctions.hpp"
#endif

#ifdef HAVE_MLPACK
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLPackFunctions.hpp"
#endif

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

using namespace TableFunctions_Namespace;

template <typename T>
std::vector<const T*> pluck_ptrs(const std::vector<std::vector<T>>& data,
                                 const int64_t start_idx,
                                 const int64_t end_idx) {
  std::vector<const T*> raw_ptrs;
  CHECK_GE(start_idx, 0L);
  CHECK_GT(end_idx, start_idx);
  CHECK_LE(end_idx, static_cast<int64_t>(data.size()));
  for (int64_t col_idx = start_idx; col_idx < end_idx; ++col_idx) {
    raw_ptrs.emplace_back(data[col_idx].data());
  }
  return raw_ptrs;
}

template <typename T>
std::vector<const T*> pluck_ptrs(const std::vector<T*>& data,
                                 const int64_t start_idx,
                                 const int64_t end_idx) {
  std::vector<const T*> raw_ptrs;
  CHECK_GE(start_idx, 0L);
  CHECK_GT(end_idx, start_idx);
  CHECK_LE(end_idx, static_cast<int64_t>(data.size()));
  for (int64_t col_idx = start_idx; col_idx < end_idx; ++col_idx) {
    raw_ptrs.emplace_back(data[col_idx]);
  }
  return raw_ptrs;
}

// clang-format off
/*
  UDTF: supported_ml_frameworks__cpu_(TableFunctionManager) ->
  Column<TextEncodingDict> ml_framework | input_id=args<>, Column<bool> is_available, Column<bool> is_default
*/
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t supported_ml_frameworks__cpu_(TableFunctionManager& mgr,
                                      Column<TextEncodingDict>& output_ml_frameworks,
                                      Column<bool>& output_availability,
                                      Column<bool>& output_default);

// clang-format off
/*
  UDTF: kmeans__cpu_template(TableFunctionManager,
   Cursor<Column<K> input_ids, ColumnList<T> input_features> data,
   int32_t num_clusters | require="num_clusters > 0" | require="num_clusters <= input_ids.size()",
   int32_t num_iterations | require="num_iterations > 0" | default=10,
   TextEncodingNone init_type | default="DEFAULT",
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>,
   Column<int32_t> cluster_id,
   K=[int64_t, TextEncodingDict], T=[float, double]
*/
// clang-format on

template <typename K, typename T>
NEVER_INLINE HOST int32_t
kmeans__cpu_template(TableFunctionManager& mgr,
                     const Column<K>& input_ids,
                     const ColumnList<T>& input_features,
                     const int num_clusters,
                     const int num_iterations,
                     const TextEncodingNone& init_type_str,
                     const TextEncodingNone& preferred_ml_framework_str,
                     Column<K>& output_ids,
                     Column<int32_t>& output_clusters) {
  mgr.set_output_row_size(input_ids.size());
  output_ids = input_ids;
  const auto kmeans_init_strategy = get_kmeans_init_type(init_type_str);
  if (kmeans_init_strategy == KMeansInitStrategy::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid KMeans initializaiton strategy: " +
                             init_type_str.getString());
  }

  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }

  const auto denulled_data = denull_data(input_features);
  const int64_t num_rows = denulled_data.masked_num_rows;
  const bool data_is_masked =
      denulled_data.masked_num_rows < denulled_data.unmasked_num_rows;
  std::vector<int32_t> denulled_output_allocation(data_is_masked ? num_rows : 0);
  int32_t* denulled_output =
      data_is_masked ? denulled_output_allocation.data() : output_clusters.ptr_;

  const auto normalized_data = z_std_normalize_data(denulled_data.data, num_rows);
  const auto normalized_ptrs = pluck_ptrs(normalized_data, 0L, normalized_data.size());

  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_kmeans_impl(normalized_ptrs,
                         denulled_output,
                         num_rows,
                         num_clusters,
                         num_iterations,
                         kmeans_init_strategy);
      did_execute = true;
    }
#endif
#ifdef HAVE_MLPACK
    if (!did_execute && (preferred_ml_framework == MLFramework::MLPACK ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      mlpack_kmeans_impl(normalized_ptrs,
                         denulled_output,
                         num_rows,
                         num_clusters,
                         num_iterations,
                         kmeans_init_strategy);
      did_execute = true;
    }
#endif
    if (!did_execute) {
      return mgr.ERROR_MESSAGE("Cannot find " + preferred_ml_framework_str.getString() +
                               " ML library to support kmeans implementation.");
    }
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }

  if (data_is_masked) {
    unmask_data(denulled_output,
                denulled_data.reverse_index_map,
                output_clusters.ptr_,
                denulled_data.unmasked_num_rows,
                inline_null_value<int32_t>());
  }
  return input_ids.size();
}

// clang-format off
/*
  UDTF: dbscan__cpu_template(TableFunctionManager,
   Cursor<Column<K> input_ids, ColumnList<T> input_features> data,
   double epsilon | require="epsilon > 0.0",
   int32_t min_observations | require="min_observations > 0",
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<int32_t> cluster_id,
   K=[int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename K, typename T>
NEVER_INLINE HOST int32_t
dbscan__cpu_template(TableFunctionManager& mgr,
                     const Column<K>& input_ids,
                     const ColumnList<T>& input_features,
                     const double epsilon,
                     const int32_t min_observations,
                     const TextEncodingNone& preferred_ml_framework_str,
                     Column<K>& output_ids,
                     Column<int32_t>& output_clusters) {
  mgr.set_output_row_size(input_ids.size());
  output_ids = input_ids;

  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }

  const auto denulled_data = denull_data(input_features);
  const int64_t num_rows = denulled_data.masked_num_rows;
  const bool data_is_masked =
      denulled_data.masked_num_rows < denulled_data.unmasked_num_rows;
  std::vector<int32_t> denulled_output_allocation(data_is_masked ? num_rows : 0);
  int32_t* denulled_output =
      data_is_masked ? denulled_output_allocation.data() : output_clusters.ptr_;

  const auto normalized_data = z_std_normalize_data(denulled_data.data, num_rows);
  const auto normalized_ptrs = pluck_ptrs(normalized_data, 0L, normalized_data.size());

  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_dbscan_impl(
          normalized_ptrs, denulled_output, num_rows, epsilon, min_observations);
      did_execute = true;
    }
#endif
#ifdef HAVE_MLPACK
    if (!did_execute && (preferred_ml_framework == MLFramework::MLPACK ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      mlpack_dbscan_impl(
          normalized_ptrs, denulled_output, num_rows, epsilon, min_observations);
      did_execute = true;
    }
#endif
    if (!did_execute) {
      return mgr.ERROR_MESSAGE("Cannot find " + preferred_ml_framework_str.getString() +
                               " ML library to support dbscan implementation.");
    }
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }

  if (data_is_masked) {
    unmask_data(denulled_output,
                denulled_data.reverse_index_map,
                output_clusters.ptr_,
                denulled_data.unmasked_num_rows,
                inline_null_value<int32_t>());
  }
  return input_ids.size();
}

// clang-format off
/*
  UDTF: linear_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[float, double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
linear_reg_fit__cpu_template(TableFunctionManager& mgr,
                             const TextEncodingNone& model_name,
                             const Column<T>& input_labels,
                             const ColumnList<T>& input_features,
                             const TextEncodingNone& preferred_ml_framework_str,
                             Column<TextEncodingDict>& output_model_name) {
  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }
  const auto denulled_data = denull_data(input_labels, input_features);
  const auto labels_ptrs = pluck_ptrs(denulled_data.data, 0L, 1L);
  const auto features_ptrs =
      pluck_ptrs(denulled_data.data, 1L, input_features.numCols() + 1);
  const int64_t num_coefs = input_features.numCols() + 1;
  mgr.set_output_row_size(num_coefs);
  std::vector<int64_t> coef_idxs(num_coefs);
  std::vector<double> coefs(num_coefs);
  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_linear_reg_fit_impl(labels_ptrs[0],
                                 features_ptrs,
                                 coef_idxs.data(),
                                 coefs.data(),
                                 denulled_data.masked_num_rows);
      did_execute = true;
    }
#endif
#ifdef HAVE_MLPACK
    if (!did_execute && (preferred_ml_framework == MLFramework::MLPACK ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      mlpack_linear_reg_fit_impl(labels_ptrs[0],
                                 features_ptrs,
                                 coef_idxs.data(),
                                 coefs.data(),
                                 denulled_data.masked_num_rows);
      did_execute = true;
    }
#endif
    if (!did_execute) {
      return mgr.ERROR_MESSAGE(
          "Cannot find " + preferred_ml_framework_str.getString() +
          " ML library to support linear regression implementation.");
    }
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
  auto model = LinearRegressionModel(coefs);
  linear_reg_models_.addModel(model_name, model);
  auto model_type{ModelType::LINEAR_REG};
  model_types_.addModel(model_name, model_type);
  const std::string model_name_str = model_name.getString();
  if (model_name_str.rfind(temp_model_prefix) != 0) {
    // model_name does not start with temp model prefix,
    // which would be the case if we called this method from
    // lienar_reg_fit_predict, which in turn means that
    // output_model_name is not safe to write to.
    // Yes a bit hacky, but was considered the least invasive
    // solution in the face of issues synthesizing string
    // dictionary proxies on the fly for dummy string columns
    const TextEncodingDict model_name_str_id =
        output_model_name.getOrAddTransient(model_name);
    output_model_name[0] = model_name_str_id;
  }
  return 1;
}

// clang-format off
/*
  UDTF: linear_reg_predict__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<K> id, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
linear_reg_predict__cpu_template(TableFunctionManager& mgr,
                                 const TextEncodingNone& model_name,
                                 const Column<K>& input_ids,
                                 const ColumnList<T>& input_features,
                                 const TextEncodingNone& preferred_ml_framework_str,
                                 Column<K>& output_ids,
                                 Column<T>& output_predictions) {
  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }
  try {
    auto model = linear_reg_models_.getModel(model_name);
    if (static_cast<int64_t>(model.coefs.size()) != input_features.numCols() + 1) {
      mgr.enable_output_allocations();
      return mgr.ERROR_MESSAGE(
          "Number of model coefficients does not match number of input features.");
    }

    mgr.set_output_row_size(input_ids.size());
    const auto denulled_data = denull_data(input_features);
    const int64_t num_rows = denulled_data.masked_num_rows;
    const bool data_is_masked =
        denulled_data.masked_num_rows < denulled_data.unmasked_num_rows;
    std::vector<T> denulled_output_allocation(data_is_masked ? num_rows : 0);
    T* denulled_output =
        data_is_masked ? denulled_output_allocation.data() : output_predictions.ptr_;

    const auto features_ptrs =
        pluck_ptrs(denulled_data.data, 0L, input_features.numCols());

    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_linear_reg_predict_impl(
          features_ptrs, denulled_output, num_rows, model.coefs.data());
      did_execute = true;
    }
#endif
#ifdef HAVE_MLPACK
    if (!did_execute && (preferred_ml_framework == MLFramework::MLPACK ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      mlpack_linear_reg_predict_impl(
          features_ptrs, denulled_output, num_rows, model.coefs.data());
      did_execute = true;
    }
#endif
    if (!did_execute) {
      return mgr.ERROR_MESSAGE("Cannot find " + preferred_ml_framework_str.getString() +
                               " ML library to support kmeans implementation.");
    }
    output_ids = input_ids;
    if (data_is_masked) {
      unmask_data(denulled_output,
                  denulled_data.reverse_index_map,
                  output_predictions.ptr_,
                  denulled_data.unmasked_num_rows,
                  inline_null_value<T>());
    }
    return input_ids.size();
  } catch (std::runtime_error& e) {
    mgr.enable_output_allocations();
    return mgr.ERROR_MESSAGE(e.what());
  }
}

// clang-format off
/*
  UDTF: linear_reg_predict__cpu_template(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model,
   Cursor<Column<K> id, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
linear_reg_predict__cpu_template(TableFunctionManager& mgr,
                                 const Column<TextEncodingDict>& model_name,
                                 const Column<K>& input_ids,
                                 const ColumnList<T>& input_features,
                                 const TextEncodingNone& preferred_ml_framework_str,
                                 Column<K>& output_ids,
                                 Column<T>& output_predictions) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model CURSOR.");
  }
  const std::string model_name_str{model_name.getString(0)};
  return linear_reg_predict__cpu_template(mgr,
                                          model_name_str,
                                          input_ids,
                                          input_features,
                                          preferred_ml_framework_str,
                                          output_ids,
                                          output_predictions);
}

template <typename T>
Column<T> create_wrapper_col(std::vector<T>& col_vec) {
  Column<T> wrapper_col(col_vec.data(), static_cast<int64_t>(col_vec.size()));
  return wrapper_col;
}

// clang-format off
/*
  UDTF: linear_reg_coefs__cpu_1(TableFunctionManager,
  TextEncodingNone model) ->
  Column<int64_t> coef_idx, Column<double> coef
 */
// clang-format on

EXTENSION_NOINLINE_HOST int32_t
linear_reg_coefs__cpu_1(TableFunctionManager& mgr,
                        const TextEncodingNone& model_name,
                        Column<int64_t>& output_coef_idx,
                        Column<double>& output_coef);

// clang-format off
/*
  UDTF: linear_reg_coefs__cpu_2(TableFunctionManager,
  Cursor<Column<TextEncodingDict> model_name> model) ->
  Column<int64_t> coef_idx, Column<double> coef
 */
// clang-format on

EXTENSION_NOINLINE_HOST int32_t
linear_reg_coefs__cpu_2(TableFunctionManager& mgr,
                        const Column<TextEncodingDict>& model_name,
                        Column<int64_t>& output_coef_idx,
                        Column<double>& output_coef);

// clang-format off
/*
  UDTF: linear_reg_fit_predict__cpu_template(TableFunctionManager,
   Cursor<Column<K> id, Column<T> labels, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
linear_reg_fit_predict__cpu_template(TableFunctionManager& mgr,
                                     const Column<K>& input_ids,
                                     const Column<T>& input_labels,
                                     const ColumnList<T>& input_features,
                                     const TextEncodingNone& preferred_ml_framework_str,
                                     Column<K>& output_ids,
                                     Column<T>& output_predictions) {
  mgr.disable_output_allocations();
  const std::string model_name_str = temp_model_prefix + std::to_string(temp_model_idx++);
  TextEncodingNone model_name(model_name_str);
  std::vector<TextEncodingDict> output_model_name_vec(1);
  Column<TextEncodingDict> output_model_name(output_model_name_vec);
  // Use the transient proxy
  // Todo: create version of linear_reg_fit that takes in the model
  // parameters so has to avoid the need to do this
  // output_model_name.string_dict_proxy_ = mgr.getStringDictionaryProxy(1, 0);

  const auto fit_ret = linear_reg_fit__cpu_template(mgr,
                                                    model_name,
                                                    input_labels,
                                                    input_features,
                                                    preferred_ml_framework_str,
                                                    output_model_name);
  mgr.enable_output_allocations();
  if (fit_ret < 0) {
    return fit_ret;
  }
  return linear_reg_predict__cpu_template(mgr,
                                          model_name,
                                          input_ids,
                                          input_features,
                                          preferred_ml_framework_str,
                                          output_ids,
                                          output_predictions);
}

// clang-format off
/*
  UDTF: random_forest_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<T> features> data,
   int64_t num_trees | require="num_trees > 0" | default=10,
   int64_t max_tree_depth | require="max_tree_depth >= 0" | default=0,
   int64_t min_obs_per_leaf_node | require="min_obs_per_leaf_node > 0" | default=5,
   int64_t min_obs_per_split_node | require="min_obs_per_leaf_node > 0" | default=2,
   bool use_histogram | default=false,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[float, double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
random_forest_reg_fit__cpu_template(TableFunctionManager& mgr,
                                    const TextEncodingNone& model_name,
                                    const Column<T>& input_labels,
                                    const ColumnList<T>& input_features,
                                    const int64_t num_trees,
                                    const int64_t max_tree_depth,
                                    const int64_t min_observations_per_leaf_node,
                                    const int64_t min_observations_per_split_node,
                                    const bool use_histogram,
                                    const TextEncodingNone& preferred_ml_framework_str,
                                    Column<TextEncodingDict>& output_model_name) {
  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }
  if (preferred_ml_framework == MLFramework::MLPACK) {
    return mgr.ERROR_MESSAGE(
        "Only OneDAL framework supported for random forest regression.");
  }
#ifndef HAVE_ONEDAL
  return mgr.ERROR_MESSAGE(
      "Only OneDAL framework supported for random forest regression.");
#endif

  const auto denulled_data = denull_data(input_labels, input_features);
  const auto labels_ptrs = pluck_ptrs(denulled_data.data, 0L, 1L);
  const auto features_ptrs =
      pluck_ptrs(denulled_data.data, 1L, input_features.numCols() + 1);
  mgr.set_output_row_size(1);
  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      if (use_histogram) {
        onedal_random_forest_reg_fit_impl<T, decision_forest::regression::training::hist>(
            model_name,
            labels_ptrs[0],
            features_ptrs,
            denulled_data.masked_num_rows,
            num_trees,
            max_tree_depth,
            min_observations_per_leaf_node,
            min_observations_per_split_node);
      } else {
        onedal_random_forest_reg_fit_impl<
            T,
            decision_forest::regression::training::defaultDense>(
            model_name,
            labels_ptrs[0],
            features_ptrs,
            denulled_data.masked_num_rows,
            num_trees,
            max_tree_depth,
            min_observations_per_leaf_node,
            min_observations_per_split_node);
      }
      const TextEncodingDict model_name_str_id =
          output_model_name.getOrAddTransient(model_name);
      output_model_name[0] = model_name_str_id;
      did_execute = true;
      auto model_type{ModelType::RANDOM_FOREST_REG};
      model_types_.addModel(model_name, model_type);
    }
#endif
    if (!did_execute) {
      return mgr.ERROR_MESSAGE(
          "Cannot find " + preferred_ml_framework_str.getString() +
          " ML library to support random forest regression implementation.");
    }
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
  return 1;
}

// clang-format off
/*
  UDTF: random_forest_reg_predict__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<K> id, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int32_t, int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t random_forest_reg_predict__cpu_template(
    TableFunctionManager& mgr,
    const TextEncodingNone& model_name,
    const Column<K>& input_ids,
    const ColumnList<T>& input_features,
    const TextEncodingNone& preferred_ml_framework_str,
    Column<K>& output_ids,
    Column<T>& output_predictions) {
  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }
  if (preferred_ml_framework == MLFramework::MLPACK) {
    return mgr.ERROR_MESSAGE(
        "Only OneDAL framework supported for random forest regression.");
  }
#ifndef HAVE_ONEDAL
  return mgr.ERROR_MESSAGE(
      "Only OneDAL framework supported for random forest regression.");
#endif

  mgr.set_output_row_size(input_ids.size());
  const auto denulled_data = denull_data(input_features);
  const int64_t num_rows = denulled_data.masked_num_rows;
  const bool data_is_masked =
      denulled_data.masked_num_rows < denulled_data.unmasked_num_rows;
  std::vector<T> denulled_output_allocation(data_is_masked ? num_rows : 0);
  T* denulled_output =
      data_is_masked ? denulled_output_allocation.data() : output_predictions.ptr_;

  const auto features_ptrs = pluck_ptrs(denulled_data.data, 0L, input_features.numCols());

  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_random_forest_reg_predict_impl(
          model_name, features_ptrs, denulled_output, num_rows);
      did_execute = true;
    }
#endif
    if (!did_execute) {
      return mgr.ERROR_MESSAGE("Cannot find " + preferred_ml_framework_str.getString() +
                               " ML library to support kmeans implementation.");
    }
  } catch (std::runtime_error& e) {
    mgr.enable_output_allocations();
    return mgr.ERROR_MESSAGE(e.what());
  }
  output_ids = input_ids;
  if (data_is_masked) {
    unmask_data(denulled_output,
                denulled_data.reverse_index_map,
                output_predictions.ptr_,
                denulled_data.unmasked_num_rows,
                inline_null_value<T>());
  }
  return input_ids.size();
}

// clang-format off
/*
  UDTF: random_forest_reg_predict__cpu_template(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model,
   Cursor<Column<K> id, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t random_forest_reg_predict__cpu_template(
    TableFunctionManager& mgr,
    const Column<TextEncodingDict>& model_name,
    const Column<K>& input_ids,
    const ColumnList<T>& input_features,
    const TextEncodingNone& preferred_ml_framework_str,
    Column<K>& output_ids,
    Column<T>& output_predictions) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model CURSOR.");
  }
  const std::string model_name_str{model_name.getString(0)};
  return random_forest_reg_predict__cpu_template(mgr,
                                                 model_name_str,
                                                 input_ids,
                                                 input_features,
                                                 preferred_ml_framework_str,
                                                 output_ids,
                                                 output_predictions);
}

// clang-format off
/*
  UDTF: r2_score__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<T> features> data) ->
   Column<double> r2, T=[float, double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t r2_score__cpu_template(TableFunctionManager& mgr,
                                                 const TextEncodingNone& model_name,
                                                 const Column<T>& input_labels,
                                                 const ColumnList<T>& input_features,
                                                 Column<double>& output_r2) {
  const int64_t num_rows = input_labels.size();
  std::vector<T> output_predictions_vec(num_rows);
  Column<T> output_predictions(output_predictions_vec);
  std::vector<int64_t> input_ids_vec(num_rows);
  std::vector<int64_t> output_ids_vec(num_rows);
  Column<int64_t> input_ids(input_ids_vec);
  Column<int64_t> output_ids(output_ids_vec);
  mgr.disable_output_allocations();
  TextEncodingNone model_name_encoding_none(model_name);
  const std::string ml_framework("DEFAULT");
  TextEncodingNone ml_framework_encoding_none(ml_framework);

  try {
    const auto model_type = model_types_.getModel(model_name);
    int32_t ret{0};
    switch (model_type) {
      case ModelType::LINEAR_REG: {
        ret = linear_reg_predict__cpu_template(mgr,
                                               model_name_encoding_none,
                                               input_ids,
                                               input_features,
                                               ml_framework_encoding_none,
                                               output_ids,
                                               output_predictions);
        break;
      }
      case ModelType::RANDOM_FOREST_REG: {
        ret = random_forest_reg_predict__cpu_template(mgr,
                                                      model_name_encoding_none,
                                                      input_ids,
                                                      input_features,
                                                      ml_framework_encoding_none,
                                                      output_ids,
                                                      output_predictions);
        break;
      }
    }
    if (ret < 0) {
      // A return of less than 0 symbolizes an error
      return ret;
    }
  } catch (std::runtime_error& e) {
    mgr.enable_output_allocations();
    return mgr.ERROR_MESSAGE(e.what());
  }

  mgr.enable_output_allocations();
  mgr.set_output_row_size(1);

  const auto labels_mean = get_column_mean(input_labels);
  const size_t max_thread_count = std::thread::hardware_concurrency();
  const size_t max_inputs_per_thread = 20000;
  const size_t num_threads = std::min(
      max_thread_count, ((num_rows + max_inputs_per_thread - 1) / max_inputs_per_thread));

  std::vector<double> local_sum_squared_regressions(num_threads, 0.0);
  std::vector<double> local_sum_squares(num_threads, 0.0);

  tbb::task_arena limited_arena(num_threads);

  limited_arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, num_rows),
        [&](const tbb::blocked_range<int64_t>& r) {
          const int64_t start_idx = r.begin();
          const int64_t end_idx = r.end();
          double local_sum_squared_regression{0.0};
          double local_sum_square{0.0};
          for (int64_t row_idx = start_idx; row_idx < end_idx; ++row_idx) {
            if (output_predictions[row_idx] != inline_null_value<int32_t>()) {
              local_sum_squared_regression +=
                  (input_labels[row_idx] - output_predictions[row_idx]) *
                  (input_labels[row_idx] - output_predictions[row_idx]);
              local_sum_square += (input_labels[row_idx] - labels_mean) *
                                  (input_labels[row_idx] - labels_mean);
            }
          }
          const size_t thread_idx = tbb::this_task_arena::current_thread_index();
          local_sum_squared_regressions[thread_idx] += local_sum_squared_regression;
          local_sum_squares[thread_idx] += local_sum_square;
        });
  });
  double sum_squared_regression{0.0};
  double sum_squares{0.0};
  for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    sum_squared_regression += local_sum_squared_regressions[thread_idx];
    sum_squares += local_sum_squares[thread_idx];
  }
  output_r2[0] = sum_squares == 0.0 ? 1.0 : 1.0 - (sum_squared_regression / sum_squares);
  return 1;
}

// clang-format off
/*
  UDTF: r2_score__cpu_template(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model,
   Cursor<Column<T> labels, ColumnList<T> features> data) ->
   Column<double> r2, T=[float, double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
r2_score__cpu_template(TableFunctionManager& mgr,
                       const Column<TextEncodingDict>& model_name,
                       const Column<T>& input_labels,
                       const ColumnList<T>& input_features,
                       Column<double>& output_r2) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model name CURSOR.");
  }
  const std::string model_name_str{model_name.getString(0)};
  return r2_score__cpu_template(
      mgr, model_name_str, input_labels, input_features, output_r2);
}

// clang-format off
/*
  UDTF: random_forest_reg_var_importance__cpu_1(TableFunctionManager,
   TextEncodingNone model) ->
   Column<int64_t> feature_id, Column<double> importance_score
 */
// clang-format on

EXTENSION_NOINLINE_HOST int32_t
random_forest_reg_var_importance__cpu_1(TableFunctionManager& mgr,
                                        const TextEncodingNone& model_name,
                                        Column<int64_t>& feature_id,
                                        Column<double>& importance_score);

// clang-format off
/*
  UDTF: random_forest_reg_var_importance__cpu_2(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model) ->
   Column<int64_t> feature_id, Column<double> importance_score
 */
// clang-format on

EXTENSION_NOINLINE_HOST int32_t
random_forest_reg_var_importance__cpu_2(TableFunctionManager& mgr,
                                        const Column<TextEncodingDict>& model_name,
                                        Column<int64_t>& feature_id,
                                        Column<double>& importance_score);
#endif  // #ifndef __CUDACC__
