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
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/OneHotEncoder.h"

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
EXTENSION_NOINLINE_HOST
void check_model_params(const std::shared_ptr<AbstractMLModel>& model,
                        const int64_t num_cat_features,
                        const int64_t num_numeric_features);

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
   K=[int64_t, TextEncodingDict], T=[double]
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
   K=[int64_t, TextEncodingDict], T=[double]
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

template <typename T>
NEVER_INLINE HOST int32_t
linear_reg_fit_impl(TableFunctionManager& mgr,
                    const TextEncodingNone& model_name,
                    const Column<T>& input_labels,
                    const ColumnList<T>& input_features,
                    const std::vector<std::vector<std::string>>& cat_feature_keys,
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
  auto model = std::make_shared<LinearRegressionModel>(coefs, cat_feature_keys);
  ml_models_.addModel(model_name, model);
  const std::string model_name_str = model_name.getString();
  const TextEncodingDict model_name_str_id =
      output_model_name.getOrAddTransient(model_name);
  output_model_name[0] = model_name_str_id;
  return 1;
}

// clang-format off
/*
  UDTF: linear_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[double]
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
  std::vector<std::vector<std::string>> empty_cat_feature_keys;
  return linear_reg_fit_impl(mgr,
                             model_name,
                             input_labels,
                             input_features,
                             empty_cat_feature_keys,
                             preferred_ml_framework_str,
                             output_model_name);
}

template <typename T>
struct CategoricalFeaturesBuilder {
 public:
  CategoricalFeaturesBuilder(const ColumnList<TextEncodingDict>& cat_features,
                             const ColumnList<T>& numeric_features,
                             const int32_t top_k_cat_attrs,
                             const float min_cat_attr_proportion,
                             const bool cat_include_others)
      : num_rows_(numeric_features.size()) {
    TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo
        one_hot_encoding_info(
            top_k_cat_attrs, min_cat_attr_proportion, cat_include_others);
    const size_t num_cat_features = static_cast<size_t>(cat_features.numCols());
    std::vector<TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo>
        one_hot_encoding_infos;
    for (size_t cat_idx = 0; cat_idx < num_cat_features; ++cat_idx) {
      one_hot_encoding_infos.emplace_back(one_hot_encoding_info);
    }
    one_hot_encoded_cols_ =
        TableFunctions_Namespace::OneHotEncoder_Namespace::one_hot_encode<T>(
            cat_features, one_hot_encoding_infos);
    for (auto& one_hot_encoded_col : one_hot_encoded_cols_) {
      cat_feature_keys_.emplace_back(one_hot_encoded_col.cat_features);
      for (auto& one_hot_encoded_vec : one_hot_encoded_col.encoded_buffers) {
        col_ptrs_.emplace_back(reinterpret_cast<int8_t*>(one_hot_encoded_vec.data()));
      }
    }
    const int64_t num_numeric_features = numeric_features.numCols();
    for (int64_t numeric_feature_idx = 0; numeric_feature_idx < num_numeric_features;
         ++numeric_feature_idx) {
      col_ptrs_.emplace_back(numeric_features.ptrs_[numeric_feature_idx]);
    }
  }

  CategoricalFeaturesBuilder(
      const ColumnList<TextEncodingDict>& cat_features,
      const ColumnList<T>& numeric_features,
      const std::vector<std::vector<std::string>>& cat_feature_keys)
      : num_rows_(numeric_features.size()), cat_feature_keys_(cat_feature_keys) {
    const size_t num_cat_features = static_cast<size_t>(cat_features.numCols());
    if (num_cat_features != cat_feature_keys_.size()) {
      throw std::runtime_error(
          "Number of provided categorical features does not match number of categorical "
          "features in the model.");
    }
    std::vector<TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo>
        one_hot_encoding_infos;
    for (size_t cat_idx = 0; cat_idx < num_cat_features; ++cat_idx) {
      one_hot_encoding_infos.emplace_back(cat_feature_keys_[cat_idx]);
    }
    one_hot_encoded_cols_ =
        TableFunctions_Namespace::OneHotEncoder_Namespace::one_hot_encode<T>(
            cat_features, one_hot_encoding_infos);
    for (auto& one_hot_encoded_col : one_hot_encoded_cols_) {
      for (auto& one_hot_encoded_vec : one_hot_encoded_col.encoded_buffers) {
        col_ptrs_.emplace_back(reinterpret_cast<int8_t*>(one_hot_encoded_vec.data()));
      }
    }
    const int64_t num_numeric_features = numeric_features.numCols();
    for (int64_t numeric_feature_idx = 0; numeric_feature_idx < num_numeric_features;
         ++numeric_feature_idx) {
      col_ptrs_.emplace_back(numeric_features.ptrs_[numeric_feature_idx]);
    }
  }

  ColumnList<T> getFeatures() {
    return ColumnList<T>(
        col_ptrs_.data(), static_cast<int64_t>(col_ptrs_.size()), num_rows_);
  }

  const std::vector<std::vector<std::string>>& getCatFeatureKeys() const {
    return cat_feature_keys_;
  }

 private:
  int64_t num_rows_;
  std::vector<TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodedCol<T>>
      one_hot_encoded_cols_;
  std::vector<std::vector<std::string>> cat_feature_keys_;
  std::vector<int8_t*> col_ptrs_;
};

// clang-format off
/*
  UDTF: linear_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<TextEncodingDict> cat_features, 
   ColumnList<T> numeric_features> data,
   int32_t top_k_cat_attrs | require="top_k_cat_attrs >= 1" | default=10,
   float min_cat_attr_proportion | require="min_cat_attr_proportion > 0.0" | require="min_cat_attr_proportion <= 1.0" | default=0.01,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
linear_reg_fit__cpu_template(TableFunctionManager& mgr,
                             const TextEncodingNone& model_name,
                             const Column<T>& input_labels,
                             const ColumnList<TextEncodingDict>& input_cat_features,
                             const ColumnList<T>& input_numeric_features,
                             const int32_t top_k_cat_attrs,
                             const float min_cat_attr_proportion,
                             const TextEncodingNone& preferred_ml_framework_str,
                             Column<TextEncodingDict>& output_model_name) {
  CategoricalFeaturesBuilder<T> cat_features_builder(input_cat_features,
                                                     input_numeric_features,
                                                     top_k_cat_attrs,
                                                     min_cat_attr_proportion,
                                                     false /* cat_include_others */);

  return linear_reg_fit_impl(mgr,
                             model_name,
                             input_labels,
                             cat_features_builder.getFeatures(),
                             cat_features_builder.getCatFeatureKeys(),
                             preferred_ml_framework_str,
                             output_model_name);
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
  Column<int64_t> coef_idx, Column<int64_t> sub_coef_idx ,Column<TextEncodingDict> sub_coef_attr | input_id=args<>,
  Column<double> coef
 */
// clang-format on

EXTENSION_NOINLINE_HOST int32_t
linear_reg_coefs__cpu_1(TableFunctionManager& mgr,
                        const TextEncodingNone& model_name,
                        Column<int64_t>& output_coef_idx,
                        Column<int64_t>& output_sub_coef_idx,
                        Column<TextEncodingDict>& output_sub_attr,
                        Column<double>& output_coef);

// clang-format off
/*
  UDTF: linear_reg_coefs__cpu_2(TableFunctionManager,
  Cursor<Column<TextEncodingDict> model_name> model) ->
  Column<int64_t> coef_idx, Column<int64_t> sub_coef_idx, Column<TextEncodingDict> sub_coef_attr | input_id=args<>,
  Column<double> coef
 */
// clang-format on

EXTENSION_NOINLINE_HOST int32_t
linear_reg_coefs__cpu_2(TableFunctionManager& mgr,
                        const Column<TextEncodingDict>& model_name,
                        Column<int64_t>& output_coef_idx,
                        Column<int64_t>& output_sub_coef_idx,
                        Column<TextEncodingDict>& output_sub_attr,
                        Column<double>& output_coef);

template <typename T>
NEVER_INLINE HOST int32_t
decision_tree_reg_impl(TableFunctionManager& mgr,
                       const TextEncodingNone& model_name,
                       const Column<T>& input_labels,
                       const ColumnList<T>& input_features,
                       const std::vector<std::vector<std::string>>& cat_feature_keys,
                       const int64_t max_tree_depth,
                       const int64_t min_observations_per_leaf_node,
                       const TextEncodingNone& preferred_ml_framework_str,
                       Column<TextEncodingDict>& output_model_name) {
  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }
  if (preferred_ml_framework == MLFramework::MLPACK) {
    return mgr.ERROR_MESSAGE(
        "Only OneDAL framework supported for decision tree regression.");
  }
#ifndef HAVE_ONEDAL
  return mgr.ERROR_MESSAGE(
      "Only OneDAL framework supported for decision tree regression.");
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
      onedal_decision_tree_reg_fit_impl<T>(model_name,
                                           labels_ptrs[0],
                                           features_ptrs,
                                           cat_feature_keys,
                                           denulled_data.masked_num_rows,
                                           max_tree_depth,
                                           min_observations_per_leaf_node);
      const TextEncodingDict model_name_str_id =
          output_model_name.getOrAddTransient(model_name);
      output_model_name[0] = model_name_str_id;
      did_execute = true;
    }
#endif
    if (!did_execute) {
      return mgr.ERROR_MESSAGE(
          "Cannot find " + preferred_ml_framework_str.getString() +
          " ML library to support decision tree regression implementation.");
    }
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
  return 1;
}

// clang-format off
/*
  UDTF: decision_tree_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<T> features> data,
   int64_t max_tree_depth | require="max_tree_depth >= 0" | default=0,
   int64_t min_obs_per_leaf_node | require="min_obs_per_leaf_node >= 0" | default=5,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
decision_tree_reg_fit__cpu_template(TableFunctionManager& mgr,
                                    const TextEncodingNone& model_name,
                                    const Column<T>& input_labels,
                                    const ColumnList<T>& input_features,
                                    const int64_t max_tree_depth,
                                    const int64_t min_observations_per_leaf_node,
                                    const TextEncodingNone& preferred_ml_framework_str,
                                    Column<TextEncodingDict>& output_model_name) {
  std::vector<std::vector<std::string>> empty_cat_feature_keys;
  return decision_tree_reg_impl(mgr,
                                model_name,
                                input_labels,
                                input_features,
                                empty_cat_feature_keys,
                                max_tree_depth,
                                min_observations_per_leaf_node,
                                preferred_ml_framework_str,
                                output_model_name);
}

// clang-format off
/*
  UDTF: decision_tree_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<TextEncodingDict> cat_features, ColumnList<T> numeric_features> data,
   int64_t max_tree_depth | require="max_tree_depth >= 0" | default=0,
   int64_t min_obs_per_leaf_node | require="min_obs_per_leaf_node >= 0" | default=5,
   int32_t top_k_cat_attrs | require="top_k_cat_attrs >= 1" | default=10,
   float min_cat_attr_proportion | require="min_cat_attr_proportion > 0.0" | require="min_cat_attr_proportion <= 1.0" | default=0.01,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t decision_tree_reg_fit__cpu_template(
    TableFunctionManager& mgr,
    const TextEncodingNone& model_name,
    const Column<T>& input_labels,
    const ColumnList<TextEncodingDict>& input_cat_features,
    const ColumnList<T>& input_numeric_features,
    const int64_t max_tree_depth,
    const int64_t min_observations_per_leaf_node,
    const int32_t top_k_cat_attrs,
    const float min_cat_attr_proportion,
    const TextEncodingNone& preferred_ml_framework_str,
    Column<TextEncodingDict>& output_model_name) {
  std::vector<std::vector<std::string>> empty_cat_feature_keys;
  CategoricalFeaturesBuilder<T> cat_features_builder(input_cat_features,
                                                     input_numeric_features,
                                                     top_k_cat_attrs,
                                                     min_cat_attr_proportion,
                                                     false /* cat_include_others */);
  return decision_tree_reg_impl(mgr,
                                model_name,
                                input_labels,
                                cat_features_builder.getFeatures(),
                                cat_features_builder.getCatFeatureKeys(),
                                max_tree_depth,
                                min_observations_per_leaf_node,
                                preferred_ml_framework_str,
                                output_model_name);
}

template <typename T>
NEVER_INLINE HOST int32_t
gbt_reg_fit_impl(TableFunctionManager& mgr,
                 const TextEncodingNone& model_name,
                 const Column<T>& input_labels,
                 const ColumnList<T>& input_features,
                 const std::vector<std::vector<std::string>>& cat_feature_keys,
                 const int64_t max_iterations,
                 const int64_t max_tree_depth,
                 const double shrinkage,
                 const double min_split_loss,
                 const double lambda,
                 const double obs_per_tree_fraction,
                 const int64_t features_per_node,
                 const int64_t min_observations_per_leaf_node,
                 const int64_t max_bins,
                 const int64_t min_bin_size,
                 const TextEncodingNone& preferred_ml_framework_str,
                 Column<TextEncodingDict>& output_model_name) {
  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }
  if (preferred_ml_framework == MLFramework::MLPACK) {
    return mgr.ERROR_MESSAGE("Only OneDAL framework supported for GBT regression.");
  }
#ifndef HAVE_ONEDAL
  return mgr.ERROR_MESSAGE("Only OneDAL framework supported for GBT regression.");
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
      onedal_gbt_reg_fit_impl<T>(model_name,
                                 labels_ptrs[0],
                                 features_ptrs,
                                 cat_feature_keys,
                                 denulled_data.masked_num_rows,
                                 max_iterations,
                                 max_tree_depth,
                                 shrinkage,
                                 min_split_loss,
                                 lambda,
                                 obs_per_tree_fraction,
                                 features_per_node,
                                 min_observations_per_leaf_node,
                                 max_bins,
                                 min_bin_size);
      const TextEncodingDict model_name_str_id =
          output_model_name.getOrAddTransient(model_name);
      output_model_name[0] = model_name_str_id;
      did_execute = true;
    }
#endif
    if (!did_execute) {
      return mgr.ERROR_MESSAGE("Cannot find " + preferred_ml_framework_str.getString() +
                               " ML library to support GBT regression implementation.");
    }
  } catch (std::runtime_error& e) {
    return mgr.ERROR_MESSAGE(e.what());
  }
  return 1;
}

// clang-format off
/*
  UDTF: gbt_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<T> features> data,
   int64_t max_iterations | require="max_iterations > 0" | default=50,
   int64_t max_tree_depth | require="max_tree_depth > 0" | default=6,
   double shrinkage | require="shrinkage > 0.0" | require="shrinkage <= 1.0" | default=0.3,
   double min_split_loss | require="min_split_loss >= 0.0" | default=0.0,
   double lambda | require="lambda >= 0.0" | default=1.0,
   double obs_per_tree_fraction | require="obs_per_tree_fraction > 0.0" | require="obs_per_tree_fraction <= 1.0" | default=1.0,
   int64_t features_per_node | require="features_per_node >= 0" | default=0,
   int64_t min_obs_per_leaf_node | require="min_obs_per_leaf_node > 0" | default=5,
   int64_t max_bins | require="max_bins > 0" | default=256,
   int64_t min_bin_size | require="min_bin_size >= 0" | default=5,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
gbt_reg_fit__cpu_template(TableFunctionManager& mgr,
                          const TextEncodingNone& model_name,
                          const Column<T>& input_labels,
                          const ColumnList<T>& input_features,
                          const int64_t max_iterations,
                          const int64_t max_tree_depth,
                          const double shrinkage,
                          const double min_split_loss,
                          const double lambda,
                          const double obs_per_tree_fraction,
                          const int64_t features_per_node,
                          const int64_t min_observations_per_leaf_node,
                          const int64_t max_bins,
                          const int64_t min_bin_size,
                          const TextEncodingNone& preferred_ml_framework_str,
                          Column<TextEncodingDict>& output_model_name) {
  std::vector<std::vector<std::string>> empty_cat_feature_keys;
  return gbt_reg_fit_impl(mgr,
                          model_name,
                          input_labels,
                          input_features,
                          empty_cat_feature_keys,
                          max_iterations,
                          max_tree_depth,
                          shrinkage,
                          min_split_loss,
                          lambda,
                          obs_per_tree_fraction,
                          features_per_node,
                          min_observations_per_leaf_node,
                          max_bins,
                          min_bin_size,
                          preferred_ml_framework_str,
                          output_model_name);
}

// clang-format off
/*
  UDTF: gbt_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<TextEncodingDict> cat_features, ColumnList<T> numeric_features> data,
   int64_t max_iterations | require="max_iterations > 0" | default=50,
   int64_t max_tree_depth | require="max_tree_depth > 0" | default=6,
   double shrinkage | require="shrinkage > 0.0" | require="shrinkage <= 1.0" | default=0.3,
   double min_split_loss | require="min_split_loss >= 0.0" | default=0.0,
   double lambda | require="lambda >= 0.0" | default=1.0,
   double obs_per_tree_fraction | require="obs_per_tree_fraction > 0.0" | require="obs_per_tree_fraction <= 1.0" | default=1.0,
   int64_t features_per_node | require="features_per_node >= 0" | default=0,
   int64_t min_obs_per_leaf_node | require="min_obs_per_leaf_node > 0" | default=5,
   int64_t max_bins | require="max_bins > 0" | default=256,
   int64_t min_bin_size | require="min_bin_size >= 0" | default=5,
   int32_t top_k_cat_attrs | require="top_k_cat_attrs >= 1" | default=10,
   float min_cat_attr_proportion | require="min_cat_attr_proportion > 0.0" | require="min_cat_attr_proportion <= 1.0" | default=0.01,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
gbt_reg_fit__cpu_template(TableFunctionManager& mgr,
                          const TextEncodingNone& model_name,
                          const Column<T>& input_labels,
                          const ColumnList<TextEncodingDict>& input_cat_features,
                          const ColumnList<T>& input_numeric_features,
                          const int64_t max_iterations,
                          const int64_t max_tree_depth,
                          const double shrinkage,
                          const double min_split_loss,
                          const double lambda,
                          const double obs_per_tree_fraction,
                          const int64_t features_per_node,
                          const int64_t min_observations_per_leaf_node,
                          const int64_t max_bins,
                          const int64_t min_bin_size,
                          const int32_t top_k_cat_attrs,
                          const float min_cat_attr_proportion,
                          const TextEncodingNone& preferred_ml_framework_str,
                          Column<TextEncodingDict>& output_model_name) {
  CategoricalFeaturesBuilder<T> cat_features_builder(input_cat_features,
                                                     input_numeric_features,
                                                     top_k_cat_attrs,
                                                     min_cat_attr_proportion,
                                                     false /* cat_include_others */);
  return gbt_reg_fit_impl(mgr,
                          model_name,
                          input_labels,
                          cat_features_builder.getFeatures(),
                          cat_features_builder.getCatFeatureKeys(),
                          max_iterations,
                          max_tree_depth,
                          shrinkage,
                          min_split_loss,
                          lambda,
                          obs_per_tree_fraction,
                          features_per_node,
                          min_observations_per_leaf_node,
                          max_bins,
                          min_bin_size,
                          preferred_ml_framework_str,
                          output_model_name);
}

template <typename T>
NEVER_INLINE HOST int32_t
random_forest_reg_fit_impl(TableFunctionManager& mgr,
                           const TextEncodingNone& model_name,
                           const Column<T>& input_labels,
                           const ColumnList<T>& input_features,
                           const std::vector<std::vector<std::string>>& cat_feature_keys,
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
                           const bool use_histogram,
                           const TextEncodingNone& var_importance_metric_str,
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
    const auto var_importance_metric =
        get_var_importance_metric(var_importance_metric_str);
    if (var_importance_metric == VarImportanceMetric::INVALID) {
      return mgr.ERROR_MESSAGE("Invalid variable importance metric: " +
                               var_importance_metric_str.getString());
    }
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      if (use_histogram) {
        onedal_random_forest_reg_fit_impl<T, decision_forest::regression::training::hist>(
            model_name,
            labels_ptrs[0],
            features_ptrs,
            cat_feature_keys,
            denulled_data.masked_num_rows,
            num_trees,
            obs_per_tree_fraction,
            max_tree_depth,
            features_per_node,
            impurity_threshold,
            bootstrap,
            min_obs_per_leaf_node,
            min_obs_per_split_node,
            min_weight_fraction_in_leaf_node,
            min_impurity_decrease_in_split_node,
            max_leaf_nodes,
            var_importance_metric);
      } else {
        onedal_random_forest_reg_fit_impl<
            T,
            decision_forest::regression::training::defaultDense>(
            model_name,
            labels_ptrs[0],
            features_ptrs,
            cat_feature_keys,
            denulled_data.masked_num_rows,
            num_trees,
            obs_per_tree_fraction,
            max_tree_depth,
            features_per_node,
            impurity_threshold,
            bootstrap,
            min_obs_per_leaf_node,
            min_obs_per_split_node,
            min_weight_fraction_in_leaf_node,
            min_impurity_decrease_in_split_node,
            max_leaf_nodes,
            var_importance_metric);
      }
      const TextEncodingDict model_name_str_id =
          output_model_name.getOrAddTransient(model_name);
      output_model_name[0] = model_name_str_id;
      did_execute = true;
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
  UDTF: random_forest_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<T> features> data,
   int64_t num_trees | require="num_trees > 0" | default=10,
   double obs_per_tree_fraction | require="obs_per_tree_fraction > 0.0" | require="obs_per_tree_fraction <= 1.0" | default=1.0,
   int64_t max_tree_depth | require="max_tree_depth >= 0" | default=0,
   int64_t features_per_node | require="features_per_node >= 0" | default=0,
   double impurity_threshold | require="impurity_threshold >= 0.0" | default=0.0,
   bool bootstrap | default=true,
   int64_t min_obs_per_leaf_node | require="min_obs_per_leaf_node > 0" | default=5,
   int64_t min_obs_per_split_node | require="min_obs_per_leaf_node > 0" | default=2,
   double min_weight_fraction_in_leaf_node | require="min_weight_fraction_in_leaf_node >= 0.0" | default=0.0,
   double min_impurity_decrease_in_split_node | require="min_impurity_decrease_in_split_node >= 0.0" | default=0.0,
   int64_t max_leaf_nodes | require="max_leaf_nodes >=0" | default=0,
   bool use_histogram | default=false,
   TextEncodingNone var_importance_metric | default="MDI",
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
random_forest_reg_fit__cpu_template(TableFunctionManager& mgr,
                                    const TextEncodingNone& model_name,
                                    const Column<T>& input_labels,
                                    const ColumnList<T>& input_features,
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
                                    const bool use_histogram,
                                    const TextEncodingNone& var_importance_metric_str,
                                    const TextEncodingNone& preferred_ml_framework_str,
                                    Column<TextEncodingDict>& output_model_name) {
  std::vector<std::vector<std::string>> empty_cat_feature_keys;
  return random_forest_reg_fit_impl(mgr,
                                    model_name,
                                    input_labels,
                                    input_features,
                                    empty_cat_feature_keys,
                                    num_trees,
                                    obs_per_tree_fraction,
                                    max_tree_depth,
                                    features_per_node,
                                    impurity_threshold,
                                    bootstrap,
                                    min_obs_per_leaf_node,
                                    min_obs_per_split_node,
                                    min_weight_fraction_in_leaf_node,
                                    min_impurity_decrease_in_split_node,
                                    max_leaf_nodes,
                                    use_histogram,
                                    var_importance_metric_str,
                                    preferred_ml_framework_str,
                                    output_model_name);
}

// clang-format off
/*
  UDTF: random_forest_reg_fit__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<TextEncodingDict> cat_features, ColumnList<T> numeric_features> data,
   int64_t num_trees | require="num_trees > 0" | default=10,
   double obs_per_tree_fraction | require="obs_per_tree_fraction > 0.0" | require="obs_per_tree_fraction <= 1.0" | default=1.0,
   int64_t max_tree_depth | require="max_tree_depth >= 0" | default=0,
   int64_t features_per_node | require="features_per_node >= 0" | default=0,
   double impurity_threshold | require="impurity_threshold >= 0.0" | default=0.0,
   bool bootstrap | default=true,
   int64_t min_obs_per_leaf_node | require="min_obs_per_leaf_node > 0" | default=5,
   int64_t min_obs_per_split_node | require="min_obs_per_leaf_node > 0" | default=2,
   double min_weight_fraction_in_leaf_node | require="min_weight_fraction_in_leaf_node >= 0.0" | default=0.0,
   double min_impurity_decrease_in_split_node | require="min_impurity_decrease_in_split_node >= 0.0" | default=0.0,
   int64_t max_leaf_nodes | require="max_leaf_nodes >=0" | default=0,
   bool use_histogram | default=false,
   TextEncodingNone var_importance_metric | default="MDI",
   int32_t top_k_cat_attrs | require="top_k_cat_attrs >= 1" | default=10,
   float min_cat_attr_proportion | require="min_cat_attr_proportion > 0.0" | require="min_cat_attr_proportion <= 1.0" | default=0.01,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<TextEncodingDict> model | input_id=args<>, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t random_forest_reg_fit__cpu_template(
    TableFunctionManager& mgr,
    const TextEncodingNone& model_name,
    const Column<T>& input_labels,
    const ColumnList<TextEncodingDict>& input_cat_features,
    const ColumnList<T>& input_numeric_features,
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
    const bool use_histogram,
    const TextEncodingNone& var_importance_metric_str,
    const int32_t top_k_cat_attrs,
    const float min_cat_attr_proportion,
    const TextEncodingNone& preferred_ml_framework_str,
    Column<TextEncodingDict>& output_model_name) {
  CategoricalFeaturesBuilder<T> cat_features_builder(input_cat_features,
                                                     input_numeric_features,
                                                     top_k_cat_attrs,
                                                     min_cat_attr_proportion,
                                                     false /* cat_include_others */);
  return random_forest_reg_fit_impl(mgr,
                                    model_name,
                                    input_labels,
                                    cat_features_builder.getFeatures(),
                                    cat_features_builder.getCatFeatureKeys(),
                                    num_trees,
                                    obs_per_tree_fraction,
                                    max_tree_depth,
                                    features_per_node,
                                    impurity_threshold,
                                    bootstrap,
                                    min_obs_per_leaf_node,
                                    min_obs_per_split_node,
                                    min_weight_fraction_in_leaf_node,
                                    min_impurity_decrease_in_split_node,
                                    max_leaf_nodes,
                                    use_histogram,
                                    var_importance_metric_str,
                                    preferred_ml_framework_str,
                                    output_model_name);
}

template <typename T, typename K>
NEVER_INLINE HOST int32_t
ml_reg_predict_impl(TableFunctionManager& mgr,
                    const std::shared_ptr<AbstractMLModel>& model,
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
  const auto denulled_data = denull_data(input_features);
  const int64_t num_rows = denulled_data.masked_num_rows;
  const bool data_is_masked =
      denulled_data.masked_num_rows < denulled_data.unmasked_num_rows;
  std::vector<T> denulled_output_allocation(data_is_masked ? num_rows : 0);
  mgr.set_output_row_size(input_ids.size());
  T* denulled_output =
      data_is_masked ? denulled_output_allocation.data() : output_predictions.ptr_;
  const auto features_ptrs = pluck_ptrs(denulled_data.data, 0L, input_features.numCols());

  try {
    bool did_execute = false;
    const auto model_type = model->getModelType();
    switch (model_type) {
      case MLModelType::LINEAR_REG: {
        const auto linear_reg_model =
            std::dynamic_pointer_cast<LinearRegressionModel>(model);
        CHECK(linear_reg_model);
#ifdef HAVE_ONEDAL
        if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                             preferred_ml_framework == MLFramework::DEFAULT)) {
          onedal_linear_reg_predict_impl(
              linear_reg_model, features_ptrs, denulled_output, num_rows);
          did_execute = true;
        }
#endif
#ifdef HAVE_MLPACK
        if (!did_execute && (preferred_ml_framework == MLFramework::MLPACK ||
                             preferred_ml_framework == MLFramework::DEFAULT)) {
          mlpack_linear_reg_predict_impl(
              linear_reg_model, features_ptrs, denulled_output, num_rows);
          did_execute = true;
        }
#endif
        break;
      }
      case MLModelType::DECISION_TREE_REG: {
#ifdef HAVE_ONEDAL
        const auto decision_tree_reg_model =
            std::dynamic_pointer_cast<DecisionTreeRegressionModel>(model);
        CHECK(decision_tree_reg_model);
        if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                             preferred_ml_framework == MLFramework::DEFAULT)) {
          onedal_decision_tree_reg_predict_impl(
              decision_tree_reg_model, features_ptrs, denulled_output, num_rows);
          did_execute = true;
        }
#endif
        break;
      }
      case MLModelType::GBT_REG: {
#ifdef HAVE_ONEDAL
        const auto gbt_reg_model = std::dynamic_pointer_cast<GbtRegressionModel>(model);
        CHECK(gbt_reg_model);
        if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                             preferred_ml_framework == MLFramework::DEFAULT)) {
          onedal_gbt_reg_predict_impl(
              gbt_reg_model, features_ptrs, denulled_output, num_rows);
          did_execute = true;
        }
#endif
        break;
      }
      case MLModelType::RANDOM_FOREST_REG: {
#ifdef HAVE_ONEDAL
        const auto random_forest_reg_model =
            std::dynamic_pointer_cast<RandomForestRegressionModel>(model);
        CHECK(random_forest_reg_model);
        if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                             preferred_ml_framework == MLFramework::DEFAULT)) {
          onedal_random_forest_reg_predict_impl(
              random_forest_reg_model, features_ptrs, denulled_output, num_rows);
          did_execute = true;
        }
#endif
        break;
      }
    }
    if (!did_execute) {
      return mgr.ERROR_MESSAGE("Cannot find " + preferred_ml_framework_str.getString() +
                               " ML library to support model implementation.");
    }
  } catch (std::runtime_error& e) {
    const std::string error_str(e.what());
    return mgr.ERROR_MESSAGE(error_str);
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
  UDTF: ml_reg_predict__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<K> id, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int32_t, int64_t, TextEncodingDict], T=[double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
ml_reg_predict__cpu_template(TableFunctionManager& mgr,
                             const TextEncodingNone& model_name,
                             const Column<K>& input_ids,
                             const ColumnList<T>& input_features,
                             const TextEncodingNone& preferred_ml_framework_str,
                             Column<K>& output_ids,
                             Column<T>& output_predictions) {
  try {
    const auto model = ml_models_.getModel(model_name);
    check_model_params(model, 0, input_features.numCols());
    return ml_reg_predict_impl(mgr,
                               model,
                               input_ids,
                               input_features,
                               preferred_ml_framework_str,
                               output_ids,
                               output_predictions);
  } catch (std::runtime_error& e) {
    const std::string error_str(e.what());
    return mgr.ERROR_MESSAGE(error_str);
  }
}

// clang-format off
/*
  UDTF: ml_reg_predict__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<K> id, ColumnList<TextEncodingDict> cat_features, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int32_t, int64_t, TextEncodingDict], T=[double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
ml_reg_predict__cpu_template(TableFunctionManager& mgr,
                             const TextEncodingNone& model_name,
                             const Column<K>& input_ids,
                             const ColumnList<TextEncodingDict>& input_cat_features,
                             const ColumnList<T>& input_numeric_features,
                             const TextEncodingNone& preferred_ml_framework_str,
                             Column<K>& output_ids,
                             Column<T>& output_predictions) {
  try {
    const auto model = ml_models_.getModel(model_name);
    check_model_params(
        model, input_cat_features.numCols(), input_numeric_features.numCols());
    CategoricalFeaturesBuilder<T> cat_features_builder(
        input_cat_features, input_numeric_features, model->getCatFeatureKeys());
    return ml_reg_predict_impl(mgr,
                               model,
                               input_ids,
                               cat_features_builder.getFeatures(),
                               preferred_ml_framework_str,
                               output_ids,
                               output_predictions);
  } catch (std::runtime_error& e) {
    const std::string error_str(e.what());
    return mgr.ERROR_MESSAGE(error_str);
  }
}

// clang-format off
/*
  UDTF: ml_reg_predict__cpu_template(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model,
   Cursor<Column<K> id, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int64_t, TextEncodingDict], T=[double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
ml_reg_predict__cpu_template(TableFunctionManager& mgr,
                             const Column<TextEncodingDict>& model_name,
                             const Column<K>& input_ids,
                             const ColumnList<T>& input_features,
                             const TextEncodingNone& preferred_ml_framework_str,
                             Column<K>& output_ids,
                             Column<T>& output_predictions) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model CURSOR.");
  }
  TextEncodingNone model_name_text_enc_none(model_name.getString(0));
  mgr.addVarlenBuffer(reinterpret_cast<int8_t*>(model_name_text_enc_none.ptr_));
  return ml_reg_predict__cpu_template(mgr,
                                      model_name_text_enc_none,
                                      input_ids,
                                      input_features,
                                      preferred_ml_framework_str,
                                      output_ids,
                                      output_predictions);
}

// clang-format off
/*
  UDTF: ml_reg_predict__cpu_template(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model,
   Cursor<Column<K> id, ColumnList<TextEncodingDict> cat_features, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework | default="DEFAULT") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int32_t, int64_t, TextEncodingDict], T=[double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
ml_reg_predict__cpu_template(TableFunctionManager& mgr,
                             const Column<TextEncodingDict>& model_name,
                             const Column<K>& input_ids,
                             const ColumnList<TextEncodingDict>& input_cat_features,
                             const ColumnList<T>& input_numeric_features,
                             const TextEncodingNone& preferred_ml_framework_str,
                             Column<K>& output_ids,
                             Column<T>& output_predictions) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model CURSOR.");
  }
  const std::string model_name_str{model_name.getString(0)};
  TextEncodingNone model_name_text_enc_none(model_name_str);
  mgr.addVarlenBuffer(reinterpret_cast<int8_t*>(model_name_text_enc_none.ptr_));
  return ml_reg_predict__cpu_template(mgr,
                                      model_name_text_enc_none,
                                      input_ids,
                                      input_cat_features,
                                      input_numeric_features,
                                      preferred_ml_framework_str,
                                      output_ids,
                                      output_predictions);
}

template <typename T>
NEVER_INLINE HOST int32_t r2_score_impl(TableFunctionManager& mgr,
                                        const std::shared_ptr<AbstractMLModel>& model,
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
  TextEncodingNone ml_framework_encoding_none("DEFAULT");

  try {
    auto ret = ml_reg_predict_impl(mgr,
                                   model,
                                   input_ids,
                                   input_features,
                                   ml_framework_encoding_none,
                                   output_ids,
                                   output_predictions);

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
            if (output_predictions[row_idx] != inline_null_value<T>()) {
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
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<T> features> data) ->
   Column<double> r2, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t r2_score__cpu_template(TableFunctionManager& mgr,
                                                 const TextEncodingNone& model_name,
                                                 const Column<T>& input_labels,
                                                 const ColumnList<T>& input_features,
                                                 Column<double>& output_r2) {
  try {
    const auto model = ml_models_.getModel(model_name);
    check_model_params(model, 0, input_features.numCols());
    return r2_score_impl(mgr, model, input_labels, input_features, output_r2);
  } catch (std::runtime_error& e) {
    const std::string error_str(e.what());
    return mgr.ERROR_MESSAGE(error_str);
  }
}

// clang-format off
/*
  UDTF: r2_score__cpu_template(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model,
   Cursor<Column<T> labels, ColumnList<T> features> data) ->
   Column<double> r2, T=[double]
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
  TextEncodingNone model_name_text_enc_none(model_name.getString(0));
  mgr.addVarlenBuffer(reinterpret_cast<int8_t*>(model_name_text_enc_none.ptr_));
  return r2_score__cpu_template(
      mgr, model_name_text_enc_none, input_labels, input_features, output_r2);
}

// clang-format off
/*
  UDTF: r2_score__cpu_template(TableFunctionManager,
   TextEncodingNone model,
   Cursor<Column<T> labels, ColumnList<TextEncodingDict> cat_features, ColumnList<T> numeric_features> data) -> Column<double> r2, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
r2_score__cpu_template(TableFunctionManager& mgr,
                       const TextEncodingNone& model_name,
                       const Column<T>& input_labels,
                       const ColumnList<TextEncodingDict>& input_cat_features,
                       const ColumnList<T>& input_numeric_features,
                       Column<double>& output_r2) {
  try {
    const auto model = ml_models_.getModel(model_name);
    check_model_params(
        model, input_cat_features.numCols(), input_numeric_features.numCols());
    CategoricalFeaturesBuilder<T> cat_features_builder(
        input_cat_features, input_numeric_features, model->getCatFeatureKeys());
    return r2_score_impl(
        mgr, model, input_labels, cat_features_builder.getFeatures(), output_r2);
  } catch (std::runtime_error& e) {
    const std::string error_str(e.what());
    return mgr.ERROR_MESSAGE(error_str);
  }
}

// clang-format off
/*
  UDTF: r2_score__cpu_template(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model,
   Cursor<Column<T> labels, ColumnList<TextEncodingDict> cat_features, ColumnList<T> numeric_features> data) -> Column<double> r2, T=[double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
r2_score__cpu_template(TableFunctionManager& mgr,
                       const Column<TextEncodingDict>& model_name,
                       const Column<T>& input_labels,
                       const ColumnList<TextEncodingDict>& input_cat_features,
                       const ColumnList<T>& input_numeric_features,
                       Column<double>& output_r2) {
  if (model_name.size() != 1) {
    return mgr.ERROR_MESSAGE("Expected only one row in model name CURSOR.");
  }
  const std::string model_name_str{model_name.getString(0)};
  try {
    const auto model = ml_models_.getModel(model_name_str);
    check_model_params(
        model, input_cat_features.numCols(), input_numeric_features.numCols());
    CategoricalFeaturesBuilder<T> cat_features_builder(
        input_cat_features, input_numeric_features, model->getCatFeatureKeys());
    return r2_score_impl(
        mgr, model, input_labels, cat_features_builder.getFeatures(), output_r2);
  } catch (std::runtime_error& e) {
    const std::string error_str(e.what());
    return mgr.ERROR_MESSAGE(error_str);
  }
}

// clang-format off
/*
  UDTF: random_forest_reg_var_importance__cpu_1(TableFunctionManager,
   TextEncodingNone model) ->
   Column<int64_t> feature_id, Column<int64_t> sub_feature_id, Column<TextEncodingDict> sub_feature | input_id=args<>, Column<double> importance_score
 */
// clang-format on

EXTENSION_NOINLINE_HOST int32_t
random_forest_reg_var_importance__cpu_1(TableFunctionManager& mgr,
                                        const TextEncodingNone& model_name,
                                        Column<int64_t>& feature_id,
                                        Column<int64_t>& sub_feature_id,
                                        Column<TextEncodingDict>& sub_feature,
                                        Column<double>& importance_score);

// clang-format off
/*
  UDTF: random_forest_reg_var_importance__cpu_2(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model) ->
   Column<int64_t> feature_id, Column<int64_t> sub_feature_id, Column<TextEncodingDict> sub_feature | input_id=args<>, Column<double> importance_score
 */
// clang-format on

EXTENSION_NOINLINE_HOST int32_t
random_forest_reg_var_importance__cpu_2(TableFunctionManager& mgr,
                                        const Column<TextEncodingDict>& model_name,
                                        Column<int64_t>& feature_id,
                                        Column<int64_t>& sub_feature_id,
                                        Column<TextEncodingDict>& sub_feature,
                                        Column<double>& importance_score);

// clang-format off
/*
  UDTF: get_decision_trees__cpu_1(TableFunctionManager,
   TextEncodingNone model) ->
   Column<int64_t> tree_id,
   Column<int64_t> entry_id,
   Column<bool> is_split_node,
   Column<int64_t> feature_id,
   Column<int64_t> left_child,
   Column<int64_t> right_child,
   Column<double> value
 */
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t get_decision_trees__cpu_1(TableFunctionManager& mgr,
                                  const TextEncodingNone& model_name,
                                  Column<int64_t>& tree_id,
                                  Column<int64_t>& entry_id,
                                  Column<bool>& is_split_node,
                                  Column<int64_t>& feature_id,
                                  Column<int64_t>& left_child,
                                  Column<int64_t>& right_child,
                                  Column<double>& value);

// clang-format off
/*
  UDTF: get_decision_trees__cpu_2(TableFunctionManager,
   Cursor<Column<TextEncodingDict> model_name> model) ->
   Column<int64_t> tree_id,
   Column<int64_t> entry_id,
   Column<bool> is_split_node,
   Column<int64_t> feature_id,
   Column<int64_t> left_child,
   Column<int64_t> right_child,
   Column<double> value
 */
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t get_decision_trees__cpu_2(TableFunctionManager& mgr,
                                  const Column<TextEncodingDict>& model_name,
                                  Column<int64_t>& tree_id,
                                  Column<int64_t>& entry_id,
                                  Column<bool>& is_split_node,
                                  Column<int64_t>& feature_id,
                                  Column<int64_t>& left_child,
                                  Column<int64_t>& right_child,
                                  Column<double>& value);

#endif  // #ifndef __CUDACC__
