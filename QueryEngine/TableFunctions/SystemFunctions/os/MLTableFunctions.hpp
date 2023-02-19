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

#ifdef HAVE_ONEDAL
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/OneDalFunctions.hpp"
#endif

#ifdef HAVE_MLPACK
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLPackFunctions.hpp"
#endif

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
   int32_t num_iterations | require="num_iterations > 0",
   TextEncodingNone init_type,
   TextEncodingNone preferred_ml_framework) ->
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
  UDTF: kmeans__cpu_template(TableFunctionManager,
   Cursor<Column<K> input_ids, ColumnList<T> input_features> data,
   int32_t num_clusters | require="num_clusters > 0" | require="num_clusters <= input_ids.size()",
   int32_t num_iterations | require="num_iterations > 0",
   TextEncodingNone init_type) ->
   Column<K> id | input_id=args<0>,
   Column<int32_t> cluster_id,
   K=[int64_t, TextEncodingDict], T=[float, double]
*/
// clang-format on

template <typename K, typename T>
NEVER_INLINE HOST int32_t kmeans__cpu_template(TableFunctionManager& mgr,
                                               const Column<K>& input_ids,
                                               const ColumnList<T>& input_features,
                                               const int num_clusters,
                                               const int num_iterations,
                                               const TextEncodingNone& init_type_str,
                                               Column<K>& output_ids,
                                               Column<int32_t>& output_clusters) {
  std::string preferred_ml_framework{"DEFAULT"};
  return kmeans__cpu_template(mgr,
                              input_ids,
                              input_features,
                              num_clusters,
                              num_iterations,
                              init_type_str,
                              preferred_ml_framework,
                              output_ids,
                              output_clusters);
}

// clang-format off
/*
  UDTF: kmeans__cpu_template(TableFunctionManager,
   Cursor<Column<K> input_ids, ColumnList<T> input_features> data,
   int32_t num_clusters | require="num_clusters > 0" | require="num_clusters <= input_ids.size()",
   int32_t num_iterations | require="num_iterations > 0") ->
   Column<K> id | input_id=args<0>,
   Column<int32_t> cluster_id,
   K=[int64_t, TextEncodingDict], T=[float, double]
*/
// clang-format on

template <typename K, typename T>
NEVER_INLINE HOST int32_t kmeans__cpu_template(TableFunctionManager& mgr,
                                               const Column<K>& input_ids,
                                               const ColumnList<T>& input_features,
                                               const int32_t num_clusters,
                                               const int32_t num_iterations,
                                               Column<K>& output_ids,
                                               Column<int32_t>& output_clusters) {
  std::string kmeans_init_strategy{"DEFAULT"};
  std::string preferred_ml_framework{"DEFAULT"};
  return kmeans__cpu_template(mgr,
                              input_ids,
                              input_features,
                              num_clusters,
                              num_iterations,
                              kmeans_init_strategy,
                              preferred_ml_framework,
                              output_ids,
                              output_clusters);
}

// clang-format off
/*
  UDTF: dbscan__cpu_template(TableFunctionManager,
   Cursor<Column<K> input_ids, ColumnList<T> input_features> data,
   double epsilon | require="epsilon > 0.0",
   int32_t min_observations | require="min_observations > 0",
   TextEncodingNone preferred_ml_framework) ->
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
  UDTF: dbscan__cpu_template(TableFunctionManager,
   Cursor<Column<K> input_ids, ColumnList<T> input_features> data,
   double epsilon | require="epsilon > 0.0",
   int32_t min_observations | require="min_observations > 0") ->
   Column<K> id | input_id=args<0>, Column<int32_t> cluster_id,
   K=[int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename K, typename T>
NEVER_INLINE HOST int32_t dbscan__cpu_template(TableFunctionManager& mgr,
                                               const Column<K>& input_ids,
                                               const ColumnList<T>& input_features,
                                               const double epsilon,
                                               const int32_t min_observations,
                                               Column<K>& output_ids,
                                               Column<int32_t>& output_clusters) {
  std::string preferred_ml_framework{"DEFAULT"};
  return dbscan__cpu_template(mgr,
                              input_ids,
                              input_features,
                              epsilon,
                              min_observations,
                              preferred_ml_framework,
                              output_ids,
                              output_clusters);
}

// clang-format off
/*
  UDTF: linear_reg_fit__cpu_template(TableFunctionManager,
   Cursor<Column<T> labels, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework) ->
   Column<int32_t> coef_idx, Column<T> coef, T=[float, double]
 */
// clang-format on

template <typename T>
NEVER_INLINE HOST int32_t
linear_reg_fit__cpu_template(TableFunctionManager& mgr,
                             const Column<T>& input_labels,
                             const ColumnList<T>& input_features,
                             const TextEncodingNone& preferred_ml_framework_str,
                             Column<int32_t>& output_coef_idxs,
                             Column<T>& output_coefs) {
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
  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_linear_reg_fit_impl(labels_ptrs[0],
                                 features_ptrs,
                                 output_coef_idxs.ptr_,
                                 output_coefs.ptr_,
                                 denulled_data.masked_num_rows);
      did_execute = true;
    }
#endif
#ifdef HAVE_MLPACK
    if (!did_execute && (preferred_ml_framework == MLFramework::MLPACK ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      mlpack_linear_reg_fit_impl(labels_ptrs[0],
                                 features_ptrs,
                                 output_coef_idxs.ptr_,
                                 output_coefs.ptr_,
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
  return num_coefs;
}

// clang-format off
/*
  UDTF: linear_reg_fit__cpu_template(TableFunctionManager,
   Cursor<Column<T> labels, ColumnList<T> features> data) ->
   Column<int32_t> coef_idx, Column<T> coef, T=[float, double]
 */
// clang-format on
template <typename T>
NEVER_INLINE HOST int32_t
linear_reg_fit__cpu_template(TableFunctionManager& mgr,
                             const Column<T>& input_labels,
                             const ColumnList<T>& input_features,
                             Column<int32_t>& output_coef_idxs,
                             Column<T>& output_coefs) {
  std::string preferred_ml_framework{"DEFAULT"};
  return linear_reg_fit__cpu_template(mgr,
                                      input_labels,
                                      input_features,
                                      preferred_ml_framework,
                                      output_coef_idxs,
                                      output_coefs);
}

template <typename T>
std::vector<T> sort_coefs(const Column<int32_t>& coef_idxs, const Column<T>& coefs) {
  const size_t num_coefs = coef_idxs.size();
  std::vector<T> ordered_coefs(num_coefs);
  for (size_t coef_idx = 0; coef_idx < num_coefs; ++coef_idx) {
    ordered_coefs[coef_idxs[coef_idx]] = coefs[coef_idx];
  }
  return ordered_coefs;
}

// clang-format off
/*
  UDTF: linear_reg_predict__cpu_template(TableFunctionManager,
   Cursor<Column<K> id, ColumnList<T> features> data,
   Cursor<Column<int32_t> coef_idx, Column<T> coef> params | require="coef_idx.size() == features.numCols() + 1",
   TextEncodingNone preferred_ml_framework) ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
linear_reg_predict__cpu_template(TableFunctionManager& mgr,
                                 const Column<K>& input_ids,
                                 const ColumnList<T>& input_features,
                                 const Column<int32_t>& coef_idxs,
                                 const Column<T>& coefs,
                                 const TextEncodingNone& preferred_ml_framework_str,
                                 Column<K>& output_ids,
                                 Column<T>& output_predictions) {
  const auto preferred_ml_framework = get_ml_framework(preferred_ml_framework_str);
  if (preferred_ml_framework == MLFramework::INVALID) {
    return mgr.ERROR_MESSAGE("Invalid ML Framework: " +
                             preferred_ml_framework_str.getString());
  }

  mgr.set_output_row_size(input_ids.size());
  const auto denulled_data = denull_data(input_features);
  const int64_t num_rows = denulled_data.masked_num_rows;
  const bool data_is_masked =
      denulled_data.masked_num_rows < denulled_data.unmasked_num_rows;
  std::vector<T> denulled_output_allocation(data_is_masked ? num_rows : 0);
  T* denulled_output =
      data_is_masked ? denulled_output_allocation.data() : output_predictions.ptr_;

  const auto features_ptrs = pluck_ptrs(denulled_data.data, 0L, input_features.numCols());

  const auto ordered_coefs = sort_coefs(coef_idxs, coefs);

  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_linear_reg_predict_impl(
          features_ptrs, denulled_output, num_rows, ordered_coefs.data());
      did_execute = true;
    }
#endif
#ifdef HAVE_MLPACK
    if (!did_execute && (preferred_ml_framework == MLFramework::MLPACK ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      mlpack_linear_reg_predict_impl(
          features_ptrs, denulled_output, num_rows, ordered_coefs.data());
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
  UDTF: linear_reg_predict__cpu_template(TableFunctionManager,
   Cursor<Column<K> id, ColumnList<T> features> data,
   Cursor<Column<int32_t> coef_idx, Column<T> coef> params | require="coef_idx.size() == features.numCols() + 1") ->
   Column<K> id | input_id=args<0>, Column<T> prediction,
   K=[int64_t, TextEncodingDict], T=[float, double]
 */
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t
linear_reg_predict__cpu_template(TableFunctionManager& mgr,
                                 const Column<K>& input_ids,
                                 const ColumnList<T>& input_features,
                                 const Column<int32_t>& coef_idxs,
                                 const Column<T>& coefs,
                                 Column<K>& output_ids,
                                 Column<T>& output_predictions) {
  std::string preferred_ml_framework{"DEFAULT"};
  return linear_reg_predict__cpu_template(mgr,
                                          input_ids,
                                          input_features,
                                          coef_idxs,
                                          coefs,
                                          preferred_ml_framework,
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
  UDTF: linear_reg_fit_predict__cpu_template(TableFunctionManager,
   Cursor<Column<K> id, Column<T> labels, ColumnList<T> features> data,
   TextEncodingNone preferred_ml_framework) ->
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
  const int64_t num_coefs = input_features.numCols() + 1;
  // Need to create backing vectors for coef column wrappers
  std::vector<int32_t> coef_idxs_vec(num_coefs);
  std::vector<T> coefs_vec(num_coefs);
  auto coef_idxs = create_wrapper_col(coef_idxs_vec);
  auto coefs = create_wrapper_col(coefs_vec);
  // Disable output allocations as we are not calling the fit function
  // through the normal table functions path, and we have already
  // allocated our coef storage with the vectors above.
  mgr.disable_output_allocations();
  const auto fit_ret = linear_reg_fit__cpu_template(
      mgr, input_labels, input_features, preferred_ml_framework_str, coef_idxs, coefs);
  mgr.enable_output_allocations();
  if (fit_ret < 0) {
    return fit_ret;
  }
  return linear_reg_predict__cpu_template(mgr,
                                          input_ids,
                                          input_features,
                                          coef_idxs,
                                          coefs,
                                          preferred_ml_framework_str,
                                          output_ids,
                                          output_predictions);
}

// clang-format off
/*
  UDTF: linear_reg_fit_predict__cpu_template(TableFunctionManager,
   Cursor<Column<K> id, Column<T> labels, ColumnList<T> features> data) ->
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
                                     Column<K>& output_ids,
                                     Column<T>& output_predictions) {
  std::string preferred_ml_framework{"DEFAULT"};
  return linear_reg_fit_predict__cpu_template(mgr,
                                              input_ids,
                                              input_labels,
                                              input_features,
                                              preferred_ml_framework,
                                              output_ids,
                                              output_predictions);
}

#endif  // #ifndef __CUDACC__
