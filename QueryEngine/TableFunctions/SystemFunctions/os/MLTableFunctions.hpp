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
                                      Column<bool>& output_default) {
  const std::vector<std::string> ml_frameworks = {"onedal", "mlpack"};
  const int32_t num_frameworks = ml_frameworks.size();
  mgr.set_output_row_size(num_frameworks);
  const std::vector<int32_t> ml_framework_string_ids =
      output_ml_frameworks.string_dict_proxy_->getOrAddTransientBulk(ml_frameworks);

  bool found_available_framework = false;

#if defined(HAVE_ONEDAL) || defined(HAVE_MLPACK)
  auto framework_found_actions = [&output_availability,
                                  &output_default,
                                  &found_available_framework](const int64_t out_row_idx) {
    output_availability[out_row_idx] = true;
    if (!found_available_framework) {
      output_default[out_row_idx] = true;
      found_available_framework = true;
    } else {
      output_default[out_row_idx] = false;
    }
  };
#endif

#if !defined(HAVE_ONEDAL) || !defined(HAVE_MLPACK)
  auto framework_not_found_actions = [&output_availability,
                                      &output_default](const int64_t out_row_idx) {
    output_availability[out_row_idx] = false;
    output_default[out_row_idx] = false;
  };
#endif

  for (int32_t out_row_idx = 0; out_row_idx < num_frameworks; ++out_row_idx) {
    output_ml_frameworks[out_row_idx] = ml_framework_string_ids[out_row_idx];
    if (ml_frameworks[out_row_idx] == "onedal") {
#ifdef HAVE_ONEDAL
      framework_found_actions(out_row_idx);
#else
      framework_not_found_actions(out_row_idx);
#endif
    } else if (ml_frameworks[out_row_idx] == "mlpack") {
#ifdef HAVE_MLPACK
      framework_found_actions(out_row_idx);
#else
      framework_not_found_actions(out_row_idx);
#endif
    }
  }
  return num_frameworks;
}

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
   K=[int32_t, int64_t, TextEncodingDict], T=[float, double]
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

  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_kmeans_impl(normalized_data,
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
      mlpack_kmeans_impl(normalized_data,
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
   K=[int32_t, int64_t, TextEncodingDict], T=[float, double]
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
   K=[int32_t, int64_t, TextEncodingDict], T=[float, double]
*/
// clang-format on

template <typename K, typename T>
NEVER_INLINE HOST int32_t kmeans__cpu_template(TableFunctionManager& mgr,
                                               const Column<K>& input_ids,
                                               const ColumnList<T>& input_features,
                                               const int num_clusters,
                                               const int num_iterations,
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
   K=[int32_t, int64_t, TextEncodingDict], T=[float, double]
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

  try {
    bool did_execute = false;
#ifdef HAVE_ONEDAL
    if (!did_execute && (preferred_ml_framework == MLFramework::ONEDAL ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      onedal_dbscan_impl(
          normalized_data, denulled_output, num_rows, epsilon, min_observations);
      did_execute = true;
    }
#endif
#ifdef HAVE_MLPACK
    if (!did_execute && (preferred_ml_framework == MLFramework::MLPACK ||
                         preferred_ml_framework == MLFramework::DEFAULT)) {
      mlpack_dbscan_impl(
          normalized_data, denulled_output, num_rows, epsilon, min_observations);
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
   K=[int32_t, int64_t, TextEncodingDict], T=[float, double]
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

#endif  // #ifndef __CUDACC__