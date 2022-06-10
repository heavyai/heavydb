/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "QueryEngine/heavydbTypes.h"

template <typename T>
NEVER_INLINE HOST std::pair<T, T> get_column_min_max(const Column<T>& col);

NEVER_INLINE HOST std::pair<int32_t, int32_t> get_column_min_max(
    const Column<TextEncodingDict>& col);

template <typename T>
NEVER_INLINE HOST double get_column_mean(const T* data, const int64_t num_rows);

template <typename T>
NEVER_INLINE HOST double get_column_mean(const Column<T>& col);

template <typename T>
NEVER_INLINE HOST double get_column_std_dev(const Column<T>& col, const double mean);

template <typename T>
NEVER_INLINE HOST double get_column_std_dev(const T* data,
                                            const int64_t num_rows,
                                            const double mean);

// Assumes nulls have been removed
template <typename T>
void z_std_normalize_col(const T* input_data,
                         T* output_data,
                         const int64_t num_rows,
                         const double mean,
                         const double std_dev);

// Assumes nulls have been removed
template <typename T>
std::vector<std::vector<T>> z_std_normalize_data(const std::vector<T*>& input_data,
                                                 const int64_t num_rows);

template <typename T>
NEVER_INLINE HOST std::tuple<T, T, bool> get_column_metadata(const Column<T>& col);

NEVER_INLINE HOST std::tuple<int32_t, int32_t, bool> get_column_metadata(
    const Column<TextEncodingDict>& col);

template <typename T1, typename T2>
NEVER_INLINE HOST T1
distance_in_meters(const T1 fromlon, const T1 fromlat, const T2 tolon, const T2 tolat);

inline int64_t x_y_bin_to_bin_index(const int64_t x_bin,
                                    const int64_t y_bin,
                                    const int64_t num_x_bins) {
  return y_bin * num_x_bins + x_bin;
}

inline std::pair<int64_t, int64_t> bin_to_x_y_bin_indexes(const int64_t bin,
                                                          const int64_t num_x_bins) {
  return std::make_pair(bin % num_x_bins, bin / num_x_bins);
}

namespace FileUtilities {
std::vector<std::filesystem::path> get_fs_paths(const std::string& file_or_directory);
}

enum BoundsType { Min, Max };

enum IntervalType { Inclusive, Exclusive };

template <typename T>
NEVER_INLINE HOST bool is_valid_tf_input(const T input,
                                         const T bounds_val,
                                         const BoundsType bounds_type,
                                         const IntervalType interval_type);

#endif  //__CUDACC__