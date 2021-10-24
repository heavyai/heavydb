/*
 * Copyright 2021 OmnSci, Inc.
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

#ifdef HAVE_SYSTEM_TFS
#ifndef __CUDACC__

#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "QueryEngine/OmniSciTypes.h"

template <typename T>
TEMPLATE_NOINLINE std::pair<T, T> get_column_min_max(const Column<T>& col);

TEMPLATE_NOINLINE std::pair<int32_t, int32_t> get_column_min_max(
    const Column<TextEncodingDict>& col);

template <typename T1, typename T2>
TEMPLATE_NOINLINE T1
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

struct CacheData {
  int8_t* data_buffer;
  size_t num_bytes;

  CacheData(const size_t num_bytes) : num_bytes(num_bytes) {
    data_buffer = new int8_t[num_bytes];
  }

  ~CacheData() { delete[] data_buffer; }
};

class DataBufferCache {
 public:
  bool isKeyCached(const std::string& key) const;

  bool isKeyCachedAndSameLength(const std::string& key, const size_t num_bytes) const;

  // Assumes dest_buffer is already appropriately sized
  template <typename T>
  void getDataForKey(const std::string& key, T* dest_buffer) const;

  template <typename T>
  const T& getDataRefForKey(const std::string& key) const;

  template <typename T>
  const T* getDataPtrForKey(const std::string& key) const;

  template <typename T>
  void putDataForKey(const std::string& key,
                     T* const data_buffer,
                     const size_t num_elements);

 private:
  const size_t parallel_copy_min_bytes{1 << 20};

  void copyData(int8_t* dest, const int8_t* source, const size_t num_bytes) const;

  std::unordered_map<std::string, std::shared_ptr<CacheData>> data_cache_;
  mutable std::shared_mutex cache_mutex_;
};

template <typename T>
class DataCache {
 public:
  bool isKeyCached(const std::string& key) const;

  std::shared_ptr<T> getDataForKey(const std::string& key) const;

  void putDataForKey(const std::string& key, std::shared_ptr<T> const data);

 private:
  std::unordered_map<std::string, std::shared_ptr<T>> data_cache_;
  mutable std::shared_mutex cache_mutex_;
};

namespace FileUtilities {
std::vector<std::filesystem::path> get_fs_paths(const std::string& file_or_directory);
}

enum BoundsType { Min, Max };

enum IntervalType { Inclusive, Exclusive };

template <typename T>
bool is_valid_tf_input(const T input,
                       const T bounds_val,
                       const BoundsType bounds_type,
                       const IntervalType interval_type);

#include "Utilities.cpp"

#endif  //__CUDACC__
#endif  // HAVE_SYSTEM_TFS
