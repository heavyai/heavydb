/*
 * Copyright 2021 OmniSci, Inc.
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

#ifndef __CUDACC__

#include <tbb/parallel_for.h>
#include <cstring>  // std::memcpy
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "Utilities.hpp"

#define NANOSECONDS_PER_SECOND 1000000000

template <typename T>
TEMPLATE_NOINLINE std::pair<T, T> get_column_min_max(const Column<T>& col) {
  T col_min = std::numeric_limits<T>::max();
  T col_max = std::numeric_limits<T>::lowest();
  const int64_t num_rows = col.size();
  for (int64_t r = 0; r < num_rows; ++r) {
    if (col[r] < col_min) {
      col_min = col[r];
    }
    if (col[r] > col_max) {
      col_max = col[r];
    }
  }
  return std::make_pair(col_min, col_max);
}

template <typename T1, typename T2>
TEMPLATE_NOINLINE T1
distance_in_meters(const T1 fromlon, const T1 fromlat, const T2 tolon, const T2 tolat) {
  T1 latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
  T1 longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
  T1 latitudeH = sin(latitudeArc * 0.5);
  latitudeH *= latitudeH;
  T1 lontitudeH = sin(longitudeArc * 0.5);
  lontitudeH *= lontitudeH;
  T1 tmp = cos(fromlat * 0.017453292519943295769236907684886) *
           cos(tolat * 0.017453292519943295769236907684886);
  return 6372797.560856 * (2.0 * asin(sqrt(latitudeH + tmp * lontitudeH)));
}

bool DataBufferCache::isKeyCached(const std::string& key) const {
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  return data_cache_.count(key) > 0;
}

bool DataBufferCache::isKeyCachedAndSameLength(const std::string& key,
                                               const size_t num_bytes) const {
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  const auto& cached_data_itr = data_cache_.find(key);
  if (cached_data_itr == data_cache_.end()) {
    return false;
  }
  return num_bytes == cached_data_itr->second->num_bytes;
}

template <typename T>
void DataBufferCache::getDataForKey(const std::string& key, T* dest_buffer) const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  const auto& cached_data_itr = data_cache_.find(key);
  if (cached_data_itr == data_cache_.end()) {
    const std::string error_msg = "Data for key " + key + " not found in cache.";
    throw std::runtime_error(error_msg);
  }
  copyData(reinterpret_cast<int8_t*>(dest_buffer),
           cached_data_itr->second->data_buffer,
           cached_data_itr->second->num_bytes);
}

template <typename T>
const T& DataBufferCache::getDataRefForKey(const std::string& key) const {
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  const auto& cached_data_itr = data_cache_.find(key);
  if (cached_data_itr == data_cache_.end()) {
    const std::string error_msg{"Data for key " + key + " not found in cache."};
    throw std::runtime_error(error_msg);
  }
  return *reinterpret_cast<const T*>(cached_data_itr->second->data_buffer);
}

template <typename T>
const T* DataBufferCache::getDataPtrForKey(const std::string& key) const {
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  const auto& cached_data_itr = data_cache_.find(key);
  if (cached_data_itr == data_cache_.end()) {
    return nullptr;
  }
  return reinterpret_cast<const T* const>(cached_data_itr->second->data_buffer);
}

template <typename T>
void DataBufferCache::putDataForKey(const std::string& key,
                                    T* const data_buffer,
                                    const size_t num_elements) {
  auto timer = DEBUG_TIMER(__func__);
  const size_t num_bytes(num_elements * sizeof(T));
  auto cache_data = std::make_shared<CacheData>(num_bytes);
  copyData(cache_data->data_buffer, reinterpret_cast<int8_t*>(data_buffer), num_bytes);
  std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);
  const auto& cached_data_itr = data_cache_.find(key);
  if (data_cache_.find(key) != data_cache_.end()) {
    const std::string warning_msg =
        "Data for key " + key + " already exists in cache. Replacing.";
    std::cout << warning_msg << std::endl;
    cached_data_itr->second.reset();
    cached_data_itr->second = cache_data;
    return;
  }
  data_cache_.insert(std::make_pair(key, cache_data));
}

void DataBufferCache::copyData(int8_t* dest,
                               const int8_t* source,
                               const size_t num_bytes) const {
  if (num_bytes < parallel_copy_min_bytes) {
    std::memcpy(dest, source, num_bytes);
    return;
  }
  const size_t max_bytes_per_thread = parallel_copy_min_bytes;
  const size_t num_threads =
      (num_bytes + max_bytes_per_thread - 1) / max_bytes_per_thread;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_threads, 1),
      [&](const tbb::blocked_range<size_t>& r) {
        const size_t end_chunk_idx = r.end();
        for (size_t chunk_idx = r.begin(); chunk_idx != end_chunk_idx; ++chunk_idx) {
          const size_t start_byte = chunk_idx * max_bytes_per_thread;
          const size_t length =
              std::min(start_byte + max_bytes_per_thread, num_bytes) - start_byte;
          std::memcpy(dest + start_byte, source + start_byte, length);
        }
      });
}

/* Definitions for DataCache */

template <typename T>
bool DataCache<T>::isKeyCached(const std::string& key) const {
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  return data_cache_.count(key) > 0;
}

template <typename T>
std::shared_ptr<T> DataCache<T>::getDataForKey(const std::string& key) const {
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  const auto& cached_data_itr = data_cache_.find(key);
  if (cached_data_itr == data_cache_.end()) {
    const std::string error_msg{"Data for key " + key + " not found in cache."};
    throw std::runtime_error(error_msg);
  }
  return cached_data_itr->second;
}

template <typename T>
void DataCache<T>::putDataForKey(const std::string& key, std::shared_ptr<T> const data) {
  std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);
  const auto& cached_data_itr = data_cache_.find(key);
  if (cached_data_itr != data_cache_.end()) {
    const std::string warning_msg =
        "Data for key " + key + " already exists in cache. Replacing.";
    std::cout << warning_msg << std::endl;
    cached_data_itr->second.reset();
    cached_data_itr->second = data;
  }
  data_cache_.insert(std::make_pair(key, data));
}

namespace FileUtilities {

std::vector<std::filesystem::path> get_fs_paths(const std::string& file_or_directory) {
  const std::filesystem::path file_or_directory_path(file_or_directory);
  const auto file_status = std::filesystem::status(file_or_directory_path);

  std::vector<std::filesystem::path> fs_paths;
  if (std::filesystem::is_regular_file(file_status)) {
    fs_paths.emplace_back(file_or_directory_path);
    return fs_paths;
  } else if (std::filesystem::is_directory(file_status)) {
    for (std::filesystem::directory_entry const& entry :
         std::filesystem::directory_iterator(file_or_directory_path)) {
      if (std::filesystem::is_regular_file(std::filesystem::status(entry))) {
        fs_paths.emplace_back(entry.path());
      }
    }
    return fs_paths;
  }
  return fs_paths;
}

}  // namespace FileUtilities

#endif  // __CUDACC__