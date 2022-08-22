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

#include <cstring>  // std::memcpy
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

struct CacheDataTf {
  int8_t* data_buffer;
  size_t num_bytes;

  CacheDataTf(const size_t num_bytes) : num_bytes(num_bytes) {
    data_buffer = new int8_t[num_bytes];
  }

  ~CacheDataTf() { delete[] data_buffer; }
};

class DataBufferCache {
 public:
  bool isKeyCached(const std::string& key) const {
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    return data_cache_.count(key) > 0;
  }

  bool isKeyCachedAndSameLength(const std::string& key, const size_t num_bytes) const {
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    const auto& cached_data_itr = data_cache_.find(key);
    if (cached_data_itr == data_cache_.end()) {
      return false;
    }
    return num_bytes == cached_data_itr->second->num_bytes;
  }

  template <typename T>
  void getDataForKey(const std::string& key, T* dest_buffer) const {
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
  const T& getDataRefForKey(const std::string& key) const {
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    const auto& cached_data_itr = data_cache_.find(key);
    if (cached_data_itr == data_cache_.end()) {
      const std::string error_msg{"Data for key " + key + " not found in cache."};
      throw std::runtime_error(error_msg);
    }
    return *reinterpret_cast<const T*>(cached_data_itr->second->data_buffer);
  }

  template <typename T>
  const T* getDataPtrForKey(const std::string& key) const {
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    const auto& cached_data_itr = data_cache_.find(key);
    if (cached_data_itr == data_cache_.end()) {
      return nullptr;
    }
    return reinterpret_cast<const T* const>(cached_data_itr->second->data_buffer);
  }

  template <typename T>
  void putDataForKey(const std::string& key,
                     T* const data_buffer,
                     const size_t num_elements) {
    auto timer = DEBUG_TIMER(__func__);
    const size_t num_bytes(num_elements * sizeof(T));
    auto cache_data = std::make_shared<CacheDataTf>(num_bytes);
    copyData(cache_data->data_buffer, reinterpret_cast<int8_t*>(data_buffer), num_bytes);
    std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);
    const auto& cached_data_itr = data_cache_.find(key);
    if (data_cache_.find(key) != data_cache_.end()) {
      if constexpr (debug_print_) {
        const std::string warning_msg =
            "Data for key " + key + " already exists in cache. Replacing.";
        std::cout << warning_msg << std::endl;
      }
      cached_data_itr->second.reset();
      cached_data_itr->second = cache_data;
      return;
    }
    data_cache_.insert(std::make_pair(key, cache_data));
  }

 private:
  const size_t parallel_copy_min_bytes{1 << 20};

  void copyData(int8_t* dest, const int8_t* source, const size_t num_bytes) const {
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

  std::unordered_map<std::string, std::shared_ptr<CacheDataTf>> data_cache_;
  mutable std::shared_mutex cache_mutex_;
  static constexpr bool debug_print_{false};
};

template <typename T>
class DataCache {
 public:
  bool isKeyCached(const std::string& key) const {
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    return data_cache_.count(key) > 0;
  }

  std::shared_ptr<T> getDataForKey(const std::string& key) const {
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    const auto& cached_data_itr = data_cache_.find(key);
    if (cached_data_itr == data_cache_.end()) {
      const std::string error_msg{"Data for key " + key + " not found in cache."};
      throw std::runtime_error(error_msg);
    }
    return cached_data_itr->second;
  }

  void putDataForKey(const std::string& key, std::shared_ptr<T> const data) {
    std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);
    const auto& cached_data_itr = data_cache_.find(key);
    if (cached_data_itr != data_cache_.end()) {
      if constexpr (debug_print_) {
        const std::string warning_msg =
            "Data for key " + key + " already exists in cache. Replacing.";
        std::cout << warning_msg << std::endl;
      }
      cached_data_itr->second.reset();
      cached_data_itr->second = data;
    }
    data_cache_.insert(std::make_pair(key, data));
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<T>> data_cache_;
  mutable std::shared_mutex cache_mutex_;
  static constexpr bool debug_print_{false};
};

#endif
