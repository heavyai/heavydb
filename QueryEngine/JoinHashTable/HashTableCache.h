/*
 * Copyright 2020 OmniSci, Inc.
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

#include <functional>
#include <mutex>
#include <vector>

#include "Logger/Logger.h"

template <class K, class V>
class HashTableCache {
 public:
  HashTableCache() {}

  std::function<void()> getCacheInvalidator() {
    return [this]() -> void {
      std::lock_guard<std::mutex> guard(mutex_);
      VLOG(1) << "Invalidating " << contents_.size() << " cached hash tables.";
      contents_.clear();
    };
  }

  V getCachedHashTable(const size_t idx) {
    std::lock_guard<std::mutex> guard(mutex_);
    CHECK_LT(idx, contents_.size());
    return contents_.at(idx).second;
  }

  size_t getNumberOfCachedHashTables() {
    std::lock_guard<std::mutex> guard(mutex_);
    return contents_.size();
  }

  void clear() {
    std::lock_guard<std::mutex> guard(mutex_);
    contents_.clear();
  }

  void insert(const K& key, V& hash_table) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto& kv : contents_) {
      if (kv.first == key) {
        auto& cached_hash_table = kv.second;
        cached_hash_table = hash_table;
        return;
      }
    }
    contents_.emplace_back(key, hash_table);
  }

  // makes a copy
  V get(const K& key) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& kv : contents_) {
      if (kv.first == key) {
        return kv.second;
      }
    }
    return nullptr;
  }

 private:
  std::vector<std::pair<K, V>> contents_;
  std::mutex mutex_;
};
