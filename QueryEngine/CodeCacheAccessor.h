/*
 * Copyright 2022 OmniSci, Inc.
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

#ifndef QUERYENGINE_CODECACHEACCESSOR_HPP
#define QUERYENGINE_CODECACHEACCESSOR_HPP

#include <mutex>
#include <string>
#include "QueryEngine/CodeCache.h"

template <typename CompilationContext>
class CodeCacheAccessor {
 public:
  CodeCacheAccessor(size_t cache_size, std::string name = "")
      : code_cache_(cache_size)
      , get_count_(0)
      , found_count_(0)
      , put_count_(0)
      , ignore_count_(0)
      , overwrite_count_(0)
      , evict_count_(0)
      , name_(std::move(name)) {}
  CodeCacheVal<CompilationContext> get_value(const CodeCacheKey& key) {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    get_count_++;
    auto it = code_cache_.find(key);
    if (it != code_cache_.cend()) {
      found_count_++;
      return it->second;
    }
    return {};
  }
  CodeCacheVal<CompilationContext>* get(const CodeCacheKey& key) {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    get_count_++;
    auto result = code_cache_.get(key);
    if (result) {
      found_count_++;
    }
    return result;
  }
  void put(const CodeCacheKey& key, CodeCacheVal<CompilationContext>& value) {
    bool warn = false;
    {
      std::lock_guard<std::mutex> lock(code_cache_mutex_);
      // if key is in cache, put is no-op
      auto it = code_cache_.find(key);
      put_count_++;
      if (it == code_cache_.cend()) {
        code_cache_.put(key, value);
      } else {
        ignore_count_++;
        warn = true;
      }
    }
    if (warn) {
      LOG(WARNING) << *this << ": code already in cache, ignoring.\n";
    }
  }
  void put(const CodeCacheKey& key, CodeCacheVal<CompilationContext>&& value) {
    bool warn = false;
    {
      std::lock_guard<std::mutex> lock(code_cache_mutex_);
      // if key is in cache, put overrides cached item
      auto it = code_cache_.find(key);
      put_count_++;
      if (it != code_cache_.cend()) {
        overwrite_count_++;
        warn = true;
      }
      code_cache_.put(key, value);
    }
    if (warn) {
      LOG(WARNING) << *this << ": code already in cache, overwriting.\n";
    }
  }
  void clear() {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    code_cache_.clear();
  }
  void evictFractionEntries(const float fraction) {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    evict_count_++;
    code_cache_.evictFractionEntries(fraction);
  }
  friend std::ostream& operator<<(std::ostream& os, CodeCacheAccessor& c) {
    std::lock_guard<std::mutex> lock(c.code_cache_mutex_);
    os << "CodeCacheAccessor<" << c.name_ << ">[current size=" << c.code_cache_.size()
       << ", total get/found count=" << c.get_count_ << "/" << c.found_count_
       << ", total put/ignore/overwrite count=" << c.put_count_ << "/" << c.ignore_count_
       << "/" << c.overwrite_count_ << ", total evict count=" << c.evict_count_ << "]";
    return os;
  }

 private:
  CodeCache<CompilationContext> code_cache_;
  int64_t get_count_, found_count_, put_count_, ignore_count_, overwrite_count_,
      evict_count_;
  const std::string name_;
  std::mutex code_cache_mutex_;
};

#endif
