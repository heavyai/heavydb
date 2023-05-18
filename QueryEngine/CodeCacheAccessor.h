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

  // TODO: replace get_value/put with get_or_wait/swap workflow.
  CodeCacheVal<CompilationContext> get_value(const CodeCacheKey& key);
  void put(const CodeCacheKey& key, CodeCacheVal<CompilationContext>& value);

  // get_or_wait and swap should be used in pair.
  CodeCacheVal<CompilationContext>* get_or_wait(const CodeCacheKey& key);
  void swap(const CodeCacheKey& key, CodeCacheVal<CompilationContext>&& value);
  void clear();

  size_t computeNumEntriesToEvict(const float fraction) {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    return code_cache_.computeNumEntriesToEvict(fraction);
  }

  void evictEntries(const size_t n) {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    evict_count_++;
    code_cache_.evictNEntries(n);
  }

  size_t getSumSizeEvicted(const size_t n) {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    auto last = code_cache_.cend();
    size_t visited = 0;
    size_t ret = 0;
    while (visited < n && last != code_cache_.cbegin()) {
      last--;
      ret += last->second->getMemSize();
      visited++;
    }
    return ret;
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
  // cumulative statistics of code cache usage
  int64_t get_count_, found_count_, put_count_, ignore_count_, overwrite_count_,
      evict_count_;
  // name of the code cache
  const std::string name_;
  // used to lock any access to the code cache:
  std::mutex code_cache_mutex_;
  // cv is used to wait until another thread finishes compilation and
  // inserts a code to cache
  std::condition_variable compilation_cv_;
};

#endif
