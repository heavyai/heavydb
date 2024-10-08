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
#include <sstream>
#include <string>

#include "QueryEngine/CodeCache.h"

extern bool g_is_test_env;

struct CodeCacheMetric {
  int64_t get_count{0};
  int64_t found_count{0};
  int64_t put_count{0};
  int64_t ignore_count{0};
  int64_t overwrite_count{0};
  int64_t evict_count{0};
};

template <typename CompilationContext>
class CodeCacheAccessor {
 public:
  CodeCacheAccessor(EvictionMetricType eviction_metric_type,
                    size_t max_cache_size,
                    std::string name)
      : code_cache_(eviction_metric_type, max_cache_size)
      , eviction_metric_type_(eviction_metric_type)
      , get_count_(0)
      , found_count_(0)
      , put_count_(0)
      , ignore_count_(0)
      , overwrite_count_(0)
      , evict_count_(0)
      , name_(std::move(name)) {
    std::ostringstream oss;
    std::string eviction_type = eviction_metric_type_ == EvictionMetricType::EntryCount
                                    ? "EntryCount"
                                    : "ByteSize";
    oss << "Initialize a code cache (name: " << name_
        << ", eviction_metric_type: " << eviction_type
        << ", max_cache_size: " << max_cache_size << ")";
    LOG(INFO) << oss.str();
  }

  // TODO: replace get_value/put with get_or_wait/reset workflow.
  CodeCacheVal<CompilationContext> get_value(const CodeCacheKey& key);
  bool put(const CodeCacheKey& key, CodeCacheVal<CompilationContext>& value);

  // get_or_wait and reset/erase should be used in pair.
  CodeCacheVal<CompilationContext>* get_or_wait(const CodeCacheKey& key);
  void reset(const CodeCacheKey& key, CodeCacheVal<CompilationContext> value);
  void erase(const CodeCacheKey& key);

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

  friend std::ostream& operator<<(std::ostream& os, CodeCacheAccessor& c) {
    std::lock_guard<std::mutex> lock(c.code_cache_mutex_);
    os << "CodeCacheAccessor<" << c.name_ << ">[current size=" << c.code_cache_.size()
       << ", total get/found count=" << c.get_count_ << "/" << c.found_count_
       << ", total put/ignore/overwrite count=" << c.put_count_ << "/" << c.ignore_count_
       << "/" << c.overwrite_count_ << ", total evict count=" << c.evict_count_ << "]";
    return os;
  }

  size_t getCacheSize() {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    return code_cache_.size();
  }

  void resetCache(size_t new_max_size) {
    CHECK(g_is_test_env) << "Call the resetCache function from non-test env.";
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    code_cache_ = CodeCache<CompilationContext>(eviction_metric_type_, new_max_size);
  }

  CodeCacheMetric getCodeCacheMetric() {
    std::lock_guard<std::mutex> lock(code_cache_mutex_);
    return {get_count_,
            found_count_,
            put_count_,
            ignore_count_,
            overwrite_count_,
            evict_count_};
  }

 private:
  CodeCache<CompilationContext> code_cache_;
  EvictionMetricType const eviction_metric_type_;
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
