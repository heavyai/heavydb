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

#include "QueryEngine/CodeCacheAccessor.h"
#include "QueryEngine/CompilationContext.h"

template <typename CompilationContext>
CodeCacheVal<CompilationContext> CodeCacheAccessor<CompilationContext>::get_value(
    const CodeCacheKey& key) {
  std::lock_guard<std::mutex> lock(code_cache_mutex_);
  get_count_++;
  auto it = code_cache_.find(key);
  if (it != code_cache_.cend()) {
    found_count_++;
    return it->second;
  }
  return {};
}

template <typename CompilationContext>
void CodeCacheAccessor<CompilationContext>::put(const CodeCacheKey& key,
                                                CodeCacheVal<CompilationContext>& value) {
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

template <typename CompilationContext>
CodeCacheVal<CompilationContext>* CodeCacheAccessor<CompilationContext>::get_or_wait(
    const CodeCacheKey& key) {
  CodeCacheVal<CompilationContext>* cached_code = nullptr;
  std::unique_lock<std::mutex> lk(code_cache_mutex_);
  get_count_++;
  cached_code = code_cache_.get(key);
  if (cached_code) {
    if (cached_code->get()) {
      found_count_++;
      return cached_code;
    } else {
      // Wait until compiling thread inserts code to cache. TODO: this
      // wait also locks other unrelated get_or_wait calls on
      // different keys. This is suboptimal as it will block in
      // addition to others threads get_or_wait(key) calls, also
      // independent get_or_wait(other_key) calls. To fix this, we'll
      // need a key specific mutex or use some other approach that
      // would allow threads with other keys to proceed.
      compilation_cv_.wait(lk, [&] { return !!(code_cache_.get(key)->get()); });
      cached_code = code_cache_.get(key);
      CHECK(cached_code->get());
      found_count_++;
      return cached_code;
    }
  }
  // This is the first time the key is used to acquire code from
  // cache. Insert null value to cache so that other threads acquiring
  // the same key will wait (see above) until the code is inserted to
  // cache (by put method below):
  CodeCacheVal<CompilationContext> not_a_code(nullptr);
  code_cache_.put(key, std::move(not_a_code));
  return nullptr;  // caller must trigger code compilation for the given key
}

template <typename CompilationContext>
void CodeCacheAccessor<CompilationContext>::put(
    const CodeCacheKey& key,
    CodeCacheVal<CompilationContext>&& value) {
  std::lock_guard<std::mutex> lock(code_cache_mutex_);
  auto result = code_cache_.get(key);
  CHECK(result);          // get_or_wait has inserted not_a_code to code cache
  CHECK(!result->get());  // ensure that result really contains not_a_code per get_or_wait
  result->swap(value);    // swap not_a_code with actual code
  compilation_cv_.notify_all();  // notify waiting get_or_wait(..) calls
}

template <typename CompilationContext>
void CodeCacheAccessor<CompilationContext>::clear() {
  std::lock_guard<std::mutex> lock(code_cache_mutex_);
  code_cache_.clear();
}

template class CodeCacheAccessor<CompilationContext>;
template class CodeCacheAccessor<CpuCompilationContext>;
template class CodeCacheAccessor<GpuCompilationContext>;
