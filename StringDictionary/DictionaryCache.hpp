/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef DICTIONARY_CACHE_HPP
#define DICTIONARY_CACHE_HPP

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>

template <typename key_t, typename value_t>
class DictionaryCache {
 public:
  DictionaryCache() {}

  void put(const key_t& key, const std::shared_ptr<value_t> value) {
    auto it = cache_items.find(key);
    if (it != cache_items.end()) {
      cache_items.erase(it);
    }
    cache_items.insert({key, value});
  }

  value_t* get(const key_t& key) {
    auto it = cache_items.find(key);
    if (it == cache_items.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  void remove(const key_t& key) { cache_items.erase(key); }

  bool is_empty() { return cache_items.empty(); }

  void invalidateInvertedIndex() noexcept {
    if (!cache_items.empty())
      cache_items.clear();
  }

 private:
  std::unordered_map<key_t, std::shared_ptr<value_t>> cache_items;
};

#endif  // DICTIONARY_CACHE_HPP
