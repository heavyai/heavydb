/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

  std::shared_ptr<value_t> get(const key_t& key) {
    auto it = cache_items.find(key);
    if (it == cache_items.end()) {
      return nullptr;
    }
    return it->second;
  }

  void remove(const key_t& key) { cache_items.erase(key); }

  bool is_empty() { return cache_items.empty(); }

  void invalidateInvertedIndex() noexcept {
    if (!cache_items.empty()) {
      cache_items.clear();
    }
  }

 private:
  std::unordered_map<key_t, std::shared_ptr<value_t>> cache_items;
};

#endif  // DICTIONARY_CACHE_HPP
