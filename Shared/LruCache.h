/*
 * File:   lrucache.hpp
 * Author: Alexander Ponomarev
 *
 * Created on June 20, 2013, 5:09 PM
 */

#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>

template <class T>
struct is_shared_ptr : std::false_type {};

template <class T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

enum class EvictionMetricType { EntryCount, ByteSize };

template <typename key_t, typename value_t, class hash_t = std::hash<key_t>>
class LruCache {
 private:
  using key_value_pair_t = typename std::pair<key_t, value_t>;
  using cache_list_t = typename std::list<key_value_pair_t>;
  using list_iterator_t = typename cache_list_t::iterator;
  using map_t = typename std::unordered_map<key_t, list_iterator_t, hash_t>;
  using map_t_iterator = typename map_t::iterator;

 public:
  LruCache(EvictionMetricType eviction_metric_type, const size_t max_size)
      : eviction_metric_type_(eviction_metric_type)
      , max_size_(max_size)
      , total_byte_size_(0) {}

  size_t put(const key_t& key, value_t&& value) {
    total_byte_size_ += getValueSize(value);
    auto it = cache_items_map_.find(key);
    cache_items_list_.emplace_front(key, std::forward<value_t&&>(value));
    return putCommon(it, key);
  }

  size_t put(const key_t& key, const value_t& value) {
    total_byte_size_ += getValueSize(value);
    auto it = cache_items_map_.find(key);
    cache_items_list_.emplace_front(key, std::forward<const value_t&&>(value));
    return putCommon(it, key);
  }

  void erase(const key_t& key) {
    auto it = cache_items_map_.find(key);
    if (it != cache_items_map_.end()) {
      total_byte_size_ -= getValueSize(it->second);
      cache_items_list_.erase(it->second);
      cache_items_map_.erase(it);
    }
  }

  value_t* get(const key_t& key) {
    auto it = cache_items_map_.find(key);
    if (it == cache_items_map_.end()) {
      return nullptr;
    }
    cache_items_list_.splice(cache_items_list_.begin(), cache_items_list_, it->second);
    return &it->second->second;
  }
  using const_list_iterator_t = typename cache_list_t::const_iterator;

  const_list_iterator_t find(const key_t& key) const {
    auto it = cache_items_map_.find(key);
    if (it == cache_items_map_.end()) {
      return cend();
    } else {
      return it->second;
    }
  }

  const_list_iterator_t cbegin() const { return (cache_items_list_.cbegin()); }

  const_list_iterator_t cend() const { return (cache_items_list_.cend()); }

  void clear() {
    cache_items_list_.clear();
    cache_items_map_.clear();
    total_byte_size_ = 0;
  }

  size_t computeNumEntriesToEvict(const float fraction) {
    return std::min(
        std::min(std::max(static_cast<size_t>(cache_items_map_.size() * fraction),
                          static_cast<size_t>(1)),
                 cache_items_map_.size()),
        cache_items_map_.size());
  }

  size_t evictNEntries(const size_t n) { return evictCommon(n); }

  size_t size() const {
    return eviction_metric_type_ == EvictionMetricType::EntryCount
               ? cache_items_list_.size()
               : total_byte_size_;
  }

 private:
  size_t putCommon(map_t_iterator& it, key_t const& key) {
    size_t entries_erased = 0;
    if (it != cache_items_map_.end()) {
      total_byte_size_ -= getValueSize(it->second);
      cache_items_list_.erase(it->second);
      cache_items_map_.erase(it);
      entries_erased++;
    }
    cache_items_map_[key] = cache_items_list_.begin();

    while ((eviction_metric_type_ == EvictionMetricType::EntryCount &&
            cache_items_map_.size() > max_size_) ||
           (eviction_metric_type_ == EvictionMetricType::ByteSize &&
            total_byte_size_ > max_size_)) {
      auto last = cache_items_list_.end();
      last--;
      auto target_it = cache_items_map_.find(last->first);
      total_byte_size_ -= getValueSize(target_it->second);
      cache_items_map_.erase(target_it);
      cache_items_list_.pop_back();
      entries_erased++;
    }
    return entries_erased;
  }

  size_t evictCommon(const size_t entries_to_evict) {
    auto last = cache_items_list_.end();
    size_t entries_erased = 0;
    while (entries_erased < entries_to_evict && last != cache_items_list_.begin()) {
      last--;
      total_byte_size_ -= getValueSize(last->second);
      cache_items_map_.erase(last->first);
      last = cache_items_list_.erase(last);
      entries_erased++;
    }
    return entries_erased;
  }

  size_t getValueSize(const value_t& value) {
    if constexpr (std::is_pointer_v<value_t> || is_shared_ptr<value_t>::value) {
      // 'value == nullptr' represents a call from the `get_or_wait` function
      return value ? value->size() : 0;
    } else {
      return value.size();
    }
  }

  size_t getValueSize(const list_iterator_t& it) { return getValueSize(it->second); }

  cache_list_t cache_items_list_;
  map_t cache_items_map_;
  EvictionMetricType eviction_metric_type_;
  size_t max_size_;
  size_t total_byte_size_;
};
