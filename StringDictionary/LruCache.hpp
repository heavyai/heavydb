/*
 * File:   lrucache.hpp
 * Author: Alexander Ponomarev
 *
 * Created on June 20, 2013, 5:09 PM
 */

#ifndef STRINGDICTIONARY_LRUCACHE_HPP
#define STRINGDICTIONARY_LRUCACHE_HPP

#include <cstddef>
#include <list>
#include <unordered_map>

template <typename key_t, typename value_t, class hash_t = std::hash<key_t>>
class LruCache {
 private:
  using key_value_pair_t = typename std::pair<key_t, value_t>;
  using cache_list_t = typename std::list<key_value_pair_t>;
  using list_iterator_t = typename cache_list_t::iterator;
  using map_t = typename std::unordered_map<key_t, list_iterator_t, hash_t>;
  using map_t_iterator = typename map_t::iterator;

 public:
  LruCache(const size_t max_size) : max_size_(max_size) {}

  void put(const key_t& key, value_t&& value) {
    auto it = cache_items_map_.find(key);
    cache_items_list_.emplace_front(key, std::forward<value_t&&>(value));
    putCommon(it, key);
  }

  void put(const key_t& key, const value_t& value) {
    auto it = cache_items_map_.find(key);
    cache_items_list_.emplace_front(key, std::forward<const value_t&&>(value));
    putCommon(it, key);
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

  const_list_iterator_t cend() const { return (cache_items_list_.cend()); }

  void clear() {
    cache_items_list_.clear();
    cache_items_map_.clear();
  }

  void evictFractionEntries(const float fraction) {
    size_t entries_to_evict =
        std::min(std::max(static_cast<size_t>(cache_items_map_.size() * fraction),
                          static_cast<size_t>(1)),
                 cache_items_map_.size());
    evictCommon(entries_to_evict);
  }

  void evictNEntries(const size_t n) {
    size_t entries_to_evict = std::min(n, cache_items_map_.size());
    evictCommon(entries_to_evict);
  }

  size_t size() const { return cache_items_list_.size(); }

 private:
  void putCommon(map_t_iterator& it, key_t const& key) {
    if (it != cache_items_map_.end()) {
      cache_items_list_.erase(it->second);
      cache_items_map_.erase(it);
    }
    cache_items_map_[key] = cache_items_list_.begin();

    if (cache_items_map_.size() > max_size_) {
      auto last = cache_items_list_.end();
      last--;
      cache_items_map_.erase(last->first);
      cache_items_list_.pop_back();
    }
  }

  void evictCommon(const size_t entries_to_evict) {
    auto last = cache_items_list_.end();
    size_t entries_erased = 0;
    while (entries_erased < entries_to_evict && last != cache_items_list_.begin()) {
      last--;
      cache_items_map_.erase(last->first);
      last = cache_items_list_.erase(last);
      entries_erased++;
    }
  }

  cache_list_t cache_items_list_;
  map_t cache_items_map_;
  size_t max_size_;
};

#endif  // STRINGDICTIONARY_LRUCACHE_HPP
