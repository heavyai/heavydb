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
  typedef typename std::pair<key_t, value_t> key_value_pair_t;
  typedef typename std::list<key_value_pair_t> cache_list_t;
  typedef typename cache_list_t::iterator list_iterator_t;
  typedef typename std::unordered_map<key_t, list_iterator_t, hash_t> map_t;
  typedef typename map_t::iterator map_t_iterator;

 public:
  LruCache(const size_t max_size) : max_size_(max_size) {}

  void put(key_t const& key, value_t&& value) {
    auto it = cache_items_map_.find(key);
    cache_items_list_.emplace_front(key, std::forward<value_t&&>(value));
    putCommon(it, key);
  }

  void put(const key_t& key, const value_t& value) {
    auto it = cache_items_map_.find(key);
    cache_items_list_.emplace_front(key, std::forward<const value_t&&>(value));
    putCommon(it, key);
  }

  const value_t* get(const key_t& key) {
    auto it = cache_items_map_.find(key);
    if (it == cache_items_map_.end()) {
      return nullptr;
    }
    cache_items_list_.splice(cache_items_list_.begin(), cache_items_list_, it->second);
    return &it->second->second;
  }
  typedef typename cache_list_t::const_iterator const_list_iterator_t;

  const_list_iterator_t find(const key_t& key) const {
    auto it = cache_items_map_.find(key);
    auto val = (it == cache_items_map_.end() ? cend() : it->second);
    return (val);
  }

  const_list_iterator_t cend() const { return (cache_items_list_.cend()); }

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

  cache_list_t cache_items_list_;
  map_t cache_items_map_;
  size_t max_size_;
};

#endif  // STRINGDICTIONARY_LRUCACHE_HPP
