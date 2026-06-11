/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
  TODO(Misiu): This algorithm can be replaced with the LruCache implemented in
  the StringDictionary.  However, that implementation is missing functionality
  such as the ability to remove arbitrary chunks, (which we need to selectively
  clear the cache).  This functionality could be added to the StringDict version
  but for now we're replicating algorithm here.
*/

#include "LRUEvictionAlgorithm.h"

const ChunkKey LRUEvictionAlgorithm::evictNextChunk() {
  heavyai::unique_lock<heavyai::shared_mutex> lock(cache_mutex_);
  if (cache_items_list_.size() < 1) {
    throw NoEntryFoundException();
  }
  auto last = cache_items_list_.back();
  CHECK(cache_items_map_.erase(last) > 0) << "Chunk not evicted!";
  cache_items_list_.pop_back();
  return last;
}

void LRUEvictionAlgorithm::touchChunk(const ChunkKey& key) {
  heavyai::unique_lock<heavyai::shared_mutex> lock(cache_mutex_);
  auto it = cache_items_map_.find(key);
  if (it != cache_items_map_.end()) {
    cache_items_list_.erase(it->second);
    cache_items_map_.erase(it);
  }
  cache_items_list_.emplace_front(key);
  cache_items_map_[key] = cache_items_list_.begin();
}

void LRUEvictionAlgorithm::removeChunk(const ChunkKey& key) {
  heavyai::unique_lock<heavyai::shared_mutex> lock(cache_mutex_);
  auto it = cache_items_map_.find(key);
  if (it == cache_items_map_.end()) {
    return;
  }
  cache_items_list_.erase(it->second);
  cache_items_map_.erase(key);
}

std::string LRUEvictionAlgorithm::dumpEvictionQueue() {
  heavyai::shared_lock<heavyai::shared_mutex> lock(cache_mutex_);
  std::string ret = "Eviction queue:\n{";
  for (auto chunk : cache_items_list_) {
    ret += show_chunk(chunk) + ", ";
  }
  ret += "}\n";
  return ret;
}
