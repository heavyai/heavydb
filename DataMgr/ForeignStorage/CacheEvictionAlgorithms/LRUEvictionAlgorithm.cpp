/*
 * Copyright 2020 OmniSci, Inc.
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

/*
  TODO(Misiu): This algorithm can be replaced with the LruCache implemented in
  the StringDictionary.  However, that implementation is missing functionality
  such as the ability to remove arbitrary chunks, (which we need to selectively
  clear the cache).  This functionality could be added to the StringDict version
  but for now we're replicating algorithm here.
*/

#include "LRUEvictionAlgorithm.h"

const ChunkKey LRUEvictionAlgorithm::evictNextChunk() {
  if (cache_items_list_.size() < 1)
    throw NoEntryFoundException();
  auto last = cache_items_list_.end();
  last--;
  cache_items_map_.erase(*last);
  const ChunkKey ret = cache_items_list_.back();
  cache_items_list_.pop_back();
  return ret;
}

void LRUEvictionAlgorithm::touchChunk(const ChunkKey& key) {
  auto it = cache_items_map_.find(key);
  cache_items_list_.emplace_front(key);
  if (it != cache_items_map_.end()) {
    cache_items_list_.erase(it->second);
    cache_items_map_.erase(it);
  }
  cache_items_map_[key] = cache_items_list_.begin();
}

void LRUEvictionAlgorithm::removeChunk(const ChunkKey& key) {
  auto it = cache_items_map_.find(key);
  if (it == cache_items_map_.end())
    return;
  cache_items_list_.erase(it->second);
  cache_items_map_.erase(key);
}

std::string LRUEvictionAlgorithm::dumpEvictionQueue() {
  std::string ret = "Eviction queue:\n{";
  for (auto chunk : cache_items_list_)
    ret += show_chunk(chunk) + ", ";
  ret += "}\n";
  return ret;
}
