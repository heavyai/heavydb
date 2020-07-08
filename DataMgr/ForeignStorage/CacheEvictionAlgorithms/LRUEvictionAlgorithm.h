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

/**
 * @file	LRUEvictionAlgorithm.h
 * @author	Misiu Godfrey <misiu.godfrey@omnisci.com>
 *
 * This file includes the class specification for the Least Recently Used cache eviction
 * algorithm used by the Foreign Storage Interface (FSI).
 *
 * // TODO(Misiu): A lot of the code here is replicated from StringDictionary/LruCache.hpp
 * with some minor extensions for deletion and changed to use a set.  It should be merged.
 *
 * This algorithm tracks which chunks were the least recently used by relying on the
 * touch_chunk function being called when they are used.  It tracks the order of use
 * in a simple queue.
 */

#pragma once

#include <cstddef>
#include <list>
#include "CacheEvictionAlgorithm.h"

class LRUEvictionAlgorithm : public CacheEvictionAlgorithm {
 public:
  ~LRUEvictionAlgorithm() override {}
  // Returns the next chunk to evict.
  const ChunkKey evictNextChunk() override;
  // Update the algorithm knowing that this chunk was recently touched by the system.
  void touchChunk(const ChunkKey&) override;
  // Removes a chunk from the eviction queue if present.
  void removeChunk(const ChunkKey&) override;
  // Used for debugging.
  std::string dumpEvictionQueue();

 private:
  std::list<ChunkKey> cache_items_list_;
  std::map<const ChunkKey, std::list<ChunkKey>::iterator> cache_items_map_;
};
