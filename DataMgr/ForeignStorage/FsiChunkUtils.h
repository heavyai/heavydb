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

#pragma once

#include <map>

#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ChunkMetadata.h"
#include "Shared/distributed.h"

namespace foreign_storage {
void init_chunk_for_column(
    const ChunkKey& chunk_key,
    const std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& chunk_metadata_map,
    const std::map<ChunkKey, AbstractBuffer*>& buffers,
    Chunk_NS::Chunk& chunk);

// Construct default metadata for given column descriptor with num_elements
std::shared_ptr<ChunkMetadata> get_placeholder_metadata(const SQLTypeInfo& type,
                                                        size_t num_elements);
/*
  Splits up a set of items to be processed into multiple partitions, with the intention
  that each thread will process a separate part.
 */
// TODO(Misiu): Change this to return a list of Views/Ranges when we support c++20.
template <typename T>
auto partition_for_threads(const std::set<T>& items, size_t max_threads) {
  const size_t items_per_thread = (items.size() + (max_threads - 1)) / max_threads;
  std::list<std::set<T>> items_by_thread;
  auto i = 0U;
  for (auto item : items) {
    if (i++ % items_per_thread == 0) {
      items_by_thread.emplace_back(std::set<T>{});
    }
    items_by_thread.back().emplace(item);
  }
  return items_by_thread;
}

/*
  Splits up a vector of items to be processed into multiple partitions, with the intention
  that each thread will process a separate part.
 */
// TODO: refactor partition_for_threads to use Requires when we support c++20
template <typename T>
auto partition_for_threads(const std::vector<T>& items, size_t max_threads) {
  const size_t items_per_thread = (items.size() + (max_threads - 1)) / max_threads;
  std::list<std::vector<T>> items_by_thread;
  auto i = 0U;
  for (auto item : items) {
    if (i++ % items_per_thread == 0) {
      items_by_thread.emplace_back(std::vector<T>{});
    }
    items_by_thread.back().emplace_back(item);
  }
  return items_by_thread;
}

template <typename Container>
std::vector<std::future<void>> create_futures_for_workers(
    const Container& items,
    size_t max_threads,
    std::function<void(const Container&)> lambda) {
  auto items_per_thread = partition_for_threads(items, max_threads);
  std::vector<std::future<void>> futures;
  for (const auto& items : items_per_thread) {
    futures.emplace_back(std::async(std::launch::async, lambda, items));
  }

  return futures;
}

const foreign_storage::ForeignTable& get_foreign_table_for_key(const ChunkKey& key);

bool is_system_table_chunk_key(const ChunkKey& chunk_key);

bool is_replicated_table_chunk_key(const ChunkKey& chunk_key);

bool is_append_table_chunk_key(const ChunkKey& chunk_key);

bool is_shardable_key(const ChunkKey& key);

bool fragment_maps_to_leaf(const ChunkKey& key);

bool key_does_not_shard_to_leaf(const ChunkKey& key);
}  // namespace foreign_storage
