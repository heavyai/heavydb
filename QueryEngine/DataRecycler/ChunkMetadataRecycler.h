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

#include "DataMgr/ChunkMetadata.h"
#include "DataRecycler.h"

constexpr DeviceIdentifier CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER =
    DataRecyclerUtil::CPU_DEVICE_IDENTIFIER;

struct ChunkMetadataMetaInfo {
  ChunkMetadataMetaInfo(const std::unordered_set<size_t> input_table_infos) {
    input_table_keys.insert(input_table_infos.begin(), input_table_infos.end());
  };

  std::unordered_set<size_t> input_table_keys;
};

class ChunkMetadataRecycler
    : public DataRecycler<std::optional<ChunkMetadataMap>, ChunkMetadataMetaInfo> {
 public:
  // resultset's chunk metadata recycler caches logical information instead of actual data
  // so we do not limit its capacity
  // thus we do not maintain a metric cache for this
  // also, we do not classify device identifier since it is logical information
  ChunkMetadataRecycler()
      : DataRecycler({CacheItemType::CHUNK_METADATA},
                     std::numeric_limits<size_t>::max(),
                     std::numeric_limits<size_t>::max(),
                     0) {}

  std::optional<ChunkMetadataMap> getItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::optional<ChunkMetadataMetaInfo> meta_info = std::nullopt) override;

  void putItemToCache(
      QueryPlanHash key,
      std::optional<ChunkMetadataMap> item,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      size_t item_size,
      size_t compute_time,
      std::optional<ChunkMetadataMetaInfo> meta_info = std::nullopt) override;

  // nothing to do with hashing scheme recycler
  void initCache() override {}

  void clearCache() override;

  void markCachedItemAsDirty(size_t table_key,
                             std::unordered_set<QueryPlanHash>& key_set,
                             CacheItemType item_type,
                             DeviceIdentifier device_identifier) override;

  std::string toString() const override;

  void addQueryPlanDagForTableKeys(size_t hashed_query_plan_dag,
                                   const std::unordered_set<size_t>& table_keys,
                                   std::lock_guard<std::mutex>& lock);

  std::optional<std::unordered_set<size_t>> getMappedQueryPlanDagsWithTableKey(
      size_t table_key) const;

  void removeTableKeyInfoFromQueryPlanDagMap(size_t table_key);

 private:
  bool hasItemInCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<ChunkMetadataMetaInfo> meta_info = std::nullopt) const override;

  // hashing scheme recycler clears the cached layouts at once
  void removeItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<ChunkMetadataMetaInfo> meta_info = std::nullopt) override;

  // hashing scheme recycler has unlimited capacity so we do not need this
  void cleanupCacheForInsertion(
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      size_t required_size,
      std::lock_guard<std::mutex>& lock,
      std::optional<ChunkMetadataMetaInfo> meta_info = std::nullopt) override {}

  // keep all table keys referenced to compute cached resultset
  std::unordered_map<size_t, std::unordered_set<size_t>> table_key_to_query_plan_dag_map_;
};
