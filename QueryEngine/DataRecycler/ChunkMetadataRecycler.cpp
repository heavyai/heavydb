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

#include "ChunkMetadataRecycler.h"

std::optional<ChunkMetadataMap> ChunkMetadataRecycler::getItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::optional<ChunkMetadataMetaInfo> meta_info) {
  if (!g_enable_data_recycler || !g_use_chunk_metadata_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return std::nullopt;
  }
  CHECK_EQ(item_type, CacheItemType::CHUNK_METADATA);
  CHECK_EQ(device_identifier, CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto chunk_metadata_cache = getCachedItemContainer(item_type, device_identifier);
  CHECK(chunk_metadata_cache);
  auto candidate_it =
      std::find_if(chunk_metadata_cache->begin(),
                   chunk_metadata_cache->end(),
                   [&key](const auto& cached_item) { return cached_item.key == key; });
  if (candidate_it != chunk_metadata_cache->end()) {
    if (candidate_it->isDirty()) {
      removeItemFromCache(
          key, item_type, device_identifier, lock, candidate_it->meta_info);
      return std::nullopt;
    }
    auto candidate_chunk_metadata = candidate_it->cached_item;
    candidate_it->item_metric->incRefCount();
    VLOG(1) << "[" << item_type << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Get cached item from cache (key: " << key << ")";
    return candidate_chunk_metadata;
  }
  return std::nullopt;
}

void ChunkMetadataRecycler::putItemToCache(
    QueryPlanHash key,
    std::optional<ChunkMetadataMap> item,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    size_t item_size,
    size_t compute_time,
    std::optional<ChunkMetadataMetaInfo> meta_info) {
  if (!g_enable_data_recycler || !g_use_chunk_metadata_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  CHECK(meta_info.has_value());
  CHECK_EQ(item_type, CacheItemType::CHUNK_METADATA);
  CHECK_EQ(device_identifier, CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto chunk_metadata_cache = getCachedItemContainer(item_type, device_identifier);
  CHECK(chunk_metadata_cache);
  auto candidate_it =
      std::find_if(chunk_metadata_cache->begin(),
                   chunk_metadata_cache->end(),
                   [&key](const auto& cached_item) { return cached_item.key == key; });
  bool has_cached_chunk_metadata = false;
  if (candidate_it != chunk_metadata_cache->end()) {
    has_cached_chunk_metadata = true;
    CHECK(candidate_it->meta_info);
    if (candidate_it->isDirty()) {
      removeItemFromCache(
          key, item_type, device_identifier, lock, candidate_it->meta_info);
      has_cached_chunk_metadata = false;
    }
  }

  if (!has_cached_chunk_metadata) {
    auto& metric_tracker = getMetricTracker(item_type);
    auto new_cache_metric_ptr = metric_tracker.putNewCacheItemMetric(
        key, device_identifier, item_size, compute_time);
    CHECK_EQ(item_size, new_cache_metric_ptr->getMemSize());
    VLOG(1) << "[" << item_type << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Put item to cache (key: " << key << ")";
    chunk_metadata_cache->emplace_back(key, item, new_cache_metric_ptr, meta_info);
    if (!meta_info->input_table_keys.empty()) {
      addQueryPlanDagForTableKeys(key, meta_info->input_table_keys, lock);
    }
  }
  return;
}

void ChunkMetadataRecycler::removeItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<ChunkMetadataMetaInfo> meta_info) {
  CHECK_EQ(item_type, CacheItemType::CHUNK_METADATA);
  CHECK_EQ(device_identifier, CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER);
  auto metadata_cache = getCachedItemContainer(item_type, device_identifier);
  auto filter = [key](auto const& item) { return item.key == key; };
  auto itr = std::find_if(metadata_cache->cbegin(), metadata_cache->cend(), filter);
  if (itr != metadata_cache->cend()) {
    VLOG(1) << "[" << item_type << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] remove cached item from cache (key: " << key << ")";
    metadata_cache->erase(itr);
  }
}

void ChunkMetadataRecycler::clearCache() {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto metadata_cache_container = getCachedItemContainer(
      CacheItemType::CHUNK_METADATA, CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER);
  VLOG(1) << "[" << CacheItemType::CHUNK_METADATA << ", "
          << DataRecyclerUtil::getDeviceIdentifierString(
                 CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER)
          << "] clear cache (# items: " << metadata_cache_container->size() << ")";
  metadata_cache_container->clear();
}

void ChunkMetadataRecycler::markCachedItemAsDirty(
    size_t table_key,
    std::unordered_set<QueryPlanHash>& key_set,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  if (!g_enable_data_recycler || !g_use_chunk_metadata_cache || key_set.empty()) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::CHUNK_METADATA);
  CHECK_EQ(device_identifier, CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto metadata_cache = getCachedItemContainer(item_type, device_identifier);
  for (auto key : key_set) {
    markCachedItemAsDirtyImpl(key, *metadata_cache);
  }
}

std::string ChunkMetadataRecycler::toString() const {
  auto chunk_metadata_cache_container = getCachedItemContainer(
      CacheItemType::CHUNK_METADATA, CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER);
  std::ostringstream oss;
  oss << "Chunk metadata cache:\n";
  oss << "Device: "
      << DataRecyclerUtil::getDeviceIdentifierString(
             CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER)
      << "\n # cached chunk metadata: " << chunk_metadata_cache_container->size();
  return oss.str();
}

bool ChunkMetadataRecycler::hasItemInCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<ChunkMetadataMetaInfo> meta_info) const {
  if (!g_enable_data_recycler || !g_use_chunk_metadata_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return false;
  }
  CHECK_EQ(item_type, CacheItemType::CHUNK_METADATA);
  CHECK_EQ(device_identifier, CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER);
  auto metadata_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_it = std::find_if(
      metadata_cache->begin(), metadata_cache->end(), [&key](const auto& cached_item) {
        return cached_item.key == key;
      });
  return candidate_it != metadata_cache->end();
}

void ChunkMetadataRecycler::addQueryPlanDagForTableKeys(
    size_t hashed_query_plan_dag,
    const std::unordered_set<size_t>& table_keys,
    std::lock_guard<std::mutex>& lock) {
  for (auto table_key : table_keys) {
    auto itr = table_key_to_query_plan_dag_map_.try_emplace(table_key).first;
    itr->second.insert(hashed_query_plan_dag);
  }
}

std::optional<std::unordered_set<size_t>>
ChunkMetadataRecycler::getMappedQueryPlanDagsWithTableKey(size_t table_key) const {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto it = table_key_to_query_plan_dag_map_.find(table_key);
  return it != table_key_to_query_plan_dag_map_.end() ? std::make_optional(it->second)
                                                      : std::nullopt;
}

void ChunkMetadataRecycler::removeTableKeyInfoFromQueryPlanDagMap(size_t table_key) {
  table_key_to_query_plan_dag_map_.erase(table_key);
}
