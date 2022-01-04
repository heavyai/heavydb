/*
 * Copyright 2021 OmniSci, Inc.
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

#include "HashTablePropertyRecycler.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"

std::optional<HashTableProperty> HashTablePropertyRecycler::getItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::optional<EMPTY_META_INFO> meta_info) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return std::nullopt;
  }
  CHECK_EQ(item_type, CacheItemType::HT_PROPERTY);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto property_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_layout = getCachedItemWithoutConsideringMetaInfo(
      key, item_type, device_identifier, *property_cache, lock);
  if (candidate_layout) {
    CHECK(!candidate_layout->isDirty());
    VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Recycle hashtable property in cache "
            << getHashtablePropertyString(*candidate_layout->cached_item);
    return candidate_layout->cached_item;
  }
  return std::nullopt;
}

void HashTablePropertyRecycler::putItemToCache(QueryPlanHash key,
                                               std::optional<HashTableProperty> item,
                                               CacheItemType item_type,
                                               DeviceIdentifier device_identifier,
                                               size_t item_size,
                                               size_t compute_time,
                                               std::optional<EMPTY_META_INFO> meta_info) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::HT_PROPERTY);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto property_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_layout = getCachedItemWithoutConsideringMetaInfo(
      key, item_type, device_identifier, *property_cache, lock);
  if (candidate_layout) {
    return;
  }
  property_cache->emplace_back(key, item, nullptr, meta_info);
  VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
          << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
          << "] Put hashtable property to cache " << getHashtablePropertyString(*item);
  return;
}

void HashTablePropertyRecycler::updateItemInCacheIfNecessary(
    QueryPlanHash key,
    std::optional<HashTableProperty> item,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::HT_PROPERTY);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto property_cache = getCachedItemContainer(item_type, device_identifier);
  auto filter = [key](auto const& item) { return item.key == key; };
  auto itr = std::find_if(property_cache->cbegin(), property_cache->cend(), filter);
  std::string update_msg;
  auto prop_str = getHashtablePropertyString(*item);
  if (itr != property_cache->cend()) {
    update_msg = "Update hashtable layout from " +
                 getHashtablePropertyString(*itr->cached_item) + " to " + prop_str;
    property_cache->erase(itr);
  } else {
    update_msg = "Put hashtable property " + prop_str;
  }
  property_cache->emplace_back(key, item, nullptr, std::nullopt);
  VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
          << DataRecyclerUtil::getDeviceIdentifierString(device_identifier) << "] "
          << update_msg;
}

void HashTablePropertyRecycler::removeItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<EMPTY_META_INFO> meta_info) {
  auto property_cache = getCachedItemContainer(item_type, device_identifier);
  auto filter = [key](auto const& item) { return item.key == key; };
  auto itr = std::find_if(property_cache->cbegin(), property_cache->cend(), filter);
  if (itr == property_cache->cend()) {
    return;
  } else {
    property_cache->erase(itr);
  }
}

void HashTablePropertyRecycler::clearCache() {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto property_cache_container = getCachedItemContainer(
      CacheItemType::HT_PROPERTY, PROPERTY_CACHE_DEVICE_IDENTIFIER);
  property_cache_container->clear();
}

void HashTablePropertyRecycler::markCachedItemAsDirty(
    size_t table_key,
    std::unordered_set<QueryPlanHash>& key_set,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache || key_set.empty()) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::HT_PROPERTY);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto property_cache = getCachedItemContainer(item_type, device_identifier);
  for (auto key : key_set) {
    markCachedItemAsDirtyImpl(key, *property_cache);
  }
  removeTableKeyInfoFromQueryPlanDagMap(table_key);
}

std::string HashTablePropertyRecycler::toString() const {
  auto property_cache_container = getCachedItemContainer(
      CacheItemType::HT_PROPERTY, PROPERTY_CACHE_DEVICE_IDENTIFIER);
  std::ostringstream oss;
  oss << "Hashing scheme cache:\n";
  oss << "Device: "
      << DataRecyclerUtil::getDeviceIdentifierString(PROPERTY_CACHE_DEVICE_IDENTIFIER)
      << "\n";
  for (auto& kv : *property_cache_container) {
    oss << "\tkey: " << kv.key
        << ", property: " << getHashtablePropertyString(*kv.cached_item) << "\n";
  }
  return oss.str();
}

bool HashTablePropertyRecycler::hasItemInCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<EMPTY_META_INFO> meta_info) const {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return false;
  }
  CHECK_EQ(item_type, CacheItemType::HT_PROPERTY);
  auto property_cache = getCachedItemContainer(item_type, device_identifier);
  return std::any_of(property_cache->begin(),
                     property_cache->end(),
                     [&key](const auto& cached_item) { return cached_item.key == key; });
}

void HashTablePropertyRecycler::addQueryPlanDagForTableKeys(
    size_t hashed_query_plan_dag,
    const std::unordered_set<size_t>& table_keys) {
  std::lock_guard<std::mutex> lock(getCacheLock());
  for (auto table_key : table_keys) {
    auto it = table_key_to_query_plan_dag_map_.find(table_key);
    if (it != table_key_to_query_plan_dag_map_.end()) {
      it->second.insert(hashed_query_plan_dag);
    } else {
      std::unordered_set<size_t> query_plan_dags{hashed_query_plan_dag};
      table_key_to_query_plan_dag_map_.emplace(table_key, query_plan_dags);
    }
  }
}

std::optional<std::unordered_set<size_t>>
HashTablePropertyRecycler::getMappedQueryPlanDagsWithTableKey(size_t table_key) const {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto it = table_key_to_query_plan_dag_map_.find(table_key);
  return it != table_key_to_query_plan_dag_map_.end() ? std::make_optional(it->second)
                                                      : std::nullopt;
}

void HashTablePropertyRecycler::removeTableKeyInfoFromQueryPlanDagMap(size_t table_key) {
  // this function is called when marking cached item for the given table_key as dirty
  // and when we do that we already acquire the cache lock so we skip to lock in this func
  table_key_to_query_plan_dag_map_.erase(table_key);
}
