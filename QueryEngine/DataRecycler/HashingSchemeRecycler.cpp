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

#include "HashingSchemeRecycler.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"

std::optional<HashType> HashingSchemeRecycler::getItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::optional<EMPTY_META_INFO> meta_info) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return std::nullopt;
  }
  CHECK_EQ(item_type, CacheItemType::HT_HASHING_SCHEME);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_layout = getCachedItemWithoutConsideringMetaInfo(
      key, item_type, device_identifier, *layout_cache, lock);
  if (candidate_layout) {
    CHECK(!candidate_layout->isDirty());
    VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Recycle hashtable layout in cache: "
            << HashJoin::getHashTypeString(*candidate_layout->cached_item);
    return candidate_layout->cached_item;
  }
  return std::nullopt;
}

void HashingSchemeRecycler::putItemToCache(QueryPlanHash key,
                                           std::optional<HashType> item,
                                           CacheItemType item_type,
                                           DeviceIdentifier device_identifier,
                                           size_t item_size,
                                           size_t compute_time,
                                           std::optional<EMPTY_META_INFO> meta_info) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::HT_HASHING_SCHEME);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_layout = getCachedItemWithoutConsideringMetaInfo(
      key, item_type, device_identifier, *layout_cache, lock);
  if (candidate_layout) {
    return;
  }
  layout_cache->emplace_back(key, item, nullptr, meta_info);
  VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
          << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
          << "] Put hashtable layout to cache";
  return;
}

void HashingSchemeRecycler::removeItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<EMPTY_META_INFO> meta_info) {
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  auto filter = [key](auto const& item) { return item.key == key; };
  auto itr = std::find_if(layout_cache->cbegin(), layout_cache->cend(), filter);
  if (itr == layout_cache->cend()) {
    return;
  } else {
    layout_cache->erase(itr);
  }
}

void HashingSchemeRecycler::clearCache() {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache_container = getCachedItemContainer(CacheItemType::HT_HASHING_SCHEME,
                                                       LAYOUT_CACHE_DEVICE_IDENTIFIER);
  layout_cache_container->clear();
}

void HashingSchemeRecycler::markCachedItemAsDirty(
    size_t table_key,
    std::unordered_set<QueryPlanHash>& key_set,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache || key_set.empty()) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::HT_HASHING_SCHEME);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  for (auto key : key_set) {
    markCachedItemAsDirtyImpl(key, *layout_cache);
  }
}

std::string HashingSchemeRecycler::toString() const {
  auto layout_cache_container = getCachedItemContainer(CacheItemType::HT_HASHING_SCHEME,
                                                       LAYOUT_CACHE_DEVICE_IDENTIFIER);
  std::ostringstream oss;
  oss << "Hashing scheme cache:\n";
  oss << "Device: "
      << DataRecyclerUtil::getDeviceIdentifierString(LAYOUT_CACHE_DEVICE_IDENTIFIER)
      << "\n";
  for (auto& kv : *layout_cache_container) {
    oss << "\tkey: " << kv.key
        << ", layout: " << HashJoin::getHashTypeString(*kv.cached_item) << "\n";
  }
  return oss.str();
}

bool HashingSchemeRecycler::hasItemInCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<EMPTY_META_INFO> meta_info) const {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return false;
  }
  CHECK_EQ(item_type, CacheItemType::HT_HASHING_SCHEME);
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  return std::any_of(layout_cache->begin(),
                     layout_cache->end(),
                     [&key](const auto& cached_item) { return cached_item.key == key; });
}
