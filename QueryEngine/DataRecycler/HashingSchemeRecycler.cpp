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

extern bool g_use_hashtable_cache;

std::optional<HashType> HashingSchemeRecycler::getItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::optional<EMPTY_META_INFO> meta_info) const {
  if (!config_->cache.enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return std::nullopt;
  }
  CHECK_EQ(item_type, CacheItemType::HT_HASHING_SCHEME);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_layout = getCachedItem(key, *layout_cache);
  if (candidate_layout) {
    VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Recycle hashtable layout for the join qual: "
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
  if (!config_->cache.enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::HT_HASHING_SCHEME);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_layout = getCachedItem(key, *layout_cache);
  if (candidate_layout) {
    return;
  }
  layout_cache->emplace_back(key, item, nullptr, meta_info);
  VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
          << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
          << "] Put hashtable layout for the join qual to cache: "
          << HashJoin::getHashTypeString(*item);
  return;
}

void HashingSchemeRecycler::clearCache() {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache_container = getCachedItemContainer(CacheItemType::HT_HASHING_SCHEME,
                                                       LAYOUT_CACHE_DEVICE_IDENTIFIER);
  layout_cache_container->clear();
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
  if (!config_->cache.enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return false;
  }
  CHECK_EQ(item_type, CacheItemType::HT_HASHING_SCHEME);
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  return getCachedItem(key, *layout_cache).has_value();
}
