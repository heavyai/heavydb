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

#include "OverlapsTuningParamRecycler.h"

extern bool g_use_hashtable_cache;
extern bool g_enable_data_recycler;

std::optional<AutoTunerMetaInfo> OverlapsTuningParamRecycler::getItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::optional<EMPTY_META_INFO> meta_info) const {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return std::nullopt;
  }
  CHECK_EQ(item_type, CacheItemType::OVERLAPS_AUTO_TUNER_PARAM);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  auto param_cache = getCachedItem(key, *layout_cache);
  if (param_cache) {
    VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Recycle auto tuner parameters for the overlaps hash join qual";
    return param_cache->cached_item;
  }
  return std::nullopt;
}

void OverlapsTuningParamRecycler::putItemToCache(
    QueryPlanHash key,
    std::optional<AutoTunerMetaInfo> item,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    size_t item_size,
    size_t compute_time,
    std::optional<EMPTY_META_INFO> meta_info) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::OVERLAPS_AUTO_TUNER_PARAM);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  auto param_cache = getCachedItem(key, *layout_cache);
  if (param_cache) {
    return;
  }
  layout_cache->emplace_back(key, item, nullptr, meta_info);
  VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
          << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
          << "] Put auto tuner parameters for the overlaps hash join qual to cache";
  return;
}

bool OverlapsTuningParamRecycler::hasItemInCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<EMPTY_META_INFO> meta_info) const {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return false;
  }
  CHECK_EQ(item_type, CacheItemType::OVERLAPS_AUTO_TUNER_PARAM);
  auto layout_cache = getCachedItemContainer(item_type, device_identifier);
  return getCachedItem(key, *layout_cache).has_value();
}

void OverlapsTuningParamRecycler::clearCache() {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto param_cache = getCachedItemContainer(CacheItemType::OVERLAPS_AUTO_TUNER_PARAM,
                                            PARAM_CACHE_DEVICE_IDENTIFIER);
  param_cache->clear();
}

std::string OverlapsTuningParamRecycler::toString() const {
  std::ostringstream oss;
  oss << "A current status of the Overlaps Join Hashtable Tuning Parameter Recycler:\n";
  oss << "\t# cached parameters:\n";
  oss << "\t\tDevice" << PARAM_CACHE_DEVICE_IDENTIFIER << "\n";
  auto param_cache = getCachedItemContainer(CacheItemType::OVERLAPS_AUTO_TUNER_PARAM,
                                            PARAM_CACHE_DEVICE_IDENTIFIER);
  for (auto& cache_container : *param_cache) {
    oss << "\t\t\tCache_key: " << cache_container.key;
    if (cache_container.cached_item.has_value()) {
      oss << ", Max_hashtable_size: " << cache_container.cached_item->max_hashtable_size
          << ", Bucket_threshold: " << cache_container.cached_item->bucket_threshold
          << ", Bucket_sizes: " << ::toString(cache_container.cached_item->bucket_sizes)
          << "\n";
    } else {
      oss << ", Params info is not available\n";
    }
  }
  return oss.str();
}
