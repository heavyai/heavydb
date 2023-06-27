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

#include "BoundingBoxIntersectTuningParamRecycler.h"

std::optional<AutoTunerMetaInfo>
BoundingBoxIntersectTuningParamRecycler::getItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::optional<EMPTY_META_INFO> meta_info) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return std::nullopt;
  }
  CHECK_EQ(item_type, CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto param_cache = getCachedItemContainer(item_type, device_identifier);
  auto cached_param = getCachedItemWithoutConsideringMetaInfo(
      key, item_type, device_identifier, *param_cache, lock);
  if (cached_param) {
    CHECK(!cached_param->isDirty());
    VLOG(1) << "[" << item_type << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Recycle auto tuner parameters in cache (key: " << key << ")";
    return cached_param->cached_item;
  }
  return std::nullopt;
}

void BoundingBoxIntersectTuningParamRecycler::putItemToCache(
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
  CHECK_EQ(item_type, CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto param_cache = getCachedItemContainer(item_type, device_identifier);
  auto cached_param = getCachedItemWithoutConsideringMetaInfo(
      key, item_type, device_identifier, *param_cache, lock);
  if (cached_param) {
    return;
  }
  param_cache->emplace_back(key, item, nullptr, meta_info);
  VLOG(1) << "[" << item_type << ", "
          << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
          << "] Put auto tuner parameters to cache (key: " << key << ")";
  return;
}

bool BoundingBoxIntersectTuningParamRecycler::hasItemInCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<EMPTY_META_INFO> meta_info) const {
  if (!g_enable_data_recycler || !g_use_hashtable_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return false;
  }
  CHECK_EQ(item_type, CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM);
  auto param_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_it = std::find_if(
      param_cache->begin(), param_cache->end(), [&key](const auto& cached_item) {
        return cached_item.key == key;
      });
  return candidate_it != param_cache->end();
}

void BoundingBoxIntersectTuningParamRecycler::removeItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<EMPTY_META_INFO> meta_info) {
  auto param_cache = getCachedItemContainer(item_type, device_identifier);
  auto filter = [key](auto const& item) { return item.key == key; };
  auto itr = std::find_if(param_cache->cbegin(), param_cache->cend(), filter);
  if (itr == param_cache->cend()) {
    return;
  } else {
    VLOG(1) << "[" << item_type << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] remove cached item from cache (key: " << key << ")";
    param_cache->erase(itr);
  }
}

void BoundingBoxIntersectTuningParamRecycler::clearCache() {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto param_cache = getCachedItemContainer(
      CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM, PARAM_CACHE_DEVICE_IDENTIFIER);
  if (!param_cache->empty()) {
    VLOG(1) << "[" << CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(PARAM_CACHE_DEVICE_IDENTIFIER)
            << "] clear cache (# items: " << param_cache->size() << ")";
    param_cache->clear();
  }
}

void BoundingBoxIntersectTuningParamRecycler::markCachedItemAsDirty(
    size_t table_key,
    std::unordered_set<QueryPlanHash>& key_set,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  if (!g_enable_data_recycler || !g_use_hashtable_cache || key_set.empty()) {
    return;
  }
  CHECK_EQ(item_type, CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM);
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto param_cache = getCachedItemContainer(item_type, device_identifier);
  for (auto key : key_set) {
    markCachedItemAsDirtyImpl(key, *param_cache);
  }
}

std::string BoundingBoxIntersectTuningParamRecycler::toString() const {
  std::ostringstream oss;
  oss << "A current status of the Bounding Box Intersection Tuning Parameter Recycler:\n";
  oss << "\t# cached parameters:\n";
  oss << "\t\tDevice" << PARAM_CACHE_DEVICE_IDENTIFIER << "\n";
  auto param_cache = getCachedItemContainer(
      CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM, PARAM_CACHE_DEVICE_IDENTIFIER);
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
