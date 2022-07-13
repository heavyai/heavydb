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

#include "DataRecycler.h"

#include <numeric>

constexpr DeviceIdentifier PARAM_CACHE_DEVICE_IDENTIFIER =
    DataRecyclerUtil::CPU_DEVICE_IDENTIFIER;

struct AutoTunerMetaInfo {
  size_t max_hashtable_size;
  double bucket_threshold;
  std::vector<double> bucket_sizes;
};

class OverlapsTuningParamRecycler
    : public DataRecycler<std::optional<AutoTunerMetaInfo>, EMPTY_META_INFO> {
 public:
  // hashing scheme recycler caches logical information instead of actual data
  // so we do not limit its capacity
  // thus we do not maintain a metric cache for hashing scheme
  OverlapsTuningParamRecycler()
      : DataRecycler({CacheItemType::OVERLAPS_AUTO_TUNER_PARAM},
                     std::numeric_limits<size_t>::max(),
                     std::numeric_limits<size_t>::max(),
                     0) {}

  std::optional<AutoTunerMetaInfo> getItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) override;

  void putItemToCache(QueryPlanHash key,
                      std::optional<AutoTunerMetaInfo> item,
                      CacheItemType item_type,
                      DeviceIdentifier device_identifier,
                      size_t item_size,
                      size_t compute_time,
                      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) override;

  // nothing to do with hashing scheme recycler
  void initCache() override {}

  void clearCache() override;

  void markCachedItemAsDirty(size_t table_key,
                             std::unordered_set<QueryPlanHash>& key_set,
                             CacheItemType item_type,
                             DeviceIdentifier device_identifier) override;

  std::string toString() const override;

 private:
  bool hasItemInCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) const override;

  // auto tuner param recycler has unlimited capacity so we do not need this
  void removeItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) override;

  // hauto tuner param has unlimited capacity so we do not need this
  void cleanupCacheForInsertion(
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      size_t required_size,
      std::lock_guard<std::mutex>& lock,
      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) override {
    UNREACHABLE();
  }
};
