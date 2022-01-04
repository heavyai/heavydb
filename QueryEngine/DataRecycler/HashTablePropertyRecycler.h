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

#pragma once

#include "DataRecycler.h"
#include "QueryEngine/QueryHint.h"

constexpr DeviceIdentifier PROPERTY_CACHE_DEVICE_IDENTIFIER =
    DataRecyclerUtil::CPU_DEVICE_IDENTIFIER;

struct HashTableProperty {
  HashTableProperty(std::optional<HashTableLayoutType> layout_in,
                    std::optional<HashTableHashingType> hashing_in)
      : layout(layout_in), hashing(hashing_in) {}

  std::optional<HashTableLayoutType> layout;
  std::optional<HashTableHashingType> hashing;
};

class HashTablePropertyRecycler
    : public DataRecycler<std::optional<HashTableProperty>, EMPTY_META_INFO> {
 public:
  // hashing scheme recycler caches logical information instead of actual data
  // so we do not limit its capacity
  // thus we do not maintain a metric cache for hashing scheme
  HashTablePropertyRecycler()
      : DataRecycler({CacheItemType::HT_PROPERTY},
                     std::numeric_limits<size_t>::max(),
                     std::numeric_limits<size_t>::max(),
                     0) {}

  std::optional<HashTableProperty> getItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) override;

  void putItemToCache(QueryPlanHash key,
                      std::optional<HashTableProperty> item,
                      CacheItemType item_type,
                      DeviceIdentifier device_identifier,
                      size_t item_size,
                      size_t compute_time,
                      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) override;

  void updateItemInCacheIfNecessary(QueryPlanHash key,
                                    std::optional<HashTableProperty> item,
                                    CacheItemType item_type,
                                    DeviceIdentifier device_identifier);

  // nothing to do with hashing scheme recycler
  void initCache() override {}

  void clearCache() override;

  void markCachedItemAsDirty(size_t table_key,
                             std::unordered_set<QueryPlanHash>& key_set,
                             CacheItemType item_type,
                             DeviceIdentifier device_identifier) override;

  std::string toString() const override;

  static HashType translateLayoutType(HashTableLayoutType layout_type) {
    // we do not consider ManyToMany type
    return layout_type == HashTableLayoutType::ONE ? HashType::OneToOne
                                                   : HashType::OneToMany;
  }

  static HashTableLayoutType translateHashType(HashType type) {
    return type == HashType::OneToOne ? HashTableLayoutType::ONE
                                      : HashTableLayoutType::MANY;
  }

  std::optional<std::unordered_set<size_t>> getMappedQueryPlanDagsWithTableKey(
      size_t table_key) const;

  void removeTableKeyInfoFromQueryPlanDagMap(size_t table_key);

  void addQueryPlanDagForTableKeys(size_t hashed_query_plan_dag,
                                   const std::unordered_set<size_t>& table_keys);

  static std::string getLayoutString(std::optional<HashTableLayoutType> layout) {
    auto layout_str =
        layout ? *layout == HashTableLayoutType::ONE ? "OneToOne" : "OneToMany" : "None";
    return layout_str;
  }

  static std::string getHashingString(std::optional<HashTableHashingType> hashing) {
    auto hashing_str =
        hashing ? *hashing == HashTableHashingType::BASELINE ? "Baseline" : "Perfect"
                : "None";
    return hashing_str;
  }

  static std::string getHashtablePropertyString(HashTableProperty prop) {
    auto ret = "{layout: " + getLayoutString(prop.layout) +
               ", hashing: " + getHashingString(prop.hashing) + "}";
    return ret;
  }

 private:
  bool hasItemInCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) const override;

  // hashing scheme recycler clears the cached layouts at once
  void removeItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) override;

  // hashing scheme recycler has unlimited capacity so we do not need this
  void cleanupCacheForInsertion(
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      size_t required_size,
      std::lock_guard<std::mutex>& lock,
      std::optional<EMPTY_META_INFO> meta_info = std::nullopt) override {}

  std::unordered_map<size_t, std::unordered_set<size_t>> table_key_to_query_plan_dag_map_;
};
