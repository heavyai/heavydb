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
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "Shared/Config.h"

struct QueryPlanMetaInfo {
  QueryPlan query_plan_dag;
  std::string inner_col_info_string;
};

struct HashtableCacheMetaInfo {
  std::optional<QueryPlanMetaInfo> query_plan_meta_info;
};

class HashtableRecycler
    : public DataRecycler<std::shared_ptr<HashTable>, HashtableCacheMetaInfo> {
 public:
  HashtableRecycler(ConfigPtr config, CacheItemType hashtable_type, int num_gpus)
      : DataRecycler({hashtable_type},
                     config->cache.hashtable_cache_total_bytes,
                     config->cache.max_cacheable_hashtable_size_bytes,
                     num_gpus)
      , config_(config) {}

  std::shared_ptr<HashTable> getItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::optional<HashtableCacheMetaInfo> meta_info = std::nullopt) const override;

  void putItemToCache(
      QueryPlanHash key,
      std::shared_ptr<HashTable> item_ptr,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      size_t item_size,
      size_t compute_time,
      std::optional<HashtableCacheMetaInfo> meta_info = std::nullopt) override;

  // nothing to do with hashtable recycler
  void initCache() override {}

  void clearCache() override;

  std::string toString() const override;

  static std::pair<QueryPlanHash, HashtableCacheMetaInfo> getHashtableCacheKey(
      const std::vector<InnerOuter>& inner_outer_pairs,
      const SQLOps op_type,
      const JoinType join_type,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      Executor* executor);

  static std::pair<QueryPlan, HashtableCacheMetaInfo> getHashtableKeyString(
      const std::vector<InnerOuter>& inner_outer_pairs,
      const SQLOps op_type,
      const JoinType join_type,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      Executor* executor);

  static std::string getJoinColumnInfoString(
      std::vector<const Analyzer::ColumnVar*>& inner_cols,
      std::vector<const Analyzer::ColumnVar*>& outer_cols,
      Executor* executor);

  static bool isSafeToCacheHashtable(const TableIdToNodeMap& table_id_to_node_map,
                                     bool need_dict_translation,
                                     const int table_id);

  // this function is required to test data recycler
  // specifically, it is tricky to get a hashtable cache key when we only know
  // a target query sql in test code
  // so this function utilizes an incorrect way to manipulate our hashtable recycler
  // but provides the cached hashtable for performing the test
  // a set "visited" contains cached hashtable keys that we have retrieved so far
  // based on that, this function iterates hashtable cache and return a cached one
  // when its hashtable cache key has not been visited yet
  // for instance, if we call this function with an empty "visited" key, we return
  // the first hashtable that its iterator visits
  std::tuple<QueryPlanHash,
             std::shared_ptr<HashTable>,
             std::optional<HashtableCacheMetaInfo>>
  getCachedHashtableWithoutCacheKey(std::set<size_t>& visited,
                                    CacheItemType hash_table_type,
                                    DeviceIdentifier device_identifier);

 private:
  bool hasItemInCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<HashtableCacheMetaInfo> meta_info = std::nullopt) const override;

  void removeItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<HashtableCacheMetaInfo> meta_info = std::nullopt) override;

  void cleanupCacheForInsertion(
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      size_t required_size,
      std::lock_guard<std::mutex>& lock,
      std::optional<HashtableCacheMetaInfo> meta_info = std::nullopt) override;

  ConfigPtr config_;
};
