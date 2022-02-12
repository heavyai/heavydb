/*
 * Copyright 2022 OmniSci, Inc.
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

#include "QueryEngine/DataRecycler/DataRecycler.h"
#include "QueryEngine/QueryHint.h"
#include "QueryEngine/RelAlgExecutionUnit.h"

struct ResultSetMetaInfo {
  ResultSetMetaInfo(const std::unordered_set<size_t> input_table_infos) {
    input_table_keys.insert(input_table_infos.begin(), input_table_infos.end());
  };

  void keepTargetExprs(std::vector<std::shared_ptr<Analyzer::Expr>>& in_target_exprs) {
    for (const auto& expr : in_target_exprs) {
      target_exprs.push_back(expr->get_shared_ptr());
    }
  }

  std::vector<std::shared_ptr<Analyzer::Expr>>& getTargetExprs() { return target_exprs; }

  std::unordered_set<size_t> input_table_keys;
  std::vector<std::shared_ptr<Analyzer::Expr>> target_exprs;
};

class ResultSetRecycler : public DataRecycler<ResultSetPtr, ResultSetMetaInfo> {
 public:
  ResultSetRecycler()
      : DataRecycler({CacheItemType::QUERY_RESULTSET},
                     g_query_resultset_cache_total_bytes,
                     g_max_cacheable_query_resultset_size_bytes,
                     0) {}

  ResultSetPtr getItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::optional<ResultSetMetaInfo> meta_info = std::nullopt) override;

  std::optional<std::vector<TargetMetaInfo>> getOutputMetaInfo(QueryPlanHash key);

  void putItemToCache(QueryPlanHash key,
                      ResultSetPtr item_ptr,
                      CacheItemType item_type,
                      DeviceIdentifier device_identifier,
                      size_t item_size,
                      size_t compute_time,
                      std::optional<ResultSetMetaInfo> meta_info = std::nullopt) override;

  // nothing to do with resultset recycler
  void initCache() override {}

  void clearCache() override;

  void markCachedItemAsDirty(size_t table_key,
                             std::unordered_set<QueryPlanHash>& key_set,
                             CacheItemType item_type,
                             DeviceIdentifier device_identifier) override;

  std::string toString() const override;

  std::tuple<QueryPlanHash, ResultSetPtr, std::optional<ResultSetMetaInfo>>
  getCachedResultSetWithoutCacheKey(std::set<size_t>& visited,
                                    DeviceIdentifier device_identifier);

  void addQueryPlanDagForTableKeys(size_t hashed_query_plan_dag,
                                   const std::unordered_set<size_t>& table_keys,
                                   std::lock_guard<std::mutex>& lock);

  std::optional<std::unordered_set<size_t>> getMappedQueryPlanDagsWithTableKey(
      size_t table_key) const;

  void removeTableKeyInfoFromQueryPlanDagMap(size_t table_key);

  bool hasItemInCache(QueryPlanHash key);

  std::vector<std::shared_ptr<Analyzer::Expr>>& getTargetExprs(QueryPlanHash key) const;

 private:
  bool hasItemInCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<ResultSetMetaInfo> meta_info = std::nullopt) const override;

  void removeItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<ResultSetMetaInfo> meta_info = std::nullopt) override;

  void cleanupCacheForInsertion(
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      size_t required_size,
      std::lock_guard<std::mutex>& lock,
      std::optional<ResultSetMetaInfo> meta_info = std::nullopt) override;

  // keep all table keys referenced to compute cached resultset
  std::unordered_map<size_t, std::unordered_set<size_t>> table_key_to_query_plan_dag_map_;
};
