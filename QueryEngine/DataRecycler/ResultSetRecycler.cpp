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

#include "ResultSetRecycler.h"

extern bool g_is_test_env;

bool ResultSetRecycler::hasItemInCache(QueryPlanHash key,
                                       CacheItemType item_type,
                                       DeviceIdentifier device_identifier,
                                       std::lock_guard<std::mutex>& lock,
                                       std::optional<ResultSetMetaInfo> meta_info) const {
  if (!g_enable_data_recycler || !g_use_query_resultset_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return false;
  }
  auto resultset_cache = getCachedItemContainer(item_type, device_identifier);
  CHECK(resultset_cache);
  auto candidate_resultset_it = std::find_if(
      resultset_cache->begin(), resultset_cache->end(), [&key](const auto& cached_item) {
        return cached_item.key == key;
      });
  return candidate_resultset_it != resultset_cache->end();
}

bool ResultSetRecycler::hasItemInCache(QueryPlanHash key) {
  std::lock_guard<std::mutex> lock(getCacheLock());
  return hasItemInCache(key,
                        CacheItemType::QUERY_RESULTSET,
                        DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                        lock,
                        std::nullopt);
}

ResultSetPtr ResultSetRecycler::getItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::optional<ResultSetMetaInfo> meta_info) {
  if (!g_enable_data_recycler || !g_use_query_resultset_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto resultset_cache = getCachedItemContainer(item_type, device_identifier);
  CHECK(resultset_cache);
  auto candidate_resultset_it = std::find_if(
      resultset_cache->begin(), resultset_cache->end(), [&key](const auto& cached_item) {
        return cached_item.key == key;
      });
  if (candidate_resultset_it != resultset_cache->end()) {
    CHECK(candidate_resultset_it->meta_info);
    if (candidate_resultset_it->isDirty()) {
      removeItemFromCache(
          key, item_type, device_identifier, lock, candidate_resultset_it->meta_info);
      return nullptr;
    }
    auto candidate_resultset = candidate_resultset_it->cached_item;
    decltype(std::chrono::steady_clock::now()) ts1, ts2;
    ts1 = std::chrono::steady_clock::now();
    // we need to copy cached resultset to support resultset recycler with concurrency
    auto copied_rs = candidate_resultset->copy();
    CHECK(copied_rs);
    copied_rs->setCached(true);
    copied_rs->initStatus();
    candidate_resultset_it->item_metric->incRefCount();
    ts2 = std::chrono::steady_clock::now();
    VLOG(1) << "[" << item_type << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Get cached query resultset from cache (key: " << key
            << ", copying it takes "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count()
            << "ms)";
    return copied_rs;
  }
  return nullptr;
}

std::optional<std::vector<TargetMetaInfo>> ResultSetRecycler::getOutputMetaInfo(
    QueryPlanHash key) {
  if (!g_enable_data_recycler || !g_use_query_resultset_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return std::nullopt;
  }
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto resultset_cache = getCachedItemContainer(CacheItemType::QUERY_RESULTSET,
                                                DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  CHECK(resultset_cache);
  auto candidate_resultset_it = std::find_if(
      resultset_cache->begin(), resultset_cache->end(), [&key](const auto& cached_item) {
        return cached_item.key == key;
      });
  if (candidate_resultset_it != resultset_cache->end()) {
    CHECK(candidate_resultset_it->meta_info);
    if (candidate_resultset_it->isDirty()) {
      removeItemFromCache(key,
                          CacheItemType::QUERY_RESULTSET,
                          DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                          lock,
                          candidate_resultset_it->meta_info);
      return std::nullopt;
    }
    auto candidate_resultset = candidate_resultset_it->cached_item;
    auto output_meta_info = candidate_resultset->getTargetMetaInfo();
    return output_meta_info;
  }
  return std::nullopt;
}

void ResultSetRecycler::putItemToCache(QueryPlanHash key,
                                       ResultSetPtr item_ptr,
                                       CacheItemType item_type,
                                       DeviceIdentifier device_identifier,
                                       size_t item_size,
                                       size_t compute_time,
                                       std::optional<ResultSetMetaInfo> meta_info) {
  if (!g_enable_data_recycler || !g_use_query_resultset_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  CHECK(meta_info.has_value());
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto resultset_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_resultset_it = std::find_if(
      resultset_cache->begin(), resultset_cache->end(), [&key](const auto& cached_item) {
        return cached_item.key == key;
      });
  bool has_cached_resultset = false;
  bool need_to_cleanup = false;
  if (candidate_resultset_it != resultset_cache->end()) {
    has_cached_resultset = true;
    CHECK(candidate_resultset_it->meta_info);
    if (candidate_resultset_it->isDirty()) {
      need_to_cleanup = true;
    } else if (candidate_resultset_it->cached_item->didOutputColumnar() !=
               item_ptr->didOutputColumnar()) {
      // we already have a cached resultset for the given query plan dag but
      // requested resultset output layout and that of cached one is different
      // so we remove the cached one and make a room for the resultset with different
      // layout
      need_to_cleanup = true;
      VLOG(1) << "Failed to recycle query resultset: mismatched cached resultset layout";
    }
  }
  if (need_to_cleanup) {
    // remove dirty cached resultset
    removeItemFromCache(
        key, item_type, device_identifier, lock, candidate_resultset_it->meta_info);
    has_cached_resultset = false;
  }

  if (!has_cached_resultset) {
    auto& metric_tracker = getMetricTracker(item_type);
    auto cache_status = metric_tracker.canAddItem(device_identifier, item_size);
    if (cache_status == CacheAvailability::UNAVAILABLE) {
      LOG(INFO) << "Failed to keep a query resultset: the size of the resultset ("
                << item_size << " bytes) exceeds the current system limit ("
                << g_max_cacheable_query_resultset_size_bytes << " bytes)";
      return;
    } else if (cache_status == CacheAvailability::AVAILABLE_AFTER_CLEANUP) {
      auto required_size = metric_tracker.calculateRequiredSpaceForItemAddition(
          device_identifier, item_size);
      LOG(INFO) << "Cleanup cached query resultset(s) to make a free space ("
                << required_size << " bytes) to cache a new resultset";
      cleanupCacheForInsertion(item_type, device_identifier, required_size, lock);
    }
    auto new_cache_metric_ptr = metric_tracker.putNewCacheItemMetric(
        key, device_identifier, item_size, compute_time);
    CHECK_EQ(item_size, new_cache_metric_ptr->getMemSize());
    item_ptr->setCached(true);
    item_ptr->initStatus();
    VLOG(1) << "[" << item_type << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Put query resultset to cache (key: " << key << ")";
    resultset_cache->emplace_back(key, item_ptr, new_cache_metric_ptr, meta_info);
    if (!meta_info->input_table_keys.empty()) {
      addQueryPlanDagForTableKeys(key, meta_info->input_table_keys, lock);
    }
  }
  return;
}

void ResultSetRecycler::removeItemFromCache(QueryPlanHash key,
                                            CacheItemType item_type,
                                            DeviceIdentifier device_identifier,
                                            std::lock_guard<std::mutex>& lock,
                                            std::optional<ResultSetMetaInfo> meta_info) {
  if (!g_enable_data_recycler || !g_use_query_resultset_cache ||
      key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  auto resultset_container = getCachedItemContainer(item_type, device_identifier);
  auto filter = [key](auto const& item) { return item.key == key; };
  auto itr =
      std::find_if(resultset_container->cbegin(), resultset_container->cend(), filter);
  if (itr == resultset_container->cend()) {
    return;
  } else {
    itr->cached_item->invalidateResultSetChunks();
    VLOG(1) << "[" << item_type << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Remove item from cache (key: " << key << ")";
    resultset_container->erase(itr);
  }
  auto& cache_metrics = getMetricTracker(item_type);
  auto cache_metric = cache_metrics.getCacheItemMetric(key, device_identifier);
  CHECK(cache_metric);
  auto resultset_size = cache_metric->getMemSize();
  cache_metrics.removeCacheItemMetric(key, device_identifier);
  cache_metrics.updateCurrentCacheSize(
      device_identifier, CacheUpdateAction::REMOVE, resultset_size);
  return;
}

void ResultSetRecycler::cleanupCacheForInsertion(
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    size_t required_size,
    std::lock_guard<std::mutex>& lock,
    std::optional<ResultSetMetaInfo> meta_info) {
  int elimination_target_offset = 0;
  size_t removed_size = 0;
  auto& metric_tracker = getMetricTracker(item_type);
  auto actual_space_to_free = metric_tracker.getTotalCacheSize() / 2;
  if (!g_is_test_env && required_size < actual_space_to_free) {
    required_size = actual_space_to_free;
  }
  metric_tracker.sortCacheInfoByQueryMetric(device_identifier);
  auto cached_item_metrics = metric_tracker.getCacheItemMetrics(device_identifier);
  sortCacheContainerByQueryMetric(item_type, device_identifier);

  for (auto& metric : cached_item_metrics) {
    auto target_size = metric->getMemSize();
    ++elimination_target_offset;
    removed_size += target_size;
    if (removed_size > required_size) {
      break;
    }
  }

  removeCachedItemFromBeginning(item_type, device_identifier, elimination_target_offset);
  metric_tracker.removeMetricFromBeginning(device_identifier, elimination_target_offset);

  metric_tracker.updateCurrentCacheSize(
      device_identifier, CacheUpdateAction::REMOVE, removed_size);
}

void ResultSetRecycler::clearCache() {
  std::lock_guard<std::mutex> lock(getCacheLock());
  for (auto& item_type : getCacheItemType()) {
    getMetricTracker(item_type).clearCacheMetricTracker();
    auto item_cache = getItemCache().find(item_type)->second;
    for (auto& kv : *item_cache) {
      std::for_each(kv.second->begin(), kv.second->end(), [](const auto& container) {
        container.cached_item->invalidateResultSetChunks();
      });
      VLOG(1) << "[" << item_type << ", "
              << DataRecyclerUtil::getDeviceIdentifierString(
                     DataRecyclerUtil::CPU_DEVICE_IDENTIFIER)
              << "] clear cache (# items: " << kv.second->size() << ")";
      kv.second->clear();
    }
  }
  table_key_to_query_plan_dag_map_.clear();
}

void ResultSetRecycler::markCachedItemAsDirty(size_t table_key,
                                              std::unordered_set<QueryPlanHash>& key_set,
                                              CacheItemType item_type,
                                              DeviceIdentifier device_identifier) {
  if (!g_enable_data_recycler || !g_use_query_resultset_cache || key_set.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto resultset_cache = getCachedItemContainer(item_type, device_identifier);
  for (auto key : key_set) {
    removeItemFromCache(key, item_type, device_identifier, lock, std::nullopt);
  }
  removeTableKeyInfoFromQueryPlanDagMap(table_key);
}

std::string ResultSetRecycler::toString() const {
  std::ostringstream oss;
  oss << "A current status of the query resultSet Recycler:\n";
  for (auto& item_type : getCacheItemType()) {
    oss << "\t" << item_type;
    auto& metric_tracker = getMetricTracker(item_type);
    oss << "\n\t# cached query resultsets:\n";
    auto item_cache = getItemCache().find(item_type)->second;
    for (auto& cache_container : *item_cache) {
      oss << "\t\tDevice"
          << DataRecyclerUtil::getDeviceIdentifierString(cache_container.first)
          << ", # query resultsets: " << cache_container.second->size() << "\n";
      for (auto& ht : *cache_container.second) {
        oss << "\t\t\tHT] " << ht.item_metric->toString() << "\n";
      }
    }
    oss << "\t" << metric_tracker.toString() << "\n";
  }
  return oss.str();
}

std::tuple<QueryPlanHash, ResultSetPtr, std::optional<ResultSetMetaInfo>>
ResultSetRecycler::getCachedResultSetWithoutCacheKey(std::set<size_t>& visited,
                                                     DeviceIdentifier device_identifier) {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto resultset_cache =
      getCachedItemContainer(CacheItemType::QUERY_RESULTSET, device_identifier);
  for (auto& rs : *resultset_cache) {
    if (!visited.count(rs.key)) {
      return std::make_tuple(rs.key, rs.cached_item, rs.meta_info);
    }
  }
  return std::make_tuple(EMPTY_HASHED_PLAN_DAG_KEY, nullptr, std::nullopt);
}

void ResultSetRecycler::addQueryPlanDagForTableKeys(
    size_t hashed_query_plan_dag,
    const std::unordered_set<size_t>& table_keys,
    std::lock_guard<std::mutex>& lock) {
  for (auto table_key : table_keys) {
    auto itr = table_key_to_query_plan_dag_map_.try_emplace(table_key).first;
    itr->second.insert(hashed_query_plan_dag);
  }
}

std::optional<std::unordered_set<size_t>>
ResultSetRecycler::getMappedQueryPlanDagsWithTableKey(size_t table_key) const {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto it = table_key_to_query_plan_dag_map_.find(table_key);
  return it != table_key_to_query_plan_dag_map_.end() ? std::make_optional(it->second)
                                                      : std::nullopt;
}

void ResultSetRecycler::removeTableKeyInfoFromQueryPlanDagMap(size_t table_key) {
  table_key_to_query_plan_dag_map_.erase(table_key);
}

std::vector<std::shared_ptr<Analyzer::Expr>>& ResultSetRecycler::getTargetExprs(
    QueryPlanHash key) const {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto resultset_cache = getCachedItemContainer(CacheItemType::QUERY_RESULTSET,
                                                DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  CHECK(resultset_cache);
  auto candidate_resultset_it = std::find_if(
      resultset_cache->begin(), resultset_cache->end(), [&key](const auto& cached_item) {
        return cached_item.key == key;
      });
  CHECK(candidate_resultset_it != resultset_cache->end());
  CHECK(candidate_resultset_it->meta_info);
  return candidate_resultset_it->meta_info->getTargetExprs();
}
