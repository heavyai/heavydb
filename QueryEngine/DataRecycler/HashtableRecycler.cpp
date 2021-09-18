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

#include "HashtableRecycler.h"

extern bool g_is_test_env;

bool HashtableRecycler::hasItemInCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<HashtableCacheMetaInfo> meta_info) const {
  if (key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return false;
  }
  auto hashtable_cache = getCachedItemContainer(item_type, device_identifier);
  // hashtable cache of the *any* device type should be properly initialized
  CHECK(hashtable_cache);
  auto candidate_ht = getCachedItem(key, *hashtable_cache);
  if (candidate_ht) {
    if (item_type == OVERLAPS_HT) {
      CHECK(candidate_ht->meta_info && candidate_ht->meta_info->overlaps_meta_info);
      CHECK(meta_info && meta_info->overlaps_meta_info);
      if (checkOverlapsHashtableBucketCompatability(
              *candidate_ht->meta_info->overlaps_meta_info,
              *meta_info->overlaps_meta_info)) {
        return true;
      }
    } else {
      return true;
    }
  }
  return false;
}

std::shared_ptr<HashTable> HashtableRecycler::getItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::optional<HashtableCacheMetaInfo> meta_info) const {
  if (key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto hashtable_cache = getCachedItemContainer(item_type, device_identifier);
  auto candidate_ht = getCachedItem(key, *hashtable_cache);
  if (candidate_ht) {
    candidate_ht->item_metric->incRefCount();
    VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Recycle item in a cache";
    return candidate_ht->cached_item;
  }
  return nullptr;
}

void HashtableRecycler::putItemToCache(QueryPlanHash key,
                                       std::shared_ptr<HashTable> item_ptr,
                                       CacheItemType item_type,
                                       DeviceIdentifier device_identifier,
                                       size_t item_size,
                                       size_t compute_time,
                                       std::optional<HashtableCacheMetaInfo> meta_info) {
  if (key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  std::lock_guard<std::mutex> lock(getCacheLock());
  if (!hasItemInCache(key, item_type, device_identifier, lock, meta_info)) {
    // check cache's space availability
    auto& metric_tracker = getMetricTracker(item_type);
    auto cache_status = metric_tracker.canAddItem(device_identifier, item_size);
    if (cache_status == CacheAvailability::UNAVAILABLE) {
      // hashtable is too large
      return;
    } else if (cache_status == CacheAvailability::AVAILABLE_AFTER_CLEANUP) {
      // we need to cleanup some cached hashtables to make a room to insert this hashtable
      // here we try to cache the new one anyway since we don't know the importance of
      // this hashtable yet and if it is not that frequently reused it is removed
      // in a near future
      auto required_size = metric_tracker.calculateRequiredSpaceForItemAddition(
          device_identifier, item_size);
      cleanupCacheForInsertion(item_type, device_identifier, required_size, lock);
    }
    // put hashtable's metric to metric tracker
    auto new_cache_metric_ptr = metric_tracker.putNewCacheItemMetric(
        key, device_identifier, item_size, compute_time);
    CHECK_EQ(item_size, new_cache_metric_ptr->getMemSize());
    metric_tracker.updateCurrentCacheSize(
        device_identifier, CacheUpdateAction::ADD, item_size);
    // put hashtable to cache
    VLOG(1) << "[" << DataRecyclerUtil::toStringCacheItemType(item_type) << ", "
            << DataRecyclerUtil::getDeviceIdentifierString(device_identifier)
            << "] Put item to cache";
    auto hashtable_cache = getCachedItemContainer(item_type, device_identifier);
    hashtable_cache->emplace_back(key, item_ptr, new_cache_metric_ptr, meta_info);
  }
  // this hashtable is already cached
  return;
}

void HashtableRecycler::removeItemFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    std::lock_guard<std::mutex>& lock,
    std::optional<HashtableCacheMetaInfo> meta_info) {
  if (key == EMPTY_HASHED_PLAN_DAG_KEY) {
    return;
  }
  auto& cache_metrics = getMetricTracker(item_type);
  // remove cached item from the cache
  auto cache_metric = cache_metrics.getCacheItemMetric(key, device_identifier);
  CHECK(cache_metric);
  auto hashtable_size = cache_metric->getMemSize();
  auto hashtable_container = getCachedItemContainer(item_type, device_identifier);
  auto filter = [key](auto const& item) { return item.key == key; };
  auto itr =
      std::find_if(hashtable_container->cbegin(), hashtable_container->cend(), filter);
  if (itr == hashtable_container->cend()) {
    return;
  } else {
    hashtable_container->erase(itr);
  }
  // remove cache metric
  cache_metrics.removeCacheItemMetric(key, device_identifier);
  // update current cache size
  cache_metrics.updateCurrentCacheSize(
      device_identifier, CacheUpdateAction::REMOVE, hashtable_size);
  return;
}

void HashtableRecycler::cleanupCacheForInsertion(
    CacheItemType item_type,
    DeviceIdentifier device_identifier,
    size_t required_size,
    std::lock_guard<std::mutex>& lock,
    std::optional<HashtableCacheMetaInfo> meta_info) {
  // sort the vector based on the importance of the cached items (by # referenced, size
  // and compute time) and then remove unimportant cached items
  int elimination_target_offset = 0;
  size_t removed_size = 0;
  auto& metric_tracker = getMetricTracker(item_type);
  auto actual_space_to_free = metric_tracker.getTotalCacheSize() / 2;
  if (!g_is_test_env && required_size < actual_space_to_free) {
    // remove enough items to avoid too frequent cache cleanup
    // we do not apply thin to test code since test scenarios are designed to
    // specific size of items and their caches
    required_size = actual_space_to_free;
  }
  metric_tracker.sortCacheInfoByQueryMetric(device_identifier);
  auto cached_item_metrics = metric_tracker.getCacheItemMetrics(device_identifier);
  sortCacheContainerByQueryMetric(item_type, device_identifier);

  // collect targets to eliminate
  for (auto& metric : cached_item_metrics) {
    auto target_size = metric->getMemSize();
    ++elimination_target_offset;
    removed_size += target_size;
    if (removed_size > required_size) {
      break;
    }
  }

  // eliminate targets in 1) cache container and 2) their metrics
  removeCachedItemFromBeginning(item_type, device_identifier, elimination_target_offset);
  metric_tracker.removeMetricFromBeginning(device_identifier, elimination_target_offset);

  // update the current cache size after this cleanup
  metric_tracker.updateCurrentCacheSize(
      device_identifier, CacheUpdateAction::REMOVE, removed_size);
}

void HashtableRecycler::clearCache() {
  std::lock_guard<std::mutex> lock(getCacheLock());
  for (auto& item_type : getCacheItemType()) {
    getMetricTracker(item_type).clearCacheMetricTracker();
    auto item_cache = getItemCache().find(item_type)->second;
    for (auto& kv : *item_cache) {
      kv.second->clear();
    }
  }
}

std::string HashtableRecycler::toString() const {
  std::ostringstream oss;
  oss << "A current status of the Hashtable Recycler:\n";
  for (auto& item_type : getCacheItemType()) {
    oss << "\t" << DataRecyclerUtil::toStringCacheItemType(item_type);
    auto& metric_tracker = getMetricTracker(item_type);
    oss << "\n\t# cached hashtables:\n";
    auto item_cache = getItemCache().find(item_type)->second;
    for (auto& cache_container : *item_cache) {
      oss << "\t\tDevice"
          << DataRecyclerUtil::getDeviceIdentifierString(cache_container.first)
          << ", # hashtables: " << cache_container.second->size() << "\n";
      for (auto& ht : *cache_container.second) {
        oss << "\t\t\tHT] " << ht.item_metric->toString() << "\n";
      }
    }
    oss << "\t" << metric_tracker.toString() << "\n";
  }
  return oss.str();
}

bool HashtableRecycler::checkOverlapsHashtableBucketCompatability(
    const OverlapsHashTableMetaInfo& candidate,
    const OverlapsHashTableMetaInfo& target) const {
  if (candidate.bucket_sizes.size() != target.bucket_sizes.size()) {
    return false;
  }
  for (size_t i = 0; i < candidate.bucket_sizes.size(); i++) {
    if (std::abs(target.bucket_sizes[i] - candidate.bucket_sizes[i]) > 1e-4) {
      return false;
    }
  }
  auto threshold_check =
      candidate.overlaps_bucket_threshold == target.overlaps_bucket_threshold;
  auto hashtable_size_check =
      candidate.overlaps_max_table_size_bytes == target.overlaps_max_table_size_bytes;
  return threshold_check && hashtable_size_check;
}

std::string HashtableRecycler::getJoinColumnInfoString(
    std::vector<const Analyzer::ColumnVar*>& inner_cols,
    std::vector<const Analyzer::ColumnVar*>& outer_cols,
    Executor* executor) {
  std::ostringstream oss;
  oss << executor->getQueryPlanDagCache().translateColVarsToInfoString(inner_cols, false);
  auto hash_table_cols_info = oss.str();
  oss << "|";
  oss << executor->getQueryPlanDagCache().translateColVarsToInfoString(outer_cols, false);
  return oss.str();
}

std::pair<QueryPlan, HashtableCacheMetaInfo> HashtableRecycler::getHashtableKeyString(
    const std::vector<InnerOuter>& inner_outer_pairs,
    const SQLOps op_type,
    const JoinType join_type,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    Executor* executor) {
  std::vector<const Analyzer::ColumnVar*> inner_cols_vec, outer_cols_vec;
  std::string inner_join_cols_info{""};
  for (auto& join_col_pair : inner_outer_pairs) {
    inner_cols_vec.push_back(join_col_pair.first);
    // extract inner join col's id
    // b/c when the inner col comes from a subquery's resulset,
    // table id / rte_index can be different even if we have the same
    // subquery's semantic, i.e., project col A from table T
    inner_join_cols_info +=
        concat(executor->getQueryPlanDagCache().getJoinColumnsInfoString(
                   join_col_pair.first, JoinColumnSide::kDirect, true),
               "|",
               ::toString(op_type),
               "|",
               ::toString(join_type),
               "|");
    auto outer_col_var = dynamic_cast<const Analyzer::ColumnVar*>(join_col_pair.second);
    if (outer_col_var) {
      outer_cols_vec.push_back(outer_col_var);
      if (join_col_pair.first->get_type_info().is_dict_encoded_string()) {
        // add comp param for dict encoded string
        inner_join_cols_info += outer_col_var->get_type_info().get_comp_param();
      }
    }
  }
  auto join_cols_info = getJoinColumnInfoString(inner_cols_vec, outer_cols_vec, executor);
  QueryPlan hashtable_access_path{EMPTY_QUERY_PLAN};
  HashtableCacheMetaInfo meta_info;
  auto it = hashtable_build_dag_map.find(join_cols_info);
  if (it != hashtable_build_dag_map.end()) {
    hashtable_access_path = it->second.second;
    hashtable_access_path += inner_join_cols_info;
    QueryPlanMetaInfo query_plan_meta_info;
    query_plan_meta_info.query_plan_dag = it->second.second;
    query_plan_meta_info.inner_col_info_string = inner_join_cols_info;
    HashtableCacheMetaInfo meta_info;
    meta_info.query_plan_meta_info = query_plan_meta_info;
    VLOG(2) << "Find hashtable access path for the hashjoin qual: " << join_cols_info
            << " -> " << hashtable_access_path;
  }
  return std::make_pair(hashtable_access_path, meta_info);
}

std::pair<QueryPlanHash, HashtableCacheMetaInfo> HashtableRecycler::getHashtableCacheKey(
    const std::vector<InnerOuter>& inner_outer_pairs,
    const SQLOps op_type,
    const JoinType join_type,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    Executor* executor) {
  auto hashtable_access_path = getHashtableKeyString(
      inner_outer_pairs, op_type, join_type, hashtable_build_dag_map, executor);
  return std::make_pair(boost::hash_value(hashtable_access_path.first),
                        hashtable_access_path.second);
}

std::tuple<QueryPlanHash,
           std::shared_ptr<HashTable>,
           std::optional<HashtableCacheMetaInfo>>
HashtableRecycler::getCachedHashtableWithoutCacheKey(std::set<size_t>& visited,
                                                     CacheItemType hash_table_type,
                                                     DeviceIdentifier device_identifier) {
  std::lock_guard<std::mutex> lock(getCacheLock());
  auto hashtable_cache = getCachedItemContainer(hash_table_type, device_identifier);
  for (auto& ht : *hashtable_cache) {
    if (!visited.count(ht.key)) {
      return std::make_tuple(ht.key, ht.cached_item, ht.meta_info);
    }
  }
  return std::make_tuple(EMPTY_HASHED_PLAN_DAG_KEY, nullptr, std::nullopt);
}
