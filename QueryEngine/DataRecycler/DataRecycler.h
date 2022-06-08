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

#include "Analyzer/Analyzer.h"
#include "QueryEngine/ColumnarResults.h"
#include "QueryEngine/Descriptors/InputDescriptors.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/JoinHashTable/HashTable.h"
#include "QueryEngine/RelAlgExecutionUnit.h"
#include "QueryEngine/ResultSet.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/misc.h"

#include <boost/functional/hash.hpp>

#include <unordered_map>

struct EMPTY_META_INFO {};

// Item type that we try to recycle
enum CacheItemType {
  PERFECT_HT = 0,           // Perfect hashtable
  BASELINE_HT,              // Baseline hashtable
  HT_HASHING_SCHEME,        // Hashtable layout
  BASELINE_HT_APPROX_CARD,  // Approximated cardinality for baseline hashtable
  // TODO (yoonmin): support the following items for recycling
  // ROW_RS,             Row-wise resultset
  // COUNTALL_CARD_EST,  Cardinality of query result
  // NDV_CARD_EST,       # Non-distinct value
  // FILTER_SEL          Selectivity of (push-downed) filter node
  NUM_CACHE_ITEM_TYPE
};

// given item to be cached, it represents whether the item can be cached when considering
// various size limitation
enum CacheAvailability {
  AVAILABLE,                // item can be cached as is
  AVAILABLE_AFTER_CLEANUP,  // item can be cached after removing already cached items
  UNAVAILABLE               // item cannot be cached due to size limitation
};

enum CacheUpdateAction { ADD, REMOVE };

// the order of enum values affects how we remove cached items when
// new item wants to be cached but there is not enough space to keep them
// regarding `REF_COUNT`, it represents how many times a cached item is referenced during
// its lifetime to numerically estimate the usefulness of this cached item
// (not to measure exact # reference count at time T as std::shared_ptr does)
enum CacheMetricType { REF_COUNT = 0, MEM_SIZE, COMPUTE_TIME, NUM_METRIC_TYPE };

// per query plan DAG metric
class CacheItemMetric {
 public:
  CacheItemMetric(QueryPlanHash query_plan_hash, size_t compute_time, size_t mem_size)
      : query_plan_hash_(query_plan_hash), metrics_({0, mem_size, compute_time}) {}

  QueryPlanHash getQueryPlanHash() const { return query_plan_hash_; }

  void incRefCount() { ++metrics_[CacheMetricType::REF_COUNT]; }

  size_t getRefCount() const { return metrics_[CacheMetricType::REF_COUNT]; }

  size_t getComputeTime() const { return metrics_[CacheMetricType::COMPUTE_TIME]; }

  size_t getMemSize() const { return metrics_[CacheMetricType::MEM_SIZE]; }

  const std::array<size_t, CacheMetricType::NUM_METRIC_TYPE>& getMetrics() const {
    return metrics_;
  }

  void setComputeTime(size_t compute_time) {
    metrics_[CacheMetricType::COMPUTE_TIME] = compute_time;
  }

  void setMemSize(const size_t mem_size) {
    metrics_[CacheMetricType::MEM_SIZE] = mem_size;
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "Query plan hash: " << query_plan_hash_
        << ", compute_time: " << metrics_[CacheMetricType::COMPUTE_TIME]
        << ", mem_size: " << metrics_[CacheMetricType::MEM_SIZE]
        << ", ref_count: " << metrics_[CacheMetricType::REF_COUNT];
    return oss.str();
  }

 private:
  const QueryPlanHash query_plan_hash_;
  std::array<size_t, CacheMetricType::NUM_METRIC_TYPE> metrics_;
};

// 0 = CPU, 1 ~ N : GPU-1 ~ GPU-N
using DeviceIdentifier = size_t;
using CacheSizeMap = std::unordered_map<DeviceIdentifier, size_t>;
using CacheMetricInfoMap =
    std::unordered_map<DeviceIdentifier, std::vector<std::shared_ptr<CacheItemMetric>>>;

class DataRecyclerUtil {
 public:
  // need to add more constants if necessary: ROW_RS, COUNTALL_CARD_EST, NDV_CARD_EST,
  // FILTER_SEL, ...
  static constexpr auto cache_item_type_str =
      shared::string_view_array("Perfect Join Hashtable",
                                "Baseline Join Hashtable",
                                "Hashing Scheme for Join Hashtable",
                                "Baseline Join Hashtable's Approximated Cardinality");
  static std::string_view toStringCacheItemType(CacheItemType item_type) {
    static_assert(cache_item_type_str.size() == NUM_CACHE_ITEM_TYPE);
    return cache_item_type_str[item_type];
  }

  static constexpr DeviceIdentifier CPU_DEVICE_IDENTIFIER = 0;

  static std::string getDeviceIdentifierString(DeviceIdentifier device_identifier) {
    std::string device_type = device_identifier == CPU_DEVICE_IDENTIFIER ? "CPU" : "GPU-";
    return device_identifier != CPU_DEVICE_IDENTIFIER
               ? device_type.append(std::to_string(device_identifier))
               : device_type;
  }
};

// contain information regarding 1) per-cache item metric: perfect ht-1, perfect ht-2,
// baseline ht-1, ... and 2) per-type size in current: perfect-ht cache size, baseline-ht
// cache size, overlaps-ht cache size, ...
class CacheMetricTracker {
 public:
  CacheMetricTracker(CacheItemType cache_item_type,
                     size_t total_cache_size,
                     size_t max_cache_item_size,
                     int num_gpus = 0)
      : item_type_(cache_item_type)
      , total_cache_size_(total_cache_size)
      , max_cache_item_size_(max_cache_item_size) {
    // initialize cache metrics for each device: CPU, GPU0, GPU1, ...
    // Currently we only consider maintaining our cache in CPU-memory
    for (int gpu_device_identifier = num_gpus; gpu_device_identifier >= 1;
         --gpu_device_identifier) {
      cache_metrics_.emplace(gpu_device_identifier,
                             std::vector<std::shared_ptr<CacheItemMetric>>());
      current_cache_size_in_bytes_.emplace(gpu_device_identifier, 0);
    }
    cache_metrics_.emplace(DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                           std::vector<std::shared_ptr<CacheItemMetric>>());
    current_cache_size_in_bytes_.emplace(DataRecyclerUtil::CPU_DEVICE_IDENTIFIER, 0);

    if (total_cache_size_ < 1024 * 1024 * 256) {
      LOG(INFO) << "The total cache size of "
                << DataRecyclerUtil::toStringCacheItemType(cache_item_type)
                << " is set too low, so we suggest raising it larger than 256MB";
    }

    if (max_cache_item_size < 1024 * 1024 * 10) {
      LOG(INFO)
          << "The maximum item size of "
          << DataRecyclerUtil::toStringCacheItemType(cache_item_type)
          << " that can be cached is set too low, we suggest raising it larger than 10MB";
    }
    if (max_cache_item_size > total_cache_size_) {
      LOG(INFO) << "The maximum item size of "
                << DataRecyclerUtil::toStringCacheItemType(cache_item_type)
                << " is set larger than its total cache size, so we force to set the "
                   "maximum item size as equal to the total cache size";
      max_cache_item_size = total_cache_size_;
    }
  }

  static inline CacheMetricInfoMap::mapped_type::const_iterator getCacheItemMetricItr(
      QueryPlanHash key,
      CacheMetricInfoMap::mapped_type const& metrics) {
    auto same_hash = [key](auto itr) { return itr->getQueryPlanHash() == key; };
    return std::find_if(metrics.cbegin(), metrics.cend(), same_hash);
  }

  static inline std::shared_ptr<CacheItemMetric> getCacheItemMetricImpl(
      QueryPlanHash key,
      CacheMetricInfoMap::mapped_type const& metrics) {
    auto itr = getCacheItemMetricItr(key, metrics);
    return itr == metrics.cend() ? nullptr : *itr;
  }

  std::vector<std::shared_ptr<CacheItemMetric>>& getCacheItemMetrics(
      DeviceIdentifier device_identifier) {
    auto itr = cache_metrics_.find(device_identifier);
    CHECK(itr != cache_metrics_.end());
    return itr->second;
  }

  std::shared_ptr<CacheItemMetric> getCacheItemMetric(
      QueryPlanHash key,
      DeviceIdentifier device_identifier) const {
    auto itr = cache_metrics_.find(device_identifier);
    return itr == cache_metrics_.cend() ? nullptr
                                        : getCacheItemMetricImpl(key, itr->second);
  }

  void setCurrentCacheSize(DeviceIdentifier device_identifier, size_t bytes) {
    if (bytes > total_cache_size_) {
      return;
    }
    auto itr = current_cache_size_in_bytes_.find(device_identifier);
    CHECK(itr != current_cache_size_in_bytes_.end());
    itr->second = bytes;
  }

  std::optional<size_t> getCurrentCacheSize(DeviceIdentifier key) const {
    auto same_hash = [key](auto itr) { return itr.first == key; };
    auto itr = std::find_if(current_cache_size_in_bytes_.cbegin(),
                            current_cache_size_in_bytes_.cend(),
                            same_hash);
    return itr == current_cache_size_in_bytes_.cend() ? std::nullopt
                                                      : std::make_optional(itr->second);
  }

  std::shared_ptr<CacheItemMetric> putNewCacheItemMetric(
      QueryPlanHash key,
      DeviceIdentifier device_identifier,
      size_t mem_size,
      size_t compute_time) {
    auto itr = cache_metrics_.find(device_identifier);
    CHECK(itr != cache_metrics_.end());
    if (auto cached_metric = getCacheItemMetricImpl(key, itr->second)) {
      return cached_metric;
    }
    auto cache_metric = std::make_shared<CacheItemMetric>(key, compute_time, mem_size);
    // we add the item to cache after we create it during query runtime
    // so it is used at least once
    cache_metric->incRefCount();
    return itr->second.emplace_back(std::move(cache_metric));
  }

  void removeCacheItemMetric(QueryPlanHash key, DeviceIdentifier device_identifier) {
    auto& cache_metrics = getCacheItemMetrics(device_identifier);
    auto itr = getCacheItemMetricItr(key, cache_metrics);
    if (itr != cache_metrics.cend()) {
      cache_metrics.erase(itr);
    }
  }

  void removeMetricFromBeginning(DeviceIdentifier device_identifier, int offset) {
    auto metrics = getCacheItemMetrics(device_identifier);
    metrics.erase(metrics.begin(), metrics.begin() + offset);
  }

  size_t calculateRequiredSpaceForItemAddition(DeviceIdentifier device_identifier,
                                               size_t item_size) const {
    auto it = current_cache_size_in_bytes_.find(device_identifier);
    CHECK(it != current_cache_size_in_bytes_.end());
    auto rem = total_cache_size_ - it->second;
    CHECK_GT(item_size, rem);
    return item_size - rem;
  }

  void clearCacheMetricTracker() {
    for (auto& kv : current_cache_size_in_bytes_) {
      auto cache_item_metrics = getCacheItemMetrics(kv.first);
      VLOG(1) << "Clear cache of " << DataRecyclerUtil::toStringCacheItemType(item_type_)
              << " from device [" << kv.first
              << "] (# cached items: " << cache_item_metrics.size() << ", " << kv.second
              << " bytes)";
      updateCurrentCacheSize(kv.first, CacheUpdateAction::REMOVE, kv.second);
      CHECK_EQ(getCurrentCacheSize(kv.first).value(), 0u);
    }
    for (auto& kv : cache_metrics_) {
      kv.second.clear();
    }
  }

  CacheAvailability canAddItem(DeviceIdentifier device_identifier,
                               size_t item_size) const {
    if (item_size > max_cache_item_size_) {
      return CacheAvailability::UNAVAILABLE;
    }
    auto current_cache_size = getCurrentCacheSize(device_identifier);
    CHECK(current_cache_size.has_value());
    if (*current_cache_size > total_cache_size_) {
      return CacheAvailability::UNAVAILABLE;
    }
    auto cache_size_after_addition = *current_cache_size + item_size;
    if (cache_size_after_addition > total_cache_size_) {
      return CacheAvailability::AVAILABLE_AFTER_CLEANUP;
    }
    return CacheAvailability::AVAILABLE;
  }

  void updateCurrentCacheSize(DeviceIdentifier device_identifier,
                              CacheUpdateAction action,
                              size_t size) {
    auto current_cache_size = getCurrentCacheSize(device_identifier);
    CHECK(current_cache_size.has_value());
    if (action == CacheUpdateAction::ADD) {
      setCurrentCacheSize(device_identifier, current_cache_size.value() + size);
    } else {
      CHECK_EQ(action, CacheUpdateAction::REMOVE);
      CHECK_LE(size, *current_cache_size);
      setCurrentCacheSize(device_identifier, current_cache_size.value() - size);
    }
  }

  void sortCacheInfoByQueryMetric(DeviceIdentifier device_identifier) {
    auto& metric_cache = getCacheItemMetrics(device_identifier);
    std::sort(metric_cache.begin(),
              metric_cache.end(),
              [](const std::shared_ptr<CacheItemMetric>& left,
                 const std::shared_ptr<CacheItemMetric>& right) {
                auto& elem1_metrics = left->getMetrics();
                auto& elem2_metrics = right->getMetrics();
                for (size_t i = 0; i < CacheMetricType::NUM_METRIC_TYPE; ++i) {
                  if (elem1_metrics[i] != elem2_metrics[i]) {
                    return elem1_metrics[i] < elem2_metrics[i];
                  }
                }
                return false;
              });
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "Current memory consumption of caches for each device:\n";
    for (auto& kv : current_cache_size_in_bytes_) {
      oss << "\t\tDevice " << kv.first << " : " << kv.second << " bytes\n";
    }
    return oss.str();
  }

  size_t getTotalCacheSize() const { return total_cache_size_; }
  size_t getMaxCacheItemSize() const { return max_cache_item_size_; }
  void setTotalCacheSize(size_t new_total_cache_size) {
    if (new_total_cache_size > 0) {
      total_cache_size_ = new_total_cache_size;
    }
  }
  void setMaxCacheItemSize(size_t new_max_cache_item_size) {
    if (new_max_cache_item_size > 0) {
      max_cache_item_size_ = new_max_cache_item_size;
    }
  }

 private:
  CacheItemType item_type_;
  size_t total_cache_size_;
  size_t max_cache_item_size_;
  // metadata of cached item that belongs to a cache of a specific device
  // 1) ref_count: how many times this cached item is recycled
  // 2) memory_usage: the size of cached item in bytes
  // 3) compute_time: an elapsed time to generate this cached item
  CacheMetricInfoMap cache_metrics_;

  // the total amount of currently cached data per device
  CacheSizeMap current_cache_size_in_bytes_;
};

template <typename CACHED_ITEM_TYPE, typename META_INFO_TYPE>
struct CachedItem {
  CachedItem(QueryPlanHash hashed_plan,
             CACHED_ITEM_TYPE item,
             std::shared_ptr<CacheItemMetric> item_metric_ptr,
             std::optional<META_INFO_TYPE> metadata = std::nullopt)
      : key(hashed_plan)
      , cached_item(item)
      , item_metric(item_metric_ptr)
      , meta_info(metadata) {}
  QueryPlanHash key;
  CACHED_ITEM_TYPE cached_item;
  std::shared_ptr<CacheItemMetric> item_metric;
  std::optional<META_INFO_TYPE> meta_info;
};

// A main class of data recycler
// note that some tests which directly accesses APIs for update/modify/delete
// (meta)data may need to disable data recycler explicitly before running test suites
// to make test scenarios as expected
// i.e., UpdelStorageTest that calls fragmenter's updateColumn API
template <typename CACHED_ITEM_TYPE, typename META_INFO_TYPE>
class DataRecycler {
 public:
  using CachedItemContainer = std::vector<CachedItem<CACHED_ITEM_TYPE, META_INFO_TYPE>>;
  using PerDeviceCacheItemContainer =
      std::unordered_map<DeviceIdentifier, std::shared_ptr<CachedItemContainer>>;
  using PerTypeCacheItemContainer =
      std::unordered_map<CacheItemType, std::shared_ptr<PerDeviceCacheItemContainer>>;
  using PerTypeCacheMetricTracker = std::unordered_map<CacheItemType, CacheMetricTracker>;

  DataRecycler(const std::vector<CacheItemType>& item_types,
               size_t total_cache_size,
               size_t max_item_size,
               int num_gpus) {
    for (auto& item_type : item_types) {
      cache_item_types_.insert(item_type);
      metric_tracker_.emplace(
          item_type,
          CacheMetricTracker(item_type, total_cache_size, max_item_size, num_gpus));
      auto item_container = std::make_shared<PerDeviceCacheItemContainer>();
      for (int gpu_device_identifier = num_gpus; gpu_device_identifier >= 1;
           --gpu_device_identifier) {
        item_container->emplace(gpu_device_identifier,
                                std::make_shared<CachedItemContainer>());
      }
      item_container->emplace(DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                              std::make_shared<CachedItemContainer>());
      cached_items_container_.emplace(item_type, item_container);
    }
  }

  virtual ~DataRecycler() = default;

  virtual CACHED_ITEM_TYPE getItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::optional<META_INFO_TYPE> meta_info = std::nullopt) const = 0;

  virtual void putItemToCache(QueryPlanHash key,
                              CACHED_ITEM_TYPE item_ptr,
                              CacheItemType item_type,
                              DeviceIdentifier device_identifier,
                              size_t item_size,
                              size_t compute_time,
                              std::optional<META_INFO_TYPE> meta_info = std::nullopt) = 0;

  virtual void initCache() = 0;

  virtual void clearCache() = 0;

  virtual std::string toString() const = 0;

  std::shared_ptr<CachedItemContainer> getCachedItemContainer(
      CacheItemType item_type,
      DeviceIdentifier device_identifier) const {
    auto item_type_container_itr = cached_items_container_.find(item_type);
    if (item_type_container_itr != cached_items_container_.end()) {
      auto device_type_container_itr =
          item_type_container_itr->second->find(device_identifier);
      return device_type_container_itr != item_type_container_itr->second->end()
                 ? device_type_container_itr->second
                 : nullptr;
    }
    return nullptr;
  }

  std::optional<CachedItem<CACHED_ITEM_TYPE, META_INFO_TYPE>> getCachedItem(
      QueryPlanHash key,
      CachedItemContainer& m) const {
    for (auto& candidate : m) {
      if (candidate.key == key) {
        return candidate;
      }
    }
    return std::nullopt;
  }

  size_t getCurrentNumCachedItems(CacheItemType item_type,
                                  DeviceIdentifier device_identifier) const {
    std::lock_guard<std::mutex> lock(cache_lock_);
    auto container = getCachedItemContainer(item_type, device_identifier);
    return container ? container->size() : 0;
  }

  size_t getCurrentCacheSizeForDevice(CacheItemType item_type,
                                      DeviceIdentifier device_identifier) const {
    std::lock_guard<std::mutex> lock(cache_lock_);
    auto metric_tracker = getMetricTracker(item_type);
    auto current_size_opt = metric_tracker.getCurrentCacheSize(device_identifier);
    return current_size_opt ? current_size_opt.value() : 0;
  }

  std::shared_ptr<CacheItemMetric> getCachedItemMetric(CacheItemType item_type,
                                                       DeviceIdentifier device_identifier,
                                                       QueryPlanHash key) const {
    std::lock_guard<std::mutex> lock(cache_lock_);
    auto cache_metric_tracker = getMetricTracker(item_type);
    return cache_metric_tracker.getCacheItemMetric(key, device_identifier);
  }

  void setTotalCacheSize(CacheItemType item_type, size_t new_total_cache_size) {
    if (new_total_cache_size > 0) {
      std::lock_guard<std::mutex> lock(cache_lock_);
      getMetricTracker(item_type).setTotalCacheSize(new_total_cache_size);
    }
  }

  void setMaxCacheItemSize(CacheItemType item_type, size_t new_max_cache_item_size) {
    if (new_max_cache_item_size > 0) {
      std::lock_guard<std::mutex> lock(cache_lock_);
      getMetricTracker(item_type).setMaxCacheItemSize(new_max_cache_item_size);
    }
  }

  std::function<void()> getCacheInvalidator() {
    return [this]() -> void { clearCache(); };
  }

 protected:
  void removeCachedItemFromBeginning(CacheItemType item_type,
                                     DeviceIdentifier device_identifier,
                                     int offset) {
    // it removes cached items located from `idx 0` to `offset`
    // so, call this function after sorting the cached items container vec
    // and we should call this function under the proper locking scheme
    auto container = getCachedItemContainer(item_type, device_identifier);
    CHECK(container);
    container->erase(container->begin(), container->begin() + offset);
  }

  void sortCacheContainerByQueryMetric(CacheItemType item_type,
                                       DeviceIdentifier device_identifier) {
    // should call this function under the proper locking scheme
    auto container = getCachedItemContainer(item_type, device_identifier);
    CHECK(container);
    std::sort(container->begin(),
              container->end(),
              [](const CachedItem<CACHED_ITEM_TYPE, META_INFO_TYPE>& left,
                 const CachedItem<CACHED_ITEM_TYPE, META_INFO_TYPE>& right) {
                auto& left_metrics = left.item_metric->getMetrics();
                auto& right_metrics = right.item_metric->getMetrics();
                for (size_t i = 0; i < CacheMetricType::NUM_METRIC_TYPE; ++i) {
                  if (left_metrics[i] != right_metrics[i]) {
                    return left_metrics[i] < right_metrics[i];
                  }
                }
                return false;
              });
  }

  std::mutex& getCacheLock() const { return cache_lock_; }

  CacheMetricTracker& getMetricTracker(CacheItemType item_type) {
    auto metric_iter = metric_tracker_.find(item_type);
    CHECK(metric_iter != metric_tracker_.end());
    return metric_iter->second;
  }

  CacheMetricTracker const& getMetricTracker(CacheItemType item_type) const {
    return const_cast<DataRecycler*>(this)->getMetricTracker(item_type);
  }

  std::unordered_set<CacheItemType> const& getCacheItemType() const {
    return cache_item_types_;
  }

  PerTypeCacheItemContainer const& getItemCache() const {
    return cached_items_container_;
  }

 private:
  // internally called under the proper locking scheme
  virtual bool hasItemInCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<META_INFO_TYPE> meta_info = std::nullopt) const = 0;

  // internally called under the proper locking scheme
  virtual void removeItemFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      std::lock_guard<std::mutex>& lock,
      std::optional<META_INFO_TYPE> meta_info = std::nullopt) = 0;

  // internally called under the proper locking scheme
  virtual void cleanupCacheForInsertion(
      CacheItemType item_type,
      DeviceIdentifier device_identifier,
      size_t required_size,
      std::lock_guard<std::mutex>& lock,
      std::optional<META_INFO_TYPE> meta_info = std::nullopt) = 0;

  // a set of cache item type that this recycler supports
  std::unordered_set<CacheItemType> cache_item_types_;

  // cache metric tracker
  PerTypeCacheMetricTracker metric_tracker_;

  // per-device cached item containers for each cached item type
  PerTypeCacheItemContainer cached_items_container_;

  mutable std::mutex cache_lock_;
};
