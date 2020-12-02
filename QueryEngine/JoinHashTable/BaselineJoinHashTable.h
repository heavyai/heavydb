/*
 * Copyright 2020 OmniSci, Inc.
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

#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <cstdint>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "Analyzer/Analyzer.h"
#include "DataMgr/MemoryLevel.h"
#include "QueryEngine/ColumnarResults.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/JoinHashTable/BaselineHashTable.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"

class Executor;

struct CompositeKeyInfo {
  std::vector<const void*> sd_inner_proxy_per_key;
  std::vector<const void*> sd_outer_proxy_per_key;
  std::vector<ChunkKey> cache_key_chunks;  // used for the cache key
};

struct HashTableCacheKey {
  const size_t num_elements;
  const std::vector<ChunkKey> chunk_keys;
  const SQLOps optype;
  const boost::optional<double> overlaps_hashjoin_bucket_threshold;

  bool operator==(const struct HashTableCacheKey& that) const {
    bool oeq;
    if (overlaps_hashjoin_bucket_threshold && that.overlaps_hashjoin_bucket_threshold) {
      oeq = (std::abs(*overlaps_hashjoin_bucket_threshold -
                      *that.overlaps_hashjoin_bucket_threshold) <= 0.00000001);
    } else {
      oeq =
          (overlaps_hashjoin_bucket_threshold == that.overlaps_hashjoin_bucket_threshold);
    }
    return num_elements == that.num_elements && chunk_keys == that.chunk_keys &&
           optype == that.optype && oeq;
  }

  bool operator<(const struct HashTableCacheKey& that) const {
    bool oeq;
    if (overlaps_hashjoin_bucket_threshold && that.overlaps_hashjoin_bucket_threshold) {
      oeq = (std::abs(*overlaps_hashjoin_bucket_threshold -
                      *that.overlaps_hashjoin_bucket_threshold) <= 0.00000001);
    } else {
      oeq =
          (overlaps_hashjoin_bucket_threshold == that.overlaps_hashjoin_bucket_threshold);
    }
    return num_elements < that.num_elements && chunk_keys < that.chunk_keys &&
           optype < that.optype && !oeq &&
           overlaps_hashjoin_bucket_threshold < that.overlaps_hashjoin_bucket_threshold;
  }
};

class HashTypeCache {
 public:
  static void set(const std::vector<ChunkKey>& key, const HashJoin::HashType hash_type);

  static std::pair<HashJoin::HashType, bool> get(const std::vector<ChunkKey>& key);

  static void clear();

 private:
  static std::map<std::vector<ChunkKey>, HashJoin::HashType> hash_type_cache_;
  static std::mutex hash_type_cache_mutex_;
};

// Representation for a hash table using the baseline layout: an open-addressing
// hash with a fill rate of 50%. It is used for equi-joins on multiple columns and
// on single sparse columns (with very wide range), typically big integer. As of
// now, such tuples must be unique within the inner table.
class BaselineJoinHashTable : public HashJoin {
 public:
  //! Make hash table from an in-flight SQL query's parse tree etc.
  static std::shared_ptr<BaselineJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const HashType preferred_hash_type,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor);

  static size_t getShardCountForCondition(
      const Analyzer::BinOper* condition,
      const Executor* executor,
      const std::vector<InnerOuter>& inner_outer_pairs);

  int64_t getJoinHashBuffer(const ExecutorDeviceType device_type,
                            const int device_id) const noexcept override;

  size_t getJoinHashBufferSize(const ExecutorDeviceType device_type,
                               const int device_id) const noexcept override {
    auto hash_table = getHashTableForDevice(device_id);
    CHECK(hash_table);
    return hash_table->getHashTableBufferSize(device_type);
  }

  std::string toString(const ExecutorDeviceType device_type,
                       const int device_id = 0,
                       bool raw = false) const override;

  std::set<DecodedJoinHashBufferEntry> toSet(const ExecutorDeviceType device_type,
                                             const int device_id) const override;

  llvm::Value* codegenSlot(const CompilationOptions&, const size_t) override;

  HashJoinMatchingSet codegenMatchingSet(const CompilationOptions&,
                                         const size_t) override;

  int getInnerTableId() const noexcept override;

  int getInnerTableRteIdx() const noexcept override;

  HashJoin::HashType getHashType() const noexcept override;

  Data_Namespace::MemoryLevel getMemoryLevel() const noexcept override {
    return memory_level_;
  };

  int getDeviceCount() const noexcept override { return device_count_; };

  size_t offsetBufferOff() const noexcept override;

  size_t countBufferOff() const noexcept override;

  size_t payloadBufferOff() const noexcept override;

  static auto yieldCacheInvalidator() -> std::function<void()> {
    VLOG(1) << "Invalidate " << hash_table_cache_.size() << " cached baseline hashtable.";
    return []() -> void {
      std::lock_guard<std::mutex> guard(hash_table_cache_mutex_);
      hash_table_cache_.clear();
      HashTypeCache::clear();
    };
  }

  static int8_t* getCachedHashTable(size_t idx) {
    std::lock_guard<std::mutex> guard(hash_table_cache_mutex_);
    CHECK(!hash_table_cache_.empty());
    CHECK_LT(idx, hash_table_cache_.size());
    auto hash_tables_for_device = hash_table_cache_.at(idx).second;
    CHECK(hash_tables_for_device);
    return hash_tables_for_device->getCpuBuffer();
  }

  static size_t getEntryCntCachedHashTable(size_t idx) {
    std::lock_guard<std::mutex> guard(hash_table_cache_mutex_);
    CHECK(!hash_table_cache_.empty());
    CHECK_LT(idx, hash_table_cache_.size());
    auto hash_tables_for_device = hash_table_cache_.at(idx).second;
    CHECK(hash_tables_for_device);
    return hash_tables_for_device->getEntryCount();
  }

  static uint64_t getNumberOfCachedHashTables() {
    std::lock_guard<std::mutex> guard(hash_table_cache_mutex_);
    return hash_table_cache_.size();
  }

  virtual ~BaselineJoinHashTable() {}

 private:
  size_t getKeyBufferSize() const noexcept;
  size_t getComponentBufferSize() const noexcept;

 protected:
  BaselineJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                        const std::vector<InputTableInfo>& query_infos,
                        const Data_Namespace::MemoryLevel memory_level,
                        const size_t entry_count,
                        ColumnCacheMap& column_cache,
                        Executor* executor,
                        const std::vector<InnerOuter>& inner_outer_pairs,
                        const int device_count);

  static int getInnerTableId(const std::vector<InnerOuter>& inner_outer_pairs);

  virtual void reifyWithLayout(const HashJoin::HashType layout);

  virtual ColumnsForDevice fetchColumnsForDevice(
      const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id,
      DeviceAllocator* dev_buff_owner);

  virtual std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<ColumnsForDevice>&) const;

  virtual size_t getKeyComponentWidth() const;

  virtual size_t getKeyComponentCount() const;

  virtual llvm::Value* codegenKey(const CompilationOptions&);

  size_t shardCount() const;

  Data_Namespace::MemoryLevel getEffectiveMemoryLevel(
      const std::vector<InnerOuter>& inner_outer_pairs) const;

  CompositeKeyInfo getCompositeKeyInfo() const;

  void reify(const HashJoin::HashType preferred_layout);

  virtual void reifyForDevice(const ColumnsForDevice& columns_for_device,
                              const HashJoin::HashType layout,
                              const int device_id,
                              const logger::ThreadId parent_thread_id);

  void checkHashJoinReplicationConstraint(const int table_id) const;

  virtual int initHashTableForDevice(
      const std::vector<JoinColumn>& join_columns,
      const std::vector<JoinColumnTypeInfo>& join_column_types,
      const std::vector<JoinBucketInfo>& join_buckets,
      const HashJoin::HashType layout,
      const Data_Namespace::MemoryLevel effective_memory_level,
      const int device_id);

  llvm::Value* hashPtr(const size_t index);

  std::shared_ptr<BaselineHashTable> initHashTableOnCpuFromCache(
      const HashTableCacheKey&);

  void putHashTableOnCpuToCache(const HashTableCacheKey&,
                                std::shared_ptr<BaselineHashTable>& hash_table);

  std::pair<std::optional<size_t>, size_t> getApproximateTupleCountFromCache(
      const HashTableCacheKey&) const;

  bool isBitwiseEq() const;

  void freeHashBufferMemory();

  BaselineHashTable* getHashTableForDevice(const size_t device_id) const {
    CHECK_LT(device_id, hash_tables_for_device_.size());
    return hash_tables_for_device_[device_id].get();
  }

  const std::shared_ptr<Analyzer::BinOper> condition_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  size_t entry_count_;         // number of keys in the hash table
  size_t emitted_keys_count_;  // number of keys emitted across all rows
  Executor* executor_;
  ColumnCacheMap& column_cache_;
  std::vector<std::shared_ptr<BaselineHashTable>> hash_tables_for_device_;
  std::mutex cpu_hash_table_buff_mutex_;

  std::vector<InnerOuter> inner_outer_pairs_;
  const Catalog_Namespace::Catalog* catalog_;
  const int device_count_;
  unsigned block_size_;
  unsigned grid_size_;

  std::optional<HashJoin::HashType>
      layout_override_;  // allows us to use a 1:many hash table for many:many

  using HashTableCacheValue = std::shared_ptr<BaselineHashTable>;

  static std::vector<std::pair<HashTableCacheKey, HashTableCacheValue>> hash_table_cache_;
  static std::mutex hash_table_cache_mutex_;

  static const int ERR_FAILED_TO_FETCH_COLUMN{-3};
  static const int ERR_FAILED_TO_JOIN_ON_VIRTUAL_COLUMN{-4};
};
