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
#include "QueryEngine/JoinHashTable/HashTableCache.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"

class Executor;

struct HashTableCacheKey {
  const size_t num_elements;
  const std::vector<ChunkKey> chunk_keys;
  const SQLOps optype;

  bool operator==(const struct HashTableCacheKey& that) const {
    return num_elements == that.num_elements && chunk_keys == that.chunk_keys &&
           optype == that.optype;
  }
};

class HashTypeCache {
 public:
  static void set(const std::vector<ChunkKey>& key, const HashType hash_type);

  static std::pair<HashType, bool> get(const std::vector<ChunkKey>& key);

  static void clear();

 private:
  static std::map<std::vector<ChunkKey>, HashType> hash_type_cache_;
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

  HashType getHashType() const noexcept override;

  Data_Namespace::MemoryLevel getMemoryLevel() const noexcept override {
    return memory_level_;
  };

  int getDeviceCount() const noexcept override { return device_count_; };

  size_t offsetBufferOff() const noexcept override;

  size_t countBufferOff() const noexcept override;

  size_t payloadBufferOff() const noexcept override;

  std::string getHashJoinType() const final { return "Baseline"; }

  static auto getCacheInvalidator() -> std::function<void()> {
    return []() -> void {
      // TODO: make hash type cache part of the main cache
      CHECK(hash_table_cache_);
      hash_table_cache_->clear();
      HashTypeCache::clear();
    };
  }

  static auto* getHashTableCache() {
    CHECK(hash_table_cache_);
    return hash_table_cache_.get();
  }

  virtual ~BaselineJoinHashTable() {}

 protected:
  BaselineJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                        const std::vector<InputTableInfo>& query_infos,
                        const Data_Namespace::MemoryLevel memory_level,
                        ColumnCacheMap& column_cache,
                        Executor* executor,
                        const std::vector<InnerOuter>& inner_outer_pairs,
                        const int device_count);

  size_t getComponentBufferSize() const noexcept override;

  size_t getKeyBufferSize() const noexcept;

  static int getInnerTableId(const std::vector<InnerOuter>& inner_outer_pairs);

  virtual void reifyWithLayout(const HashType layout);

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

  void reify(const HashType preferred_layout);

  virtual void reifyForDevice(const ColumnsForDevice& columns_for_device,
                              const HashType layout,
                              const int device_id,
                              const size_t entry_count,
                              const size_t emitted_keys_count,
                              const logger::ThreadId parent_thread_id);

  virtual int initHashTableForDevice(
      const std::vector<JoinColumn>& join_columns,
      const std::vector<JoinColumnTypeInfo>& join_column_types,
      const std::vector<JoinBucketInfo>& join_buckets,
      const HashType layout,
      const Data_Namespace::MemoryLevel effective_memory_level,
      const size_t entry_count,
      const size_t emitted_keys_count,
      const int device_id);

  llvm::Value* hashPtr(const size_t index);

  std::shared_ptr<HashTable> initHashTableOnCpuFromCache(const HashTableCacheKey&);

  void putHashTableOnCpuToCache(const HashTableCacheKey&,
                                std::shared_ptr<HashTable>& hash_table);

  std::pair<std::optional<size_t>, size_t> getApproximateTupleCountFromCache(
      const HashTableCacheKey&) const;

  bool isBitwiseEq() const;

  const std::shared_ptr<Analyzer::BinOper> condition_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  Executor* executor_;
  ColumnCacheMap& column_cache_;
  std::mutex cpu_hash_table_buff_mutex_;

  std::vector<InnerOuter> inner_outer_pairs_;
  const Catalog_Namespace::Catalog* catalog_;
  const int device_count_;

  std::optional<HashType>
      layout_override_;  // allows us to use a 1:many hash table for many:many

  using HashTableCacheValue = std::shared_ptr<HashTable>;
  static std::unique_ptr<HashTableCache<HashTableCacheKey, HashTableCacheValue>>
      hash_table_cache_;
};
