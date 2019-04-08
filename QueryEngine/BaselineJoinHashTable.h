/*
 * Copyright 2017 MapD Technologies, Inc.
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
#ifndef QUERYENGINE_BASELINEJOINHASHTABLE_H
#define QUERYENGINE_BASELINEJOINHASHTABLE_H

#include "../Analyzer/Analyzer.h"
#include "../DataMgr/MemoryLevel.h"
#include "ColumnarResults.h"
#include "HashJoinRuntime.h"
#include "InputMetadata.h"
#include "JoinHashTableInterface.h"
#include "ResultRows.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <cstdint>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

class Executor;

// Representation for a hash table using the baseline layout: an open-addressing
// hash with a fill rate of 50%. It is used for equi-joins on multiple columns and
// on single sparse columns (with very wide range), typically big integer. As of
// now, such tuples must be unique within the inner table.
class BaselineJoinHashTable : public JoinHashTableInterface {
 public:
  static std::shared_ptr<BaselineJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const RelAlgExecutionUnit& ra_exe_unit,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      ColumnCacheMap& column_map,
      Executor* executor);

  static size_t getShardCountForCondition(const Analyzer::BinOper* condition,
                                          const RelAlgExecutionUnit& ra_exe_unit,
                                          const Executor* executor);

  int64_t getJoinHashBuffer(const ExecutorDeviceType device_type,
                            const int device_id) noexcept override;

  llvm::Value* codegenSlot(const CompilationOptions&, const size_t) override;

  HashJoinMatchingSet codegenMatchingSet(const CompilationOptions&,
                                         const size_t) override;

  int getInnerTableId() const noexcept override;

  int getInnerTableRteIdx() const noexcept override;

  JoinHashTableInterface::HashType getHashType() const noexcept override;

  static auto yieldCacheInvalidator() -> std::function<void()> {
    return []() -> void {
      std::lock_guard<std::mutex> guard(hash_table_cache_mutex_);
      hash_table_cache_.clear();
    };
  }

  virtual ~BaselineJoinHashTable() {}

 protected:
  BaselineJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                        const std::vector<InputTableInfo>& query_infos,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        const Data_Namespace::MemoryLevel memory_level,
                        const size_t entry_count,
                        ColumnCacheMap& column_map,
                        Executor* executor);

  static int getInnerTableId(const Analyzer::BinOper* condition,
                             const Executor* executor);

  virtual void reifyWithLayout(const int device_count,
                               const JoinHashTableInterface::HashType layout);

  struct ColumnsForDevice {
    const std::vector<JoinColumn> join_columns;
    const std::vector<JoinColumnTypeInfo> join_column_types;
    const std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
    const std::vector<JoinBucketInfo> join_buckets;
  };

  virtual ColumnsForDevice fetchColumnsForDevice(
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id);

  virtual std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<ColumnsForDevice>&) const;

  virtual size_t getKeyComponentWidth() const;

  virtual size_t getKeyComponentCount() const;

  virtual int initHashTableOnCpu(const std::vector<JoinColumn>& join_columns,
                                 const std::vector<JoinColumnTypeInfo>& join_column_types,
                                 const std::vector<JoinBucketInfo>& join_bucket_info,
                                 const JoinHashTableInterface::HashType layout);

  virtual int initHashTableOnGpu(const std::vector<JoinColumn>& join_columns,
                                 const std::vector<JoinColumnTypeInfo>& join_column_types,
                                 const std::vector<JoinBucketInfo>& join_bucket_info,
                                 const JoinHashTableInterface::HashType layout,
                                 const size_t key_component_width,
                                 const size_t key_component_count,
                                 const int device_id);

  virtual llvm::Value* codegenKey(const CompilationOptions&);

  std::pair<const int8_t*, size_t> getAllColumnFragments(
      const Analyzer::ColumnVar& hash_col,
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner);

  size_t shardCount() const;

  Data_Namespace::MemoryLevel getEffectiveMemoryLevel(
      const std::vector<InnerOuter>& inner_outer_pairs) const;

  struct CompositeKeyInfo {
    std::vector<const void*> sd_inner_proxy_per_key;
    std::vector<const void*> sd_outer_proxy_per_key;
    std::vector<ChunkKey> cache_key_chunks;  // used for the cache key
  };

  CompositeKeyInfo getCompositeKeyInfo(
      const std::vector<InnerOuter>& inner_outer_pairs) const;

  void reify(const int device_count);

  JoinColumn fetchColumn(const Analyzer::ColumnVar* inner_col,
                         const Data_Namespace::MemoryLevel& effective_memory_level,
                         const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
                         std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
                         const int device_id);

  void reifyForDevice(const ColumnsForDevice& columns_for_device,
                      const JoinHashTableInterface::HashType layout,
                      const int device_id);

  void checkHashJoinReplicationConstraint(const int table_id) const;

  int initHashTableForDevice(const std::vector<JoinColumn>& join_columns,
                             const std::vector<JoinColumnTypeInfo>& join_column_types,
                             const std::vector<JoinBucketInfo>& join_buckets,
                             const JoinHashTableInterface::HashType layout,
                             const Data_Namespace::MemoryLevel effective_memory_level,
                             const int device_id);

  llvm::Value* hashPtr(const size_t index);

  struct HashTableCacheKey {
    const size_t num_elements;
    const std::vector<ChunkKey> chunk_keys;
    const SQLOps optype;

    bool operator==(const struct HashTableCacheKey& that) const {
      return num_elements == that.num_elements && chunk_keys == that.chunk_keys &&
             optype == that.optype;
    }
  };

  void initHashTableOnCpuFromCache(const HashTableCacheKey&);

  void putHashTableOnCpuToCache(const HashTableCacheKey&);

  std::pair<ssize_t, size_t> getApproximateTupleCountFromCache(
      const HashTableCacheKey&) const;

  bool isBitwiseEq() const;

  void freeHashBufferMemory();
  void freeHashBufferGpuMemory();
  void freeHashBufferCpuMemory();

  const std::shared_ptr<Analyzer::BinOper> condition_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  size_t entry_count_;         // number of keys in the hash table
  size_t emitted_keys_count_;  // number of keys emitted across all rows
  Executor* executor_;
  const RelAlgExecutionUnit& ra_exe_unit_;
  ColumnCacheMap& column_cache_;
  std::shared_ptr<std::vector<int8_t>> cpu_hash_table_buff_;
  std::mutex cpu_hash_table_buff_mutex_;
#ifdef HAVE_CUDA
  std::vector<Data_Namespace::AbstractBuffer*> gpu_hash_table_buff_;
#endif
  typedef std::pair<const int8_t*, size_t> LinearizedColumn;
  typedef std::pair<int, int> LinearizedColumnCacheKey;
  std::map<LinearizedColumnCacheKey, LinearizedColumn> linearized_multifrag_columns_;
  std::mutex linearized_multifrag_column_mutex_;
  RowSetMemoryOwner linearized_multifrag_column_owner_;
  JoinHashTableInterface::HashType layout_;

  struct HashTableCacheValue {
    const std::shared_ptr<std::vector<int8_t>> buffer;
    const JoinHashTableInterface::HashType type;
    const size_t entry_count;
    const size_t emitted_keys_count;
  };

  static std::vector<std::pair<HashTableCacheKey, HashTableCacheValue>> hash_table_cache_;
  static std::mutex hash_table_cache_mutex_;

  static const int ERR_FAILED_TO_FETCH_COLUMN{-3};
  static const int ERR_FAILED_TO_JOIN_ON_VIRTUAL_COLUMN{-4};
};

class HashTypeCache {
 public:
  static void set(const std::vector<ChunkKey>& key,
                  const JoinHashTableInterface::HashType hash_type);

  static std::pair<JoinHashTableInterface::HashType, bool> get(
      const std::vector<ChunkKey>& key);

 private:
  static std::map<std::vector<ChunkKey>, JoinHashTableInterface::HashType>
      hash_type_cache_;
  static std::mutex hash_type_cache_mutex_;
};

#endif  // QUERYENGINE_BASELINEJOINHASHTABLE_H
