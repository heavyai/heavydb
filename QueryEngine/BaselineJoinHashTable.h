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

#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <cstdint>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

class Executor;

// Representation for a hash table using the baseline layout: an open-addressing
// hash with a fill rate of 50%. It is used for equi-joins on multiple columns and
// on single sparse columns (with very wide range), typically big integer. As of
// now, such tuples must be unique within the inner table.
class BaselineJoinHashTable : public JoinHashTableInterface {
 public:
  static std::shared_ptr<BaselineJoinHashTable> getInstance(const std::shared_ptr<Analyzer::BinOper> condition,
                                                            const std::vector<InputTableInfo>& query_infos,
                                                            const RelAlgExecutionUnit& ra_exe_unit,
                                                            const Data_Namespace::MemoryLevel memory_level,
                                                            const int device_count,
                                                            const std::unordered_set<int>& skip_tables,
                                                            ColumnCacheMap& column_map,
                                                            Executor* executor);

  int64_t getJoinHashBuffer(const ExecutorDeviceType device_type, const int device_id) noexcept override;

  llvm::Value* codegenSlotIsValid(const CompilationOptions&, const size_t) override;

  llvm::Value* codegenSlot(const CompilationOptions&, const size_t) override;

  HashJoinMatchingSet codegenMatchingSet(const CompilationOptions&, const size_t) override;

  int getInnerTableId() const noexcept override;

  int getInnerTableRteIdx() const noexcept override;

  JoinHashTableInterface::HashType getHashType() const noexcept override;

 private:
  BaselineJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                        const std::vector<InputTableInfo>& query_infos,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        const Data_Namespace::MemoryLevel memory_level,
                        const size_t entry_count,
                        ColumnCacheMap& column_map,
                        Executor* executor);

  static int getInnerTableId(const Analyzer::BinOper* condition, const Executor* executor);

  std::pair<const int8_t*, size_t> getAllColumnFragments(
      const Analyzer::ColumnVar& hash_col,
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner);

  size_t shardCount() const;

  size_t computeShardCount() const;

  int reify(const int device_count);

  int reifyWithLayout(const int device_count, const JoinHashTableInterface::HashType layout);

  struct ColumnsForDevice {
    const std::vector<JoinColumn> join_columns;
    const std::vector<JoinColumnTypeInfo> join_column_types;
    const std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
    const int err;
  };

  size_t approximateTupleCount(const std::vector<ColumnsForDevice>&) const;

  ColumnsForDevice fetchColumnsForDevice(const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
                                         const int device_id);

  int reifyForDevice(const ColumnsForDevice& columns_for_device,
                     const JoinHashTableInterface::HashType layout,
                     const int device_id);

  void checkHashJoinReplicationConstraint(const int table_id) const;

  int initHashTableForDevice(const std::vector<JoinColumn>& join_columns,
                             const std::vector<JoinColumnTypeInfo>& join_column_types,
                             const JoinHashTableInterface::HashType layout,
                             const Data_Namespace::MemoryLevel effective_memory_level,
                             const int device_id);

  int initHashTableOnCpu(const std::vector<JoinColumn>& join_columns,
                         const std::vector<JoinColumnTypeInfo>& join_column_types,
                         const JoinHashTableInterface::HashType layout);

  int initHashTableOnGpu(const std::vector<JoinColumn>& join_columns,
                         const std::vector<JoinColumnTypeInfo>& join_column_types,
                         const JoinHashTableInterface::HashType layout,
                         const size_t key_component_width,
                         const size_t key_component_count,
                         const int device_id);

  llvm::Value* hashPtr(const size_t index);

  llvm::Value* codegenOneToManySlot(const CompilationOptions& co,
                                    const size_t index,
                                    llvm::Value* key_buff_lv,
                                    const size_t key_component_count,
                                    const size_t key_component_width);

  llvm::Value* codegenKey(const CompilationOptions&);

  struct HashTableCacheKey {
    const size_t num_elements;
    const std::vector<ChunkKey> chunk_keys;
    const SQLOps optype;

    bool operator==(const struct HashTableCacheKey& that) const {
      return num_elements == that.num_elements && chunk_keys == that.chunk_keys && optype == that.optype;
    }
  };

  void initHashTableOnCpuFromCache(const HashTableCacheKey&);

  void putHashTableOnCpuToCache(const HashTableCacheKey&);

  ssize_t getApproximateTupleCountFromCache(const HashTableCacheKey&) const;

  bool isBitwiseEq() const;

  const std::shared_ptr<Analyzer::BinOper> condition_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  size_t entry_count_;
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
  };

  static std::vector<std::pair<HashTableCacheKey, HashTableCacheValue>> hash_table_cache_;
  static std::mutex hash_table_cache_mutex_;

  static const int ERR_FAILED_TO_FETCH_COLUMN{-3};
  static const int ERR_FAILED_TO_JOIN_ON_VIRTUAL_COLUMN{-4};
};

class HashTypeCache {
 public:
  static void set(const std::vector<ChunkKey>& key, const JoinHashTableInterface::HashType hash_type);

  static std::pair<JoinHashTableInterface::HashType, bool> get(const std::vector<ChunkKey>& key);

 private:
  static std::map<std::vector<ChunkKey>, JoinHashTableInterface::HashType> hash_type_cache_;
  static std::mutex hash_type_cache_mutex_;
};

// TODO(alex): Should be unified with get_shard_count, doesn't belong here.

size_t get_baseline_shard_count(const Analyzer::BinOper* join_condition,
                                const RelAlgExecutionUnit& ra_exe_unit,
                                const Executor* executor);

#endif  // QUERYENGINE_BASELINEJOINHASHTABLE_H
