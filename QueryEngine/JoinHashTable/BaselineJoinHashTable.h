/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include "QueryEngine/DataRecycler/HashingSchemeRecycler.h"
#include "QueryEngine/DataRecycler/HashtableRecycler.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/JoinHashTable/BaselineHashTable.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"

class Executor;

using StrProxyTranslationMapsPtrsAndOffsets =
    std::pair<std::vector<const int32_t*>, std::vector<int32_t>>;

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
      const JoinType join_type,
      const HashType preferred_hash_type,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const TableIdToNodeMap& table_id_to_node_map);

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

  static void invalidateCache() {
    CHECK(hash_table_layout_cache_);
    hash_table_cache_->clearCache();

    CHECK(hash_table_cache_);
    hash_table_layout_cache_->clearCache();
  }

  static void markCachedItemAsDirty(size_t table_key) {
    CHECK(hash_table_cache_);
    CHECK(hash_table_layout_cache_);
    auto candidate_table_keys =
        hash_table_cache_->getMappedQueryPlanDagsWithTableKey(table_key);
    if (candidate_table_keys.has_value()) {
      hash_table_layout_cache_->markCachedItemAsDirty(
          table_key,
          *candidate_table_keys,
          CacheItemType::HT_HASHING_SCHEME,
          DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      hash_table_cache_->markCachedItemAsDirty(table_key,
                                               *candidate_table_keys,
                                               CacheItemType::BASELINE_HT,
                                               DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
    }
  }

  static HashtableRecycler* getHashTableCache() {
    CHECK(hash_table_cache_);
    return hash_table_cache_.get();
  }
  static HashingSchemeRecycler* getHashingSchemeCache() {
    CHECK(hash_table_layout_cache_);
    return hash_table_layout_cache_.get();
  }

  virtual ~BaselineJoinHashTable() {}

 protected:
  BaselineJoinHashTable(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const JoinType join_type,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const std::vector<InnerOuter>& inner_outer_pairs,
      const std::vector<InnerOuterStringOpInfos>& col_pairs_string_op_infos,
      const int device_count,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const TableIdToNodeMap& table_id_to_node_map);

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

  void copyCpuHashTableToGpu(std::shared_ptr<BaselineHashTable>& cpu_hash_table,
                             const int device_id,
                             Data_Namespace::DataMgr* data_mgr);

  virtual void reifyForDevice(const ColumnsForDevice& columns_for_device,
                              const HashType layout,
                              const int device_id,
                              const size_t entry_count,
                              const size_t emitted_keys_count,
                              const logger::QueryId,
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

  std::shared_ptr<HashTable> initHashTableOnCpuFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier);

  void putHashTableOnCpuToCache(QueryPlanHash key,
                                CacheItemType item_type,
                                std::shared_ptr<HashTable> hashtable_ptr,
                                DeviceIdentifier device_identifier,
                                size_t hashtable_building_time);

  bool isBitwiseEq() const override;

  ChunkKey genChunkKey(
      const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) const;

  struct AlternativeCacheKeyForBaselineHashJoin {
    std::vector<InnerOuter> inner_outer_pairs;
    std::vector<InnerOuterStringOpInfos> inner_outer_string_op_infos_pairs;
    const size_t num_elements;
    const SQLOps optype;
    const JoinType join_type;
    const ChunkKey chunk_key;
  };

  static QueryPlanHash getAlternativeCacheKey(
      AlternativeCacheKeyForBaselineHashJoin& info) {
    auto hash = boost::hash_value(::toString(info.optype));
    for (InnerOuter inner_outer : info.inner_outer_pairs) {
      auto inner_col = inner_outer.first;
      auto rhs_col_var = dynamic_cast<const Analyzer::ColumnVar*>(inner_outer.second);
      auto outer_col = rhs_col_var ? rhs_col_var : inner_col;
      boost::hash_combine(hash, inner_col->toString());
      if (inner_col->get_type_info().is_string()) {
        boost::hash_combine(hash, outer_col->toString());
      }
    }
    if (info.inner_outer_string_op_infos_pairs.size()) {
      boost::hash_combine(hash, ::toString(info.inner_outer_string_op_infos_pairs));
    }
    boost::hash_combine(hash, info.num_elements);
    boost::hash_combine(hash, info.join_type);
    return hash;
  }

  const std::shared_ptr<Analyzer::BinOper> condition_;
  const JoinType join_type_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  Executor* executor_;
  ColumnCacheMap& column_cache_;
  std::mutex cpu_hash_table_buff_mutex_;
  std::mutex str_proxy_translation_mutex_;
  std::vector<const StringDictionaryProxy::IdMap*> str_proxy_translation_maps_;

  std::vector<InnerOuter> inner_outer_pairs_;
  std::vector<InnerOuterStringOpInfos> inner_outer_string_op_infos_pairs_;
  const Catalog_Namespace::Catalog* catalog_;
  const int device_count_;
  mutable bool needs_dict_translation_;
  std::optional<HashType>
      layout_override_;  // allows us to use a 1:many hash table for many:many

  HashTableBuildDagMap hashtable_build_dag_map_;
  // per-device cache key to cover hash table for sharded table
  std::vector<QueryPlanHash> hashtable_cache_key_;
  HashtableCacheMetaInfo hashtable_cache_meta_info_;
  std::unordered_set<size_t> table_keys_;
  const TableIdToNodeMap table_id_to_node_map_;

  static std::unique_ptr<HashtableRecycler> hash_table_cache_;
  static std::unique_ptr<HashingSchemeRecycler> hash_table_layout_cache_;
};
