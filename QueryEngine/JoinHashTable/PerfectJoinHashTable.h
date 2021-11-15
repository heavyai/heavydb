/*
 * Copyright 2017 OmniSci, Inc.
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

/*
 * @file    JoinHashTable.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 */

#pragma once

#include "Analyzer/Analyzer.h"
#include "DataMgr/Allocators/ThrustAllocator.h"
#include "DataMgr/Chunk/Chunk.h"
#include "QueryEngine/ColumnarResults.h"
#include "QueryEngine/DataRecycler/HashingSchemeRecycler.h"
#include "QueryEngine/DataRecycler/HashtableRecycler.h"
#include "QueryEngine/Descriptors/InputDescriptors.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/ExpressionRange.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JoinHashTable/PerfectHashTable.h"

#include <llvm/IR/Value.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>

struct HashEntryInfo;

class PerfectJoinHashTable : public HashJoin {
 public:
  //! Make hash table from an in-flight SQL query's parse tree etc.
  static std::shared_ptr<PerfectJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const JoinType join_type,
      const HashType preferred_hash_type,
      const int device_count,
      DataProvider* data_provider,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const TableIdToNodeMap& table_id_to_node_map);

  std::string toString(const ExecutorDeviceType device_type,
                       const int device_id = 0,
                       bool raw = false) const override;

  std::set<DecodedJoinHashBufferEntry> toSet(const ExecutorDeviceType device_type,
                                             const int device_id) const override;

  llvm::Value* codegenSlot(const CompilationOptions&, const size_t) override;

  HashJoinMatchingSet codegenMatchingSet(const CompilationOptions&,
                                         const size_t) override;

  int getInnerTableId() const noexcept override {
    return col_var_.get()->get_table_id();
  };

  int getInnerTableRteIdx() const noexcept override {
    return col_var_.get()->get_rte_idx();
  };

  HashType getHashType() const noexcept override { return hash_type_; }

  Data_Namespace::MemoryLevel getMemoryLevel() const noexcept override {
    return memory_level_;
  };

  int getDeviceCount() const noexcept override { return device_count_; };

  size_t offsetBufferOff() const noexcept override;

  size_t countBufferOff() const noexcept override;

  size_t payloadBufferOff() const noexcept override;

  std::string getHashJoinType() const final { return "Perfect"; }

  static HashtableRecycler* getHashTableCache() {
    CHECK(hash_table_cache_);
    return hash_table_cache_.get();
  }
  static HashingSchemeRecycler* getHashingSchemeCache() {
    CHECK(hash_table_layout_cache_);
    return hash_table_layout_cache_.get();
  }

  static auto getCacheInvalidator() -> std::function<void()> {
    CHECK(hash_table_cache_);
    CHECK(hash_table_layout_cache_);
    return []() -> void {
      auto layout_cache_invalidator = hash_table_layout_cache_->getCacheInvalidator();
      layout_cache_invalidator();

      auto main_cache_invalidator = hash_table_cache_->getCacheInvalidator();
      main_cache_invalidator();
    };
  }

  virtual ~PerfectJoinHashTable() {}

 private:
  // We don't want to create JoinHashTable for big ranges
  // with small number of valid entries. Therefore we
  // define the minimal load level (in percent).
  static constexpr size_t huge_join_hash_min_load_ = 10;

  // Equijoin API
  ColumnsForDevice fetchColumnsForDevice(
      const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id,
      DeviceAllocator* dev_buff_owner);

  void reifyForDevice(const ChunkKey& hash_table_key,
                      const ColumnsForDevice& columns_for_device,
                      const HashType layout,
                      const int device_id,
                      const logger::ThreadId parent_thread_id);

  int initHashTableForDevice(const ChunkKey& chunk_key,
                             const JoinColumn& join_column,
                             const InnerOuter& cols,
                             const HashType layout,
                             const Data_Namespace::MemoryLevel effective_memory_level,
                             const int device_id);

  Data_Namespace::MemoryLevel getEffectiveMemoryLevel(
      const std::vector<InnerOuter>& inner_outer_pairs) const;

  std::vector<InnerOuter> inner_outer_pairs_;

  PerfectJoinHashTable(const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
                       const Analyzer::ColumnVar* col_var,
                       const std::vector<InputTableInfo>& query_infos,
                       const Data_Namespace::MemoryLevel memory_level,
                       const JoinType join_type,
                       const HashType preferred_hash_type,
                       const ExpressionRange& col_range,
                       DataProvider* data_provider,
                       ColumnCacheMap& column_cache,
                       Executor* executor,
                       const int device_count,
                       QueryPlanHash hashtable_cache_key,
                       HashtableCacheMetaInfo hashtable_cache_meta_info,
                       const TableIdToNodeMap& table_id_to_node_map)
      : HashJoin(data_provider)
      , qual_bin_oper_(qual_bin_oper)
      , join_type_(join_type)
      , col_var_(std::dynamic_pointer_cast<Analyzer::ColumnVar>(col_var->deep_copy()))
      , query_infos_(query_infos)
      , memory_level_(memory_level)
      , hash_type_(preferred_hash_type)
      , col_range_(col_range)
      , executor_(executor)
      , column_cache_(column_cache)
      , device_count_(device_count)
      , needs_dict_translation_(false)
      , table_id_to_node_map_(table_id_to_node_map)
      , hashtable_cache_key_(hashtable_cache_key)
      , hashtable_cache_meta_info_(hashtable_cache_meta_info) {
    CHECK(col_range.getType() == ExpressionRangeType::Integer);
    CHECK_GT(device_count_, 0);
    hash_tables_for_device_.resize(device_count_);
  }

  ChunkKey genChunkKey(const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
                       const Analyzer::Expr* outer_col,
                       const Analyzer::ColumnVar* inner_col) const;

  void reify();
  std::shared_ptr<PerfectHashTable> initHashTableOnCpuFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier);
  void putHashTableOnCpuToCache(QueryPlanHash key,
                                CacheItemType item_type,
                                std::shared_ptr<PerfectHashTable> hashtable_ptr,
                                DeviceIdentifier device_identifier,
                                size_t hashtable_building_time);

  const InputTableInfo& getInnerQueryInfo(const Analyzer::ColumnVar* inner_col) const;

  size_t shardCount() const;

  llvm::Value* codegenHashTableLoad(const size_t table_idx);

  std::vector<llvm::Value*> getHashJoinArgs(llvm::Value* hash_ptr,
                                            const Analyzer::Expr* key_col,
                                            const CompilationOptions& co);

  bool isBitwiseEq() const override;

  size_t getComponentBufferSize() const noexcept override;

  HashTable* getHashTableForDevice(const size_t device_id) const;

  struct AlternativeCacheKeyForPerfectHashJoin {
    const ExpressionRange col_range;
    const Analyzer::ColumnVar* inner_col;
    const Analyzer::ColumnVar* outer_col;
    const ChunkKey chunk_key;
    const size_t num_elements;
    const SQLOps optype;
    const JoinType join_type;
  };

  static QueryPlanHash getAlternativeCacheKey(
      AlternativeCacheKeyForPerfectHashJoin& info) {
    auto hash = boost::hash_value(::toString(info.chunk_key));
    boost::hash_combine(hash, info.inner_col->toString());
    if (info.inner_col->get_type_info().is_string()) {
      boost::hash_combine(hash, info.outer_col->toString());
    }
    boost::hash_combine(hash, info.col_range.toString());
    boost::hash_combine(hash, info.num_elements);
    boost::hash_combine(hash, ::toString(info.optype));
    boost::hash_combine(hash, ::toString(info.join_type));
    return hash;
  }

  std::shared_ptr<Analyzer::BinOper> qual_bin_oper_;
  const JoinType join_type_;
  std::shared_ptr<Analyzer::ColumnVar> col_var_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  HashType hash_type_;

  std::mutex cpu_hash_table_buff_mutex_;
  ExpressionRange col_range_;
  Executor* executor_;
  ColumnCacheMap& column_cache_;
  const int device_count_;
  mutable bool needs_dict_translation_;
  const TableIdToNodeMap table_id_to_node_map_;
  QueryPlanHash hashtable_cache_key_;
  HashtableCacheMetaInfo hashtable_cache_meta_info_;

  static std::unique_ptr<HashtableRecycler> hash_table_cache_;
  static std::unique_ptr<HashingSchemeRecycler> hash_table_layout_cache_;
};

bool needs_dictionary_translation(const Analyzer::ColumnVar* inner_col,
                                  const Analyzer::Expr* outer_col,
                                  const Executor* executor);

const InputTableInfo& get_inner_query_info(
    const int inner_table_id,
    const std::vector<InputTableInfo>& query_infos);

size_t get_entries_per_device(const size_t total_entries,
                              const size_t shard_count,
                              const size_t device_count,
                              const Data_Namespace::MemoryLevel memory_level);
