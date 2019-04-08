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

/*
 * @file    JoinHashTable.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_JOINHASHTABLE_H
#define QUERYENGINE_JOINHASHTABLE_H

#include "../Analyzer/Analyzer.h"
#include "../Catalog/Catalog.h"
#include "../Chunk/Chunk.h"
#include "ColumnarResults.h"
#include "ExpressionRange.h"
#include "InputDescriptors.h"
#include "InputMetadata.h"
#include "JoinHashTableInterface.h"
#include "ResultRows.h"
#include "ThrustAllocator.h"

#include <llvm/IR/Value.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>

class Executor;

class JoinHashTable : public JoinHashTableInterface {
 public:
  static std::shared_ptr<JoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
      const std::vector<InputTableInfo>& query_infos,
      const RelAlgExecutionUnit& ra_exe_unit,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor);

  int64_t getJoinHashBuffer(const ExecutorDeviceType device_type,
                            const int device_id) noexcept override {
    if (device_type == ExecutorDeviceType::CPU && !cpu_hash_table_buff_) {
      return 0;
    }
#ifdef HAVE_CUDA
    CHECK_LT(static_cast<size_t>(device_id), gpu_hash_table_buff_.size());
    if (device_type == ExecutorDeviceType::CPU) {
      return reinterpret_cast<int64_t>(&(*cpu_hash_table_buff_)[0]);
    } else {
      return gpu_hash_table_buff_[device_id]
                 ? reinterpret_cast<CUdeviceptr>(
                       gpu_hash_table_buff_[device_id]->getMemoryPtr())
                 : reinterpret_cast<CUdeviceptr>(nullptr);
    }
#else
    CHECK(device_type == ExecutorDeviceType::CPU);
    return reinterpret_cast<int64_t>(&(*cpu_hash_table_buff_)[0]);
#endif
  }

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

  static HashJoinMatchingSet codegenMatchingSet(
      const std::vector<llvm::Value*>& hash_join_idx_args_in,
      const bool is_sharded,
      const bool col_is_nullable,
      const bool is_bw_eq,
      const int64_t sub_buff_size,
      Executor* executor);

  static llvm::Value* codegenHashTableLoad(const size_t table_idx, Executor* executor);

  static auto yieldCacheInvalidator() -> std::function<void()> {
    return []() -> void {
      std::lock_guard<std::mutex> guard(join_hash_table_cache_mutex_);
      join_hash_table_cache_.clear();
    };
  }

  virtual ~JoinHashTable() {}

 private:
  JoinHashTable(const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
                const Analyzer::ColumnVar* col_var,
                const std::vector<InputTableInfo>& query_infos,
                const RelAlgExecutionUnit& ra_exe_unit,
                const Data_Namespace::MemoryLevel memory_level,
                const ExpressionRange& col_range,
                ColumnCacheMap& column_cache,
                Executor* executor,
                const int device_count)
      : qual_bin_oper_(qual_bin_oper)
      , col_var_(std::dynamic_pointer_cast<Analyzer::ColumnVar>(col_var->deep_copy()))
      , query_infos_(query_infos)
      , memory_level_(memory_level)
      , hash_type_(HashType::OneToOne)
      , hash_entry_count_(0)
      , col_range_(col_range)
      , executor_(executor)
      , ra_exe_unit_(ra_exe_unit)
      , column_cache_(column_cache)
      , device_count_(device_count) {
    CHECK(col_range.getType() == ExpressionRangeType::Integer);
  }

  std::pair<const int8_t*, size_t> getColumnFragment(
      const Analyzer::ColumnVar& hash_col,
      const Fragmenter_Namespace::FragmentInfo& fragment,
      const Data_Namespace::MemoryLevel effective_mem_lvl,
      const int device_id,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner);

  std::pair<const int8_t*, size_t> getAllColumnFragments(
      const Analyzer::ColumnVar& hash_col,
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner);

  ChunkKey genHashTableKey(
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      const Analyzer::Expr* outer_col,
      const Analyzer::ColumnVar* inner_col) const;

  void reify(const int device_count);
  void reifyOneToOneForDevice(
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id);
  void reifyOneToManyForDevice(
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id);
  void checkHashJoinReplicationConstraint(const int table_id) const;
  void initHashTableForDevice(
      const ChunkKey& chunk_key,
      const int8_t* col_buff,
      const size_t num_elements,
      const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
      const Data_Namespace::MemoryLevel effective_memory_level,
      const int device_id);
  void initOneToManyHashTable(
      const ChunkKey& chunk_key,
      const int8_t* col_buff,
      const size_t num_elements,
      const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
      const Data_Namespace::MemoryLevel effective_memory_level,
      const int device_id);
  void initHashTableOnCpuFromCache(
      const ChunkKey& chunk_key,
      const size_t num_elements,
      const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols);
  void putHashTableOnCpuToCache(
      const ChunkKey& chunk_key,
      const size_t num_elements,
      const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols);
  void initHashTableOnCpu(
      const int8_t* col_buff,
      const size_t num_elements,
      const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
      const size_t hash_entry_count,
      const int32_t hash_join_invalid_val);
  void initOneToManyHashTableOnCpu(
      const int8_t* col_buff,
      const size_t num_elements,
      const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
      const size_t hash_entry_count,
      const int32_t hash_join_invalid_val);

  const InputTableInfo& getInnerQueryInfo(const Analyzer::ColumnVar* inner_col) const;

  size_t shardCount() const;

  llvm::Value* codegenHashTableLoad(const size_t table_idx);

  std::vector<llvm::Value*> getHashJoinArgs(llvm::Value* hash_ptr,
                                            const Analyzer::Expr* key_col,
                                            const int shard_count,
                                            const CompilationOptions& co);

  std::pair<const int8_t*, size_t> fetchFragments(
      const Analyzer::ColumnVar* hash_col,
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragment_info,
      const Data_Namespace::MemoryLevel effective_memory_level,
      const int device_id,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      ThrustAllocator& dev_buff_owner);

  bool isBitwiseEq() const;

  void freeHashBufferMemory();
  void freeHashBufferGpuMemory();
  void freeHashBufferCpuMemory();

  std::shared_ptr<Analyzer::BinOper> qual_bin_oper_;
  std::shared_ptr<Analyzer::ColumnVar> col_var_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  HashType hash_type_;
  size_t hash_entry_count_;
  std::shared_ptr<std::vector<int32_t>> cpu_hash_table_buff_;
  std::mutex cpu_hash_table_buff_mutex_;
#ifdef HAVE_CUDA
  std::vector<Data_Namespace::AbstractBuffer*> gpu_hash_table_buff_;
  std::vector<Data_Namespace::AbstractBuffer*> gpu_hash_table_err_buff_;
#endif
  ExpressionRange col_range_;
  Executor* executor_;
  const RelAlgExecutionUnit& ra_exe_unit_;
  ColumnCacheMap& column_cache_;
  const int device_count_;
  std::pair<const int8_t*, size_t> linearized_multifrag_column_;
  std::mutex linearized_multifrag_column_mutex_;
  RowSetMemoryOwner linearized_multifrag_column_owner_;

  struct JoinHashTableCacheKey {
    const ExpressionRange col_range;
    const Analyzer::ColumnVar inner_col;
    const Analyzer::ColumnVar outer_col;
    const size_t num_elements;
    const ChunkKey chunk_key;
    const SQLOps optype;

    bool operator==(const struct JoinHashTableCacheKey& that) const {
      return col_range == that.col_range && inner_col == that.inner_col &&
             outer_col == that.outer_col && num_elements == that.num_elements &&
             chunk_key == that.chunk_key && optype == that.optype;
    }
  };

  static std::vector<
      std::pair<JoinHashTableCacheKey, std::shared_ptr<std::vector<int32_t>>>>
      join_hash_table_cache_;
  static std::mutex join_hash_table_cache_mutex_;
};

inline std::string get_table_name_by_id(const int table_id,
                                        const Catalog_Namespace::Catalog& cat) {
  if (table_id >= 1) {
    const auto td = cat.getMetadataForTable(table_id);
    CHECK(td);
    return td->tableName;
  }
  return "$TEMPORARY_TABLE" + std::to_string(-table_id);
}

// TODO(alex): Functions below need to be moved to a separate translation unit, they don't
// belong here.

size_t get_shard_count(const Analyzer::BinOper* join_condition,
                       const RelAlgExecutionUnit& ra_exe_unit,
                       const Executor* executor);

size_t get_shard_count(
    std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*> equi_pair,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Executor* executor);

bool needs_dictionary_translation(const Analyzer::ColumnVar* inner_col,
                                  const Analyzer::Expr* outer_col,
                                  const Executor* executor);

// Swap the columns if needed and make the inner column the first component.
InnerOuter normalize_column_pair(const Analyzer::Expr* lhs,
                                 const Analyzer::Expr* rhs,
                                 const Catalog_Namespace::Catalog& cat,
                                 const TemporaryTables* temporary_tables,
                                 const bool is_overlaps_join = false);

// Normalize each expression tuple
std::vector<InnerOuter> normalize_column_pairs(const Analyzer::BinOper* condition,
                                               const Catalog_Namespace::Catalog& cat,
                                               const TemporaryTables* temporary_tables);

std::deque<Fragmenter_Namespace::FragmentInfo> only_shards_for_device(
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    const int device_count);

const InputTableInfo& get_inner_query_info(
    const int inner_table_id,
    const std::vector<InputTableInfo>& query_infos);

size_t get_entries_per_device(const size_t total_entries,
                              const size_t shard_count,
                              const size_t device_count,
                              const Data_Namespace::MemoryLevel memory_level);

#endif  // QUERYENGINE_JOINHASHTABLE_H
