/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analyzer/Analyzer.h"
#include "DataMgr/MemoryLevel.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/ExpressionRange.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"

#include <memory>
#include <set>

class HashtableRecycler;

class BitmapJoinHashTable : public HashJoin {
 public:
  static std::shared_ptr<BitmapJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const JoinType join_type,
      const std::set<int>& device_ids,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const TableIdToNodeMap& table_id_to_node_map);

  std::string toString(const ExecutorDeviceType device_type,
                       const int device_id = 0,
                       bool raw = false) const override;

  std::set<DecodedJoinHashBufferEntry> toSet(const ExecutorDeviceType,
                                             const int) const override {
    return {};
  }

  llvm::Value* codegenSlot(const CompilationOptions&, const size_t) override;

  HashJoinMatchingSet codegenMatchingSet(const CompilationOptions&,
                                         const size_t) override;

  shared::TableKey getInnerTableId() const noexcept override {
    return inner_col_->getTableKey();
  }

  int getInnerTableRteIdx() const noexcept override { return inner_col_->get_rte_idx(); }

  HashType getHashType() const noexcept override { return HashType::OneToOne; }

  Data_Namespace::MemoryLevel getMemoryLevel() const noexcept override {
    return memory_level_;
  }

  size_t offsetBufferOff() const noexcept override { return 0; }

  size_t countBufferOff() const noexcept override { return 0; }

  size_t payloadBufferOff() const noexcept override { return 0; }

  std::string getHashJoinType() const override { return "Bitmap"; }

  static HashtableRecycler* getHashTableCache();
  static void invalidateCache();
  static void markCachedItemAsDirty(size_t table_key);

 private:
  BitmapJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                      const Analyzer::ColumnVar* inner_col,
                      const Analyzer::Expr* outer_expr,
                      const ExpressionRange& col_range,
                      const size_t bit_count,
                      const size_t bitmap_bytes,
                      const std::vector<InputTableInfo>& query_infos,
                      const Data_Namespace::MemoryLevel memory_level,
                      const JoinType join_type,
                      const std::set<int>& device_ids,
                      ColumnCacheMap& column_cache,
                      Executor* executor);

  void reify();

  void reifyForDevice(const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
                      const int device_id,
                      const logger::ThreadLocalIds parent_thread_local_ids);

  bool isBitwiseEq() const override { return false; }

  size_t getComponentBufferSize() const noexcept override { return 0; }

  std::shared_ptr<Analyzer::BinOper> condition_;
  std::shared_ptr<Analyzer::ColumnVar> inner_col_;
  std::shared_ptr<Analyzer::Expr> outer_expr_;
  ExpressionRange col_range_;
  size_t bit_count_;
  size_t bitmap_bytes_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  const JoinType join_type_;
  const std::set<int> device_ids_;
  ColumnCacheMap& column_cache_;
  Executor* executor_;

  static std::unique_ptr<HashtableRecycler> hash_table_cache_;
};
