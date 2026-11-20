/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Copyright 2026 HEAVY.AI, Inc.
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
#include "DataMgr/MemoryLevel.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/ExpressionRange.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"

#include <atomic>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <vector>

class HashtableRecycler;
class RankedBitmapHashTable;

namespace Data_Namespace {
class DataMgr;
}

#ifdef HAVE_CUDA
void allreduce_payload_free_ranked_bitmaps_for_test(
    const std::vector<int>& device_ids,
    const std::vector<std::shared_ptr<RankedBitmapHashTable>>& hash_tables,
    Data_Namespace::DataMgr* data_mgr,
    size_t expected_distinct_count);
#endif

class RankedBitmapJoinHashTable : public HashJoin {
 public:
  struct BuildSideFilter {
    std::shared_ptr<Analyzer::ColumnVar> column;
    bool has_lower_bound{false};
    bool lower_bound_inclusive{false};
    int64_t lower_bound{0};
    bool has_upper_bound{false};
    bool upper_bound_inclusive{false};
    int64_t upper_bound{0};
  };

  static std::shared_ptr<RankedBitmapJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const JoinType join_type,
      const std::set<int>& device_ids,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const TableIdToNodeMap& table_id_to_node_map,
      const RegisteredQueryHint& query_hint,
      const std::list<std::shared_ptr<Analyzer::Expr>>& build_side_quals = {},
      bool payload_free_unique_probe = false);

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

  HashType getHashType() const noexcept override {
    return hash_type_.load(std::memory_order_relaxed);
  }

  Data_Namespace::MemoryLevel getMemoryLevel() const noexcept override {
    return memory_level_;
  }

  bool usesBuildSideGlobalRowIds() const noexcept override {
    return !payload_free_unique_probe_;
  }

  size_t offsetBufferOff() const noexcept override { return 0; }

  size_t countBufferOff() const noexcept override { return 0; }

  size_t payloadBufferOff() const noexcept override { return 0; }

  std::string getHashJoinType() const override { return "RankedBitmap"; }

  bool isBuildSideQualifierPushedDown(const Analyzer::Expr* qual) const override;

  static HashtableRecycler* getHashTableCache();
  static void invalidateCache();
  static void markCachedItemAsDirty(size_t table_key);

 private:
  RankedBitmapJoinHashTable(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const Analyzer::ColumnVar* inner_col,
      const Analyzer::Expr* outer_expr,
      const ExpressionRange& col_range,
      const size_t bit_count,
      const size_t payload_count,
      const size_t table_bytes,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const JoinType join_type,
      const std::set<int>& device_ids,
      ColumnCacheMap& column_cache,
      Executor* executor,
      std::optional<BuildSideFilter> build_side_filter,
      std::vector<std::shared_ptr<Analyzer::Expr>> pushed_down_build_quals,
      bool allow_range_pruning,
      bool payload_free_unique_probe);

  void reify();

  void reifyForDevice(const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
                      const int device_id,
                      const logger::ThreadLocalIds parent_thread_local_ids,
                      bool defer_payload_free_rank_index);

  bool isBitwiseEq() const override { return false; }

  size_t getComponentBufferSize() const noexcept override { return 0; }

  std::shared_ptr<Analyzer::BinOper> condition_;
  std::shared_ptr<Analyzer::ColumnVar> inner_col_;
  std::shared_ptr<Analyzer::Expr> outer_expr_;
  ExpressionRange col_range_;
  size_t bit_count_;
  size_t payload_count_;
  size_t table_bytes_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;
  const JoinType join_type_;
  const std::set<int> device_ids_;
  ColumnCacheMap& column_cache_;
  Executor* executor_;
  std::optional<BuildSideFilter> build_side_filter_;
  std::vector<std::shared_ptr<Analyzer::Expr>> pushed_down_build_quals_;
  const bool allow_range_pruning_;
  const bool payload_free_unique_probe_;
  std::atomic<HashType> hash_type_{HashType::OneToOne};

  static std::unique_ptr<HashtableRecycler> hash_table_cache_;
};
