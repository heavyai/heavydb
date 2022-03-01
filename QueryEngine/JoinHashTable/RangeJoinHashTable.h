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

#include "QueryEngine/JoinHashTable/OverlapsJoinHashTable.h"

class RangeJoinHashTable final : public OverlapsJoinHashTable {
 public:
  RangeJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                     const JoinType join_type,
                     const Analyzer::RangeOper* range_expr,
                     std::shared_ptr<Analyzer::ColumnVar> inner_col_expr,
                     const std::vector<InputTableInfo>& query_infos,
                     const Data_Namespace::MemoryLevel memory_level,
                     ColumnCacheMap& column_cache,
                     Executor* executor,
                     const std::vector<InnerOuter>& inner_outer_pairs,
                     const int device_count,
                     QueryPlan query_plan_dag,
                     HashtableCacheMetaInfo hashtable_cache_meta_info,
                     const HashTableBuildDagMap& hashtable_build_dag_map,
                     const TableIdToNodeMap& table_id_to_node_map)
      : OverlapsJoinHashTable(condition,
                              join_type,
                              query_infos,
                              memory_level,
                              column_cache,
                              executor,
                              inner_outer_pairs,
                              device_count,
                              query_plan_dag,
                              hashtable_cache_meta_info,
                              table_id_to_node_map)
      , range_expr_(range_expr)
      , inner_col_expr_(std::move(inner_col_expr)) {}

  ~RangeJoinHashTable() override = default;

  static std::shared_ptr<RangeJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const Analyzer::RangeOper* range_expr,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const JoinType join_type,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const RegisteredQueryHint& query_hint,
      const TableIdToNodeMap& table_id_to_node_map);

 protected:
  void reifyWithLayout(const HashType layout) override;

  void reifyForDevice(const ColumnsForDevice& columns_for_device,
                      const HashType layout,
                      const size_t entry_count,
                      const size_t emitted_keys_count,
                      const int device_id,
                      const logger::ThreadId parent_thread_id);

  std::shared_ptr<BaselineHashTable> initHashTableOnCpu(
      const std::vector<JoinColumn>& join_columns,
      const std::vector<JoinColumnTypeInfo>& join_column_types,
      const std::vector<JoinBucketInfo>& join_bucket_info,
      const HashType layout,
      const size_t entry_count,
      const size_t emitted_keys_count);

#ifdef HAVE_CUDA
  std::shared_ptr<BaselineHashTable> initHashTableOnGpu(
      const std::vector<JoinColumn>& join_columns,
      const std::vector<JoinColumnTypeInfo>& join_column_types,
      const std::vector<JoinBucketInfo>& join_bucket_info,
      const HashType layout,
      const size_t entry_count,
      const size_t emitted_keys_count,
      const size_t device_id);
#endif

  HashType getHashType() const noexcept override { return HashType::OneToMany; }

  std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<double>& inverse_bucket_sizes_for_dimension,
      std::vector<ColumnsForDevice>& columns_per_device,
      const size_t chosen_max_hashtable_size,
      const double chosen_bucket_threshold) override;

  std::pair<size_t, size_t> computeRangeHashTableCounts(
      const size_t shard_count,
      std::vector<ColumnsForDevice>& columns_per_device);

 public:
  llvm::Value* codegenKey(const CompilationOptions& co, llvm::Value* offset);

  HashJoinMatchingSet codegenMatchingSetWithOffset(const CompilationOptions&,
                                                   const size_t,
                                                   llvm::Value*);

 private:
  const Analyzer::RangeOper* range_expr_;
  std::shared_ptr<Analyzer::ColumnVar> inner_col_expr_;
  const double bucket_threshold_{std::numeric_limits<double>::max()};
  const size_t max_hashtable_size_{std::numeric_limits<size_t>::max()};
  HashtableCacheMetaInfo overlaps_hashtable_cache_meta_info_;
};
