/*
 * Copyright 2018 OmniSci, Inc.
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

#ifndef QUERYENGINE_OVERLAPSHASHTABLE_H
#define QUERYENGINE_OVERLAPSHASHTABLE_H

#include "BaselineJoinHashTable.h"

class OverlapsJoinHashTable : public BaselineJoinHashTable {
 public:
  OverlapsJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                        const std::vector<InputTableInfo>& query_infos,
                        const Data_Namespace::MemoryLevel memory_level,
                        const size_t entry_count,
                        ColumnCacheMap& column_cache,
                        Executor* executor,
                        const std::vector<InnerOuter>& inner_outer_pairs,
                        const int device_count)
      : BaselineJoinHashTable(condition,
                              query_infos,
                              memory_level,
                              JoinHashTableInterface::HashType::OneToOne,
                              entry_count,
                              column_cache,
                              executor,
                              inner_outer_pairs,
                              device_count) {}

  ~OverlapsJoinHashTable() override {}

  //! Make hash table from an in-flight SQL query's parse tree etc.
  static std::shared_ptr<OverlapsJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor);

  static auto yieldCacheInvalidator() -> std::function<void()> {
    return []() -> void {
      std::lock_guard<std::mutex> guard(auto_tuner_cache_mutex_);
      auto_tuner_cache_.clear();
    };
  }

 protected:
  void reifyWithLayout(const JoinHashTableInterface::HashType layout) override;

  std::pair<size_t, size_t> calculateCounts(
      size_t shard_count,
      const Fragmenter_Namespace::TableInfo& query_info,
      std::vector<BaselineJoinHashTable::ColumnsForDevice>& columns_per_device);

  size_t calculateHashTableSize(size_t number_of_dimensions,
                                size_t emitted_keys_count,
                                size_t entry_count) const;

  ColumnsForDevice fetchColumnsForDevice(
      const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id,
      ThrustAllocator& dev_buff_owner) override;

  std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<ColumnsForDevice>&) const override;

  size_t getKeyComponentWidth() const override;

  size_t getKeyComponentCount() const override;

  int initHashTableOnCpu(const std::vector<JoinColumn>& join_columns,
                         const std::vector<JoinColumnTypeInfo>& join_column_types,
                         const std::vector<JoinBucketInfo>& join_bucket_info,
                         const JoinHashTableInterface::HashType layout) override;

  int initHashTableOnGpu(const std::vector<JoinColumn>& join_columns,
                         const std::vector<JoinColumnTypeInfo>& join_column_types,
                         const std::vector<JoinBucketInfo>& join_bucket_info,
                         const JoinHashTableInterface::HashType layout,
                         const size_t key_component_width,
                         const size_t key_component_count,
                         const int device_id) override;

  llvm::Value* codegenKey(const CompilationOptions&) override;

  static std::map<HashTableCacheKey, double> auto_tuner_cache_;
  static std::mutex auto_tuner_cache_mutex_;

 private:
  void computeBucketSizes(std::vector<double>& bucket_sizes_for_dimension,
                          const JoinColumn& join_column,
                          const JoinColumnTypeInfo& join_column_type,
                          const std::vector<InnerOuter>& inner_outer_pairs);

  std::vector<double> bucket_sizes_for_dimension_;
  double overlaps_hashjoin_bucket_threshold_;
};

#endif  // QUERYENGINE_OVERLAPSHASHTABLE_H
