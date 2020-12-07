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

#pragma once

#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"  // HashTableCacheKey
#include "QueryEngine/JoinHashTable/HashJoin.h"

class OverlapsJoinHashTable : public HashJoin {
 public:
  OverlapsJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                        const std::vector<InputTableInfo>& query_infos,
                        const Data_Namespace::MemoryLevel memory_level,
                        HashType hash_layout_type,
                        const size_t entry_count,
                        ColumnCacheMap& column_cache,
                        Executor* executor,
                        const std::vector<InnerOuter>& inner_outer_pairs,
                        const int device_count)
      : condition_(condition)
      , query_infos_(query_infos)
      , memory_level_(memory_level)
      , executor_(executor)
      , column_cache_(column_cache)
      , inner_outer_pairs_(inner_outer_pairs)
      , device_count_(device_count) {
    CHECK_GT(device_count_, 0);
    hash_tables_for_device_.resize(std::max(device_count_, 1));
  }

  virtual ~OverlapsJoinHashTable() {}

  //! Make hash table from an in-flight SQL query's parse tree etc.
  static std::shared_ptr<OverlapsJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor);

  static auto getCacheInvalidator() -> std::function<void()> {
    VLOG(1) << "Invalidate " << auto_tuner_cache_.size() << " cached overlaps hashtable.";
    return []() -> void {
      std::lock_guard<std::mutex> guard(auto_tuner_cache_mutex_);
      auto_tuner_cache_.clear();
    };
  }

 protected:
  void reify(const HashType preferred_layout);

  void reifyForDevice(const ColumnsForDevice& columns_for_device,
                      const HashType layout,
                      const size_t entry_count,
                      const size_t emitted_keys_count,
                      const int device_id,
                      const logger::ThreadId parent_thread_id);

  void reifyWithLayout(const HashType layout);

  std::pair<size_t, size_t> calculateCounts(
      size_t shard_count,
      const Fragmenter_Namespace::TableInfo& query_info,
      std::vector<ColumnsForDevice>& columns_per_device);

  size_t calculateHashTableSize(size_t number_of_dimensions,
                                size_t emitted_keys_count,
                                size_t entry_count) const;

  ColumnsForDevice fetchColumnsForDevice(
      const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id,
      DeviceAllocator* dev_buff_owner);

  std::vector<JoinBucketInfo> computeBucketInfo(
      const std::vector<JoinColumn>& join_columns,
      const std::vector<JoinColumnTypeInfo>& join_column_types,
      const int device_id);

  std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<ColumnsForDevice>&) const;

  size_t getKeyComponentWidth() const;

  size_t getKeyComponentCount() const;

  HashType getHashType() const noexcept override {
    if (layout_override_) {
      return *layout_override_;
    }
    auto hash_table = getHashTableForDevice(0);
    CHECK(hash_table);
    return hash_table->getLayout();
  }

  Data_Namespace::MemoryLevel getMemoryLevel() const noexcept override {
    return memory_level_;
  }

  int getDeviceCount() const noexcept override { return device_count_; };

  void initHashTableOnCpu(const std::vector<JoinColumn>& join_columns,
                          const std::vector<JoinColumnTypeInfo>& join_column_types,
                          const std::vector<JoinBucketInfo>& join_bucket_info,
                          const HashType layout,
                          const size_t entry_count,
                          const size_t emitted_keys_count);

  HashJoinMatchingSet codegenMatchingSet(const CompilationOptions&,
                                         const size_t) override;

  std::string toString(const ExecutorDeviceType device_type,
                       const int device_id = 0,
                       bool raw = false) const override;

  DecodedJoinHashBufferSet toSet(const ExecutorDeviceType device_type,
                                 const int device_id) const override;

  llvm::Value* codegenSlot(const CompilationOptions&, const size_t) override {
    UNREACHABLE();  // not applicable for overlaps join
    return nullptr;
  }

  static std::map<HashTableCacheKey, double> auto_tuner_cache_;
  static std::mutex auto_tuner_cache_mutex_;

 private:
  size_t getEntryCount() const {
    auto hash_table = getHashTableForDevice(0);
    CHECK(hash_table);
    return hash_table->getEntryCount();
  }

  size_t getEmittedKeysCount() const {
    auto hash_table = getHashTableForDevice(0);
    CHECK(hash_table);
    return hash_table->getEmittedKeysCount();
  }

  size_t getComponentBufferSize() const noexcept override {
    CHECK(!hash_tables_for_device_.empty());
    auto hash_table = hash_tables_for_device_.front();
    CHECK(hash_table);
    return hash_table->getEntryCount() * sizeof(int32_t);
  }

  size_t shardCount() const {
    if (memory_level_ != Data_Namespace::GPU_LEVEL) {
      return 0;
    }
    return BaselineJoinHashTable::getShardCountForCondition(
        condition_.get(), executor_, inner_outer_pairs_);
  }

  Data_Namespace::MemoryLevel getEffectiveMemoryLevel(
      const std::vector<InnerOuter>& inner_outer_pairs) const {
    return memory_level_;
  }

  int getInnerTableId() const noexcept override;

  int getInnerTableRteIdx() const noexcept override {
    CHECK(!inner_outer_pairs_.empty());
    const auto first_inner_col = inner_outer_pairs_.front().first;
    return first_inner_col->get_rte_idx();
  }

  size_t getKeyBufferSize() const noexcept {
    const auto key_component_width = getKeyComponentWidth();
    CHECK(key_component_width == 4 || key_component_width == 8);
    const auto key_component_count = getKeyComponentCount();
    if (layoutRequiresAdditionalBuffers(getHashType())) {
      return getEntryCount() * key_component_count * key_component_width;
    } else {
      return getEntryCount() * (key_component_count + 1) * key_component_width;
    }
  }

  size_t offsetBufferOff() const noexcept override { return getKeyBufferSize(); }

  size_t countBufferOff() const noexcept override {
    if (layoutRequiresAdditionalBuffers(getHashType())) {
      return offsetBufferOff() + getComponentBufferSize();
    } else {
      return getKeyBufferSize();
    }
  }

  size_t payloadBufferOff() const noexcept override {
    if (layoutRequiresAdditionalBuffers(getHashType())) {
      return countBufferOff() + getComponentBufferSize();
    } else {
      return getKeyBufferSize();
    }
  }

  std::string getHashJoinType() const final { return "Overlaps"; }

  std::shared_ptr<HashTable> initHashTableOnCpuFromCache(const HashTableCacheKey& key);

  std::pair<std::optional<size_t>, size_t> getApproximateTupleCountFromCache(
      const HashTableCacheKey&) const;

  void putHashTableOnCpuToCache(const HashTableCacheKey& key,
                                std::shared_ptr<HashTable>& hash_table);

  void computeBucketSizes(std::vector<double>& bucket_sizes_for_dimension,
                          const JoinColumn& join_column,
                          const JoinColumnTypeInfo& join_column_type,
                          const std::vector<InnerOuter>& inner_outer_pairs);

  llvm::Value* codegenKey(const CompilationOptions&);
  std::vector<llvm::Value*> codegenManyKey(const CompilationOptions&);

  const std::shared_ptr<Analyzer::BinOper> condition_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;

  Executor* executor_;
  ColumnCacheMap& column_cache_;

  std::vector<InnerOuter> inner_outer_pairs_;
  const int device_count_;

  std::vector<double> bucket_sizes_for_dimension_;
  double overlaps_hashjoin_bucket_threshold_{0.1};

  std::optional<HashType>
      layout_override_;  // allows us to use a 1:many hash table for many:many

  using HashTableCacheValue = std::shared_ptr<HashTable>;
  static std::unique_ptr<HashTableCache<HashTableCacheKey, HashTableCacheValue>>
      hash_table_cache_;
};
