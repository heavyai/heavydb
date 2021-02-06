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

#include "QueryEngine/JoinHashTable/BaselineHashTable.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JoinHashTable/HashTableCache.h"

struct OverlapsHashTableCacheKey {
  const size_t num_elements;
  const std::vector<ChunkKey> chunk_keys;
  const SQLOps optype;
  const std::vector<double> bucket_sizes;

  bool operator==(const struct OverlapsHashTableCacheKey& that) const {
    if (bucket_sizes.size() != that.bucket_sizes.size()) {
      return false;
    }
    for (size_t i = 0; i < bucket_sizes.size(); i++) {
      // bucket sizes within 10^-4 are considered close enough
      if (std::abs(bucket_sizes[i] - that.bucket_sizes[i]) > 1e-4) {
        return false;
      }
    }
    return num_elements == that.num_elements && chunk_keys == that.chunk_keys &&
           optype == that.optype;
  }

  OverlapsHashTableCacheKey(const size_t num_elements,
                            const std::vector<ChunkKey>& chunk_keys,
                            const SQLOps& optype,
                            const std::vector<double> bucket_sizes)
      : num_elements(num_elements)
      , chunk_keys(chunk_keys)
      , optype(optype)
      , bucket_sizes(bucket_sizes) {}

  // "copy" constructor
  OverlapsHashTableCacheKey(const HashTableCacheKey& that,
                            const std::vector<double>& bucket_sizes)
      : num_elements(that.num_elements)
      , chunk_keys(that.chunk_keys)
      , optype(that.optype)
      , bucket_sizes(bucket_sizes) {}
};

template <class K, class V>
class OverlapsHashTableCache : public HashTableCache<K, V> {
 public:
  std::optional<std::pair<K, V>> getWithKey(const K& key) {
    std::lock_guard<std::mutex> guard(this->mutex_);
    for (const auto& kv : this->contents_) {
      if (kv.first == key) {
        return kv;
      }
    }
    return std::nullopt;
  }
};

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
    query_hint_ = QueryHint::defaults();
  }

  virtual ~OverlapsJoinHashTable() {}

  //! Make hash table from an in-flight SQL query's parse tree etc.
  static std::shared_ptr<OverlapsJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const QueryHint& query_hint);

  static auto getCacheInvalidator() -> std::function<void()> {
    return []() -> void {
      CHECK(auto_tuner_cache_);
      auto auto_tuner_cache_invalidator = auto_tuner_cache_->getCacheInvalidator();
      auto_tuner_cache_invalidator();

      CHECK(hash_table_cache_);
      auto main_cache_invalidator = hash_table_cache_->getCacheInvalidator();
      main_cache_invalidator();
    };
  }

  static size_t getCombinedHashTableCacheSize() {
    // for unit tests
    CHECK(hash_table_cache_ && auto_tuner_cache_);
    return hash_table_cache_->getNumberOfCachedHashTables() +
           auto_tuner_cache_->getNumberOfCachedHashTables();
  }

 protected:
  void reify(const HashType preferred_layout);

  void reifyWithLayout(const HashType layout);

  void reifyImpl(std::vector<ColumnsForDevice>& columns_per_device,
                 const Fragmenter_Namespace::TableInfo& query_info,
                 const HashType layout,
                 const size_t shard_count,
                 const size_t entry_count,
                 const size_t emitted_keys_count);

  void reifyForDevice(const ColumnsForDevice& columns_for_device,
                      const HashType layout,
                      const size_t entry_count,
                      const size_t emitted_keys_count,
                      const int device_id,
                      const logger::ThreadId parent_thread_id);

  size_t calculateHashTableSize(size_t number_of_dimensions,
                                size_t emitted_keys_count,
                                size_t entry_count) const;

  ColumnsForDevice fetchColumnsForDevice(
      const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id,
      DeviceAllocator* dev_buff_owner);

  // returns entry_count, emitted_keys_count
  std::pair<size_t, size_t> approximateTupleCount(const std::vector<ColumnsForDevice>&);

  // returns entry_count, emitted_keys_count
  std::pair<size_t, size_t> computeHashTableCounts(
      const size_t shard_count,
      std::vector<ColumnsForDevice>& columns_per_device);

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

  std::shared_ptr<BaselineHashTable> copyCpuHashTableToGpu(
      std::shared_ptr<BaselineHashTable>&& cpu_hash_table,
      const HashType layout,
      const size_t entry_count,
      const size_t emitted_keys_count,
      const size_t device_id);
#endif  // HAVE_CUDA

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

  const QueryHint& getRegisteredQueryHint() { return query_hint_; }

  void registerQueryHint(const QueryHint& query_hint) { query_hint_ = query_hint; }

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
      const std::vector<InnerOuter>& inner_outer_pairs) const;

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

  std::shared_ptr<HashTable> initHashTableOnCpuFromCache(
      const OverlapsHashTableCacheKey& key);

  std::optional<std::pair<size_t, size_t>> getApproximateTupleCountFromCache(
      const OverlapsHashTableCacheKey&);

  void putHashTableOnCpuToCache(const OverlapsHashTableCacheKey& key,
                                std::shared_ptr<HashTable> hash_table);

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

  std::optional<HashType>
      layout_override_;  // allows us to use a 1:many hash table for many:many

  std::mutex cpu_hash_table_buff_mutex_;

  using HashTableCacheValue = std::shared_ptr<HashTable>;
  // includes bucket threshold
  static std::unique_ptr<
      OverlapsHashTableCache<OverlapsHashTableCacheKey, HashTableCacheValue>>
      hash_table_cache_;
  // skips bucket threshold
  using BucketThreshold = double;
  using BucketSizes = std::vector<double>;
  static std::unique_ptr<
      HashTableCache<HashTableCacheKey, std::pair<BucketThreshold, BucketSizes>>>
      auto_tuner_cache_;

  QueryHint query_hint_;
};
