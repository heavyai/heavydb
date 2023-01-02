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

#include "QueryEngine/DataRecycler/OverlapsTuningParamRecycler.h"
#include "QueryEngine/JoinHashTable/BaselineHashTable.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"

class OverlapsJoinHashTable : public HashJoin {
 public:
  OverlapsJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                        const JoinType join_type,
                        const std::vector<InputTableInfo>& query_infos,
                        const Data_Namespace::MemoryLevel memory_level,
                        ColumnCacheMap& column_cache,
                        Executor* executor,
                        const std::vector<InnerOuter>& inner_outer_pairs,
                        const int device_count,
                        const RegisteredQueryHint& query_hints,
                        const HashTableBuildDagMap& hashtable_build_dag_map,
                        const TableIdToNodeMap& table_id_to_node_map)
      : condition_(condition)
      , join_type_(join_type)
      , query_infos_(query_infos)
      , memory_level_(memory_level)
      , executor_(executor)
      , column_cache_(column_cache)
      , inner_outer_pairs_(inner_outer_pairs)
      , device_count_(device_count)
      , query_hints_(query_hints)
      , hashtable_build_dag_map_(hashtable_build_dag_map)
      , table_id_to_node_map_(table_id_to_node_map) {
    CHECK_GT(device_count_, 0);
    hash_tables_for_device_.resize(std::max(device_count_, 1));
  }

  virtual ~OverlapsJoinHashTable() {}

  //! Make hash table from an in-flight SQL query's parse tree etc.
  static std::shared_ptr<OverlapsJoinHashTable> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      const JoinType join_type,
      const int device_count,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const RegisteredQueryHint& query_hint,
      const TableIdToNodeMap& table_id_to_node_map);

  static void invalidateCache() {
    CHECK(auto_tuner_cache_);
    auto_tuner_cache_->clearCache();

    CHECK(hash_table_cache_);
    hash_table_cache_->clearCache();
  }

  static void markCachedItemAsDirty(size_t table_key) {
    CHECK(auto_tuner_cache_);
    CHECK(hash_table_cache_);
    auto candidate_table_keys =
        hash_table_cache_->getMappedQueryPlanDagsWithTableKey(table_key);
    if (candidate_table_keys.has_value()) {
      auto_tuner_cache_->markCachedItemAsDirty(table_key,
                                               *candidate_table_keys,
                                               CacheItemType::OVERLAPS_AUTO_TUNER_PARAM,
                                               DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      hash_table_cache_->markCachedItemAsDirty(table_key,
                                               *candidate_table_keys,
                                               CacheItemType::OVERLAPS_HT,
                                               DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
    }
  }

  static HashtableRecycler* getHashTableCache() {
    CHECK(hash_table_cache_);
    return hash_table_cache_.get();
  }

  static OverlapsTuningParamRecycler* getOverlapsTuningParamCache() {
    CHECK(auto_tuner_cache_);
    return auto_tuner_cache_.get();
  }

 protected:
  void reify(const HashType preferred_layout);

  virtual void reifyWithLayout(const HashType layout);

  virtual void reifyImpl(std::vector<ColumnsForDevice>& columns_per_device,
                         const Fragmenter_Namespace::TableInfo& query_info,
                         const HashType layout,
                         const size_t shard_count,
                         const size_t entry_count,
                         const size_t emitted_keys_count,
                         const bool skip_hashtable_caching,
                         const size_t chosen_max_hashtable_size,
                         const double chosen_bucket_threshold);

  void reifyForDevice(const ColumnsForDevice& columns_for_device,
                      const HashType layout,
                      const size_t entry_count,
                      const size_t emitted_keys_count,
                      const bool skip_hashtable_caching,
                      const int device_id,
                      const logger::ThreadLocalIds parent_thread_local_ids);

  size_t calculateHashTableSize(size_t number_of_dimensions,
                                size_t emitted_keys_count,
                                size_t entry_count) const;

  ColumnsForDevice fetchColumnsForDevice(
      const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
      const int device_id,
      DeviceAllocator* dev_buff_owner);

  // returns entry_count, emitted_keys_count
  virtual std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<double>& inverse_bucket_sizes_for_dimension,
      std::vector<ColumnsForDevice>&,
      const size_t chosen_max_hashtable_size,
      const double chosen_bucket_threshold);

  // returns entry_count, emitted_keys_count
  virtual std::pair<size_t, size_t> computeHashTableCounts(
      const size_t shard_count,
      const std::vector<double>& inverse_bucket_sizes_for_dimension,
      std::vector<ColumnsForDevice>& columns_per_device,
      const size_t chosen_max_hashtable_size,
      const double chosen_bucket_threshold);

  void setInverseBucketSizeInfo(const std::vector<double>& inverse_bucket_sizes,
                                std::vector<ColumnsForDevice>& columns_per_device,
                                const size_t device_count);

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
      const size_t emitted_keys_count,
      const bool skip_hashtable_caching);

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
      std::shared_ptr<BaselineHashTable>& cpu_hash_table,
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

  const RegisteredQueryHint& getRegisteredQueryHint() { return query_hints_; }

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

  shared::TableKey getInnerTableId() const noexcept override;

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

  bool isBitwiseEq() const override;

  std::shared_ptr<HashTable> initHashTableOnCpuFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier);

  std::optional<std::pair<size_t, size_t>> getApproximateTupleCountFromCache(
      QueryPlanHash key,
      CacheItemType item_type,
      DeviceIdentifier device_identifier);

  void putHashTableOnCpuToCache(QueryPlanHash key,
                                CacheItemType item_type,
                                std::shared_ptr<HashTable> hashtable_ptr,
                                DeviceIdentifier device_identifier,
                                size_t hashtable_building_time);

  llvm::Value* codegenKey(const CompilationOptions&);
  std::vector<llvm::Value*> codegenManyKey(const CompilationOptions&);

  std::optional<OverlapsHashTableMetaInfo> getOverlapsHashTableMetaInfo() {
    return hashtable_cache_meta_info_.overlaps_meta_info;
  }

  struct AlternativeCacheKeyForOverlapsHashJoin {
    std::vector<InnerOuter> inner_outer_pairs;
    const size_t num_elements;
    const size_t chunk_key_hash;
    const SQLOps optype;
    const size_t max_hashtable_size;
    const double bucket_threshold;
    const std::vector<double> inverse_bucket_sizes = {};
  };

  QueryPlanHash getAlternativeCacheKey(AlternativeCacheKeyForOverlapsHashJoin& info) {
    auto hash = info.chunk_key_hash;
    for (InnerOuter inner_outer : info.inner_outer_pairs) {
      auto inner_col = inner_outer.first;
      auto rhs_col_var = dynamic_cast<const Analyzer::ColumnVar*>(inner_outer.second);
      auto outer_col = rhs_col_var ? rhs_col_var : inner_col;
      boost::hash_combine(hash, inner_col->toString());
      if (inner_col->get_type_info().is_string()) {
        boost::hash_combine(hash, outer_col->toString());
      }
    }
    boost::hash_combine(hash, info.num_elements);
    boost::hash_combine(hash, info.optype);
    boost::hash_combine(hash, info.max_hashtable_size);
    boost::hash_combine(hash, info.bucket_threshold);
    boost::hash_combine(hash, info.inverse_bucket_sizes);
    return hash;
  }

  void generateCacheKey(
      const size_t max_hashtable_size,
      const double bucket_threshold,
      const std::vector<double>& bucket_sizes,
      std::vector<std::vector<Fragmenter_Namespace::FragmentInfo>>& fragments_per_device,
      int device_count) {
    for (int device_id = 0; device_id < device_count; ++device_id) {
      auto hash_val = boost::hash_value(hashtable_cache_key_[device_id]);
      boost::hash_combine(hash_val, max_hashtable_size);
      boost::hash_combine(hash_val, bucket_threshold);
      boost::hash_combine(hash_val, bucket_sizes);
      boost::hash_combine(hash_val,
                          HashJoin::collectFragmentIds(fragments_per_device[device_id]));
      hashtable_cache_key_[device_id] = hash_val;
      hash_table_cache_->addQueryPlanDagForTableKeys(hashtable_cache_key_[device_id],
                                                     table_keys_);
    }
  }

  QueryPlanHash getCacheKey(int device_id) const {
    return hashtable_cache_key_[device_id];
  }

  const std::vector<InnerOuter>& getInnerOuterPairs() const { return inner_outer_pairs_; }

  void setOverlapsHashtableMetaInfo(size_t max_table_size_bytes,
                                    double bucket_threshold,
                                    std::vector<double>& bucket_sizes) {
    OverlapsHashTableMetaInfo overlaps_meta_info;
    overlaps_meta_info.bucket_sizes = bucket_sizes;
    overlaps_meta_info.overlaps_max_table_size_bytes = max_table_size_bytes;
    overlaps_meta_info.overlaps_bucket_threshold = bucket_threshold;
    HashtableCacheMetaInfo meta_info;
    meta_info.overlaps_meta_info = overlaps_meta_info;
    hashtable_cache_meta_info_ = meta_info;
  }

  const std::shared_ptr<Analyzer::BinOper> condition_;
  const JoinType join_type_;
  const std::vector<InputTableInfo>& query_infos_;
  const Data_Namespace::MemoryLevel memory_level_;

  Executor* executor_;
  ColumnCacheMap& column_cache_;

  std::vector<InnerOuter> inner_outer_pairs_;
  const int device_count_;
  RegisteredQueryHint query_hints_;

  std::vector<double> inverse_bucket_sizes_for_dimension_;
  double chosen_overlaps_bucket_threshold_;
  size_t chosen_overlaps_max_table_size_bytes_;
  CompositeKeyInfo composite_key_info_;

  std::optional<HashType>
      layout_override_;  // allows us to use a 1:many hash table for many:many

  std::mutex cpu_hash_table_buff_mutex_;

  // cache a hashtable based on the cache key C
  // C = query plan dag D + join col J + hashtable params P
  // by varying overlaps join hashtable parameters P, we can build
  // multiple (and different) hashtables for the same query plan dag D
  // in this scenario, the rule we follow is cache everything
  // with the assumption that varying P is intended by user
  // for the performance and so worth to keep it for future recycling
  static std::unique_ptr<HashtableRecycler> hash_table_cache_;
  // auto tuner cache is maintained separately with hashtable cache
  static std::unique_ptr<OverlapsTuningParamRecycler> auto_tuner_cache_;

  HashTableBuildDagMap hashtable_build_dag_map_;
  QueryPlanDAG query_plan_dag_;
  std::vector<QueryPlanHash> hashtable_cache_key_;
  HashtableCacheMetaInfo hashtable_cache_meta_info_;
  std::unordered_set<size_t> table_keys_;
  const TableIdToNodeMap table_id_to_node_map_;
};
