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

#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"

#include <atomic>
#include <future>
#include <numeric>
#include <thread>

#include "Logger/Logger.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/JoinHashTable/Builders/PerfectHashTableBuilder.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "QueryEngine/RuntimeFunctions.h"

// let's only consider CPU hahstable recycler at this moment
std::unique_ptr<HashtableRecycler> PerfectJoinHashTable::hash_table_cache_ =
    std::make_unique<HashtableRecycler>(CacheItemType::PERFECT_HT,
                                        DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
std::unique_ptr<HashingSchemeRecycler> PerfectJoinHashTable::hash_table_layout_cache_ =
    std::make_unique<HashingSchemeRecycler>();

namespace {
std::pair<InnerOuter, InnerOuterStringOpInfos> get_cols(
    const Analyzer::BinOper* qual_bin_oper,
    const Catalog_Namespace::Catalog& cat,
    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  return HashJoin::normalizeColumnPair(lhs, rhs, cat, temporary_tables);
}

HashEntryInfo get_bucketized_hash_entry_info(SQLTypeInfo const& context_ti,
                                             ExpressionRange const& col_range,
                                             bool const is_bw_eq) {
  using EmptyRangeSize = boost::optional<size_t>;
  auto empty_range_check = [](ExpressionRange const& col_range,
                              bool const is_bw_eq) -> EmptyRangeSize {
    if (col_range.getIntMin() > col_range.getIntMax()) {
      CHECK_EQ(col_range.getIntMin(), int64_t(0));
      CHECK_EQ(col_range.getIntMax(), int64_t(-1));
      if (is_bw_eq) {
        return size_t(1);
      }
      return size_t(0);
    }
    return EmptyRangeSize{};
  };

  auto empty_range = empty_range_check(col_range, is_bw_eq);
  if (empty_range) {
    return {size_t(*empty_range), 1};
  }

  int64_t bucket_normalization =
      context_ti.get_type() == kDATE ? col_range.getBucket() : 1;
  CHECK_GT(bucket_normalization, 0);
  return {size_t(col_range.getIntMax() - col_range.getIntMin() + 1 + (is_bw_eq ? 1 : 0)),
          bucket_normalization};
}

size_t get_hash_entry_count(const ExpressionRange& col_range, const bool is_bw_eq) {
  if (col_range.getIntMin() > col_range.getIntMax()) {
    CHECK_EQ(col_range.getIntMin(), int64_t(0));
    CHECK_EQ(col_range.getIntMax(), int64_t(-1));
    return is_bw_eq ? 1 : 0;
  }
  return col_range.getIntMax() - col_range.getIntMin() + 1 + (is_bw_eq ? 1 : 0);
}

}  // namespace

namespace {

bool shard_count_less_or_equal_device_count(const int inner_table_id,
                                            const Executor* executor) {
  const auto inner_table_info = executor->getTableInfo(inner_table_id);
  std::unordered_set<int> device_holding_fragments;
  auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
  const int device_count = cuda_mgr ? cuda_mgr->getDeviceCount() : 1;
  for (const auto& fragment : inner_table_info.fragments) {
    if (fragment.shard != -1) {
      const auto it_ok = device_holding_fragments.emplace(fragment.shard % device_count);
      if (!it_ok.second) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

size_t get_shard_count(
    std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*> equi_pair,
    const Executor* executor) {
  const auto inner_col = equi_pair.first;
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(equi_pair.second);
  if (!outer_col || inner_col->get_table_id() < 0 || outer_col->get_table_id() < 0) {
    return 0;
  }
  if (outer_col->get_rte_idx()) {
    return 0;
  }
  if (inner_col->get_type_info() != outer_col->get_type_info()) {
    return 0;
  }
  const auto catalog = executor->getCatalog();
  const auto inner_td = catalog->getMetadataForTable(inner_col->get_table_id());
  CHECK(inner_td);
  const auto outer_td = catalog->getMetadataForTable(outer_col->get_table_id());
  CHECK(outer_td);
  if (inner_td->shardedColumnId == 0 || outer_td->shardedColumnId == 0 ||
      inner_td->nShards != outer_td->nShards) {
    return 0;
  }
  if (!shard_count_less_or_equal_device_count(inner_td->tableId, executor)) {
    return 0;
  }
  // The two columns involved must be the ones on which the tables have been sharded on.
  return (inner_td->shardedColumnId == inner_col->get_column_id() &&
          outer_td->shardedColumnId == outer_col->get_column_id()) ||
                 (outer_td->shardedColumnId == inner_col->get_column_id() &&
                  inner_td->shardedColumnId == inner_col->get_column_id())
             ? inner_td->nShards
             : 0;
}

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<PerfectJoinHashTable> PerfectJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map) {
  CHECK(IS_EQUIVALENCE(qual_bin_oper->get_optype()));
  const auto cols_and_string_op_infos =
      get_cols(qual_bin_oper.get(), *executor->getCatalog(), executor->temporary_tables_);
  const auto& cols = cols_and_string_op_infos.first;
  const auto& inner_outer_string_op_infos = cols_and_string_op_infos.second;
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  auto col_range =
      getExpressionRange(ti.is_string() ? cols.second : inner_col, query_infos, executor);
  if (col_range.getType() == ExpressionRangeType::Invalid) {
    throw HashJoinFail(
        "Could not compute range for the expressions involved in the equijoin");
  }
  const auto rhs_source_col_range =
      ti.is_string() ? getExpressionRange(inner_col, query_infos, executor) : col_range;
  if (ti.is_string()) {
    // The nullable info must be the same as the source column.
    if (rhs_source_col_range.getType() == ExpressionRangeType::Invalid) {
      throw HashJoinFail(
          "Could not compute range for the expressions involved in the equijoin");
    }
    if (rhs_source_col_range.getIntMin() > rhs_source_col_range.getIntMax()) {
      // If the inner column expression range is empty, use the inner col range
      CHECK_EQ(rhs_source_col_range.getIntMin(), int64_t(0));
      CHECK_EQ(rhs_source_col_range.getIntMax(), int64_t(-1));
      col_range = rhs_source_col_range;
    } else {
      col_range = ExpressionRange::makeIntRange(
          std::min(rhs_source_col_range.getIntMin(), col_range.getIntMin()),
          std::max(rhs_source_col_range.getIntMax(), col_range.getIntMax()),
          0,
          rhs_source_col_range.hasNulls());
    }
  }

  // We can't allocate more than 2GB contiguous memory on GPU and each entry is 4 bytes.
  const auto max_hash_entry_count =
      memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL
          ? static_cast<size_t>(std::numeric_limits<int32_t>::max() / sizeof(int32_t))
          : static_cast<size_t>(std::numeric_limits<int32_t>::max());

  auto bucketized_entry_count_info = get_bucketized_hash_entry_info(
      ti, col_range, qual_bin_oper->get_optype() == kBW_EQ);
  auto bucketized_entry_count = bucketized_entry_count_info.getNormalizedHashEntryCount();

  if (bucketized_entry_count > max_hash_entry_count) {
    throw TooManyHashEntries();
  }

  if (qual_bin_oper->get_optype() == kBW_EQ &&
      col_range.getIntMax() >= std::numeric_limits<int64_t>::max()) {
    throw HashJoinFail("Cannot translate null value for kBW_EQ");
  }
  decltype(std::chrono::steady_clock::now()) ts1, ts2;
  if (VLOGGING(1)) {
    ts1 = std::chrono::steady_clock::now();
  }

  auto join_hash_table = std::shared_ptr<PerfectJoinHashTable>(
      new PerfectJoinHashTable(qual_bin_oper,
                               inner_col,
                               query_infos,
                               memory_level,
                               join_type,
                               preferred_hash_type,
                               col_range,
                               rhs_source_col_range,
                               column_cache,
                               executor,
                               device_count,
                               hashtable_build_dag_map,
                               table_id_to_node_map,
                               inner_outer_string_op_infos));
  try {
    join_hash_table->reify();
  } catch (const TableMustBeReplicated& e) {
    // Throw a runtime error to abort the query
    join_hash_table->freeHashBufferMemory();
    throw std::runtime_error(e.what());
  } catch (const HashJoinFail& e) {
    // HashJoinFail exceptions log an error and trigger a retry with a join loop (if
    // possible)
    join_hash_table->freeHashBufferMemory();
    throw HashJoinFail(std::string("Could not build a 1-to-1 correspondence for columns "
                                   "involved in equijoin | ") +
                       e.what());
  } catch (const ColumnarConversionNotSupported& e) {
    throw HashJoinFail(std::string("Could not build hash tables for equijoin | ") +
                       e.what());
  } catch (const OutOfMemory& e) {
    throw HashJoinFail(
        std::string("Ran out of memory while building hash tables for equijoin | ") +
        e.what());
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("Fatal error while attempting to build hash tables for join: ") +
        e.what());
  }
  if (VLOGGING(1)) {
    ts2 = std::chrono::steady_clock::now();
    VLOG(1) << "Built perfect hash table "
            << getHashTypeString(join_hash_table->getHashType()) << " in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count()
            << " ms";
  }
  return join_hash_table;
}

bool needs_dictionary_translation(
    const InnerOuter& inner_outer_col_pair,
    const InnerOuterStringOpInfos& inner_outer_string_op_infos,
    const Executor* executor) {
  if (inner_outer_string_op_infos.first.size() ||
      inner_outer_string_op_infos.second.size()) {
    return true;
  }
  auto inner_col = inner_outer_col_pair.first;
  auto outer_col_expr = inner_outer_col_pair.second;
  const auto catalog = executor->getCatalog();
  CHECK(catalog);
  const auto inner_cd = get_column_descriptor_maybe(
      inner_col->get_column_id(), inner_col->get_table_id(), *catalog);
  const auto& inner_ti = get_column_type(inner_col->get_column_id(),
                                         inner_col->get_table_id(),
                                         inner_cd,
                                         executor->getTemporaryTables());
  // Only strings may need dictionary translation.
  if (!inner_ti.is_string()) {
    return false;
  }
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(outer_col_expr);
  CHECK(outer_col);
  const auto outer_cd = get_column_descriptor_maybe(
      outer_col->get_column_id(), outer_col->get_table_id(), *catalog);
  // Don't want to deal with temporary tables for now, require translation.
  if (!inner_cd || !outer_cd) {
    return true;
  }
  const auto& outer_ti = get_column_type(outer_col->get_column_id(),
                                         outer_col->get_table_id(),
                                         outer_cd,
                                         executor->getTemporaryTables());
  CHECK_EQ(inner_ti.is_string(), outer_ti.is_string());
  // If the two columns don't share the dictionary, translation is needed.
  if (outer_ti.get_comp_param() != inner_ti.get_comp_param()) {
    return true;
  }
  const auto inner_str_dict_proxy =
      executor->getStringDictionaryProxy(inner_col->get_comp_param(), true);
  CHECK(inner_str_dict_proxy);
  const auto outer_str_dict_proxy =
      executor->getStringDictionaryProxy(inner_col->get_comp_param(), true);
  CHECK(outer_str_dict_proxy);

  return *inner_str_dict_proxy != *outer_str_dict_proxy;
}

std::vector<Fragmenter_Namespace::FragmentInfo> only_shards_for_device(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    const int device_count) {
  std::vector<Fragmenter_Namespace::FragmentInfo> shards_for_device;
  for (const auto& fragment : fragments) {
    CHECK_GE(fragment.shard, 0);
    if (fragment.shard % device_count == device_id) {
      shards_for_device.push_back(fragment);
    }
  }
  return shards_for_device;
}

bool PerfectJoinHashTable::isOneToOneHashPossible(
    const std::vector<ColumnsForDevice>& columns_per_device) const {
  CHECK(!inner_outer_pairs_.empty());
  const auto& rhs_col_ti = inner_outer_pairs_.front().first->get_type_info();
  const auto max_unique_hash_input_entries =
      get_bucketized_hash_entry_info(
          rhs_col_ti, rhs_source_col_range_, qual_bin_oper_->get_optype() == kBW_EQ)
          .getNormalizedHashEntryCount() +
      rhs_source_col_range_.hasNulls();
  for (const auto& device_columns : columns_per_device) {
    CHECK(!device_columns.join_columns.empty());
    const auto rhs_join_col_num_entries = device_columns.join_columns.front().num_elems;
    if (rhs_join_col_num_entries > max_unique_hash_input_entries) {
      VLOG(1) << "Skipping attempt to build perfect hash one-to-one table as number of "
                 "rhs column entries ("
              << rhs_join_col_num_entries << ") exceeds range for rhs join column ("
              << max_unique_hash_input_entries << ").";
      return false;
    }
  }
  return true;
}

void PerfectJoinHashTable::reify() {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_LT(0, device_count_);
  auto catalog = const_cast<Catalog_Namespace::Catalog*>(executor_->getCatalog());
  const auto cols =
      get_cols(qual_bin_oper_.get(), *catalog, executor_->temporary_tables_).first;
  const auto inner_col = cols.first;
  HashJoin::checkHashJoinReplicationConstraint(
      inner_col->get_table_id(),
      get_shard_count(qual_bin_oper_.get(), executor_),
      executor_);
  const auto& query_info = getInnerQueryInfo(inner_col).info;
  if (query_info.fragments.empty()) {
    return;
  }
  if (query_info.getNumTuplesUpperBound() >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw TooManyHashEntries();
  }
  std::vector<std::future<void>> init_threads;
  const int shard_count = shardCount();

  inner_outer_pairs_.push_back(cols);
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));
  // Todo(todd): Clean up the fact that we store the inner outer column pairs as a vector,
  // even though only one is ever valid for perfect hash layout. Either move to 1 or keep
  // the vector but move it to the HashTable parent class
  needs_dict_translation_ = needs_dictionary_translation(
      inner_outer_pairs_.front(), inner_outer_string_op_infos_, executor_);

  std::vector<std::vector<Fragmenter_Namespace::FragmentInfo>> fragments_per_device;
  std::vector<ColumnsForDevice> columns_per_device;
  std::vector<std::unique_ptr<CudaAllocator>> dev_buff_owners;

  auto data_mgr = executor_->getDataMgr();
  // check the existence of cached hash table here before fetching columns
  // if available, skip the rest of logic and copy it to GPU if necessary
  // there are few considerable things:
  // 1. if table is sharded? --> deploy per-device logic
  // here, each device may load different set of fragments, so their cache keys are
  // different accordingly
  // 2. otherwise, each device has the same hash table built from "all" fragments
  // and their cache keys are the same (but we stick to per-device cache key vector)
  // here, for CPU, we consider its # device to be one
  // for GPU, each device builds its own hash table, or we build a single hash table on
  // CPU and then copy it to each device
  // 3. if cache key is not available? --> use alternative cache key

  // retrieve fragment lists and chunk key per device
  std::vector<ChunkKey> chunk_key_per_device;
  auto outer_col =
      dynamic_cast<const Analyzer::ColumnVar*>(inner_outer_pairs_.front().second);
  for (int device_id = 0; device_id < device_count_; ++device_id) {
    fragments_per_device.emplace_back(
        shard_count
            ? only_shards_for_device(query_info.fragments, device_id, device_count_)
            : query_info.fragments);
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      dev_buff_owners.emplace_back(std::make_unique<CudaAllocator>(
          data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id)));
    }
    const auto chunk_key =
        genChunkKey(fragments_per_device[device_id], outer_col, inner_col);
    chunk_key_per_device.emplace_back(std::move(chunk_key));
  }

  // try to extract cache key for hash table and its relevant info
  auto hashtable_access_path_info =
      HashtableRecycler::getHashtableAccessPathInfo(inner_outer_pairs_,
                                                    {inner_outer_string_op_infos_},
                                                    qual_bin_oper_->get_optype(),
                                                    join_type_,
                                                    hashtable_build_dag_map_,
                                                    device_count_,
                                                    shard_count,
                                                    fragments_per_device,
                                                    executor_);
  hashtable_cache_key_ = hashtable_access_path_info.hashed_query_plan_dag;
  hashtable_cache_meta_info_ = hashtable_access_path_info.meta_info;
  table_keys_ = hashtable_access_path_info.table_keys;

  if (table_keys_.empty()) {
    // the actual chunks fetched per device can be different but they constitute the same
    // table in the same db, so we can exploit this to create an alternative table key
    table_keys_ = DataRecyclerUtil::getAlternativeTableKeys(
        chunk_key_per_device,
        executor_->getCatalog()->getDatabaseId(),
        getInnerTableId());
  }
  CHECK(!table_keys_.empty());

  if (HashtableRecycler::isInvalidHashTableCacheKey(hashtable_cache_key_) &&
      getInnerTableId() > 0) {
    // sometimes we cannot retrieve query plan dag, so try to recycler cache
    // with the old-fashioned cache key if we deal with hashtable of non-temporary table
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      const auto num_tuples = std::accumulate(
          fragments_per_device[device_id].begin(),
          fragments_per_device[device_id].end(),
          size_t(0),
          [](size_t sum, const auto& fragment) { return sum + fragment.getNumTuples(); });
      AlternativeCacheKeyForPerfectHashJoin cache_key{col_range_,
                                                      inner_col,
                                                      outer_col ? outer_col : inner_col,
                                                      inner_outer_string_op_infos_,
                                                      chunk_key_per_device[device_id],
                                                      num_tuples,
                                                      qual_bin_oper_->get_optype(),
                                                      join_type_};
      hashtable_cache_key_[device_id] = getAlternativeCacheKey(cache_key);
    }
  }

  // register a mapping between cache key and its input table info for per-table cache
  // invalidation if we have valid cache key for "all" devices (otherwise, we skip to use
  // cached hash table for safety)
  const bool invalid_cache_key =
      HashtableRecycler::isInvalidHashTableCacheKey(hashtable_cache_key_);
  if (!invalid_cache_key) {
    if (!shard_count) {
      hash_table_cache_->addQueryPlanDagForTableKeys(hashtable_cache_key_.front(),
                                                     table_keys_);
    } else {
      std::for_each(hashtable_cache_key_.cbegin(),
                    hashtable_cache_key_.cend(),
                    [this](QueryPlanHash key) {
                      hash_table_cache_->addQueryPlanDagForTableKeys(key, table_keys_);
                    });
    }
  }

  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);

  // Assume we will need one-to-many if we have a string operation, as these tend
  // to be cardinality-reducting operations, i.e. |S(t)| < |t|
  // Todo(todd): Ostensibly only string ops on the rhs/inner expression cause rhs dups and
  // so we may be too conservative here, but validate

  const bool has_string_ops = inner_outer_string_op_infos_.first.size() ||
                              inner_outer_string_op_infos_.second.size();

  // Also check if on the number of entries per column exceeds the rhs join hash table
  // range, and skip trying to build a One-to-One hash table if so. There is a slight edge
  // case where this can be overly pessimistic, and that is if the non-null values are all
  // unique, but there are multiple null values, but we currently don't have the metadata
  // to track null counts (only column nullability from the ddl and null existence from
  // the encoded data), and this is probably too much of an edge case to worry about for
  // now given the general performance benfits of skipping 1:1 if we are fairly confident
  // it is doomed up front

  // Now check if on the number of entries per column exceeds the rhs join hash table
  // range, and skip trying to build a One-to-One hash table if so
  if (hash_type_ == HashType::OneToOne &&
      (has_string_ops || !isOneToOneHashPossible(columns_per_device))) {
    hash_type_ = HashType::OneToMany;
  }

  // todo (yoonmin) : support dictionary proxy cache for join including string op(s)
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    // construct string dictionary proxies if necessary
    std::unique_lock<std::mutex> str_proxy_translation_lock(str_proxy_translation_mutex_);
    if (needs_dict_translation_ && !str_proxy_translation_map_) {
      CHECK_GE(inner_outer_pairs_.size(), 1UL);
      str_proxy_translation_map_ =
          HashJoin::translateInnerToOuterStrDictProxies(inner_outer_pairs_.front(),
                                                        inner_outer_string_op_infos_,
                                                        col_range_,
                                                        executor_);
    }
  }

  auto allow_hashtable_recycling =
      HashtableRecycler::isSafeToCacheHashtable(table_id_to_node_map_,
                                                needs_dict_translation_,
                                                {inner_outer_string_op_infos_},
                                                inner_col->get_table_id());
  bool has_invalid_cached_hash_table = false;
  if (effective_memory_level == Data_Namespace::CPU_LEVEL &&
      HashJoin::canAccessHashTable(
          allow_hashtable_recycling, invalid_cache_key, join_type_)) {
    // build a hash table on CPU, and we have a chance to recycle the cached one if
    // available
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      auto hash_table =
          initHashTableOnCpuFromCache(hashtable_cache_key_[device_id],
                                      CacheItemType::PERFECT_HT,
                                      DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      if (hash_table) {
        hash_tables_for_device_[device_id] = hash_table;
        hash_type_ = hash_table->getLayout();
      } else {
        has_invalid_cached_hash_table = true;
        break;
      }
    }

    if (has_invalid_cached_hash_table) {
      hash_tables_for_device_.clear();
      hash_tables_for_device_.resize(device_count_);
    } else {
      if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
        for (int device_id = 0; device_id < device_count_; ++device_id) {
          auto cpu_hash_table = std::dynamic_pointer_cast<PerfectHashTable>(
              hash_tables_for_device_[device_id]);
          copyCpuHashTableToGpu(cpu_hash_table, device_id, data_mgr);
        }
#else
        UNREACHABLE();
#endif
      }
      return;
    }
  }

  // we have no cached hash table for this qual
  // so, start building the hash table by fetching columns for devices
  for (int device_id = 0; device_id < device_count_; ++device_id) {
    columns_per_device.emplace_back(
        fetchColumnsForDevice(fragments_per_device[device_id],
                              device_id,
                              memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL
                                  ? dev_buff_owners[device_id].get()
                                  : nullptr,
                              *catalog));
  }

  try {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      const auto chunk_key = genChunkKey(fragments_per_device[device_id],
                                         inner_outer_pairs_.front().second,
                                         inner_outer_pairs_.front().first);
      init_threads.push_back(std::async(std::launch::async,
                                        &PerfectJoinHashTable::reifyForDevice,
                                        this,
                                        chunk_key,
                                        columns_per_device[device_id],
                                        hash_type_,
                                        device_id,
                                        logger::query_id(),
                                        logger::thread_id()));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }
  } catch (const NeedsOneToManyHash& e) {
    VLOG(1) << "RHS/Inner hash join values detected to not be unique, falling back to "
               "One-to-Many hash layout.";
    CHECK(hash_type_ == HashType::OneToOne);
    hash_type_ = HashType::OneToMany;
    freeHashBufferMemory();
    init_threads.clear();
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      CHECK_EQ(dev_buff_owners.size(), size_t(device_count_));
    }
    CHECK_EQ(columns_per_device.size(), size_t(device_count_));
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      const auto chunk_key = genChunkKey(fragments_per_device[device_id],
                                         inner_outer_pairs_.front().second,
                                         inner_outer_pairs_.front().first);
      init_threads.push_back(std::async(std::launch::async,
                                        &PerfectJoinHashTable::reifyForDevice,
                                        this,
                                        chunk_key,
                                        columns_per_device[device_id],
                                        hash_type_,
                                        device_id,
                                        logger::query_id(),
                                        logger::thread_id()));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }
  }
}

Data_Namespace::MemoryLevel PerfectJoinHashTable::getEffectiveMemoryLevel(
    const std::vector<InnerOuter>& inner_outer_pairs) const {
  if (needs_dictionary_translation(
          inner_outer_pairs.front(), inner_outer_string_op_infos_, executor_)) {
    needs_dict_translation_ = true;
    return Data_Namespace::CPU_LEVEL;
  }
  return memory_level_;
}

ColumnsForDevice PerfectJoinHashTable::fetchColumnsForDevice(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    DeviceAllocator* dev_buff_owner,
    const Catalog_Namespace::Catalog& catalog) {
  std::vector<JoinColumn> join_columns;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<JoinColumnTypeInfo> join_column_types;
  std::vector<JoinBucketInfo> join_bucket_info;
  std::vector<std::shared_ptr<void>> malloc_owner;
  const auto effective_memory_level =
      get_effective_memory_level(memory_level_, needs_dict_translation_);
  for (const auto& inner_outer_pair : inner_outer_pairs_) {
    const auto inner_col = inner_outer_pair.first;
    const auto inner_cd = get_column_descriptor_maybe(
        inner_col->get_column_id(), inner_col->get_table_id(), catalog);
    if (inner_cd && inner_cd->isVirtualCol) {
      throw FailedToJoinOnVirtualColumn();
    }
    join_columns.emplace_back(fetchJoinColumn(inner_col,
                                              fragments,
                                              effective_memory_level,
                                              device_id,
                                              chunks_owner,
                                              dev_buff_owner,
                                              malloc_owner,
                                              executor_,
                                              &column_cache_));
    const auto& ti = inner_col->get_type_info();
    join_column_types.emplace_back(JoinColumnTypeInfo{static_cast<size_t>(ti.get_size()),
                                                      0,
                                                      0,
                                                      inline_fixed_encoding_null_val(ti),
                                                      isBitwiseEq(),
                                                      0,
                                                      get_join_column_type_kind(ti)});
  }
  return {join_columns, join_column_types, chunks_owner, join_bucket_info, malloc_owner};
}

void PerfectJoinHashTable::reifyForDevice(const ChunkKey& chunk_key,
                                          const ColumnsForDevice& columns_for_device,
                                          const HashType layout,
                                          const int device_id,
                                          const logger::QueryId query_id,
                                          const logger::ThreadId parent_thread_id) {
  logger::QidScopeGuard qsg = logger::set_thread_local_query_id(query_id);
  DEBUG_TIMER_NEW_THREAD(parent_thread_id);
  const auto effective_memory_level =
      get_effective_memory_level(memory_level_, needs_dict_translation_);

  CHECK_EQ(columns_for_device.join_columns.size(), size_t(1));
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));
  auto& join_column = columns_for_device.join_columns.front();
  if (layout == HashType::OneToOne) {
    const auto err = initHashTableForDevice(chunk_key,
                                            join_column,
                                            inner_outer_pairs_.front(),
                                            layout,

                                            effective_memory_level,
                                            device_id);
    if (err) {
      throw NeedsOneToManyHash();
    }
  } else {
    const auto err = initHashTableForDevice(chunk_key,
                                            join_column,
                                            inner_outer_pairs_.front(),
                                            HashType::OneToMany,
                                            effective_memory_level,
                                            device_id);
    if (err) {
      throw std::runtime_error("Unexpected error building one to many hash table: " +
                               std::to_string(err));
    }
  }
}

int PerfectJoinHashTable::initHashTableForDevice(
    const ChunkKey& chunk_key,
    const JoinColumn& join_column,
    const InnerOuter& cols,
    const HashType layout,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id) {
  auto timer = DEBUG_TIMER(__func__);
  const auto inner_col = cols.first;
  CHECK(inner_col);

  auto hash_entry_info = get_bucketized_hash_entry_info(
      inner_col->get_type_info(), col_range_, isBitwiseEq());
  if (!hash_entry_info && layout == HashType::OneToOne) {
    // TODO: what is this for?
    return 0;
  }

#ifndef HAVE_CUDA
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  int err{0};
  const int32_t hash_join_invalid_val{-1};
  auto hashtable_layout = layout;
  auto allow_hashtable_recycling =
      HashtableRecycler::isSafeToCacheHashtable(table_id_to_node_map_,
                                                needs_dict_translation_,
                                                {inner_outer_string_op_infos_},
                                                inner_col->get_table_id());
  if (allow_hashtable_recycling) {
    auto cached_hashtable_layout_type = hash_table_layout_cache_->getItemFromCache(
        hashtable_cache_key_[device_id],
        CacheItemType::HT_HASHING_SCHEME,
        DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
        {});
    if (cached_hashtable_layout_type) {
      hash_type_ = *cached_hashtable_layout_type;
      hashtable_layout = hash_type_;
    }
  }
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    CHECK(!chunk_key.empty());
    std::shared_ptr<PerfectHashTable> hash_table{nullptr};
    decltype(std::chrono::steady_clock::now()) ts1, ts2;
    ts1 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      PerfectJoinHashTableBuilder builder;
      if (hashtable_layout == HashType::OneToOne) {
        builder.initOneToOneHashTableOnCpu(join_column,
                                           col_range_,
                                           isBitwiseEq(),
                                           cols,
                                           str_proxy_translation_map_,
                                           join_type_,
                                           hashtable_layout,
                                           hash_entry_info,
                                           hash_join_invalid_val,
                                           executor_);
        hash_table = builder.getHashTable();
      } else {
        builder.initOneToManyHashTableOnCpu(join_column,
                                            col_range_,
                                            isBitwiseEq(),
                                            cols,
                                            str_proxy_translation_map_,
                                            hash_entry_info,
                                            hash_join_invalid_val,
                                            executor_);
        hash_table = builder.getHashTable();
      }
      ts2 = std::chrono::steady_clock::now();
      auto build_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count();
      hash_table->setHashEntryInfo(hash_entry_info);
      hash_table->setColumnNumElems(join_column.num_elems);
      if (allow_hashtable_recycling && hash_table) {
        // add ht-related items to cache iff we have a valid hashtable
        hash_table_layout_cache_->putItemToCache(hashtable_cache_key_[device_id],
                                                 hashtable_layout,
                                                 CacheItemType::HT_HASHING_SCHEME,
                                                 DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                                                 0,
                                                 0,
                                                 {});
        putHashTableOnCpuToCache(hashtable_cache_key_[device_id],
                                 CacheItemType::PERFECT_HT,
                                 hash_table,
                                 DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                                 build_time);
      }
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
      const auto& ti = inner_col->get_type_info();
      CHECK(ti.is_string());
      auto data_mgr = executor_->getDataMgr();
      copyCpuHashTableToGpu(hash_table, device_id, data_mgr);
#else
      UNREACHABLE();
#endif
    } else {
      CHECK(hash_table);
      CHECK_LT(static_cast<size_t>(device_id), hash_tables_for_device_.size());
      hash_tables_for_device_[device_id] = hash_table;
    }
  } else {
#ifdef HAVE_CUDA
    PerfectJoinHashTableBuilder builder;
    CHECK_EQ(Data_Namespace::GPU_LEVEL, effective_memory_level);
    builder.allocateDeviceMemory(join_column,
                                 hashtable_layout,
                                 hash_entry_info,
                                 shardCount(),
                                 device_id,
                                 device_count_,
                                 executor_);
    builder.initHashTableOnGpu(chunk_key,
                               join_column,
                               col_range_,
                               isBitwiseEq(),
                               cols,
                               join_type_,
                               hashtable_layout,
                               hash_entry_info,
                               shardCount(),
                               hash_join_invalid_val,
                               device_id,
                               device_count_,
                               executor_);
    CHECK_LT(static_cast<size_t>(device_id), hash_tables_for_device_.size());
    hash_tables_for_device_[device_id] = builder.getHashTable();
    if (!err && allow_hashtable_recycling && hash_tables_for_device_[device_id]) {
      // add layout to cache iff we have a valid hashtable
      hash_table_layout_cache_->putItemToCache(
          hashtable_cache_key_[device_id],
          hash_tables_for_device_[device_id]->getLayout(),
          CacheItemType::HT_HASHING_SCHEME,
          DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
          0,
          0,
          {});
    }
#else
    UNREACHABLE();
#endif
  }
  return err;
}

ChunkKey PerfectJoinHashTable::genChunkKey(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const Analyzer::Expr* outer_col_expr,
    const Analyzer::ColumnVar* inner_col) const {
  ChunkKey chunk_key{executor_->getCatalog()->getCurrentDB().dbId,
                     inner_col->get_table_id(),
                     inner_col->get_column_id()};
  const auto& ti = inner_col->get_type_info();
  std::for_each(fragments.cbegin(), fragments.cend(), [&chunk_key](const auto& fragment) {
    // collect all frag ids to correctly generated cache key for a cached hash table
    chunk_key.push_back(fragment.fragmentId);
  });
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(outer_col_expr);
    CHECK(outer_col);
    const auto& outer_query_info = getInnerQueryInfo(outer_col).info;
    size_t outer_elem_count =
        std::accumulate(outer_query_info.fragments.begin(),
                        outer_query_info.fragments.end(),
                        size_t(0),
                        [&chunk_key](size_t sum, const auto& fragment) {
                          chunk_key.push_back(fragment.fragmentId);
                          return sum + fragment.getNumTuples();
                        });
    chunk_key.push_back(outer_elem_count);
  }

  return chunk_key;
}

std::shared_ptr<PerfectHashTable> PerfectJoinHashTable::initHashTableOnCpuFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  CHECK(hash_table_cache_);
  auto timer = DEBUG_TIMER(__func__);
  VLOG(1) << "Checking CPU hash table cache.";
  auto hashtable_ptr =
      hash_table_cache_->getItemFromCache(key, item_type, device_identifier);
  if (hashtable_ptr) {
    return std::dynamic_pointer_cast<PerfectHashTable>(hashtable_ptr);
  }
  return nullptr;
}

void PerfectJoinHashTable::putHashTableOnCpuToCache(
    QueryPlanHash key,
    CacheItemType item_type,
    std::shared_ptr<PerfectHashTable> hashtable_ptr,
    DeviceIdentifier device_identifier,
    size_t hashtable_building_time) {
  CHECK(hash_table_cache_);
  CHECK(hashtable_ptr && !hashtable_ptr->getGpuBuffer());
  hash_table_cache_->putItemToCache(
      key,
      hashtable_ptr,
      item_type,
      device_identifier,
      hashtable_ptr->getHashTableBufferSize(ExecutorDeviceType::CPU),
      hashtable_building_time);
}

llvm::Value* PerfectJoinHashTable::codegenHashTableLoad(const size_t table_idx) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto hash_ptr = HashJoin::codegenHashTableLoad(table_idx, executor_);
  if (hash_ptr->getType()->isIntegerTy(64)) {
    return hash_ptr;
  }
  CHECK(hash_ptr->getType()->isPointerTy());
  return executor_->cgen_state_->ir_builder_.CreatePtrToInt(
      get_arg_by_name(executor_->cgen_state_->row_func_, "join_hash_tables"),
      llvm::Type::getInt64Ty(executor_->cgen_state_->context_));
}

std::vector<llvm::Value*> PerfectJoinHashTable::getHashJoinArgs(
    llvm::Value* hash_ptr,
    llvm::Value* key_lv,
    const Analyzer::Expr* key_col,
    const int shard_count,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CodeGenerator code_generator(executor_);
  CHECK(key_lv);
  // Todo(todd): Fix below, it's gross (but didn't want to redo the plumbing yet)
  // const auto key_lv = key_lvs.size() && key_lvs[0]
  //                        ? key_lvs[0]
  //                        : code_generator.codegen(key_col, true, co)[0];
  auto const& key_col_ti = key_col->get_type_info();
  auto hash_entry_info =
      get_bucketized_hash_entry_info(key_col_ti, col_range_, isBitwiseEq());

  std::vector<llvm::Value*> hash_join_idx_args{
      hash_ptr,
      executor_->cgen_state_->castToTypeIn(key_lv, 64),
      executor_->cgen_state_->llInt(col_range_.getIntMin()),
      executor_->cgen_state_->llInt(col_range_.getIntMax())};
  if (shard_count) {
    const auto expected_hash_entry_count =
        get_hash_entry_count(col_range_, isBitwiseEq());
    const auto entry_count_per_shard =
        (expected_hash_entry_count + shard_count - 1) / shard_count;
    hash_join_idx_args.push_back(
        executor_->cgen_state_->llInt<uint32_t>(entry_count_per_shard));
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt<uint32_t>(shard_count));
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt<uint32_t>(device_count_));
  }
  auto key_col_logical_ti = get_logical_type_info(key_col->get_type_info());
  if (!key_col_logical_ti.get_notnull() || isBitwiseEq()) {
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
        inline_fixed_encoding_null_val(key_col_logical_ti)));
  }
  auto special_date_bucketization_case = key_col_ti.get_type() == kDATE;
  if (isBitwiseEq()) {
    if (special_date_bucketization_case) {
      hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
          col_range_.getIntMax() / hash_entry_info.bucket_normalization + 1));
    } else {
      hash_join_idx_args.push_back(
          executor_->cgen_state_->llInt(col_range_.getIntMax() + 1));
    }
  }

  if (special_date_bucketization_case) {
    hash_join_idx_args.emplace_back(
        executor_->cgen_state_->llInt(hash_entry_info.bucket_normalization));
  }

  return hash_join_idx_args;
}

HashJoinMatchingSet PerfectJoinHashTable::codegenMatchingSet(const CompilationOptions& co,
                                                             const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto cols =
      get_cols(
          qual_bin_oper_.get(), *executor_->getCatalog(), executor_->temporary_tables_)
          .first;
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  auto pos_ptr = codegenHashTableLoad(index);
  CHECK(pos_ptr);
  const int shard_count = shardCount();
  const auto key_col_var = dynamic_cast<const Analyzer::ColumnVar*>(key_col);
  const auto val_col_var = dynamic_cast<const Analyzer::ColumnVar*>(val_col);
  if (key_col_var && val_col_var &&
      self_join_not_covered_by_left_deep_tree(
          key_col_var,
          val_col_var,
          get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
    throw std::runtime_error(
        "Query execution fails because the query contains not supported self-join "
        "pattern. We suspect the query requires multiple left-deep join tree due to "
        "the "
        "join condition of the self-join and is not supported for now. Please consider "
        "rewriting table order in "
        "FROM clause.");
  }
  CodeGenerator code_generator(executor_);

  auto key_lv = HashJoin::codegenColOrStringOper(
      key_col, inner_outer_string_op_infos_.second, code_generator, co);

  auto hash_join_idx_args = getHashJoinArgs(pos_ptr, key_lv, key_col, shard_count, co);
  const int64_t sub_buff_size = getComponentBufferSize();
  const auto& key_col_ti = key_col->get_type_info();

  auto bucketize = (key_col_ti.get_type() == kDATE);
  return HashJoin::codegenMatchingSet(hash_join_idx_args,
                                      shard_count,
                                      !key_col_ti.get_notnull(),
                                      isBitwiseEq(),
                                      sub_buff_size,
                                      executor_,
                                      bucketize);
}

size_t PerfectJoinHashTable::offsetBufferOff() const noexcept {
  return 0;
}

size_t PerfectJoinHashTable::countBufferOff() const noexcept {
  return getComponentBufferSize();
}

size_t PerfectJoinHashTable::payloadBufferOff() const noexcept {
  return 2 * getComponentBufferSize();
}

size_t PerfectJoinHashTable::getComponentBufferSize() const noexcept {
  if (hash_tables_for_device_.empty()) {
    return 0;
  }
  auto hash_table = hash_tables_for_device_.front();
  if (hash_table && hash_table->getLayout() == HashType::OneToMany) {
    return hash_table->getEntryCount() * sizeof(int32_t);
  } else {
    return 0;
  }
}

HashTable* PerfectJoinHashTable::getHashTableForDevice(const size_t device_id) const {
  CHECK_LT(device_id, hash_tables_for_device_.size());
  return hash_tables_for_device_[device_id].get();
}

void PerfectJoinHashTable::copyCpuHashTableToGpu(
    std::shared_ptr<PerfectHashTable>& cpu_hash_table,
    const int device_id,
    Data_Namespace::DataMgr* data_mgr) {
  CHECK_EQ(memory_level_, Data_Namespace::MemoryLevel::GPU_LEVEL);
  CHECK(data_mgr);
  CHECK(cpu_hash_table);

  std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
  PerfectJoinHashTableBuilder gpu_builder;
  gpu_builder.allocateDeviceMemory(cpu_hash_table->getColumnNumElems(),
                                   cpu_hash_table->getLayout(),
                                   cpu_hash_table->getHashEntryInfo(),
                                   shardCount(),
                                   device_id,
                                   device_count_,
                                   executor_);

  std::shared_ptr<PerfectHashTable> gpu_hash_table = gpu_builder.getHashTable();
  CHECK(gpu_hash_table);
  auto gpu_buffer_ptr = gpu_hash_table->getGpuBuffer();
  CHECK(gpu_buffer_ptr);

  CHECK_LE(cpu_hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU),
           gpu_hash_table->getHashTableBufferSize(ExecutorDeviceType::GPU));

  auto device_allocator = std::make_unique<CudaAllocator>(
      data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id));
  device_allocator->copyToDevice(
      gpu_buffer_ptr,
      cpu_hash_table->getCpuBuffer(),
      cpu_hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU));
  CHECK_LT(static_cast<size_t>(device_id), hash_tables_for_device_.size());
  hash_tables_for_device_[device_id] = std::move(gpu_hash_table);
}

std::string PerfectJoinHashTable::toString(const ExecutorDeviceType device_type,
                                           const int device_id,
                                           bool raw) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto buffer_size = getJoinHashBufferSize(device_type, device_id);
  auto hash_table = getHashTableForDevice(device_id);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);

    auto data_mgr = executor_->getDataMgr();
    auto device_allocator = std::make_unique<CudaAllocator>(
        data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id));
    device_allocator->copyFromDevice(buffer_copy.get(), buffer, buffer_size);
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : reinterpret_cast<const int8_t*>(buffer);
#else
  auto ptr1 = reinterpret_cast<const int8_t*>(buffer);
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  return HashTable::toString("perfect",
                             getHashTypeString(hash_type_),
                             0,
                             0,
                             hash_table ? hash_table->getEntryCount() : 0,
                             ptr1,
                             ptr2,
                             ptr3,
                             ptr4,
                             buffer_size,
                             raw);
}

std::set<DecodedJoinHashBufferEntry> PerfectJoinHashTable::toSet(
    const ExecutorDeviceType device_type,
    const int device_id) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto buffer_size = getJoinHashBufferSize(device_type, device_id);
  auto hash_table = getHashTableForDevice(device_id);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);

    auto data_mgr = executor_->getDataMgr();
    auto device_allocator = std::make_unique<CudaAllocator>(
        data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id));
    device_allocator->copyFromDevice(buffer_copy.get(), buffer, buffer_size);
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : reinterpret_cast<const int8_t*>(buffer);
#else
  auto ptr1 = reinterpret_cast<const int8_t*>(buffer);
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  return HashTable::toSet(0,
                          0,
                          hash_table ? hash_table->getEntryCount() : 0,
                          ptr1,
                          ptr2,
                          ptr3,
                          ptr4,
                          buffer_size);
}

llvm::Value* PerfectJoinHashTable::codegenSlot(const CompilationOptions& co,
                                               const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  using namespace std::string_literals;

  CHECK(getHashType() == HashType::OneToOne);
  const auto cols_and_string_op_infos = get_cols(
      qual_bin_oper_.get(), *executor_->getCatalog(), executor_->temporary_tables_);
  const auto& cols = cols_and_string_op_infos.first;
  const auto& inner_outer_string_op_infos = cols_and_string_op_infos.second;
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  CodeGenerator code_generator(executor_);
  const auto key_col_var = dynamic_cast<const Analyzer::ColumnVar*>(key_col);
  const auto val_col_var = dynamic_cast<const Analyzer::ColumnVar*>(val_col);
  if (key_col_var && val_col_var &&
      self_join_not_covered_by_left_deep_tree(
          key_col_var,
          val_col_var,
          get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
    throw std::runtime_error(
        "Query execution failed because the query contains not supported self-join "
        "pattern. We suspect the query requires multiple left-deep join tree due to "
        "the join condition of the self-join and is not supported for now. Please "
        "consider chaning the table order in the FROM clause.");
  }

  auto key_lv = HashJoin::codegenColOrStringOper(
      key_col, inner_outer_string_op_infos.second, code_generator, co);

  // CHECK_EQ(size_t(1), key_lvs.size());
  auto hash_ptr = codegenHashTableLoad(index);
  CHECK(hash_ptr);
  const int shard_count = shardCount();
  const auto hash_join_idx_args =
      getHashJoinArgs(hash_ptr, key_lv, key_col, shard_count, co);

  const auto& key_col_ti = key_col->get_type_info();
  std::string fname((key_col_ti.get_type() == kDATE) ? "bucketized_hash_join_idx"s
                                                     : "hash_join_idx"s);

  if (isBitwiseEq()) {
    fname += "_bitwise";
  }
  if (shard_count) {
    fname += "_sharded";
  }

  if (!isBitwiseEq() && !key_col_ti.get_notnull()) {
    fname += "_nullable";
  }
  return executor_->cgen_state_->emitCall(fname, hash_join_idx_args);
}

const InputTableInfo& PerfectJoinHashTable::getInnerQueryInfo(
    const Analyzer::ColumnVar* inner_col) const {
  return get_inner_query_info(inner_col->get_table_id(), query_infos_);
}

const InputTableInfo& get_inner_query_info(
    const int inner_table_id,
    const std::vector<InputTableInfo>& query_infos) {
  std::optional<size_t> ti_idx;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (inner_table_id == query_infos[i].table_id) {
      ti_idx = i;
      break;
    }
  }
  CHECK(ti_idx);
  return query_infos[*ti_idx];
}

size_t get_entries_per_device(const size_t total_entries,
                              const size_t shard_count,
                              const size_t device_count,
                              const Data_Namespace::MemoryLevel memory_level) {
  const auto entries_per_shard =
      shard_count ? (total_entries + shard_count - 1) / shard_count : total_entries;
  size_t entries_per_device = entries_per_shard;
  if (memory_level == Data_Namespace::GPU_LEVEL && shard_count) {
    const auto shards_per_device = (shard_count + device_count - 1) / device_count;
    CHECK_GT(shards_per_device, 0u);
    entries_per_device = entries_per_shard * shards_per_device;
  }
  return entries_per_device;
}

size_t PerfectJoinHashTable::shardCount() const {
  return memory_level_ == Data_Namespace::GPU_LEVEL
             ? get_shard_count(qual_bin_oper_.get(), executor_)
             : 0;
}

bool PerfectJoinHashTable::isBitwiseEq() const {
  return qual_bin_oper_->get_optype() == kBW_EQ;
}
