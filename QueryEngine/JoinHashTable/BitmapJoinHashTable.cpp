/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/JoinHashTable/BitmapJoinHashTable.h"

#include <future>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include "DataMgr/Allocators/CudaAllocator.h"
#include "Logger/Logger.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/DataRecycler/HashtableRecycler.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/JoinHashTable/BitmapHashTable.h"
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "Shared/InlineNullValues.h"

std::unique_ptr<HashtableRecycler> BitmapJoinHashTable::hash_table_cache_ =
    std::make_unique<HashtableRecycler>(CacheItemType::BITMAP_HT,
                                        DataRecyclerUtil::MAX_GPU_CACHE_DEVICE_COUNT);

namespace {

DeviceIdentifier gpu_cache_device_identifier(const int device_id) {
  CHECK_GE(device_id, 0);
  return static_cast<DeviceIdentifier>(device_id + 1);
}

QueryPlanHash hash_bitmap_cache_key(const std::string& key) {
  auto hashed_key = boost::hash_value(key);
  if (hashed_key == EMPTY_HASHED_PLAN_DAG_KEY) {
    boost::hash_combine(hashed_key, key.size());
  }
  return hashed_key;
}

void mark_bitmap_cache_key_for_tables(const QueryPlanHash key,
                                      const std::unordered_set<size_t>& table_keys) {
  if (key == EMPTY_HASHED_PLAN_DAG_KEY || table_keys.empty()) {
    return;
  }
  BitmapJoinHashTable::getHashTableCache()->addQueryPlanDagForTableKeys(key, table_keys);
}

bool supported_bitmap_type(const SQLTypeInfo& ti) {
  return (ti.is_integer() || ti.is_time() || ti.is_boolean()) && !ti.is_string() &&
         !ti.is_array() && !ti.is_fp();
}

size_t checked_bit_count(const ExpressionRange& range) {
  if (range.getIntMin() > range.getIntMax()) {
    return 0;
  }
  const auto bit_count = static_cast<__int128>(range.getIntMax()) -
                         static_cast<__int128>(range.getIntMin()) + 1;
  if (bit_count < 0) {
    throw TooManyHashEntries("Bitmap join range is invalid");
  }
  if (bit_count > std::numeric_limits<size_t>::max()) {
    throw TooManyHashEntries("Bitmap join range exceeds addressable host size");
  }
  return static_cast<size_t>(bit_count);
}

void check_bitmap_size(const size_t bitmap_bytes,
                       const size_t tuple_count,
                       const size_t key_width) {
  const size_t baseline_entry_width = 2 * key_width;
  const auto baseline_size_estimate =
      tuple_count > std::numeric_limits<size_t>::max() / (2 * baseline_entry_width)
          ? std::numeric_limits<size_t>::max()
          : 2 * tuple_count * baseline_entry_width;
  if (bitmap_bytes > baseline_size_estimate) {
    std::ostringstream oss;
    oss << "Bitmap join range is too sparse for an exact bitmap (# bitmap bytes: "
        << bitmap_bytes << ", estimated baseline bytes: " << baseline_size_estimate
        << ", # input rows: " << tuple_count << ")";
    throw TooManyHashEntries(oss.str());
  }
}

std::string bitmap_temp_build_cache_key(const Analyzer::ColumnVar* inner_col,
                                        const ExpressionRange& col_range,
                                        const size_t bit_count,
                                        const size_t bitmap_bytes,
                                        const Data_Namespace::MemoryLevel memory_level,
                                        const JoinType join_type,
                                        const std::set<int>& device_ids) {
  CHECK(inner_col);
  const auto column_key = inner_col->getColumnKey();
  if (column_key.table_id >= 0) {
    return {};
  }

  std::ostringstream oss;
  oss << "bitmap-temp:v1"
      << "|db=" << column_key.db_id << "|table=" << column_key.table_id
      << "|col=" << column_key.column_id << "|range_min=" << col_range.getIntMin()
      << "|range_max=" << col_range.getIntMax() << "|bits=" << bit_count
      << "|bytes=" << bitmap_bytes << "|memory=" << static_cast<int>(memory_level)
      << "|join_type=" << static_cast<int>(join_type) << "|devices=";
  for (const auto device_id : device_ids) {
    oss << device_id << ",";
  }
  return oss.str();
}

std::string bitmap_stored_build_cache_key(
    const Analyzer::ColumnVar* inner_col,
    const ExpressionRange& col_range,
    const size_t bit_count,
    const size_t bitmap_bytes,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const SQLOps op_type,
    const std::set<int>& device_ids,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  CHECK(inner_col);
  const auto column_key = inner_col->getColumnKey();
  if (column_key.table_id <= 0) {
    return {};
  }

  std::ostringstream oss;
  oss << "bitmap-stored:v1"
      << "|db=" << column_key.db_id << "|table=" << column_key.table_id
      << "|col=" << column_key.column_id
      << "|type=" << inner_col->get_type_info().toString()
      << "|range_min=" << col_range.getIntMin() << "|range_max=" << col_range.getIntMax()
      << "|bits=" << bit_count << "|bytes=" << bitmap_bytes
      << "|memory=" << static_cast<int>(memory_level)
      << "|join_type=" << static_cast<int>(join_type)
      << "|op=" << static_cast<int>(op_type) << "|fragments=";
  for (const auto& fragment : fragments) {
    oss << fragment.fragmentId << ",";
  }
  oss << "|devices=";
  for (const auto device_id : device_ids) {
    oss << device_id << ",";
  }
  return oss.str();
}

std::string bitmap_semantic_build_cache_key(
    const Analyzer::ColumnVar* inner_col,
    const Analyzer::Expr* outer_expr,
    const ExpressionRange& col_range,
    const size_t bit_count,
    const size_t bitmap_bytes,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const SQLOps op_type,
    const std::set<int>& device_ids,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    Executor* executor,
    std::unordered_set<size_t>* table_keys) {
  CHECK(inner_col);
  CHECK(outer_expr);
  CHECK(executor);
  if (!HashtableRecycler::isSafeToCacheHashtable(
          table_id_to_node_map, false, {}, inner_col->getTableKey())) {
    return {};
  }

  std::unordered_map<int, std::vector<Fragmenter_Namespace::FragmentInfo>>
      frags_for_device;
  for (const auto device_id : device_ids) {
    frags_for_device.emplace(device_id, fragments);
  }
  const std::vector<InnerOuter> inner_outer_pairs{{inner_col, outer_expr}};
  const auto access_path_info =
      HashtableRecycler::getHashtableAccessPathInfo(inner_outer_pairs,
                                                    {},
                                                    op_type,
                                                    join_type,
                                                    hashtable_build_dag_map,
                                                    device_ids,
                                                    0,
                                                    frags_for_device,
                                                    executor);
  if (HashtableRecycler::isInvalidHashTableCacheKey(
          access_path_info.hashed_query_plan_dag)) {
    return {};
  }
  if (table_keys) {
    *table_keys = access_path_info.table_keys;
  }

  std::ostringstream oss;
  oss << "bitmap-semantic:v1";
  for (const auto device_id : device_ids) {
    oss << "|device=" << device_id
        << ":plan=" << access_path_info.hashed_query_plan_dag.at(device_id);
  }
  oss << "|range_min=" << col_range.getIntMin() << "|range_max=" << col_range.getIntMax()
      << "|bits=" << bit_count << "|bytes=" << bitmap_bytes
      << "|memory=" << static_cast<int>(memory_level)
      << "|join_type=" << static_cast<int>(join_type) << "|devices=";
  for (const auto device_id : device_ids) {
    oss << device_id << ",";
  }
  return oss.str();
}

struct BitmapCacheKey {
  std::string key;
  std::string source;
  std::unordered_set<size_t> table_keys;
  bool globally_recyclable{false};
};

BitmapCacheKey bitmap_build_cache_key(
    const Analyzer::ColumnVar* inner_col,
    const Analyzer::Expr* outer_expr,
    const ExpressionRange& col_range,
    const size_t bit_count,
    const size_t bitmap_bytes,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const SQLOps op_type,
    const std::set<int>& device_ids,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    Executor* executor) {
  std::unordered_set<size_t> semantic_table_keys;
  auto semantic_key = bitmap_semantic_build_cache_key(inner_col,
                                                      outer_expr,
                                                      col_range,
                                                      bit_count,
                                                      bitmap_bytes,
                                                      memory_level,
                                                      join_type,
                                                      op_type,
                                                      device_ids,
                                                      hashtable_build_dag_map,
                                                      table_id_to_node_map,
                                                      fragments,
                                                      executor,
                                                      &semantic_table_keys);
  if (!semantic_key.empty()) {
    if (semantic_table_keys.empty() && inner_col->getTableKey().table_id > 0) {
      semantic_table_keys.insert(inner_col->getTableKey().hash());
    }
    const bool globally_recyclable = !semantic_table_keys.empty();
    return {std::move(semantic_key),
            "semantic",
            std::move(semantic_table_keys),
            globally_recyclable};
  }
  auto stored_key = bitmap_stored_build_cache_key(inner_col,
                                                  col_range,
                                                  bit_count,
                                                  bitmap_bytes,
                                                  memory_level,
                                                  join_type,
                                                  op_type,
                                                  device_ids,
                                                  fragments);
  if (!stored_key.empty()) {
    return {
        std::move(stored_key), "stored-table", {inner_col->getTableKey().hash()}, true};
  }
  return {bitmap_temp_build_cache_key(inner_col,
                                      col_range,
                                      bit_count,
                                      bitmap_bytes,
                                      memory_level,
                                      join_type,
                                      device_ids),
          "temporary-table",
          {},
          false};
}

void reuse_bitmap_hash_tables(BitmapJoinHashTable& target,
                              const std::shared_ptr<BitmapJoinHashTable>& cached,
                              const std::set<int>& device_ids) {
  CHECK(cached);
  for (const auto device_id : device_ids) {
    target.putHashTableForDevice(cached->getHashTableForDevice(device_id), device_id);
  }
}

}  // namespace

HashtableRecycler* BitmapJoinHashTable::getHashTableCache() {
  CHECK(hash_table_cache_);
  return hash_table_cache_.get();
}

void BitmapJoinHashTable::invalidateCache() {
  CHECK(hash_table_cache_);
  hash_table_cache_->clearCache();
}

void BitmapJoinHashTable::markCachedItemAsDirty(size_t table_key) {
  CHECK(hash_table_cache_);
  auto candidate_table_keys =
      hash_table_cache_->getMappedQueryPlanDagsWithTableKey(table_key);
  if (!candidate_table_keys.has_value()) {
    return;
  }
  for (int device_identifier = 1;
       device_identifier <= DataRecyclerUtil::MAX_GPU_CACHE_DEVICE_COUNT;
       ++device_identifier) {
    hash_table_cache_->markCachedItemAsDirty(
        table_key, *candidate_table_keys, CacheItemType::BITMAP_HT, device_identifier);
  }
}

std::shared_ptr<BitmapJoinHashTable> BitmapJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const std::set<int>& device_ids,
    ColumnCacheMap& column_cache,
    Executor* executor,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map) {
  if (join_type != JoinType::SEMI && join_type != JoinType::ANTI) {
    throw HashJoinFail("Bitmap join is only valid for SEMI/ANTI joins");
  }
  if (!IS_EQUIVALENCE(condition->get_optype()) || condition->get_optype() == kBW_EQ) {
    throw HashJoinFail("Bitmap join only supports regular equality predicates");
  }
  const auto normalized =
      HashJoin::normalizeColumnPairs(condition.get(), executor->getTemporaryTables());
  if (normalized.first.size() != 1 || normalized.second.size() != 1 ||
      normalized.second.front().first.size() || normalized.second.front().second.size()) {
    throw HashJoinFail("Bitmap join only supports one integer column pair");
  }
  const auto inner_col = normalized.first.front().first;
  const auto outer_expr = normalized.first.front().second;
  CHECK(inner_col);
  CHECK(outer_expr);
  const auto inner_logical_ti = HashJoin::getColumnTypeForJoin(
      inner_col, executor->getTemporaryTables(), table_id_to_node_map, true);
  const auto outer_logical_ti = HashJoin::getExpressionTypeForJoin(
      outer_expr, executor->getTemporaryTables(), table_id_to_node_map, true);
  if (inner_logical_ti.is_string() || outer_logical_ti.is_string()) {
    throw HashJoinFail("Bitmap join does not support dictionary-encoded string keys");
  }
  if (!supported_bitmap_type(inner_logical_ti) ||
      !supported_bitmap_type(outer_logical_ti)) {
    throw HashJoinFail("Bitmap join only supports integer-like join keys");
  }
  const auto& ti = inner_col->get_type_info();

  auto col_range = getExpressionRange(inner_col, query_infos, executor);
  if (col_range.getType() == ExpressionRangeType::Invalid) {
    throw HashJoinFail("Could not compute range for bitmap join key");
  }
  const auto bit_count = checked_bit_count(col_range);
  if (bit_count == 0) {
    throw HashJoinFail("Bitmap join input range is empty");
  }
  const auto bitmap_bytes =
      BitmapHashTable::wordsForBytes(BitmapHashTable::bytesForBits(bit_count)) *
      sizeof(uint32_t);
  const auto& inner_table_info =
      get_inner_query_info(inner_col->getTableKey(), query_infos).info;
  const auto tuple_count = get_hash_join_table_num_tuples(inner_table_info);
  const auto key_width = std::max<size_t>(sizeof(int32_t), ti.get_logical_size());
  check_bitmap_size(bitmap_bytes, tuple_count, key_width);
  auto cache_key_info = bitmap_build_cache_key(inner_col,
                                               outer_expr,
                                               col_range,
                                               bit_count,
                                               bitmap_bytes,
                                               memory_level,
                                               join_type,
                                               condition->get_optype(),
                                               device_ids,
                                               hashtable_build_dag_map,
                                               table_id_to_node_map,
                                               inner_table_info.fragments,
                                               executor);

  auto join_hash_table =
      std::shared_ptr<BitmapJoinHashTable>(new BitmapJoinHashTable(condition,
                                                                   inner_col,
                                                                   outer_expr,
                                                                   col_range,
                                                                   bit_count,
                                                                   bitmap_bytes,
                                                                   query_infos,
                                                                   memory_level,
                                                                   join_type,
                                                                   device_ids,
                                                                   column_cache,
                                                                   executor));
  const auto row_set_mem_owner = executor->getRowSetMemoryOwner();
  if (!cache_key_info.key.empty() && row_set_mem_owner) {
    if (const auto cached_base =
            row_set_mem_owner->getCachedJoinHashTable(cache_key_info.key)) {
      auto cached = std::dynamic_pointer_cast<BitmapJoinHashTable>(cached_base);
      if (cached) {
        CHECK_EQ(cached->bit_count_, bit_count);
        CHECK_EQ(cached->bitmap_bytes_, bitmap_bytes);
        CHECK_EQ(cached->col_range_.getIntMin(), col_range.getIntMin());
        CHECK_EQ(cached->col_range_.getIntMax(), col_range.getIntMax());
        CHECK_EQ(cached->memory_level_, memory_level);
        reuse_bitmap_hash_tables(*join_hash_table, cached, device_ids);
        return join_hash_table;
      }
    }
  }
  const bool can_use_global_gpu_cache =
      cache_key_info.globally_recyclable &&
      memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL &&
      !cache_key_info.key.empty();
  const auto global_cache_key = can_use_global_gpu_cache
                                    ? hash_bitmap_cache_key(cache_key_info.key)
                                    : EMPTY_HASHED_PLAN_DAG_KEY;
  if (can_use_global_gpu_cache) {
    bool found_all_devices = true;
    std::map<int, std::shared_ptr<HashTable>> cached_hash_tables;
    for (const auto device_id : device_ids) {
      auto cached_hash_table =
          hash_table_cache_->getItemFromCache(global_cache_key,
                                              CacheItemType::BITMAP_HT,
                                              gpu_cache_device_identifier(device_id));
      if (!std::dynamic_pointer_cast<BitmapHashTable>(cached_hash_table)) {
        found_all_devices = false;
        break;
      }
      cached_hash_tables.emplace(device_id, std::move(cached_hash_table));
    }
    if (found_all_devices) {
      for (auto& [device_id, hash_table] : cached_hash_tables) {
        join_hash_table->putHashTableForDevice(std::move(hash_table), device_id);
      }
      return join_hash_table;
    }
  }
  join_hash_table->reify();
  if (can_use_global_gpu_cache) {
    mark_bitmap_cache_key_for_tables(global_cache_key, cache_key_info.table_keys);
    for (const auto device_id : device_ids) {
      auto hash_table = join_hash_table->getHashTableForDevice(device_id);
      hash_table_cache_->putItemToCache(
          global_cache_key,
          hash_table,
          CacheItemType::BITMAP_HT,
          gpu_cache_device_identifier(device_id),
          hash_table->getHashTableBufferSize(ExecutorDeviceType::GPU),
          0);
    }
  }
  if (!cache_key_info.key.empty() && row_set_mem_owner) {
    row_set_mem_owner->putCachedJoinHashTable(cache_key_info.key, join_hash_table);
  }
  return join_hash_table;
}

BitmapJoinHashTable::BitmapJoinHashTable(
    const std::shared_ptr<Analyzer::BinOper> condition,
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
    Executor* executor)
    : condition_(condition)
    , inner_col_(std::dynamic_pointer_cast<Analyzer::ColumnVar>(inner_col->deep_copy()))
    , outer_expr_(outer_expr->deep_copy())
    , col_range_(col_range)
    , bit_count_(bit_count)
    , bitmap_bytes_(bitmap_bytes)
    , query_infos_(query_infos)
    , memory_level_(memory_level)
    , join_type_(join_type)
    , device_ids_(device_ids)
    , column_cache_(column_cache)
    , executor_(executor) {
  CHECK(condition_);
  CHECK(inner_col_);
  CHECK(outer_expr_);
  CHECK_GT(device_ids_.size(), 0u);
}

void BitmapJoinHashTable::reify() {
  const auto& query_info = get_inner_query_info(getInnerTableId(), query_infos_).info;
  if (query_info.fragments.empty()) {
    return;
  }

  std::vector<std::future<void>> init_threads;
  for (const auto device_id : device_ids_) {
    init_threads.push_back(std::async(std::launch::async,
                                      &BitmapJoinHashTable::reifyForDevice,
                                      this,
                                      query_info.fragments,
                                      device_id,
                                      logger::thread_local_ids()));
  }
  for (auto& init_thread : init_threads) {
    init_thread.wait();
  }
  for (auto& init_thread : init_threads) {
    init_thread.get();
  }
}

void BitmapJoinHashTable::reifyForDevice(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    const logger::ThreadLocalIds parent_thread_local_ids) {
  logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();

  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<std::shared_ptr<void>> malloc_owner;
  DeviceAllocator* device_allocator{nullptr};
  const auto effective_memory_level = memory_level_;
#ifdef HAVE_CUDA
  if (effective_memory_level == Data_Namespace::GPU_LEVEL) {
    device_allocator = executor_->getCudaAllocator(device_id);
    CHECK(device_allocator);
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif

  auto join_column = fetchJoinColumn(inner_col_.get(),
                                     fragments,
                                     effective_memory_level,
                                     device_id,
                                     chunks_owner,
                                     device_allocator,
                                     malloc_owner,
                                     executor_,
                                     &column_cache_);
  // The fetched build column retains its physical encoding. Match the established
  // perfect-hash build path so fixed-width and DATE_IN_DAYS values are decoded by the
  // JoinColumn iterator with the correct element width and null sentinel.
  const auto& physical_ti = inner_col_->get_type_info();
  JoinColumnTypeInfo type_info{static_cast<size_t>(physical_ti.get_size()),
                               col_range_.getIntMin(),
                               col_range_.getIntMax(),
                               inline_fixed_encoding_null_val(physical_ti),
                               false,
                               0,
                               get_join_column_type_kind(physical_ti)};

  std::shared_ptr<BitmapHashTable> bitmap_hash_table;
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    bitmap_hash_table = std::make_shared<BitmapHashTable>(
        ExecutorDeviceType::CPU, bit_count_, executor_->maxCpuSlabSize());
    const int thread_count = cpu_threads();
    std::vector<std::future<void>> fill_threads;
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      if (bitmap_hash_table->hasSegmentedLayout()) {
        fill_threads.emplace_back(
            std::async(std::launch::async,
                       fill_join_bitmap_segmented,
                       bitmap_hash_table->getCpuBitmapPtrs(),
                       static_cast<int64_t>(bitmap_hash_table->getBitmapChunkWordCount()),
                       join_column,
                       type_info,
                       make_empty_join_column_filter(),
                       col_range_.getIntMin(),
                       col_range_.getIntMax(),
                       thread_idx,
                       thread_count));
      } else {
        fill_threads.emplace_back(std::async(std::launch::async,
                                             fill_join_bitmap,
                                             bitmap_hash_table->getCpuBitmap(),
                                             join_column,
                                             type_info,
                                             make_empty_join_column_filter(),
                                             col_range_.getIntMin(),
                                             col_range_.getIntMax(),
                                             thread_idx,
                                             thread_count));
      }
    }
    for (auto& child : fill_threads) {
      child.wait();
    }
    for (auto& child : fill_threads) {
      child.get();
    }
  } else {
#ifdef HAVE_CUDA
    bitmap_hash_table = std::make_shared<BitmapHashTable>(ExecutorDeviceType::GPU,
                                                          bit_count_,
                                                          executor_->maxGpuSlabSize(),
                                                          executor_->getDataMgr(),
                                                          device_id);
    const auto cuda_stream = executor_->getCudaStream(device_id);
    if (bitmap_hash_table->hasSegmentedLayout()) {
      bitmap_hash_table->copyGpuHeaderToDevice(device_allocator);
      for (size_t chunk_idx = 0; chunk_idx < bitmap_hash_table->getBitmapChunkCount();
           ++chunk_idx) {
        device_allocator->zeroDeviceMem(
            reinterpret_cast<int8_t*>(bitmap_hash_table->getGpuBitmapChunk(chunk_idx)),
            bitmap_hash_table->getBitmapChunkWordCount(chunk_idx) * sizeof(uint32_t));
      }
      fill_join_bitmap_segmented_on_device(bitmap_hash_table->getGpuBitmapPtrs(),
                                           bitmap_hash_table->getBitmapChunkWordCount(),
                                           join_column,
                                           type_info,
                                           make_empty_join_column_filter(),
                                           col_range_.getIntMin(),
                                           col_range_.getIntMax(),
                                           cuda_stream);
    } else {
      device_allocator->zeroDeviceMem(bitmap_hash_table->getGpuBuffer(),
                                      bitmap_hash_table->getAllocatedBytes());
      fill_join_bitmap_on_device(
          reinterpret_cast<uint32_t*>(bitmap_hash_table->getGpuBuffer()),
          join_column,
          type_info,
          make_empty_join_column_filter(),
          col_range_.getIntMin(),
          col_range_.getIntMax(),
          cuda_stream);
    }
#else
    UNREACHABLE();
#endif
  }
  std::shared_ptr<HashTable> hash_table = std::move(bitmap_hash_table);
  moveHashTableForDevice(std::move(hash_table), device_id);
}

llvm::Value* BitmapJoinHashTable::codegenSlot(const CompilationOptions& co,
                                              const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->getCgenStatePtr());
  CodeGenerator code_generator(executor_);
  auto key_lv =
      HashJoin::codegenColOrStringOper(outer_expr_.get(), {}, code_generator, co);
  auto cgen_state = executor_->getCgenStatePtr();
  auto hash_ptr = HashJoin::codegenHashTableLoad(index, executor_);
  auto bitmap_ptr_ty = llvm::Type::getInt8PtrTy(cgen_state->context_);
  llvm::Value* bitmap_ptr{nullptr};
  if (hash_ptr->getType()->isPointerTy()) {
    bitmap_ptr = cgen_state->ir_builder_.CreatePointerCast(hash_ptr, bitmap_ptr_ty);
  } else {
    bitmap_ptr = cgen_state->ir_builder_.CreateIntToPtr(hash_ptr, bitmap_ptr_ty);
  }

  const auto key_i64 = cgen_state->castToTypeIn(key_lv, 64);
  const auto key_ti = get_logical_type_info(outer_expr_->get_type_info());
  const auto hash_table = std::dynamic_pointer_cast<BitmapHashTable>(
      getHashTableForDevice(*device_ids_.begin()));
  CHECK(hash_table);
  auto is_set = cgen_state->emitCall(
      hash_table->hasSegmentedLayout() ? "segmented_bit_is_set" : "bit_is_set",
      {bitmap_ptr,
       key_i64,
       cgen_state->llInt(col_range_.getIntMin()),
       cgen_state->llInt(col_range_.getIntMax()),
       cgen_state->llInt(inline_fixed_encoding_null_val(key_ti)),
       cgen_state->llInt(int8_t(0))});
  // bit_is_set returns SQL BOOLEAN NULL for a null probe key. NULL is not a matching
  // row for either SEMI or ANTI join loop semantics, so only the exact TRUE value is a
  // membership hit.
  auto match = cgen_state->ir_builder_.CreateICmpEQ(is_set, cgen_state->llInt(int8_t(1)));
  return cgen_state->ir_builder_.CreateSelect(
      match, cgen_state->llInt(int64_t(0)), cgen_state->llInt(int64_t(-1)));
}

HashJoinMatchingSet BitmapJoinHashTable::codegenMatchingSet(const CompilationOptions&,
                                                            const size_t) {
  UNREACHABLE();
  return {nullptr, nullptr, nullptr, nullptr};
}

std::string BitmapJoinHashTable::toString(const ExecutorDeviceType device_type,
                                          const int device_id,
                                          bool) const {
  auto hash_table = getHashTableForDevice(device_id);
  const auto bitmap_hash_table = std::dynamic_pointer_cast<BitmapHashTable>(hash_table);
  std::ostringstream oss;
  oss << (bitmap_hash_table && bitmap_hash_table->hasSegmentedLayout()
              ? "segmented bitmap "
              : "bitmap ")
      << ::toString(device_type) << " join hash table"
      << ", bits: " << bit_count_ << ", bytes: "
      << (hash_table ? hash_table->getHashTableBufferSize(device_type) : bitmap_bytes_)
      << ", range: [" << col_range_.getIntMin() << ", " << col_range_.getIntMax() << "]";
  return oss.str();
}
