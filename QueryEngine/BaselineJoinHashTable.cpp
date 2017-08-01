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

#include "BaselineJoinHashTable.h"
#include "ExpressionRewrite.h"
#include "Execute.h"
#include "HashJoinRuntime.h"

#include <future>
#include <type_traits>

std::vector<std::pair<BaselineJoinHashTable::HashTableCacheKey, std::shared_ptr<std::vector<int8_t>>>>
    BaselineJoinHashTable::hash_table_cache_;
std::mutex BaselineJoinHashTable::hash_table_cache_mutex_;

std::shared_ptr<BaselineJoinHashTable> BaselineJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> condition_in,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    const std::unordered_set<int>& skip_tables,
    Executor* executor) {
  const auto condition =
      std::dynamic_pointer_cast<Analyzer::BinOper>(redirect_expr(condition_in.get(), ra_exe_unit.input_col_descs));
  // Already handled the table
  if (skip_tables.count(getInnerTableId(condition.get(), executor))) {
    throw HashJoinFail("A hash table is already built for the table of this column");
  }
  const auto& query_info = get_inner_query_info(getInnerTableId(condition.get(), executor), query_infos).info;
  const auto shard_count_info = memory_level == Data_Namespace::GPU_LEVEL
                                    ? get_baseline_shard_count(condition.get(), ra_exe_unit, executor)
                                    : ShardCountInfo{0, nullptr};
  const auto shard_count = shard_count_info.count;
  const auto total_entries = 2 * query_info.getNumTuplesUpperBound();
  if (total_entries > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw TooManyHashEntries();
  }
  const auto entries_per_shard = shard_count ? (total_entries + shard_count - 1) / shard_count : total_entries;
  size_t entries_per_device = entries_per_shard;
  if (memory_level == Data_Namespace::GPU_LEVEL && shard_count) {
    const auto shards_per_device = (shard_count + device_count - 1) / device_count;
    CHECK_GT(shards_per_device, 0);
    entries_per_device = entries_per_shard * shards_per_device;
  }
  auto join_hash_table = std::shared_ptr<BaselineJoinHashTable>(
      new BaselineJoinHashTable(condition, query_infos, ra_exe_unit, memory_level, entries_per_device, executor));
  join_hash_table->checkHashJoinReplicationConstraint(getInnerTableId(condition.get(), executor));
  const int err = join_hash_table->reify(device_count);
  if (err) {
    throw HashJoinFail("Could not build a 1-to-1 correspondence for columns involved in equijoin");
  }
  return join_hash_table;
}

BaselineJoinHashTable::BaselineJoinHashTable(const std::shared_ptr<Analyzer::BinOper> condition,
                                             const std::vector<InputTableInfo>& query_infos,
                                             const RelAlgExecutionUnit& ra_exe_unit,
                                             const Data_Namespace::MemoryLevel memory_level,
                                             const size_t entry_count,
                                             Executor* executor)
    : condition_(condition),
      query_infos_(query_infos),
      memory_level_(memory_level),
      entry_count_(entry_count),
      executor_(executor),
      ra_exe_unit_(ra_exe_unit) {}

int64_t BaselineJoinHashTable::getJoinHashBuffer(const ExecutorDeviceType device_type, const int device_id) noexcept {
  if (device_type == ExecutorDeviceType::CPU && !cpu_hash_table_buff_) {
    return 0;
  }
#ifdef HAVE_CUDA
  CHECK_LT(static_cast<size_t>(device_id), gpu_hash_table_buff_.size());
  return device_type == ExecutorDeviceType::CPU ? reinterpret_cast<int64_t>(&(*cpu_hash_table_buff_)[0])
                                                : gpu_hash_table_buff_[device_id];
#else
  CHECK(device_type == ExecutorDeviceType::CPU);
  return reinterpret_cast<int64_t>(&(*cpu_hash_table_buff_)[0]);
#endif
}

int BaselineJoinHashTable::reify(const int device_count) {
  CHECK_LT(0, device_count);
#ifdef HAVE_CUDA
  gpu_hash_table_buff_.resize(device_count);
#endif  // HAVE_CUDA
  std::vector<int> errors(device_count);
  std::vector<std::thread> init_threads;
  const auto shard_info = computeShardCount();
  const auto& query_info = get_inner_query_info(getInnerTableId(), query_infos_).info;
  if (query_info.fragments.empty()) {
    return 0;
  }
  for (int device_id = 0; device_id < device_count; ++device_id) {
    const auto fragments =
        shard_info.count ? only_shards_for_device(query_info.fragments, device_id, device_count) : query_info.fragments;
    init_threads.emplace_back(
        [&errors, device_id, fragments, this] { errors[device_id] = reifyForDevice(fragments, device_id); });
  }
  for (auto& init_thread : init_threads) {
    init_thread.join();
  }
  for (const int err : errors) {
    if (err) {
      return err;
    }
  }
  return 0;
}

namespace {

typedef std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*> InnerOuter;

std::vector<InnerOuter> normalize_column_pairs(const Analyzer::BinOper* condition,
                                               const Catalog_Namespace::Catalog& cat,
                                               const TemporaryTables* temporary_tables) {
  std::vector<InnerOuter> result;
  const auto lhs_tuple_expr = dynamic_cast<const Analyzer::ColumnVarTuple*>(condition->get_left_operand());
  const auto rhs_tuple_expr = dynamic_cast<const Analyzer::ColumnVarTuple*>(condition->get_right_operand());
  CHECK(lhs_tuple_expr && rhs_tuple_expr);
  const auto& lhs_tuple = lhs_tuple_expr->getTuple();
  const auto& rhs_tuple = rhs_tuple_expr->getTuple();
  CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
  for (size_t i = 0; i < lhs_tuple.size(); ++i) {
    result.push_back(normalize_column_pair(lhs_tuple[i].get(), rhs_tuple[i].get(), cat, temporary_tables));
  }
  return result;
}

}  // namespace

int BaselineJoinHashTable::reifyForDevice(const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
                                          const int device_id) {
  static std::mutex fragment_fetch_mutex;
  const bool has_multi_frag = fragments.size() > 1;
  const auto& catalog = *executor_->getCatalog();
  const auto inner_outer_pairs = normalize_column_pairs(condition_.get(), catalog, executor_->getTemporaryTables());
  std::vector<JoinColumn> join_columns;
  std::vector<JoinColumnTypeInfo> join_column_types;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::map<int, std::shared_ptr<const ColumnarResults>> frags_owner;
  const auto& first_frag = fragments.front();
  auto effective_memory_level = memory_level_;
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    if (needs_dictionary_translation(inner_outer_pair.first, inner_outer_pair.second, executor_)) {
      effective_memory_level = Data_Namespace::CPU_LEVEL;
      break;
    }
  }
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto inner_col = inner_outer_pair.first;
    const auto& ti = inner_col->get_type_info();
    const auto inner_cd = get_column_descriptor_maybe(inner_col->get_column_id(), inner_col->get_table_id(), catalog);
    if (inner_cd && inner_cd->isVirtualCol) {
      return ERR_FAILED_TO_JOIN_ON_VIRTUAL_COLUMN;
    }
    const int8_t* col_buff = nullptr;
    size_t elem_count = 0;
    const size_t elem_width = inner_col->get_type_info().get_size();
    auto& data_mgr = catalog.get_dataMgr();
    ThrustAllocator dev_buff_owner(&data_mgr, device_id);
    if (has_multi_frag) {
      try {
        std::tie(col_buff, elem_count) = getAllColumnFragments(*inner_col, fragments, chunks_owner, frags_owner);
      } catch (...) {
        return ERR_FAILED_TO_FETCH_COLUMN;
      }
    }
    {
      std::lock_guard<std::mutex> fragment_fetch_lock(fragment_fetch_mutex);
      if (has_multi_frag) {
        if (effective_memory_level == Data_Namespace::GPU_LEVEL) {
          CHECK(col_buff != nullptr);
          CHECK_NE(elem_count, size_t(0));
          int8_t* dev_col_buff = nullptr;
          dev_col_buff = dev_buff_owner.allocate(elem_count * elem_width);
          copy_to_gpu(
              &data_mgr, reinterpret_cast<CUdeviceptr>(dev_col_buff), col_buff, elem_count * elem_width, device_id);
          col_buff = dev_col_buff;
        }
      } else {
        try {
          std::tie(col_buff, elem_count) = Executor::ExecutionDispatch::getColumnFragment(
              executor_, *inner_col, first_frag, effective_memory_level, device_id, chunks_owner, frags_owner);
        } catch (...) {
          return ERR_FAILED_TO_FETCH_COLUMN;
        }
      }
    }
    join_columns.emplace_back(JoinColumn{col_buff, elem_count});
    join_column_types.emplace_back(
        JoinColumnTypeInfo{static_cast<size_t>(ti.get_size()), 0, inline_fixed_encoding_null_val(ti)});
  }
  int err{0};
  try {
    err = initHashTableForDevice(join_columns, join_column_types, effective_memory_level, device_id);
  } catch (...) {
    return -1;
  }
  return err;
}

std::pair<const int8_t*, size_t> BaselineJoinHashTable::getAllColumnFragments(
    const Analyzer::ColumnVar& hash_col,
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    std::map<int, std::shared_ptr<const ColumnarResults>>& frags_owner) {
  std::lock_guard<std::mutex> linearized_multifrag_column_lock(linearized_multifrag_column_mutex_);
  auto linearized_column_cache_key = std::make_pair(hash_col.get_table_id(), hash_col.get_column_id());
  const auto cache_it = linearized_multifrag_columns_.find(linearized_column_cache_key);
  if (cache_it != linearized_multifrag_columns_.end()) {
    return cache_it->second;
  }
  const int8_t* col_buff;
  size_t total_elem_count;
  std::tie(col_buff, total_elem_count) =
      Executor::ExecutionDispatch::getAllColumnFragments(executor_, hash_col, fragments, chunks_owner, frags_owner);
  linearized_multifrag_column_owner_.addColBuffer(col_buff);
  const auto shard_count_info = shardCount();
  if (!shard_count_info.count) {
    const auto it_ok = linearized_multifrag_columns_.emplace(linearized_column_cache_key,
                                                             LinearizedColumn{col_buff, total_elem_count});
    CHECK(it_ok.second);
  }
  return {col_buff, total_elem_count};
}

ShardCountInfo BaselineJoinHashTable::shardCount() const {
  if (memory_level_ != Data_Namespace::GPU_LEVEL) {
    return {0, nullptr};
  }
  return computeShardCount();
}

ShardCountInfo BaselineJoinHashTable::computeShardCount() const {
  return get_baseline_shard_count(condition_.get(), ra_exe_unit_, executor_);
}

ShardCountInfo get_baseline_shard_count(const Analyzer::BinOper* condition,
                                        const RelAlgExecutionUnit& ra_exe_unit,
                                        const Executor* executor) {
  const auto inner_outer_pairs =
      normalize_column_pairs(condition, *executor->getCatalog(), executor->getTemporaryTables());
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto pair_shard_count = get_shard_count(inner_outer_pair, ra_exe_unit, executor);
    if (pair_shard_count) {
      return {pair_shard_count, inner_outer_pair.first};
    }
  }
  return {0, nullptr};
}

namespace {

size_t get_key_component_width(const std::shared_ptr<Analyzer::BinOper> condition, const Executor* executor) {
  const auto inner_outer_pairs =
      normalize_column_pairs(condition.get(), *executor->getCatalog(), executor->getTemporaryTables());
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto inner_col = inner_outer_pair.first;
    if (inner_col->get_type_info().get_type() == kBIGINT) {
      return 8;
    }
  }
  return 4;
}

template <class T>
T* transfer_pod_vector_to_gpu(const std::vector<T>& vec, Data_Namespace::DataMgr* data_mgr, const int device_id) {
  static_assert(std::is_pod<T>::value, "Transferring a vector to GPU only works for POD elements");
  const auto vec_bytes = vec.size() * sizeof(T);
  auto gpu_vec = alloc_gpu_mem(data_mgr, vec_bytes, device_id, nullptr);
  copy_to_gpu(data_mgr, gpu_vec, &vec[0], vec_bytes, device_id);
  return reinterpret_cast<T*>(gpu_vec);
}

}  // namespace

int BaselineJoinHashTable::initHashTableOnCpu(const std::vector<JoinColumn>& join_columns,
                                              const std::vector<JoinColumnTypeInfo>& join_column_types) {
  const auto col_tuple_expr = std::dynamic_pointer_cast<Analyzer::ColumnVarTuple>(condition_->get_own_right_operand());
  CHECK(col_tuple_expr);
  const auto inner_outer_pairs =
      normalize_column_pairs(condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  const auto key_component_count = inner_outer_pairs.size();
  std::vector<const void*> sd_inner_proxy_per_key;
  std::vector<const void*> sd_outer_proxy_per_key;
  std::vector<ChunkKey> cache_key_chunks;  // used for the cache key
  for (size_t i = 0; i < key_component_count; ++i) {
    const auto& inner_outer_pair = inner_outer_pairs[i];
    const auto inner_col = inner_outer_pair.first;
    const auto outer_col = inner_outer_pair.second;
    const auto& inner_ti = inner_col->get_type_info();
    const auto& outer_ti = outer_col->get_type_info();
    ChunkKey cache_key_chunks_for_column{
        executor_->getCatalog()->get_currentDB().dbId, inner_col->get_table_id(), inner_col->get_column_id()};
    if (inner_ti.is_string()) {
      CHECK(outer_ti.is_string());
      CHECK(inner_ti.get_compression() == kENCODING_DICT && outer_ti.get_compression() == kENCODING_DICT);
      const auto sd_inner_proxy =
          executor_->getStringDictionaryProxy(inner_ti.get_comp_param(), executor_->row_set_mem_owner_, true);
      const auto sd_outer_proxy =
          executor_->getStringDictionaryProxy(outer_ti.get_comp_param(), executor_->row_set_mem_owner_, true);
      CHECK(sd_inner_proxy && sd_outer_proxy);
      sd_inner_proxy_per_key.push_back(sd_inner_proxy);
      sd_outer_proxy_per_key.push_back(sd_outer_proxy);
      cache_key_chunks_for_column.push_back(sd_outer_proxy->getGeneration());
    } else {
      sd_inner_proxy_per_key.emplace_back();
      sd_outer_proxy_per_key.emplace_back();
    }
    cache_key_chunks.push_back(cache_key_chunks_for_column);
  }
  CHECK(!join_columns.empty());
  HashTableCacheKey cache_key{join_columns.front().num_elems, cache_key_chunks};
  initHashTableOnCpuFromCache(cache_key);
  if (cpu_hash_table_buff_) {
    return 0;
  }
  const auto key_component_width = get_key_component_width(condition_, executor_);
  const auto entry_size = (inner_outer_pairs.size() + 1) * key_component_width;
  cpu_hash_table_buff_.reset(new std::vector<int8_t>(entry_size * entry_count_));
  int thread_count = cpu_threads();
  std::vector<std::future<void>> init_cpu_buff_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    init_cpu_buff_threads.emplace_back(
        std::async(std::launch::async, [this, key_component_count, key_component_width, thread_idx, thread_count] {
          switch (key_component_width) {
            case 4:
              init_baseline_hash_join_buff_32(
                  &(*cpu_hash_table_buff_)[0], entry_count_, key_component_count, -1, thread_idx, thread_count);
              break;
            case 8:
              init_baseline_hash_join_buff_64(
                  &(*cpu_hash_table_buff_)[0], entry_count_, key_component_count, -1, thread_idx, thread_count);
              break;
            default:
              CHECK(false);
          }
        }));
  }
  for (auto& child : init_cpu_buff_threads) {
    child.get();
  }
  std::vector<std::future<int>> fill_cpu_buff_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    fill_cpu_buff_threads.emplace_back(
        std::async(std::launch::async,
                   [this,
                    &join_columns,
                    &join_column_types,
                    &sd_inner_proxy_per_key,
                    &sd_outer_proxy_per_key,
                    key_component_count,
                    key_component_width,
                    thread_idx,
                    thread_count] {
                     switch (key_component_width) {
                       case 4:
                         return fill_baseline_hash_join_buff_32(&(*cpu_hash_table_buff_)[0],
                                                                entry_count_,
                                                                -1,
                                                                key_component_count,
                                                                join_columns,
                                                                join_column_types,
                                                                sd_inner_proxy_per_key,
                                                                sd_outer_proxy_per_key,
                                                                thread_idx,
                                                                thread_count);
                         break;
                       case 8:
                         return fill_baseline_hash_join_buff_64(&(*cpu_hash_table_buff_)[0],
                                                                entry_count_,
                                                                -1,
                                                                key_component_count,
                                                                join_columns,
                                                                join_column_types,
                                                                sd_inner_proxy_per_key,
                                                                sd_outer_proxy_per_key,
                                                                thread_idx,
                                                                thread_count);
                         break;
                       default:
                         CHECK(false);
                     }
                     return -1;
                   }));
  }
  int err = 0;
  for (auto& child : fill_cpu_buff_threads) {
    int partial_err = child.get();
    if (partial_err) {
      err = partial_err;
    }
  }
  if (err) {
    cpu_hash_table_buff_.reset();
  } else {
    putHashTableOnCpuToCache(cache_key);
  }
  return err;
}

int BaselineJoinHashTable::initHashTableForDevice(const std::vector<JoinColumn>& join_columns,
                                                  const std::vector<JoinColumnTypeInfo>& join_column_types,
                                                  const Data_Namespace::MemoryLevel effective_memory_level,
                                                  const int device_id) {
  const auto catalog = executor_->getCatalog();
  const auto col_tuple_expr = std::dynamic_pointer_cast<Analyzer::ColumnVarTuple>(condition_->get_own_right_operand());
  CHECK(col_tuple_expr);
  const auto inner_outer_pairs = normalize_column_pairs(condition_.get(), *catalog, executor_->getTemporaryTables());
  int err = 0;
#ifdef HAVE_CUDA
  const auto key_component_width = get_key_component_width(condition_, executor_);
  const auto key_component_count = inner_outer_pairs.size();
  auto& data_mgr = catalog->get_dataMgr();
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    const auto entry_size = (key_component_count + 1) * key_component_width;
    gpu_hash_table_buff_[device_id] = alloc_gpu_mem(&data_mgr, entry_size * entry_count_, device_id, nullptr);
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
    err = initHashTableOnCpu(join_columns, join_column_types);
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    // Don't transfer the buffer if there was an error since we'll bail anyway.
    if (memory_level_ == Data_Namespace::GPU_LEVEL && !err) {
#ifdef HAVE_CUDA
      copy_to_gpu(&data_mgr,
                  gpu_hash_table_buff_[device_id],
                  &(*cpu_hash_table_buff_)[0],
                  cpu_hash_table_buff_->size() * sizeof((*cpu_hash_table_buff_)[0]),
                  device_id);
#else
      CHECK(false);
#endif
    }
  } else {
#ifdef HAVE_CUDA
    CHECK_EQ(Data_Namespace::GPU_LEVEL, effective_memory_level);
    auto dev_err_buff = alloc_gpu_mem(&data_mgr, sizeof(int), device_id, nullptr);
    copy_to_gpu(&data_mgr, dev_err_buff, &err, sizeof(err), device_id);
    switch (key_component_width) {
      case 4:
        init_baseline_hash_join_buff_on_device_32(reinterpret_cast<int8_t*>(gpu_hash_table_buff_[device_id]),
                                                  entry_count_,
                                                  key_component_count,
                                                  -1,
                                                  executor_->blockSize(),
                                                  executor_->gridSize());
        break;
      case 8:
        init_baseline_hash_join_buff_on_device_64(reinterpret_cast<int8_t*>(gpu_hash_table_buff_[device_id]),
                                                  entry_count_,
                                                  key_component_count,
                                                  -1,
                                                  executor_->blockSize(),
                                                  executor_->gridSize());
        break;
      default:
        CHECK(false);
    }
    auto join_columns_gpu = transfer_pod_vector_to_gpu(join_columns, &data_mgr, device_id);
    auto join_column_types_gpu = transfer_pod_vector_to_gpu(join_column_types, &data_mgr, device_id);
    auto hash_buff = reinterpret_cast<int8_t*>(gpu_hash_table_buff_[device_id]);
    switch (key_component_width) {
      case 4: {
        fill_baseline_hash_join_buff_on_device_32(hash_buff,
                                                  entry_count_,
                                                  -1,
                                                  key_component_count,
                                                  reinterpret_cast<int*>(dev_err_buff),
                                                  join_columns_gpu,
                                                  join_column_types_gpu,
                                                  executor_->blockSize(),
                                                  executor_->gridSize());
        copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
        break;
      }
      case 8: {
        fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                                  entry_count_,
                                                  -1,
                                                  key_component_count,
                                                  reinterpret_cast<int*>(dev_err_buff),
                                                  join_columns_gpu,
                                                  join_column_types_gpu,
                                                  executor_->blockSize(),
                                                  executor_->gridSize());
        copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
        break;
      }
      default:
        CHECK(false);
    }
#else
    CHECK(false);
#endif
  }
  return err;
}

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->ll_int(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

llvm::Value* BaselineJoinHashTable::codegenSlot(const CompilationOptions& co, const size_t index) {
  const auto hash_ptr = hashPtr(index);
  const auto key_component_width = get_key_component_width(condition_, executor_);
  CHECK(key_component_width == 4 || key_component_width == 8);
  const auto inner_outer_pairs =
      normalize_column_pairs(condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  const auto key_size_lv = LL_INT(inner_outer_pairs.size() * key_component_width);
  llvm::Value* key_buff_lv{nullptr};
  switch (key_component_width) {
    case 4:
      key_buff_lv = LL_BUILDER.CreateAlloca(llvm::Type::getInt32Ty(LL_CONTEXT), key_size_lv);
      break;
    case 8:
      key_buff_lv = LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
      break;
    default:
      CHECK(false);
  }
  for (size_t i = 0; i < inner_outer_pairs.size(); ++i) {
    const auto key_comp_dest_lv = LL_BUILDER.CreateGEP(key_buff_lv, LL_INT(i));
    const auto& inner_outer_pair = inner_outer_pairs[i];
    const auto outer_col = inner_outer_pair.second;
    const auto col_lvs = executor_->codegen(outer_col, true, co);
    CHECK_EQ(size_t(1), col_lvs.size());
    const auto col_lv = LL_BUILDER.CreateSExt(col_lvs.front(), get_int_type(key_component_width * 8, LL_CONTEXT));
    LL_BUILDER.CreateStore(col_lv, key_comp_dest_lv);
  }
  const auto key_ptr_lv = LL_BUILDER.CreatePointerCast(key_buff_lv, llvm::Type::getInt8PtrTy(LL_CONTEXT));
  const auto slot_lv =
      executor_->cgen_state_->emitExternalCall("baseline_hash_join_idx_" + std::to_string(key_component_width * 8),
                                               get_int_type(64, LL_CONTEXT),
                                               {hash_ptr, key_ptr_lv, key_size_lv, LL_INT(entry_count_)});
  CHECK(!inner_outer_pairs.empty());
  const auto first_inner_col = inner_outer_pairs.front().first;
  const auto it_ok = executor_->cgen_state_->scan_idx_to_hash_pos_.emplace(first_inner_col->get_rte_idx(), slot_lv);
  CHECK(it_ok.second);
  const auto slot_valid_lv =
      executor_->cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_SGE, slot_lv, executor_->ll_int(int64_t(0)));
  return slot_valid_lv;
}

llvm::Value* BaselineJoinHashTable::hashPtr(const size_t index) {
  const auto total_table_count = executor_->plan_state_->join_info_.join_hash_tables_.size();
  CHECK_LT(index, total_table_count);
  CHECK(executor_->plan_state_->join_info_.join_hash_tables_[index]);
  llvm::Value* hash_ptr = nullptr;
  if (total_table_count > 1) {
    auto hash_tables_ptr = get_arg_by_name(ROW_FUNC, "join_hash_tables");
    auto hash_pptr =
        index > 0 ? LL_BUILDER.CreateGEP(hash_tables_ptr, LL_INT(static_cast<int64_t>(index))) : hash_tables_ptr;
    hash_ptr = LL_BUILDER.CreateLoad(hash_pptr);
    hash_ptr = LL_BUILDER.CreatePointerCast(hash_ptr, llvm::Type::getInt8PtrTy(LL_CONTEXT));
  } else {
    hash_ptr = LL_BUILDER.CreatePointerCast(get_arg_by_name(ROW_FUNC, "join_hash_tables"),
                                            llvm::Type::getInt8PtrTy(LL_CONTEXT));
  }
  return hash_ptr;
}

#undef ROW_FUNC
#undef LL_INT
#undef LL_BUILDER
#undef LL_CONTEXT

int BaselineJoinHashTable::getInnerTableId() const noexcept {
  try {
    return getInnerTableId(condition_.get(), executor_);
  } catch (...) {
    CHECK(false);
  }
  return 0;
}

int BaselineJoinHashTable::getInnerTableId(const Analyzer::BinOper* condition, const Executor* executor) {
  const auto inner_outer_pairs =
      normalize_column_pairs(condition, *executor->getCatalog(), executor->getTemporaryTables());
  CHECK(!inner_outer_pairs.empty());
  const auto first_inner_col = inner_outer_pairs.front().first;
  return first_inner_col->get_table_id();
}

void BaselineJoinHashTable::checkHashJoinReplicationConstraint(const int table_id) const {
  if (!g_cluster) {
    return;
  }
  if (table_id >= 0) {
    const auto inner_td = executor_->getCatalog()->getMetadataForTable(table_id);
    CHECK(inner_td);
    const auto shard_count_info = computeShardCount();
    if (!shard_count_info.count && !table_is_replicated(inner_td)) {
      throw std::runtime_error("Join table " + inner_td->tableName + " must be replicated");
    }
  }
}

void BaselineJoinHashTable::initHashTableOnCpuFromCache(const HashTableCacheKey& key) {
  std::lock_guard<std::mutex> hash_table_cache_lock(hash_table_cache_mutex_);
  for (const auto& kv : hash_table_cache_) {
    if (kv.first == key) {
      cpu_hash_table_buff_ = kv.second;
      break;
    }
  }
}

void BaselineJoinHashTable::putHashTableOnCpuToCache(const HashTableCacheKey& key) {
  std::lock_guard<std::mutex> hash_table_cache_lock(hash_table_cache_mutex_);
  for (const auto& kv : hash_table_cache_) {
    if (kv.first == key) {
      return;
    }
  }
  hash_table_cache_.emplace_back(key, cpu_hash_table_buff_);
}
