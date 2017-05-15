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

#include "JoinHashTable.h"
#include "ThrustAllocator.h"
#include "Execute.h"
#include "ExpressionRewrite.h"
#include "HashJoinRuntime.h"
#include "RuntimeFunctions.h"

#include <glog/logging.h>
#include <numeric>
#include <thread>

namespace {

std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*> get_cols(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const Catalog_Namespace::Catalog& cat,
    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  const auto& lhs_ti = lhs->get_type_info();
  const auto& rhs_ti = rhs->get_type_info();
  if (lhs_ti.get_type() != rhs_ti.get_type()) {
    throw HashJoinFail("Equijoin types must be identical, found: " + lhs_ti.get_type_name() + ", " +
                       rhs_ti.get_type_name());
  }
  if (!lhs_ti.is_integer() && !lhs_ti.is_string()) {
    throw HashJoinFail("Cannot apply hash join to " + lhs_ti.get_type_name());
  }
  const auto lhs_cast = dynamic_cast<const Analyzer::UOper*>(lhs);
  const auto rhs_cast = dynamic_cast<const Analyzer::UOper*>(rhs);
  if (static_cast<bool>(lhs_cast) != static_cast<bool>(rhs_cast) || (lhs_cast && lhs_cast->get_optype() != kCAST) ||
      (rhs_cast && rhs_cast->get_optype() != kCAST)) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  CHECK_EQ(kENCODING_NONE, lhs_ti.get_compression());
  CHECK_EQ(kENCODING_NONE, rhs_ti.get_compression());
  const auto lhs_col = lhs_cast ? dynamic_cast<const Analyzer::ColumnVar*>(lhs_cast->get_operand())
                                : dynamic_cast<const Analyzer::ColumnVar*>(lhs);
  const auto rhs_col = rhs_cast ? dynamic_cast<const Analyzer::ColumnVar*>(rhs_cast->get_operand())
                                : dynamic_cast<const Analyzer::ColumnVar*>(rhs);
  if (!lhs_col || !rhs_col) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  const Analyzer::ColumnVar* inner_col{nullptr};
  const Analyzer::ColumnVar* outer_col{nullptr};
#ifdef ENABLE_EQUIJOIN_FOLD
  if (lhs_col->get_rte_idx() == 0 && rhs_col->get_rte_idx() > 0) {
#else
  if (lhs_col->get_rte_idx() == 0 && rhs_col->get_rte_idx() == 1) {
#endif
    inner_col = rhs_col;
    outer_col = lhs_col;
  } else {
#ifdef ENABLE_EQUIJOIN_FOLD
    CHECK_GT(lhs_col->get_rte_idx(), 0);
#else
    CHECK_EQ(lhs_col->get_rte_idx(), 1);
#endif
    CHECK_EQ(rhs_col->get_rte_idx(), 0);
    inner_col = lhs_col;
    outer_col = rhs_col;
  }
  // We need to fetch the actual type information from the catalog since Analyzer
  // always reports nullable as true for inner table columns in left joins.
  const auto inner_col_cd = get_column_descriptor_maybe(inner_col->get_column_id(), inner_col->get_table_id(), cat);
  const auto& inner_col_real_ti =
      get_column_type(inner_col->get_column_id(), inner_col->get_table_id(), inner_col_cd, temporary_tables);
  const auto& outer_col_ti = outer_col->get_type_info();
  if (outer_col_ti.get_notnull() != inner_col_real_ti.get_notnull()) {
    throw HashJoinFail("For hash join, both sides must have the same nullability");
  }
  if (!(inner_col_real_ti.is_integer() ||
        (inner_col_real_ti.is_string() && inner_col_real_ti.get_compression() == kENCODING_DICT))) {
    throw HashJoinFail("Can only apply hash join to integer-like types and dictionary encoded strings");
  }
  return {inner_col, outer_col};
}

}  // namespace

std::vector<std::pair<JoinHashTable::JoinHashTableCacheKey, std::shared_ptr<std::vector<int32_t>>>>
    JoinHashTable::join_hash_table_cache_;
std::mutex JoinHashTable::join_hash_table_cache_mutex_;

size_t get_shard_count(const Analyzer::BinOper* join_condition, const Catalog_Namespace::Catalog& catalog) {
  return 0;  // Sharded joins aren't enabled yet.
}

std::shared_ptr<JoinHashTable> JoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const Catalog_Namespace::Catalog& cat,
    const std::vector<InputTableInfo>& query_infos,
    const std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    Executor* executor) {
  CHECK_EQ(kEQ, qual_bin_oper->get_optype());
  const auto redirected_bin_oper =
      std::dynamic_pointer_cast<Analyzer::BinOper>(redirect_expr(qual_bin_oper.get(), input_col_descs));
  CHECK(redirected_bin_oper);
  const auto cols = get_cols(redirected_bin_oper, cat, executor->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  auto col_range = getExpressionRange(ti.is_string() ? cols.second : inner_col, query_infos, executor);
  if (col_range.getType() == ExpressionRangeType::Invalid) {
    throw HashJoinFail("Could not compute range for the expressions involved in the equijoin");
  }
  if (ti.is_string()) {
    // The nullable info must be the same as the source column.
    const auto source_col_range = getExpressionRange(inner_col, query_infos, executor);
    if (source_col_range.getType() == ExpressionRangeType::Invalid) {
      throw HashJoinFail("Could not compute range for the expressions involved in the equijoin");
    }
    col_range = ExpressionRange::makeIntRange(std::min(source_col_range.getIntMin(), col_range.getIntMin()),
                                              std::max(source_col_range.getIntMax(), col_range.getIntMax()),
                                              0,
                                              source_col_range.hasNulls());
  }
  auto join_hash_table = std::shared_ptr<JoinHashTable>(
      new JoinHashTable(qual_bin_oper, inner_col, cat, query_infos, memory_level, col_range, executor, device_count));
  const int err = join_hash_table->reify(device_count);
  if (err) {
#ifndef ENABLE_MULTIFRAG_JOIN
    if (err == ERR_MULTI_FRAG) {
      const auto cols = get_cols(qual_bin_oper, cat, executor->temporary_tables_);
      const auto inner_col = cols.first;
      CHECK(inner_col);
      const auto& table_info = join_hash_table->getInnerQueryInfo(inner_col);
      throw HashJoinFail("Multi-fragment inner table '" + get_table_name_by_id(table_info.table_id, cat) +
                         "' not supported yet");
    }
#endif
    if (err == ERR_FAILED_TO_FETCH_COLUMN) {
      throw HashJoinFail("Not enough memory for the columns involved in join");
    }
    if (err == ERR_FAILED_TO_JOIN_ON_VIRTUAL_COLUMN) {
      throw HashJoinFail("Cannot join on rowid");
    }
    throw HashJoinFail("Could not build a 1-to-1 correspondence for columns involved in equijoin");
  }
  return join_hash_table;
}

std::pair<const int8_t*, size_t> JoinHashTable::getColumnFragment(
    const Analyzer::ColumnVar& hash_col,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const Data_Namespace::MemoryLevel effective_mem_lvl,
    const int device_id,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    std::map<int, std::shared_ptr<const ColumnarResults>>& frags_owner) {
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(hash_col.get_column_id());
  CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
  const auto cd = get_column_descriptor_maybe(hash_col.get_column_id(), hash_col.get_table_id(), cat_);
  CHECK(!cd || !(cd->isVirtualCol));
  const int8_t* col_buff = nullptr;
  if (cd) {
    ChunkKey chunk_key{
        cat_.get_currentDB().dbId, hash_col.get_table_id(), hash_col.get_column_id(), fragment.fragmentId};
    const auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                                 &cat_.get_dataMgr(),
                                                 chunk_key,
                                                 effective_mem_lvl,
                                                 effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id,
                                                 chunk_meta_it->second.numBytes,
                                                 chunk_meta_it->second.numElements);
    chunks_owner.push_back(chunk);
    CHECK(chunk);
    auto ab = chunk->get_buffer();
    CHECK(ab->getMemoryPtr());
    col_buff = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
  } else {
    const auto frag_id = fragment.fragmentId;
    auto frag_it = frags_owner.find(frag_id);
    if (frag_it == frags_owner.end()) {
      std::shared_ptr<const ColumnarResults> col_frag(
          columnarize_result(executor_->row_set_mem_owner_,
                             get_temporary_table(executor_->temporary_tables_, hash_col.get_table_id()),
                             frag_id));
      auto res = frags_owner.insert(std::make_pair(frag_id, col_frag));
      CHECK(res.second);
      frag_it = res.first;
    }
    col_buff = Executor::ExecutionDispatch::getColumn(frag_it->second.get(),
                                                      hash_col.get_column_id(),
                                                      &cat_.get_dataMgr(),
                                                      effective_mem_lvl,
                                                      effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id);
  }
  return {col_buff, fragment.getNumTuples()};
}

std::pair<const int8_t*, size_t> JoinHashTable::getAllColumnFragments(
    const Analyzer::ColumnVar& hash_col,
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    std::map<int, std::shared_ptr<const ColumnarResults>>& frags_owner) {
  CHECK(!fragments.empty());
  const size_t elem_width = hash_col.get_type_info().get_size();
  std::vector<const int8_t*> col_frags;
  std::vector<size_t> elem_counts;
  for (auto& frag : fragments) {
    const int8_t* col_frag = nullptr;
    size_t elem_count = 0;
    std::tie(col_frag, elem_count) =
        getColumnFragment(hash_col, frag, Data_Namespace::CPU_LEVEL, 0, chunks_owner, frags_owner);
    CHECK(col_frag != nullptr);
    CHECK_NE(elem_count, size_t(0));
    col_frags.push_back(col_frag);
    elem_counts.push_back(elem_count);
  }
  CHECK(!col_frags.empty());
  CHECK_EQ(col_frags.size(), elem_counts.size());
  const auto total_elem_count = std::accumulate(elem_counts.begin(), elem_counts.end(), size_t(0));
  auto col_buff = reinterpret_cast<int8_t*>(checked_malloc(total_elem_count * elem_width));
  for (size_t i = 0, offset = 0; i < col_frags.size(); ++i) {
    memcpy(col_buff + offset, col_frags[i], elem_counts[i] * elem_width);
    offset += elem_counts[i] * elem_width;
  }
  return {col_buff, total_elem_count};
}

int JoinHashTable::reify(const int device_count) {
  CHECK_LT(0, device_count);
  const auto cols = get_cols(qual_bin_oper_, cat_, executor_->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& query_info = getInnerQueryInfo(inner_col).info;
  if (query_info.fragments.empty()) {
    return 0;
  }
#ifndef ENABLE_MULTIFRAG_JOIN
  if (query_info.fragments.size() != 1) {  // we don't support multiple fragment inner tables (yet)
    return ERR_MULTI_FRAG;
  }
#else
  const bool has_multi_frag = query_info.fragments.size() > 1;
#endif
  const auto cd = get_column_descriptor_maybe(inner_col->get_column_id(), inner_col->get_table_id(), cat_);
  if (cd && cd->isVirtualCol) {
    return ERR_FAILED_TO_JOIN_ON_VIRTUAL_COLUMN;
  }
  CHECK(!cd || !(cd->isVirtualCol));
  const auto& ti =
      get_column_type(inner_col->get_column_id(), inner_col->get_table_id(), cd, executor_->temporary_tables_);
  // Since we don't have the string dictionary payloads on the GPU, we'll build
  // the join hash table on the CPU and transfer it to the GPU.
  const auto effective_memory_level = ti.is_string() ? Data_Namespace::CPU_LEVEL : memory_level_;
#ifdef HAVE_CUDA
  gpu_hash_table_buff_.resize(device_count);
#endif
  std::vector<int> errors(device_count);
  std::vector<std::thread> init_threads;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::map<int, std::shared_ptr<const ColumnarResults>> frags_owner;
  const auto& first_frag = query_info.fragments.front();
  ChunkKey chunk_key{cat_.get_currentDB().dbId, inner_col->get_table_id(), inner_col->get_column_id()};
  const int8_t* col_buff = nullptr;
  size_t elem_count = 0;

#ifdef ENABLE_MULTIFRAG_JOIN
  const size_t elem_width = inner_col->get_type_info().get_size();
  auto& data_mgr = cat_.get_dataMgr();
  RowSetMemoryOwner col_buff_owner;
  std::vector<ThrustAllocator> dev_buff_owner;
  if (has_multi_frag) {
    try {
      std::tie(col_buff, elem_count) =
          getAllColumnFragments(*inner_col, query_info.fragments, chunks_owner, frags_owner);
      col_buff_owner.addColBuffer(col_buff);
    } catch (...) {
      return ERR_FAILED_TO_FETCH_COLUMN;
    }
  } else
#endif
  {
    chunk_key.push_back(first_frag.fragmentId);
  }

  for (int device_id = 0; device_id < device_count; ++device_id) {
#ifdef ENABLE_MULTIFRAG_JOIN
    dev_buff_owner.emplace_back(&data_mgr, device_id);
    if (has_multi_frag) {
      if (effective_memory_level == Data_Namespace::GPU_LEVEL) {
        CHECK(col_buff != nullptr);
        CHECK_NE(elem_count, size_t(0));
        int8_t* dev_col_buff = nullptr;
        dev_col_buff = dev_buff_owner[device_id].allocate(elem_count * elem_width);
        copy_to_gpu(
            &data_mgr, reinterpret_cast<CUdeviceptr>(dev_col_buff), col_buff, elem_count * elem_width, device_id);
        col_buff = dev_col_buff;
      }
    } else
#endif
    {
      try {
        std::tie(col_buff, elem_count) =
            getColumnFragment(*inner_col, first_frag, effective_memory_level, device_id, chunks_owner, frags_owner);
      } catch (...) {
        return ERR_FAILED_TO_FETCH_COLUMN;
      }
    }
    init_threads.emplace_back(
        [&errors, &chunk_key, &cols, elem_count, col_buff, effective_memory_level, device_id, this] {
          try {
            errors[device_id] =
                initHashTableForDevice(chunk_key, col_buff, elem_count, cols, effective_memory_level, device_id);
          } catch (...) {
            errors[device_id] = -1;
          }
        });
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

int JoinHashTable::initHashTableOnCpu(const int8_t* col_buff,
                                      const size_t num_elements,
                                      const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols,
                                      const int32_t hash_entry_count,
                                      const int32_t hash_join_invalid_val) {
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  int err = 0;
  if (!cpu_hash_table_buff_) {
    cpu_hash_table_buff_ = std::make_shared<std::vector<int32_t>>(hash_entry_count);
    const StringDictionaryProxy* sd_inner_proxy{nullptr};
    const StringDictionaryProxy* sd_outer_proxy{nullptr};
    if (ti.is_string()) {
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
      sd_inner_proxy =
          executor_->getStringDictionaryProxy(inner_col->get_comp_param(), executor_->row_set_mem_owner_, true);
      CHECK(sd_inner_proxy);
      sd_outer_proxy =
          executor_->getStringDictionaryProxy(cols.second->get_comp_param(), executor_->row_set_mem_owner_, true);
      CHECK(sd_outer_proxy);
    }
    int thread_count = cpu_threads();
    std::vector<std::thread> init_cpu_buff_threads;
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      init_cpu_buff_threads.emplace_back([this, hash_entry_count, hash_join_invalid_val, thread_idx, thread_count] {
        init_hash_join_buff(
            &(*cpu_hash_table_buff_)[0], hash_entry_count, hash_join_invalid_val, thread_idx, thread_count);
      });
    }
    for (auto& t : init_cpu_buff_threads) {
      t.join();
    }
    init_cpu_buff_threads.clear();
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      init_cpu_buff_threads.emplace_back([this,
                                          hash_join_invalid_val,
                                          col_buff,
                                          num_elements,
                                          sd_inner_proxy,
                                          sd_outer_proxy,
                                          thread_idx,
                                          thread_count,
                                          &ti,
                                          &err] {
        int partial_err = fill_hash_join_buff(&(*cpu_hash_table_buff_)[0],
                                              hash_join_invalid_val,
                                              {col_buff, num_elements},
                                              {static_cast<size_t>(ti.get_size()),
                                               col_range_.getIntMin(),
                                               inline_fixed_encoding_null_val(ti),
                                               col_range_.getIntMax() + 1},
                                              sd_inner_proxy,
                                              sd_outer_proxy,
                                              thread_idx,
                                              thread_count);
        __sync_val_compare_and_swap(&err, 0, partial_err);
      });
    }
    for (auto& t : init_cpu_buff_threads) {
      t.join();
    }
    if (err) {
      cpu_hash_table_buff_.reset();
    }
  }
  return err;
}

namespace {

#ifdef HAVE_CUDA
// Number of entries per shard, rounded up.
size_t get_entries_per_shard(const size_t total_entry_count, const size_t shard_count) {
  CHECK_NE(size_t(0), shard_count);
  return (total_entry_count + shard_count - 1) / shard_count;
}
#endif  // HAVE_CUDA

// Number of entries required for the given range.
size_t get_hash_entry_count(const ExpressionRange& col_range) {
  CHECK_LE(col_range.getIntMin(), col_range.getIntMax());
  return col_range.getIntMax() - col_range.getIntMin() + 1 + (col_range.hasNulls() ? 1 : 0);
}

}  // namespace

int JoinHashTable::initHashTableForDevice(const ChunkKey& chunk_key,
                                          const int8_t* col_buff,
                                          const size_t num_elements,
                                          const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols,
                                          const Data_Namespace::MemoryLevel effective_memory_level,
                                          const int device_id) {
  auto hash_entry_count = get_hash_entry_count(col_range_);
#ifdef HAVE_CUDA
  const auto shard_count = get_shard_count(qual_bin_oper_.get(), cat_);
  const size_t entries_per_shard{shard_count ? get_entries_per_shard(hash_entry_count, shard_count) : 0};
  // Even if we join on dictionary encoded strings, the memory on the GPU is still needed
  // once the join hash table has been built on the CPU.
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    auto& data_mgr = cat_.get_dataMgr();
    if (shard_count) {
      const auto shards_per_device = (shard_count + device_count_ - 1) / device_count_;
      CHECK_GT(shards_per_device, 0);
      hash_entry_count = entries_per_shard * shards_per_device;
    }
    gpu_hash_table_buff_[device_id] = alloc_gpu_mem(&data_mgr, hash_entry_count * sizeof(int32_t), device_id, nullptr);
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  const auto inner_col = cols.first;
  CHECK(inner_col);
#ifdef HAVE_CUDA
  const auto& ti = inner_col->get_type_info();
#endif
  int err = 0;
  const int32_t hash_join_invalid_val{-1};
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    initHashTableOnCpuFromCache(chunk_key, num_elements, cols);
    {
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      err = initHashTableOnCpu(col_buff, num_elements, cols, hash_entry_count, hash_join_invalid_val);
    }
    if (!err && inner_col->get_table_id() > 0) {
      putHashTableOnCpuToCache(chunk_key, num_elements, cols);
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    // Don't transfer the buffer if there was an error since we'll bail anyway.
    if (memory_level_ == Data_Namespace::GPU_LEVEL && !err) {
#ifdef HAVE_CUDA
      CHECK(ti.is_string());
      auto& data_mgr = cat_.get_dataMgr();
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
    auto& data_mgr = cat_.get_dataMgr();
    auto dev_err_buff = alloc_gpu_mem(&data_mgr, sizeof(int), device_id, nullptr);
    copy_to_gpu(&data_mgr, dev_err_buff, &err, sizeof(err), device_id);
    init_hash_join_buff_on_device(reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]),
                                  hash_entry_count,
                                  hash_join_invalid_val,
                                  executor_->blockSize(),
                                  executor_->gridSize());
    JoinColumn join_column{col_buff, num_elements};
    JoinColumnTypeInfo type_info{static_cast<size_t>(ti.get_size()),
                                 col_range_.getIntMin(),
                                 inline_fixed_encoding_null_val(ti),
                                 col_range_.getIntMax() + 1};
    if (shard_count) {
      CHECK_GT(device_count_, 0);
      for (size_t shard = device_id; shard < shard_count; shard += device_count_) {
        ShardInfo shard_info{shard, entries_per_shard, shard_count, device_count_};
        fill_hash_join_buff_on_device_sharded(reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]),
                                              hash_join_invalid_val,
                                              reinterpret_cast<int*>(dev_err_buff),
                                              join_column,
                                              type_info,
                                              shard_info,
                                              executor_->blockSize(),
                                              executor_->gridSize());
      }
    } else {
      fill_hash_join_buff_on_device(reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]),
                                    hash_join_invalid_val,
                                    reinterpret_cast<int*>(dev_err_buff),
                                    join_column,
                                    type_info,
                                    executor_->blockSize(),
                                    executor_->gridSize());
    }
    copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
#else
    CHECK(false);
#endif
  }
  return err;
}

void JoinHashTable::initHashTableOnCpuFromCache(
    const ChunkKey& chunk_key,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols) {
  JoinHashTableCacheKey cache_key{col_range_, *cols.first, *cols.second, num_elements, chunk_key};
  std::lock_guard<std::mutex> join_hash_table_cache_lock(join_hash_table_cache_mutex_);
  for (const auto& kv : join_hash_table_cache_) {
    if (kv.first == cache_key) {
      cpu_hash_table_buff_ = kv.second;
      break;
    }
  }
}

void JoinHashTable::putHashTableOnCpuToCache(
    const ChunkKey& chunk_key,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols) {
  JoinHashTableCacheKey cache_key{col_range_, *cols.first, *cols.second, num_elements, chunk_key};
  std::lock_guard<std::mutex> join_hash_table_cache_lock(join_hash_table_cache_mutex_);
  for (const auto& kv : join_hash_table_cache_) {
    if (kv.first == cache_key) {
      return;
    }
  }
  join_hash_table_cache_.emplace_back(cache_key, cpu_hash_table_buff_);
}

llvm::Value* JoinHashTable::codegenSlot(const CompilationOptions& co, const size_t index) noexcept {
  CHECK(executor_->plan_state_->join_info_.join_impl_type_ == Executor::JoinImplType::HashOneToOne);
  const auto cols = get_cols(qual_bin_oper_, cat_, executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  const auto key_lvs = executor_->codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  const auto total_table_count = executor_->plan_state_->join_info_.join_hash_tables_.size();
  CHECK_LT(index, total_table_count);
  CHECK(executor_->plan_state_->join_info_.join_hash_tables_[index]);
  llvm::Value* hash_ptr = nullptr;
  if (total_table_count > 1) {
    auto hash_tables_ptr = get_arg_by_name(executor_->cgen_state_->row_func_, "join_hash_tables");
    auto hash_pptr = index > 0
                         ? executor_->cgen_state_->ir_builder_.CreateGEP(hash_tables_ptr,
                                                                         executor_->ll_int(static_cast<int64_t>(index)))
                         : hash_tables_ptr;
    hash_ptr = executor_->cgen_state_->ir_builder_.CreateLoad(hash_pptr);
  } else {
    hash_ptr = executor_->cgen_state_->ir_builder_.CreatePtrToInt(
        get_arg_by_name(executor_->cgen_state_->row_func_, "join_hash_tables"),
        llvm::Type::getInt64Ty(executor_->cgen_state_->context_));
  }
  CHECK(hash_ptr);
  std::vector<llvm::Value*> hash_join_idx_args{hash_ptr,
                                               executor_->castToTypeIn(key_lvs.front(), 64),
                                               executor_->ll_int(col_range_.getIntMin()),
                                               executor_->ll_int(col_range_.getIntMax())};
  const int shard_count = get_shard_count(qual_bin_oper_.get(), cat_);
  if (shard_count) {
    const auto hash_entry_count = get_hash_entry_count(col_range_);
    const auto entry_count_per_shard = (hash_entry_count + shard_count - 1) / shard_count;
    hash_join_idx_args.push_back(executor_->ll_int<uint32_t>(entry_count_per_shard));
    hash_join_idx_args.push_back(executor_->ll_int<uint32_t>(shard_count));
    hash_join_idx_args.push_back(executor_->ll_int<uint32_t>(device_count_));
  }
  if (col_range_.hasNulls()) {
    hash_join_idx_args.push_back(executor_->ll_int(inline_fixed_encoding_null_val(key_col->get_type_info())));
  }
  std::string fname{"hash_join_idx"};
  if (shard_count) {
    fname += "_sharded";
  }
  if (col_range_.hasNulls()) {
    fname += "_nullable";
  }
  const auto slot_lv = executor_->cgen_state_->emitCall(fname, hash_join_idx_args);
  const auto it_ok = executor_->cgen_state_->scan_idx_to_hash_pos_.emplace(val_col->get_rte_idx(), slot_lv);
  CHECK(it_ok.second);
  const auto slot_valid_lv =
      executor_->cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_SGE, slot_lv, executor_->ll_int(int64_t(0)));
  return slot_valid_lv;
}

const InputTableInfo& JoinHashTable::getInnerQueryInfo(const Analyzer::ColumnVar* inner_col) {
  ssize_t ti_idx = -1;
  for (size_t i = 0; i < query_infos_.size(); ++i) {
    if (inner_col->get_table_id() == query_infos_[i].table_id) {
      ti_idx = i;
      break;
    }
  }
  CHECK_NE(ssize_t(-1), ti_idx);
  return query_infos_[ti_idx];
}
