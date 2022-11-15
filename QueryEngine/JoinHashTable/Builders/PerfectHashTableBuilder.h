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

#include "QueryEngine/JoinHashTable/PerfectHashTable.h"
#include "QueryEngine/QueryEngine.h"

#include "Shared/scope.h"

class PerfectJoinHashTableBuilder {
 public:
  PerfectJoinHashTableBuilder() {}

  void allocateDeviceMemory(const size_t num_column_elems,
                            const HashType layout,
                            BucketizedHashEntryInfo hash_entry_info,
                            const size_t shard_count,
                            const int device_id,
                            const int device_count,
                            const Executor* executor) {
#ifdef HAVE_CUDA
    if (shard_count) {
      const auto shards_per_device = (shard_count + device_count - 1) / device_count;
      CHECK_GT(shards_per_device, 0u);
      const size_t entries_per_shard =
          get_entries_per_shard(hash_entry_info.bucketized_hash_entry_count, shard_count);
      hash_entry_info.bucketized_hash_entry_count = entries_per_shard * shards_per_device;
    }
    const size_t total_count =
        layout == HashType::OneToOne
            ? hash_entry_info.getNormalizedHashEntryCount()
            : 2 * hash_entry_info.getNormalizedHashEntryCount() + num_column_elems;
    CHECK(!hash_table_);
    hash_table_ =
        std::make_unique<PerfectHashTable>(executor->getDataMgr(),
                                           layout,
                                           ExecutorDeviceType::GPU,
                                           hash_entry_info.getNormalizedHashEntryCount(),
                                           num_column_elems);
    hash_table_->allocateGpuMemory(total_count, device_id);
#else
    UNREACHABLE();
#endif  // HAVE_CUDA
  }

  void allocateDeviceMemory(const JoinColumn& join_column,
                            const HashType layout,
                            BucketizedHashEntryInfo& hash_entry_info,
                            const size_t shard_count,
                            const int device_id,
                            const int device_count,
                            const Executor* executor) {
    allocateDeviceMemory(join_column.num_elems,
                         layout,
                         hash_entry_info,
                         shard_count,
                         device_id,
                         device_count,
                         executor);
  }

#ifdef HAVE_CUDA
  void initHashTableOnGpu(const ChunkKey& chunk_key,
                          const JoinColumn& join_column,
                          const ExpressionRange& col_range,
                          const bool is_bitwise_eq,
                          const InnerOuter& cols,
                          const JoinType join_type,
                          const HashType layout,
                          const BucketizedHashEntryInfo hash_entry_info,
                          const size_t shard_count,
                          const int32_t hash_join_invalid_val,
                          const int device_id,
                          const int device_count,
                          const Executor* executor) {
    auto timer = DEBUG_TIMER(__func__);
    auto data_mgr = executor->getDataMgr();
    Data_Namespace::AbstractBuffer* gpu_hash_table_err_buff =
        CudaAllocator::allocGpuAbstractBuffer(data_mgr, sizeof(int), device_id);
    ScopeGuard cleanup_error_buff = [&data_mgr, gpu_hash_table_err_buff]() {
      data_mgr->free(gpu_hash_table_err_buff);
    };
    CHECK(gpu_hash_table_err_buff);
    auto dev_err_buff = gpu_hash_table_err_buff->getMemoryPtr();
    int err{0};

    auto allocator = std::make_unique<CudaAllocator>(
        data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id));
    allocator->copyToDevice(dev_err_buff, &err, sizeof(err));

    CHECK(hash_table_);
    auto gpu_hash_table_buff = hash_table_->getGpuBuffer();

    init_hash_join_buff_on_device(reinterpret_cast<int32_t*>(gpu_hash_table_buff),
                                  hash_entry_info.getNormalizedHashEntryCount(),
                                  hash_join_invalid_val);
    if (chunk_key.empty()) {
      return;
    }

    // TODO: pass this in? duplicated in JoinHashTable currently
    const auto inner_col = cols.first;
    CHECK(inner_col);
    const auto& ti = inner_col->get_type_info();
    auto translated_null_val = col_range.getIntMax() + 1;
    if (col_range.getIntMax() < col_range.getIntMin()) {
      translated_null_val = col_range.getIntMin() - 1;
    }
    JoinColumnTypeInfo type_info{static_cast<size_t>(ti.get_size()),
                                 col_range.getIntMin(),
                                 col_range.getIntMax(),
                                 inline_fixed_encoding_null_val(ti),
                                 is_bitwise_eq,
                                 translated_null_val,
                                 get_join_column_type_kind(ti)};
    auto use_bucketization = inner_col->get_type_info().get_type() == kDATE;
    if (shard_count) {
      const size_t entries_per_shard =
          get_entries_per_shard(hash_entry_info.bucketized_hash_entry_count, shard_count);
      CHECK_GT(device_count, 0);
      for (size_t shard = device_id; shard < shard_count; shard += device_count) {
        ShardInfo shard_info{shard, entries_per_shard, shard_count, device_count};
        if (layout == HashType::OneToOne) {
          fill_hash_join_buff_on_device_sharded_bucketized(
              reinterpret_cast<int32_t*>(gpu_hash_table_buff),
              hash_join_invalid_val,
              for_semi_anti_join(join_type),
              reinterpret_cast<int*>(dev_err_buff),
              join_column,
              type_info,
              shard_info,
              hash_entry_info.bucket_normalization);
        } else {
          fill_one_to_many_hash_table_on_device_sharded(
              reinterpret_cast<int32_t*>(gpu_hash_table_buff),
              hash_entry_info,
              join_column,
              type_info,
              shard_info);
        }
      }
    } else {
      if (layout == HashType::OneToOne) {
        fill_hash_join_buff_on_device_bucketized(
            reinterpret_cast<int32_t*>(gpu_hash_table_buff),
            hash_join_invalid_val,
            for_semi_anti_join(join_type),
            reinterpret_cast<int*>(dev_err_buff),
            join_column,
            type_info,
            hash_entry_info.bucket_normalization);
      } else {
        if (use_bucketization) {
          fill_one_to_many_hash_table_on_device_bucketized(
              reinterpret_cast<int32_t*>(gpu_hash_table_buff),
              hash_entry_info,
              join_column,
              type_info);
        } else {
          fill_one_to_many_hash_table_on_device(
              reinterpret_cast<int32_t*>(gpu_hash_table_buff),
              hash_entry_info,
              join_column,
              type_info,
              join_type == JoinType::WINDOW_FUNCTION_FRAMING);
        }
      }
    }
    allocator->copyFromDevice(&err, dev_err_buff, sizeof(err));
    if (err) {
      if (layout == HashType::OneToOne) {
        throw NeedsOneToManyHash();
      } else {
        throw std::runtime_error("Unexpected error when building perfect hash table: " +
                                 std::to_string(err));
      }
    }
  }
#endif

  void initOneToOneHashTableOnCpu(
      const JoinColumn& join_column,
      const ExpressionRange& col_range,
      const bool is_bitwise_eq,
      const InnerOuter& cols,
      const StringDictionaryProxy::IdMap* str_proxy_translation_map,
      const JoinType join_type,
      const HashType hash_type,
      const BucketizedHashEntryInfo hash_entry_info,
      const int32_t hash_join_invalid_val,
      const Executor* executor) {
    auto timer = DEBUG_TIMER(__func__);
    const auto inner_col = cols.first;
    CHECK(inner_col);
    const auto& ti = inner_col->get_type_info();

    CHECK(!hash_table_);
    hash_table_ =
        std::make_unique<PerfectHashTable>(executor->getDataMgr(),
                                           hash_type,
                                           ExecutorDeviceType::CPU,
                                           hash_entry_info.getNormalizedHashEntryCount(),
                                           0);

    auto cpu_hash_table_buff = reinterpret_cast<int32_t*>(hash_table_->getCpuBuffer());
    const int thread_count = cpu_threads();
    std::vector<std::thread> init_cpu_buff_threads;

    {
      auto timer_init = DEBUG_TIMER("CPU One-To-One Perfect-Hash: init_hash_join_buff");
#ifdef HAVE_TBB
      init_hash_join_buff_tbb(cpu_hash_table_buff,
                              hash_entry_info.getNormalizedHashEntryCount(),
                              hash_join_invalid_val);
#else   // #ifdef HAVE_TBB
      for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        init_cpu_buff_threads.emplace_back([hash_entry_info,
                                            hash_join_invalid_val,
                                            thread_idx,
                                            thread_count,
                                            cpu_hash_table_buff] {
          init_hash_join_buff(cpu_hash_table_buff,
                              hash_entry_info.getNormalizedHashEntryCount(),
                              hash_join_invalid_val,
                              thread_idx,
                              thread_count);
        });
      }
      for (auto& t : init_cpu_buff_threads) {
        t.join();
      }
      init_cpu_buff_threads.clear();
#endif  // !HAVE_TBB
    }
    const bool for_semi_join = for_semi_anti_join(join_type);
    std::atomic<int> err{0};
    {
      auto timer_fill =
          DEBUG_TIMER("CPU One-To-One Perfect-Hash: fill_hash_join_buff_bucketized");
      for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        init_cpu_buff_threads.emplace_back([hash_join_invalid_val,
                                            &join_column,
                                            str_proxy_translation_map,
                                            thread_idx,
                                            thread_count,
                                            &ti,
                                            &err,
                                            &col_range,
                                            &is_bitwise_eq,
                                            &for_semi_join,
                                            cpu_hash_table_buff,
                                            hash_entry_info] {
          auto translated_null_val = col_range.getIntMax() + 1;
          if (col_range.getIntMax() < col_range.getIntMin()) {
            translated_null_val = col_range.getIntMin() - 1;
          }
          int partial_err = fill_hash_join_buff_bucketized(
              cpu_hash_table_buff,
              hash_join_invalid_val,
              for_semi_join,
              join_column,
              {static_cast<size_t>(ti.get_size()),
               col_range.getIntMin(),
               col_range.getIntMax(),
               inline_fixed_encoding_null_val(ti),
               is_bitwise_eq,
               translated_null_val,
               get_join_column_type_kind(ti)},
              str_proxy_translation_map ? str_proxy_translation_map->data() : nullptr,
              str_proxy_translation_map ? str_proxy_translation_map->domainStart()
                                        : 0,  // 0 is dummy value
              thread_idx,
              thread_count,
              hash_entry_info.bucket_normalization);
          int zero{0};
          err.compare_exchange_strong(zero, partial_err);
        });
      }
      for (auto& t : init_cpu_buff_threads) {
        t.join();
      }
    }
    if (err) {
      // Too many hash entries, need to retry with a 1:many table
      hash_table_ = nullptr;  // clear the hash table buffer
      throw NeedsOneToManyHash();
    }
  }

  void initOneToManyHashTableOnCpu(
      const JoinColumn& join_column,
      const ExpressionRange& col_range,
      const bool is_bitwise_eq,
      const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
      const StringDictionaryProxy::IdMap* str_proxy_translation_map,
      const JoinType join_type,
      const BucketizedHashEntryInfo hash_entry_info,
      const int32_t hash_join_invalid_val,
      const Executor* executor) {
    auto timer = DEBUG_TIMER(__func__);
    const auto inner_col = cols.first;
    CHECK(inner_col);
    const auto& ti = inner_col->get_type_info();
    CHECK(!hash_table_);
    hash_table_ = std::make_unique<PerfectHashTable>(
        executor->getDataMgr(),
        HashType::OneToMany,
        ExecutorDeviceType::CPU,
        hash_entry_info.getNormalizedHashEntryCount(),
        join_column.num_elems,
        join_type == JoinType::WINDOW_FUNCTION_FRAMING);
    auto cpu_hash_table_buff = reinterpret_cast<int32_t*>(hash_table_->getCpuBuffer());

    int thread_count = cpu_threads();
    {
      auto timer_init =
          DEBUG_TIMER("CPU One-To-Many Perfect Hash Table Builder: init_hash_join_buff");
#ifdef HAVE_TBB
      init_hash_join_buff_tbb(cpu_hash_table_buff,
                              hash_entry_info.getNormalizedHashEntryCount(),
                              hash_join_invalid_val);
#else   // #ifdef HAVE_TBB
      std::vector<std::future<void> > init_threads;
      for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        init_threads.emplace_back(
            std::async(std::launch::async,
                       init_hash_join_buff,
                       cpu_hash_table_buff,
                       hash_entry_info.getNormalizedHashEntryCount(),
                       hash_join_invalid_val,
                       thread_idx,
                       thread_count));
      }
      for (auto& child : init_threads) {
        child.wait();
      }
      for (auto& child : init_threads) {
        child.get();
      }
#endif  // !HAVE_TBB
    }
    {
      auto timer_fill = DEBUG_TIMER(
          "CPU One-To-Many Perfect Hash Table Builder: fill_hash_join_buff_bucketized");
      auto translated_null_val = col_range.getIntMax() + 1;
      if (col_range.getIntMax() < col_range.getIntMin()) {
        translated_null_val = col_range.getIntMin() - 1;
      }
      if (ti.get_type() == kDATE) {
        fill_one_to_many_hash_table_bucketized(
            cpu_hash_table_buff,
            hash_entry_info,
            join_column,
            {static_cast<size_t>(ti.get_size()),
             col_range.getIntMin(),
             col_range.getIntMax(),
             inline_fixed_encoding_null_val(ti),
             is_bitwise_eq,
             translated_null_val,
             get_join_column_type_kind(ti)},
            str_proxy_translation_map ? str_proxy_translation_map->data() : nullptr,
            str_proxy_translation_map ? str_proxy_translation_map->domainStart()
                                      : 0 /*dummy*/,
            thread_count);
      } else {
        fill_one_to_many_hash_table(
            cpu_hash_table_buff,
            hash_entry_info,
            join_column,
            {static_cast<size_t>(ti.get_size()),
             col_range.getIntMin(),
             col_range.getIntMax(),
             inline_fixed_encoding_null_val(ti),
             is_bitwise_eq,
             translated_null_val,
             get_join_column_type_kind(ti)},
            str_proxy_translation_map ? str_proxy_translation_map->data() : nullptr,
            str_proxy_translation_map ? str_proxy_translation_map->domainStart()
                                      : 0 /*dummy*/,
            thread_count,
            join_type == JoinType::WINDOW_FUNCTION_FRAMING);
      }
    }
  }

  std::unique_ptr<PerfectHashTable> getHashTable() { return std::move(hash_table_); }

  static size_t get_entries_per_shard(const size_t total_entry_count,
                                      const size_t shard_count) {
    CHECK_NE(size_t(0), shard_count);
    return (total_entry_count + shard_count - 1) / shard_count;
  }

  const bool for_semi_anti_join(const JoinType join_type) {
    return join_type == JoinType::SEMI || join_type == JoinType::ANTI;
  }

 private:
  std::unique_ptr<PerfectHashTable> hash_table_;
};
