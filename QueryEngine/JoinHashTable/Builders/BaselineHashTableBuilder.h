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

#include "DataMgr/Allocators/CudaAllocator.h"
#include "QueryEngine/JoinHashTable/BaselineHashTable.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinKeyHandlers.h"
#include "QueryEngine/JoinHashTable/Runtime/JoinHashTableGpuUtils.h"
#include "QueryEngine/QueryEngine.h"
#include "Shared/thread_count.h"

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 4, SIZE>::type* = nullptr>
int fill_baseline_hash_join_buff(int8_t* hash_buff,
                                 const size_t entry_count,
                                 const int32_t invalid_slot_val,
                                 const bool for_semi_join,
                                 const size_t key_component_count,
                                 const bool with_val_slot,
                                 const KEY_HANDLER* key_handler,
                                 const size_t num_elems,
                                 const int32_t cpu_thread_idx,
                                 const int32_t cpu_thread_count) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    return fill_baseline_hash_join_buff_32(hash_buff,
                                           entry_count,
                                           invalid_slot_val,
                                           for_semi_join,
                                           key_component_count,
                                           with_val_slot,
                                           key_handler,
                                           num_elems,
                                           cpu_thread_idx,
                                           cpu_thread_count);
  } else if constexpr (std::is_same<KEY_HANDLER, RangeKeyHandler>::value) {
    return range_fill_baseline_hash_join_buff_32(hash_buff,
                                                 entry_count,
                                                 invalid_slot_val,
                                                 key_component_count,
                                                 with_val_slot,
                                                 key_handler,
                                                 num_elems,
                                                 cpu_thread_idx,
                                                 cpu_thread_count);
  } else {
    static_assert(
        std::is_same<KEY_HANDLER, BoundingBoxIntersectKeyHandler>::value,
        "Only Generic, Bounding Box Intersect, and Range Key Handlers are supported.");
    return bbox_intersect_fill_baseline_hash_join_buff_32(hash_buff,
                                                          entry_count,
                                                          invalid_slot_val,
                                                          key_component_count,
                                                          with_val_slot,
                                                          key_handler,
                                                          num_elems,
                                                          cpu_thread_idx,
                                                          cpu_thread_count);
  }
}

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 8, SIZE>::type* = nullptr>
int fill_baseline_hash_join_buff(int8_t* hash_buff,
                                 const size_t entry_count,
                                 const int32_t invalid_slot_val,
                                 const bool for_semi_join,
                                 const size_t key_component_count,
                                 const bool with_val_slot,
                                 const KEY_HANDLER* key_handler,
                                 const size_t num_elems,
                                 const int32_t cpu_thread_idx,
                                 const int32_t cpu_thread_count) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    return fill_baseline_hash_join_buff_64(hash_buff,
                                           entry_count,
                                           invalid_slot_val,
                                           for_semi_join,
                                           key_component_count,
                                           with_val_slot,
                                           key_handler,
                                           num_elems,
                                           cpu_thread_idx,
                                           cpu_thread_count);
  } else if constexpr (std::is_same<KEY_HANDLER, RangeKeyHandler>::value) {
    return range_fill_baseline_hash_join_buff_64(hash_buff,
                                                 entry_count,
                                                 invalid_slot_val,
                                                 key_component_count,
                                                 with_val_slot,
                                                 key_handler,
                                                 num_elems,
                                                 cpu_thread_idx,
                                                 cpu_thread_count);
  } else {
    static_assert(
        std::is_same<KEY_HANDLER, BoundingBoxIntersectKeyHandler>::value,
        "Only Generic, Bounding Box Intersection, and Range Key Handlers are supported.");
    return bbox_intersect_fill_baseline_hash_join_buff_64(hash_buff,
                                                          entry_count,
                                                          invalid_slot_val,
                                                          key_component_count,
                                                          with_val_slot,
                                                          key_handler,
                                                          num_elems,
                                                          cpu_thread_idx,
                                                          cpu_thread_count);
  }
}

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 4, SIZE>::type* = nullptr>
void fill_baseline_hash_join_buff_on_device(int8_t* hash_buff,
                                            const size_t entry_count,
                                            const int32_t invalid_slot_val,
                                            const bool for_semi_join,
                                            const size_t key_component_count,
                                            const bool with_val_slot,
                                            int* dev_err_buff,
                                            const KEY_HANDLER* key_handler,
                                            const size_t num_elems,
                                            CUstream cuda_stream) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_baseline_hash_join_buff_on_device_32(hash_buff,
                                              entry_count,
                                              invalid_slot_val,
                                              for_semi_join,
                                              key_component_count,
                                              with_val_slot,
                                              dev_err_buff,
                                              key_handler,
                                              num_elems,
                                              cuda_stream);
  } else if constexpr (std::is_same<KEY_HANDLER, RangeKeyHandler>::value) {
    UNREACHABLE();
  } else {
    static_assert(
        std::is_same<KEY_HANDLER, BoundingBoxIntersectKeyHandler>::value,
        "Only Generic, Bounding Box Intersection, and Range Key Handlers are supported.");
    LOG(FATAL) << "32-bit keys not yet supported for bounding box intersect.";
  }
}

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 8, SIZE>::type* = nullptr>
void fill_baseline_hash_join_buff_on_device(int8_t* hash_buff,
                                            const size_t entry_count,
                                            const int32_t invalid_slot_val,
                                            const bool for_semi_join,
                                            const size_t key_component_count,
                                            const bool with_val_slot,
                                            int* dev_err_buff,
                                            const KEY_HANDLER* key_handler,
                                            const size_t num_elems,
                                            CUstream cuda_stream) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                              entry_count,
                                              invalid_slot_val,
                                              for_semi_join,
                                              key_component_count,
                                              with_val_slot,
                                              dev_err_buff,
                                              key_handler,
                                              num_elems,
                                              cuda_stream);
  } else if constexpr (std::is_same<KEY_HANDLER, RangeKeyHandler>::value) {
    range_fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                                    entry_count,
                                                    invalid_slot_val,
                                                    key_component_count,
                                                    with_val_slot,
                                                    dev_err_buff,
                                                    key_handler,
                                                    num_elems,
                                                    cuda_stream);
  } else {
    static_assert(
        std::is_same<KEY_HANDLER, BoundingBoxIntersectKeyHandler>::value,
        "Only Generic, Bounding Box Intersect, and Range Key Handlers are supported.");
    bbox_intersect_fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                                             entry_count,
                                                             invalid_slot_val,
                                                             key_component_count,
                                                             with_val_slot,
                                                             dev_err_buff,
                                                             key_handler,
                                                             num_elems,
                                                             cuda_stream);
  }
}

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 4, SIZE>::type* = nullptr>
void fill_one_to_many_baseline_hash_table_on_device(int32_t* buff,
                                                    const SIZE* composite_key_dict,
                                                    const size_t hash_entry_count,
                                                    const size_t key_component_count,
                                                    const KEY_HANDLER* key_handler,
                                                    const size_t num_elems,
                                                    const bool for_window_framing,
                                                    CUstream cuda_stream) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_one_to_many_baseline_hash_table_on_device_32(buff,
                                                      composite_key_dict,
                                                      hash_entry_count,
                                                      key_component_count,
                                                      key_handler,
                                                      num_elems,
                                                      for_window_framing,
                                                      cuda_stream);
  } else {
    static_assert(
        std::is_same<KEY_HANDLER, BoundingBoxIntersectKeyHandler>::value ||
            std::is_same<KEY_HANDLER, RangeKeyHandler>::value,
        "Only Generic, Bounding Box Intersection, and Range Key Handlers are supported.");
    LOG(FATAL) << "32-bit keys not yet supported for bounding box intersect.";
  }
}

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 8, SIZE>::type* = nullptr>
void fill_one_to_many_baseline_hash_table_on_device(int32_t* buff,
                                                    const SIZE* composite_key_dict,
                                                    const size_t hash_entry_count,
                                                    const size_t key_component_count,
                                                    const KEY_HANDLER* key_handler,
                                                    const size_t num_elems,
                                                    const bool for_window_framing,
                                                    CUstream cuda_stream) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_one_to_many_baseline_hash_table_on_device_64(buff,
                                                      composite_key_dict,
                                                      hash_entry_count,
                                                      key_handler,
                                                      num_elems,
                                                      for_window_framing,
                                                      cuda_stream);
  } else if constexpr (std::is_same<KEY_HANDLER, RangeKeyHandler>::value) {
    range_fill_one_to_many_baseline_hash_table_on_device_64(
        buff, composite_key_dict, hash_entry_count, key_handler, num_elems, cuda_stream);
  } else {
    static_assert(
        std::is_same<KEY_HANDLER, BoundingBoxIntersectKeyHandler>::value,
        "Only Generic, Bounding Box Intersect, and Range Key Handlers are supported.");
    bbox_intersect_fill_one_to_many_baseline_hash_table_on_device_64(
        buff, composite_key_dict, hash_entry_count, key_handler, num_elems, cuda_stream);
  }
}

class BaselineJoinHashTableBuilder {
 public:
  BaselineJoinHashTableBuilder() = default;
  template <class KEY_HANDLER>
  int initHashTableOnCpu(KEY_HANDLER* key_handler,
                         const CompositeKeyInfo& composite_key_info,
                         const std::vector<JoinColumn>& join_columns,
                         const std::vector<JoinColumnTypeInfo>& join_column_types,
                         const std::vector<JoinBucketInfo>& join_bucket_info,
                         const StrProxyTranslationMapsPtrsAndOffsets&
                             str_proxy_translation_maps_ptrs_and_offsets,
                         const BaselineHashTableEntryInfo hash_table_entry_info,
                         const JoinType join_type,
                         const Executor* executor,
                         const RegisteredQueryHint& query_hint) {
    auto timer = DEBUG_TIMER(__func__);
    auto const hash_table_layout = hash_table_entry_info.getHashTableLayout();
    size_t const hash_table_size = hash_table_entry_info.computeHashTableSize();
    if (query_hint.isHintRegistered(QueryHint::kMaxJoinHashTableSize) &&
        hash_table_size > query_hint.max_join_hash_table_size) {
      throw JoinHashTableTooBig(hash_table_size, query_hint.max_join_hash_table_size);
    }
    const bool for_semi_join =
        (join_type == JoinType::SEMI || join_type == JoinType::ANTI) &&
        hash_table_layout == HashType::OneToOne;
    hash_table_ = std::make_unique<BaselineHashTable>(
        MemoryLevel::CPU_LEVEL, hash_table_entry_info, nullptr, -1);
    setHashLayout(hash_table_layout);
    if (hash_table_entry_info.getNumKeys() == 0) {
      VLOG(1) << "Stop building a hash table: the input table is empty";
      return 0;
    }
    auto cpu_hash_table_ptr = hash_table_->getCpuBuffer();
    int thread_count = cpu_threads();
    std::vector<std::future<void>> init_cpu_buff_threads;
    {
      auto timer_init = DEBUG_TIMER("Initialize CPU Baseline Join Hash Table");
#ifdef HAVE_TBB
      switch (hash_table_entry_info.getJoinKeysSize()) {
        case 4:
          init_baseline_hash_join_buff_tbb_32(cpu_hash_table_ptr,
                                              hash_table_entry_info.getNumHashEntries(),
                                              hash_table_entry_info.getNumJoinKeys(),
                                              hash_table_layout == HashType::OneToOne,
                                              -1);
          break;
        case 8:
          init_baseline_hash_join_buff_tbb_64(cpu_hash_table_ptr,
                                              hash_table_entry_info.getNumHashEntries(),
                                              hash_table_entry_info.getNumJoinKeys(),
                                              hash_table_layout == HashType::OneToOne,
                                              -1);
          break;
        default:
          CHECK(false);
      }
#else   // #ifdef HAVE_TBB
      for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        init_cpu_buff_threads.emplace_back(std::async(
            std::launch::async,
            [keyspace_entry_count,
             key_component_count,
             key_component_width,
             thread_idx,
             thread_count,
             cpu_hash_table_ptr,
             layout,
             parent_thread_local_ids = logger::thread_local_ids()] {
              logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
              DEBUG_TIMER_NEW_THREAD(parent_thread_local_ids.thread_id_);
              switch (key_component_width) {
                case 4:
                  init_baseline_hash_join_buff_32(cpu_hash_table_ptr,
                                                  keyspace_entry_count,
                                                  key_component_count,
                                                  layout == HashType::OneToOne,
                                                  -1,
                                                  thread_idx,
                                                  thread_count);
                  break;
                case 8:
                  init_baseline_hash_join_buff_64(cpu_hash_table_ptr,
                                                  keyspace_entry_count,
                                                  key_component_count,
                                                  layout == HashType::OneToOne,
                                                  -1,
                                                  thread_idx,
                                                  thread_count);
                  break;
                default:
                  UNREACHABLE();
              }
            }));
      }
      for (auto& child : init_cpu_buff_threads) {
        child.get();
      }
#endif  // !HAVE_TBB
    }
    std::vector<std::future<int>> fill_cpu_buff_threads;
    auto timer_fill = DEBUG_TIMER("Fill CPU Baseline Join Hash Table");
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      fill_cpu_buff_threads.emplace_back(std::async(
          std::launch::async,
          [key_handler,
           &join_columns,
           hash_table_entry_info,
           thread_idx,
           cpu_hash_table_ptr,
           thread_count,
           for_semi_join,
           hash_table_layout,
           parent_thread_local_ids = logger::thread_local_ids()] {
            logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
            DEBUG_TIMER_NEW_THREAD(parent_thread_local_ids.thread_id_);
            switch (hash_table_entry_info.getJoinKeysSize()) {
              case 4: {
                return fill_baseline_hash_join_buff<int32_t>(
                    cpu_hash_table_ptr,
                    hash_table_entry_info.getNumHashEntries(),
                    -1,
                    for_semi_join,
                    hash_table_entry_info.getNumJoinKeys(),
                    hash_table_layout == HashType::OneToOne,
                    key_handler,
                    join_columns[0].num_elems,
                    thread_idx,
                    thread_count);
              }
              case 8: {
                return fill_baseline_hash_join_buff<int64_t>(
                    cpu_hash_table_ptr,
                    hash_table_entry_info.getNumHashEntries(),
                    -1,
                    for_semi_join,
                    hash_table_entry_info.getNumJoinKeys(),
                    hash_table_layout == HashType::OneToOne,
                    key_handler,
                    join_columns[0].num_elems,
                    thread_idx,
                    thread_count);
              }
              default:
                UNREACHABLE() << "Unexpected hash join key size: "
                              << hash_table_entry_info.getJoinKeysSize();
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
      return err;
    }
    if (HashJoin::layoutRequiresAdditionalBuffers(hash_table_layout)) {
      auto one_to_many_buff = reinterpret_cast<int32_t*>(
          cpu_hash_table_ptr + hash_table_entry_info.getNumHashEntries() *
                                   hash_table_entry_info.computeKeySize());
      {
        auto timer_init_additional_buffers =
            DEBUG_TIMER("Initialize Additional Buffers for CPU Baseline Join Hash Table");
        init_hash_join_buff(
            one_to_many_buff, hash_table_entry_info.getNumHashEntries(), -1, 0, 1);
      }
      bool is_geo_compressed = false;
      if constexpr (std::is_same_v<KEY_HANDLER, RangeKeyHandler>) {
        if (const auto range_handler =
                reinterpret_cast<const RangeKeyHandler*>(key_handler)) {
          is_geo_compressed = range_handler->is_compressed_;
        }
      }
      auto timer_fill_additional_buffers =
          DEBUG_TIMER("Fill Additional Buffers for CPU Baseline Join Hash Table");
      setHashLayout(hash_table_layout);
      switch (hash_table_entry_info.getJoinKeysSize()) {
        case 4: {
          const auto composite_key_dict = reinterpret_cast<int32_t*>(cpu_hash_table_ptr);
          fill_one_to_many_baseline_hash_table_32(
              one_to_many_buff,
              composite_key_dict,
              hash_table_entry_info.getNumHashEntries(),
              hash_table_entry_info.getNumJoinKeys(),
              join_columns,
              join_column_types,
              join_bucket_info,
              str_proxy_translation_maps_ptrs_and_offsets.first,
              str_proxy_translation_maps_ptrs_and_offsets.second,
              thread_count,
              std::is_same_v<KEY_HANDLER, RangeKeyHandler>,
              is_geo_compressed,
              join_type == JoinType::WINDOW_FUNCTION_FRAMING);
          break;
        }
        case 8: {
          const auto composite_key_dict = reinterpret_cast<int64_t*>(cpu_hash_table_ptr);
          fill_one_to_many_baseline_hash_table_64(
              one_to_many_buff,
              composite_key_dict,
              hash_table_entry_info.getNumHashEntries(),
              hash_table_entry_info.getNumJoinKeys(),
              join_columns,
              join_column_types,
              join_bucket_info,
              str_proxy_translation_maps_ptrs_and_offsets.first,
              str_proxy_translation_maps_ptrs_and_offsets.second,
              thread_count,
              std::is_same_v<KEY_HANDLER, RangeKeyHandler>,
              is_geo_compressed,
              join_type == JoinType::WINDOW_FUNCTION_FRAMING);
          break;
        }
        default:
          CHECK(false);
      }
    }
    return 0;
  }

  void allocateDeviceMemory(const BaselineHashTableEntryInfo hash_table_entry_info,
                            const int device_id,
                            const Executor* executor,
                            const RegisteredQueryHint& query_hint) {
#ifdef HAVE_CUDA
    const size_t hash_table_size = hash_table_entry_info.computeHashTableSize();
    if (query_hint.isHintRegistered(QueryHint::kMaxJoinHashTableSize) &&
        hash_table_size > query_hint.max_join_hash_table_size) {
      throw JoinHashTableTooBig(hash_table_size, query_hint.max_join_hash_table_size);
    }
    if (hash_table_size > executor->maxGpuSlabSize()) {
      throw JoinHashTableTooBig(hash_table_size, executor->maxGpuSlabSize());
    }

    hash_table_ = std::make_unique<BaselineHashTable>(
        MemoryLevel::GPU_LEVEL, hash_table_entry_info, executor->getDataMgr(), device_id);
#else
    UNREACHABLE();
#endif
  }

  template <class KEY_HANDLER>
  int initHashTableOnGpu(KEY_HANDLER* key_handler,
                         const std::vector<JoinColumn>& join_columns,
                         const JoinType join_type,
                         const BaselineHashTableEntryInfo hash_table_entry_info,
                         const int device_id,
                         const Executor* executor,
                         const RegisteredQueryHint& query_hint) {
    auto timer = DEBUG_TIMER(__func__);
    int err = 0;
#ifdef HAVE_CUDA
    allocateDeviceMemory(hash_table_entry_info, device_id, executor, query_hint);
    auto const hash_table_layout = hash_table_entry_info.getHashTableLayout();
    setHashLayout(hash_table_layout);
    if (hash_table_entry_info.getNumKeys() == 0) {
      VLOG(1) << "Stop building a hash table based on a column: an input table is empty";
      return 0;
    }
    auto device_allocator = executor->getCudaAllocator(device_id);
    CHECK(device_allocator);
    auto cuda_stream = executor->getCudaStream(device_id);
    auto dev_err_buff = device_allocator->alloc(sizeof(int));
    device_allocator->copyToDevice(
        dev_err_buff, &err, sizeof(err), "Baseline join hashtable error buffer");
    auto gpu_hash_table_buff = hash_table_->getGpuBuffer();
    CHECK(gpu_hash_table_buff);
    const bool for_semi_join =
        (join_type == JoinType::SEMI || join_type == JoinType::ANTI) &&
        hash_table_layout == HashType::OneToOne;
    const auto key_handler_gpu = transfer_flat_object_to_gpu(
        *key_handler, *device_allocator, "Baseline hash join key handler");
    {
      auto timer_init = DEBUG_TIMER("Initialize GPU Baseline Join Hash Table");
      switch (hash_table_entry_info.getJoinKeysSize()) {
        case 4:
          init_baseline_hash_join_buff_on_device_32(
              gpu_hash_table_buff,
              hash_table_entry_info.getNumHashEntries(),
              hash_table_entry_info.getNumJoinKeys(),
              hash_table_layout == HashType::OneToOne,
              -1,
              cuda_stream);
          break;
        case 8:
          init_baseline_hash_join_buff_on_device_64(
              gpu_hash_table_buff,
              hash_table_entry_info.getNumHashEntries(),
              hash_table_entry_info.getNumJoinKeys(),
              hash_table_layout == HashType::OneToOne,
              -1,
              cuda_stream);
          break;
        default:
          UNREACHABLE();
      }
    }
    auto timer_fill = DEBUG_TIMER("Fill GPU Baseline Join Hash Table");
    switch (hash_table_entry_info.getJoinKeysSize()) {
      case 4: {
        fill_baseline_hash_join_buff_on_device<int32_t>(
            gpu_hash_table_buff,
            hash_table_entry_info.getNumHashEntries(),
            -1,
            for_semi_join,
            hash_table_entry_info.getNumJoinKeys(),
            hash_table_layout == HashType::OneToOne,
            reinterpret_cast<int*>(dev_err_buff),
            key_handler_gpu,
            join_columns.front().num_elems,
            cuda_stream);
        device_allocator->copyFromDevice(
            &err, dev_err_buff, sizeof(err), "Baseline join hashtable error code");
        break;
      }
      case 8: {
        fill_baseline_hash_join_buff_on_device<int64_t>(
            gpu_hash_table_buff,
            hash_table_entry_info.getNumHashEntries(),
            -1,
            for_semi_join,
            hash_table_entry_info.getNumJoinKeys(),
            hash_table_layout == HashType::OneToOne,
            reinterpret_cast<int*>(dev_err_buff),
            key_handler_gpu,
            join_columns.front().num_elems,
            cuda_stream);
        device_allocator->copyFromDevice(
            &err, dev_err_buff, sizeof(err), "Baseline join hashtable error code");
        break;
      }
      default:
        UNREACHABLE();
    }
    if (err) {
      return err;
    }
    if (HashJoin::layoutRequiresAdditionalBuffers(hash_table_layout)) {
      auto one_to_many_buff = reinterpret_cast<int32_t*>(
          gpu_hash_table_buff + hash_table_entry_info.getNumHashEntries() *
                                    hash_table_entry_info.computeKeySize());
      {
        auto timer_init_additional_buf =
            DEBUG_TIMER("Initialize Additional Buffer for GPU Baseline Join Hash Table");
        init_hash_join_buff_on_device(
            one_to_many_buff, hash_table_entry_info.getNumHashEntries(), -1, cuda_stream);
      }
      setHashLayout(hash_table_layout);
      auto timer_fill_additional_buf =
          DEBUG_TIMER("Fill Additional Buffer for GPU Baseline Join Hash Table");
      switch (hash_table_entry_info.getJoinKeysSize()) {
        case 4: {
          const auto composite_key_dict = reinterpret_cast<int32_t*>(gpu_hash_table_buff);
          fill_one_to_many_baseline_hash_table_on_device<int32_t>(
              one_to_many_buff,
              composite_key_dict,
              hash_table_entry_info.getNumHashEntries(),
              hash_table_entry_info.getNumJoinKeys(),
              key_handler_gpu,
              join_columns.front().num_elems,
              join_type == JoinType::WINDOW_FUNCTION_FRAMING,
              cuda_stream);

          break;
        }
        case 8: {
          const auto composite_key_dict = reinterpret_cast<int64_t*>(gpu_hash_table_buff);
          fill_one_to_many_baseline_hash_table_on_device<int64_t>(
              one_to_many_buff,
              composite_key_dict,
              hash_table_entry_info.getNumHashEntries(),
              hash_table_entry_info.getNumJoinKeys(),
              key_handler_gpu,
              join_columns.front().num_elems,
              join_type == JoinType::WINDOW_FUNCTION_FRAMING,
              cuda_stream);

          break;
        }
        default:
          UNREACHABLE();
      }
    }
#else
    UNREACHABLE();
#endif
    return err;
  }

  std::unique_ptr<BaselineHashTable> getHashTable() {
    return std::move(hash_table_);
  }

  void setHashLayout(HashType layout) {
    layout_ = layout;
  }

  HashType getHashLayout() const {
    return layout_;
  }

 private:
  std::unique_ptr<BaselineHashTable> hash_table_;
  HashType layout_;
};
