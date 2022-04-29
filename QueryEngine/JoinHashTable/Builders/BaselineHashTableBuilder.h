/*
 * Copyright 2020 OmniSci, Inc.
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

#include "DataMgr/Allocators/GpuAllocator.h"
#include "QueryEngine/JoinHashTable/BaselineHashTable.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinKeyHandlers.h"
#include "QueryEngine/JoinHashTable/Runtime/JoinHashTableGpuUtils.h"
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
  auto timer = DEBUG_TIMER(__func__);
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
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic, Overlaps, and Range Key Handlers are supported.");
    return overlaps_fill_baseline_hash_join_buff_32(hash_buff,
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
  auto timer = DEBUG_TIMER(__func__);
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
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic, Overlaps, and Range Key Handlers are supported.");
    return overlaps_fill_baseline_hash_join_buff_64(hash_buff,
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
                                            const size_t num_elems) {
  auto timer = DEBUG_TIMER(__func__);
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_baseline_hash_join_buff_on_device_32(hash_buff,
                                              entry_count,
                                              invalid_slot_val,
                                              for_semi_join,
                                              key_component_count,
                                              with_val_slot,
                                              dev_err_buff,
                                              key_handler,
                                              num_elems);
  } else if constexpr (std::is_same<KEY_HANDLER, RangeKeyHandler>::value) {
    UNREACHABLE();
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic, Overlaps, and Range Key Handlers are supported.");
    LOG(FATAL) << "32-bit keys not yet supported for overlaps join.";
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
                                            const size_t num_elems) {
  auto timer = DEBUG_TIMER(__func__);
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                              entry_count,
                                              invalid_slot_val,
                                              for_semi_join,
                                              key_component_count,
                                              with_val_slot,
                                              dev_err_buff,
                                              key_handler,
                                              num_elems);
  } else if constexpr (std::is_same<KEY_HANDLER, RangeKeyHandler>::value) {
    range_fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                                    entry_count,
                                                    invalid_slot_val,
                                                    key_component_count,
                                                    with_val_slot,
                                                    dev_err_buff,
                                                    key_handler,
                                                    num_elems);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic, Overlaps, and Range Key Handlers are supported.");
    overlaps_fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                                       entry_count,
                                                       invalid_slot_val,
                                                       key_component_count,
                                                       with_val_slot,
                                                       dev_err_buff,
                                                       key_handler,
                                                       num_elems);
  }
}

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 4, SIZE>::type* = nullptr>
void fill_one_to_many_baseline_hash_table_on_device(int32_t* buff,
                                                    const SIZE* composite_key_dict,
                                                    const size_t hash_entry_count,
                                                    const int32_t invalid_slot_val,
                                                    const size_t key_component_count,
                                                    const KEY_HANDLER* key_handler,
                                                    const size_t num_elems) {
  auto timer = DEBUG_TIMER(__func__);
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_one_to_many_baseline_hash_table_on_device_32(buff,
                                                      composite_key_dict,
                                                      hash_entry_count,
                                                      invalid_slot_val,
                                                      key_component_count,
                                                      key_handler,
                                                      num_elems);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value ||
                      std::is_same<KEY_HANDLER, RangeKeyHandler>::value,
                  "Only Generic, Overlaps, and Range Key Handlers are supported.");
    LOG(FATAL) << "32-bit keys not yet supported for overlaps join.";
  }
}

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 8, SIZE>::type* = nullptr>
void fill_one_to_many_baseline_hash_table_on_device(int32_t* buff,
                                                    const SIZE* composite_key_dict,
                                                    const size_t hash_entry_count,
                                                    const int32_t invalid_slot_val,
                                                    const size_t key_component_count,
                                                    const KEY_HANDLER* key_handler,
                                                    const size_t num_elems) {
  auto timer = DEBUG_TIMER(__func__);
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_one_to_many_baseline_hash_table_on_device_64(buff,
                                                      composite_key_dict,
                                                      hash_entry_count,
                                                      invalid_slot_val,
                                                      key_handler,
                                                      num_elems);
  } else if constexpr (std::is_same<KEY_HANDLER, RangeKeyHandler>::value) {
    range_fill_one_to_many_baseline_hash_table_on_device_64(buff,
                                                            composite_key_dict,
                                                            hash_entry_count,
                                                            invalid_slot_val,
                                                            key_handler,
                                                            num_elems);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic, Overlaps, and Range Key Handlers are supported.");
    overlaps_fill_one_to_many_baseline_hash_table_on_device_64(buff,
                                                               composite_key_dict,
                                                               hash_entry_count,
                                                               invalid_slot_val,
                                                               key_handler,
                                                               num_elems);
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
                         const size_t keyspace_entry_count,
                         const size_t keys_for_all_rows,
                         const HashType layout,
                         const JoinType join_type,
                         const size_t key_component_width,
                         const size_t key_component_count) {
    auto timer = DEBUG_TIMER(__func__);
    const auto entry_size =
        (key_component_count + (layout == HashType::OneToOne ? 1 : 0)) *
        key_component_width;
    const size_t one_to_many_hash_entries =
        HashJoin::layoutRequiresAdditionalBuffers(layout)
            ? 2 * keyspace_entry_count + keys_for_all_rows
            : 0;
    const size_t hash_table_size =
        entry_size * keyspace_entry_count + one_to_many_hash_entries * sizeof(int32_t);

    // We can't allocate more than 2GB contiguous memory on GPU and each entry is 4 bytes.
    if (hash_table_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      throw TooManyHashEntries(
          "Hash tables for GPU requiring larger than 2GB contigious memory not supported "
          "yet");
    }
    const bool for_semi_join =
        (join_type == JoinType::SEMI || join_type == JoinType::ANTI) &&
        layout == HashType::OneToOne;

    VLOG(1) << "Initializing CPU Join Hash Table with " << keyspace_entry_count
            << " hash entries and " << one_to_many_hash_entries
            << " entries in the one to many buffer";
    VLOG(1) << "Total hash table size: " << hash_table_size << " Bytes";

    hash_table_ = std::make_unique<BaselineHashTable>(
        layout, keyspace_entry_count, keys_for_all_rows, hash_table_size);
    auto cpu_hash_table_ptr = hash_table_->getCpuBuffer();
    int thread_count = cpu_threads();
    std::vector<std::future<void>> init_cpu_buff_threads;
    setHashLayout(layout);
    {
      auto timer_init = DEBUG_TIMER("CPU Baseline-Hash: init_baseline_hash_join_buff_32");
#ifdef HAVE_TBB
      switch (key_component_width) {
        case 4:
          init_baseline_hash_join_buff_tbb_32(cpu_hash_table_ptr,
                                              keyspace_entry_count,
                                              key_component_count,
                                              layout == HashType::OneToOne,
                                              -1);
          break;
        case 8:
          init_baseline_hash_join_buff_tbb_64(cpu_hash_table_ptr,
                                              keyspace_entry_count,
                                              key_component_count,
                                              layout == HashType::OneToOne,
                                              -1);
          break;
        default:
          CHECK(false);
      }
#else   // #ifdef HAVE_TBB
      for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        init_cpu_buff_threads.emplace_back(
            std::async(std::launch::async,
                       [keyspace_entry_count,
                        key_component_count,
                        key_component_width,
                        thread_idx,
                        thread_count,
                        cpu_hash_table_ptr,
                        layout] {
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
                             CHECK(false);
                         }
                       }));
      }
      for (auto& child : init_cpu_buff_threads) {
        child.get();
      }
#endif  // !HAVE_TBB
    }
    std::vector<std::future<int>> fill_cpu_buff_threads;
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      fill_cpu_buff_threads.emplace_back(std::async(
          std::launch::async,
          [key_handler,
           keyspace_entry_count,
           &join_columns,
           key_component_count,
           key_component_width,
           layout,
           thread_idx,
           cpu_hash_table_ptr,
           thread_count,
           for_semi_join] {
            switch (key_component_width) {
              case 4: {
                return fill_baseline_hash_join_buff<int32_t>(cpu_hash_table_ptr,
                                                             keyspace_entry_count,
                                                             -1,
                                                             for_semi_join,
                                                             key_component_count,
                                                             layout == HashType::OneToOne,
                                                             key_handler,
                                                             join_columns[0].num_elems,
                                                             thread_idx,
                                                             thread_count);
                break;
              }
              case 8: {
                return fill_baseline_hash_join_buff<int64_t>(cpu_hash_table_ptr,
                                                             keyspace_entry_count,
                                                             -1,
                                                             for_semi_join,
                                                             key_component_count,
                                                             layout == HashType::OneToOne,
                                                             key_handler,
                                                             join_columns[0].num_elems,
                                                             thread_idx,
                                                             thread_count);
                break;
              }
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
      return err;
    }
    if (HashJoin::layoutRequiresAdditionalBuffers(layout)) {
      auto one_to_many_buff = reinterpret_cast<int32_t*>(
          cpu_hash_table_ptr + keyspace_entry_count * entry_size);
      {
        auto timer_init_additional_buffers =
            DEBUG_TIMER("CPU Baseline-Hash: Additional Buffers init_hash_join_buff");
        init_hash_join_buff(one_to_many_buff, keyspace_entry_count, -1, 0, 1);
      }
      setHashLayout(layout);
      switch (key_component_width) {
        case 4: {
          const auto composite_key_dict = reinterpret_cast<int32_t*>(cpu_hash_table_ptr);
          fill_one_to_many_baseline_hash_table_32(
              one_to_many_buff,
              composite_key_dict,
              keyspace_entry_count,
              -1,
              key_component_count,
              join_columns,
              join_column_types,
              join_bucket_info,
              str_proxy_translation_maps_ptrs_and_offsets.first,
              str_proxy_translation_maps_ptrs_and_offsets.second,
              thread_count,
              std::is_same_v<KEY_HANDLER, RangeKeyHandler>);
          break;
        }
        case 8: {
          const auto composite_key_dict = reinterpret_cast<int64_t*>(cpu_hash_table_ptr);
          fill_one_to_many_baseline_hash_table_64(
              one_to_many_buff,
              composite_key_dict,
              keyspace_entry_count,
              -1,
              key_component_count,
              join_columns,
              join_column_types,
              join_bucket_info,
              str_proxy_translation_maps_ptrs_and_offsets.first,
              str_proxy_translation_maps_ptrs_and_offsets.second,
              thread_count,
              std::is_same_v<KEY_HANDLER, RangeKeyHandler>);
          break;
        }
        default:
          CHECK(false);
      }
    }
    return err;
  }

  void allocateDeviceMemory(const HashType layout,
                            const size_t key_component_width,
                            const size_t key_component_count,
                            const size_t keyspace_entry_count,
                            const size_t emitted_keys_count,
                            const int device_id,
                            const Executor* executor) {
#ifdef HAVE_CUDA
    const auto entry_size =
        (key_component_count + (layout == HashType::OneToOne ? 1 : 0)) *
        key_component_width;
    const size_t one_to_many_hash_entries =
        HashJoin::layoutRequiresAdditionalBuffers(layout)
            ? 2 * keyspace_entry_count + emitted_keys_count
            : 0;
    const size_t hash_table_size =
        entry_size * keyspace_entry_count + one_to_many_hash_entries * sizeof(int32_t);

    // We can't allocate more than 2GB contiguous memory on GPU and each entry is 4 bytes.
    if (hash_table_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      throw TooManyHashEntries(
          "Hash tables for GPU requiring larger than 2GB contigious memory not supported "
          "yet");
    }

    VLOG(1) << "Initializing GPU Hash Table for device " << device_id << " with "
            << keyspace_entry_count << " hash entries and " << one_to_many_hash_entries
            << " entries in the " << HashJoin::getHashTypeString(layout) << " buffer";
    VLOG(1) << "Total hash table size: " << hash_table_size << " Bytes";

    hash_table_ = std::make_unique<BaselineHashTable>(executor->getBufferProvider(),
                                                      layout,
                                                      keyspace_entry_count,
                                                      emitted_keys_count,
                                                      hash_table_size,
                                                      device_id);
#else
    UNREACHABLE();
#endif
  }

  template <class KEY_HANDLER>
  int initHashTableOnGpu(KEY_HANDLER* key_handler,
                         const std::vector<JoinColumn>& join_columns,
                         const HashType layout,
                         const JoinType join_type,
                         const size_t key_component_width,
                         const size_t key_component_count,
                         const size_t keyspace_entry_count,
                         const size_t emitted_keys_count,
                         const int device_id,
                         const Executor* executor) {
    auto timer = DEBUG_TIMER(__func__);
    int err = 0;
#ifdef HAVE_CUDA
    allocateDeviceMemory(layout,
                         key_component_width,
                         key_component_count,
                         keyspace_entry_count,
                         emitted_keys_count,
                         device_id,
                         executor);
    if (!keyspace_entry_count) {
      // need to "allocate" the empty hash table first
      CHECK(!emitted_keys_count);
      return 0;
    }
    auto buffer_provider = executor->getBufferProvider();
    GpuAllocator allocator(buffer_provider, device_id);
    auto dev_err_buff = reinterpret_cast<CUdeviceptr>(allocator.alloc(sizeof(int)));
    buffer_provider->copyToDevice(reinterpret_cast<int8_t*>(dev_err_buff),
                                  reinterpret_cast<const int8_t*>(&err),
                                  sizeof(err),
                                  device_id);
    auto gpu_hash_table_buff = hash_table_->getGpuBuffer();
    CHECK(gpu_hash_table_buff);
    const bool for_semi_join =
        (join_type == JoinType::SEMI || join_type == JoinType::ANTI) &&
        layout == HashType::OneToOne;
    setHashLayout(layout);
    const auto key_handler_gpu = transfer_flat_object_to_gpu(*key_handler, allocator);
    switch (key_component_width) {
      case 4:
        init_baseline_hash_join_buff_on_device_32(gpu_hash_table_buff,
                                                  keyspace_entry_count,
                                                  key_component_count,
                                                  layout == HashType::OneToOne,
                                                  -1);
        break;
      case 8:
        init_baseline_hash_join_buff_on_device_64(gpu_hash_table_buff,
                                                  keyspace_entry_count,
                                                  key_component_count,
                                                  layout == HashType::OneToOne,
                                                  -1);
        break;
      default:
        UNREACHABLE();
    }
    switch (key_component_width) {
      case 4: {
        fill_baseline_hash_join_buff_on_device<int32_t>(
            gpu_hash_table_buff,
            keyspace_entry_count,
            -1,
            for_semi_join,
            key_component_count,
            layout == HashType::OneToOne,
            reinterpret_cast<int*>(dev_err_buff),
            key_handler_gpu,
            join_columns.front().num_elems);
        buffer_provider->copyFromDevice(reinterpret_cast<int8_t*>(&err),
                                        reinterpret_cast<const int8_t*>(dev_err_buff),
                                        sizeof(err),
                                        device_id);
        break;
      }
      case 8: {
        fill_baseline_hash_join_buff_on_device<int64_t>(
            gpu_hash_table_buff,
            keyspace_entry_count,
            -1,
            for_semi_join,
            key_component_count,
            layout == HashType::OneToOne,
            reinterpret_cast<int*>(dev_err_buff),
            key_handler_gpu,
            join_columns.front().num_elems);
        buffer_provider->copyFromDevice(reinterpret_cast<int8_t*>(&err),
                                        reinterpret_cast<const int8_t*>(dev_err_buff),
                                        sizeof(err),
                                        device_id);
        break;
      }
      default:
        UNREACHABLE();
    }
    if (err) {
      return err;
    }
    if (HashJoin::layoutRequiresAdditionalBuffers(layout)) {
      const auto entry_size = key_component_count * key_component_width;
      auto one_to_many_buff = reinterpret_cast<int32_t*>(
          gpu_hash_table_buff + keyspace_entry_count * entry_size);
      init_hash_join_buff_on_device(one_to_many_buff, keyspace_entry_count, -1);
      setHashLayout(layout);
      switch (key_component_width) {
        case 4: {
          const auto composite_key_dict = reinterpret_cast<int32_t*>(gpu_hash_table_buff);
          fill_one_to_many_baseline_hash_table_on_device<int32_t>(
              one_to_many_buff,
              composite_key_dict,
              keyspace_entry_count,
              -1,
              key_component_count,
              key_handler_gpu,
              join_columns.front().num_elems);

          break;
        }
        case 8: {
          const auto composite_key_dict = reinterpret_cast<int64_t*>(gpu_hash_table_buff);
          fill_one_to_many_baseline_hash_table_on_device<int64_t>(
              one_to_many_buff,
              composite_key_dict,
              keyspace_entry_count,
              -1,
              key_component_count,
              key_handler_gpu,
              join_columns.front().num_elems);

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

  std::unique_ptr<BaselineHashTable> getHashTable() { return std::move(hash_table_); }

  void setHashLayout(HashType layout) { layout_ = layout; }

  HashType getHashLayout() const { return layout_; }

 private:
  std::unique_ptr<BaselineHashTable> hash_table_;
  HashType layout_;
};
