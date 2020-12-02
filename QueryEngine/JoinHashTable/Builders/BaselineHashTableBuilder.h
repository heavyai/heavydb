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

#include "DataMgr/Allocators/CudaAllocator.h"
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
                                           key_component_count,
                                           with_val_slot,
                                           key_handler,
                                           num_elems,
                                           cpu_thread_idx,
                                           cpu_thread_count);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic or Overlaps Key Handlers are supported.");
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
                                           key_component_count,
                                           with_val_slot,
                                           key_handler,
                                           num_elems,
                                           cpu_thread_idx,
                                           cpu_thread_count);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic or Overlaps Key Handlers are supported.");
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
                                            const size_t key_component_count,
                                            const bool with_val_slot,
                                            int* dev_err_buff,
                                            const KEY_HANDLER* key_handler,
                                            const size_t num_elems,
                                            const size_t block_size_x,
                                            const size_t grid_size_x) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_baseline_hash_join_buff_on_device_32(hash_buff,
                                              entry_count,
                                              invalid_slot_val,
                                              key_component_count,
                                              with_val_slot,
                                              dev_err_buff,
                                              key_handler,
                                              num_elems,
                                              block_size_x,
                                              grid_size_x);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic or Overlaps Key Handlers are supported.");
    LOG(FATAL) << "32-bit keys not yet supported for overlaps join.";
  }
}

template <typename SIZE,
          class KEY_HANDLER,
          typename std::enable_if<sizeof(SIZE) == 8, SIZE>::type* = nullptr>
void fill_baseline_hash_join_buff_on_device(int8_t* hash_buff,
                                            const size_t entry_count,
                                            const int32_t invalid_slot_val,
                                            const size_t key_component_count,
                                            const bool with_val_slot,
                                            int* dev_err_buff,
                                            const KEY_HANDLER* key_handler,
                                            const size_t num_elems,
                                            const size_t block_size_x,
                                            const size_t grid_size_x) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                              entry_count,
                                              invalid_slot_val,
                                              key_component_count,
                                              with_val_slot,
                                              dev_err_buff,
                                              key_handler,
                                              num_elems,
                                              block_size_x,
                                              grid_size_x);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic or Overlaps Key Handlers are supported.");
    overlaps_fill_baseline_hash_join_buff_on_device_64(hash_buff,
                                                       entry_count,
                                                       invalid_slot_val,
                                                       key_component_count,
                                                       with_val_slot,
                                                       dev_err_buff,
                                                       key_handler,
                                                       num_elems,
                                                       block_size_x,
                                                       grid_size_x);
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
                                                    const size_t num_elems,
                                                    const size_t block_size_x,
                                                    const size_t grid_size_x) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_one_to_many_baseline_hash_table_on_device_32(buff,
                                                      composite_key_dict,
                                                      hash_entry_count,
                                                      invalid_slot_val,
                                                      key_component_count,
                                                      key_handler,
                                                      num_elems,
                                                      block_size_x,
                                                      grid_size_x);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic or Overlaps Key Handlers are supported.");
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
                                                    const size_t num_elems,
                                                    const size_t block_size_x,
                                                    const size_t grid_size_x) {
  if constexpr (std::is_same<KEY_HANDLER, GenericKeyHandler>::value) {
    fill_one_to_many_baseline_hash_table_on_device_64(buff,
                                                      composite_key_dict,
                                                      hash_entry_count,
                                                      invalid_slot_val,
                                                      key_handler,
                                                      num_elems,
                                                      block_size_x,
                                                      grid_size_x);
  } else {
    static_assert(std::is_same<KEY_HANDLER, OverlapsKeyHandler>::value,
                  "Only Generic or Overlaps Key Handlers are supported.");
    overlaps_fill_one_to_many_baseline_hash_table_on_device_64(buff,
                                                               composite_key_dict,
                                                               hash_entry_count,
                                                               invalid_slot_val,
                                                               key_handler,
                                                               num_elems,
                                                               block_size_x,
                                                               grid_size_x);
  }
}

class BaselineJoinHashTableBuilder {
 public:
  BaselineJoinHashTableBuilder(const Catalog_Namespace::Catalog* catalog)
      : catalog_(catalog) {}

  template <class KEY_HANDLER>
  int initHashTableOnCpu(KEY_HANDLER* key_handler,
                         const CompositeKeyInfo& composite_key_info,
                         const std::vector<JoinColumn>& join_columns,
                         const std::vector<JoinColumnTypeInfo>& join_column_types,
                         const std::vector<JoinBucketInfo>& join_bucket_info,
                         const size_t keyspace_entry_count,
                         const size_t keys_for_all_rows,
                         const HashType layout,
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

    VLOG(1) << "Initializing CPU Join Hash Table with " << keyspace_entry_count
            << " hash entries and " << one_to_many_hash_entries
            << " entries in the one to many buffer";
    VLOG(1) << "Total hash table size: " << hash_table_size << " Bytes";

    hash_table_ = std::make_unique<BaselineHashTable>(
        catalog_, layout, keyspace_entry_count, keys_for_all_rows, hash_table_size);
    auto cpu_hash_table_ptr = hash_table_->getCpuBuffer();
    int thread_count = cpu_threads();
    std::vector<std::future<void>> init_cpu_buff_threads;
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
           thread_count] {
            switch (key_component_width) {
              case 4: {
                return fill_baseline_hash_join_buff<int32_t>(cpu_hash_table_ptr,
                                                             keyspace_entry_count,
                                                             -1,
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
      init_hash_join_buff(one_to_many_buff, keyspace_entry_count, -1, 0, 1);
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
              composite_key_info.sd_inner_proxy_per_key,
              composite_key_info.sd_outer_proxy_per_key,
              thread_count);
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
              composite_key_info.sd_inner_proxy_per_key,
              composite_key_info.sd_outer_proxy_per_key,
              thread_count);
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
                            const int device_id) {
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
            << " entries in the one to many buffer";
    VLOG(1) << "Total hash table size: " << hash_table_size << " Bytes";

    hash_table_ = std::make_unique<BaselineHashTable>(catalog_,
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
                         const size_t key_component_width,
                         const size_t key_component_count,
                         const size_t keyspace_entry_count,
                         const size_t emitted_keys_count,
                         const int device_id,
                         const unsigned block_size,
                         const unsigned grid_size) {
    auto timer = DEBUG_TIMER(__func__);
    int err = 0;
#ifdef HAVE_CUDA
    allocateDeviceMemory(layout,
                         key_component_width,
                         key_component_count,
                         keyspace_entry_count,
                         emitted_keys_count,
                         device_id);
    if (!keyspace_entry_count) {
      // need to "allocate" the empty hash table first
      CHECK(!emitted_keys_count);
      return 0;
    }
    auto& data_mgr = catalog_->getDataMgr();
    CudaAllocator allocator(&data_mgr, device_id);
    auto dev_err_buff = reinterpret_cast<CUdeviceptr>(allocator.alloc(sizeof(int)));
    copy_to_gpu(&data_mgr, dev_err_buff, &err, sizeof(err), device_id);
    auto gpu_hash_table_buff = hash_table_->getGpuBuffer();
    CHECK(gpu_hash_table_buff);

    switch (key_component_width) {
      case 4:
        init_baseline_hash_join_buff_on_device_32(gpu_hash_table_buff,
                                                  keyspace_entry_count,
                                                  key_component_count,
                                                  layout == HashType::OneToOne,
                                                  -1,
                                                  block_size,
                                                  grid_size);
        break;
      case 8:
        init_baseline_hash_join_buff_on_device_64(gpu_hash_table_buff,
                                                  keyspace_entry_count,
                                                  key_component_count,
                                                  layout == HashType::OneToOne,
                                                  -1,
                                                  block_size,
                                                  grid_size);
        break;
      default:
        UNREACHABLE();
    }
    const auto key_handler_gpu = transfer_flat_object_to_gpu(*key_handler, allocator);
    switch (key_component_width) {
      case 4: {
        fill_baseline_hash_join_buff_on_device<int32_t>(
            gpu_hash_table_buff,
            keyspace_entry_count,
            -1,
            key_component_count,
            layout == HashType::OneToOne,
            reinterpret_cast<int*>(dev_err_buff),
            key_handler_gpu,
            join_columns.front().num_elems,
            block_size,
            grid_size);
        copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
        break;
      }
      case 8: {
        fill_baseline_hash_join_buff_on_device<int64_t>(
            gpu_hash_table_buff,
            keyspace_entry_count,
            -1,
            key_component_count,
            layout == HashType::OneToOne,
            reinterpret_cast<int*>(dev_err_buff),
            key_handler_gpu,
            join_columns.front().num_elems,
            block_size,
            grid_size);
        copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
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
      init_hash_join_buff_on_device(
          one_to_many_buff, keyspace_entry_count, -1, block_size, grid_size);
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
              join_columns.front().num_elems,
              block_size,
              grid_size);

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
              join_columns.front().num_elems,
              block_size,
              grid_size);

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

 private:
  const Catalog_Namespace::Catalog* catalog_;

  std::unique_ptr<BaselineHashTable> hash_table_;
};
