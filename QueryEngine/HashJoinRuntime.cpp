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

#include "HashJoinRuntime.h"
#ifdef __CUDACC__
#include "DecodersImpl.h"
#include "JoinHashImpl.h"
#else
#include "RuntimeFunctions.h"
#include "../StringDictionary/StringDictionary.h"
#include "../StringDictionary/StringDictionaryProxy.h"
#include <glog/logging.h>
#endif
#include "../Shared/funcannotations.h"

DEVICE void SUFFIX(init_hash_join_buff)(int32_t* groups_buffer,
                                        const int32_t hash_entry_count,
                                        const int32_t invalid_slot_val,
                                        const int32_t cpu_thread_idx,
                                        const int32_t cpu_thread_count) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  for (int32_t i = start; i < hash_entry_count; i += step) {
    groups_buffer[i] = invalid_slot_val;
  }
}

#ifdef __CUDACC__
#define mapd_cas(address, compare, val) atomicCAS(address, compare, val)
#else
#define mapd_cas(address, compare, val) __sync_val_compare_and_swap(address, compare, val)
#endif

DEVICE int SUFFIX(fill_hash_join_buff)(int32_t* buff,
                                       const int32_t invalid_slot_val,
                                       const JoinColumn join_column,
                                       const JoinColumnTypeInfo type_info,
                                       const void* sd_inner_proxy,
                                       const void* sd_outer_proxy,
                                       const int32_t cpu_thread_idx,
                                       const int32_t cpu_thread_count) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  for (size_t i = start; i < join_column.num_elems; i += step) {
    int64_t elem = SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    if (elem == type_info.null_val) {
      elem = type_info.translated_null_val;
    }
#ifndef __CUDACC__
    if (sd_inner_proxy && elem != type_info.translated_null_val) {
      CHECK(sd_outer_proxy);
      const auto sd_inner_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_inner_proxy);
      const auto sd_outer_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_outer_proxy);
      const auto elem_str = sd_inner_dict_proxy->getString(elem);
      const auto outer_id = sd_outer_dict_proxy->getIdOfString(elem_str);
      if (outer_id == StringDictionary::INVALID_STR_ID) {
        continue;
      }
      elem = outer_id;
    }
#endif
    int32_t* entry_ptr = SUFFIX(get_hash_slot)(buff, elem, type_info.min_val);
    if (mapd_cas(entry_ptr, invalid_slot_val, i) != invalid_slot_val) {
      return -1;
    }
  }
  return 0;
}

DEVICE int SUFFIX(fill_hash_join_buff_sharded)(int32_t* buff,
                                               const int32_t invalid_slot_val,
                                               const JoinColumn join_column,
                                               const JoinColumnTypeInfo type_info,
                                               const ShardInfo shard_info,
                                               const void* sd_inner_proxy,
                                               const void* sd_outer_proxy,
                                               const int32_t cpu_thread_idx,
                                               const int32_t cpu_thread_count) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  for (size_t i = start; i < join_column.num_elems; i += step) {
    int64_t elem = SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    if (elem % shard_info.num_shards != shard_info.shard) {
      continue;
    }
    if (elem == type_info.null_val) {
      elem = type_info.translated_null_val;
    }
#ifndef __CUDACC__
    if (sd_inner_proxy && elem != type_info.translated_null_val) {
      CHECK(sd_outer_proxy);
      const auto sd_inner_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_inner_proxy);
      const auto sd_outer_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_outer_proxy);
      const auto elem_str = sd_inner_dict_proxy->getString(elem);
      const auto outer_id = sd_outer_dict_proxy->getIdOfString(elem_str);
      if (outer_id == StringDictionary::INVALID_STR_ID) {
        continue;
      }
      elem = outer_id;
    }
#endif
    int32_t* entry_ptr = SUFFIX(get_hash_slot_sharded)(buff,
                                                       elem,
                                                       type_info.min_val,
                                                       shard_info.entry_count_per_shard,
                                                       shard_info.num_shards,
                                                       shard_info.device_count);
    if (mapd_cas(entry_ptr, invalid_slot_val, i) != invalid_slot_val) {
      return -1;
    }
  }
  return 0;
}

#undef mapd_cas

#ifdef __CUDACC__

__global__ void fill_hash_join_buff_wrapper(int32_t* buff,
                                            const int32_t invalid_slot_val,
                                            const JoinColumn join_column,
                                            const JoinColumnTypeInfo type_info,
                                            int* err) {
  int partial_err = SUFFIX(fill_hash_join_buff)(buff, invalid_slot_val, join_column, type_info, NULL, NULL, -1, -1);
  atomicCAS(err, 0, partial_err);
}

void fill_hash_join_buff_on_device(int32_t* buff,
                                   const int32_t invalid_slot_val,
                                   int* dev_err_buff,
                                   const JoinColumn join_column,
                                   const JoinColumnTypeInfo type_info,
                                   const size_t block_size_x,
                                   const size_t grid_size_x) {
  fill_hash_join_buff_wrapper<<<grid_size_x, block_size_x>>>(
      buff, invalid_slot_val, join_column, type_info, dev_err_buff);
}

__global__ void fill_hash_join_buff_wrapper_sharded(int32_t* buff,
                                                    const int32_t invalid_slot_val,
                                                    const JoinColumn join_column,
                                                    const JoinColumnTypeInfo type_info,
                                                    const ShardInfo shard_info,
                                                    int* err) {
  int partial_err = SUFFIX(fill_hash_join_buff_sharded)(
      buff, invalid_slot_val, join_column, type_info, shard_info, NULL, NULL, -1, -1);
  atomicCAS(err, 0, partial_err);
}

void fill_hash_join_buff_on_device_sharded(int32_t* buff,
                                           const int32_t invalid_slot_val,
                                           int* dev_err_buff,
                                           const JoinColumn join_column,
                                           const JoinColumnTypeInfo type_info,
                                           const ShardInfo shard_info,
                                           const size_t block_size_x,
                                           const size_t grid_size_x) {
  fill_hash_join_buff_wrapper_sharded<<<grid_size_x, block_size_x>>>(
      buff, invalid_slot_val, join_column, type_info, shard_info, dev_err_buff);
}

__global__ void init_hash_join_buff_wrapper(int32_t* buff,
                                            const int32_t hash_entry_count,
                                            const int32_t invalid_slot_val) {
  SUFFIX(init_hash_join_buff)(buff, hash_entry_count, invalid_slot_val, -1, -1);
}

void init_hash_join_buff_on_device(int32_t* buff,
                                   const int32_t hash_entry_count,
                                   const int32_t invalid_slot_val,
                                   const size_t block_size_x,
                                   const size_t grid_size_x) {
  init_hash_join_buff_wrapper<<<grid_size_x, block_size_x>>>(buff, hash_entry_count, invalid_slot_val);
}

#endif
