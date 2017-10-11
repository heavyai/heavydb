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
#include "CompareKeysInl.h"
#include "HyperLogLogRank.h"
#include "MurmurHash1Inl.h"
#ifdef __CUDACC__
#include "DecodersImpl.h"
#include "GpuRtConstants.h"
#include "JoinHashImpl.h"
#else
#include "RuntimeFunctions.h"
#include "../Shared/likely.h"
#include "../StringDictionary/StringDictionary.h"
#include "../StringDictionary/StringDictionaryProxy.h"
#include <glog/logging.h>

#include <future>
#endif

#if HAVE_CUDA
#include <thrust/scan.h>
#endif
#include "../Shared/funcannotations.h"

#include <numeric>

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
    int64_t elem = type_info.is_unsigned
                       ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                       : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_proxy && (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
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
    int64_t elem = type_info.is_unsigned
                       ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                       : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    if (elem % shard_info.num_shards != shard_info.shard) {
      continue;
    }
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_proxy && (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
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

template <typename T>
DEVICE void SUFFIX(init_baseline_hash_join_buff)(int8_t* hash_buff,
                                                 const size_t entry_count,
                                                 const size_t key_component_count,
                                                 const bool with_val_slot,
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
  const T empty_key = SUFFIX(get_invalid_key)<T>();
  for (uint32_t h = start; h < entry_count; h += step) {
    uint32_t off = h * (key_component_count + (with_val_slot ? 1 : 0)) * sizeof(T);
    auto row_ptr = reinterpret_cast<T*>(hash_buff + off);
    for (size_t i = 0; i < key_component_count; ++i) {
      row_ptr[i] = empty_key;
    }
    if (with_val_slot) {
      row_ptr[key_component_count] = invalid_slot_val;
    }
  }
}

#ifdef __CUDACC__
template <typename T>
__device__ T* get_matching_baseline_hash_slot_at(int8_t* hash_buff,
                                                 const uint32_t h,
                                                 const T* key,
                                                 const size_t key_component_count,
                                                 const bool with_val_slot) {
  uint32_t off = h * (key_component_count + (with_val_slot ? 1 : 0)) * sizeof(T);
  auto row_ptr = reinterpret_cast<T*>(hash_buff + off);
  const T empty_key = SUFFIX(get_invalid_key)<T>();
  {
    const T old = atomicCAS(row_ptr, empty_key, *key);
    if (empty_key == old && key_component_count > 1) {
      for (size_t i = 1; i <= key_component_count - 1; ++i) {
        atomicExch(row_ptr + i, key[i]);
      }
    }
  }
  if (key_component_count > 1) {
    while (atomicAdd(row_ptr + key_component_count - 1, 0) == empty_key) {
      // spin until the winning thread has finished writing the entire key and the init value
    }
  }
  bool match = true;
  for (uint32_t i = 0; i < key_component_count; ++i) {
    if (row_ptr[i] != key[i]) {
      match = false;
      break;
    }
  }

  if (match) {
    return reinterpret_cast<T*>(row_ptr + key_component_count);
  }
  return nullptr;
}
#else

#define cas_cst(ptr, expected, desired) \
  __atomic_compare_exchange_n(ptr, expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define store_cst(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST)
#define load_cst(ptr) __atomic_load_n(ptr, __ATOMIC_SEQ_CST)

template <typename T>
T* get_matching_baseline_hash_slot_at(int8_t* hash_buff,
                                      const uint32_t h,
                                      const T* key,
                                      const size_t key_component_count,
                                      const bool with_val_slot) {
  uint32_t off = h * (key_component_count + (with_val_slot ? 1 : 0)) * sizeof(T);
  auto row_ptr = reinterpret_cast<T*>(hash_buff + off);
  T empty_key = SUFFIX(get_invalid_key)<T>();
  T write_pending = SUFFIX(get_invalid_key)<T>() - 1;
  if (UNLIKELY(*key == write_pending)) {
    // Address the singularity case where the first column contains the pending
    // write special value. Should never happen, but avoid doing wrong things.
    return nullptr;
  }
  const bool success = cas_cst(row_ptr, &empty_key, write_pending);
  if (success) {
    if (key_component_count > 1) {
      memcpy(row_ptr + 1, key + 1, (key_component_count - 1) * sizeof(T));
    }
    store_cst(row_ptr, *key);
    return reinterpret_cast<T*>(row_ptr + key_component_count);
  }
  while (load_cst(row_ptr) == write_pending) {
    // spin until the winning thread has finished writing the entire key
  }
  for (size_t i = 0; i < key_component_count; ++i) {
    if (load_cst(row_ptr + i) != key[i]) {
      return nullptr;
    }
  }
  return reinterpret_cast<T*>(row_ptr + key_component_count);
}

#undef load_cst
#undef store_cst
#undef cas_cst

#endif  // __CUDACC__

template <typename T>
DEVICE int write_baseline_hash_slot(const int32_t val,
                                    int8_t* hash_buff,
                                    const size_t entry_count,
                                    const T* key,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const int32_t invalid_slot_val) {
  const uint32_t h = MurmurHash1Impl(key, key_component_count * sizeof(T), 0) % entry_count;
  T* matching_group = get_matching_baseline_hash_slot_at(hash_buff, h, key, key_component_count, with_val_slot);
  if (!matching_group) {
    uint32_t h_probe = (h + 1) % entry_count;
    while (h_probe != h) {
      matching_group = get_matching_baseline_hash_slot_at(hash_buff, h_probe, key, key_component_count, with_val_slot);
      if (matching_group) {
        break;
      }
      h_probe = (h_probe + 1) % entry_count;
    }
  }
  if (!matching_group) {
    return -2;
  }
  if (!with_val_slot) {
    return 0;
  }
  if (mapd_cas(matching_group, invalid_slot_val, val) != invalid_slot_val) {
    return -1;
  }
  return 0;
}

template <typename T>
DEVICE int SUFFIX(fill_baseline_hash_join_buff)(int8_t* hash_buff,
                                                const size_t entry_count,
                                                const int32_t invalid_slot_val,
                                                const size_t key_component_count,
                                                const bool with_val_slot,
                                                const JoinColumn* join_column_per_key,
                                                const JoinColumnTypeInfo* type_info_per_key,
                                                const void* const* sd_inner_proxy_per_key,
                                                const void* const* sd_outer_proxy_per_key,
                                                const int32_t cpu_thread_idx,
                                                const int32_t cpu_thread_count) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  const auto num_elems = join_column_per_key[0].num_elems;
  T key_scratch_buff[g_maximum_conditions_to_coalesce];
  for (size_t i = start; i < num_elems; i += step) {
    bool skip_entry = false;
    for (size_t key_component_index = 0; key_component_index < key_component_count; ++key_component_index) {
      const auto& join_column = join_column_per_key[key_component_index];
      const auto& type_info = type_info_per_key[key_component_index];
#ifndef __CUDACC__
      const auto sd_inner_proxy = sd_inner_proxy_per_key[key_component_index];
      const auto sd_outer_proxy = sd_outer_proxy_per_key[key_component_index];
#endif
      int64_t elem = type_info.is_unsigned
                         ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                         : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
      if (elem == type_info.null_val && !type_info.uses_bw_eq) {
        skip_entry = true;
        break;
      }
#ifndef __CUDACC__
      if (sd_inner_proxy && elem != type_info.null_val) {
        CHECK(sd_outer_proxy);
        const auto sd_inner_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_inner_proxy);
        const auto sd_outer_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_outer_proxy);
        const auto elem_str = sd_inner_dict_proxy->getString(elem);
        const auto outer_id = sd_outer_dict_proxy->getIdOfString(elem_str);
        if (outer_id == StringDictionary::INVALID_STR_ID) {
          skip_entry = true;
          break;
        }
        elem = outer_id;
      }
#endif
      key_scratch_buff[key_component_index] = elem;
    }
    if (!skip_entry) {
      int err = write_baseline_hash_slot<T>(
          i, hash_buff, entry_count, key_scratch_buff, key_component_count, with_val_slot, invalid_slot_val);
      if (err) {
        return err;
      }
    }
  }
  return 0;
}

#undef mapd_cas

#ifdef __CUDACC__
#define mapd_add(address, val) atomicAdd(address, val)
#else
#define mapd_add(address, val) __sync_fetch_and_add(address, val)
#endif

GLOBAL void SUFFIX(count_matches)(int32_t* count_buff,
                                  const int32_t invalid_slot_val,
                                  const JoinColumn join_column,
                                  const JoinColumnTypeInfo type_info
#ifndef __CUDACC__
                                  ,
                                  const void* sd_inner_proxy,
                                  const void* sd_outer_proxy,
                                  const int32_t cpu_thread_idx,
                                  const int32_t cpu_thread_count
#endif
                                  ) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  for (size_t i = start; i < join_column.num_elems; i += step) {
    int64_t elem = type_info.is_unsigned
                       ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                       : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_proxy && (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
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
    int32_t* entry_ptr = SUFFIX(get_hash_slot)(count_buff, elem, type_info.min_val);
    mapd_add(entry_ptr, int32_t(1));
  }
}

GLOBAL void SUFFIX(count_matches_sharded)(int32_t* count_buff,
                                          const int32_t invalid_slot_val,
                                          const JoinColumn join_column,
                                          const JoinColumnTypeInfo type_info,
                                          const ShardInfo shard_info
#ifndef __CUDACC__
                                          ,
                                          const void* sd_inner_proxy,
                                          const void* sd_outer_proxy,
                                          const int32_t cpu_thread_idx,
                                          const int32_t cpu_thread_count
#endif
                                          ) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  for (size_t i = start; i < join_column.num_elems; i += step) {
    int64_t elem = type_info.is_unsigned
                       ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                       : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_proxy && (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
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
    int32_t* entry_ptr = SUFFIX(get_hash_slot_sharded)(count_buff,
                                                       elem,
                                                       type_info.min_val,
                                                       shard_info.entry_count_per_shard,
                                                       shard_info.num_shards,
                                                       shard_info.device_count);
    mapd_add(entry_ptr, int32_t(1));
  }
}

template <typename T>
DEVICE NEVER_INLINE const T* SUFFIX(get_matching_baseline_hash_slot_readonly)(const T* key,
                                                                              const size_t key_component_count,
                                                                              const T* composite_key_dict,
                                                                              const size_t entry_count) {
  const uint32_t h = MurmurHash1Impl(key, key_component_count * sizeof(T), 0) % entry_count;
  uint32_t off = h * key_component_count;
  if (keys_are_equal(&composite_key_dict[off], key, key_component_count)) {
    return &composite_key_dict[off];
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    off = h_probe * key_component_count;
    if (keys_are_equal(&composite_key_dict[off], key, key_component_count)) {
      return &composite_key_dict[off];
    }
    h_probe = (h_probe + 1) % entry_count;
  }
#ifndef __CUDACC__
  CHECK(false);
#else
  assert(false);
#endif
  return nullptr;
}

template <typename T>
GLOBAL void SUFFIX(count_matches_baseline)(int32_t* count_buff,
                                           const T* composite_key_dict,
                                           const size_t entry_count,
                                           const int32_t invalid_slot_val,
                                           const size_t key_component_count,
                                           const JoinColumn* join_column_per_key,
                                           const JoinColumnTypeInfo* type_info_per_key
#ifndef __CUDACC__
                                           ,
                                           const void* const* sd_inner_proxy_per_key,
                                           const void* const* sd_outer_proxy_per_key,
                                           const int32_t cpu_thread_idx,
                                           const int32_t cpu_thread_count
#endif
                                           ) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  const auto num_elems = join_column_per_key[0].num_elems;
  T key_scratch_buff[g_maximum_conditions_to_coalesce];
  for (size_t i = start; i < num_elems; i += step) {
    bool skip_entry = false;
    for (size_t key_component_index = 0; key_component_index < key_component_count; ++key_component_index) {
      const auto& join_column = join_column_per_key[key_component_index];
      const auto& type_info = type_info_per_key[key_component_index];
#ifndef __CUDACC__
      const auto sd_inner_proxy = sd_inner_proxy_per_key[key_component_index];
      const auto sd_outer_proxy = sd_outer_proxy_per_key[key_component_index];
#endif
      int64_t elem = type_info.is_unsigned
                         ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                         : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
      if (elem == type_info.null_val && !type_info.uses_bw_eq) {
        skip_entry = true;
        break;
      }
#ifndef __CUDACC__
      if (sd_inner_proxy && elem != type_info.null_val) {
        CHECK(sd_outer_proxy);
        const auto sd_inner_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_inner_proxy);
        const auto sd_outer_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_outer_proxy);
        const auto elem_str = sd_inner_dict_proxy->getString(elem);
        const auto outer_id = sd_outer_dict_proxy->getIdOfString(elem_str);
        if (outer_id == StringDictionary::INVALID_STR_ID) {
          skip_entry = true;
          break;
        }
        elem = outer_id;
      }
#endif
      key_scratch_buff[key_component_index] = elem;
    }
#ifdef __CUDACC__
    assert(composite_key_dict);
#endif
    if (!skip_entry) {
      const auto matching_group = SUFFIX(get_matching_baseline_hash_slot_readonly)(
          key_scratch_buff, key_component_count, composite_key_dict, entry_count);
      const auto entry_idx = (matching_group - composite_key_dict) / key_component_count;
      mapd_add(&count_buff[entry_idx], int32_t(1));
    }
  }
}

GLOBAL void SUFFIX(fill_row_ids)(int32_t* buff,
                                 const int32_t hash_entry_count,
                                 const int32_t invalid_slot_val,
                                 const JoinColumn join_column,
                                 const JoinColumnTypeInfo type_info
#ifndef __CUDACC__
                                 ,
                                 const void* sd_inner_proxy,
                                 const void* sd_outer_proxy,
                                 const int32_t cpu_thread_idx,
                                 const int32_t cpu_thread_count
#endif
                                 ) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  int32_t* id_buff = count_buff + hash_entry_count;

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  for (size_t i = start; i < join_column.num_elems; i += step) {
    int64_t elem = type_info.is_unsigned
                       ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                       : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_proxy && (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
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
    int32_t* pos_ptr = SUFFIX(get_hash_slot)(pos_buff, elem, type_info.min_val);
#ifndef __CUDACC__
    CHECK_NE(*pos_ptr, invalid_slot_val);
#endif
    const auto bin_idx = pos_ptr - pos_buff;
    const auto id_buff_idx = mapd_add(count_buff + bin_idx, 1) + *pos_ptr;
    id_buff[id_buff_idx] = static_cast<int32_t>(i);
  }
}

GLOBAL void SUFFIX(fill_row_ids_sharded)(int32_t* buff,
                                         const int32_t hash_entry_count,
                                         const int32_t invalid_slot_val,
                                         const JoinColumn join_column,
                                         const JoinColumnTypeInfo type_info,
                                         const ShardInfo shard_info
#ifndef __CUDACC__
                                         ,
                                         const void* sd_inner_proxy,
                                         const void* sd_outer_proxy,
                                         const int32_t cpu_thread_idx,
                                         const int32_t cpu_thread_count
#endif
                                         ) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  int32_t* id_buff = count_buff + hash_entry_count;

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  for (size_t i = start; i < join_column.num_elems; i += step) {
    int64_t elem = type_info.is_unsigned
                       ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                       : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_proxy && (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
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
    int32_t* pos_ptr = SUFFIX(get_hash_slot_sharded)(pos_buff,
                                                     elem,
                                                     type_info.min_val,
                                                     shard_info.entry_count_per_shard,
                                                     shard_info.num_shards,
                                                     shard_info.device_count);
#ifndef __CUDACC__
    CHECK_NE(*pos_ptr, invalid_slot_val);
#endif
    const auto bin_idx = pos_ptr - pos_buff;
    const auto id_buff_idx = mapd_add(count_buff + bin_idx, 1) + *pos_ptr;
    id_buff[id_buff_idx] = static_cast<int32_t>(i);
  }
}

template <typename T>
GLOBAL void SUFFIX(fill_row_ids_baseline)(int32_t* buff,
                                          const T* composite_key_dict,
                                          const size_t hash_entry_count,
                                          const int32_t invalid_slot_val,
                                          const size_t key_component_count,
                                          const JoinColumn* join_column_per_key,
                                          const JoinColumnTypeInfo* type_info_per_key
#ifndef __CUDACC__
                                          ,
                                          const void* const* sd_inner_proxy_per_key,
                                          const void* const* sd_outer_proxy_per_key,
                                          const int32_t cpu_thread_idx,
                                          const int32_t cpu_thread_count
#endif
                                          ) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  int32_t* id_buff = count_buff + hash_entry_count;
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  const auto num_elems = join_column_per_key[0].num_elems;
  T key_scratch_buff[g_maximum_conditions_to_coalesce];
  for (size_t i = start; i < num_elems; i += step) {
    bool skip_entry = false;
    for (size_t key_component_index = 0; key_component_index < key_component_count; ++key_component_index) {
      const auto& join_column = join_column_per_key[key_component_index];
      const auto& type_info = type_info_per_key[key_component_index];
#ifndef __CUDACC__
      const auto sd_inner_proxy = sd_inner_proxy_per_key[key_component_index];
      const auto sd_outer_proxy = sd_outer_proxy_per_key[key_component_index];
#endif
      int64_t elem = type_info.is_unsigned
                         ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
                         : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
      if (elem == type_info.null_val && !type_info.uses_bw_eq) {
        skip_entry = true;
        break;
      }
#ifndef __CUDACC__
      if (sd_inner_proxy && elem != type_info.null_val) {
        CHECK(sd_outer_proxy);
        const auto sd_inner_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_inner_proxy);
        const auto sd_outer_dict_proxy = static_cast<const StringDictionaryProxy*>(sd_outer_proxy);
        const auto elem_str = sd_inner_dict_proxy->getString(elem);
        const auto outer_id = sd_outer_dict_proxy->getIdOfString(elem_str);
        if (outer_id == StringDictionary::INVALID_STR_ID) {
          skip_entry = true;
          break;
        }
        elem = outer_id;
      }
#endif
      key_scratch_buff[key_component_index] = elem;
    }
#ifdef __CUDACC__
    assert(composite_key_dict);
#endif
    if (!skip_entry) {
      const T* matching_group = SUFFIX(get_matching_baseline_hash_slot_readonly)(
          key_scratch_buff, key_component_count, composite_key_dict, hash_entry_count);
      const auto entry_idx = (matching_group - composite_key_dict) / key_component_count;
      int32_t* pos_ptr = pos_buff + entry_idx;
#ifndef __CUDACC__
      CHECK_NE(*pos_ptr, invalid_slot_val);
#endif
      const auto bin_idx = pos_ptr - pos_buff;
      const auto id_buff_idx = mapd_add(count_buff + bin_idx, 1) + *pos_ptr;
      id_buff[id_buff_idx] = static_cast<int32_t>(i);
    }
  }
  return;
}

#undef mapd_add

namespace {

GLOBAL void SUFFIX(approximate_distinct_tuples_impl)(uint8_t* hll_buffer,
                                                     const uint32_t b,
                                                     const size_t key_component_count,
                                                     const JoinColumn* join_column_per_key,
                                                     const JoinColumnTypeInfo* type_info_per_key
#ifndef __CUDACC__
                                                     ,
                                                     const int32_t cpu_thread_idx,
                                                     const int32_t cpu_thread_count
#endif
                                                     ) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  const auto num_elems = join_column_per_key[0].num_elems;
  int64_t key_scratch_buff[g_maximum_conditions_to_coalesce];
  for (size_t i = start; i < num_elems; i += step) {
    for (size_t key_component_index = 0; key_component_index < key_component_count; ++key_component_index) {
      const auto& join_column = join_column_per_key[key_component_index];
      const auto& type_info = type_info_per_key[key_component_index];
      key_scratch_buff[key_component_index] =
          type_info.is_unsigned
              ? SUFFIX(fixed_width_unsigned_decode_noinline)(join_column.col_buff, type_info.elem_sz, i)
              : SUFFIX(fixed_width_int_decode_noinline)(join_column.col_buff, type_info.elem_sz, i);
    }
    const uint64_t hash = MurmurHash64AImpl(key_scratch_buff, key_component_count * sizeof(int64_t), 0);
    const uint32_t index = hash >> (64 - b);
    const auto rank = get_rank(hash << b, 64 - b);
#ifdef __CUDACC__
    atomicMax(reinterpret_cast<int32_t*>(hll_buffer) + index, rank);
#else
    hll_buffer[index] = std::max(hll_buffer[index], rank);
#endif
  }
}

}  // namespace

#ifndef __CUDACC__

template <typename InputIterator, typename OutputIterator>
void inclusive_scan(InputIterator first, InputIterator last, OutputIterator out, const size_t thread_count) {
  typedef typename InputIterator::value_type ElementType;
  typedef typename InputIterator::difference_type OffsetType;
  const OffsetType elem_count = last - first;
  if (elem_count < 10000 || thread_count <= 1) {
    ElementType sum = 0;
    for (auto iter = first; iter != last; ++iter, ++out) {
      *out = sum += *iter;
    }
    return;
  }

  const OffsetType step = (elem_count + thread_count - 1) / thread_count;
  OffsetType start_off = 0;
  OffsetType end_off = std::min(step, elem_count);
  std::vector<ElementType> partial_sums(thread_count);
  std::vector<std::future<void>> counter_threads;
  for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx,
              start_off = std::min(start_off + step, elem_count),
              end_off = std::min(start_off + step, elem_count)) {
    counter_threads.push_back(std::async(
        std::launch::async,
        [first, out](ElementType& partial_sum, const OffsetType start, const OffsetType end) {
          ElementType sum = 0;
          for (auto in_iter = first + start, out_iter = out + start; in_iter != (first + end); ++in_iter, ++out_iter) {
            *out_iter = sum += *in_iter;
          }
          partial_sum = sum;
        },
        std::ref(partial_sums[thread_idx]),
        start_off,
        end_off));
  }
  for (auto& child : counter_threads) {
    child.get();
  }

  ElementType sum = 0;
  for (auto& s : partial_sums) {
    s += sum;
    sum = s;
  }

  counter_threads.clear();
  start_off = std::min(step, elem_count);
  end_off = std::min(start_off + step, elem_count);
  for (size_t thread_idx = 0; thread_idx < thread_count - 1; ++thread_idx,
              start_off = std::min(start_off + step, elem_count),
              end_off = std::min(start_off + step, elem_count)) {
    counter_threads.push_back(
        std::async(std::launch::async,
                   [out](const ElementType prev_sum, const OffsetType start, const OffsetType end) {
                     for (auto iter = out + start; iter != (out + end); ++iter) {
                       *iter += prev_sum;
                     }
                   },
                   partial_sums[thread_idx],
                   start_off,
                   end_off));
  }
  for (auto& child : counter_threads) {
    child.get();
  }
}

void fill_one_to_many_hash_table(int32_t* buff,
                                 const int32_t hash_entry_count,
                                 const int32_t invalid_slot_val,
                                 const JoinColumn& join_column,
                                 const JoinColumnTypeInfo& type_info,
                                 const void* sd_inner_proxy,
                                 const void* sd_outer_proxy,
                                 const int32_t cpu_thread_count) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  memset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  std::vector<std::future<void>> counter_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    counter_threads.push_back(std::async(std::launch::async,
                                         count_matches,
                                         count_buff,
                                         invalid_slot_val,
                                         join_column,
                                         type_info,
                                         sd_inner_proxy,
                                         sd_outer_proxy,
                                         cpu_thread_idx,
                                         cpu_thread_count));
  }

  for (auto& child : counter_threads) {
    child.get();
  }

  std::vector<int32_t> count_copy(hash_entry_count, 0);
  CHECK_GT(hash_entry_count, int32_t(0));
  memcpy(&count_copy[1], count_buff, (hash_entry_count - 1) * sizeof(int32_t));
#if HAVE_CUDA
  thrust::inclusive_scan(count_copy.begin(), count_copy.end(), count_copy.begin());
#else
  inclusive_scan(count_copy.begin(), count_copy.end(), count_copy.begin(), cpu_thread_count);
#endif
  std::vector<std::future<void>> pos_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    pos_threads.push_back(std::async(std::launch::async,
                                     [&](const int thread_idx) {
                                       for (int i = thread_idx; i < hash_entry_count; i += cpu_thread_count) {
                                         if (count_buff[i]) {
                                           pos_buff[i] = count_copy[i];
                                         }
                                       }
                                     },
                                     cpu_thread_idx));
  }
  for (auto& child : pos_threads) {
    child.get();
  }

  memset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  std::vector<std::future<void>> rowid_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    rowid_threads.push_back(std::async(std::launch::async,
                                       SUFFIX(fill_row_ids),
                                       buff,
                                       hash_entry_count,
                                       invalid_slot_val,
                                       std::ref(join_column),
                                       std::ref(type_info),
                                       sd_inner_proxy,
                                       sd_outer_proxy,
                                       cpu_thread_idx,
                                       cpu_thread_count));
  }

  for (auto& child : rowid_threads) {
    child.get();
  }
}

void fill_one_to_many_hash_table_sharded(int32_t* buff,
                                         const int32_t hash_entry_count,
                                         const int32_t invalid_slot_val,
                                         const JoinColumn& join_column,
                                         const JoinColumnTypeInfo& type_info,
                                         const ShardInfo& shard_info,
                                         const void* sd_inner_proxy,
                                         const void* sd_outer_proxy,
                                         const int32_t cpu_thread_count) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  memset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  std::vector<std::future<void>> counter_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    counter_threads.push_back(std::async(std::launch::async,
                                         count_matches_sharded,
                                         count_buff,
                                         invalid_slot_val,
                                         std::ref(join_column),
                                         std::ref(type_info),
                                         std::ref(shard_info),
                                         sd_inner_proxy,
                                         sd_outer_proxy,
                                         cpu_thread_idx,
                                         cpu_thread_count));
  }

  for (auto& child : counter_threads) {
    child.get();
  }

  std::vector<int32_t> count_copy(hash_entry_count, 0);
  CHECK_GT(hash_entry_count, int32_t(0));
  memcpy(&count_copy[1], count_buff, (hash_entry_count - 1) * sizeof(int32_t));
  inclusive_scan(count_copy.begin(), count_copy.end(), count_copy.begin(), cpu_thread_count);
  std::vector<std::future<void>> pos_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    pos_threads.push_back(std::async(std::launch::async,
                                     [&](const int thread_idx) {
                                       for (int i = thread_idx; i < hash_entry_count; i += cpu_thread_count) {
                                         if (count_buff[i]) {
                                           pos_buff[i] = count_copy[i];
                                         }
                                       }
                                     },
                                     cpu_thread_idx));
  }
  for (auto& child : pos_threads) {
    child.get();
  }

  memset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  std::vector<std::future<void>> rowid_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    rowid_threads.push_back(std::async(std::launch::async,
                                       SUFFIX(fill_row_ids_sharded),
                                       buff,
                                       hash_entry_count,
                                       invalid_slot_val,
                                       std::ref(join_column),
                                       std::ref(type_info),
                                       std::ref(shard_info),
                                       sd_inner_proxy,
                                       sd_outer_proxy,
                                       cpu_thread_idx,
                                       cpu_thread_count));
  }

  for (auto& child : rowid_threads) {
    child.get();
  }
}

void init_baseline_hash_join_buff_32(int8_t* hash_join_buff,
                                     const int32_t entry_count,
                                     const size_t key_component_count,
                                     const bool with_val_slot,
                                     const int32_t invalid_slot_val,
                                     const int32_t cpu_thread_idx,
                                     const int32_t cpu_thread_count) {
  init_baseline_hash_join_buff<int32_t>(hash_join_buff,
                                        entry_count,
                                        key_component_count,
                                        with_val_slot,
                                        invalid_slot_val,
                                        cpu_thread_idx,
                                        cpu_thread_count);
}

void init_baseline_hash_join_buff_64(int8_t* hash_join_buff,
                                     const int32_t entry_count,
                                     const size_t key_component_count,
                                     const bool with_val_slot,
                                     const int32_t invalid_slot_val,
                                     const int32_t cpu_thread_idx,
                                     const int32_t cpu_thread_count) {
  init_baseline_hash_join_buff<int64_t>(hash_join_buff,
                                        entry_count,
                                        key_component_count,
                                        with_val_slot,
                                        invalid_slot_val,
                                        cpu_thread_idx,
                                        cpu_thread_count);
}

int fill_baseline_hash_join_buff_32(int8_t* hash_buff,
                                    const size_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const std::vector<JoinColumn>& join_column_per_key,
                                    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                    const std::vector<const void*>& sd_inner_proxy_per_key,
                                    const std::vector<const void*>& sd_outer_proxy_per_key,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count) {
  return fill_baseline_hash_join_buff<int32_t>(hash_buff,
                                               entry_count,
                                               invalid_slot_val,
                                               key_component_count,
                                               with_val_slot,
                                               &join_column_per_key[0],
                                               &type_info_per_key[0],
                                               &sd_inner_proxy_per_key[0],
                                               &sd_outer_proxy_per_key[0],
                                               cpu_thread_idx,
                                               cpu_thread_count);
}

int fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                    const size_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const std::vector<JoinColumn>& join_column_per_key,
                                    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                    const std::vector<const void*>& sd_inner_proxy_per_key,
                                    const std::vector<const void*>& sd_outer_proxy_per_key,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count) {
  return fill_baseline_hash_join_buff<int64_t>(hash_buff,
                                               entry_count,
                                               invalid_slot_val,
                                               key_component_count,
                                               with_val_slot,
                                               &join_column_per_key[0],
                                               &type_info_per_key[0],
                                               &sd_inner_proxy_per_key[0],
                                               &sd_outer_proxy_per_key[0],
                                               cpu_thread_idx,
                                               cpu_thread_count);
}

template <typename T>
void fill_one_to_many_baseline_hash_table(int32_t* buff,
                                          const T* composite_key_dict,
                                          const size_t hash_entry_count,
                                          const int32_t invalid_slot_val,
                                          const size_t key_component_count,
                                          const std::vector<JoinColumn>& join_column_per_key,
                                          const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                          const std::vector<const void*>& sd_inner_proxy_per_key,
                                          const std::vector<const void*>& sd_outer_proxy_per_key,
                                          const int32_t cpu_thread_count) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  memset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  std::vector<std::future<void>> counter_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    counter_threads.push_back(std::async(std::launch::async,
                                         &count_matches_baseline<T>,
                                         count_buff,
                                         composite_key_dict,
                                         hash_entry_count,
                                         invalid_slot_val,
                                         key_component_count,
                                         &join_column_per_key[0],
                                         &type_info_per_key[0],
                                         &sd_inner_proxy_per_key[0],
                                         &sd_outer_proxy_per_key[0],
                                         cpu_thread_idx,
                                         cpu_thread_count));
  }

  for (auto& child : counter_threads) {
    child.get();
  }

  std::vector<int32_t> count_copy(hash_entry_count, 0);
  CHECK_GT(hash_entry_count, int32_t(0));
  memcpy(&count_copy[1], count_buff, (hash_entry_count - 1) * sizeof(int32_t));
  inclusive_scan(count_copy.begin(), count_copy.end(), count_copy.begin(), cpu_thread_count);
  std::vector<std::future<void>> pos_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    pos_threads.push_back(std::async(std::launch::async,
                                     [&](const int thread_idx) {
                                       for (size_t i = thread_idx; i < hash_entry_count; i += cpu_thread_count) {
                                         if (count_buff[i]) {
                                           pos_buff[i] = count_copy[i];
                                         }
                                       }
                                     },
                                     cpu_thread_idx));
  }
  for (auto& child : pos_threads) {
    child.get();
  }

  memset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  std::vector<std::future<void>> rowid_threads;
  for (int cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    SUFFIX(fill_row_ids_baseline)
    (buff,
     composite_key_dict,
     hash_entry_count,
     invalid_slot_val,
     key_component_count,
     &join_column_per_key[0],
     &type_info_per_key[0],
     &sd_inner_proxy_per_key[0],
     &sd_outer_proxy_per_key[0],
     cpu_thread_idx,
     cpu_thread_count);
  }

  for (auto& child : rowid_threads) {
    child.get();
  }
}

void fill_one_to_many_baseline_hash_table_32(int32_t* buff,
                                             const int32_t* composite_key_dict,
                                             const size_t hash_entry_count,
                                             const int32_t invalid_slot_val,
                                             const size_t key_component_count,
                                             const std::vector<JoinColumn>& join_column_per_key,
                                             const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                             const std::vector<const void*>& sd_inner_proxy_per_key,
                                             const std::vector<const void*>& sd_outer_proxy_per_key,
                                             const int32_t cpu_thread_count) {
  fill_one_to_many_baseline_hash_table<int32_t>(buff,
                                                composite_key_dict,
                                                hash_entry_count,
                                                invalid_slot_val,
                                                key_component_count,
                                                join_column_per_key,
                                                type_info_per_key,
                                                sd_inner_proxy_per_key,
                                                sd_outer_proxy_per_key,
                                                cpu_thread_count);
}

void fill_one_to_many_baseline_hash_table_64(int32_t* buff,
                                             const int64_t* composite_key_dict,
                                             const size_t hash_entry_count,
                                             const int32_t invalid_slot_val,
                                             const size_t key_component_count,
                                             const std::vector<JoinColumn>& join_column_per_key,
                                             const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                             const std::vector<const void*>& sd_inner_proxy_per_key,
                                             const std::vector<const void*>& sd_outer_proxy_per_key,
                                             const int32_t cpu_thread_count) {
  fill_one_to_many_baseline_hash_table<int64_t>(buff,
                                                composite_key_dict,
                                                hash_entry_count,
                                                invalid_slot_val,
                                                key_component_count,
                                                join_column_per_key,
                                                type_info_per_key,
                                                sd_inner_proxy_per_key,
                                                sd_outer_proxy_per_key,
                                                cpu_thread_count);
}

void approximate_distinct_tuples(uint8_t* hll_buffer_all_cpus,
                                 const uint32_t b,
                                 const size_t padded_size_bytes,
                                 const std::vector<JoinColumn>& join_column_per_key,
                                 const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                 const int thread_count) {
  CHECK_EQ(join_column_per_key.size(), type_info_per_key.size());
  CHECK(!join_column_per_key.empty());
  std::vector<std::future<void>> approx_distinct_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    approx_distinct_threads.push_back(std::async(std::launch::async,
                                                 [&join_column_per_key,
                                                  &type_info_per_key,
                                                  b,
                                                  hll_buffer_all_cpus,
                                                  padded_size_bytes,
                                                  thread_idx,
                                                  thread_count] {
                                                   auto hll_buffer =
                                                       hll_buffer_all_cpus + thread_idx * padded_size_bytes;
                                                   approximate_distinct_tuples_impl(hll_buffer,
                                                                                    b,
                                                                                    join_column_per_key.size(),
                                                                                    &join_column_per_key[0],
                                                                                    &type_info_per_key[0],
                                                                                    thread_idx,
                                                                                    thread_count);
                                                 }));
  }
  for (auto& child : approx_distinct_threads) {
    child.get();
  }
}

#endif
