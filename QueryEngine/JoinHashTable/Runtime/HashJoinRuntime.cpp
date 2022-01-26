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

#include "QueryEngine/CompareKeysInl.h"
#include "QueryEngine/HyperLogLogRank.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinKeyHandlers.h"
#include "QueryEngine/JoinHashTable/Runtime/JoinColumnIterator.h"
#include "QueryEngine/MurmurHash1Inl.h"
#ifdef __CUDACC__
#include "QueryEngine/DecodersImpl.h"
#include "QueryEngine/GpuRtConstants.h"
#include "QueryEngine/JoinHashTable/Runtime/JoinHashImpl.h"
#else
#include "Logger/Logger.h"

#include "QueryEngine/RuntimeFunctions.h"
#include "Shared/likely.h"
#include "StringDictionary/StringDictionary.h"
#include "StringDictionary/StringDictionaryProxy.h"

#include <x86intrin.h>

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#endif

#include <future>
#endif

#if HAVE_CUDA
#include <thrust/scan.h>
#endif
#include "Shared/funcannotations.h"

#include <cmath>
#include <numeric>

#ifndef __CUDACC__
namespace {

/**
 * Joins between two dictionary encoded string columns without a shared string dictionary
 * are computed by translating the inner dictionary to the outer dictionary while filling
 * the  hash table. The translation works as follows:
 *
 * Given two tables t1 and t2, with t1 the outer table and t2 the inner table, and two
 * columns t1.x and t2.x, both dictionary encoded strings without a shared dictionary, we
 * read each value in t2.x and do a lookup in the dictionary for t1.x. If the lookup
 * returns a valid ID, we insert that ID into the hash table. Otherwise, we skip adding an
 * entry into the hash table for the inner column. We can also skip adding any entries
 * that are outside the range of the outer column.
 *
 * Consider a join of the form SELECT x, n FROM (SELECT x, COUNT(*) n FROM t1 GROUP BY x
 * HAVING n > 10), t2 WHERE t1.x = t2.x; Let the result of the subquery be t1_s.
 * Due to the HAVING clause, the range of all IDs in t1_s must be less than or equal to
 * the range of all IDs in t1. Suppose we have an element a in t2.x that is also in
 * t1_s.x. Then the ID of a must be within the range of t1_s. Therefore it is safe to
 * ignore any element ID that is not in the dictionary corresponding to t1_s.x or is
 * outside the range of column t1_s.
 */
inline int64_t translate_str_id_to_outer_dict(const int64_t elem,
                                              const int64_t min_elem,
                                              const int64_t max_elem,
                                              const void* sd_inner_proxy,
                                              const void* sd_outer_proxy) {
  CHECK(sd_outer_proxy);
  const auto sd_inner_dict_proxy =
      static_cast<const StringDictionaryProxy*>(sd_inner_proxy);
  const auto sd_outer_dict_proxy =
      static_cast<const StringDictionaryProxy*>(sd_outer_proxy);
  const auto elem_str = sd_inner_dict_proxy->getString(elem);
  const auto outer_id = sd_outer_dict_proxy->getIdOfString(elem_str);
  if (outer_id > max_elem || outer_id < min_elem) {
    return StringDictionary::INVALID_STR_ID;
  }
  return outer_id;
}

inline int64_t map_str_id_to_outer_dict(const int64_t inner_elem,
                                        const int64_t min_inner_elem,
                                        const int64_t min_outer_elem,
                                        const int64_t max_outer_elem,
                                        const int32_t* inner_to_outer_translation_map) {
  const auto outer_id = inner_to_outer_translation_map[inner_elem - min_inner_elem];
  if (outer_id > max_outer_elem || outer_id < min_outer_elem) {
    return StringDictionary::INVALID_STR_ID;
  }
  return outer_id;
}

/**
 * For non-AVX512 we are fine with auto-vectorized loop.
 */
__attribute__((target("default"))) void init_hash_join_buff_cpu(
    int32_t* groups_buffer,
    const int32_t invalid_slot_val,
    const int64_t start,
    const int64_t end) {
  for (int64_t pos = start; pos < end; ++pos) {
    groups_buffer[pos] = invalid_slot_val;
  }
}

/**
 * For AVX512 target we perform manual vectorization to use non-temporal stores.
 */
__attribute__((target("avx512f"), optimize("no-tree-vectorize"))) void
init_hash_join_buff_cpu(int32_t* groups_buffer,
                        const int32_t invalid_slot_val,
                        const int64_t start,
                        const int64_t end) {
  int64_t pos = start;

  // Align buffer pointer.
  int64_t align_iters =
      ((64ULL - reinterpret_cast<uint64_t>(groups_buffer + pos)) & 0x3F) /
      sizeof(invalid_slot_val);
  int64_t align_end = std::min(pos + align_iters, end);
  while (pos < align_end) {
    groups_buffer[pos++] = invalid_slot_val;
  }

  // Fill using 512-byte vector template.
  __m512i vec_val = (__m512i)(__v16si){invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val,
                                       invalid_slot_val};
  int64_t vec_end = pos + (end - pos) / 16 * 16;
  while (pos < vec_end) {
    _mm512_stream_si512(reinterpret_cast<__m512i*>(groups_buffer + pos), vec_val);
    pos += 16;
  }

  // Scalar tail.
  while (pos < end) {
    groups_buffer[pos++] = invalid_slot_val;
  }
}

}  // namespace
#endif

DEVICE void SUFFIX(init_hash_join_buff)(int32_t* groups_buffer,
                                        const int64_t hash_entry_count,
                                        const int32_t invalid_slot_val,
                                        const int32_t cpu_thread_idx,
                                        const int32_t cpu_thread_count) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
  for (int64_t i = start; i < hash_entry_count; i += step) {
    groups_buffer[i] = invalid_slot_val;
  }
#else
  int64_t start = hash_entry_count * cpu_thread_idx / cpu_thread_count;
  int32_t end = hash_entry_count * (cpu_thread_idx + 1) / cpu_thread_count;
  init_hash_join_buff_cpu(groups_buffer, invalid_slot_val, start, end);
#endif
}

#ifndef __CUDACC__
#ifdef HAVE_TBB

void SUFFIX(init_hash_join_buff_tbb)(int32_t* groups_buffer,
                                     const int64_t hash_entry_count,
                                     const int32_t invalid_slot_val) {
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, hash_entry_count),
                    [=](const tbb::blocked_range<int64_t>& r) {
                      const auto start_idx = r.begin();
                      const auto end_idx = r.end();
                      for (auto entry_idx = start_idx; entry_idx != end_idx;
                           ++entry_idx) {
                        groups_buffer[entry_idx] = invalid_slot_val;
                      }
                    });
}

#endif  // #ifdef HAVE_TBB
#endif  // #ifndef __CUDACC__

#ifdef __CUDACC__
#define mapd_cas(address, compare, val) atomicCAS(address, compare, val)
#elif defined(_MSC_VER)
#define mapd_cas(address, compare, val)                                 \
  InterlockedCompareExchange(reinterpret_cast<volatile long*>(address), \
                             static_cast<long>(val),                    \
                             static_cast<long>(compare))
#else
#define mapd_cas(address, compare, val) __sync_val_compare_and_swap(address, compare, val)
#endif

template <typename HASHTABLE_FILLING_FUNC>
DEVICE auto fill_hash_join_buff_impl(int32_t* buff,
                                     const int32_t invalid_slot_val,
                                     const JoinColumn join_column,
                                     const JoinColumnTypeInfo type_info,
                                     const int32_t* sd_inner_to_outer_translation_map,
                                     const int32_t min_inner_elem,
                                     const int32_t cpu_thread_idx,
                                     const int32_t cpu_thread_count,
                                     HASHTABLE_FILLING_FUNC filling_func) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  JoinColumnTyped col{&join_column, &type_info};
  for (auto item : col.slice(start, step)) {
    const size_t index = item.index;
    int64_t elem = item.element;
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_to_outer_translation_map &&
        (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
      const auto outer_id = map_str_id_to_outer_dict(elem,
                                                     min_inner_elem,
                                                     type_info.min_val,
                                                     type_info.max_val,
                                                     sd_inner_to_outer_translation_map);
      if (outer_id == StringDictionary::INVALID_STR_ID) {
        continue;
      }
      elem = outer_id;
    }
#endif
    if (filling_func(elem, index)) {
      return -1;
    }
  }
  return 0;
};

DEVICE int SUFFIX(fill_hash_join_buff_bucketized)(
    int32_t* buff,
    const int32_t invalid_slot_val,
    const bool for_semi_join,
    const JoinColumn join_column,
    const JoinColumnTypeInfo type_info,
    const int32_t* sd_inner_to_outer_translation_map,
    const int32_t min_inner_elem,
    const int32_t cpu_thread_idx,
    const int32_t cpu_thread_count,
    const int64_t bucket_normalization) {
  auto filling_func = for_semi_join ? SUFFIX(fill_hashtable_for_semi_join)
                                    : SUFFIX(fill_one_to_one_hashtable);
  auto hashtable_filling_func = [&](auto elem, size_t index) {
    auto entry_ptr = SUFFIX(get_bucketized_hash_slot)(
        buff, elem, type_info.min_val, bucket_normalization);
    return filling_func(index, entry_ptr, invalid_slot_val);
  };

  return fill_hash_join_buff_impl(buff,
                                  invalid_slot_val,
                                  join_column,
                                  type_info,
                                  sd_inner_to_outer_translation_map,
                                  min_inner_elem,
                                  cpu_thread_idx,
                                  cpu_thread_count,
                                  hashtable_filling_func);
}

DEVICE int SUFFIX(fill_hash_join_buff)(int32_t* buff,
                                       const int32_t invalid_slot_val,
                                       const bool for_semi_join,
                                       const JoinColumn join_column,
                                       const JoinColumnTypeInfo type_info,
                                       const int32_t* sd_inner_to_outer_translation_map,
                                       const int32_t min_inner_elem,
                                       const int32_t cpu_thread_idx,
                                       const int32_t cpu_thread_count) {
  auto filling_func = for_semi_join ? SUFFIX(fill_hashtable_for_semi_join)
                                    : SUFFIX(fill_one_to_one_hashtable);
  auto hashtable_filling_func = [&](auto elem, size_t index) {
    auto entry_ptr = SUFFIX(get_hash_slot)(buff, elem, type_info.min_val);
    return filling_func(index, entry_ptr, invalid_slot_val);
  };

  return fill_hash_join_buff_impl(buff,
                                  invalid_slot_val,
                                  join_column,
                                  type_info,
                                  sd_inner_to_outer_translation_map,
                                  min_inner_elem,
                                  cpu_thread_idx,
                                  cpu_thread_count,
                                  hashtable_filling_func);
}

template <typename T>
DEVICE void SUFFIX(init_baseline_hash_join_buff)(int8_t* hash_buff,
                                                 const int64_t entry_count,
                                                 const size_t key_component_count,
                                                 const bool with_val_slot,
                                                 const int32_t invalid_slot_val,
                                                 const int32_t cpu_thread_idx,
                                                 const int32_t cpu_thread_count) {
#ifdef __CUDACC__
  const int64_t start = threadIdx.x + blockDim.x * blockIdx.x;
  const int64_t end = entry_count;
  const int64_t step = blockDim.x * gridDim.x;
#else
  const int64_t start = cpu_thread_idx * entry_count / cpu_thread_count;
  const int64_t end = (cpu_thread_idx + 1) * entry_count / cpu_thread_count;
  const int32_t step = 1;
#endif
  auto hash_entry_size = (key_component_count + (with_val_slot ? 1 : 0)) * sizeof(T);
  const T empty_key = SUFFIX(get_invalid_key)<T>();
  for (int64_t h = start; h < end; h += step) {
    int64_t off = h * hash_entry_size;
    auto row_ptr = reinterpret_cast<T*>(hash_buff + off);
    for (size_t i = 0; i < key_component_count; ++i) {
      row_ptr[i] = empty_key;
    }
    if (with_val_slot) {
      row_ptr[key_component_count] = invalid_slot_val;
    }
  }
}

#ifndef __CUDACC__
#ifdef HAVE_TBB

template <typename T>
DEVICE void SUFFIX(init_baseline_hash_join_buff_tbb)(int8_t* hash_buff,
                                                     const int64_t entry_count,
                                                     const size_t key_component_count,
                                                     const bool with_val_slot,
                                                     const int32_t invalid_slot_val) {
  const auto hash_entry_size =
      (key_component_count + (with_val_slot ? 1 : 0)) * sizeof(T);
  const T empty_key = SUFFIX(get_invalid_key)<T>();
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, entry_count),
                    [=](const tbb::blocked_range<int64_t>& r) {
                      const auto start_idx = r.begin();
                      const auto end_idx = r.end();
                      for (int64_t entry_idx = start_idx; entry_idx < end_idx;
                           ++entry_idx) {
                        const int64_t offset = entry_idx * hash_entry_size;
                        auto row_ptr = reinterpret_cast<T*>(hash_buff + offset);
                        for (size_t k = 0; k < key_component_count; ++k) {
                          row_ptr[k] = empty_key;
                        }
                        if (with_val_slot) {
                          row_ptr[key_component_count] = invalid_slot_val;
                        }
                      }
                    });
}

#endif  // #ifdef HAVE_TBB
#endif  // #ifndef __CUDACC__

#ifdef __CUDACC__
template <typename T>
__device__ T* get_matching_baseline_hash_slot_at(int8_t* hash_buff,
                                                 const uint32_t h,
                                                 const T* key,
                                                 const size_t key_component_count,
                                                 const int64_t hash_entry_size) {
  uint32_t off = h * hash_entry_size;
  auto row_ptr = reinterpret_cast<T*>(hash_buff + off);
  const T empty_key = SUFFIX(get_invalid_key)<T>();
  {
    const T old = atomicCAS(row_ptr, empty_key, *key);
    if (empty_key == old && key_component_count > 1) {
      for (int64_t i = 1; i <= key_component_count - 1; ++i) {
        atomicExch(row_ptr + i, key[i]);
      }
    }
  }
  if (key_component_count > 1) {
    while (atomicAdd(row_ptr + key_component_count - 1, 0) == empty_key) {
      // spin until the winning thread has finished writing the entire key and the init
      // value
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

#ifdef _MSC_VER
#define cas_cst(ptr, expected, desired)                                      \
  (InterlockedCompareExchangePointer(reinterpret_cast<void* volatile*>(ptr), \
                                     reinterpret_cast<void*>(&desired),      \
                                     expected) == expected)
#define store_cst(ptr, val)                                          \
  InterlockedExchangePointer(reinterpret_cast<void* volatile*>(ptr), \
                             reinterpret_cast<void*>(val))
#define load_cst(ptr) \
  InterlockedCompareExchange(reinterpret_cast<volatile long*>(ptr), 0, 0)
#else
#define cas_cst(ptr, expected, desired) \
  __atomic_compare_exchange_n(          \
      ptr, expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define store_cst(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST)
#define load_cst(ptr) __atomic_load_n(ptr, __ATOMIC_SEQ_CST)
#endif

template <typename T>
T* get_matching_baseline_hash_slot_at(int8_t* hash_buff,
                                      const uint32_t h,
                                      const T* key,
                                      const size_t key_component_count,
                                      const int64_t hash_entry_size) {
  uint32_t off = h * hash_entry_size;
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
                                    const int64_t entry_count,
                                    const T* key,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const int32_t invalid_slot_val,
                                    const size_t key_size_in_bytes,
                                    const size_t hash_entry_size) {
  const uint32_t h = MurmurHash1Impl(key, key_size_in_bytes, 0) % entry_count;
  T* matching_group = get_matching_baseline_hash_slot_at(
      hash_buff, h, key, key_component_count, hash_entry_size);
  if (!matching_group) {
    uint32_t h_probe = (h + 1) % entry_count;
    while (h_probe != h) {
      matching_group = get_matching_baseline_hash_slot_at(
          hash_buff, h_probe, key, key_component_count, hash_entry_size);
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
DEVICE int write_baseline_hash_slot_for_semi_join(const int32_t val,
                                                  int8_t* hash_buff,
                                                  const int64_t entry_count,
                                                  const T* key,
                                                  const size_t key_component_count,
                                                  const bool with_val_slot,
                                                  const int32_t invalid_slot_val,
                                                  const size_t key_size_in_bytes,
                                                  const size_t hash_entry_size) {
  const uint32_t h = MurmurHash1Impl(key, key_size_in_bytes, 0) % entry_count;
  T* matching_group = get_matching_baseline_hash_slot_at(
      hash_buff, h, key, key_component_count, hash_entry_size);
  if (!matching_group) {
    uint32_t h_probe = (h + 1) % entry_count;
    while (h_probe != h) {
      matching_group = get_matching_baseline_hash_slot_at(
          hash_buff, h_probe, key, key_component_count, hash_entry_size);
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
  mapd_cas(matching_group, invalid_slot_val, val);
  return 0;
}

template <typename T, typename FILL_HANDLER>
DEVICE int SUFFIX(fill_baseline_hash_join_buff)(int8_t* hash_buff,
                                                const int64_t entry_count,
                                                const int32_t invalid_slot_val,
                                                const bool for_semi_join,
                                                const size_t key_component_count,
                                                const bool with_val_slot,
                                                const FILL_HANDLER* f,
                                                const int64_t num_elems,
                                                const int32_t cpu_thread_idx,
                                                const int32_t cpu_thread_count) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif

  T key_scratch_buff[g_maximum_conditions_to_coalesce];
  const size_t key_size_in_bytes = key_component_count * sizeof(T);
  const size_t hash_entry_size =
      (key_component_count + (with_val_slot ? 1 : 0)) * sizeof(T);
  auto key_buff_handler = [hash_buff,
                           entry_count,
                           with_val_slot,
                           invalid_slot_val,
                           key_size_in_bytes,
                           hash_entry_size,
                           &for_semi_join](const int64_t entry_idx,
                                           const T* key_scratch_buffer,
                                           const size_t key_component_count) {
    if (for_semi_join) {
      return write_baseline_hash_slot_for_semi_join<T>(entry_idx,
                                                       hash_buff,
                                                       entry_count,
                                                       key_scratch_buffer,
                                                       key_component_count,
                                                       with_val_slot,
                                                       invalid_slot_val,
                                                       key_size_in_bytes,
                                                       hash_entry_size);
    } else {
      return write_baseline_hash_slot<T>(entry_idx,
                                         hash_buff,
                                         entry_count,
                                         key_scratch_buffer,
                                         key_component_count,
                                         with_val_slot,
                                         invalid_slot_val,
                                         key_size_in_bytes,
                                         hash_entry_size);
    }
  };

  JoinColumnTuple cols(
      f->get_number_of_columns(), f->get_join_columns(), f->get_join_column_type_infos());
  for (auto& it : cols.slice(start, step)) {
    const auto err = (*f)(it.join_column_iterators, key_scratch_buff, key_buff_handler);
    if (err) {
      return err;
    }
  }
  return 0;
}

#undef mapd_cas

#ifdef __CUDACC__
#define mapd_add(address, val) atomicAdd(address, val)
#elif defined(_MSC_VER)
#define mapd_add(address, val)                                      \
  InterlockedExchangeAdd(reinterpret_cast<volatile long*>(address), \
                         static_cast<long>(val))
#else
#define mapd_add(address, val) __sync_fetch_and_add(address, val)
#endif

template <typename SLOT_SELECTOR>
DEVICE void count_matches_impl(int32_t* count_buff,
                               const int32_t invalid_slot_val,
                               const JoinColumn join_column,
                               const JoinColumnTypeInfo type_info
#ifndef __CUDACC__
                               ,
                               const int32_t* sd_inner_to_outer_translation_map,
                               const int32_t min_inner_elem,
                               const int32_t cpu_thread_idx,
                               const int32_t cpu_thread_count
#endif
                               ,
                               SLOT_SELECTOR slot_selector) {
#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t step = blockDim.x * gridDim.x;
#else
  int32_t start = cpu_thread_idx;
  int32_t step = cpu_thread_count;
#endif
  JoinColumnTyped col{&join_column, &type_info};
  for (auto item : col.slice(start, step)) {
    int64_t elem = item.element;
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_to_outer_translation_map &&
        (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
      const auto outer_id = map_str_id_to_outer_dict(elem,
                                                     min_inner_elem,
                                                     type_info.min_val,
                                                     type_info.max_val,
                                                     sd_inner_to_outer_translation_map);
      if (outer_id == StringDictionary::INVALID_STR_ID) {
        continue;
      }
      elem = outer_id;
    }
#endif
    auto* entry_ptr = slot_selector(count_buff, elem);
    mapd_add(entry_ptr, int32_t(1));
  }
}

GLOBAL void SUFFIX(count_matches)(int32_t* count_buff,
                                  const int32_t invalid_slot_val,
                                  const JoinColumn join_column,
                                  const JoinColumnTypeInfo type_info
#ifndef __CUDACC__
                                  ,
                                  const int32_t* sd_inner_to_outer_translation_map,
                                  const int32_t min_inner_elem,
                                  const int32_t cpu_thread_idx,
                                  const int32_t cpu_thread_count
#endif
) {
  auto slot_sel = [&type_info](auto count_buff, auto elem) {
    return SUFFIX(get_hash_slot)(count_buff, elem, type_info.min_val);
  };
  count_matches_impl(count_buff,
                     invalid_slot_val,
                     join_column,
                     type_info
#ifndef __CUDACC__
                     ,
                     sd_inner_to_outer_translation_map,
                     min_inner_elem,
                     cpu_thread_idx,
                     cpu_thread_count
#endif
                     ,
                     slot_sel);
}

GLOBAL void SUFFIX(count_matches_bucketized)(
    int32_t* count_buff,
    const int32_t invalid_slot_val,
    const JoinColumn join_column,
    const JoinColumnTypeInfo type_info
#ifndef __CUDACC__
    ,
    const int32_t* sd_inner_to_outer_translation_map,
    const int32_t min_inner_elem,
    const int32_t cpu_thread_idx,
    const int32_t cpu_thread_count
#endif
    ,
    const int64_t bucket_normalization) {
  auto slot_sel = [bucket_normalization, &type_info](auto count_buff, auto elem) {
    return SUFFIX(get_bucketized_hash_slot)(
        count_buff, elem, type_info.min_val, bucket_normalization);
  };
  count_matches_impl(count_buff,
                     invalid_slot_val,
                     join_column,
                     type_info
#ifndef __CUDACC__
                     ,
                     sd_inner_to_outer_translation_map,
                     min_inner_elem,
                     cpu_thread_idx,
                     cpu_thread_count
#endif
                     ,
                     slot_sel);
}

template <typename T>
DEVICE NEVER_INLINE const T* SUFFIX(get_matching_baseline_hash_slot_readonly)(
    const T* key,
    const size_t key_component_count,
    const T* composite_key_dict,
    const int64_t entry_count,
    const size_t key_size_in_bytes) {
  const uint32_t h = MurmurHash1Impl(key, key_size_in_bytes, 0) % entry_count;
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

template <typename T, typename KEY_HANDLER>
GLOBAL void SUFFIX(count_matches_baseline)(int32_t* count_buff,
                                           const T* composite_key_dict,
                                           const int64_t entry_count,
                                           const KEY_HANDLER* f,
                                           const int64_t num_elems
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
#ifdef __CUDACC__
  assert(composite_key_dict);
#endif
  T key_scratch_buff[g_maximum_conditions_to_coalesce];
  const size_t key_size_in_bytes = f->get_key_component_count() * sizeof(T);
  auto key_buff_handler = [composite_key_dict,
                           entry_count,
                           count_buff,
                           key_size_in_bytes](const int64_t row_entry_idx,
                                              const T* key_scratch_buff,
                                              const size_t key_component_count) {
    const auto matching_group =
        SUFFIX(get_matching_baseline_hash_slot_readonly)(key_scratch_buff,
                                                         key_component_count,
                                                         composite_key_dict,
                                                         entry_count,
                                                         key_size_in_bytes);
    const auto entry_idx = (matching_group - composite_key_dict) / key_component_count;
    mapd_add(&count_buff[entry_idx], int32_t(1));
    return 0;
  };

  JoinColumnTuple cols(
      f->get_number_of_columns(), f->get_join_columns(), f->get_join_column_type_infos());
  for (auto& it : cols.slice(start, step)) {
    (*f)(it.join_column_iterators, key_scratch_buff, key_buff_handler);
  }
}

template <typename SLOT_SELECTOR>
DEVICE void fill_row_ids_impl(int32_t* buff,
                              const int64_t hash_entry_count,
                              const int32_t invalid_slot_val,
                              const JoinColumn join_column,
                              const JoinColumnTypeInfo type_info
#ifndef __CUDACC__
                              ,
                              const int32_t* sd_inner_to_outer_translation_map,
                              const int32_t min_inner_elem,
                              const int32_t cpu_thread_idx,
                              const int32_t cpu_thread_count
#endif
                              ,
                              SLOT_SELECTOR slot_selector) {
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
  JoinColumnTyped col{&join_column, &type_info};
  for (auto item : col.slice(start, step)) {
    const size_t index = item.index;
    int64_t elem = item.element;
    if (elem == type_info.null_val) {
      if (type_info.uses_bw_eq) {
        elem = type_info.translated_null_val;
      } else {
        continue;
      }
    }
#ifndef __CUDACC__
    if (sd_inner_to_outer_translation_map &&
        (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
      const auto outer_id = map_str_id_to_outer_dict(elem,
                                                     min_inner_elem,
                                                     type_info.min_val,
                                                     type_info.max_val,
                                                     sd_inner_to_outer_translation_map);
      if (outer_id == StringDictionary::INVALID_STR_ID) {
        continue;
      }
      elem = outer_id;
    }
#endif
    auto pos_ptr = slot_selector(pos_buff, elem);
    const auto bin_idx = pos_ptr - pos_buff;
    const auto id_buff_idx = mapd_add(count_buff + bin_idx, 1) + *pos_ptr;
    id_buff[id_buff_idx] = static_cast<int32_t>(index);
  }
}

GLOBAL void SUFFIX(fill_row_ids)(int32_t* buff,
                                 const int64_t hash_entry_count,
                                 const int32_t invalid_slot_val,
                                 const JoinColumn join_column,
                                 const JoinColumnTypeInfo type_info
#ifndef __CUDACC__
                                 ,
                                 const int32_t* sd_inner_to_outer_translation_map,
                                 const int32_t min_inner_elem,
                                 const int32_t cpu_thread_idx,
                                 const int32_t cpu_thread_count
#endif
) {
  auto slot_sel = [&type_info](auto pos_buff, auto elem) {
    return SUFFIX(get_hash_slot)(pos_buff, elem, type_info.min_val);
  };

  fill_row_ids_impl(buff,
                    hash_entry_count,
                    invalid_slot_val,
                    join_column,
                    type_info
#ifndef __CUDACC__
                    ,
                    sd_inner_to_outer_translation_map,
                    min_inner_elem,
                    cpu_thread_idx,
                    cpu_thread_count
#endif
                    ,
                    slot_sel);
}

GLOBAL void SUFFIX(fill_row_ids_bucketized)(
    int32_t* buff,
    const int64_t hash_entry_count,
    const int32_t invalid_slot_val,
    const JoinColumn join_column,
    const JoinColumnTypeInfo type_info
#ifndef __CUDACC__
    ,
    const int32_t* sd_inner_to_outer_translation_map,
    const int32_t min_inner_elem,
    const int32_t cpu_thread_idx,
    const int32_t cpu_thread_count
#endif
    ,
    const int64_t bucket_normalization) {
  auto slot_sel = [&type_info, bucket_normalization](auto pos_buff, auto elem) {
    return SUFFIX(get_bucketized_hash_slot)(
        pos_buff, elem, type_info.min_val, bucket_normalization);
  };
  fill_row_ids_impl(buff,
                    hash_entry_count,
                    invalid_slot_val,
                    join_column,
                    type_info
#ifndef __CUDACC__
                    ,
                    sd_inner_to_outer_translation_map,
                    min_inner_elem,
                    cpu_thread_idx,
                    cpu_thread_count
#endif
                    ,
                    slot_sel);
}

template <typename T, typename KEY_HANDLER>
GLOBAL void SUFFIX(fill_row_ids_baseline)(int32_t* buff,
                                          const T* composite_key_dict,
                                          const int64_t hash_entry_count,
                                          const int32_t invalid_slot_val,
                                          const KEY_HANDLER* f,
                                          const int64_t num_elems
#ifndef __CUDACC__
                                          ,
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

  T key_scratch_buff[g_maximum_conditions_to_coalesce];
#ifdef __CUDACC__
  assert(composite_key_dict);
#endif
  const size_t key_size_in_bytes = f->get_key_component_count() * sizeof(T);
  auto key_buff_handler = [composite_key_dict,
                           hash_entry_count,
                           pos_buff,
                           invalid_slot_val,
                           count_buff,
                           id_buff,
                           key_size_in_bytes](const int64_t row_index,
                                              const T* key_scratch_buff,
                                              const size_t key_component_count) {
    const T* matching_group =
        SUFFIX(get_matching_baseline_hash_slot_readonly)(key_scratch_buff,
                                                         key_component_count,
                                                         composite_key_dict,
                                                         hash_entry_count,
                                                         key_size_in_bytes);
    const auto entry_idx = (matching_group - composite_key_dict) / key_component_count;
    int32_t* pos_ptr = pos_buff + entry_idx;
    const auto bin_idx = pos_ptr - pos_buff;
    const auto id_buff_idx = mapd_add(count_buff + bin_idx, 1) + *pos_ptr;
    id_buff[id_buff_idx] = static_cast<int32_t>(row_index);
    return 0;
  };

  JoinColumnTuple cols(
      f->get_number_of_columns(), f->get_join_columns(), f->get_join_column_type_infos());
  for (auto& it : cols.slice(start, step)) {
    (*f)(it.join_column_iterators, key_scratch_buff, key_buff_handler);
  }
  return;
}

#undef mapd_add

template <typename KEY_HANDLER>
GLOBAL void SUFFIX(approximate_distinct_tuples_impl)(uint8_t* hll_buffer,
                                                     int32_t* row_count_buffer,
                                                     const uint32_t b,
                                                     const int64_t num_elems,
                                                     const KEY_HANDLER* f
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

  auto key_buff_handler = [b, hll_buffer, row_count_buffer](
                              const int64_t entry_idx,
                              const int64_t* key_scratch_buff,
                              const size_t key_component_count) {
    if (row_count_buffer) {
      row_count_buffer[entry_idx] += 1;
    }

    const uint64_t hash =
        MurmurHash64AImpl(key_scratch_buff, key_component_count * sizeof(int64_t), 0);
    const uint32_t index = hash >> (64 - b);
    const auto rank = get_rank(hash << b, 64 - b);
#ifdef __CUDACC__
    atomicMax(reinterpret_cast<int32_t*>(hll_buffer) + index, rank);
#else
    hll_buffer[index] = std::max(hll_buffer[index], rank);
#endif

    return 0;
  };

  int64_t key_scratch_buff[g_maximum_conditions_to_coalesce];

  JoinColumnTuple cols(
      f->get_number_of_columns(), f->get_join_columns(), f->get_join_column_type_infos());
  for (auto& it : cols.slice(start, step)) {
    (*f)(it.join_column_iterators, key_scratch_buff, key_buff_handler);
  }
}

#ifdef __CUDACC__
namespace {
// TODO(adb): put these in a header file so they are not duplicated between here and
// cuda_mapd_rt.cu
__device__ double atomicMin(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(min(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}
}  // namespace
#endif

template <size_t N>
GLOBAL void SUFFIX(compute_bucket_sizes_impl)(double* bucket_sizes_for_thread,
                                              const JoinColumn* join_column,
                                              const JoinColumnTypeInfo* type_info,
                                              const double* bucket_size_thresholds
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
  JoinColumnIterator it(join_column, type_info, start, step);
  for (; it; ++it) {
    // We expect the bounds column to be (min, max) e.g. (x_min, y_min, x_max, y_max)
    double bounds[2 * N];
    for (size_t j = 0; j < 2 * N; j++) {
      bounds[j] = SUFFIX(fixed_width_double_decode_noinline)(it.ptr(), j);
    }

    for (size_t j = 0; j < N; j++) {
      const auto diff = bounds[j + N] - bounds[j];
#ifdef __CUDACC__
      if (diff > bucket_size_thresholds[j]) {
        atomicMin(&bucket_sizes_for_thread[j], diff);
      }
#else
      if (diff < bucket_size_thresholds[j] && diff > bucket_sizes_for_thread[j]) {
        bucket_sizes_for_thread[j] = diff;
      }
#endif
    }
  }
}

#ifndef __CUDACC__

template <typename InputIterator, typename OutputIterator>
void inclusive_scan(InputIterator first,
                    InputIterator last,
                    OutputIterator out,
                    const size_t thread_count) {
  using ElementType = typename InputIterator::value_type;
  using OffsetType = typename InputIterator::difference_type;
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
        [first, out](
            ElementType& partial_sum, const OffsetType start, const OffsetType end) {
          ElementType sum = 0;
          for (auto in_iter = first + start, out_iter = out + start;
               in_iter != (first + end);
               ++in_iter, ++out_iter) {
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
    counter_threads.push_back(std::async(
        std::launch::async,
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

template <typename COUNT_MATCHES_LAUNCH_FUNCTOR, typename FILL_ROW_IDS_LAUNCH_FUNCTOR>
void fill_one_to_many_hash_table_impl(int32_t* buff,
                                      const int64_t hash_entry_count,
                                      const int32_t invalid_slot_val,
                                      const JoinColumn& join_column,
                                      const JoinColumnTypeInfo& type_info,
                                      const int32_t* sd_inner_to_outer_translation_map,
                                      const int32_t min_inner_elem,
                                      const unsigned cpu_thread_count,
                                      COUNT_MATCHES_LAUNCH_FUNCTOR count_matches_func,
                                      FILL_ROW_IDS_LAUNCH_FUNCTOR fill_row_ids_func) {
  auto timer = DEBUG_TIMER(__func__);
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  memset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  std::vector<std::future<void>> counter_threads;
  for (unsigned cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    counter_threads.push_back(std::async(
        std::launch::async, count_matches_func, cpu_thread_idx, cpu_thread_count));
  }

  for (auto& child : counter_threads) {
    child.get();
  }

  std::vector<int32_t> count_copy(hash_entry_count, 0);
  CHECK_GT(hash_entry_count, int64_t(0));
  memcpy(count_copy.data() + 1, count_buff, (hash_entry_count - 1) * sizeof(int32_t));
#if HAVE_CUDA
  thrust::inclusive_scan(count_copy.begin(), count_copy.end(), count_copy.begin());
#else
  ::inclusive_scan(
      count_copy.begin(), count_copy.end(), count_copy.begin(), cpu_thread_count);
#endif
  std::vector<std::future<void>> pos_threads;
  for (size_t cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    pos_threads.push_back(std::async(
        std::launch::async,
        [&](size_t thread_idx) {
          for (int64_t i = thread_idx; i < hash_entry_count; i += cpu_thread_count) {
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
  for (size_t cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    rowid_threads.push_back(std::async(
        std::launch::async, fill_row_ids_func, cpu_thread_idx, cpu_thread_count));
  }

  for (auto& child : rowid_threads) {
    child.get();
  }
}

void fill_one_to_many_hash_table(int32_t* buff,
                                 const HashEntryInfo hash_entry_info,
                                 const int32_t invalid_slot_val,
                                 const JoinColumn& join_column,
                                 const JoinColumnTypeInfo& type_info,
                                 const int32_t* sd_inner_to_outer_translation_map,
                                 const int32_t min_inner_elem,
                                 const unsigned cpu_thread_count) {
  auto timer = DEBUG_TIMER(__func__);
  auto launch_count_matches = [count_buff = buff + hash_entry_info.hash_entry_count,
                               invalid_slot_val,
                               &join_column,
                               &type_info,
                               sd_inner_to_outer_translation_map,
                               min_inner_elem](auto cpu_thread_idx,
                                               auto cpu_thread_count) {
    SUFFIX(count_matches)
    (count_buff,
     invalid_slot_val,
     join_column,
     type_info,
     sd_inner_to_outer_translation_map,
     min_inner_elem,
     cpu_thread_idx,
     cpu_thread_count);
  };
  auto launch_fill_row_ids = [hash_entry_count = hash_entry_info.hash_entry_count,
                              buff,
                              invalid_slot_val,
                              &join_column,
                              &type_info,
                              sd_inner_to_outer_translation_map,
                              min_inner_elem](auto cpu_thread_idx,
                                              auto cpu_thread_count) {
    SUFFIX(fill_row_ids)
    (buff,
     hash_entry_count,
     invalid_slot_val,
     join_column,
     type_info,
     sd_inner_to_outer_translation_map,
     min_inner_elem,
     cpu_thread_idx,
     cpu_thread_count);
  };

  fill_one_to_many_hash_table_impl(buff,
                                   hash_entry_info.hash_entry_count,
                                   invalid_slot_val,
                                   join_column,
                                   type_info,
                                   sd_inner_to_outer_translation_map,
                                   min_inner_elem,
                                   cpu_thread_count,
                                   launch_count_matches,
                                   launch_fill_row_ids);
}

void fill_one_to_many_hash_table_bucketized(
    int32_t* buff,
    const HashEntryInfo hash_entry_info,
    const int32_t invalid_slot_val,
    const JoinColumn& join_column,
    const JoinColumnTypeInfo& type_info,
    const int32_t* sd_inner_to_outer_translation_map,
    const int32_t min_inner_elem,
    const unsigned cpu_thread_count) {
  auto timer = DEBUG_TIMER(__func__);
  auto bucket_normalization = hash_entry_info.bucket_normalization;
  auto hash_entry_count = hash_entry_info.getNormalizedHashEntryCount();
  auto launch_count_matches = [bucket_normalization,
                               count_buff = buff + hash_entry_count,
                               invalid_slot_val,
                               &join_column,
                               &type_info,
                               sd_inner_to_outer_translation_map,
                               min_inner_elem](auto cpu_thread_idx,
                                               auto cpu_thread_count) {
    SUFFIX(count_matches_bucketized)
    (count_buff,
     invalid_slot_val,
     join_column,
     type_info,
     sd_inner_to_outer_translation_map,
     min_inner_elem,
     cpu_thread_idx,
     cpu_thread_count,
     bucket_normalization);
  };
  auto launch_fill_row_ids = [bucket_normalization,
                              hash_entry_count,
                              buff,
                              invalid_slot_val,
                              &join_column,
                              &type_info,
                              sd_inner_to_outer_translation_map,
                              min_inner_elem](auto cpu_thread_idx,
                                              auto cpu_thread_count) {
    SUFFIX(fill_row_ids_bucketized)
    (buff,
     hash_entry_count,
     invalid_slot_val,
     join_column,
     type_info,
     sd_inner_to_outer_translation_map,
     min_inner_elem,
     cpu_thread_idx,
     cpu_thread_count,
     bucket_normalization);
  };

  fill_one_to_many_hash_table_impl(buff,
                                   hash_entry_count,
                                   invalid_slot_val,
                                   join_column,
                                   type_info,
                                   sd_inner_to_outer_translation_map,
                                   min_inner_elem,
                                   cpu_thread_count,
                                   launch_count_matches,
                                   launch_fill_row_ids);
}

void init_baseline_hash_join_buff_32(int8_t* hash_join_buff,
                                     const int64_t entry_count,
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
                                     const int64_t entry_count,
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

#ifndef __CUDACC__
#ifdef HAVE_TBB

void init_baseline_hash_join_buff_tbb_32(int8_t* hash_join_buff,
                                         const int64_t entry_count,
                                         const size_t key_component_count,
                                         const bool with_val_slot,
                                         const int32_t invalid_slot_val) {
  init_baseline_hash_join_buff_tbb<int32_t>(
      hash_join_buff, entry_count, key_component_count, with_val_slot, invalid_slot_val);
}

void init_baseline_hash_join_buff_tbb_64(int8_t* hash_join_buff,
                                         const int64_t entry_count,
                                         const size_t key_component_count,
                                         const bool with_val_slot,
                                         const int32_t invalid_slot_val) {
  init_baseline_hash_join_buff_tbb<int64_t>(
      hash_join_buff, entry_count, key_component_count, with_val_slot, invalid_slot_val);
}

#endif  // #ifdef HAVE_TBB
#endif  // #ifndef __CUDACC__

int fill_baseline_hash_join_buff_32(int8_t* hash_buff,
                                    const int64_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const bool for_semi_join,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const GenericKeyHandler* key_handler,
                                    const int64_t num_elems,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count) {
  return fill_baseline_hash_join_buff<int32_t>(hash_buff,
                                               entry_count,
                                               invalid_slot_val,
                                               for_semi_join,
                                               key_component_count,
                                               with_val_slot,
                                               key_handler,
                                               num_elems,
                                               cpu_thread_idx,
                                               cpu_thread_count);
}

int overlaps_fill_baseline_hash_join_buff_32(int8_t* hash_buff,
                                             const int64_t entry_count,
                                             const int32_t invalid_slot_val,
                                             const size_t key_component_count,
                                             const bool with_val_slot,
                                             const OverlapsKeyHandler* key_handler,
                                             const int64_t num_elems,
                                             const int32_t cpu_thread_idx,
                                             const int32_t cpu_thread_count) {
  return fill_baseline_hash_join_buff<int32_t>(hash_buff,
                                               entry_count,
                                               invalid_slot_val,
                                               false,
                                               key_component_count,
                                               with_val_slot,
                                               key_handler,
                                               num_elems,
                                               cpu_thread_idx,
                                               cpu_thread_count);
}

int range_fill_baseline_hash_join_buff_32(int8_t* hash_buff,
                                          const size_t entry_count,
                                          const int32_t invalid_slot_val,
                                          const size_t key_component_count,
                                          const bool with_val_slot,
                                          const RangeKeyHandler* key_handler,
                                          const size_t num_elems,
                                          const int32_t cpu_thread_idx,
                                          const int32_t cpu_thread_count) {
  return fill_baseline_hash_join_buff<int32_t>(hash_buff,
                                               entry_count,
                                               invalid_slot_val,
                                               false,
                                               key_component_count,
                                               with_val_slot,
                                               key_handler,
                                               num_elems,
                                               cpu_thread_idx,
                                               cpu_thread_count);
}

int fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                    const int64_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const bool for_semi_join,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const GenericKeyHandler* key_handler,
                                    const int64_t num_elems,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count) {
  return fill_baseline_hash_join_buff<int64_t>(hash_buff,
                                               entry_count,
                                               invalid_slot_val,
                                               for_semi_join,
                                               key_component_count,
                                               with_val_slot,
                                               key_handler,
                                               num_elems,
                                               cpu_thread_idx,
                                               cpu_thread_count);
}

int overlaps_fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                             const int64_t entry_count,
                                             const int32_t invalid_slot_val,
                                             const size_t key_component_count,
                                             const bool with_val_slot,
                                             const OverlapsKeyHandler* key_handler,
                                             const int64_t num_elems,
                                             const int32_t cpu_thread_idx,
                                             const int32_t cpu_thread_count) {
  return fill_baseline_hash_join_buff<int64_t>(hash_buff,
                                               entry_count,
                                               invalid_slot_val,
                                               false,
                                               key_component_count,
                                               with_val_slot,
                                               key_handler,
                                               num_elems,
                                               cpu_thread_idx,
                                               cpu_thread_count);
}

int range_fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                          const size_t entry_count,
                                          const int32_t invalid_slot_val,
                                          const size_t key_component_count,
                                          const bool with_val_slot,
                                          const RangeKeyHandler* key_handler,
                                          const size_t num_elems,
                                          const int32_t cpu_thread_idx,
                                          const int32_t cpu_thread_count) {
  return fill_baseline_hash_join_buff<int64_t>(hash_buff,
                                               entry_count,
                                               invalid_slot_val,
                                               false,
                                               key_component_count,
                                               with_val_slot,
                                               key_handler,
                                               num_elems,
                                               cpu_thread_idx,
                                               cpu_thread_count);
}

template <typename T>
void fill_one_to_many_baseline_hash_table(
    int32_t* buff,
    const T* composite_key_dict,
    const int64_t hash_entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_buckets_per_key,
    const std::vector<const void*>& sd_inner_proxy_per_key,
    const std::vector<const void*>& sd_outer_proxy_per_key,
    const size_t cpu_thread_count,
    const bool is_range_join) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  memset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  std::vector<std::future<void>> counter_threads;
  for (size_t cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    if (is_range_join) {
      counter_threads.push_back(std::async(
          std::launch::async,
          [count_buff,
           composite_key_dict,
           &hash_entry_count,
           &join_buckets_per_key,
           &join_column_per_key,
           cpu_thread_idx,
           cpu_thread_count] {
            const auto key_handler = RangeKeyHandler(
                false,
                join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.size(),
                &join_column_per_key[0],
                join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.data());
            count_matches_baseline(count_buff,
                                   composite_key_dict,
                                   hash_entry_count,
                                   &key_handler,
                                   join_column_per_key[0].num_elems,
                                   cpu_thread_idx,
                                   cpu_thread_count);
          }));
    } else if (join_buckets_per_key.size() > 0) {
      counter_threads.push_back(std::async(
          std::launch::async,
          [count_buff,
           composite_key_dict,
           &hash_entry_count,
           &join_buckets_per_key,
           &join_column_per_key,
           cpu_thread_idx,
           cpu_thread_count] {
            const auto key_handler = OverlapsKeyHandler(
                join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.size(),
                &join_column_per_key[0],
                join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.data());
            count_matches_baseline(count_buff,
                                   composite_key_dict,
                                   hash_entry_count,
                                   &key_handler,
                                   join_column_per_key[0].num_elems,
                                   cpu_thread_idx,
                                   cpu_thread_count);
          }));
    } else {
      counter_threads.push_back(std::async(
          std::launch::async,
          [count_buff,
           composite_key_dict,
           &key_component_count,
           &hash_entry_count,
           &join_column_per_key,
           &type_info_per_key,
           &sd_inner_proxy_per_key,
           &sd_outer_proxy_per_key,
           cpu_thread_idx,
           cpu_thread_count] {
            const auto key_handler = GenericKeyHandler(key_component_count,
                                                       true,
                                                       &join_column_per_key[0],
                                                       &type_info_per_key[0],
                                                       &sd_inner_proxy_per_key[0],
                                                       &sd_outer_proxy_per_key[0]);
            count_matches_baseline(count_buff,
                                   composite_key_dict,
                                   hash_entry_count,
                                   &key_handler,
                                   join_column_per_key[0].num_elems,
                                   cpu_thread_idx,
                                   cpu_thread_count);
          }));
    }
  }

  for (auto& child : counter_threads) {
    child.get();
  }

  std::vector<int32_t> count_copy(hash_entry_count, 0);
  CHECK_GT(hash_entry_count, int64_t(0));
  memcpy(&count_copy[1], count_buff, (hash_entry_count - 1) * sizeof(int32_t));
  ::inclusive_scan(
      count_copy.begin(), count_copy.end(), count_copy.begin(), cpu_thread_count);
  std::vector<std::future<void>> pos_threads;
  for (size_t cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    pos_threads.push_back(std::async(
        std::launch::async,
        [&](const int thread_idx) {
          for (int64_t i = thread_idx; i < hash_entry_count; i += cpu_thread_count) {
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
  for (size_t cpu_thread_idx = 0; cpu_thread_idx < cpu_thread_count; ++cpu_thread_idx) {
    if (is_range_join) {
      rowid_threads.push_back(std::async(
          std::launch::async,
          [buff,
           composite_key_dict,
           hash_entry_count,
           invalid_slot_val,
           &join_column_per_key,
           &join_buckets_per_key,
           cpu_thread_idx,
           cpu_thread_count] {
            const auto key_handler = RangeKeyHandler(
                false,
                join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.size(),
                &join_column_per_key[0],
                join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.data());
            SUFFIX(fill_row_ids_baseline)
            (buff,
             composite_key_dict,
             hash_entry_count,
             invalid_slot_val,
             &key_handler,
             join_column_per_key[0].num_elems,
             cpu_thread_idx,
             cpu_thread_count);
          }));
    } else if (join_buckets_per_key.size() > 0) {
      rowid_threads.push_back(std::async(
          std::launch::async,
          [buff,
           composite_key_dict,
           hash_entry_count,
           invalid_slot_val,
           &join_column_per_key,
           &join_buckets_per_key,
           cpu_thread_idx,
           cpu_thread_count] {
            const auto key_handler = OverlapsKeyHandler(
                join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.size(),
                &join_column_per_key[0],
                join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.data());
            SUFFIX(fill_row_ids_baseline)
            (buff,
             composite_key_dict,
             hash_entry_count,
             invalid_slot_val,
             &key_handler,
             join_column_per_key[0].num_elems,
             cpu_thread_idx,
             cpu_thread_count);
          }));
    } else {
      rowid_threads.push_back(std::async(std::launch::async,
                                         [buff,
                                          composite_key_dict,
                                          hash_entry_count,
                                          invalid_slot_val,
                                          key_component_count,
                                          &join_column_per_key,
                                          &type_info_per_key,
                                          &sd_inner_proxy_per_key,
                                          &sd_outer_proxy_per_key,
                                          cpu_thread_idx,
                                          cpu_thread_count] {
                                           const auto key_handler = GenericKeyHandler(
                                               key_component_count,
                                               true,
                                               &join_column_per_key[0],
                                               &type_info_per_key[0],
                                               &sd_inner_proxy_per_key[0],
                                               &sd_outer_proxy_per_key[0]);
                                           SUFFIX(fill_row_ids_baseline)
                                           (buff,
                                            composite_key_dict,
                                            hash_entry_count,
                                            invalid_slot_val,
                                            &key_handler,
                                            join_column_per_key[0].num_elems,
                                            cpu_thread_idx,
                                            cpu_thread_count);
                                         }));
    }
  }

  for (auto& child : rowid_threads) {
    child.get();
  }
}

void fill_one_to_many_baseline_hash_table_32(
    int32_t* buff,
    const int32_t* composite_key_dict,
    const int64_t hash_entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const std::vector<const void*>& sd_inner_proxy_per_key,
    const std::vector<const void*>& sd_outer_proxy_per_key,
    const int32_t cpu_thread_count,
    const bool is_range_join) {
  fill_one_to_many_baseline_hash_table<int32_t>(buff,
                                                composite_key_dict,
                                                hash_entry_count,
                                                invalid_slot_val,
                                                key_component_count,
                                                join_column_per_key,
                                                type_info_per_key,
                                                join_bucket_info,
                                                sd_inner_proxy_per_key,
                                                sd_outer_proxy_per_key,
                                                cpu_thread_count,
                                                is_range_join);
}

void fill_one_to_many_baseline_hash_table_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const int64_t hash_entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const std::vector<const void*>& sd_inner_proxy_per_key,
    const std::vector<const void*>& sd_outer_proxy_per_key,
    const int32_t cpu_thread_count,
    const bool is_range_join) {
  fill_one_to_many_baseline_hash_table<int64_t>(buff,
                                                composite_key_dict,
                                                hash_entry_count,
                                                invalid_slot_val,
                                                key_component_count,
                                                join_column_per_key,
                                                type_info_per_key,
                                                join_bucket_info,
                                                sd_inner_proxy_per_key,
                                                sd_outer_proxy_per_key,
                                                cpu_thread_count,
                                                is_range_join);
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
    approx_distinct_threads.push_back(std::async(
        std::launch::async,
        [&join_column_per_key,
         &type_info_per_key,
         b,
         hll_buffer_all_cpus,
         padded_size_bytes,
         thread_idx,
         thread_count] {
          auto hll_buffer = hll_buffer_all_cpus + thread_idx * padded_size_bytes;

          const auto key_handler = GenericKeyHandler(join_column_per_key.size(),
                                                     false,
                                                     &join_column_per_key[0],
                                                     &type_info_per_key[0],
                                                     nullptr,
                                                     nullptr);
          approximate_distinct_tuples_impl(hll_buffer,
                                           nullptr,
                                           b,
                                           join_column_per_key[0].num_elems,
                                           &key_handler,
                                           thread_idx,
                                           thread_count);
        }));
  }
  for (auto& child : approx_distinct_threads) {
    child.get();
  }
}

void approximate_distinct_tuples_overlaps(
    uint8_t* hll_buffer_all_cpus,
    std::vector<int32_t>& row_counts,
    const uint32_t b,
    const size_t padded_size_bytes,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_buckets_per_key,
    const int thread_count) {
  CHECK_EQ(join_column_per_key.size(), join_buckets_per_key.size());
  CHECK_EQ(join_column_per_key.size(), type_info_per_key.size());
  CHECK(!join_column_per_key.empty());

  std::vector<std::future<void>> approx_distinct_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    approx_distinct_threads.push_back(std::async(
        std::launch::async,
        [&join_column_per_key,
         &join_buckets_per_key,
         &row_counts,
         b,
         hll_buffer_all_cpus,
         padded_size_bytes,
         thread_idx,
         thread_count] {
          auto hll_buffer = hll_buffer_all_cpus + thread_idx * padded_size_bytes;

          const auto key_handler = OverlapsKeyHandler(
              join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.size(),
              &join_column_per_key[0],
              join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.data());
          approximate_distinct_tuples_impl(hll_buffer,
                                           row_counts.data(),
                                           b,
                                           join_column_per_key[0].num_elems,
                                           &key_handler,
                                           thread_idx,
                                           thread_count);
        }));
  }
  for (auto& child : approx_distinct_threads) {
    child.get();
  }

  ::inclusive_scan(
      row_counts.begin(), row_counts.end(), row_counts.begin(), thread_count);
}

void approximate_distinct_tuples_range(
    uint8_t* hll_buffer_all_cpus,
    std::vector<int32_t>& row_counts,
    const uint32_t b,
    const size_t padded_size_bytes,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_buckets_per_key,
    const bool is_compressed,
    const int thread_count) {
  CHECK_EQ(join_column_per_key.size(), join_buckets_per_key.size());
  CHECK_EQ(join_column_per_key.size(), type_info_per_key.size());
  CHECK(!join_column_per_key.empty());

  std::vector<std::future<void>> approx_distinct_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    approx_distinct_threads.push_back(std::async(
        std::launch::async,
        [&join_column_per_key,
         &join_buckets_per_key,
         &row_counts,
         b,
         hll_buffer_all_cpus,
         padded_size_bytes,
         thread_idx,
         is_compressed,
         thread_count] {
          auto hll_buffer = hll_buffer_all_cpus + thread_idx * padded_size_bytes;

          const auto key_handler = RangeKeyHandler(
              is_compressed,
              join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.size(),
              &join_column_per_key[0],
              join_buckets_per_key[0].inverse_bucket_sizes_for_dimension.data());
          approximate_distinct_tuples_impl(hll_buffer,
                                           row_counts.data(),
                                           b,
                                           join_column_per_key[0].num_elems,
                                           &key_handler,
                                           thread_idx,
                                           thread_count);
        }));
  }
  for (auto& child : approx_distinct_threads) {
    child.get();
  }

  ::inclusive_scan(
      row_counts.begin(), row_counts.end(), row_counts.begin(), thread_count);
}

void compute_bucket_sizes_on_cpu(std::vector<double>& bucket_sizes_for_dimension,
                                 const JoinColumn& join_column,
                                 const JoinColumnTypeInfo& type_info,
                                 const std::vector<double>& bucket_size_thresholds,
                                 const int thread_count) {
  std::vector<std::vector<double>> bucket_sizes_for_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    bucket_sizes_for_threads.emplace_back(bucket_sizes_for_dimension.size(), 0.0);
  }
  std::vector<std::future<void>> threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    threads.push_back(std::async(std::launch::async,
                                 compute_bucket_sizes_impl<2>,
                                 bucket_sizes_for_threads[thread_idx].data(),
                                 &join_column,
                                 &type_info,
                                 bucket_size_thresholds.data(),
                                 thread_idx,
                                 thread_count));
  }
  for (auto& child : threads) {
    child.get();
  }

  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    for (size_t i = 0; i < bucket_sizes_for_dimension.size(); i++) {
      if (bucket_sizes_for_threads[thread_idx][i] > bucket_sizes_for_dimension[i]) {
        bucket_sizes_for_dimension[i] = bucket_sizes_for_threads[thread_idx][i];
      }
    }
  }
}

#endif  // ifndef __CUDACC__
