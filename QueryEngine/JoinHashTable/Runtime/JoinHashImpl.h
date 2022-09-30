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

/**
 * @file    JoinHashImpl.h
 * @brief
 *
 */

#ifndef QUERYENGINE_GROUPBYFASTIMPL_H
#define QUERYENGINE_GROUPBYFASTIMPL_H

#include <cstdint>
#include <functional>
#include "../../../Shared/funcannotations.h"
#include "../../../Shared/shard_key.h"

#ifdef __CUDACC__
#define insert_key_cas(address, compare, val) atomicCAS(address, compare, val)
#elif defined(_WIN32)
#include "Shared/clean_windows.h"
#define insert_key_cas(address, compare, val)                           \
  InterlockedCompareExchange(reinterpret_cast<volatile long*>(address), \
                             static_cast<long>(val),                    \
                             static_cast<long>(compare))
#else
#define insert_key_cas(address, compare, val) \
  __sync_val_compare_and_swap(address, compare, val)
#endif

extern "C" ALWAYS_INLINE DEVICE int SUFFIX(fill_one_to_one_hashtable)(
    size_t idx,
    int32_t* entry_ptr,
    const int32_t invalid_slot_val) {
  if (insert_key_cas(entry_ptr, invalid_slot_val, idx) != invalid_slot_val) {
    return -1;
  }
  return 0;
}

extern "C" ALWAYS_INLINE DEVICE int SUFFIX(fill_hashtable_for_semi_join)(
    size_t idx,
    int32_t* entry_ptr,
    const int32_t invalid_slot_val) {
  // just mark the existence of value to the corresponding hash slot
  // regardless of hashtable collision
  insert_key_cas(entry_ptr, invalid_slot_val, idx);
  return 0;
}

#undef insert_key_cas

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_bucketized_hash_slot)(
    int32_t* buff,
    const int64_t key,
    const int64_t min_key,
    const int64_t translated_null_val,
    const int64_t bucket_normalization) {
  auto hash_slot = key / bucket_normalization - min_key + (key == translated_null_val);
  return buff + hash_slot;
}

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_hash_slot)(int32_t* buff,
                                                               const int64_t key,
                                                               const int64_t min_key) {
  return buff + (key - min_key);
}

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_bucketized_hash_slot_sharded)(
    int32_t* buff,
    const int64_t key,
    const int64_t min_key,
    const int64_t translated_null_val,
    const uint32_t entry_count_per_shard,
    const uint32_t num_shards,
    const uint32_t device_count,
    const int64_t bucket_normalization) {
  const uint32_t shard = SHARD_FOR_KEY(key, num_shards);
  const uint32_t shard_buffer_index =
      shard / device_count;  // shard sub-buffer index within `buff`
  int32_t* shard_buffer = buff + shard_buffer_index * entry_count_per_shard;
  auto hash_slot = ((key / bucket_normalization) - min_key) / num_shards +
                   (key == translated_null_val);
  return shard_buffer + hash_slot;
}

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_hash_slot_sharded)(
    int32_t* buff,
    const int64_t key,
    const int64_t min_key,
    const uint32_t entry_count_per_shard,
    const uint32_t num_shards,
    const uint32_t device_count) {
  const uint32_t shard = SHARD_FOR_KEY(key, num_shards);
  const uint32_t shard_buffer_index =
      shard / device_count;  // shard sub-buffer index within `buff`
  int32_t* shard_buffer = buff + shard_buffer_index * entry_count_per_shard;
  return shard_buffer + (key - min_key) / num_shards;
}

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_bucketized_hash_slot_sharded_opt)(
    int32_t* buff,
    const int64_t key,
    const int64_t min_key,
    const int64_t translated_null_val,
    const uint32_t entry_count_per_shard,
    const uint32_t shard,
    const uint32_t num_shards,
    const uint32_t device_count,
    const int64_t bucket_normalization) {
  const uint32_t shard_buffer_index =
      shard / device_count;  // shard sub-buffer index within `buff`
  int32_t* shard_buffer = buff + shard_buffer_index * entry_count_per_shard;
  int64_t hash_slot = ((key / bucket_normalization) - min_key) / num_shards +
                      (key == translated_null_val);
  return shard_buffer + hash_slot;
}

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_hash_slot_sharded_opt)(
    int32_t* buff,
    const int64_t key,
    const int64_t min_key,
    const uint32_t entry_count_per_shard,
    const uint32_t shard,
    const uint32_t num_shards,
    const uint32_t device_count) {
  const uint32_t shard_buffer_index =
      shard / device_count;  // shard sub-buffer index within `buff`
  int32_t* shard_buffer = buff + shard_buffer_index * entry_count_per_shard;
  return shard_buffer + (key - min_key) / num_shards;
}

#endif  // QUERYENGINE_GROUPBYFASTIMPL_H
