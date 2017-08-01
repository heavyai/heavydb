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

#include "MurmurHash.h"

DEVICE bool compare_to_key(const int8_t* entry, const int8_t* key, const size_t key_bytes) {
  for (size_t i = 0; i < key_bytes; ++i) {
    if (entry[i] != key[i]) {
      return false;
    }
  }
  return true;
}

template <class T>
DEVICE int64_t get_matching_slot(const int8_t* hash_buff, const uint32_t h, const int8_t* key, const size_t key_bytes) {
  const auto lookup_result_ptr = hash_buff + h * (key_bytes + sizeof(T));
  if (compare_to_key(lookup_result_ptr, key, key_bytes)) {
    return *reinterpret_cast<const T*>(lookup_result_ptr + key_bytes);
  }
  return -1;
}

template <class T>
FORCE_INLINE DEVICE int64_t baseline_hash_join_idx_impl(const int8_t* hash_buff,
                                                        const int8_t* key,
                                                        const size_t key_bytes,
                                                        const size_t entry_count) {
  const uint32_t h = MurmurHash1(key, key_bytes, 0) % entry_count;
  int64_t matching_slot = get_matching_slot<T>(hash_buff, h, key, key_bytes);
  if (matching_slot != -1) {
    return matching_slot;
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    matching_slot = get_matching_slot<T>(hash_buff, h_probe, key, key_bytes);
    if (matching_slot != -1) {
      return matching_slot;
    }
    h_probe = (h_probe + 1) % entry_count;
  }
  return -1;
}

extern "C" NEVER_INLINE DEVICE int64_t baseline_hash_join_idx_32(const int8_t* hash_buff,
                                                                 const int8_t* key,
                                                                 const size_t key_bytes,
                                                                 const size_t entry_count) {
  return baseline_hash_join_idx_impl<int32_t>(hash_buff, key, key_bytes, entry_count);
}

extern "C" NEVER_INLINE DEVICE int64_t baseline_hash_join_idx_64(const int8_t* hash_buff,
                                                                 const int8_t* key,
                                                                 const size_t key_bytes,
                                                                 const size_t entry_count) {
  return baseline_hash_join_idx_impl<int64_t>(hash_buff, key, key_bytes, entry_count);
}
