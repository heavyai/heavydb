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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../Shared/geo_compression_runtime.h"
#include "../CompareKeysInl.h"
#include "../MurmurHash.h"

DEVICE bool compare_to_key(const int8_t* entry,
                           const int8_t* key,
                           const size_t key_bytes) {
  for (size_t i = 0; i < key_bytes; ++i) {
    if (entry[i] != key[i]) {
      return false;
    }
  }
  return true;
}

namespace {

const int kNoMatch = -1;
const int kNotPresent = -2;

}  // namespace

template <class T>
DEVICE int64_t get_matching_slot(const int8_t* hash_buff,
                                 const uint32_t h,
                                 const int8_t* key,
                                 const size_t key_bytes) {
  const auto lookup_result_ptr = hash_buff + h * (key_bytes + sizeof(T));
  if (compare_to_key(lookup_result_ptr, key, key_bytes)) {
    return *reinterpret_cast<const T*>(lookup_result_ptr + key_bytes);
  }
  if (*reinterpret_cast<const T*>(lookup_result_ptr) ==
      SUFFIX(get_invalid_key) < T > ()) {
    return kNotPresent;
  }
  return kNoMatch;
}

template <class T>
FORCE_INLINE DEVICE int64_t baseline_hash_join_idx_impl(const int8_t* hash_buff,
                                                        const int8_t* key,
                                                        const size_t key_bytes,
                                                        const size_t entry_count) {
  if (!entry_count) {
    return kNoMatch;
  }
  const uint32_t h = MurmurHash1(key, key_bytes, 0) % entry_count;
  int64_t matching_slot = get_matching_slot<T>(hash_buff, h, key, key_bytes);
  if (matching_slot != kNoMatch) {
    return matching_slot;
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    matching_slot = get_matching_slot<T>(hash_buff, h_probe, key, key_bytes);
    if (matching_slot != kNoMatch) {
      return matching_slot;
    }
    h_probe = (h_probe + 1) % entry_count;
  }
  return kNoMatch;
}

extern "C" NEVER_INLINE DEVICE int64_t
baseline_hash_join_idx_32(const int8_t* hash_buff,
                          const int8_t* key,
                          const size_t key_bytes,
                          const size_t entry_count) {
  return baseline_hash_join_idx_impl<int32_t>(hash_buff, key, key_bytes, entry_count);
}

extern "C" NEVER_INLINE DEVICE int64_t
baseline_hash_join_idx_64(const int8_t* hash_buff,
                          const int8_t* key,
                          const size_t key_bytes,
                          const size_t entry_count) {
  return baseline_hash_join_idx_impl<int64_t>(hash_buff, key, key_bytes, entry_count);
}

template <typename T>
FORCE_INLINE DEVICE int64_t get_bucket_key_for_value_impl(const T value,
                                                          const double bucket_size) {
  return static_cast<int64_t>(floor(static_cast<double>(value) * bucket_size));
}

extern "C" NEVER_INLINE DEVICE int64_t
get_bucket_key_for_range_double(const int8_t* range_bytes,
                                const size_t range_component_index,
                                const double bucket_size) {
  const auto range = reinterpret_cast<const double*>(range_bytes);
  return get_bucket_key_for_value_impl(range[range_component_index], bucket_size);
}

FORCE_INLINE DEVICE int64_t
get_bucket_key_for_range_compressed_impl(const int8_t* range,
                                         const size_t range_component_index,
                                         const double bucket_size) {
  const auto range_ptr = reinterpret_cast<const int32_t*>(range);
  if (range_component_index % 2 == 0) {
    return get_bucket_key_for_value_impl(
        Geo_namespace::decompress_longitude_coord_geoint32(
            range_ptr[range_component_index]),
        bucket_size);
  } else {
    return get_bucket_key_for_value_impl(
        Geo_namespace::decompress_lattitude_coord_geoint32(
            range_ptr[range_component_index]),
        bucket_size);
  }
}

extern "C" NEVER_INLINE DEVICE int64_t
get_bucket_key_for_range_compressed(const int8_t* range,
                                    const size_t range_component_index,
                                    const double bucket_size) {
  return get_bucket_key_for_range_compressed_impl(
      range, range_component_index, bucket_size);
}

template <typename T>
FORCE_INLINE DEVICE int64_t get_composite_key_index_impl(const T* key,
                                                         const size_t key_component_count,
                                                         const T* composite_key_dict,
                                                         const size_t entry_count) {
  const uint32_t h = MurmurHash1(key, key_component_count * sizeof(T), 0) % entry_count;
  uint32_t off = h * key_component_count;
  if (keys_are_equal(&composite_key_dict[off], key, key_component_count)) {
    return h;
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    off = h_probe * key_component_count;
    if (keys_are_equal(&composite_key_dict[off], key, key_component_count)) {
      return h_probe;
    }
    if (composite_key_dict[off] == SUFFIX(get_invalid_key) < T > ()) {
      return -1;
    }
    h_probe = (h_probe + 1) % entry_count;
  }
  return -1;
}

extern "C" NEVER_INLINE DEVICE int64_t
get_composite_key_index_32(const int32_t* key,
                           const size_t key_component_count,
                           const int32_t* composite_key_dict,
                           const size_t entry_count) {
  return get_composite_key_index_impl(
      key, key_component_count, composite_key_dict, entry_count);
}

extern "C" NEVER_INLINE DEVICE int64_t
get_composite_key_index_64(const int64_t* key,
                           const size_t key_component_count,
                           const int64_t* composite_key_dict,
                           const size_t entry_count) {
  return get_composite_key_index_impl(
      key, key_component_count, composite_key_dict, entry_count);
}

extern "C" NEVER_INLINE DEVICE int32_t insert_sorted(int32_t* arr,
                                                     size_t elem_count,
                                                     int32_t elem) {
  for (size_t i = 0; i < elem_count; i++) {
    if (elem == arr[i])
      return 0;

    if (elem > arr[i])
      continue;

    for (size_t j = elem_count; i < j; j--) {
      arr[j] = arr[j - 1];
    }
    arr[i] = elem;
    return 1;
  }

  arr[elem_count] = elem;
  return 1;
}

extern "C" ALWAYS_INLINE DEVICE int64_t overlaps_hash_join_idx(int64_t hash_buff,
                                                               const int64_t key,
                                                               const int64_t min_key,
                                                               const int64_t max_key) {
  if (key >= min_key && key <= max_key) {
    return *(reinterpret_cast<int32_t*>(hash_buff) + (key - min_key));
  }
  return -1;
}

struct BufferRange {
  const int32_t* buffer = nullptr;
  const int64_t element_count = 0;
};

ALWAYS_INLINE DEVICE BufferRange
get_row_id_buffer_ptr(int64_t* hash_table_ptr,
                      const int64_t* key,
                      const int64_t key_component_count,
                      const int64_t entry_count,
                      const int64_t offset_buffer_ptr_offset,
                      const int64_t sub_buff_size) {
  const int64_t min_key = 0;
  const int64_t max_key = entry_count - 1;

  auto key_idx =
      get_composite_key_index_64(key, key_component_count, hash_table_ptr, entry_count);

  if (key_idx < -1) {
    return BufferRange{.buffer = nullptr, .element_count = 0};
  }

  int8_t* one_to_many_ptr = reinterpret_cast<int8_t*>(hash_table_ptr);
  one_to_many_ptr += offset_buffer_ptr_offset;

  // Returns an index used to fetch row count and row ids.
  const auto slot = overlaps_hash_join_idx(
      reinterpret_cast<int64_t>(one_to_many_ptr), key_idx, min_key, max_key);
  if (slot < 0) {
    return BufferRange{.buffer = nullptr, .element_count = 0};
  }

  // Offset into the row count section of buffer
  int8_t* count_ptr = one_to_many_ptr + sub_buff_size;

  const int64_t matched_row_count = overlaps_hash_join_idx(
      reinterpret_cast<int64_t>(count_ptr), key_idx, min_key, max_key);

  // Offset into payload section, containing an array of row ids
  int32_t* rowid_buffer = (int32_t*)(one_to_many_ptr + 2 * sub_buff_size);
  const auto rowidoff_ptr = &rowid_buffer[slot];

  return BufferRange{.buffer = rowidoff_ptr, .element_count = matched_row_count};
}

struct Bounds {
  const double min_X;
  const double min_Y;
  const double max_X;
  const double max_Y;
};

/// Getting overlapping candidates for the overlaps join algorithm
/// works as follows:
///
/// 1. Take the bounds of the Polygon and use the bucket sizes
/// to split the bounding box into the hash keys it falls into.
///
/// 2. Iterate over the keys and look them up in the hash
/// table.
///
/// 3. When looking up the values of each key, we use a
/// series of offsets to get to the array of row ids.
///
/// 4. Since it is possible (likely) we encounter the same
/// row id in several buckets, we need to ensure we remove
/// the duplicates we encounter. A simple ordered insertion
/// is used which ignores duplicate values. Since the N elements
/// we insert can be considered relatively small (N < 200) this
/// exhibits a good tradeoff to conserve space since we are constrained
/// by the stack size on the GPU.
///
/// RETURNS:
/// Unique Row IDs are placed on the fixed size stack array that is passed
/// in as out_arr.
/// The number of row ids in this array is returned.
extern "C" NEVER_INLINE DEVICE int64_t
get_candidate_rows(int32_t* out_arr,
                   const uint32_t max_arr_size,
                   const int8_t* range_bytes,
                   const int32_t range_component_index,
                   const double bucket_size_x,
                   const double bucket_size_y,
                   const int32_t keys_count,
                   const int64_t key_component_count,
                   int64_t* hash_table_ptr,
                   const int64_t entry_count,
                   const int64_t offset_buffer_ptr_offset,
                   const int64_t sub_buff_size) {
  const auto range = reinterpret_cast<const double*>(range_bytes);

  size_t elem_count = 0;

  const auto bounds =
      Bounds{.min_X = range[0], .min_Y = range[1], .max_X = range[2], .max_Y = range[3]};

  for (int64_t x = floor(bounds.min_X * bucket_size_x);
       x <= floor(bounds.max_X * bucket_size_x);
       x++) {
    for (int64_t y = floor(bounds.min_Y * bucket_size_y);
         y <= floor(bounds.max_Y * bucket_size_y);
         y++) {
      int64_t cur_key[2];
      cur_key[0] = static_cast<int64_t>(x);
      cur_key[1] = static_cast<int64_t>(y);

      const auto buffer_range = get_row_id_buffer_ptr(hash_table_ptr,
                                                      cur_key,
                                                      key_component_count,
                                                      entry_count,
                                                      offset_buffer_ptr_offset,
                                                      sub_buff_size);

      for (int64_t j = 0; j < buffer_range.element_count; j++) {
        const auto rowid = buffer_range.buffer[j];
        elem_count += insert_sorted(out_arr, elem_count, rowid);
        assert(max_arr_size >= elem_count);
      }
    }
  }

  return elem_count;
}

// /// Given the bounding box and the bucket size,
// /// return the number of buckets the bounding box
// /// will be split into.
extern "C" NEVER_INLINE DEVICE int32_t
get_num_buckets_for_bounds(const int8_t* range_bytes,
                           const int32_t range_component_index,
                           const double bucket_size_x,
                           const double bucket_size_y) {
  const auto range = reinterpret_cast<const double*>(range_bytes);

  const auto bounds_min_x = range[0];
  const auto bounds_min_y = range[1];
  const auto bounds_max_x = range[2];
  const auto bounds_max_y = range[3];

  const auto num_x =
      floor(bounds_max_x * bucket_size_x) - floor(bounds_min_x * bucket_size_x);
  const auto num_y =
      floor(bounds_max_y * bucket_size_y) - floor(bounds_min_y * bucket_size_y);
  const auto num_buckets = (num_x + 1) * (num_y + 1);

  return num_buckets;
}
