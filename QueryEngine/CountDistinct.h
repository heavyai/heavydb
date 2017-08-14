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

/**
 * @file    CountDistinct.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Functions used to work with (approximate) count distinct sets.
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_COUNTDISTINCT_H
#define QUERYENGINE_COUNTDISTINCT_H

#include "CountDistinctDescriptor.h"
#include "HyperLogLog.h"

#include <bitset>
#include <set>
#include <vector>

typedef std::vector<CountDistinctDescriptor> CountDistinctDescriptors;

inline size_t bitmap_set_size(const int8_t* bitmap, const size_t bitmap_byte_sz) {
  const auto bitmap_word_count = bitmap_byte_sz >> 3;
  const auto bitmap_rem_bytes = bitmap_byte_sz & 7;
  const auto bitmap64 = reinterpret_cast<const int64_t*>(bitmap);
  size_t set_size = 0;
  for (size_t i = 0; i < bitmap_word_count; ++i) {
    std::bitset<64> word_bitset(bitmap64[i]);
    set_size += word_bitset.count();
  }
  const auto rem_bitmap = reinterpret_cast<const int8_t*>(&bitmap64[bitmap_word_count]);
  for (size_t i = 0; i < bitmap_rem_bytes; ++i) {
    std::bitset<8> byte_bitset(rem_bitmap[i]);
    set_size += byte_bitset.count();
  }
  return set_size;
}

inline void bitmap_set_union(int8_t* lhs, int8_t* rhs, const size_t bitmap_sz) {
  for (size_t i = 0; i < bitmap_sz; ++i) {
    lhs[i] = rhs[i] = lhs[i] | rhs[i];
  }
}

// Bring all the set bits in the multiple sub-bitmaps into the first sub-bitmap.
inline void partial_bitmap_union(int8_t* set_vals, const CountDistinctDescriptor& count_distinct_desc) {
  auto partial_set_vals = set_vals;
  CHECK_EQ(size_t(0), count_distinct_desc.bitmapPaddedSizeBytes() % count_distinct_desc.sub_bitmap_count);
  const auto partial_padded_size = count_distinct_desc.bitmapPaddedSizeBytes() / count_distinct_desc.sub_bitmap_count;
  for (size_t i = 1; i < count_distinct_desc.sub_bitmap_count; ++i) {
    partial_set_vals += partial_padded_size;
    bitmap_set_union(set_vals, partial_set_vals, count_distinct_desc.bitmapSizeBytes());
  }
}

inline int64_t count_distinct_set_size(const int64_t set_handle,
                                       const int target_idx,
                                       const CountDistinctDescriptors& count_distinct_descriptors) {
  if (!set_handle) {
    return 0;
  }
  CHECK_LT(target_idx, count_distinct_descriptors.size());
  const auto& count_distinct_desc = count_distinct_descriptors[target_idx];
  if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
    auto set_vals = reinterpret_cast<int8_t*>(set_handle);
    if (count_distinct_desc.approximate) {
      CHECK_GT(count_distinct_desc.bitmap_sz_bits, 0);
      return count_distinct_desc.device_type == ExecutorDeviceType::GPU
                 ? hll_size(reinterpret_cast<const int32_t*>(set_vals), count_distinct_desc.bitmap_sz_bits)
                 : hll_size(reinterpret_cast<const int8_t*>(set_vals), count_distinct_desc.bitmap_sz_bits);
    }
    if (count_distinct_desc.sub_bitmap_count > 1) {
      partial_bitmap_union(set_vals, count_distinct_desc);
    }
    return bitmap_set_size(set_vals, count_distinct_desc.bitmapSizeBytes());
  }
  CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet);
  return reinterpret_cast<std::set<int64_t>*>(set_handle)->size();
}

inline void count_distinct_set_union(const int64_t new_set_handle,
                                     const int64_t old_set_handle,
                                     const CountDistinctDescriptor& new_count_distinct_desc,
                                     const CountDistinctDescriptor& old_count_distinct_desc) {
  if (new_count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
    auto new_set = reinterpret_cast<int8_t*>(new_set_handle);
    auto old_set = reinterpret_cast<int8_t*>(old_set_handle);
    if (new_count_distinct_desc.approximate) {
      CHECK(old_count_distinct_desc.approximate);
      if (new_count_distinct_desc.device_type == ExecutorDeviceType::GPU &&
          old_count_distinct_desc.device_type == ExecutorDeviceType::GPU) {
        hll_unify(reinterpret_cast<int32_t*>(new_set),
                  reinterpret_cast<int32_t*>(old_set),
                  1 << old_count_distinct_desc.bitmap_sz_bits);
      } else if (new_count_distinct_desc.device_type == ExecutorDeviceType::GPU &&
                 old_count_distinct_desc.device_type == ExecutorDeviceType::CPU) {
        hll_unify(reinterpret_cast<int32_t*>(new_set),
                  reinterpret_cast<int8_t*>(old_set),
                  1 << old_count_distinct_desc.bitmap_sz_bits);
      } else if (new_count_distinct_desc.device_type == ExecutorDeviceType::CPU &&
                 old_count_distinct_desc.device_type == ExecutorDeviceType::GPU) {
        hll_unify(reinterpret_cast<int8_t*>(new_set),
                  reinterpret_cast<int32_t*>(old_set),
                  1 << old_count_distinct_desc.bitmap_sz_bits);
      } else {
        CHECK(old_count_distinct_desc.device_type == ExecutorDeviceType::CPU &&
              new_count_distinct_desc.device_type == ExecutorDeviceType::CPU);
        hll_unify(reinterpret_cast<int8_t*>(new_set),
                  reinterpret_cast<int8_t*>(old_set),
                  1 << old_count_distinct_desc.bitmap_sz_bits);
      }
    } else {
      CHECK_EQ(new_count_distinct_desc.sub_bitmap_count, old_count_distinct_desc.sub_bitmap_count);
      CHECK_GE(old_count_distinct_desc.sub_bitmap_count, size_t(1));
      // NB: For low cardinality input and if the query ran on GPU the bitmap is
      // composed of multiple padded sub-bitmaps. Treat them as if they are regular
      // bitmaps and let count_distinct_set_size take care of additional reduction.
      const auto bitmap_byte_sz = old_count_distinct_desc.sub_bitmap_count == 1
                                      ? old_count_distinct_desc.bitmapSizeBytes()
                                      : old_count_distinct_desc.bitmapPaddedSizeBytes();
      bitmap_set_union(new_set, old_set, bitmap_byte_sz);
    }
  } else {
    CHECK(old_count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet);
    auto old_set = reinterpret_cast<std::set<int64_t>*>(old_set_handle);
    auto new_set = reinterpret_cast<std::set<int64_t>*>(new_set_handle);
    new_set->insert(old_set->begin(), old_set->end());
    old_set->insert(new_set->begin(), new_set->end());
  }
}

inline void count_distinct_set_union(const int64_t new_set_handle,
                                     const int64_t old_set_handle,
                                     const CountDistinctDescriptor& count_distinct_desc) {
  count_distinct_set_union(new_set_handle, old_set_handle, count_distinct_desc, count_distinct_desc);
}

#endif
