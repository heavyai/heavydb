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

inline int64_t count_distinct_set_size(const int64_t set_handle,
                                       const int target_idx,
                                       const CountDistinctDescriptors& count_distinct_descriptors) {
  if (!set_handle) {
    return 0;
  }
  CHECK_LT(target_idx, count_distinct_descriptors.size());
  const auto& count_distinct_desc = count_distinct_descriptors[target_idx];
  if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
    auto set_vals = reinterpret_cast<const int8_t*>(set_handle);
    if (count_distinct_desc.approximate) {
      return count_distinct_desc.device_type == ExecutorDeviceType::GPU
                 ? hll_size(reinterpret_cast<const int32_t*>(set_vals), count_distinct_desc)
                 : hll_size(reinterpret_cast<const int8_t*>(set_vals), count_distinct_desc);
    }
    return bitmap_set_size(set_vals, count_distinct_desc.bitmapSizeBytes());
  }
  CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet);
  return reinterpret_cast<std::set<int64_t>*>(set_handle)->size();
}

inline void bitmap_set_union(int8_t* lhs, int8_t* rhs, const size_t bitmap_sz) {
  for (size_t i = 0; i < bitmap_sz; ++i) {
    lhs[i] = rhs[i] = lhs[i] | rhs[i];
  }
}

inline void count_distinct_set_union(const int64_t new_set_handle,
                                     const int64_t old_set_handle,
                                     const CountDistinctDescriptor& count_distinct_desc) {
  if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
    auto old_set = reinterpret_cast<int8_t*>(old_set_handle);
    auto new_set = reinterpret_cast<int8_t*>(new_set_handle);
    if (count_distinct_desc.approximate) {
      if (count_distinct_desc.device_type == ExecutorDeviceType::GPU) {
        hll_unify(reinterpret_cast<int32_t*>(new_set),
                  reinterpret_cast<int32_t*>(old_set),
                  1 << count_distinct_desc.bitmap_sz_bits);
      } else {
        hll_unify(reinterpret_cast<int8_t*>(new_set),
                  reinterpret_cast<int8_t*>(old_set),
                  1 << count_distinct_desc.bitmap_sz_bits);
      }
    } else {
      bitmap_set_union(new_set, old_set, count_distinct_desc.bitmapSizeBytes());
    }
  } else {
    CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet);
    auto old_set = reinterpret_cast<std::set<int64_t>*>(old_set_handle);
    auto new_set = reinterpret_cast<std::set<int64_t>*>(new_set_handle);
    new_set->insert(old_set->begin(), old_set->end());
    old_set->insert(new_set->begin(), new_set->end());
  }
}

#endif
