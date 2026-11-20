/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    CountDistinctDescriptor.h
 * @brief   Descriptor for the storage layout use for (approximate) count distinct
 * operations.
 *
 */

#ifndef QUERYENGINE_COUNTDISTINCTDESCRIPTOR_H
#define QUERYENGINE_COUNTDISTINCTDESCRIPTOR_H

#include "../BufferCompaction.h"
#include "../CompilationOptions.h"
#include "Logger/Logger.h"

#include <limits>
#include <stdexcept>

inline size_t bitmap_bits_to_bytes(const size_t bitmap_sz) {
  size_t bitmap_byte_sz = bitmap_sz / 8;
  if (bitmap_sz % 8) {
    ++bitmap_byte_sz;
  }
  return bitmap_byte_sz;
}

enum class CountDistinctImplType { Invalid, Bitmap, UnorderedSet };

struct CountDistinctDescriptor {
  CountDistinctImplType impl_type_;
  int64_t min_val;
  int64_t bucket_size;
  // When used in the approximate count distinct algorithm, bitmap_sz_bits has a
  // different meaning than the bitmap size: https://en.wikipedia.org/wiki/HyperLogLog
  int64_t bitmap_sz_bits;
  bool approximate;
  ExecutorDeviceType device_type;
  size_t sub_bitmap_count;

  size_t bitmapSizeBytes() const {
    if (impl_type_ != CountDistinctImplType::Bitmap || bitmap_sz_bits < 0) {
      throw std::invalid_argument("Invalid count-distinct bitmap descriptor");
    }
    const size_t approx_reg_bytes =
        device_type == ExecutorDeviceType::GPU ? sizeof(int32_t) : 1;
    if (!approximate) {
      return bitmap_bits_to_bytes(static_cast<size_t>(bitmap_sz_bits));
    }
    if (static_cast<uint64_t>(bitmap_sz_bits) >= std::numeric_limits<size_t>::digits ||
        approx_reg_bytes > (std::numeric_limits<size_t>::max() >> bitmap_sz_bits)) {
      throw std::overflow_error("Count-distinct bitmap size overflow");
    }
    return approx_reg_bytes << bitmap_sz_bits;
  }

  size_t bitmapPaddedSizeBytes() const {
    const auto effective_size = bitmapSizeBytes();
    if (sub_bitmap_count == 0) {
      throw std::invalid_argument("Count-distinct bitmap has no sub-bitmaps");
    }
    size_t padded_size = effective_size;
    if (device_type == ExecutorDeviceType::GPU || sub_bitmap_count > 1) {
      if (effective_size > std::numeric_limits<size_t>::max() - (sizeof(int64_t) - 1)) {
        throw std::overflow_error("Count-distinct bitmap padding overflow");
      }
      padded_size = align_to_int64(effective_size);
    }
    if (padded_size > std::numeric_limits<size_t>::max() / sub_bitmap_count) {
      throw std::overflow_error("Count-distinct bitmap allocation size overflow");
    }
    return padded_size * sub_bitmap_count;
  }
};

inline bool operator==(const CountDistinctDescriptor& lhs,
                       const CountDistinctDescriptor& rhs) {
  return lhs.impl_type_ == rhs.impl_type_ && lhs.min_val == rhs.min_val &&
         lhs.bucket_size == rhs.bucket_size && lhs.bitmap_sz_bits == rhs.bitmap_sz_bits &&
         lhs.approximate == rhs.approximate && lhs.device_type == rhs.device_type &&
         lhs.sub_bitmap_count == rhs.sub_bitmap_count;
}

inline bool operator!=(const CountDistinctDescriptor& lhs,
                       const CountDistinctDescriptor& rhs) {
  return !(lhs == rhs);
}

#endif  // QUERYENGINE_COUNTDISTINCTDESCRIPTOR_H
