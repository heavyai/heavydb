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
 * @file    CountDistinctDescriptor.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Descriptor for the storage layout use for (approximate) count distinct operations.
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_COUNTDISTINCTDESCRIPTOR_H
#define QUERYENGINE_COUNTDISTINCTDESCRIPTOR_H

#include "BufferCompaction.h"
#include "CompilationOptions.h"

#include <glog/logging.h>

inline size_t bitmap_bits_to_bytes(const size_t bitmap_sz) {
  size_t bitmap_byte_sz = bitmap_sz / 8;
  if (bitmap_sz % 8) {
    ++bitmap_byte_sz;
  }
  return bitmap_byte_sz;
}

enum class CountDistinctImplType { Invalid, Bitmap, StdSet };

struct CountDistinctDescriptor {
  CountDistinctImplType impl_type_;
  int64_t min_val;
  int64_t bitmap_sz_bits;
  bool approximate;
  ExecutorDeviceType device_type;
  size_t sub_bitmap_count;

  size_t bitmapSizeBytes() const {
    CHECK(impl_type_ == CountDistinctImplType::Bitmap);
    const auto approx_reg_bytes = (device_type == ExecutorDeviceType::GPU ? sizeof(int32_t) : 1);
    return approximate ? (1 << bitmap_sz_bits) * approx_reg_bytes : bitmap_bits_to_bytes(bitmap_sz_bits);
  }

  size_t bitmapPaddedSizeBytes() const {
    const auto effective_size = bitmapSizeBytes();
    const auto padded_size = (device_type == ExecutorDeviceType::GPU || sub_bitmap_count > 1)
                                 ? align_to_int64(effective_size)
                                 : effective_size;
    return padded_size * sub_bitmap_count;
  }
};

inline bool operator==(const CountDistinctDescriptor& lhs, const CountDistinctDescriptor& rhs) {
  return lhs.impl_type_ == rhs.impl_type_ && lhs.min_val == rhs.min_val && lhs.bitmap_sz_bits == rhs.bitmap_sz_bits &&
         lhs.approximate == rhs.approximate && lhs.device_type == rhs.device_type;
}

#endif  // QUERYENGINE_COUNTDISTINCTDESCRIPTOR_H
