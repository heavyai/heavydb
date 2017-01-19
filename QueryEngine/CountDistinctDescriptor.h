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

  size_t bitmapSizeBytes() const {
    CHECK(impl_type_ == CountDistinctImplType::Bitmap);
    const auto approx_reg_bytes = (device_type == ExecutorDeviceType::GPU ? sizeof(int32_t) : 1);
    return approximate ? (1 << bitmap_sz_bits) * approx_reg_bytes : bitmap_bits_to_bytes(bitmap_sz_bits);
  }

  size_t bitmapPaddedSizeBytes() const {
    const auto effective_size = bitmapSizeBytes();
    return device_type == ExecutorDeviceType::GPU ? align_to_int64(effective_size) : effective_size;
  }
};

#endif  // QUERYENGINE_COUNTDISTINCTDESCRIPTOR_H
