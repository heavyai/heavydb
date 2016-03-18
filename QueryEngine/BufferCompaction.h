/*
 * @file    BufferCompaction.h
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Macros and functions for groupby buffer compaction
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef BUFFER_COMPACTION_H
#define BUFFER_COMPACTION_H

#include "../Shared/funcannotations.h"
#include <stdint.h>

#ifndef __CUDACC__
#include <algorithm>
#endif

#define SMALLEST_BIT_WIDTH_TO_COMPACT 64

extern "C" FORCE_INLINE DEVICE unsigned compact_bit_width(unsigned qw) {
#ifdef __CUDACC__
  return max((qw << 3), SMALLEST_BIT_WIDTH_TO_COMPACT);
#else
  return std::max((qw << 3), static_cast<unsigned>(SMALLEST_BIT_WIDTH_TO_COMPACT));
#endif
}

extern "C" FORCE_INLINE DEVICE unsigned compact_byte_width(unsigned qw) {
#ifdef __CUDACC__
  return max(qw, (SMALLEST_BIT_WIDTH_TO_COMPACT >> 3));
#else
  return std::max(qw, static_cast<unsigned>(SMALLEST_BIT_WIDTH_TO_COMPACT >> 3));
#endif
}

template <typename T>
FORCE_INLINE DEVICE T align_to_int64(T addr) {
  static_assert(sizeof(T) <= sizeof(int64_t), "Unsupported template type");
  addr += sizeof(int64_t) - 1;
  return (T)(((uint64_t)addr >> 3) << 3);
}

#endif /* BUFFER_COMPACTION_H */
