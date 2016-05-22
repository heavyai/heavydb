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

#define MAX_BYTE_WIDTH_SUPPORTED 8

#ifndef __CUDACC__
inline unsigned compact_byte_width(unsigned qw, unsigned low_bound) {
  return std::max(qw, low_bound);
}
#endif

template <typename T>
FORCE_INLINE DEVICE T align_to_int64(T addr) {
  addr += sizeof(int64_t) - 1;
  return (T)(((uint64_t)addr >> 3) << 3);
}

#endif /* BUFFER_COMPACTION_H */
