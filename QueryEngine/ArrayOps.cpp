/**
 * @file    ArrayOps.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Functions to support array operations used by the executor.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <stdint.h>
#include "../Shared/funcannotations.h"
#include "../Utils/ChunkIter.h"

#ifdef EXECUTE_INCLUDE

extern "C" DEVICE
uint32_t array_size(int8_t* chunk_iter_, const uint64_t row_pos, const uint32_t elem_sz) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null ? 0 : ad.length / elem_sz;
}

#define ARRAY_AT(width)                                              \
extern "C" DEVICE                                                    \
int##width##_t array_at_i##width(int8_t* chunk_iter_,                \
                                 const uint64_t row_pos,             \
                                 const uint##width##_t elem_idx) {   \
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_); \
  ArrayDatum ad;                                                     \
  bool is_end;                                                       \
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);              \
  return reinterpret_cast<int##width##_t*>(ad.pointer)[elem_idx];    \
}

ARRAY_AT(16)
ARRAY_AT(32)
ARRAY_AT(64)

#undef ARRAY_AT

#endif  // EXECUTE_INCLUDE
