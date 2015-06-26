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
uint32_t array_size(int8_t* chunk_iter_, const uint64_t row_pos, const uint32_t elem_log_sz) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null ? 0 : ad.length >> elem_log_sz;
}

#define ARRAY_AT(width)                                              \
extern "C" DEVICE                                                    \
int##width##_t array_at_i##width(int8_t* chunk_iter_,                \
                                 const uint64_t row_pos,             \
                                 const uint32_t elem_idx) {          \
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

#define ARRAY_AT_CHECKED(width)                                                  \
extern "C" DEVICE                                                                \
int##width##_t array_at_i##width##_checked(int8_t* chunk_iter_,                  \
                                           const uint64_t row_pos,               \
                                           const int64_t elem_idx,               \
                                           const int##width##_t null_val) {      \
  if (elem_idx < 0) {                                                            \
    return null_val;                                                             \
  }                                                                              \
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);             \
  ArrayDatum ad;                                                                 \
  bool is_end;                                                                   \
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                          \
  if (ad.is_null || static_cast<size_t>(elem_idx) >= ad.length / (width >> 3)) { \
    return null_val;                                                             \
  }                                                                              \
  return reinterpret_cast<int##width##_t*>(ad.pointer)[elem_idx];                \
}

ARRAY_AT_CHECKED(16)
ARRAY_AT_CHECKED(32)
ARRAY_AT_CHECKED(64)

#undef ARRAY_AT_CHECKED

extern "C" DEVICE
int8_t* array_buff(int8_t* chunk_iter_, const uint64_t row_pos) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.pointer;
}

#endif  // EXECUTE_INCLUDE
