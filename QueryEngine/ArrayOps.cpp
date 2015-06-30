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

extern "C" DEVICE
bool array_is_null(int8_t* chunk_iter_, const uint64_t row_pos) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null;
}

#define ARRAY_AT(type)                                               \
extern "C" DEVICE                                                    \
type array_at_##type(int8_t* chunk_iter_,                            \
                     const uint64_t row_pos,                         \
                     const uint32_t elem_idx) {                      \
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_); \
  ArrayDatum ad;                                                     \
  bool is_end;                                                       \
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);              \
  return reinterpret_cast<type*>(ad.pointer)[elem_idx];              \
}

ARRAY_AT(int16_t)
ARRAY_AT(int32_t)
ARRAY_AT(int64_t)
ARRAY_AT(float)
ARRAY_AT(double)

#undef ARRAY_AT

#define ARRAY_AT_CHECKED(type)                                                   \
extern "C" DEVICE                                                                \
type array_at_##type##_checked(int8_t* chunk_iter_,                              \
                               const uint64_t row_pos,                           \
                               const int64_t elem_idx,                           \
                               const type null_val) {                            \
  if (elem_idx < 0) {                                                            \
    return null_val;                                                             \
  }                                                                              \
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);             \
  ArrayDatum ad;                                                                 \
  bool is_end;                                                                   \
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                          \
  if (ad.is_null || static_cast<size_t>(elem_idx) >= ad.length / sizeof(type)) { \
    return null_val;                                                             \
  }                                                                              \
  return reinterpret_cast<type*>(ad.pointer)[elem_idx];                          \
}

ARRAY_AT_CHECKED(int16_t)
ARRAY_AT_CHECKED(int32_t)
ARRAY_AT_CHECKED(int64_t)
ARRAY_AT_CHECKED(float)
ARRAY_AT_CHECKED(double)

#undef ARRAY_AT_CHECKED

extern "C" DEVICE
int8_t* array_buff(int8_t* chunk_iter_, const uint64_t row_pos) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.pointer;
}

#ifndef __CUDACC__

#include <set>

extern "C" ALWAYS_INLINE
int64_t elem_bitcast_int16_t(const int16_t val) {
  return val;
}

extern "C" ALWAYS_INLINE
int64_t elem_bitcast_int32_t(const int32_t val) {
  return val;
}

extern "C" ALWAYS_INLINE
int64_t elem_bitcast_int64_t(const int64_t val) {
  return val;
}

extern "C" ALWAYS_INLINE
int64_t elem_bitcast_float(const float val) {
  const double dval { val };
  return *reinterpret_cast<const int64_t*>(&dval);
}

extern "C" ALWAYS_INLINE
int64_t elem_bitcast_double(const double val) {
  return *reinterpret_cast<const int64_t*>(&val);
}

#define COUNT_DISTINCT_ARRAY(type)                                   \
extern "C"                                                           \
void agg_count_distinct_array_##type(int64_t* agg,                   \
                                     int8_t* chunk_iter_,            \
                                     const uint64_t row_pos,         \
                                     const type null_val) {          \
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_); \
  ArrayDatum ad;                                                     \
  bool is_end;                                                       \
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);              \
  const size_t elem_count { ad.length / sizeof(type) };              \
  for (size_t i = 0; i < elem_count; ++i) {                          \
    const auto val = reinterpret_cast<type*>(ad.pointer)[i];         \
    if (val != null_val) {                                           \
      reinterpret_cast<std::set<int64_t>*>(*agg)->insert(            \
        elem_bitcast_##type(val));                                   \
    }                                                                \
  }                                                                  \
}

COUNT_DISTINCT_ARRAY(int16_t)
COUNT_DISTINCT_ARRAY(int32_t)
COUNT_DISTINCT_ARRAY(int64_t)
COUNT_DISTINCT_ARRAY(float)
COUNT_DISTINCT_ARRAY(double)

#undef COUNT_DISTINCT_ARRAY

#endif

#endif  // EXECUTE_INCLUDE
