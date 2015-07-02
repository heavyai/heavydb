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

#define ARRAY_ANY(type, needle_type, oper_name, oper)                           \
extern "C" DEVICE                                                               \
bool array_any_##oper_name##_##type##_##needle_type(int8_t* chunk_iter_,        \
                                                    const uint64_t row_pos,     \
                                                    const needle_type needle) { \
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);            \
  ArrayDatum ad;                                                                \
  bool is_end;                                                                  \
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                         \
  const size_t elem_count = ad.length / sizeof(type);                           \
  for (size_t i = 0; i < elem_count; ++i) {                                     \
    const needle_type val = reinterpret_cast<type*>(ad.pointer)[i];             \
    if (val oper needle) {                                                      \
      return true;                                                              \
    }                                                                           \
  }                                                                             \
  return false;                                                                 \
}

#define ARRAY_ALL(type, needle_type, oper_name, oper)                           \
extern "C" DEVICE                                                               \
bool array_all_##oper_name##_##type##_##needle_type(int8_t* chunk_iter_,        \
                                                    const uint64_t row_pos,     \
                                                    const needle_type needle) { \
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);            \
  ArrayDatum ad;                                                                \
  bool is_end;                                                                  \
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                         \
  const size_t elem_count = ad.length / sizeof(type);                           \
  for (size_t i = 0; i < elem_count; ++i) {                                     \
    const needle_type val = reinterpret_cast<type*>(ad.pointer)[i];             \
    if (!(val oper needle)) {                                                   \
      return false;                                                             \
    }                                                                           \
  }                                                                             \
  return true;                                                                  \
}

#define ARRAY_ALL_ANY_ALL_TYPES(oper_name, oper, needle_type) \
ARRAY_ANY(int16_t, needle_type, oper_name, oper)              \
ARRAY_ALL(int16_t, needle_type, oper_name, oper)              \
ARRAY_ANY(int32_t, needle_type, oper_name, oper)              \
ARRAY_ALL(int32_t, needle_type, oper_name, oper)              \
ARRAY_ANY(int64_t, needle_type, oper_name, oper)              \
ARRAY_ALL(int64_t, needle_type, oper_name, oper)              \
ARRAY_ANY(float, needle_type, oper_name, oper)                \
ARRAY_ALL(float, needle_type, oper_name, oper)                \
ARRAY_ANY(double, needle_type, oper_name, oper)               \
ARRAY_ALL(double, needle_type, oper_name, oper)

ARRAY_ALL_ANY_ALL_TYPES(eq, ==, int16_t)
ARRAY_ALL_ANY_ALL_TYPES(ne, !=, int16_t)
ARRAY_ALL_ANY_ALL_TYPES(lt, <, int16_t)
ARRAY_ALL_ANY_ALL_TYPES(le, <=, int16_t)
ARRAY_ALL_ANY_ALL_TYPES(gt, >, int16_t)
ARRAY_ALL_ANY_ALL_TYPES(ge, >=, int16_t)

ARRAY_ALL_ANY_ALL_TYPES(eq, ==, int32_t)
ARRAY_ALL_ANY_ALL_TYPES(ne, !=, int32_t)
ARRAY_ALL_ANY_ALL_TYPES(lt, <, int32_t)
ARRAY_ALL_ANY_ALL_TYPES(le, <=, int32_t)
ARRAY_ALL_ANY_ALL_TYPES(gt, >, int32_t)
ARRAY_ALL_ANY_ALL_TYPES(ge, >=, int32_t)

ARRAY_ALL_ANY_ALL_TYPES(eq, ==, int64_t)
ARRAY_ALL_ANY_ALL_TYPES(ne, !=, int64_t)
ARRAY_ALL_ANY_ALL_TYPES(lt, <, int64_t)
ARRAY_ALL_ANY_ALL_TYPES(le, <=, int64_t)
ARRAY_ALL_ANY_ALL_TYPES(gt, >, int64_t)
ARRAY_ALL_ANY_ALL_TYPES(ge, >=, int64_t)

ARRAY_ALL_ANY_ALL_TYPES(eq, ==, float)
ARRAY_ALL_ANY_ALL_TYPES(ne, !=, float)
ARRAY_ALL_ANY_ALL_TYPES(lt, <, float)
ARRAY_ALL_ANY_ALL_TYPES(le, <=, float)
ARRAY_ALL_ANY_ALL_TYPES(gt, >, float)
ARRAY_ALL_ANY_ALL_TYPES(ge, >=, float)

ARRAY_ALL_ANY_ALL_TYPES(eq, ==, double)
ARRAY_ALL_ANY_ALL_TYPES(ne, !=, double)
ARRAY_ALL_ANY_ALL_TYPES(lt, <, double)
ARRAY_ALL_ANY_ALL_TYPES(le, <=, double)
ARRAY_ALL_ANY_ALL_TYPES(gt, >, double)
ARRAY_ALL_ANY_ALL_TYPES(ge, >=, double)

#undef ARRAY_ALL_ANY_ALL_TYPES
#undef ARRAY_ALL
#undef ARRAY_ANY

#define ARRAY_AT_CHECKED(type)                                                   \
extern "C" DEVICE                                                                \
type array_at_##type##_checked(int8_t* chunk_iter_,                              \
                               const uint64_t row_pos,                           \
                               const int64_t elem_idx,                           \
                               const type null_val) {                            \
  if (elem_idx <= 0) {                                                           \
    return null_val;                                                             \
  }                                                                              \
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);             \
  ArrayDatum ad;                                                                 \
  bool is_end;                                                                   \
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                          \
  if (ad.is_null || static_cast<size_t>(elem_idx) > ad.length / sizeof(type)) {  \
    return null_val;                                                             \
  }                                                                              \
  return reinterpret_cast<type*>(ad.pointer)[elem_idx - 1];                      \
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
