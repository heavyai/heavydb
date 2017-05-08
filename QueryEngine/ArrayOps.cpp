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
 * @file    ArrayOps.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Functions to support array operations used by the executor.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <stdint.h>
#include "TypePunning.h"
#include "../Shared/funcannotations.h"
#include "../Utils/ChunkIter.h"

#ifdef EXECUTE_INCLUDE

extern "C" DEVICE uint32_t array_size(int8_t* chunk_iter_, const uint64_t row_pos, const uint32_t elem_log_sz) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null ? 0 : ad.length >> elem_log_sz;
}

extern "C" DEVICE bool array_is_null(int8_t* chunk_iter_, const uint64_t row_pos) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null;
}

#define ARRAY_AT(type)                                                                                           \
  extern "C" DEVICE type array_at_##type(int8_t* chunk_iter_, const uint64_t row_pos, const uint32_t elem_idx) { \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                                           \
    ArrayDatum ad;                                                                                               \
    bool is_end;                                                                                                 \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                                                        \
    return reinterpret_cast<type*>(ad.pointer)[elem_idx];                                                        \
  }

ARRAY_AT(int8_t)
ARRAY_AT(int16_t)
ARRAY_AT(int32_t)
ARRAY_AT(int64_t)
ARRAY_AT(float)
ARRAY_AT(double)

#undef ARRAY_AT

#define ARRAY_ANY(type, needle_type, oper_name, oper)                                               \
  extern "C" DEVICE bool array_any_##oper_name##_##type##_##needle_type(                            \
      int8_t* chunk_iter_, const uint64_t row_pos, const needle_type needle, const type null_val) { \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                              \
    ArrayDatum ad;                                                                                  \
    bool is_end;                                                                                    \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                                           \
    const size_t elem_count = ad.length / sizeof(type);                                             \
    for (size_t i = 0; i < elem_count; ++i) {                                                       \
      const needle_type val = reinterpret_cast<type*>(ad.pointer)[i];                               \
      if (val != null_val && val oper needle) {                                                     \
        return true;                                                                                \
      }                                                                                             \
    }                                                                                               \
    return false;                                                                                   \
  }

#define ARRAY_ALL(type, needle_type, oper_name, oper)                                               \
  extern "C" DEVICE bool array_all_##oper_name##_##type##_##needle_type(                            \
      int8_t* chunk_iter_, const uint64_t row_pos, const needle_type needle, const type null_val) { \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                              \
    ArrayDatum ad;                                                                                  \
    bool is_end;                                                                                    \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                                           \
    const size_t elem_count = ad.length / sizeof(type);                                             \
    for (size_t i = 0; i < elem_count; ++i) {                                                       \
      const needle_type val = reinterpret_cast<type*>(ad.pointer)[i];                               \
      if (!(val != null_val && val oper needle)) {                                                  \
        return false;                                                                               \
      }                                                                                             \
    }                                                                                               \
    return true;                                                                                    \
  }

#define ARRAY_ALL_ANY_ALL_TYPES(oper_name, oper, needle_type) \
  ARRAY_ANY(int8_t, needle_type, oper_name, oper)             \
  ARRAY_ALL(int8_t, needle_type, oper_name, oper)             \
  ARRAY_ANY(int16_t, needle_type, oper_name, oper)            \
  ARRAY_ALL(int16_t, needle_type, oper_name, oper)            \
  ARRAY_ANY(int32_t, needle_type, oper_name, oper)            \
  ARRAY_ALL(int32_t, needle_type, oper_name, oper)            \
  ARRAY_ANY(int64_t, needle_type, oper_name, oper)            \
  ARRAY_ALL(int64_t, needle_type, oper_name, oper)            \
  ARRAY_ANY(float, needle_type, oper_name, oper)              \
  ARRAY_ALL(float, needle_type, oper_name, oper)              \
  ARRAY_ANY(double, needle_type, oper_name, oper)             \
  ARRAY_ALL(double, needle_type, oper_name, oper)

ARRAY_ALL_ANY_ALL_TYPES(eq, ==, int8_t)
ARRAY_ALL_ANY_ALL_TYPES(ne, !=, int8_t)
ARRAY_ALL_ANY_ALL_TYPES(lt, <, int8_t)
ARRAY_ALL_ANY_ALL_TYPES(le, <=, int8_t)
ARRAY_ALL_ANY_ALL_TYPES(gt, >, int8_t)
ARRAY_ALL_ANY_ALL_TYPES(ge, >=, int8_t)

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

#define ARRAY_AT_CHECKED(type)                                                                    \
  extern "C" DEVICE type array_at_##type##_checked(                                               \
      int8_t* chunk_iter_, const uint64_t row_pos, const int64_t elem_idx, const type null_val) { \
    if (elem_idx <= 0) {                                                                          \
      return null_val;                                                                            \
    }                                                                                             \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                            \
    ArrayDatum ad;                                                                                \
    bool is_end;                                                                                  \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                                         \
    if (ad.is_null || static_cast<size_t>(elem_idx) > ad.length / sizeof(type)) {                 \
      return null_val;                                                                            \
    }                                                                                             \
    return reinterpret_cast<type*>(ad.pointer)[elem_idx - 1];                                     \
  }

ARRAY_AT_CHECKED(int8_t)
ARRAY_AT_CHECKED(int16_t)
ARRAY_AT_CHECKED(int32_t)
ARRAY_AT_CHECKED(int64_t)
ARRAY_AT_CHECKED(float)
ARRAY_AT_CHECKED(double)

#undef ARRAY_AT_CHECKED

extern "C" DEVICE int8_t* array_buff(int8_t* chunk_iter_, const uint64_t row_pos) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.pointer;
}

#ifndef __CUDACC__

#include <set>

extern "C" ALWAYS_INLINE int64_t elem_bitcast_int8_t(const int8_t val) {
  return val;
}

extern "C" ALWAYS_INLINE int64_t elem_bitcast_int16_t(const int16_t val) {
  return val;
}

extern "C" ALWAYS_INLINE int64_t elem_bitcast_int32_t(const int32_t val) {
  return val;
}

extern "C" ALWAYS_INLINE int64_t elem_bitcast_int64_t(const int64_t val) {
  return val;
}

extern "C" ALWAYS_INLINE int64_t elem_bitcast_float(const float val) {
  const double dval{val};
  return *reinterpret_cast<const int64_t*>(may_alias_ptr(&dval));
}

extern "C" ALWAYS_INLINE int64_t elem_bitcast_double(const double val) {
  return *reinterpret_cast<const int64_t*>(may_alias_ptr(&val));
}

#define COUNT_DISTINCT_ARRAY(type)                                                      \
  extern "C" void agg_count_distinct_array_##type(                                      \
      int64_t* agg, int8_t* chunk_iter_, const uint64_t row_pos, const type null_val) { \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                  \
    ArrayDatum ad;                                                                      \
    bool is_end;                                                                        \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                               \
    const size_t elem_count{ad.length / sizeof(type)};                                  \
    for (size_t i = 0; i < elem_count; ++i) {                                           \
      const auto val = reinterpret_cast<type*>(ad.pointer)[i];                          \
      if (val != null_val) {                                                            \
        reinterpret_cast<std::set<int64_t>*>(*agg)->insert(elem_bitcast_##type(val));   \
      }                                                                                 \
    }                                                                                   \
  }

COUNT_DISTINCT_ARRAY(int8_t)
COUNT_DISTINCT_ARRAY(int16_t)
COUNT_DISTINCT_ARRAY(int32_t)
COUNT_DISTINCT_ARRAY(int64_t)
COUNT_DISTINCT_ARRAY(float)
COUNT_DISTINCT_ARRAY(double)

#undef COUNT_DISTINCT_ARRAY

#include <string>

extern "C" uint64_t string_decompress(const int32_t string_id, const int64_t string_dict_handle);

#define ARRAY_STR_ANY(type, oper_name, oper)                                           \
  extern "C" bool array_any_##oper_name##_str_##type(int8_t* chunk_iter_,              \
                                                     const uint64_t row_pos,           \
                                                     const char* needle_ptr,           \
                                                     const uint32_t needle_len,        \
                                                     const int64_t string_dict_handle, \
                                                     const type null_val) {            \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                 \
    ArrayDatum ad;                                                                     \
    bool is_end;                                                                       \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                              \
    const size_t elem_count = ad.length / sizeof(type);                                \
    std::string needle_str(needle_ptr, needle_len);                                    \
    for (size_t i = 0; i < elem_count; ++i) {                                          \
      const type val = reinterpret_cast<type*>(ad.pointer)[i];                         \
      if (val != null_val) {                                                           \
        uint64_t str_and_len = string_decompress(val, string_dict_handle);             \
        const char* str = reinterpret_cast<const char*>(str_and_len & 0xffffffffffff); \
        const uint16_t len = str_and_len >> 48;                                        \
        std::string val_str(str, len);                                                 \
        if (val_str oper needle_str) {                                                 \
          return true;                                                                 \
        }                                                                              \
      }                                                                                \
    }                                                                                  \
    return false;                                                                      \
  }

#define ARRAY_STR_ALL(type, oper_name, oper)                                           \
  extern "C" bool array_all_##oper_name##_str_##type(int8_t* chunk_iter_,              \
                                                     const uint64_t row_pos,           \
                                                     const char* needle_ptr,           \
                                                     const uint32_t needle_len,        \
                                                     const int64_t string_dict_handle, \
                                                     const type null_val) {            \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                 \
    ArrayDatum ad;                                                                     \
    bool is_end;                                                                       \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                              \
    const size_t elem_count = ad.length / sizeof(type);                                \
    std::string needle_str(needle_ptr, needle_len);                                    \
    for (size_t i = 0; i < elem_count; ++i) {                                          \
      const type val = reinterpret_cast<type*>(ad.pointer)[i];                         \
      if (val == null_val) {                                                           \
        return false;                                                                  \
      }                                                                                \
      uint64_t str_and_len = string_decompress(val, string_dict_handle);               \
      const char* str = reinterpret_cast<const char*>(str_and_len & 0xffffffffffff);   \
      const uint16_t len = str_and_len >> 48;                                          \
      std::string val_str(str, len);                                                   \
      if (!(val_str oper needle_str)) {                                                \
        return false;                                                                  \
      }                                                                                \
    }                                                                                  \
    return true;                                                                       \
  }

#define ARRAY_STR_ALL_ANY_ALL_TYPES(oper_name, oper) \
  ARRAY_STR_ANY(int8_t, oper_name, oper)             \
  ARRAY_STR_ALL(int8_t, oper_name, oper)             \
  ARRAY_STR_ANY(int16_t, oper_name, oper)            \
  ARRAY_STR_ALL(int16_t, oper_name, oper)            \
  ARRAY_STR_ANY(int32_t, oper_name, oper)            \
  ARRAY_STR_ALL(int32_t, oper_name, oper)            \
  ARRAY_STR_ANY(int64_t, oper_name, oper)            \
  ARRAY_STR_ALL(int64_t, oper_name, oper)

ARRAY_STR_ALL_ANY_ALL_TYPES(eq, ==)
ARRAY_STR_ALL_ANY_ALL_TYPES(ne, !=)
ARRAY_STR_ALL_ANY_ALL_TYPES(lt, <)
ARRAY_STR_ALL_ANY_ALL_TYPES(le, <=)
ARRAY_STR_ALL_ANY_ALL_TYPES(gt, >)
ARRAY_STR_ALL_ANY_ALL_TYPES(ge, >=)

#undef ARRAY_ALL_ANY_ALL_TYPES
#undef ARRAY_STR_ALL
#undef ARRAY_STR_ANY

#endif

#endif  // EXECUTE_INCLUDE
