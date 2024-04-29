/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @brief   Functions to support array operations used by the executor.
 *
 */

#include <cstdint>
#include "../Shared/funcannotations.h"
#include "../Utils/ChunkIter.h"
#include "TypePunning.h"

#include <boost/preprocessor/seq/for_each_product.hpp>

#ifdef EXECUTE_INCLUDE

extern "C" DEVICE RUNTIME_EXPORT int32_t array_size(int8_t* chunk_iter_,
                                                    const uint64_t row_pos,
                                                    const uint32_t elem_log_sz) {
  if (!chunk_iter_) {
    return 0;
  }
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null ? 0 : ad.length >> elem_log_sz;
}

extern "C" DEVICE RUNTIME_EXPORT int32_t array_size_nullable(int8_t* chunk_iter_,
                                                             const uint64_t row_pos,
                                                             const uint32_t elem_log_sz,
                                                             const int32_t null_val) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null ? null_val : ad.length >> elem_log_sz;
}

extern "C" DEVICE RUNTIME_EXPORT int32_t array_size_1_nullable(int8_t* chunk_iter_,
                                                               const uint64_t row_pos,
                                                               const int32_t null_val) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null ? null_val : 1;
}

extern "C" DEVICE RUNTIME_EXPORT bool array_is_null(int8_t* chunk_iter_,
                                                    const uint64_t row_pos) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null;
}

extern "C" DEVICE RUNTIME_EXPORT bool point_coord_array_is_null(int8_t* chunk_iter_,
                                                                const uint64_t row_pos) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth_point_coords(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null;
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
point_coord_array_size(int8_t* chunk_iter_,
                       const uint64_t row_pos,
                       const uint32_t elem_log_sz) {
  if (!chunk_iter_) {
    return 0;
  }
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth_point_coords(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null ? 0 : ad.length >> elem_log_sz;
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
point_coord_array_size_nullable(int8_t* chunk_iter_,
                                const uint64_t row_pos,
                                const uint32_t elem_log_sz,
                                const int32_t null_val) {
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth_point_coords(chunk_iter, row_pos, &ad, &is_end);
  return ad.is_null ? null_val : ad.length >> elem_log_sz;
}

namespace dot_product {

// Define structs to provide a uniform API to access the array data:
//  * ArrayExpr arrays are passed in by their array pointer directly.
//  * Chunk data requires ChunkIter_get_nth() to access the array data.

template <typename T>
struct ArrayExpr {
  T const* ptr;
  size_t array_size;
  DEVICE inline ArrayExpr(int8_t* ptr, uint64_t array_size, uint32_t)
      : ptr(reinterpret_cast<T const*>(ptr))
      , array_size(static_cast<size_t>(array_size)) {}

  DEVICE inline std::pair<T const*, size_t> getArrayPointerAndSize() const {
    return std::make_pair(ptr, array_size);
  }
};

template <typename T>
struct Chunk {
  ChunkIter* ptr;
  uint64_t row_pos;
  uint32_t elem_log_size;
  DEVICE inline Chunk(int8_t* ptr, uint64_t row_pos, uint32_t elem_log_size)
      : ptr(reinterpret_cast<ChunkIter*>(ptr))
      , row_pos(row_pos)
      , elem_log_size(elem_log_size) {}

  DEVICE inline std::pair<T const*, size_t> getArrayPointerAndSize() const {
    ArrayDatum array_datum;
    bool is_end;
    ChunkIter_get_nth(ptr, row_pos, &array_datum, &is_end);
    if (array_datum.is_null) {
      return {nullptr, 0u};
    }
    size_t const array_size = array_datum.length >> elem_log_size;
    return std::make_pair(reinterpret_cast<T const*>(array_datum.pointer), array_size);
  }
};

template <typename T, typename STRUCT1, typename STRUCT2>
DEVICE inline T calculate(STRUCT1 const arg1, STRUCT2 const arg2, T const null_val) {
  if (arg1.ptr && arg2.ptr) {
    auto [array_ptr1, array_size1] = arg1.getArrayPointerAndSize();
    auto [array_ptr2, array_size2] = arg2.getArrayPointerAndSize();
    if (array_ptr1 && array_ptr2 && array_size1 == array_size2) {
      T sum{0};
      for (size_t i = 0; i < array_size1; ++i) {
        sum += static_cast<T>(array_ptr1[i]) * static_cast<T>(array_ptr2[i]);
      }
      return sum;
    }
  }
  return null_val;
}

}  // namespace dot_product

#define ARRAY_DOT_PRODUCT(ret_type, struct1, type1, struct2, type2)                      \
  extern "C" DEVICE RUNTIME_EXPORT ret_type                                              \
      array_dot_product_##struct1##_##type1##_##struct2##_##type2(                       \
          int8_t* const ptr1,            /* ChunkIter or array ptr */                    \
          uint64_t const n1,             /* row_pos or array_size */                     \
          uint32_t const elem_log_size1, /* unused for ArrayExpr */                      \
          int8_t* const ptr2,            /* ChunkIter or array ptr */                    \
          uint64_t const n2,             /* row_pos or array_size */                     \
          uint32_t const elem_log_size2, /* unused for ArrayExpr */                      \
          ret_type const null_val) {                                                     \
    return dot_product::calculate(dot_product::struct1<type1>{ptr1, n1, elem_log_size1}, \
                                  dot_product::struct2<type2>{ptr2, n2, elem_log_size2}, \
                                  null_val);                                             \
  }

#define EXPAND_ARRAY_DOT_PRODUCT(r, cross_product)                   \
  EXPAND_ARRAY_DOT_PRODUCT_IMPL(BOOST_PP_SEQ_ELEM(0, cross_product), \
                                BOOST_PP_SEQ_ELEM(1, cross_product), \
                                BOOST_PP_SEQ_ELEM(2, cross_product), \
                                BOOST_PP_SEQ_ELEM(3, cross_product), \
                                BOOST_PP_SEQ_ELEM(4, cross_product))

#define EXPAND_ARRAY_DOT_PRODUCT_IMPL(ret_type, struct1, type1, struct2, type2) \
  ARRAY_DOT_PRODUCT(ret_type, struct1, type1, struct2, type2)

// clang-format off
// Define array_dot_product_* functions in sets of cartesian products.
// Must match the ret_types table in QueryEngine/DotProductReturnTypes.h
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_ARRAY_DOT_PRODUCT,
    ((int32_t))           // ret_type
    ((ArrayExpr)(Chunk))  // struct1
    ((int8_t))            // type1
    ((ArrayExpr)(Chunk))  // struct2
    ((int8_t)))           // type2
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_ARRAY_DOT_PRODUCT,
    ((int32_t))
    ((ArrayExpr)(Chunk))
    ((int16_t))
    ((ArrayExpr)(Chunk))
    ((int8_t)(int16_t)))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_ARRAY_DOT_PRODUCT,
    ((int64_t))
    ((ArrayExpr)(Chunk))
    ((int32_t))
    ((ArrayExpr)(Chunk))
    ((int8_t)(int16_t)(int32_t)))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_ARRAY_DOT_PRODUCT,
    ((int64_t))
    ((ArrayExpr)(Chunk))
    ((int64_t))
    ((ArrayExpr)(Chunk))
    ((int8_t)(int16_t)(int32_t)(int64_t)))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_ARRAY_DOT_PRODUCT,
    ((float))
    ((ArrayExpr)(Chunk))
    ((float))
    ((ArrayExpr)(Chunk))
    ((int8_t)(int16_t)(float)))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_ARRAY_DOT_PRODUCT,
    ((double))
    ((ArrayExpr)(Chunk))
    ((float))
    ((ArrayExpr)(Chunk))
    ((int32_t)(int64_t)))
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_ARRAY_DOT_PRODUCT,
    ((double))
    ((ArrayExpr)(Chunk))
    ((double))
    ((ArrayExpr)(Chunk))
    ((int8_t)(int16_t)(int32_t)(int64_t)(float)(double)))
// clang-format on

#undef EXPAND_ARRAY_DOT_PRODUCT_IMPL
#undef EXPAND_ARRAY_DOT_PRODUCT
#undef ARRAY_DOT_PRODUCT

#define ARRAY_AT(type)                                                        \
  extern "C" DEVICE RUNTIME_EXPORT type array_at_##type(                      \
      int8_t* chunk_iter_, const uint64_t row_pos, const uint32_t elem_idx) { \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);        \
    ArrayDatum ad;                                                            \
    bool is_end;                                                              \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                     \
    return reinterpret_cast<type*>(ad.pointer)[elem_idx];                     \
  }

ARRAY_AT(int8_t)
ARRAY_AT(int16_t)
ARRAY_AT(int32_t)
ARRAY_AT(int64_t)
ARRAY_AT(float)
ARRAY_AT(double)

#undef ARRAY_AT

#define VARLEN_ARRAY_AT(type)                                                 \
  extern "C" DEVICE RUNTIME_EXPORT type varlen_array_at_##type(               \
      int8_t* chunk_iter_, const uint64_t row_pos, const uint32_t elem_idx) { \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);        \
    ArrayDatum ad;                                                            \
    bool is_end;                                                              \
    ChunkIter_get_nth_varlen(chunk_iter, row_pos, &ad, &is_end);              \
    return reinterpret_cast<type*>(ad.pointer)[elem_idx];                     \
  }

VARLEN_ARRAY_AT(int8_t)
VARLEN_ARRAY_AT(int16_t)
VARLEN_ARRAY_AT(int32_t)
VARLEN_ARRAY_AT(int64_t)
VARLEN_ARRAY_AT(float)
VARLEN_ARRAY_AT(double)

#undef VARLEN_ARRAY_AT

#define VARLEN_NOTNULL_ARRAY_AT(type)                                         \
  extern "C" DEVICE RUNTIME_EXPORT type varlen_notnull_array_at_##type(       \
      int8_t* chunk_iter_, const uint64_t row_pos, const uint32_t elem_idx) { \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);        \
    ArrayDatum ad;                                                            \
    bool is_end;                                                              \
    ChunkIter_get_nth_varlen_notnull(chunk_iter, row_pos, &ad, &is_end);      \
    return reinterpret_cast<type*>(ad.pointer)[elem_idx];                     \
  }

VARLEN_NOTNULL_ARRAY_AT(int8_t)
VARLEN_NOTNULL_ARRAY_AT(int16_t)
VARLEN_NOTNULL_ARRAY_AT(int32_t)
VARLEN_NOTNULL_ARRAY_AT(int64_t)
VARLEN_NOTNULL_ARRAY_AT(float)
VARLEN_NOTNULL_ARRAY_AT(double)

#undef VARLEN_NOTNULL_ARRAY_AT

#define ARRAY_ANY(type, needle_type, oper_name, oper)                                   \
  extern "C" DEVICE RUNTIME_EXPORT bool array_any_##oper_name##_##type##_##needle_type( \
      int8_t* chunk_iter_,                                                              \
      const uint64_t row_pos,                                                           \
      const needle_type needle,                                                         \
      const type null_val) {                                                            \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                  \
    ArrayDatum ad;                                                                      \
    bool is_end;                                                                        \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                               \
    const size_t elem_count = ad.length / sizeof(type);                                 \
    for (size_t i = 0; i < elem_count; ++i) {                                           \
      const needle_type val = reinterpret_cast<type*>(ad.pointer)[i];                   \
      if (val != null_val && val oper needle) {                                         \
        return true;                                                                    \
      }                                                                                 \
    }                                                                                   \
    return false;                                                                       \
  }

#define ARRAY_ALL(type, needle_type, oper_name, oper)                                   \
  extern "C" DEVICE RUNTIME_EXPORT bool array_all_##oper_name##_##type##_##needle_type( \
      int8_t* chunk_iter_,                                                              \
      const uint64_t row_pos,                                                           \
      const needle_type needle,                                                         \
      const type null_val) {                                                            \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                  \
    ArrayDatum ad;                                                                      \
    bool is_end;                                                                        \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                               \
    const size_t elem_count = ad.length / sizeof(type);                                 \
    for (size_t i = 0; i < elem_count; ++i) {                                           \
      const needle_type val = reinterpret_cast<type*>(ad.pointer)[i];                   \
      if (!(val != null_val && val oper needle)) {                                      \
        return false;                                                                   \
      }                                                                                 \
    }                                                                                   \
    return true;                                                                        \
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

#define ARRAY_AT_CHECKED(type)                                                    \
  extern "C" DEVICE RUNTIME_EXPORT type array_at_##type##_checked(                \
      int8_t* chunk_iter_,                                                        \
      const uint64_t row_pos,                                                     \
      const int64_t elem_idx,                                                     \
      const type null_val) {                                                      \
    if (elem_idx <= 0) {                                                          \
      return null_val;                                                            \
    }                                                                             \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);            \
    ArrayDatum ad;                                                                \
    bool is_end;                                                                  \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                         \
    if (ad.is_null || static_cast<size_t>(elem_idx) > ad.length / sizeof(type)) { \
      return null_val;                                                            \
    }                                                                             \
    return reinterpret_cast<type*>(ad.pointer)[elem_idx - 1];                     \
  }

ARRAY_AT_CHECKED(int8_t)
ARRAY_AT_CHECKED(int16_t)
ARRAY_AT_CHECKED(int32_t)
ARRAY_AT_CHECKED(int64_t)
ARRAY_AT_CHECKED(float)
ARRAY_AT_CHECKED(double)

#undef ARRAY_AT_CHECKED

extern "C" DEVICE RUNTIME_EXPORT int8_t* allocate_varlen_buffer(int64_t element_count,
                                                                int64_t element_size) {
#ifndef __CUDACC__
  int8_t* varlen_buffer =
      reinterpret_cast<int8_t*>(checked_malloc((element_count + 1) * element_size));
  return varlen_buffer;
#else
  return nullptr;
#endif
}

extern "C" DEVICE RUNTIME_EXPORT ALWAYS_INLINE int32_t
fast_fixlen_array_size(int8_t* chunk_iter_, const uint32_t elem_log_sz) {
  ChunkIter* it = reinterpret_cast<ChunkIter*>(chunk_iter_);
  return it->skip_size >> elem_log_sz;
}

extern "C" DEVICE RUNTIME_EXPORT ALWAYS_INLINE int8_t* fast_fixlen_array_buff(
    int8_t* chunk_iter_,
    const uint64_t row_pos) {
  if (!chunk_iter_) {
    return nullptr;
  }
  ChunkIter* it = reinterpret_cast<ChunkIter*>(chunk_iter_);
  auto n = static_cast<int>(row_pos);
  int8_t* current_pos = it->start_pos + n * it->skip_size;
  return current_pos;
}

extern "C" DEVICE RUNTIME_EXPORT ALWAYS_INLINE int64_t
determine_fixed_array_len(int8_t* chunk_iter, int64_t valid_len) {
  return chunk_iter ? valid_len : 0;
}

extern "C" DEVICE RUNTIME_EXPORT int8_t* array_buff(int8_t* chunk_iter_,
                                                    const uint64_t row_pos) {
  if (!chunk_iter_) {
    return nullptr;
  }
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  return ad.pointer;
}

#ifndef __CUDACC__

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t elem_bitcast_int8_t(const int8_t val) {
  return val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t elem_bitcast_int16_t(const int16_t val) {
  return val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t elem_bitcast_int32_t(const int32_t val) {
  return val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t elem_bitcast_int64_t(const int64_t val) {
  return val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t elem_bitcast_float(const float val) {
  const double dval{val};
  return *reinterpret_cast<const int64_t*>(may_alias_ptr(&dval));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t elem_bitcast_double(const double val) {
  return *reinterpret_cast<const int64_t*>(may_alias_ptr(&val));
}

#define COUNT_DISTINCT_ARRAY(type)                                                      \
  extern "C" RUNTIME_EXPORT void agg_count_distinct_array_##type(                       \
      int64_t* agg, int8_t* chunk_iter_, const uint64_t row_pos, const type null_val) { \
    ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);                  \
    ArrayDatum ad;                                                                      \
    bool is_end;                                                                        \
    ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);                               \
    const size_t elem_count{ad.length / sizeof(type)};                                  \
    for (size_t i = 0; i < elem_count; ++i) {                                           \
      const auto val = reinterpret_cast<type*>(ad.pointer)[i];                          \
      if (val != null_val) {                                                            \
        reinterpret_cast<CountDistinctSet*>(*agg)->insert(elem_bitcast_##type(val));    \
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

#include <functional>
#include <string_view>

extern "C" RUNTIME_EXPORT StringView string_decompress(const int32_t string_id,
                                                       const int64_t string_dict_handle);

template <typename T>
bool array_any(int8_t* const chunk_iter_i8,
               uint64_t const row_pos,
               std::string_view const needle_str,
               int64_t const string_dict_handle,
               T const null_val,
               std::function<bool(std::string_view, std::string_view)> const cmp) {
  ChunkIter* const chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_i8);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  size_t const elem_count = ad.length / sizeof(T);
  for (size_t i = 0; i < elem_count; ++i) {
    T const val = reinterpret_cast<T*>(ad.pointer)[i];
    if (val != null_val) {
      StringView const sv = string_decompress(val, string_dict_handle);
      if (cmp(sv.stringView(), needle_str)) {
        return true;
      }
    }
  }
  return false;
}

template <typename T>
bool array_all(int8_t* const chunk_iter_i8,
               uint64_t const row_pos,
               std::string_view const needle_str,
               int64_t const string_dict_handle,
               T const null_val,
               std::function<bool(std::string_view, std::string_view)> const cmp) {
  ChunkIter* const chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_i8);
  ArrayDatum ad;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, row_pos, &ad, &is_end);
  size_t const elem_count = ad.length / sizeof(T);
  for (size_t i = 0; i < elem_count; ++i) {
    T const val = reinterpret_cast<T*>(ad.pointer)[i];
    if (val == null_val) {
      return false;
    }
    StringView const sv = string_decompress(val, string_dict_handle);
    if (!cmp(sv.stringView(), needle_str)) {
      return false;
    }
  }
  return true;
}

#define ARRAY_STR_ANY(type, oper_name, oper)                         \
  extern "C" RUNTIME_EXPORT bool array_any_##oper_name##_str_##type( \
      int8_t* const chunk_iter_i8,                                   \
      uint64_t const row_pos,                                        \
      char const* const needle_ptr,                                  \
      uint32_t const needle_len,                                     \
      int64_t const string_dict_handle,                              \
      type const null_val) {                                         \
    return array_any(chunk_iter_i8,                                  \
                     row_pos,                                        \
                     std::string_view{needle_ptr, needle_len},       \
                     string_dict_handle,                             \
                     null_val,                                       \
                     std::oper<std::string_view>{});                 \
  }

#define ARRAY_STR_ALL(type, oper_name, oper)                         \
  extern "C" RUNTIME_EXPORT bool array_all_##oper_name##_str_##type( \
      int8_t* const chunk_iter_i8,                                   \
      uint64_t const row_pos,                                        \
      char const* const needle_ptr,                                  \
      uint32_t const needle_len,                                     \
      int64_t const string_dict_handle,                              \
      type const null_val) {                                         \
    return array_all(chunk_iter_i8,                                  \
                     row_pos,                                        \
                     std::string_view{needle_ptr, needle_len},       \
                     string_dict_handle,                             \
                     null_val,                                       \
                     std::oper<std::string_view>{});                 \
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

ARRAY_STR_ALL_ANY_ALL_TYPES(eq, equal_to)
ARRAY_STR_ALL_ANY_ALL_TYPES(ne, not_equal_to)
ARRAY_STR_ALL_ANY_ALL_TYPES(lt, less)
ARRAY_STR_ALL_ANY_ALL_TYPES(le, less_equal)
ARRAY_STR_ALL_ANY_ALL_TYPES(gt, greater)
ARRAY_STR_ALL_ANY_ALL_TYPES(ge, greater_equal)

#undef ARRAY_ALL_ANY_ALL_TYPES
#undef ARRAY_STR_ALL
#undef ARRAY_STR_ANY

#endif

#endif  // EXECUTE_INCLUDE
