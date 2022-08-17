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

#ifdef __CUDACC__
#error This code is not intended to be compiled with a CUDA C++ compiler
#endif  // __CUDACC__

#include "RuntimeFunctions.h"
#include "../Shared/Datum.h"
#include "../Shared/funcannotations.h"
#include "BufferCompaction.h"
#include "HyperLogLogRank.h"
#include "MurmurHash.h"
#include "Shared/quantile.h"
#include "TypePunning.h"
#include "Utils/SegmentTreeUtils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <tuple>

// decoder implementations

#include "DecodersImpl.h"

// arithmetic operator implementations

#define DEF_ARITH_NULLABLE(type, null_type, opname, opsym)                 \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE type opname##_##type##_nullable( \
      const type lhs, const type rhs, const null_type null_val) {          \
    if (lhs != null_val && rhs != null_val) {                              \
      return lhs opsym rhs;                                                \
    }                                                                      \
    return null_val;                                                       \
  }

#define DEF_ARITH_NULLABLE_LHS(type, null_type, opname, opsym)                 \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE type opname##_##type##_nullable_lhs( \
      const type lhs, const type rhs, const null_type null_val) {              \
    if (lhs != null_val) {                                                     \
      return lhs opsym rhs;                                                    \
    }                                                                          \
    return null_val;                                                           \
  }

#define DEF_ARITH_NULLABLE_RHS(type, null_type, opname, opsym)                 \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE type opname##_##type##_nullable_rhs( \
      const type lhs, const type rhs, const null_type null_val) {              \
    if (rhs != null_val) {                                                     \
      return lhs opsym rhs;                                                    \
    }                                                                          \
    return null_val;                                                           \
  }

#define DEF_CMP_NULLABLE(type, null_type, opname, opsym)                     \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t opname##_##type##_nullable( \
      const type lhs,                                                        \
      const type rhs,                                                        \
      const null_type null_val,                                              \
      const int8_t null_bool_val) {                                          \
    if (lhs != null_val && rhs != null_val) {                                \
      return lhs opsym rhs;                                                  \
    }                                                                        \
    return null_bool_val;                                                    \
  }

#define DEF_CMP_NULLABLE_LHS(type, null_type, opname, opsym)                     \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t opname##_##type##_nullable_lhs( \
      const type lhs,                                                            \
      const type rhs,                                                            \
      const null_type null_val,                                                  \
      const int8_t null_bool_val) {                                              \
    if (lhs != null_val) {                                                       \
      return lhs opsym rhs;                                                      \
    }                                                                            \
    return null_bool_val;                                                        \
  }

#define DEF_CMP_NULLABLE_RHS(type, null_type, opname, opsym)                     \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t opname##_##type##_nullable_rhs( \
      const type lhs,                                                            \
      const type rhs,                                                            \
      const null_type null_val,                                                  \
      const int8_t null_bool_val) {                                              \
    if (rhs != null_val) {                                                       \
      return lhs opsym rhs;                                                      \
    }                                                                            \
    return null_bool_val;                                                        \
  }

#define DEF_SAFE_DIV_NULLABLE(type, null_type, opname)            \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE type safe_div_##type(   \
      const type lhs, const type rhs, const null_type null_val) { \
    if (lhs != null_val && rhs != null_val && rhs != 0) {         \
      return lhs / rhs;                                           \
    }                                                             \
    return null_val;                                              \
  }

#define DEF_BINARY_NULLABLE_ALL_OPS(type, null_type) \
  DEF_ARITH_NULLABLE(type, null_type, add, +)        \
  DEF_ARITH_NULLABLE(type, null_type, sub, -)        \
  DEF_ARITH_NULLABLE(type, null_type, mul, *)        \
  DEF_ARITH_NULLABLE(type, null_type, div, /)        \
  DEF_SAFE_DIV_NULLABLE(type, null_type, safe_div)   \
  DEF_ARITH_NULLABLE_LHS(type, null_type, add, +)    \
  DEF_ARITH_NULLABLE_LHS(type, null_type, sub, -)    \
  DEF_ARITH_NULLABLE_LHS(type, null_type, mul, *)    \
  DEF_ARITH_NULLABLE_LHS(type, null_type, div, /)    \
  DEF_ARITH_NULLABLE_RHS(type, null_type, add, +)    \
  DEF_ARITH_NULLABLE_RHS(type, null_type, sub, -)    \
  DEF_ARITH_NULLABLE_RHS(type, null_type, mul, *)    \
  DEF_ARITH_NULLABLE_RHS(type, null_type, div, /)    \
  DEF_CMP_NULLABLE(type, null_type, eq, ==)          \
  DEF_CMP_NULLABLE(type, null_type, ne, !=)          \
  DEF_CMP_NULLABLE(type, null_type, lt, <)           \
  DEF_CMP_NULLABLE(type, null_type, gt, >)           \
  DEF_CMP_NULLABLE(type, null_type, le, <=)          \
  DEF_CMP_NULLABLE(type, null_type, ge, >=)          \
  DEF_CMP_NULLABLE_LHS(type, null_type, eq, ==)      \
  DEF_CMP_NULLABLE_LHS(type, null_type, ne, !=)      \
  DEF_CMP_NULLABLE_LHS(type, null_type, lt, <)       \
  DEF_CMP_NULLABLE_LHS(type, null_type, gt, >)       \
  DEF_CMP_NULLABLE_LHS(type, null_type, le, <=)      \
  DEF_CMP_NULLABLE_LHS(type, null_type, ge, >=)      \
  DEF_CMP_NULLABLE_RHS(type, null_type, eq, ==)      \
  DEF_CMP_NULLABLE_RHS(type, null_type, ne, !=)      \
  DEF_CMP_NULLABLE_RHS(type, null_type, lt, <)       \
  DEF_CMP_NULLABLE_RHS(type, null_type, gt, >)       \
  DEF_CMP_NULLABLE_RHS(type, null_type, le, <=)      \
  DEF_CMP_NULLABLE_RHS(type, null_type, ge, >=)

DEF_BINARY_NULLABLE_ALL_OPS(int8_t, int64_t)
DEF_BINARY_NULLABLE_ALL_OPS(int16_t, int64_t)
DEF_BINARY_NULLABLE_ALL_OPS(int32_t, int64_t)
DEF_BINARY_NULLABLE_ALL_OPS(int64_t, int64_t)
DEF_BINARY_NULLABLE_ALL_OPS(float, float)
DEF_BINARY_NULLABLE_ALL_OPS(double, double)
DEF_ARITH_NULLABLE(int8_t, int64_t, mod, %)
DEF_ARITH_NULLABLE(int16_t, int64_t, mod, %)
DEF_ARITH_NULLABLE(int32_t, int64_t, mod, %)
DEF_ARITH_NULLABLE(int64_t, int64_t, mod, %)
DEF_ARITH_NULLABLE_LHS(int8_t, int64_t, mod, %)
DEF_ARITH_NULLABLE_LHS(int16_t, int64_t, mod, %)
DEF_ARITH_NULLABLE_LHS(int32_t, int64_t, mod, %)
DEF_ARITH_NULLABLE_LHS(int64_t, int64_t, mod, %)
DEF_ARITH_NULLABLE_RHS(int8_t, int64_t, mod, %)
DEF_ARITH_NULLABLE_RHS(int16_t, int64_t, mod, %)
DEF_ARITH_NULLABLE_RHS(int32_t, int64_t, mod, %)
DEF_ARITH_NULLABLE_RHS(int64_t, int64_t, mod, %)

#undef DEF_BINARY_NULLABLE_ALL_OPS
#undef DEF_SAFE_DIV_NULLABLE
#undef DEF_CMP_NULLABLE_RHS
#undef DEF_CMP_NULLABLE_LHS
#undef DEF_CMP_NULLABLE
#undef DEF_ARITH_NULLABLE_RHS
#undef DEF_ARITH_NULLABLE_LHS
#undef DEF_ARITH_NULLABLE

#define DEF_MAP_STRING_TO_DATUM(value_type, value_name)                        \
  extern "C" ALWAYS_INLINE DEVICE value_type map_string_to_datum_##value_name( \
      const int32_t string_id,                                                 \
      const int64_t translation_map_handle,                                    \
      const int32_t min_source_id) {                                           \
    const Datum* translation_map =                                             \
        reinterpret_cast<const Datum*>(translation_map_handle);                \
    const Datum& out_datum = translation_map[string_id - min_source_id];       \
    return out_datum.value_name##val;                                          \
  }

DEF_MAP_STRING_TO_DATUM(int8_t, bool)
DEF_MAP_STRING_TO_DATUM(int8_t, tinyint)
DEF_MAP_STRING_TO_DATUM(int16_t, smallint)
DEF_MAP_STRING_TO_DATUM(int32_t, int)
DEF_MAP_STRING_TO_DATUM(int64_t, bigint)
DEF_MAP_STRING_TO_DATUM(float, float)
DEF_MAP_STRING_TO_DATUM(double, double)

#undef DEF_MAP_STRING_TO_DATUM

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
scale_decimal_up(const int64_t operand,
                 const uint64_t scale,
                 const int64_t operand_null_val,
                 const int64_t result_null_val) {
  return operand != operand_null_val ? operand * scale : result_null_val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
scale_decimal_down_nullable(const int64_t operand,
                            const int64_t scale,
                            const int64_t null_val) {
  // rounded scale down of a decimal
  if (operand == null_val) {
    return null_val;
  }

  int64_t tmp = scale >> 1;
  tmp = operand >= 0 ? operand + tmp : operand - tmp;
  return tmp / scale;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
scale_decimal_down_not_nullable(const int64_t operand,
                                const int64_t scale,
                                const int64_t null_val) {
  int64_t tmp = scale >> 1;
  tmp = operand >= 0 ? operand + tmp : operand - tmp;
  return tmp / scale;
}

// Return floor(dividend / divisor).
// Assumes 0 < divisor.
extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t floor_div_lhs(const int64_t dividend,
                                                              const int64_t divisor) {
  return (dividend < 0 ? dividend - (divisor - 1) : dividend) / divisor;
}

// Return floor(dividend / divisor) or NULL if dividend IS NULL.
// Assumes 0 < divisor.
extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
floor_div_nullable_lhs(const int64_t dividend,
                       const int64_t divisor,
                       const int64_t null_val) {
  return dividend == null_val ? null_val : floor_div_lhs(dividend, divisor);
}

#define DEF_UMINUS_NULLABLE(type, null_type)                             \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE type uminus_##type##_nullable( \
      const type operand, const null_type null_val) {                    \
    return operand == null_val ? null_val : -operand;                    \
  }

DEF_UMINUS_NULLABLE(int8_t, int8_t)
DEF_UMINUS_NULLABLE(int16_t, int16_t)
DEF_UMINUS_NULLABLE(int32_t, int32_t)
DEF_UMINUS_NULLABLE(int64_t, int64_t)
DEF_UMINUS_NULLABLE(float, float)
DEF_UMINUS_NULLABLE(double, double)

#undef DEF_UMINUS_NULLABLE

#define DEF_CAST_NULLABLE(from_type, to_type)                                   \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE to_type                               \
      cast_##from_type##_to_##to_type##_nullable(const from_type operand,       \
                                                 const from_type from_null_val, \
                                                 const to_type to_null_val) {   \
    return operand == from_null_val ? to_null_val : operand;                    \
  }

#define DEF_CAST_SCALED_NULLABLE(from_type, to_type)                                   \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE to_type                                      \
      cast_##from_type##_to_##to_type##_scaled_nullable(const from_type operand,       \
                                                        const from_type from_null_val, \
                                                        const to_type to_null_val,     \
                                                        const to_type multiplier) {    \
    return operand == from_null_val ? to_null_val : multiplier * operand;              \
  }

#define DEF_CAST_NULLABLE_BIDIR(type1, type2) \
  DEF_CAST_NULLABLE(type1, type2)             \
  DEF_CAST_NULLABLE(type2, type1)

#define DEF_ROUND_NULLABLE(from_type, to_type)                                  \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE to_type                               \
      cast_##from_type##_to_##to_type##_nullable(const from_type operand,       \
                                                 const from_type from_null_val, \
                                                 const to_type to_null_val) {   \
    return operand == from_null_val                                             \
               ? to_null_val                                                    \
               : static_cast<to_type>(operand + (operand < from_type(0)         \
                                                     ? from_type(-0.5)          \
                                                     : from_type(0.5)));        \
  }

DEF_CAST_NULLABLE_BIDIR(int8_t, int16_t)
DEF_CAST_NULLABLE_BIDIR(int8_t, int32_t)
DEF_CAST_NULLABLE_BIDIR(int8_t, int64_t)
DEF_CAST_NULLABLE_BIDIR(int16_t, int32_t)
DEF_CAST_NULLABLE_BIDIR(int16_t, int64_t)
DEF_CAST_NULLABLE_BIDIR(int32_t, int64_t)
DEF_CAST_NULLABLE_BIDIR(float, double)

DEF_CAST_NULLABLE(int8_t, float)
DEF_CAST_NULLABLE(int16_t, float)
DEF_CAST_NULLABLE(int32_t, float)
DEF_CAST_NULLABLE(int64_t, float)
DEF_CAST_NULLABLE(int8_t, double)
DEF_CAST_NULLABLE(int16_t, double)
DEF_CAST_NULLABLE(int32_t, double)
DEF_CAST_NULLABLE(int64_t, double)

DEF_ROUND_NULLABLE(float, int8_t)
DEF_ROUND_NULLABLE(float, int16_t)
DEF_ROUND_NULLABLE(float, int32_t)
DEF_ROUND_NULLABLE(float, int64_t)
DEF_ROUND_NULLABLE(double, int8_t)
DEF_ROUND_NULLABLE(double, int16_t)
DEF_ROUND_NULLABLE(double, int32_t)
DEF_ROUND_NULLABLE(double, int64_t)

DEF_CAST_NULLABLE(uint8_t, int32_t)
DEF_CAST_NULLABLE(uint16_t, int32_t)
DEF_CAST_SCALED_NULLABLE(int64_t, float)
DEF_CAST_SCALED_NULLABLE(int64_t, double)

#undef DEF_ROUND_NULLABLE
#undef DEF_CAST_NULLABLE_BIDIR
#undef DEF_CAST_SCALED_NULLABLE
#undef DEF_CAST_NULLABLE

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t logical_not(const int8_t operand,
                                                           const int8_t null_val) {
  return operand == null_val ? operand : (operand ? 0 : 1);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t logical_and(const int8_t lhs,
                                                           const int8_t rhs,
                                                           const int8_t null_val) {
  if (lhs == null_val) {
    return rhs == 0 ? rhs : null_val;
  }
  if (rhs == null_val) {
    return lhs == 0 ? lhs : null_val;
  }
  return (lhs && rhs) ? 1 : 0;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t logical_or(const int8_t lhs,
                                                          const int8_t rhs,
                                                          const int8_t null_val) {
  if (lhs == null_val) {
    return rhs == 0 ? null_val : rhs;
  }
  if (rhs == null_val) {
    return lhs == 0 ? null_val : lhs;
  }
  return (lhs || rhs) ? 1 : 0;
}

// aggregator implementations

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint64_t agg_count(uint64_t* agg, const int64_t) {
  return (*agg)++;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void
agg_count_distinct_bitmap(int64_t* agg, const int64_t val, const int64_t min_val) {
  const uint64_t bitmap_idx = val - min_val;
  reinterpret_cast<int8_t*>(*agg)[bitmap_idx >> 3] |= (1 << (bitmap_idx & 7));
}

#ifdef _MSC_VER
#define GPU_RT_STUB NEVER_INLINE
#else
#define GPU_RT_STUB NEVER_INLINE __attribute__((optnone))
#endif

extern "C" GPU_RT_STUB void agg_count_distinct_bitmap_gpu(int64_t*,
                                                          const int64_t,
                                                          const int64_t,
                                                          const int64_t,
                                                          const int64_t,
                                                          const uint64_t,
                                                          const uint64_t) {}

extern "C" RUNTIME_EXPORT NEVER_INLINE void
agg_approximate_count_distinct(int64_t* agg, const int64_t key, const uint32_t b) {
  const uint64_t hash = MurmurHash64A(&key, sizeof(key), 0);
  const uint32_t index = hash >> (64 - b);
  const uint8_t rank = get_rank(hash << b, 64 - b);
  uint8_t* M = reinterpret_cast<uint8_t*>(*agg);
  M[index] = std::max(M[index], rank);
}

extern "C" GPU_RT_STUB void agg_approximate_count_distinct_gpu(int64_t*,
                                                               const int64_t,
                                                               const uint32_t,
                                                               const int64_t,
                                                               const int64_t) {}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t bit_is_set(const int64_t bitset,
                                                          const int64_t val,
                                                          const int64_t min_val,
                                                          const int64_t max_val,
                                                          const int64_t null_val,
                                                          const int8_t null_bool_val) {
  if (val == null_val) {
    return null_bool_val;
  }
  if (val < min_val || val > max_val) {
    return 0;
  }
  if (!bitset) {
    return 0;
  }
  const uint64_t bitmap_idx = val - min_val;
  return (reinterpret_cast<const int8_t*>(bitset))[bitmap_idx >> 3] &
                 (1 << (bitmap_idx & 7))
             ? 1
             : 0;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
compute_int64_t_lower_bound(const int64_t entry_cnt,
                            const int64_t target_value,
                            const int64_t* col_buf) {
  int64_t l = 0;
  int64_t h = entry_cnt - 1;
  while (l < h) {
    int64_t mid = l + (h - l) / 2;
    if (target_value < col_buf[mid]) {
      h = mid;
    } else {
      l = mid + 1;
    }
  }
  return l;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
get_valid_buf_start_pos(const int64_t null_start_pos, const int64_t null_end_pos) {
  return null_start_pos == 0 ? null_end_pos + 1 : 0;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
get_valid_buf_end_pos(const int64_t num_elems,
                      const int64_t null_start_pos,
                      const int64_t null_end_pos) {
  return null_end_pos == num_elems ? null_start_pos : num_elems;
}

template <typename T>
inline int64_t compute_current_row_idx_in_frame(const int64_t num_elems,
                                                const int64_t cur_row_idx,
                                                const T* col_buf,
                                                const int32_t* partition_rowid_buf,
                                                const int64_t* ordered_index_buf,
                                                const T null_val,
                                                const int64_t null_start_pos,
                                                const int64_t null_end_pos) {
  const auto target_value = col_buf[cur_row_idx];
  if (target_value == null_val) {
    for (int64_t target_offset = null_start_pos; target_offset < null_end_pos;
         target_offset++) {
      const auto candidate_offset = partition_rowid_buf[ordered_index_buf[target_offset]];
      if (candidate_offset == cur_row_idx) {
        return target_offset;
      }
    }
  }
  int64_t l = get_valid_buf_start_pos(null_start_pos, null_end_pos);
  int64_t h = get_valid_buf_end_pos(num_elems, null_start_pos, null_end_pos);
  while (l <= h) {
    int64_t mid = l + (h - l) / 2;
    auto row_idx_in_frame = partition_rowid_buf[ordered_index_buf[mid]];
    auto cur_value = col_buf[row_idx_in_frame];
    if (cur_value == target_value) {
      return mid;
    } else if (cur_value < target_value) {
      l = mid + 1;
    } else {
      h = mid - 1;
    }
  }
  return -1;
}

#define DEF_COMPUTE_CURRENT_ROW_IDX_IN_FRAME(value_type)                     \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t                            \
      compute_##value_type##_current_row_idx_in_frame(                       \
          const int64_t num_elems,                                           \
          const int64_t cur_row_idx,                                         \
          const value_type* col_buf,                                         \
          const int32_t* partition_rowid_buf,                                \
          const int64_t* ordered_index_buf,                                  \
          const value_type null_val,                                         \
          const int64_t null_start_pos,                                      \
          const int64_t null_end_pos) {                                      \
    return compute_current_row_idx_in_frame<value_type>(num_elems,           \
                                                        cur_row_idx,         \
                                                        col_buf,             \
                                                        partition_rowid_buf, \
                                                        ordered_index_buf,   \
                                                        null_val,            \
                                                        null_start_pos,      \
                                                        null_end_pos);       \
  }
DEF_COMPUTE_CURRENT_ROW_IDX_IN_FRAME(int8_t)
DEF_COMPUTE_CURRENT_ROW_IDX_IN_FRAME(int16_t)
DEF_COMPUTE_CURRENT_ROW_IDX_IN_FRAME(int32_t)
DEF_COMPUTE_CURRENT_ROW_IDX_IN_FRAME(int64_t)
DEF_COMPUTE_CURRENT_ROW_IDX_IN_FRAME(float)
DEF_COMPUTE_CURRENT_ROW_IDX_IN_FRAME(double)
#undef DEF_COMPUTE_CURRENT_ROW_IDX_IN_FRAME

template <typename TARGET_VAL_TYPE, typename COL_TYPE, typename NULL_TYPE>
inline int64_t compute_lower_bound_from_ordered_partition_index(
    const int64_t num_elems,
    const TARGET_VAL_TYPE target_val,
    const COL_TYPE* col_buf,
    const int32_t* partition_rowid_buf,
    const int64_t* ordered_index_buf,
    const NULL_TYPE null_val,
    const int64_t null_start_offset,
    const int64_t null_end_offset) {
  if (target_val == null_val) {
    return null_start_offset;
  }
  int64_t l = get_valid_buf_start_pos(null_start_offset, null_end_offset);
  int64_t h = get_valid_buf_end_pos(num_elems, null_start_offset, null_end_offset);
  while (l < h) {
    int64_t mid = l + (h - l) / 2;
    if (target_val <= col_buf[partition_rowid_buf[ordered_index_buf[mid]]]) {
      h = mid;
    } else {
      l = mid + 1;
    }
  }
  return l;
}

#define DEF_RANGE_MODE_FRAME_LOWER_BOUND(                                                     \
    target_val_type, col_type, null_type, opname, opsym)                                      \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t                                             \
      range_mode_##target_val_type##_##col_type##_##null_type##_##opname##_frame_lower_bound( \
          const int64_t num_elems,                                                            \
          const target_val_type target_value,                                                 \
          const col_type* col_buf,                                                            \
          const int32_t* partition_rowid_buf,                                                 \
          const int64_t* ordered_index_buf,                                                   \
          const int64_t frame_bound_val,                                                      \
          const null_type null_val,                                                           \
          const int64_t null_start_pos,                                                       \
          const int64_t null_end_pos) {                                                       \
    if (target_value == null_val) {                                                           \
      return null_start_pos;                                                                  \
    }                                                                                         \
    target_val_type new_val = target_value opsym frame_bound_val;                             \
    return compute_lower_bound_from_ordered_partition_index<target_val_type,                  \
                                                            col_type,                         \
                                                            null_type>(                       \
        num_elems,                                                                            \
        new_val,                                                                              \
        col_buf,                                                                              \
        partition_rowid_buf,                                                                  \
        ordered_index_buf,                                                                    \
        null_val,                                                                             \
        null_start_pos,                                                                       \
        null_end_pos);                                                                        \
  }
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int8_t, int8_t, int8_t, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int8_t, int8_t, int8_t, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int16_t, int16_t, int16_t, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int16_t, int16_t, int16_t, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int16_t, int16_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int16_t, int16_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int32_t, int32_t, int32_t, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int32_t, int32_t, int32_t, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int32_t, int32_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int32_t, int32_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int64_t, int16_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int64_t, int16_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int64_t, int32_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int64_t, int32_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int64_t, int64_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(int64_t, int64_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(float, float, float, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(float, float, float, sub, -)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(double, double, double, add, +)
DEF_RANGE_MODE_FRAME_LOWER_BOUND(double, double, double, sub, -)
#undef DEF_RANGE_MODE_FRAME_LOWER_BOUND

template <typename TARGET_VAL_TYPE, typename COL_TYPE, typename NULL_TYPE>
inline int64_t compute_upper_bound_from_ordered_partition_index(
    const int64_t num_elems,
    const TARGET_VAL_TYPE target_val,
    const COL_TYPE* col_buf,
    const int32_t* partition_rowid_buf,
    const int64_t* ordered_index_buf,
    const NULL_TYPE null_val,
    const int64_t null_start_offset,
    const int64_t null_end_offset) {
  if (target_val == null_val) {
    return null_end_offset;
  }
  int64_t l = get_valid_buf_start_pos(null_start_offset, null_end_offset);
  int64_t h = get_valid_buf_end_pos(num_elems, null_start_offset, null_end_offset);
  while (l < h) {
    int64_t mid = l + (h - l) / 2;
    if (target_val >= col_buf[partition_rowid_buf[ordered_index_buf[mid]]]) {
      l = mid + 1;
    } else {
      h = mid;
    }
  }
  return l;
}

#define DEF_RANGE_MODE_FRAME_UPPER_BOUND(                                                     \
    target_val_type, col_type, null_type, opname, opsym)                                      \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t                                             \
      range_mode_##target_val_type##_##col_type##_##null_type##_##opname##_frame_upper_bound( \
          const int64_t num_elems,                                                            \
          const target_val_type target_value,                                                 \
          const col_type* col_buf,                                                            \
          const int32_t* partition_rowid_buf,                                                 \
          const int64_t* ordered_index_buf,                                                   \
          const int64_t frame_bound_val,                                                      \
          const null_type null_val,                                                           \
          const int64_t null_start_pos,                                                       \
          const int64_t null_end_pos) {                                                       \
    if (target_value == null_val) {                                                           \
      return null_end_pos;                                                                    \
    }                                                                                         \
    target_val_type new_val = target_value opsym frame_bound_val;                             \
    return compute_upper_bound_from_ordered_partition_index<target_val_type,                  \
                                                            col_type,                         \
                                                            null_type>(                       \
        num_elems,                                                                            \
        new_val,                                                                              \
        col_buf,                                                                              \
        partition_rowid_buf,                                                                  \
        ordered_index_buf,                                                                    \
        null_val,                                                                             \
        null_start_pos,                                                                       \
        null_end_pos);                                                                        \
  }
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int8_t, int8_t, int8_t, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int8_t, int8_t, int8_t, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int16_t, int16_t, int16_t, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int16_t, int16_t, int16_t, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int16_t, int16_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int16_t, int16_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int32_t, int32_t, int32_t, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int32_t, int32_t, int32_t, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int32_t, int32_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int32_t, int32_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int64_t, int16_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int64_t, int16_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int64_t, int32_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int64_t, int32_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int64_t, int64_t, int64_t, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(int64_t, int64_t, int64_t, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(float, float, float, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(float, float, float, sub, -)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(double, double, double, add, +)
DEF_RANGE_MODE_FRAME_UPPER_BOUND(double, double, double, sub, -)
#undef DEF_RANGE_MODE_FRAME_UPPER_BOUND

template <typename COL_TYPE, typename LOGICAL_TYPE>
inline LOGICAL_TYPE get_value_in_window_frame(const int64_t target_row_idx_in_frame,
                                              const int64_t frame_start_offset,
                                              const int64_t frame_end_offset,
                                              const COL_TYPE* col_buf,
                                              const int32_t* partition_rowid_buf,
                                              const int64_t* ordered_index_buf,
                                              const LOGICAL_TYPE logical_null_val,
                                              const LOGICAL_TYPE col_null_val) {
  if (target_row_idx_in_frame < frame_start_offset ||
      target_row_idx_in_frame >= frame_end_offset) {
    return logical_null_val;
  }
  const auto target_offset =
      partition_rowid_buf[ordered_index_buf[target_row_idx_in_frame]];
  LOGICAL_TYPE target_val = col_buf[target_offset];
  if (target_val == col_null_val) {
    return logical_null_val;
  }
  return target_val;
}

#define DEF_GET_VALUE_IN_FRAME(col_type, logical_type)                                \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE logical_type                                \
      get_##col_type##_value_##logical_type##_type_in_frame(                          \
          const int64_t target_row_idx_in_frame,                                      \
          const int64_t frame_start_offset,                                           \
          const int64_t frame_end_offset,                                             \
          const col_type* col_buf,                                                    \
          const int32_t* partition_rowid_buf,                                         \
          const int64_t* ordered_index_buf,                                           \
          const logical_type logical_null_val,                                        \
          const logical_type col_null_val) {                                          \
    return get_value_in_window_frame<col_type, logical_type>(target_row_idx_in_frame, \
                                                             frame_start_offset,      \
                                                             frame_end_offset,        \
                                                             col_buf,                 \
                                                             partition_rowid_buf,     \
                                                             ordered_index_buf,       \
                                                             logical_null_val,        \
                                                             col_null_val);           \
  }
DEF_GET_VALUE_IN_FRAME(int8_t, int8_t)
DEF_GET_VALUE_IN_FRAME(int8_t, int16_t)
DEF_GET_VALUE_IN_FRAME(int8_t, int32_t)
DEF_GET_VALUE_IN_FRAME(int8_t, int64_t)
DEF_GET_VALUE_IN_FRAME(int16_t, int16_t)
DEF_GET_VALUE_IN_FRAME(int16_t, int32_t)
DEF_GET_VALUE_IN_FRAME(int16_t, int64_t)
DEF_GET_VALUE_IN_FRAME(int32_t, int32_t)
DEF_GET_VALUE_IN_FRAME(int32_t, int64_t)
DEF_GET_VALUE_IN_FRAME(int64_t, int64_t)
DEF_GET_VALUE_IN_FRAME(float, float)
DEF_GET_VALUE_IN_FRAME(double, double)
#undef DEF_GET_VALUE_IN_FRAME

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t encode_date(int64_t decoded_val,
                                                            int64_t null_val,
                                                            int64_t multiplier) {
  return decoded_val == null_val ? decoded_val : decoded_val * multiplier;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
compute_row_mode_start_index_sub(int64_t candidate_index,
                                 int64_t current_partition_start_offset,
                                 int64_t frame_bound) {
  int64_t index = candidate_index - current_partition_start_offset - frame_bound;
  return index < 0 ? 0 : index;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
compute_row_mode_start_index_add(int64_t candidate_index,
                                 int64_t current_partition_start_offset,
                                 int64_t frame_bound,
                                 int64_t num_current_partition_elem) {
  int64_t index = candidate_index - current_partition_start_offset + frame_bound;
  return index >= num_current_partition_elem ? num_current_partition_elem : index;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
compute_row_mode_end_index_sub(int64_t candidate_index,
                               int64_t current_partition_start_offset,
                               int64_t frame_bound) {
  int64_t index = candidate_index - current_partition_start_offset - frame_bound;
  return index < 0 ? 0 : index + 1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
compute_row_mode_end_index_add(int64_t candidate_index,
                               int64_t current_partition_start_offset,
                               int64_t frame_bound,
                               int64_t num_current_partition_elem) {
  int64_t index = candidate_index - current_partition_start_offset + frame_bound;
  return index >= num_current_partition_elem ? num_current_partition_elem : index + 1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t* get_integer_aggregation_tree(
    int64_t** aggregation_trees,
    size_t partition_idx) {
  return aggregation_trees[partition_idx];
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE double* get_double_aggregation_tree(
    int64_t** aggregation_trees,
    size_t partition_idx) {
  double** casted_aggregation_trees = reinterpret_cast<double**>(aggregation_trees);
  return casted_aggregation_trees[partition_idx];
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE SumAndCountPair<int64_t>*
get_integer_derived_aggregation_tree(int64_t** aggregation_trees, size_t partition_idx) {
  SumAndCountPair<int64_t>** casted_aggregation_trees =
      reinterpret_cast<SumAndCountPair<int64_t>**>(aggregation_trees);
  return casted_aggregation_trees[partition_idx];
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE SumAndCountPair<double>*
get_double_derived_aggregation_tree(int64_t** aggregation_trees, size_t partition_idx) {
  SumAndCountPair<double>** casted_aggregation_trees =
      reinterpret_cast<SumAndCountPair<double>**>(aggregation_trees);
  return casted_aggregation_trees[partition_idx];
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE size_t
getStartOffsetForSegmentTreeTraversal(size_t level, size_t tree_fanout) {
  size_t offset = 0;
  for (size_t i = 0; i < level; i++) {
    offset += pow(tree_fanout, i);
  }
  return offset;
}
namespace {
enum class AggFuncType { MIN, MAX, SUM };

template <AggFuncType AGG_FUNC_TYPE, typename AGG_TYPE>
inline AGG_TYPE agg_func(AGG_TYPE const lhs, AGG_TYPE const rhs) {
  if constexpr (AGG_FUNC_TYPE == AggFuncType::MIN) {
    return std::min(lhs, rhs);
  } else if constexpr (AGG_FUNC_TYPE == AggFuncType::MAX) {
    return std::max(lhs, rhs);
  } else {
    return lhs + rhs;
  }
}
}  // namespace

template <AggFuncType AGG_FUNC_TYPE, typename AGG_TYPE>
inline AGG_TYPE compute_window_func_via_aggregation_tree(
    AGG_TYPE* aggregation_tree_for_partition,
    size_t query_range_start_idx,
    size_t query_range_end_idx,
    size_t leaf_level,
    size_t tree_fanout,
    AGG_TYPE init_val,
    AGG_TYPE invalid_val,
    AGG_TYPE null_val) {
  size_t leaf_start_idx = getStartOffsetForSegmentTreeTraversal(leaf_level, tree_fanout);
  size_t begin = leaf_start_idx + query_range_start_idx;
  size_t end = leaf_start_idx + query_range_end_idx;
  AGG_TYPE res = init_val;
  bool all_nulls = true;
  for (int level = leaf_level; level >= 0; level--) {
    size_t parentBegin = begin / tree_fanout;
    size_t parentEnd = (end - 1) / tree_fanout;
    if (parentBegin == parentEnd) {
      for (size_t pos = begin; pos < end; pos++) {
        if (aggregation_tree_for_partition[pos] != null_val) {
          all_nulls = false;
          res = agg_func<AGG_FUNC_TYPE>(res, aggregation_tree_for_partition[pos]);
        }
      }
      return all_nulls ? null_val : res;
    } else if (parentBegin > parentEnd) {
      return null_val;
    }
    size_t group_begin = (parentBegin * tree_fanout) + 1;
    if (begin != group_begin) {
      size_t limit = (parentBegin * tree_fanout) + tree_fanout + 1;
      for (size_t pos = begin; pos < limit; pos++) {
        if (aggregation_tree_for_partition[pos] != null_val) {
          all_nulls = false;
          res = agg_func<AGG_FUNC_TYPE>(res, aggregation_tree_for_partition[pos]);
        }
      }
      parentBegin++;
    }
    size_t group_end = (parentEnd * tree_fanout) + 1;
    if (end != group_end) {
      for (size_t pos = group_end; pos < end; pos++) {
        if (aggregation_tree_for_partition[pos] != null_val) {
          all_nulls = false;
          res = agg_func<AGG_FUNC_TYPE>(res, aggregation_tree_for_partition[pos]);
        }
      }
    }
    begin = parentBegin;
    end = parentEnd;
  }
  return invalid_val;
}

#define DEF_SEARCH_AGGREGATION_TREE(agg_value_type)                                      \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE agg_value_type                                 \
      search_##agg_value_type##_aggregation_tree(                                        \
          agg_value_type* aggregated_tree_for_partition,                                 \
          size_t query_range_start_idx,                                                  \
          size_t query_range_end_idx,                                                    \
          size_t leaf_level,                                                             \
          size_t tree_fanout,                                                            \
          bool decimal_type,                                                             \
          size_t scale,                                                                  \
          agg_value_type invalid_val,                                                    \
          agg_value_type null_val,                                                       \
          int32_t agg_type) {                                                            \
    if (!aggregated_tree_for_partition || query_range_start_idx > query_range_end_idx) { \
      return null_val;                                                                   \
    }                                                                                    \
    switch (agg_type) {                                                                  \
      case 1: {                                                                          \
        return compute_window_func_via_aggregation_tree<AggFuncType::MIN>(               \
            aggregated_tree_for_partition,                                               \
            query_range_start_idx,                                                       \
            query_range_end_idx,                                                         \
            leaf_level,                                                                  \
            tree_fanout,                                                                 \
            std::numeric_limits<agg_value_type>::max(),                                  \
            invalid_val,                                                                 \
            null_val);                                                                   \
      }                                                                                  \
      case 2: {                                                                          \
        return compute_window_func_via_aggregation_tree<AggFuncType::MAX>(               \
            aggregated_tree_for_partition,                                               \
            query_range_start_idx,                                                       \
            query_range_end_idx,                                                         \
            leaf_level,                                                                  \
            tree_fanout,                                                                 \
            std::numeric_limits<agg_value_type>::lowest(),                               \
            invalid_val,                                                                 \
            null_val);                                                                   \
      }                                                                                  \
      default: {                                                                         \
        return compute_window_func_via_aggregation_tree<AggFuncType::SUM>(               \
            aggregated_tree_for_partition,                                               \
            query_range_start_idx,                                                       \
            query_range_end_idx,                                                         \
            leaf_level,                                                                  \
            tree_fanout,                                                                 \
            static_cast<agg_value_type>(0),                                              \
            invalid_val,                                                                 \
            null_val);                                                                   \
      }                                                                                  \
    }                                                                                    \
  }

DEF_SEARCH_AGGREGATION_TREE(int64_t)
DEF_SEARCH_AGGREGATION_TREE(double)
#undef DEF_SEARCH_AGGREGATION_TREE

template <typename AGG_VALUE_TYPE>
inline void compute_derived_aggregates(
    SumAndCountPair<AGG_VALUE_TYPE>* aggregation_tree_for_partition,
    SumAndCountPair<AGG_VALUE_TYPE>& res,
    size_t query_range_start_idx,
    size_t query_range_end_idx,
    size_t leaf_level,
    size_t tree_fanout,
    AGG_VALUE_TYPE invalid_val,
    AGG_VALUE_TYPE null_val) {
  size_t leaf_start_idx = getStartOffsetForSegmentTreeTraversal(leaf_level, tree_fanout);
  size_t begin = leaf_start_idx + query_range_start_idx;
  size_t end = leaf_start_idx + query_range_end_idx;
  SumAndCountPair<AGG_VALUE_TYPE> null_res{null_val, 0};
  SumAndCountPair<AGG_VALUE_TYPE> invalid_res{invalid_val, 0};
  bool all_nulls = true;
  for (int level = leaf_level; level >= 0; level--) {
    size_t parentBegin = begin / tree_fanout;
    size_t parentEnd = (end - 1) / tree_fanout;
    if (parentBegin == parentEnd) {
      for (size_t pos = begin; pos < end; pos++) {
        if (aggregation_tree_for_partition[pos].sum != null_val) {
          all_nulls = false;
          res.sum += aggregation_tree_for_partition[pos].sum;
          res.count += aggregation_tree_for_partition[pos].count;
        }
      }
      if (all_nulls) {
        res = null_res;
      }
      return;
    } else if (parentBegin > parentEnd) {
      res = null_res;
      return;
    }
    size_t group_begin = (parentBegin * tree_fanout) + 1;
    if (begin != group_begin) {
      size_t limit = (parentBegin * tree_fanout) + tree_fanout + 1;
      for (size_t pos = begin; pos < limit; pos++) {
        if (aggregation_tree_for_partition[pos].sum != null_val) {
          all_nulls = false;
          res.sum += aggregation_tree_for_partition[pos].sum;
          res.count += aggregation_tree_for_partition[pos].count;
        }
      }
      parentBegin++;
    }
    size_t group_end = (parentEnd * tree_fanout) + 1;
    if (end != group_end) {
      for (size_t pos = group_end; pos < end; pos++) {
        if (aggregation_tree_for_partition[pos].sum != null_val) {
          all_nulls = false;
          res.sum += aggregation_tree_for_partition[pos].sum;
          res.count += aggregation_tree_for_partition[pos].count;
        }
      }
    }
    begin = parentBegin;
    end = parentEnd;
  }
  res = invalid_res;
  return;
}

#define DEF_SEARCH_DERIVED_AGGREGATION_TREE(agg_value_type)                              \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE double                                         \
      search_##agg_value_type##_derived_aggregation_tree(                                \
          SumAndCountPair<agg_value_type>* aggregated_tree_for_partition,                \
          size_t query_range_start_idx,                                                  \
          size_t query_range_end_idx,                                                    \
          size_t leaf_level,                                                             \
          size_t tree_fanout,                                                            \
          bool decimal_type,                                                             \
          size_t scale,                                                                  \
          agg_value_type invalid_val,                                                    \
          agg_value_type null_val,                                                       \
          int32_t agg_type) {                                                            \
    if (!aggregated_tree_for_partition || query_range_start_idx > query_range_end_idx) { \
      return null_val;                                                                   \
    }                                                                                    \
    SumAndCountPair<agg_value_type> res{0, 0};                                           \
    compute_derived_aggregates<agg_value_type>(aggregated_tree_for_partition,            \
                                               res,                                      \
                                               query_range_start_idx,                    \
                                               query_range_end_idx,                      \
                                               leaf_level,                               \
                                               tree_fanout,                              \
                                               invalid_val,                              \
                                               null_val);                                \
    if (res.sum == null_val) {                                                           \
      return null_val;                                                                   \
    } else if (res.count > 0) {                                                          \
      if (decimal_type) {                                                                \
        return (static_cast<double>(res.sum) / pow(10, scale)) / res.count;              \
      }                                                                                  \
      return (static_cast<double>(res.sum)) / res.count;                                 \
    } else {                                                                             \
      return invalid_val;                                                                \
    }                                                                                    \
  }

DEF_SEARCH_DERIVED_AGGREGATION_TREE(int64_t)
DEF_SEARCH_DERIVED_AGGREGATION_TREE(double)
#undef DEF_SEARCH_DERIVED_AGGREGATION_TREE

#define DEF_HANDLE_NULL_FOR_WINDOW_FRAMING_AGG(agg_type, null_type)            \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE agg_type                             \
      handle_null_val_##agg_type##_##null_type##_window_framing_agg(           \
          agg_type res, null_type agg_null_val, agg_type input_col_null_val) { \
    if (res == agg_null_val) {                                                 \
      return input_col_null_val;                                               \
    }                                                                          \
    return res;                                                                \
  }
DEF_HANDLE_NULL_FOR_WINDOW_FRAMING_AGG(double, int64_t)
DEF_HANDLE_NULL_FOR_WINDOW_FRAMING_AGG(double, double)
DEF_HANDLE_NULL_FOR_WINDOW_FRAMING_AGG(int64_t, int64_t)
#undef DEF_HANDLE_NULL_FOR_WINDOW_FRAMING_AGG

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t agg_sum(int64_t* agg, const int64_t val) {
  const auto old = *agg;
  *agg += val;
  return old;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_max(int64_t* agg, const int64_t val) {
  *agg = std::max(*agg, val);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_min(int64_t* agg, const int64_t val) {
  *agg = std::min(*agg, val);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_id(int64_t* agg, const int64_t val) {
  *agg = val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t* agg_id_varlen(int8_t* varlen_buffer,
                                                              const int64_t offset,
                                                              const int8_t* value,
                                                              const int64_t size_bytes) {
  for (auto i = 0; i < size_bytes; i++) {
    varlen_buffer[offset + i] = value[i];
  }
  return &varlen_buffer[offset];
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t
checked_single_agg_id(int64_t* agg, const int64_t val, const int64_t null_val) {
  if (val == null_val) {
    return 0;
  }

  if (*agg == val) {
    return 0;
  } else if (*agg == null_val) {
    *agg = val;
    return 0;
  } else {
    // see Execute::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES
    return 15;
  }
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_count_distinct_bitmap_skip_val(
    int64_t* agg,
    const int64_t val,
    const int64_t min_val,
    const int64_t skip_val) {
  if (val != skip_val) {
    agg_count_distinct_bitmap(agg, val, min_val);
  }
}

extern "C" GPU_RT_STUB void agg_count_distinct_bitmap_skip_val_gpu(int64_t*,
                                                                   const int64_t,
                                                                   const int64_t,
                                                                   const int64_t,
                                                                   const int64_t,
                                                                   const int64_t,
                                                                   const uint64_t,
                                                                   const uint64_t) {}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint32_t agg_count_int32(uint32_t* agg,
                                                                 const int32_t) {
  return (*agg)++;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t agg_sum_int32(int32_t* agg,
                                                              const int32_t val) {
  const auto old = *agg;
  *agg += val;
  return old;
}

#define DEF_AGG_MAX_INT(n)                                                            \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_max_int##n(int##n##_t* agg,        \
                                                              const int##n##_t val) { \
    *agg = std::max(*agg, val);                                                       \
  }

DEF_AGG_MAX_INT(32)
DEF_AGG_MAX_INT(16)
DEF_AGG_MAX_INT(8)
#undef DEF_AGG_MAX_INT

#define DEF_AGG_MIN_INT(n)                                                            \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_min_int##n(int##n##_t* agg,        \
                                                              const int##n##_t val) { \
    *agg = std::min(*agg, val);                                                       \
  }

DEF_AGG_MIN_INT(32)
DEF_AGG_MIN_INT(16)
DEF_AGG_MIN_INT(8)
#undef DEF_AGG_MIN_INT

#define DEF_AGG_ID_INT(n)                                                            \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_id_int##n(int##n##_t* agg,        \
                                                             const int##n##_t val) { \
    *agg = val;                                                                      \
  }

#define DEF_CHECKED_SINGLE_AGG_ID_INT(n)                                        \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t checked_single_agg_id_int##n( \
      int##n##_t* agg, const int##n##_t val, const int##n##_t null_val) {       \
    if (val == null_val) {                                                      \
      return 0;                                                                 \
    }                                                                           \
    if (*agg == val) {                                                          \
      return 0;                                                                 \
    } else if (*agg == null_val) {                                              \
      *agg = val;                                                               \
      return 0;                                                                 \
    } else {                                                                    \
      /* see Execute::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES*/                  \
      return 15;                                                                \
    }                                                                           \
  }

DEF_AGG_ID_INT(32)
DEF_AGG_ID_INT(16)
DEF_AGG_ID_INT(8)

DEF_CHECKED_SINGLE_AGG_ID_INT(32)
DEF_CHECKED_SINGLE_AGG_ID_INT(16)
DEF_CHECKED_SINGLE_AGG_ID_INT(8)

#undef DEF_AGG_ID_INT
#undef DEF_CHECKED_SINGLE_AGG_ID_INT

#define DEF_WRITE_PROJECTION_INT(n)                                     \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE void write_projection_int##n( \
      int8_t* slot_ptr, const int##n##_t val, const int64_t init_val) { \
    if (val != init_val) {                                              \
      *reinterpret_cast<int##n##_t*>(slot_ptr) = val;                   \
    }                                                                   \
  }

DEF_WRITE_PROJECTION_INT(64)
DEF_WRITE_PROJECTION_INT(32)
#undef DEF_WRITE_PROJECTION_INT

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t agg_sum_skip_val(int64_t* agg,
                                                                 const int64_t val,
                                                                 const int64_t skip_val) {
  const auto old = *agg;
  if (val != skip_val) {
    if (old != skip_val) {
      return agg_sum(agg, val);
    } else {
      *agg = val;
    }
  }
  return old;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t
agg_sum_int32_skip_val(int32_t* agg, const int32_t val, const int32_t skip_val) {
  const auto old = *agg;
  if (val != skip_val) {
    if (old != skip_val) {
      return agg_sum_int32(agg, val);
    } else {
      *agg = val;
    }
  }
  return old;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint64_t
agg_count_skip_val(uint64_t* agg, const int64_t val, const int64_t skip_val) {
  if (val != skip_val) {
    return agg_count(agg, val);
  }
  return *agg;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint32_t
agg_count_int32_skip_val(uint32_t* agg, const int32_t val, const int32_t skip_val) {
  if (val != skip_val) {
    return agg_count_int32(agg, val);
  }
  return *agg;
}

#define DEF_SKIP_AGG_ADD(base_agg_func)                                  \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE void base_agg_func##_skip_val( \
      DATA_T* agg, const DATA_T val, const DATA_T skip_val) {            \
    if (val != skip_val) {                                               \
      base_agg_func(agg, val);                                           \
    }                                                                    \
  }

#define DEF_SKIP_AGG(base_agg_func)                                      \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE void base_agg_func##_skip_val( \
      DATA_T* agg, const DATA_T val, const DATA_T skip_val) {            \
    if (val != skip_val) {                                               \
      const DATA_T old_agg = *agg;                                       \
      if (old_agg != skip_val) {                                         \
        base_agg_func(agg, val);                                         \
      } else {                                                           \
        *agg = val;                                                      \
      }                                                                  \
    }                                                                    \
  }

#define DATA_T int64_t
DEF_SKIP_AGG(agg_max)
DEF_SKIP_AGG(agg_min)
#undef DATA_T

#define DATA_T int32_t
DEF_SKIP_AGG(agg_max_int32)
DEF_SKIP_AGG(agg_min_int32)
#undef DATA_T

#define DATA_T int16_t
DEF_SKIP_AGG(agg_max_int16)
DEF_SKIP_AGG(agg_min_int16)
#undef DATA_T

#define DATA_T int8_t
DEF_SKIP_AGG(agg_max_int8)
DEF_SKIP_AGG(agg_min_int8)
#undef DATA_T

#undef DEF_SKIP_AGG_ADD
#undef DEF_SKIP_AGG

// TODO(alex): fix signature

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint64_t agg_count_double(uint64_t* agg,
                                                                  const double val) {
  return (*agg)++;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_sum_double(int64_t* agg,
                                                            const double val) {
  const auto r = *reinterpret_cast<const double*>(agg) + val;
  *agg = *reinterpret_cast<const int64_t*>(may_alias_ptr(&r));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_max_double(int64_t* agg,
                                                            const double val) {
  const auto r = std::max(*reinterpret_cast<const double*>(agg), val);
  *agg = *(reinterpret_cast<const int64_t*>(may_alias_ptr(&r)));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_min_double(int64_t* agg,
                                                            const double val) {
  const auto r = std::min(*reinterpret_cast<const double*>(agg), val);
  *agg = *(reinterpret_cast<const int64_t*>(may_alias_ptr(&r)));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_id_double(int64_t* agg,
                                                           const double val) {
  *agg = *(reinterpret_cast<const int64_t*>(may_alias_ptr(&val)));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t
checked_single_agg_id_double(int64_t* agg, const double val, const double null_val) {
  if (val == null_val) {
    return 0;
  }

  if (*agg == *(reinterpret_cast<const int64_t*>(may_alias_ptr(&val)))) {
    return 0;
  } else if (*agg == *(reinterpret_cast<const int64_t*>(may_alias_ptr(&null_val)))) {
    *agg = *(reinterpret_cast<const int64_t*>(may_alias_ptr(&val)));
    return 0;
  } else {
    // see Execute::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES
    return 15;
  }
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint32_t agg_count_float(uint32_t* agg,
                                                                 const float val) {
  return (*agg)++;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_sum_float(int32_t* agg,
                                                           const float val) {
  const auto r = *reinterpret_cast<const float*>(agg) + val;
  *agg = *reinterpret_cast<const int32_t*>(may_alias_ptr(&r));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_max_float(int32_t* agg,
                                                           const float val) {
  const auto r = std::max(*reinterpret_cast<const float*>(agg), val);
  *agg = *(reinterpret_cast<const int32_t*>(may_alias_ptr(&r)));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_min_float(int32_t* agg,
                                                           const float val) {
  const auto r = std::min(*reinterpret_cast<const float*>(agg), val);
  *agg = *(reinterpret_cast<const int32_t*>(may_alias_ptr(&r)));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void agg_id_float(int32_t* agg, const float val) {
  *agg = *(reinterpret_cast<const int32_t*>(may_alias_ptr(&val)));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t
checked_single_agg_id_float(int32_t* agg, const float val, const float null_val) {
  if (val == null_val) {
    return 0;
  }

  if (*agg == *(reinterpret_cast<const int32_t*>(may_alias_ptr(&val)))) {
    return 0;
  } else if (*agg == *(reinterpret_cast<const int32_t*>(may_alias_ptr(&null_val)))) {
    *agg = *(reinterpret_cast<const int32_t*>(may_alias_ptr(&val)));
    return 0;
  } else {
    // see Execute::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES
    return 15;
  }
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint64_t
agg_count_double_skip_val(uint64_t* agg, const double val, const double skip_val) {
  if (val != skip_val) {
    return agg_count_double(agg, val);
  }
  return *agg;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint32_t
agg_count_float_skip_val(uint32_t* agg, const float val, const float skip_val) {
  if (val != skip_val) {
    return agg_count_float(agg, val);
  }
  return *agg;
}

#define DEF_SKIP_AGG_ADD(base_agg_func)                                  \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE void base_agg_func##_skip_val( \
      ADDR_T* agg, const DATA_T val, const DATA_T skip_val) {            \
    if (val != skip_val) {                                               \
      base_agg_func(agg, val);                                           \
    }                                                                    \
  }

#define DEF_SKIP_AGG(base_agg_func)                                                \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE void base_agg_func##_skip_val(           \
      ADDR_T* agg, const DATA_T val, const DATA_T skip_val) {                      \
    if (val != skip_val) {                                                         \
      const ADDR_T old_agg = *agg;                                                 \
      if (old_agg != *reinterpret_cast<const ADDR_T*>(may_alias_ptr(&skip_val))) { \
        base_agg_func(agg, val);                                                   \
      } else {                                                                     \
        *agg = *reinterpret_cast<const ADDR_T*>(may_alias_ptr(&val));              \
      }                                                                            \
    }                                                                              \
  }

#define DATA_T double
#define ADDR_T int64_t
DEF_SKIP_AGG(agg_sum_double)
DEF_SKIP_AGG(agg_max_double)
DEF_SKIP_AGG(agg_min_double)
#undef ADDR_T
#undef DATA_T

#define DATA_T float
#define ADDR_T int32_t
DEF_SKIP_AGG(agg_sum_float)
DEF_SKIP_AGG(agg_max_float)
DEF_SKIP_AGG(agg_min_float)
#undef ADDR_T
#undef DATA_T

#undef DEF_SKIP_AGG_ADD
#undef DEF_SKIP_AGG

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t decimal_floor(const int64_t x,
                                                              const int64_t scale) {
  if (x >= 0) {
    return x / scale * scale;
  }
  if (!(x % scale)) {
    return x;
  }
  return x / scale * scale - scale;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t decimal_ceil(const int64_t x,
                                                             const int64_t scale) {
  return decimal_floor(x, scale) + (x % scale ? scale : 0);
}

// Shared memory aggregators. Should never be called,
// real implementations are in cuda_mapd_rt.cu.
#define DEF_SHARED_AGG_RET_STUBS(base_agg_func)                                     \
  extern "C" GPU_RT_STUB uint64_t base_agg_func##_shared(uint64_t* agg,             \
                                                         const int64_t val) {       \
    return 0;                                                                       \
  }                                                                                 \
                                                                                    \
  extern "C" GPU_RT_STUB uint64_t base_agg_func##_skip_val_shared(                  \
      uint64_t* agg, const int64_t val, const int64_t skip_val) {                   \
    return 0;                                                                       \
  }                                                                                 \
  extern "C" GPU_RT_STUB uint32_t base_agg_func##_int32_shared(uint32_t* agg,       \
                                                               const int32_t val) { \
    return 0;                                                                       \
  }                                                                                 \
                                                                                    \
  extern "C" GPU_RT_STUB uint32_t base_agg_func##_int32_skip_val_shared(            \
      uint32_t* agg, const int32_t val, const int32_t skip_val) {                   \
    return 0;                                                                       \
  }                                                                                 \
                                                                                    \
  extern "C" GPU_RT_STUB uint64_t base_agg_func##_double_shared(uint64_t* agg,      \
                                                                const double val) { \
    return 0;                                                                       \
  }                                                                                 \
                                                                                    \
  extern "C" GPU_RT_STUB uint64_t base_agg_func##_double_skip_val_shared(           \
      uint64_t* agg, const double val, const double skip_val) {                     \
    return 0;                                                                       \
  }                                                                                 \
  extern "C" GPU_RT_STUB uint32_t base_agg_func##_float_shared(uint32_t* agg,       \
                                                               const float val) {   \
    return 0;                                                                       \
  }                                                                                 \
                                                                                    \
  extern "C" GPU_RT_STUB uint32_t base_agg_func##_float_skip_val_shared(            \
      uint32_t* agg, const float val, const float skip_val) {                       \
    return 0;                                                                       \
  }

#define DEF_SHARED_AGG_STUBS(base_agg_func)                                              \
  extern "C" GPU_RT_STUB void base_agg_func##_shared(int64_t* agg, const int64_t val) {} \
                                                                                         \
  extern "C" GPU_RT_STUB void base_agg_func##_skip_val_shared(                           \
      int64_t* agg, const int64_t val, const int64_t skip_val) {}                        \
  extern "C" GPU_RT_STUB void base_agg_func##_int32_shared(int32_t* agg,                 \
                                                           const int32_t val) {}         \
  extern "C" GPU_RT_STUB void base_agg_func##_int16_shared(int16_t* agg,                 \
                                                           const int16_t val) {}         \
  extern "C" GPU_RT_STUB void base_agg_func##_int8_shared(int8_t* agg,                   \
                                                          const int8_t val) {}           \
                                                                                         \
  extern "C" GPU_RT_STUB void base_agg_func##_int32_skip_val_shared(                     \
      int32_t* agg, const int32_t val, const int32_t skip_val) {}                        \
                                                                                         \
  extern "C" GPU_RT_STUB void base_agg_func##_double_shared(int64_t* agg,                \
                                                            const double val) {}         \
                                                                                         \
  extern "C" GPU_RT_STUB void base_agg_func##_double_skip_val_shared(                    \
      int64_t* agg, const double val, const double skip_val) {}                          \
  extern "C" GPU_RT_STUB void base_agg_func##_float_shared(int32_t* agg,                 \
                                                           const float val) {}           \
                                                                                         \
  extern "C" GPU_RT_STUB void base_agg_func##_float_skip_val_shared(                     \
      int32_t* agg, const float val, const float skip_val) {}

DEF_SHARED_AGG_RET_STUBS(agg_count)
DEF_SHARED_AGG_STUBS(agg_max)
DEF_SHARED_AGG_STUBS(agg_min)
DEF_SHARED_AGG_STUBS(agg_id)

extern "C" GPU_RT_STUB int8_t* agg_id_varlen_shared(int8_t* varlen_buffer,
                                                    const int64_t offset,
                                                    const int8_t* value,
                                                    const int64_t size_bytes) {
  return nullptr;
}

extern "C" GPU_RT_STUB int32_t checked_single_agg_id_shared(int64_t* agg,
                                                            const int64_t val,
                                                            const int64_t null_val) {
  return 0;
}

extern "C" GPU_RT_STUB int32_t
checked_single_agg_id_int32_shared(int32_t* agg,
                                   const int32_t val,
                                   const int32_t null_val) {
  return 0;
}
extern "C" GPU_RT_STUB int32_t
checked_single_agg_id_int16_shared(int16_t* agg,
                                   const int16_t val,
                                   const int16_t null_val) {
  return 0;
}
extern "C" GPU_RT_STUB int32_t checked_single_agg_id_int8_shared(int8_t* agg,
                                                                 const int8_t val,
                                                                 const int8_t null_val) {
  return 0;
}

extern "C" GPU_RT_STUB int32_t
checked_single_agg_id_double_shared(int64_t* agg,
                                    const double val,
                                    const double null_val) {
  return 0;
}

extern "C" GPU_RT_STUB int32_t checked_single_agg_id_float_shared(int32_t* agg,
                                                                  const float val,
                                                                  const float null_val) {
  return 0;
}

extern "C" GPU_RT_STUB void agg_max_int16_skip_val_shared(int16_t* agg,
                                                          const int16_t val,
                                                          const int16_t skip_val) {}

extern "C" GPU_RT_STUB void agg_max_int8_skip_val_shared(int8_t* agg,
                                                         const int8_t val,
                                                         const int8_t skip_val) {}

extern "C" GPU_RT_STUB void agg_min_int16_skip_val_shared(int16_t* agg,
                                                          const int16_t val,
                                                          const int16_t skip_val) {}

extern "C" GPU_RT_STUB void agg_min_int8_skip_val_shared(int8_t* agg,
                                                         const int8_t val,
                                                         const int8_t skip_val) {}

extern "C" GPU_RT_STUB void agg_id_double_shared_slow(int64_t* agg, const double* val) {}

extern "C" GPU_RT_STUB int64_t agg_sum_shared(int64_t* agg, const int64_t val) {
  return 0;
}

extern "C" GPU_RT_STUB int64_t agg_sum_skip_val_shared(int64_t* agg,
                                                       const int64_t val,
                                                       const int64_t skip_val) {
  return 0;
}
extern "C" GPU_RT_STUB int32_t agg_sum_int32_shared(int32_t* agg, const int32_t val) {
  return 0;
}

extern "C" GPU_RT_STUB int32_t agg_sum_int32_skip_val_shared(int32_t* agg,
                                                             const int32_t val,
                                                             const int32_t skip_val) {
  return 0;
}

extern "C" GPU_RT_STUB void agg_sum_double_shared(int64_t* agg, const double val) {}

extern "C" GPU_RT_STUB void agg_sum_double_skip_val_shared(int64_t* agg,
                                                           const double val,
                                                           const double skip_val) {}
extern "C" GPU_RT_STUB void agg_sum_float_shared(int32_t* agg, const float val) {}

extern "C" GPU_RT_STUB void agg_sum_float_skip_val_shared(int32_t* agg,
                                                          const float val,
                                                          const float skip_val) {}

extern "C" GPU_RT_STUB void force_sync() {}

extern "C" GPU_RT_STUB void sync_warp() {}
extern "C" GPU_RT_STUB void sync_warp_protected(int64_t thread_pos, int64_t row_count) {}
extern "C" GPU_RT_STUB void sync_threadblock() {}

extern "C" GPU_RT_STUB void write_back_non_grouped_agg(int64_t* input_buffer,
                                                       int64_t* output_buffer,
                                                       const int32_t num_agg_cols){};
// x64 stride functions

extern "C" RUNTIME_EXPORT NEVER_INLINE int32_t pos_start_impl(int32_t* error_code) {
  int32_t row_index_resume{0};
  if (error_code) {
    row_index_resume = error_code[0];
    error_code[0] = 0;
  }
  return row_index_resume;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE int32_t group_buff_idx_impl() {
  return pos_start_impl(nullptr);
}

extern "C" RUNTIME_EXPORT NEVER_INLINE int32_t pos_step_impl() {
  return 1;
}

extern "C" GPU_RT_STUB int8_t thread_warp_idx(const int8_t warp_sz) {
  return 0;
}

extern "C" GPU_RT_STUB int64_t get_thread_index() {
  return 0;
}

extern "C" GPU_RT_STUB int64_t* declare_dynamic_shared_memory() {
  return nullptr;
}

extern "C" GPU_RT_STUB int64_t get_block_index() {
  return 0;
}

#undef GPU_RT_STUB

extern "C" RUNTIME_EXPORT ALWAYS_INLINE void record_error_code(const int32_t err_code,
                                                               int32_t* error_codes) {
  // NB: never override persistent error codes (with code greater than zero).
  // On GPU, a projection query with a limit can run out of slots without it
  // being an actual error if the limit has been hit. If a persistent error
  // (division by zero, for example) occurs before running out of slots, we
  // have to avoid overriding it, because there's a risk that the query would
  // go through if we override with a potentially benign out-of-slots code.
  if (err_code && error_codes[pos_start_impl(nullptr)] <= 0) {
    error_codes[pos_start_impl(nullptr)] = err_code;
  }
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t get_error_code(int32_t* error_codes) {
  return error_codes[pos_start_impl(nullptr)];
}

// group by helpers

extern "C" RUNTIME_EXPORT NEVER_INLINE const int64_t* init_shared_mem_nop(
    const int64_t* groups_buffer,
    const int32_t groups_buffer_size) {
  return groups_buffer;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE void write_back_nop(int64_t* dest,
                                                           int64_t* src,
                                                           const int32_t sz) {
#ifndef _WIN32
  // the body is not really needed, just make sure the call is not optimized away
  assert(dest);
#endif
}

extern "C" RUNTIME_EXPORT int64_t* init_shared_mem(const int64_t* global_groups_buffer,
                                                   const int32_t groups_buffer_size) {
  return nullptr;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE void init_group_by_buffer_gpu(
    int64_t* groups_buffer,
    const int64_t* init_vals,
    const uint32_t groups_buffer_entry_count,
    const uint32_t key_qw_count,
    const uint32_t agg_col_count,
    const bool keyless,
    const int8_t warp_size) {
#ifndef _WIN32
  // the body is not really needed, just make sure the call is not optimized away
  assert(groups_buffer);
#endif
}

extern "C" RUNTIME_EXPORT NEVER_INLINE void init_columnar_group_by_buffer_gpu(
    int64_t* groups_buffer,
    const int64_t* init_vals,
    const uint32_t groups_buffer_entry_count,
    const uint32_t key_qw_count,
    const uint32_t agg_col_count,
    const bool keyless,
    const bool blocks_share_memory,
    const int32_t frag_idx) {
#ifndef _WIN32
  // the body is not really needed, just make sure the call is not optimized away
  assert(groups_buffer);
#endif
}

extern "C" RUNTIME_EXPORT NEVER_INLINE void init_group_by_buffer_impl(
    int64_t* groups_buffer,
    const int64_t* init_vals,
    const uint32_t groups_buffer_entry_count,
    const uint32_t key_qw_count,
    const uint32_t agg_col_count,
    const bool keyless,
    const int8_t warp_size) {
#ifndef _WIN32
  // the body is not really needed, just make sure the call is not optimized away
  assert(groups_buffer);
#endif
}

template <typename T>
ALWAYS_INLINE int64_t* get_matching_group_value(int64_t* groups_buffer,
                                                const uint32_t h,
                                                const T* key,
                                                const uint32_t key_count,
                                                const uint32_t row_size_quad) {
  auto off = h * row_size_quad;
  auto row_ptr = reinterpret_cast<T*>(groups_buffer + off);
  if (*row_ptr == get_empty_key<T>()) {
    memcpy(row_ptr, key, key_count * sizeof(T));
    auto row_ptr_i8 = reinterpret_cast<int8_t*>(row_ptr + key_count);
    return reinterpret_cast<int64_t*>(align_to_int64(row_ptr_i8));
  }
  if (memcmp(row_ptr, key, key_count * sizeof(T)) == 0) {
    auto row_ptr_i8 = reinterpret_cast<int8_t*>(row_ptr + key_count);
    return reinterpret_cast<int64_t*>(align_to_int64(row_ptr_i8));
  }
  return nullptr;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t* get_matching_group_value(
    int64_t* groups_buffer,
    const uint32_t h,
    const int64_t* key,
    const uint32_t key_count,
    const uint32_t key_width,
    const uint32_t row_size_quad) {
  switch (key_width) {
    case 4:
      return get_matching_group_value(groups_buffer,
                                      h,
                                      reinterpret_cast<const int32_t*>(key),
                                      key_count,
                                      row_size_quad);
    case 8:
      return get_matching_group_value(groups_buffer, h, key, key_count, row_size_quad);
    default:;
  }
  return nullptr;
}

template <typename T>
ALWAYS_INLINE int32_t get_matching_group_value_columnar_slot(int64_t* groups_buffer,
                                                             const uint32_t entry_count,
                                                             const uint32_t h,
                                                             const T* key,
                                                             const uint32_t key_count) {
  auto off = h;
  auto key_buffer = reinterpret_cast<T*>(groups_buffer);
  if (key_buffer[off] == get_empty_key<T>()) {
    for (size_t i = 0; i < key_count; ++i) {
      key_buffer[off] = key[i];
      off += entry_count;
    }
    return h;
  }
  off = h;
  for (size_t i = 0; i < key_count; ++i) {
    if (key_buffer[off] != key[i]) {
      return -1;
    }
    off += entry_count;
  }
  return h;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t
get_matching_group_value_columnar_slot(int64_t* groups_buffer,
                                       const uint32_t entry_count,
                                       const uint32_t h,
                                       const int64_t* key,
                                       const uint32_t key_count,
                                       const uint32_t key_width) {
  switch (key_width) {
    case 4:
      return get_matching_group_value_columnar_slot(groups_buffer,
                                                    entry_count,
                                                    h,
                                                    reinterpret_cast<const int32_t*>(key),
                                                    key_count);
    case 8:
      return get_matching_group_value_columnar_slot(
          groups_buffer, entry_count, h, key, key_count);
    default:
      return -1;
  }
  return -1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t* get_matching_group_value_columnar(
    int64_t* groups_buffer,
    const uint32_t h,
    const int64_t* key,
    const uint32_t key_qw_count,
    const size_t entry_count) {
  auto off = h;
  if (groups_buffer[off] == EMPTY_KEY_64) {
    for (size_t i = 0; i < key_qw_count; ++i) {
      groups_buffer[off] = key[i];
      off += entry_count;
    }
    return &groups_buffer[off];
  }
  off = h;
  for (size_t i = 0; i < key_qw_count; ++i) {
    if (groups_buffer[off] != key[i]) {
      return nullptr;
    }
    off += entry_count;
  }
  return &groups_buffer[off];
}

/*
 * For a particular hashed_index, returns the row-wise offset
 * to the first matching agg column in memory.
 * It also checks the corresponding group column, and initialize all
 * available keys if they are not empty (it is assumed all group columns are
 * 64-bit wide).
 *
 * Memory layout:
 *
 * | prepended group columns (64-bit each) | agg columns |
 */
extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t* get_matching_group_value_perfect_hash(
    int64_t* groups_buffer,
    const uint32_t hashed_index,
    const int64_t* key,
    const uint32_t key_count,
    const uint32_t row_size_quad) {
  uint32_t off = hashed_index * row_size_quad;
  if (groups_buffer[off] == EMPTY_KEY_64) {
    for (uint32_t i = 0; i < key_count; ++i) {
      groups_buffer[off + i] = key[i];
    }
  }
  return groups_buffer + off + key_count;
}

/**
 * For a particular hashed index (only used with multi-column perfect hash group by)
 * it returns the row-wise offset of the group in the output buffer.
 * Since it is intended for keyless hash use, it assumes there is no group columns
 * prepending the output buffer.
 */
extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t*
get_matching_group_value_perfect_hash_keyless(int64_t* groups_buffer,
                                              const uint32_t hashed_index,
                                              const uint32_t row_size_quad) {
  return groups_buffer + row_size_quad * hashed_index;
}

/*
 * For a particular hashed_index, find and initialize (if necessary) all the group
 * columns corresponding to a key. It is assumed that all group columns are 64-bit wide.
 */
extern "C" RUNTIME_EXPORT ALWAYS_INLINE void
set_matching_group_value_perfect_hash_columnar(int64_t* groups_buffer,
                                               const uint32_t hashed_index,
                                               const int64_t* key,
                                               const uint32_t key_count,
                                               const uint32_t entry_count) {
  if (groups_buffer[hashed_index] == EMPTY_KEY_64) {
    for (uint32_t i = 0; i < key_count; i++) {
      groups_buffer[i * entry_count + hashed_index] = key[i];
    }
  }
}

#include "GeoOpsRuntime.cpp"
#include "GroupByRuntime.cpp"
#include "JoinHashTable/Runtime/JoinHashTableQueryRuntime.cpp"

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t* get_group_value_fast_keyless(
    int64_t* groups_buffer,
    const int64_t key,
    const int64_t min_key,
    const int64_t /* bucket */,
    const uint32_t row_size_quad) {
  return groups_buffer + row_size_quad * (key - min_key);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t* get_group_value_fast_keyless_semiprivate(
    int64_t* groups_buffer,
    const int64_t key,
    const int64_t min_key,
    const int64_t /* bucket */,
    const uint32_t row_size_quad,
    const uint8_t thread_warp_idx,
    const uint8_t warp_size) {
  return groups_buffer + row_size_quad * (warp_size * (key - min_key) + thread_warp_idx);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int8_t* extract_str_ptr(
    const uint64_t str_and_len) {
  return reinterpret_cast<int8_t*>(str_and_len & 0xffffffffffff);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t
extract_str_len(const uint64_t str_and_len) {
  return static_cast<int64_t>(str_and_len) >> 48;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE int8_t* extract_str_ptr_noinline(
    const uint64_t str_and_len) {
  return extract_str_ptr(str_and_len);
}

extern "C" RUNTIME_EXPORT NEVER_INLINE int32_t
extract_str_len_noinline(const uint64_t str_and_len) {
  return extract_str_len(str_and_len);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE uint64_t string_pack(const int8_t* ptr,
                                                             const int32_t len) {
  return (reinterpret_cast<const uint64_t>(ptr) & 0xffffffffffff) |
         (static_cast<const uint64_t>(len) << 48);
}

#ifdef __clang__
#include "../Utils/StringLike.cpp"
#endif

#ifndef __CUDACC__
#include "TopKRuntime.cpp"
#endif

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
char_length(const char* str, const int32_t str_len) {
  return str_len;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
char_length_nullable(const char* str, const int32_t str_len, const int32_t int_null) {
  if (!str) {
    return int_null;
  }
  return str_len;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
key_for_string_encoded(const int32_t str_id) {
  return str_id;
}

extern "C" ALWAYS_INLINE DEVICE int32_t
map_string_dict_id(const int32_t string_id,
                   const int64_t translation_map_handle,
                   const int32_t min_source_id) {
  const int32_t* translation_map =
      reinterpret_cast<const int32_t*>(translation_map_handle);
  return translation_map[string_id - min_source_id];
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE bool sample_ratio(
    const double proportion,
    const int64_t row_offset) {
  const int64_t threshold = 4294967296 * proportion;
  return (row_offset * 2654435761) % 4294967296 < threshold;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
width_bucket(const double target_value,
             const double lower_bound,
             const double upper_bound,
             const double scale_factor,
             const int32_t partition_count) {
  if (target_value < lower_bound) {
    return 0;
  } else if (target_value >= upper_bound) {
    return partition_count + 1;
  }
  return ((target_value - lower_bound) * scale_factor) + 1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
width_bucket_reversed(const double target_value,
                      const double lower_bound,
                      const double upper_bound,
                      const double scale_factor,
                      const int32_t partition_count) {
  if (target_value > lower_bound) {
    return 0;
  } else if (target_value <= upper_bound) {
    return partition_count + 1;
  }
  return ((lower_bound - target_value) * scale_factor) + 1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t
width_bucket_nullable(const double target_value,
                      const double lower_bound,
                      const double upper_bound,
                      const double scale_factor,
                      const int32_t partition_count,
                      const double null_val) {
  if (target_value == null_val) {
    return INT32_MIN;
  }
  return width_bucket(
      target_value, lower_bound, upper_bound, scale_factor, partition_count);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int32_t
width_bucket_reversed_nullable(const double target_value,
                               const double lower_bound,
                               const double upper_bound,
                               const double scale_factor,
                               const int32_t partition_count,
                               const double null_val) {
  if (target_value == null_val) {
    return INT32_MIN;
  }
  return width_bucket_reversed(
      target_value, lower_bound, upper_bound, scale_factor, partition_count);
}

// width_bucket with no out-of-bound check version which can be called
// if we can assure the input target_value expr always resides in the valid range
// (so we can also avoid null checking)
extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
width_bucket_no_oob_check(const double target_value,
                          const double lower_bound,
                          const double scale_factor) {
  return ((target_value - lower_bound) * scale_factor) + 1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
width_bucket_reversed_no_oob_check(const double target_value,
                                   const double lower_bound,
                                   const double scale_factor) {
  return ((lower_bound - target_value) * scale_factor) + 1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
width_bucket_expr(const double target_value,
                  const bool reversed,
                  const double lower_bound,
                  const double upper_bound,
                  const int32_t partition_count) {
  if (reversed) {
    return width_bucket_reversed(target_value,
                                 lower_bound,
                                 upper_bound,
                                 partition_count / (lower_bound - upper_bound),
                                 partition_count);
  }
  return width_bucket(target_value,
                      lower_bound,
                      upper_bound,
                      partition_count / (upper_bound - lower_bound),
                      partition_count);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
width_bucket_expr_nullable(const double target_value,
                           const bool reversed,
                           const double lower_bound,
                           const double upper_bound,
                           const int32_t partition_count,
                           const double null_val) {
  if (target_value == null_val) {
    return INT32_MIN;
  }
  return width_bucket_expr(
      target_value, reversed, lower_bound, upper_bound, partition_count);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
width_bucket_expr_no_oob_check(const double target_value,
                               const bool reversed,
                               const double lower_bound,
                               const double upper_bound,
                               const int32_t partition_count) {
  if (reversed) {
    return width_bucket_reversed_no_oob_check(
        target_value, lower_bound, partition_count / (lower_bound - upper_bound));
  }
  return width_bucket_no_oob_check(
      target_value, lower_bound, partition_count / (upper_bound - lower_bound));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE int64_t
row_number_window_func(const int64_t output_buff, const int64_t pos) {
  return reinterpret_cast<const int64_t*>(output_buff)[pos];
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE double percent_window_func(
    const int64_t output_buff,
    const int64_t pos) {
  return reinterpret_cast<const double*>(output_buff)[pos];
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE double load_double(const int64_t* agg) {
  return *reinterpret_cast<const double*>(may_alias_ptr(agg));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE float load_float(const int32_t* agg) {
  return *reinterpret_cast<const float*>(may_alias_ptr(agg));
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE double load_avg_int(const int64_t* sum,
                                                            const int64_t* count,
                                                            const double null_val) {
  return *count != 0 ? static_cast<double>(*sum) / *count : null_val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE double load_avg_decimal(const int64_t* sum,
                                                                const int64_t* count,
                                                                const double null_val,
                                                                const uint32_t scale) {
  return *count != 0 ? (static_cast<double>(*sum) / pow(10, scale)) / *count : null_val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE double load_avg_double(const int64_t* agg,
                                                               const int64_t* count,
                                                               const double null_val) {
  return *count != 0 ? *reinterpret_cast<const double*>(may_alias_ptr(agg)) / *count
                     : null_val;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE double load_avg_float(const int32_t* agg,
                                                              const int32_t* count,
                                                              const double null_val) {
  return *count != 0 ? *reinterpret_cast<const float*>(may_alias_ptr(agg)) / *count
                     : null_val;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE void linear_probabilistic_count(
    uint8_t* bitmap,
    const uint32_t bitmap_bytes,
    const uint8_t* key_bytes,
    const uint32_t key_len) {
  const uint32_t bit_pos = MurmurHash3(key_bytes, key_len, 0) % (bitmap_bytes * 8);
  const uint32_t word_idx = bit_pos / 32;
  const uint32_t bit_idx = bit_pos % 32;
  reinterpret_cast<uint32_t*>(bitmap)[word_idx] |= 1 << bit_idx;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE void query_stub_hoisted_literals(
    const int8_t** col_buffers,
    const int8_t* literals,
    const int64_t* num_rows,
    const uint64_t* frag_row_offsets,
    const int32_t* max_matched,
    const int64_t* init_agg_value,
    int64_t** out,
    uint32_t frag_idx,
    const int64_t* join_hash_tables,
    int32_t* error_code,
    int32_t* total_matched) {
#ifndef _WIN32
  assert(col_buffers || literals || num_rows || frag_row_offsets || max_matched ||
         init_agg_value || out || frag_idx || error_code || join_hash_tables ||
         total_matched);
#endif
}

extern "C" RUNTIME_EXPORT void multifrag_query_hoisted_literals(
    const int8_t*** col_buffers,
    const uint64_t* num_fragments,
    const int8_t* literals,
    const int64_t* num_rows,
    const uint64_t* frag_row_offsets,
    const int32_t* max_matched,
    int32_t* total_matched,
    const int64_t* init_agg_value,
    int64_t** out,
    int32_t* error_code,
    const uint32_t* num_tables_ptr,
    const int64_t* join_hash_tables) {
  for (uint32_t i = 0; i < *num_fragments; ++i) {
    query_stub_hoisted_literals(col_buffers ? col_buffers[i] : nullptr,
                                literals,
                                &num_rows[i * (*num_tables_ptr)],
                                &frag_row_offsets[i * (*num_tables_ptr)],
                                max_matched,
                                init_agg_value,
                                out,
                                i,
                                join_hash_tables,
                                total_matched,
                                error_code);
  }
}

extern "C" RUNTIME_EXPORT NEVER_INLINE void query_stub(const int8_t** col_buffers,
                                                       const int64_t* num_rows,
                                                       const uint64_t* frag_row_offsets,
                                                       const int32_t* max_matched,
                                                       const int64_t* init_agg_value,
                                                       int64_t** out,
                                                       uint32_t frag_idx,
                                                       const int64_t* join_hash_tables,
                                                       int32_t* error_code,
                                                       int32_t* total_matched) {
#ifndef _WIN32
  assert(col_buffers || num_rows || frag_row_offsets || max_matched || init_agg_value ||
         out || frag_idx || error_code || join_hash_tables || total_matched);
#endif
}

extern "C" RUNTIME_EXPORT void multifrag_query(const int8_t*** col_buffers,
                                               const uint64_t* num_fragments,
                                               const int64_t* num_rows,
                                               const uint64_t* frag_row_offsets,
                                               const int32_t* max_matched,
                                               int32_t* total_matched,
                                               const int64_t* init_agg_value,
                                               int64_t** out,
                                               int32_t* error_code,
                                               const uint32_t* num_tables_ptr,
                                               const int64_t* join_hash_tables) {
  for (uint32_t i = 0; i < *num_fragments; ++i) {
    query_stub(col_buffers ? col_buffers[i] : nullptr,
               &num_rows[i * (*num_tables_ptr)],
               &frag_row_offsets[i * (*num_tables_ptr)],
               max_matched,
               init_agg_value,
               out,
               i,
               join_hash_tables,
               total_matched,
               error_code);
  }
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE bool check_interrupt() {
  if (check_interrupt_init(static_cast<unsigned>(INT_CHECK))) {
    return true;
  }
  return false;
}

extern "C" RUNTIME_EXPORT bool check_interrupt_init(unsigned command) {
  static std::atomic_bool runtime_interrupt_flag{false};

  if (command == static_cast<unsigned>(INT_CHECK)) {
    if (runtime_interrupt_flag.load()) {
      return true;
    }
    return false;
  }
  if (command == static_cast<unsigned>(INT_ABORT)) {
    runtime_interrupt_flag.store(true);
    return false;
  }
  if (command == static_cast<unsigned>(INT_RESET)) {
    runtime_interrupt_flag.store(false);
    return false;
  }
  return false;
}
