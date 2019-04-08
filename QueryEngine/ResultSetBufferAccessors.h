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
 * @file    ResultSetBufferAccessors.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Utility functions for easy access to the result set buffers.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_RESULTSETBUFFERACCESSORS_H
#define QUERYENGINE_RESULTSETBUFFERACCESSORS_H

#include "BufferCompaction.h"
#include "SqlTypesLayout.h"
#include "TypePunning.h"

#include "../Shared/unreachable.h"

#ifndef __CUDACC__

#include "QueryMemoryDescriptor.h"

#include <algorithm>

inline bool is_real_str_or_array(const TargetInfo& target_info) {
  return (!target_info.is_agg || target_info.agg_kind == kSAMPLE) &&
         (target_info.sql_type.is_array() ||
          (target_info.sql_type.is_string() &&
           target_info.sql_type.get_compression() == kENCODING_NONE));
}

inline size_t get_slots_for_geo_target(const TargetInfo& target_info,
                                       const bool separate_varlen_storage) {
  // Aggregates on geospatial types are serialized directly by rewriting the underlying
  // buffer. Even if separate varlen storage is valid, treat aggregates the same on
  // distributed and single node
  if (separate_varlen_storage && !target_info.is_agg) {
    return 1;
  } else {
    return 2 * target_info.sql_type.get_physical_coord_cols();
  }
}

inline size_t get_slots_for_target(const TargetInfo& target_info,
                                   const bool separate_varlen_storage) {
  if (target_info.is_agg) {
    if (target_info.agg_kind == kAVG || is_real_str_or_array(target_info)) {
      return 2;
    } else {
      return 1;
    }
  } else {
    if (is_real_str_or_array(target_info) && !separate_varlen_storage) {
      return 2;
    } else {
      return 1;
    }
  }
}

inline size_t advance_slot(const size_t j,
                           const TargetInfo& target_info,
                           const bool separate_varlen_storage) {
  if (target_info.sql_type.is_geometry()) {
    return j + get_slots_for_geo_target(target_info, separate_varlen_storage);
  }
  return j + get_slots_for_target(target_info, separate_varlen_storage);
}

inline size_t slot_offset_rowwise(const size_t entry_idx,
                                  const size_t slot_idx,
                                  const size_t key_count,
                                  const size_t slot_count) {
  return (key_count + slot_count) * entry_idx + (key_count + slot_idx);
}

inline size_t slot_offset_colwise(const size_t entry_idx,
                                  const size_t slot_idx,
                                  const size_t key_count,
                                  const size_t entry_count) {
  return (key_count + slot_idx) * entry_count + entry_idx;
}

inline size_t key_offset_rowwise(const size_t entry_idx,
                                 const size_t key_count,
                                 const size_t slot_count) {
  return (key_count + slot_count) * entry_idx;
}

inline size_t key_offset_colwise(const size_t entry_idx,
                                 const size_t key_idx,
                                 const size_t entry_count) {
  return key_idx * entry_count + entry_idx;
}

template <class T>
inline T advance_to_next_columnar_target_buff(T target_ptr,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              const size_t target_slot_idx) {
  auto new_target_ptr = target_ptr;
  const auto column_size = query_mem_desc.getEntryCount() *
                           query_mem_desc.getPaddedColumnWidthBytes(target_slot_idx);
  new_target_ptr += align_to_int64(column_size);

  return new_target_ptr;
}

template <class T>
inline T get_cols_ptr(T buff, const QueryMemoryDescriptor& query_mem_desc) {
  CHECK(query_mem_desc.didOutputColumnar());
  return buff + query_mem_desc.getColOffInBytes(0);
}

inline size_t get_key_bytes_rowwise(const QueryMemoryDescriptor& query_mem_desc) {
  if (query_mem_desc.hasKeylessHash()) {
    return 0;
  }
  auto consist_key_width = query_mem_desc.getEffectiveKeyWidth();
  CHECK(consist_key_width);
  return consist_key_width * query_mem_desc.groupColWidthsSize();
}

inline size_t get_row_bytes(const QueryMemoryDescriptor& query_mem_desc) {
  size_t result = align_to_int64(get_key_bytes_rowwise(query_mem_desc));  // plus padding
  return result + query_mem_desc.getRowWidth();
}

template <class T>
inline T row_ptr_rowwise(T buff,
                         const QueryMemoryDescriptor& query_mem_desc,
                         const size_t entry_idx) {
  const auto row_bytes = get_row_bytes(query_mem_desc);
  return buff + entry_idx * row_bytes;
}

template <class T>
inline T advance_target_ptr_row_wise(T target_ptr,
                                     const TargetInfo& target_info,
                                     const size_t slot_idx,
                                     const QueryMemoryDescriptor& query_mem_desc,
                                     const bool separate_varlen_storage) {
  auto result = target_ptr + query_mem_desc.getColumnWidth(slot_idx).compact;
  if ((target_info.is_agg && target_info.agg_kind == kAVG) ||
      ((!separate_varlen_storage || target_info.is_agg) &&
       is_real_str_or_array(target_info))) {
    return result + query_mem_desc.getColumnWidth(slot_idx + 1).compact;
  }
  if (target_info.sql_type.is_geometry() &&
      (!separate_varlen_storage || target_info.is_agg)) {
    for (auto i = 1; i < 2 * target_info.sql_type.get_physical_coord_cols(); ++i) {
      result += query_mem_desc.getColumnWidth(slot_idx + i).compact;
    }
  }
  return result;
}

template <class T>
inline T advance_target_ptr_col_wise(T target_ptr,
                                     const TargetInfo& target_info,
                                     const size_t slot_idx,
                                     const QueryMemoryDescriptor& query_mem_desc,
                                     const bool separate_varlen_storage) {
  auto result =
      advance_to_next_columnar_target_buff(target_ptr, query_mem_desc, slot_idx);
  if ((target_info.is_agg && target_info.agg_kind == kAVG) ||
      (is_real_str_or_array(target_info) && !separate_varlen_storage)) {
    return advance_to_next_columnar_target_buff(result, query_mem_desc, slot_idx + 1);
  } else if (target_info.sql_type.is_geometry() && !separate_varlen_storage) {
    for (auto i = 1; i < 2 * target_info.sql_type.get_physical_coord_cols(); ++i) {
      result = advance_to_next_columnar_target_buff(result, query_mem_desc, slot_idx + i);
    }
    return result;
  } else {
    return result;
  }
}

inline size_t get_slot_off_quad(const QueryMemoryDescriptor& query_mem_desc) {
  return align_to_int64(get_key_bytes_rowwise(query_mem_desc)) / sizeof(int64_t);
}

#endif  // __CUDACC__

inline double pair_to_double(const std::pair<int64_t, int64_t>& fp_pair,
                             const SQLTypeInfo& ti,
                             const bool float_argument_input) {
  double dividend{0.0};
  int64_t null_val{0};
  switch (ti.get_type()) {
    case kFLOAT: {
      if (float_argument_input) {
        dividend = static_cast<double>(
            *reinterpret_cast<const float*>(may_alias_ptr(&fp_pair.first)));
      } else {
        dividend = *reinterpret_cast<const double*>(may_alias_ptr(&fp_pair.first));
      }
      double null_float = inline_fp_null_val(ti);
      null_val = *reinterpret_cast<const int64_t*>(may_alias_ptr(&null_float));
      break;
    }
    case kDOUBLE: {
      dividend = *reinterpret_cast<const double*>(may_alias_ptr(&fp_pair.first));
      double null_double = inline_fp_null_val(ti);
      null_val = *reinterpret_cast<const int64_t*>(may_alias_ptr(&null_double));
      break;
    }
    default: {
      CHECK(ti.is_integer() || ti.is_decimal());
      dividend = static_cast<double>(fp_pair.first);
      null_val = inline_int_null_val(ti);
      break;
    }
  }
  if (!ti.get_notnull() && null_val == fp_pair.first) {
    return inline_fp_null_val(SQLTypeInfo(kDOUBLE, false));
  }

  return ti.is_integer() || ti.is_decimal()
             ? (dividend / exp_to_scale(ti.is_decimal() ? ti.get_scale() : 0)) /
                   static_cast<double>(fp_pair.second)
             : dividend / static_cast<double>(fp_pair.second);
}

inline int64_t null_val_bit_pattern(const SQLTypeInfo& ti,
                                    const bool float_argument_input) {
  if (ti.is_fp()) {
    if (float_argument_input && ti.get_type() == kFLOAT) {
      int64_t float_null_val = 0;
      *reinterpret_cast<float*>(may_alias_ptr(&float_null_val)) =
          static_cast<float>(inline_fp_null_val(ti));
      return float_null_val;
    }
    const auto double_null_val = inline_fp_null_val(ti);
    return *reinterpret_cast<const int64_t*>(may_alias_ptr(&double_null_val));
  }
  if ((ti.is_string() && ti.get_compression() == kENCODING_NONE) || ti.is_array() ||
      ti.is_geometry()) {
    return 0;
  }
  return inline_int_null_val(ti);
}

// Interprets ptr as an integer of compact_sz byte width and reads it.
inline int64_t read_int_from_buff(const int8_t* ptr, const int8_t compact_sz) {
  switch (compact_sz) {
    case 8: {
      return *reinterpret_cast<const int64_t*>(ptr);
    }
    case 4: {
      return *reinterpret_cast<const int32_t*>(ptr);
    }
    case 2: {
      return *reinterpret_cast<const int16_t*>(ptr);
    }
    case 1: {
      return *reinterpret_cast<const int8_t*>(ptr);
    }
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return 0;
}

#endif  // QUERYENGINE_RESULTSETBUFFERACCESSORS_H
