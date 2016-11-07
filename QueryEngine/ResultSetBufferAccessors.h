/**
 * @file    ResultSetBufferAccessors.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Utility functions for easy access to the result set buffers.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_RESULTSETBUFFERACCESSORS_H
#define QUERYENGINE_RESULTSETBUFFERACCESSORS_H

#include "SqlTypesLayout.h"

#include "../Shared/unreachable.h"

#ifndef __CUDACC__

#include "QueryMemoryDescriptor.h"

inline bool is_real_str_or_array(const TargetInfo& target_info) {
  return !target_info.is_agg &&
         (target_info.sql_type.is_array() ||
          (target_info.sql_type.is_string() && target_info.sql_type.get_compression() == kENCODING_NONE));
}

inline size_t advance_slot(const size_t j, const TargetInfo& target_info) {
  return j + ((target_info.agg_kind == kAVG || is_real_str_or_array(target_info)) ? 2 : 1);
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

inline size_t key_offset_rowwise(const size_t entry_idx, const size_t key_count, const size_t slot_count) {
  return (key_count + slot_count) * entry_idx;
}

inline size_t key_offset_colwise(const size_t entry_idx, const size_t key_idx, const size_t entry_count) {
  return key_idx * entry_count + entry_idx;
}

template <class T>
inline T advance_to_next_columnar_target_buff(T target_ptr,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              const size_t target_slot_idx) {
  CHECK_LT(target_slot_idx, query_mem_desc.agg_col_widths.size());
  auto new_target_ptr =
      target_ptr + query_mem_desc.entry_count * query_mem_desc.agg_col_widths[target_slot_idx].compact;
  if (!query_mem_desc.target_column_pad_bytes.empty()) {
    CHECK_LT(target_slot_idx, query_mem_desc.target_column_pad_bytes.size());
    new_target_ptr += query_mem_desc.target_column_pad_bytes[target_slot_idx];
  }
  return new_target_ptr;
}

inline size_t get_groupby_col_count(const QueryMemoryDescriptor& query_mem_desc) {
  return query_mem_desc.group_col_widths.size();
}

inline size_t get_key_count_for_descriptor(const QueryMemoryDescriptor& query_mem_desc) {
  return query_mem_desc.keyless_hash ? 0 : get_groupby_col_count(query_mem_desc);
}

inline size_t get_buffer_col_slot_count(const QueryMemoryDescriptor& query_mem_desc) {
  return query_mem_desc.agg_col_widths.size();
}

template <class T>
inline T get_cols_ptr(T buff, const QueryMemoryDescriptor& query_mem_desc) {
  CHECK(query_mem_desc.output_columnar);
  auto cols_ptr = buff;
  if (query_mem_desc.keyless_hash) {
    CHECK(query_mem_desc.key_column_pad_bytes.empty());
  } else {
    CHECK_EQ(query_mem_desc.key_column_pad_bytes.empty(), query_mem_desc.target_column_pad_bytes.empty());
  }
  const bool has_key_col_padding = !query_mem_desc.key_column_pad_bytes.empty();
  const auto key_count = get_key_count_for_descriptor(query_mem_desc);
  if (has_key_col_padding) {
    CHECK_EQ(key_count, query_mem_desc.key_column_pad_bytes.size());
  }
  for (size_t key_idx = 0; key_idx < key_count; ++key_idx) {
    cols_ptr += query_mem_desc.group_col_widths[key_idx] * query_mem_desc.entry_count;
    if (has_key_col_padding) {
      cols_ptr += query_mem_desc.key_column_pad_bytes[key_idx];
    }
  }
  return cols_ptr;
}

inline size_t get_key_bytes_rowwise(const QueryMemoryDescriptor& query_mem_desc) {
  if (query_mem_desc.keyless_hash) {
    return 0;
  }
  size_t result = 0;
  for (const auto& group_width : query_mem_desc.group_col_widths) {
    result += group_width;
  }
  return result;
}

inline size_t get_row_bytes(const QueryMemoryDescriptor& query_mem_desc) {
  size_t result = get_key_bytes_rowwise(query_mem_desc);
  for (const auto& target_width : query_mem_desc.agg_col_widths) {
    result += target_width.compact;
  }
  return result;
}

template <class T>
inline T row_ptr_rowwise(T buff, const QueryMemoryDescriptor& query_mem_desc, const size_t entry_idx) {
  const auto row_bytes = get_row_bytes(query_mem_desc);
  return buff + entry_idx * row_bytes;
}

template <class T>
inline T advance_target_ptr(T target_ptr,
                            const TargetInfo& target_info,
                            const size_t slot_idx,
                            const QueryMemoryDescriptor& query_mem_desc) {
  auto result = target_ptr + query_mem_desc.agg_col_widths[slot_idx].compact;
  if ((target_info.is_agg && target_info.agg_kind == kAVG) || is_real_str_or_array(target_info)) {
    return result + query_mem_desc.agg_col_widths[slot_idx + 1].compact;
  }
  return result;
}

#endif  // __CUDACC__

inline double pair_to_double(const std::pair<int64_t, int64_t>& fp_pair, const SQLTypeInfo& ti) {
  double dividend{0.0};
  int64_t null_val{0};
  switch (ti.get_type()) {
    case kFLOAT: {
      dividend = *reinterpret_cast<const double*>(&fp_pair.first);
      double null_float = inline_fp_null_val(ti);
      null_val = *reinterpret_cast<int64_t*>(&null_float);
      break;
    }
    case kDOUBLE: {
      dividend = *reinterpret_cast<const double*>(&fp_pair.first);
      double null_double = inline_fp_null_val(ti);
      null_val = *reinterpret_cast<int64_t*>(&null_double);
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
             ? (dividend / exp_to_scale(ti.is_decimal() ? ti.get_scale() : 0)) / static_cast<double>(fp_pair.second)
             : dividend / static_cast<double>(fp_pair.second);
}

inline int64_t null_val_bit_pattern(const SQLTypeInfo& ti) {
  if (ti.is_fp()) {
    const auto double_null_val = inline_fp_null_val(ti);
    return *reinterpret_cast<const int64_t*>(&double_null_val);
  }
  if ((ti.is_string() && ti.get_compression() == kENCODING_NONE) || ti.is_array()) {
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
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return 0;
}

#endif  // QUERYENGINE_RESULTSETBUFFERACCESSORS_H
