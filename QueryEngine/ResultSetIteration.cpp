/**
 * @file    ResultSetIteration.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Iteration part of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultSet.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"

namespace {

double pair_to_double(const std::pair<int64_t, int64_t>& fp_pair, const SQLTypeInfo& ti) {
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

// Interprets ptr as an integer of compact_sz byte width and reads it.
int64_t read_int_from_buff(const int8_t* ptr, const int8_t compact_sz) {
  switch (compact_sz) {
    case 8: {
      return *reinterpret_cast<const int64_t*>(ptr);
    }
    case 4: {
      return *reinterpret_cast<const int32_t*>(ptr);
    }
    default:
      CHECK(false);
  }
  CHECK(false);
  return 0;
}

// Interprets ptr1, ptr2 as the sum and count pair used for AVG.
TargetValue make_avg_target_value(const int8_t* ptr1,
                                  const int8_t compact_sz1,
                                  const int8_t* ptr2,
                                  const int8_t compact_sz2,
                                  const TargetInfo& target_info) {
  int64_t sum{0};
  if (target_info.agg_arg_type.is_integer()) {
    sum = read_int_from_buff(ptr1, compact_sz1);
  } else if (target_info.agg_arg_type.is_fp()) {
    switch (compact_sz1) {
      case 8: {
        double d = *reinterpret_cast<const double*>(ptr1);
        sum = *reinterpret_cast<const int64_t*>(&d);
        break;
      }
      case 4: {
        double d = *reinterpret_cast<const float*>(ptr1);
        sum = *reinterpret_cast<const int64_t*>(&d);
        break;
      }
      default:
        CHECK(false);
    }
  } else {
    CHECK(false);
  }
  const auto count = read_int_from_buff(ptr2, compact_sz2);
  return pair_to_double({sum, count}, target_info.sql_type);
}

// Reads an integer or a float from ptr based on the type and the byte width.
TargetValue make_target_value(const int8_t* ptr, const int8_t compact_sz, const SQLTypeInfo& ti) {
  if (ti.is_integer()) {
    return read_int_from_buff(ptr, compact_sz);
  }
  if (ti.is_fp()) {
    switch (compact_sz) {
      case 8: {
        return *reinterpret_cast<const double*>(ptr);
      }
      case 4: {
        CHECK_EQ(kFLOAT, ti.get_type());
        return *reinterpret_cast<const float*>(ptr);
      }
      default:
        CHECK(false);
    }
  }
  if (ti.is_string() && ti.get_compression() == kENCODING_DICT) {
    return read_int_from_buff(ptr, compact_sz);
  }
  CHECK(false);
  return TargetValue(int64_t(0));
}

// Gets the TargetValue stored at position entry_idx in the col1_ptr and col2_ptr
// column buffers. The second column is only used for AVG.
TargetValue get_target_value_from_buffer_colwise(const int8_t* col1_ptr,
                                                 const int8_t compact_sz1,
                                                 const int8_t* col2_ptr,
                                                 const int8_t compact_sz2,
                                                 const size_t entry_idx,
                                                 const TargetInfo& target_info,
                                                 const QueryMemoryDescriptor& query_mem_desc) {
  CHECK(query_mem_desc.output_columnar);
  const auto& ti = target_info.sql_type;
  const auto ptr1 = col1_ptr + compact_sz1 * entry_idx;
  if (target_info.agg_kind == kAVG) {
    CHECK(col2_ptr);
    CHECK(compact_sz2);
    const auto ptr2 = col2_ptr + compact_sz2 * entry_idx;
    return make_avg_target_value(ptr1, compact_sz1, ptr2, compact_sz2, target_info);
  }
  return make_target_value(ptr1, compact_sz1, ti);
}

// Gets the TargetValue stored in slot_idx (and slot_idx for AVG) of rowwise_target_ptr.
TargetValue get_target_value_from_buffer_rowwise(const int8_t* rowwise_target_ptr,
                                                 const TargetInfo& target_info,
                                                 const size_t slot_idx,
                                                 const QueryMemoryDescriptor& query_mem_desc) {
  auto ptr1 = rowwise_target_ptr;
  auto compact_sz1 = query_mem_desc.agg_col_widths[slot_idx].compact;
  const int8_t* ptr2{nullptr};
  int8_t compact_sz2{0};
  if (target_info.is_agg && target_info.agg_kind == kAVG) {
    ptr2 = rowwise_target_ptr + query_mem_desc.agg_col_widths[slot_idx].compact;
    compact_sz2 = query_mem_desc.agg_col_widths[slot_idx + 1].compact;
  }
  if (target_info.agg_kind == kAVG) {
    CHECK(ptr2);
    return make_avg_target_value(ptr1, compact_sz1, ptr2, compact_sz2, target_info);
  }
  CHECK(!ptr2);
  CHECK_EQ(0, compact_sz2);
  return make_target_value(ptr1, compact_sz1, target_info.sql_type);
}

// Gets the byte offset, starting from the beginning of the row targets buffer, of
// the value in position slot_idx (only makes sense for row-wise representation).
size_t get_byteoff_of_slot(const size_t slot_idx, const QueryMemoryDescriptor& query_mem_desc) {
  size_t result = 0;
  for (size_t i = 0; i < slot_idx; ++i) {
    result += query_mem_desc.agg_col_widths[i].compact;
  }
  return result;
}

// Given the entire buffer for the result set, buff, finds the beginning of the
// column for slot_idx. Only makes sense for column-wise representation.
const int8_t* advance_col_buff_to_slot(const int8_t* buff,
                                       const QueryMemoryDescriptor& query_mem_desc,
                                       const std::vector<TargetInfo>& targets,
                                       const size_t slot_idx) {
  auto crt_col_ptr = get_cols_ptr(buff, query_mem_desc);
  const auto buffer_col_count = get_buffer_col_slot_count(query_mem_desc);
  size_t agg_col_idx{0};
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    if (agg_col_idx == slot_idx) {
      return crt_col_ptr;
    }
    CHECK_LT(agg_col_idx, buffer_col_count);
    const auto& agg_info = targets[target_idx];
    crt_col_ptr = advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc, agg_col_idx);
    if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
      if (agg_col_idx + 1 == slot_idx) {
        return crt_col_ptr;
      }
      crt_col_ptr = advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc, agg_col_idx + 1);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info);
  }
  CHECK(false);
  return nullptr;
}

}  // namespace

std::vector<TargetValue> ResultSet::getNextRow(const bool /* translate_strings */,
                                               const bool /* decimal_to_double */) const {
  advanceCursorToNextEntry();
  if (keep_first_ && crt_row_buff_idx_ >= drop_first_ + keep_first_) {
    return {};
  }
  if (crt_row_buff_idx_ >= query_mem_desc_.entry_count) {
    CHECK_EQ(query_mem_desc_.entry_count, crt_row_buff_idx_);
    return {};
  }
  const auto buff = storage_->buff_;
  CHECK(buff);
  std::vector<TargetValue> row;
  size_t agg_col_idx = 0;
  const auto buffer_col_count = get_buffer_col_slot_count(storage_->query_mem_desc_);
  const int8_t* rowwise_target_ptr{nullptr};
  const int8_t* crt_col_ptr{nullptr};
  if (query_mem_desc_.output_columnar) {
    crt_col_ptr = get_cols_ptr(buff, query_mem_desc_);
  } else {
    rowwise_target_ptr =
        row_ptr_rowwise(buff, query_mem_desc_, crt_row_buff_idx_) + get_key_bytes_rowwise(query_mem_desc_);
  }
  for (size_t target_idx = 0; target_idx < storage_->targets_.size(); ++target_idx) {
    CHECK_LT(agg_col_idx, buffer_col_count);
    const auto& agg_info = storage_->targets_[target_idx];
    if (query_mem_desc_.output_columnar) {
      const auto next_col_ptr = advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc_, agg_col_idx);
      const auto col2_ptr = (agg_info.is_agg && agg_info.agg_kind == kAVG) ? next_col_ptr : nullptr;
      const auto compact_sz2 =
          (agg_info.is_agg && agg_info.agg_kind == kAVG) ? query_mem_desc_.agg_col_widths[agg_col_idx + 1].compact : 0;
      row.push_back(get_target_value_from_buffer_colwise(crt_col_ptr,
                                                         query_mem_desc_.agg_col_widths[agg_col_idx].compact,
                                                         col2_ptr,
                                                         compact_sz2,
                                                         crt_row_buff_idx_,
                                                         agg_info,
                                                         query_mem_desc_));
      crt_col_ptr = next_col_ptr;
      if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
        crt_col_ptr = advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
      }
    } else {
      row.push_back(get_target_value_from_buffer_rowwise(rowwise_target_ptr, agg_info, agg_col_idx, query_mem_desc_));
      rowwise_target_ptr = advance_target_ptr(rowwise_target_ptr, agg_info, agg_col_idx, query_mem_desc_);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info);
  }
  ++crt_row_buff_idx_;
  return row;
}

// Not all entries in the buffer represent a valid row. Advance the internal cursor
// used for the getNextRow method to the next row which is valid.
void ResultSet::advanceCursorToNextEntry() const {
  CHECK(GroupByColRangeType::OneColKnownRange == storage_->query_mem_desc_.hash_type ||
        GroupByColRangeType::MultiColPerfectHash == storage_->query_mem_desc_.hash_type ||
        GroupByColRangeType::MultiCol == storage_->query_mem_desc_.hash_type);
  while (crt_row_buff_idx_ < query_mem_desc_.entry_count) {
    if (!storage_->isEmptyEntry(crt_row_buff_idx_, storage_->buff_)) {
      break;
    }
    ++crt_row_buff_idx_;
  }
}

// Returns true iff the entry at position entry_idx in buff contains a valid row.
bool ResultSetStorage::isEmptyEntry(const size_t entry_idx, const int8_t* buff) const {
  CHECK(query_mem_desc_.hash_type == GroupByColRangeType::OneColKnownRange ||
        query_mem_desc_.hash_type == GroupByColRangeType::MultiColPerfectHash ||
        query_mem_desc_.hash_type == GroupByColRangeType::MultiCol);
  if (query_mem_desc_.keyless_hash) {
    CHECK(query_mem_desc_.hash_type == GroupByColRangeType::OneColKnownRange ||
          query_mem_desc_.hash_type == GroupByColRangeType::MultiColPerfectHash);
    CHECK_GE(query_mem_desc_.idx_target_as_key, 0);
    CHECK_LT(static_cast<size_t>(query_mem_desc_.idx_target_as_key), target_init_vals_.size());
    if (query_mem_desc_.output_columnar) {
      const auto col_buff =
          advance_col_buff_to_slot(buff, query_mem_desc_, targets_, query_mem_desc_.idx_target_as_key);
      const auto entry_buff =
          col_buff + entry_idx * query_mem_desc_.agg_col_widths[query_mem_desc_.idx_target_as_key].compact;
      return read_int_from_buff(entry_buff,
                                query_mem_desc_.agg_col_widths[query_mem_desc_.idx_target_as_key].compact) ==
             target_init_vals_[query_mem_desc_.idx_target_as_key];
    }
    const auto rowwise_target_ptr =
        row_ptr_rowwise(buff, query_mem_desc_, entry_idx) + get_key_bytes_rowwise(query_mem_desc_);
    const auto target_slot_off = get_byteoff_of_slot(query_mem_desc_.idx_target_as_key, query_mem_desc_);
    return read_int_from_buff(rowwise_target_ptr + target_slot_off,
                              query_mem_desc_.agg_col_widths[query_mem_desc_.idx_target_as_key].compact) ==
           target_init_vals_[query_mem_desc_.idx_target_as_key];
  }
  // TODO(alex): Don't assume 64-bit keys, we could compact them as well.
  if (query_mem_desc_.output_columnar) {
    return reinterpret_cast<const int64_t*>(buff)[entry_idx] == EMPTY_KEY_64;
  }
  const auto keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, entry_idx);
  return *reinterpret_cast<const int64_t*>(keys_ptr) == EMPTY_KEY_64;
}

bool ResultSet::isNull(const SQLTypeInfo& ti, const InternalTargetValue& val) {
  if (val.isInt()) {
    if (!ti.is_fp()) {
      return val.i1 == inline_int_null_val(ti);
    }
    const auto null_val = inline_fp_null_val(ti);
    return val.i1 == *reinterpret_cast<const int64_t*>(&null_val);
  }
  if (val.isPair()) {
    CHECK(val.i2);
    return pair_to_double({val.i1, val.i2}, ti) == NULL_DOUBLE;
  }
  if (val.isStr()) {
    return false;
  }
  CHECK(val.isNull());
  return true;
}
