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
 * @file    ResultSetIteration.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Iteration part of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "Execute.h"
#include "ResultSet.h"
#include "ResultRows.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"
#include "TypePunning.h"
#include "../Shared/likely.h"

namespace {

// Interprets ptr1, ptr2 as the sum and count pair used for AVG.
TargetValue make_avg_target_value(const int8_t* ptr1,
                                  const int8_t compact_sz1,
                                  const int8_t* ptr2,
                                  const int8_t compact_sz2,
                                  const TargetInfo& target_info) {
  int64_t sum{0};
  CHECK(target_info.agg_kind == kAVG);
  const bool float_argument_input = takes_float_argument(target_info);
  const auto actual_compact_sz1 = float_argument_input ? sizeof(float) : compact_sz1;
  if (target_info.agg_arg_type.is_integer() || target_info.agg_arg_type.is_decimal()) {
    sum = read_int_from_buff(ptr1, actual_compact_sz1);
  } else if (target_info.agg_arg_type.is_fp()) {
    switch (actual_compact_sz1) {
      case 8: {
        double d = *reinterpret_cast<const double*>(ptr1);
        sum = *reinterpret_cast<const int64_t*>(may_alias_ptr(&d));
        break;
      }
      case 4: {
        double d = *reinterpret_cast<const float*>(ptr1);
        sum = *reinterpret_cast<const int64_t*>(may_alias_ptr(&d));
        break;
      }
      default:
        CHECK(false);
    }
  } else {
    CHECK(false);
  }
  const auto count = read_int_from_buff(ptr2, compact_sz2);
  return pair_to_double({sum, count}, target_info.sql_type, false);
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
                                       const size_t slot_idx,
                                       const bool none_encoded_strings_valid) {
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
    agg_col_idx = advance_slot(agg_col_idx, agg_info, none_encoded_strings_valid);
  }
  CHECK(false);
  return nullptr;
}

}  // namespace

std::vector<TargetValue> ResultSet::getRowAt(const size_t global_entry_idx,
                                             const bool translate_strings,
                                             const bool decimal_to_double,
                                             const bool fixup_count_distinct_pointers) const {
  const auto storage_lookup_result = fixup_count_distinct_pointers
                                         ? StorageLookupResult{storage_.get(), global_entry_idx, 0}
                                         : findStorage(global_entry_idx);
  const auto storage = storage_lookup_result.storage_ptr;
  const auto local_entry_idx = storage_lookup_result.fixedup_entry_idx;
  if (!fixup_count_distinct_pointers && storage->isEmptyEntry(local_entry_idx)) {
    return {};
  }

  const auto buff = storage->buff_;
  CHECK(buff);
  std::vector<TargetValue> row;
  size_t agg_col_idx = 0;
  int8_t* rowwise_target_ptr{nullptr};
  int8_t* keys_ptr{nullptr};
  const int8_t* crt_col_ptr{nullptr};
  if (query_mem_desc_.output_columnar) {
    crt_col_ptr = get_cols_ptr(buff, query_mem_desc_);
  } else {
    keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, local_entry_idx);
    const auto key_bytes_with_padding = align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
    rowwise_target_ptr = keys_ptr + key_bytes_with_padding;
  }
  for (size_t target_idx = 0; target_idx < storage_->targets_.size(); ++target_idx) {
    const auto& agg_info = storage_->targets_[target_idx];
    if (query_mem_desc_.output_columnar) {
      const auto next_col_ptr = advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc_, agg_col_idx);
      const auto col2_ptr = (agg_info.is_agg && agg_info.agg_kind == kAVG) ? next_col_ptr : nullptr;
      const auto compact_sz2 =
          (agg_info.is_agg && agg_info.agg_kind == kAVG) ? query_mem_desc_.agg_col_widths[agg_col_idx + 1].compact : 0;
      row.push_back(getTargetValueFromBufferColwise(crt_col_ptr,
                                                    query_mem_desc_.agg_col_widths[agg_col_idx].compact,
                                                    col2_ptr,
                                                    compact_sz2,
                                                    global_entry_idx,
                                                    agg_info,
                                                    target_idx,
                                                    translate_strings,
                                                    decimal_to_double));
      crt_col_ptr = next_col_ptr;
      if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
        crt_col_ptr = advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
      }
    } else {
      row.push_back(getTargetValueFromBufferRowwise(rowwise_target_ptr,
                                                    keys_ptr,
                                                    global_entry_idx,
                                                    agg_info,
                                                    target_idx,
                                                    agg_col_idx,
                                                    translate_strings,
                                                    decimal_to_double,
                                                    fixup_count_distinct_pointers));
      rowwise_target_ptr =
          advance_target_ptr(rowwise_target_ptr, agg_info, agg_col_idx, query_mem_desc_, none_encoded_strings_valid_);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info, none_encoded_strings_valid_);
  }

  return row;
}

TargetValue ResultSet::getRowAt(const size_t row_idx,
                                const size_t col_idx,
                                const bool translate_strings,
                                const bool decimal_to_double /* = true */) const {
  std::lock_guard<std::mutex> lock(row_iteration_mutex_);
  moveToBegin();
  for (size_t i = 0; i < row_idx; ++i) {
    auto crt_row = getNextRowUnlocked(translate_strings, decimal_to_double);
    CHECK(!crt_row.empty());
  }
  auto crt_row = getNextRowUnlocked(translate_strings, decimal_to_double);
  CHECK(!crt_row.empty());
  return crt_row[col_idx];
}

OneIntegerColumnRow ResultSet::getOneColRow(const size_t global_entry_idx) const {
  const auto storage_lookup_result = findStorage(global_entry_idx);
  const auto storage = storage_lookup_result.storage_ptr;
  const auto local_entry_idx = storage_lookup_result.fixedup_entry_idx;
  if (storage->isEmptyEntry(local_entry_idx)) {
    return {0, false};
  }
  const auto buff = storage->buff_;
  CHECK(buff);
  CHECK(!query_mem_desc_.output_columnar);
  const auto keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, local_entry_idx);
  const auto key_bytes_with_padding = align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
  const auto rowwise_target_ptr = keys_ptr + key_bytes_with_padding;
  const auto tv = getTargetValueFromBufferRowwise(
      rowwise_target_ptr, keys_ptr, global_entry_idx, targets_.front(), 0, 0, false, false, false);
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  CHECK(scalar_tv);
  const auto ival_ptr = boost::get<int64_t>(scalar_tv);
  CHECK(ival_ptr);
  return {*ival_ptr, true};
}

std::vector<TargetValue> ResultSet::getRowAt(const size_t logical_index) const {
  if (logical_index >= entryCount()) {
    return {};
  }
  const auto entry_idx = permutation_.empty() ? logical_index : permutation_[logical_index];
  return getRowAt(entry_idx, true, false, false);
}

std::vector<TargetValue> ResultSet::getRowAtNoTranslations(const size_t logical_index) const {
  if (logical_index >= entryCount()) {
    return {};
  }
  const auto entry_idx = permutation_.empty() ? logical_index : permutation_[logical_index];
  return getRowAt(entry_idx, false, false, false);
}

bool ResultSet::isRowAtEmpty(const size_t logical_index) const {
  if (logical_index >= entryCount()) {
    return true;
  }
  const auto entry_idx = permutation_.empty() ? logical_index : permutation_[logical_index];
  const auto storage_lookup_result = findStorage(entry_idx);
  const auto storage = storage_lookup_result.storage_ptr;
  const auto local_entry_idx = storage_lookup_result.fixedup_entry_idx;
  return storage->isEmptyEntry(local_entry_idx);
}

std::vector<TargetValue> ResultSet::getNextRow(const bool translate_strings, const bool decimal_to_double) const {
  std::lock_guard<std::mutex> lock(row_iteration_mutex_);
  if (!storage_ && !just_explain_) {
    return {};
  }
  return getNextRowUnlocked(translate_strings, decimal_to_double);
}

std::vector<TargetValue> ResultSet::getNextRowUnlocked(const bool translate_strings,
                                                       const bool decimal_to_double) const {
  if (just_explain_) {
    if (fetched_so_far_) {
      return {};
    }
    fetched_so_far_ = 1;
    return {explanation_};
  }
  while (fetched_so_far_ < drop_first_) {
    const auto row = getNextRowImpl(translate_strings, decimal_to_double);
    if (row.empty()) {
      return row;
    }
  }
  return getNextRowImpl(translate_strings, decimal_to_double);
}

std::vector<TargetValue> ResultSet::getNextRowImpl(const bool translate_strings, const bool decimal_to_double) const {
  auto entry_buff_idx = advanceCursorToNextEntry();
  if (keep_first_ && fetched_so_far_ >= drop_first_ + keep_first_) {
    return {};
  }

  if (crt_row_buff_idx_ >= entryCount()) {
    CHECK_EQ(entryCount(), crt_row_buff_idx_);
    return {};
  }
  auto row = getRowAt(entry_buff_idx, translate_strings, decimal_to_double, false);
  CHECK(!row.empty());
  ++crt_row_buff_idx_;
  ++fetched_so_far_;

  return row;
}

namespace {

const int8_t* columnar_elem_ptr(const size_t entry_idx, const int8_t* col1_ptr, const int8_t compact_sz1) {
  return col1_ptr + compact_sz1 * entry_idx;
}

int64_t int_resize_cast(const int64_t ival, const size_t sz) {
  switch (sz) {
    case 8:
      return ival;
    case 4:
      return static_cast<int32_t>(ival);
    case 2:
      return static_cast<int16_t>(ival);
    case 1:
      return static_cast<int8_t>(ival);
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return 0;
}

}  // namespace

InternalTargetValue ResultSet::getColumnInternal(const int8_t* buff,
                                                 const size_t entry_idx,
                                                 const size_t target_logical_idx,
                                                 const StorageLookupResult& storage_lookup_result) const {
  CHECK(buff);
  const int8_t* rowwise_target_ptr{nullptr};
  const int8_t* keys_ptr{nullptr};
  const int8_t* crt_col_ptr{nullptr};
  const size_t key_width{query_mem_desc_.getEffectiveKeyWidth()};
  if (query_mem_desc_.output_columnar) {
    crt_col_ptr = get_cols_ptr(buff, query_mem_desc_);
  } else {
    keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, entry_idx);
    const auto key_bytes_with_padding = align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
    rowwise_target_ptr = keys_ptr + key_bytes_with_padding;
  }
  CHECK_LT(target_logical_idx, storage_->targets_.size());
  size_t agg_col_idx = 0;
  // TODO(alex): remove this loop, offsets can be computed only once
  for (size_t target_idx = 0; target_idx < storage_->targets_.size(); ++target_idx) {
    const auto& agg_info = storage_->targets_[target_idx];
    if (query_mem_desc_.output_columnar) {
      const auto next_col_ptr = advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc_, agg_col_idx);
      const auto col2_ptr = (agg_info.is_agg && agg_info.agg_kind == kAVG) ? next_col_ptr : nullptr;
      const auto compact_sz2 =
          (agg_info.is_agg && agg_info.agg_kind == kAVG) ? query_mem_desc_.agg_col_widths[agg_col_idx + 1].compact : 0;
      if (target_idx == target_logical_idx) {
        const auto compact_sz1 = query_mem_desc_.agg_col_widths[agg_col_idx].compact;
        const auto i1 =
            lazyReadInt(read_int_from_buff(columnar_elem_ptr(entry_idx, crt_col_ptr, compact_sz1), compact_sz1),
                        target_logical_idx,
                        storage_lookup_result);
        if (col2_ptr) {
          const auto i2 = read_int_from_buff(columnar_elem_ptr(entry_idx, col2_ptr, compact_sz2), compact_sz2);
          return InternalTargetValue(i1, i2);
        } else {
          return InternalTargetValue(
              agg_info.sql_type.is_fp() ? i1 : int_resize_cast(i1, agg_info.sql_type.get_logical_size()));
        }
      }
      crt_col_ptr = next_col_ptr;
      if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
        crt_col_ptr = advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
      }
    } else {
      auto ptr1 = rowwise_target_ptr;
      if (!query_mem_desc_.target_groupby_indices.empty()) {
        CHECK_LT(target_logical_idx, query_mem_desc_.target_groupby_indices.size());
        if (query_mem_desc_.target_groupby_indices[target_logical_idx] >= 0) {
          ptr1 = keys_ptr + query_mem_desc_.target_groupby_indices[target_logical_idx] * key_width;
        }
      }
      const auto compact_sz1 = query_mem_desc_.agg_col_widths[agg_col_idx].compact
                                   ? query_mem_desc_.agg_col_widths[agg_col_idx].compact
                                   : key_width;
      const int8_t* ptr2{nullptr};
      int8_t compact_sz2{0};
      if ((agg_info.is_agg && agg_info.agg_kind == kAVG) || is_real_str_or_array(agg_info)) {
        ptr2 = rowwise_target_ptr + query_mem_desc_.agg_col_widths[agg_col_idx].compact;
        compact_sz2 = query_mem_desc_.agg_col_widths[agg_col_idx + 1].compact;
      }
      if (target_idx == target_logical_idx) {
        const auto i1 = lazyReadInt(read_int_from_buff(ptr1, compact_sz1), target_logical_idx, storage_lookup_result);
        if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
          CHECK(ptr2);
          const auto i2 = read_int_from_buff(ptr2, compact_sz2);
          return InternalTargetValue(i1, i2);
        } else {
          if (agg_info.sql_type.is_string() && agg_info.sql_type.get_compression() == kENCODING_NONE) {
            CHECK(!agg_info.is_agg);
            if (!lazy_fetch_info_.empty()) {
              CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
              const auto& col_lazy_fetch = lazy_fetch_info_[target_logical_idx];
              if (col_lazy_fetch.is_lazily_fetched) {
                return InternalTargetValue(reinterpret_cast<const std::string*>(i1));
              }
            }
            if (none_encoded_strings_valid_) {
              if (i1 < 0) {
                CHECK_EQ(-1, i1);
                return InternalTargetValue(static_cast<const std::string*>(nullptr));
              }
              CHECK_LT(i1, none_encoded_strings_.size());
              CHECK_LT(storage_lookup_result.storage_idx, none_encoded_strings_.size());
              const auto& none_encoded_strings_for_fragment = none_encoded_strings_[storage_lookup_result.storage_idx];
              return InternalTargetValue(&none_encoded_strings_for_fragment[i1]);
            }
            CHECK(ptr2);
            const auto str_len = read_int_from_buff(ptr2, compact_sz2);
            CHECK_GE(str_len, 0);
            return getVarlenOrderEntry(i1, str_len);
          }
          return InternalTargetValue(
              agg_info.sql_type.is_fp() ? i1 : int_resize_cast(i1, agg_info.sql_type.get_logical_size()));
        }
      }
      rowwise_target_ptr =
          advance_target_ptr(rowwise_target_ptr, agg_info, agg_col_idx, query_mem_desc_, none_encoded_strings_valid_);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info, none_encoded_strings_valid_);
  }
  CHECK(false);
  return InternalTargetValue(int64_t(0));
}

InternalTargetValue ResultSet::getVarlenOrderEntry(const int64_t str_ptr, const size_t str_len) const {
  char* host_str_ptr{nullptr};
  std::vector<int8_t> cpu_buffer;
  if (device_type_ == ExecutorDeviceType::GPU) {
    cpu_buffer.resize(str_len);
    auto& data_mgr = query_mem_desc_.executor_->catalog_->get_dataMgr();
    copy_from_gpu(&data_mgr, &cpu_buffer[0], static_cast<CUdeviceptr>(str_ptr), str_len, device_id_);
    host_str_ptr = reinterpret_cast<char*>(&cpu_buffer[0]);
  } else {
    CHECK(device_type_ == ExecutorDeviceType::CPU);
    host_str_ptr = reinterpret_cast<char*>(str_ptr);
  }
  std::string str(host_str_ptr, str_len);
  return InternalTargetValue(row_set_mem_owner_->addString(str));
}

int64_t ResultSet::lazyReadInt(const int64_t ival,
                               const size_t target_logical_idx,
                               const StorageLookupResult& storage_lookup_result) const {
  if (!lazy_fetch_info_.empty()) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    const auto& col_lazy_fetch = lazy_fetch_info_[target_logical_idx];
    if (col_lazy_fetch.is_lazily_fetched) {
      CHECK_LT(static_cast<size_t>(storage_lookup_result.storage_idx), col_buffers_.size());
      int64_t ival_copy = ival;
      auto& frag_col_buffers =
          getColumnFrag(static_cast<size_t>(storage_lookup_result.storage_idx), target_logical_idx, ival_copy);
      auto& frag_col_buffer = frag_col_buffers[col_lazy_fetch.local_col_id];
      CHECK_LT(target_logical_idx, targets_.size());
      const TargetInfo& target_info = targets_[target_logical_idx];
      CHECK(!target_info.is_agg);
      if (target_info.sql_type.is_string() && target_info.sql_type.get_compression() == kENCODING_NONE) {
        VarlenDatum vd;
        bool is_end{false};
        ChunkIter_get_nth(reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffer)),
                          storage_lookup_result.fixedup_entry_idx,
                          false,
                          &vd,
                          &is_end);
        CHECK(!is_end);
        if (vd.is_null) {
          return 0;
        }
        std::string fetched_str(reinterpret_cast<char*>(vd.pointer), vd.length);
        return reinterpret_cast<int64_t>(row_set_mem_owner_->addString(fetched_str));
      }
      return lazy_decode(col_lazy_fetch, frag_col_buffer, ival_copy);
    }
  }
  return ival;
}

// Not all entries in the buffer represent a valid row. Advance the internal cursor
// used for the getNextRow method to the next row which is valid.
size_t ResultSet::advanceCursorToNextEntry() const {
  while (crt_row_buff_idx_ < entryCount()) {
    const auto entry_idx = permutation_.empty() ? crt_row_buff_idx_ : permutation_[crt_row_buff_idx_];
    const auto storage_lookup_result = findStorage(entry_idx);
    const auto storage = storage_lookup_result.storage_ptr;
    const auto fixedup_entry_idx = storage_lookup_result.fixedup_entry_idx;
    if (!storage->isEmptyEntry(fixedup_entry_idx)) {
      break;
    }
    ++crt_row_buff_idx_;
  }
  if (permutation_.empty()) {
    return crt_row_buff_idx_;
  }
  CHECK_LE(crt_row_buff_idx_, permutation_.size());
  return crt_row_buff_idx_ == permutation_.size() ? crt_row_buff_idx_ : permutation_[crt_row_buff_idx_];
}

size_t ResultSet::entryCount() const {
  return permutation_.empty() ? (query_mem_desc_.entry_count + query_mem_desc_.entry_count_small) : permutation_.size();
}

size_t ResultSet::getBufferSizeBytes(const ExecutorDeviceType device_type) const {
  CHECK(storage_);
  return storage_->query_mem_desc_.getBufferSizeBytes(device_type);
}

int64_t lazy_decode(const ColumnLazyFetchInfo& col_lazy_fetch, const int8_t* byte_stream, const int64_t pos) {
  CHECK(col_lazy_fetch.is_lazily_fetched);
  const auto& type_info = col_lazy_fetch.type;
  if (type_info.is_fp()) {
    if (type_info.get_type() == kFLOAT) {
      double fval = fixed_width_float_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<const int64_t*>(may_alias_ptr(&fval));
    } else {
      double fval = fixed_width_double_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<const int64_t*>(may_alias_ptr(&fval));
    }
  }
  CHECK(type_info.is_integer() || type_info.is_decimal() || type_info.is_time() || type_info.is_boolean() ||
        type_info.is_string());
  size_t type_bitwidth = get_bit_width(type_info);
  if (type_info.get_compression() == kENCODING_FIXED) {
    type_bitwidth = type_info.get_comp_param();
  } else if (type_info.get_compression() == kENCODING_DICT) {
    type_bitwidth = 8 * type_info.get_size();
  }
  CHECK_EQ(size_t(0), type_bitwidth % 8);
  auto val = type_info.get_compression() == kENCODING_DICT && type_info.get_size() < type_info.get_logical_size()
                 ? fixed_width_unsigned_decode_noinline(byte_stream, type_bitwidth / 8, pos)
                 : fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
  if (type_info.get_compression() != kENCODING_NONE) {
    CHECK(type_info.get_compression() == kENCODING_FIXED || type_info.get_compression() == kENCODING_DICT);
    auto encoding = type_info.get_compression();
    if (encoding == kENCODING_FIXED) {
      encoding = kENCODING_NONE;
    }
    SQLTypeInfo col_logical_ti(type_info.get_type(),
                               type_info.get_dimension(),
                               type_info.get_scale(),
                               false,
                               encoding,
                               0,
                               type_info.get_subtype());
    if (val == inline_fixed_encoding_null_val(type_info)) {
      return inline_int_null_val(col_logical_ti);
    }
  }
  return val;
}

namespace {

template <class T>
ScalarTargetValue make_scalar_tv(const T val) {
  return ScalarTargetValue(static_cast<int64_t>(val));
}

template <>
ScalarTargetValue make_scalar_tv(const float val) {
  return ScalarTargetValue(val);
}

template <>
ScalarTargetValue make_scalar_tv(const double val) {
  return ScalarTargetValue(val);
}

template <class T>
TargetValue build_array_target_value(const int8_t* buff,
                                     const size_t buff_sz,
                                     std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  std::vector<ScalarTargetValue> values;
  auto buff_elems = reinterpret_cast<const T*>(buff);
  CHECK_EQ(size_t(0), buff_sz % sizeof(T));
  const size_t num_elems = buff_sz / sizeof(T);
  for (size_t i = 0; i < num_elems; ++i) {
    values.push_back(make_scalar_tv<T>(buff_elems[i]));
  }
  return TargetValue(values);
}

TargetValue build_string_array_target_value(const int32_t* buff,
                                            const size_t buff_sz,
                                            const int dict_id,
                                            std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                            const Executor* executor) {
  std::vector<ScalarTargetValue> values;
  CHECK_EQ(size_t(0), buff_sz % sizeof(int32_t));
  const size_t num_elems = buff_sz / sizeof(int32_t);
  for (size_t i = 0; i < num_elems; ++i) {
    const auto string_id = buff[i];
    values.emplace_back(
        string_id == NULL_INT
            ? NullableString(nullptr)
            : NullableString(
                  executor->getStringDictionaryProxy(dict_id, row_set_mem_owner, false)->getString(string_id)));
  }
  return values;
}

TargetValue build_array_target_value(const SQLTypeInfo& array_ti,
                                     const int8_t* buff,
                                     const size_t buff_sz,
                                     std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                     const Executor* executor) {
  CHECK(array_ti.is_array());
  const auto& elem_ti = array_ti.get_elem_type();
  CHECK_EQ(elem_ti.get_size(), elem_ti.get_logical_size());  // no fixed encoding for arrays yet
  if (elem_ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, elem_ti.get_compression());
    CHECK_EQ(size_t(4), elem_ti.get_size());
    return build_string_array_target_value(
        reinterpret_cast<const int32_t*>(buff), buff_sz, elem_ti.get_comp_param(), row_set_mem_owner, executor);
  }
  switch (elem_ti.get_size()) {
    case 1:
      return build_array_target_value<int8_t>(buff, buff_sz, row_set_mem_owner);
    case 2:
      return build_array_target_value<int16_t>(buff, buff_sz, row_set_mem_owner);
    case 4:
      if (elem_ti.is_fp()) {
        return build_array_target_value<float>(buff, buff_sz, row_set_mem_owner);
      } else {
        return build_array_target_value<int32_t>(buff, buff_sz, row_set_mem_owner);
      }
    case 8:
      if (elem_ti.is_fp()) {
        return build_array_target_value<double>(buff, buff_sz, row_set_mem_owner);
      } else {
        return build_array_target_value<int64_t>(buff, buff_sz, row_set_mem_owner);
      }
    default:
      CHECK(false);
  }
  CHECK(false);
  return TargetValue(nullptr);
}

}  // namespace

const std::vector<const int8_t*>& ResultSet::getColumnFrag(const size_t storage_idx,
                                                           const size_t col_logical_idx,
                                                           int64_t& global_idx) const {
  CHECK_LT(static_cast<size_t>(storage_idx), col_buffers_.size());
#ifdef ENABLE_MULTIFRAG_JOIN
  if (col_buffers_[storage_idx].size() > 1) {
    int64_t frag_id = 0;
    int64_t local_idx = global_idx;
    if (consistent_frag_sizes_[storage_idx][col_logical_idx] != -1) {
      frag_id = global_idx / consistent_frag_sizes_[storage_idx][col_logical_idx];
      local_idx = global_idx % consistent_frag_sizes_[storage_idx][col_logical_idx];
    } else {
      std::tie(frag_id, local_idx) = get_frag_id_and_local_idx(frag_offsets_[storage_idx], col_logical_idx, global_idx);
      CHECK_LE(local_idx, global_idx);
    }
    CHECK_GE(frag_id, int64_t(0));
    CHECK_LT(frag_id, col_buffers_[storage_idx].size());
    global_idx = local_idx;
    return col_buffers_[storage_idx][frag_id];
  } else
#endif
  {
    CHECK_EQ(size_t(1), col_buffers_[storage_idx].size());
    return col_buffers_[storage_idx][0];
  }
}

// Interprets ptr1, ptr2 as the ptr and len pair used for variable length data.
TargetValue ResultSet::makeVarlenTargetValue(const int8_t* ptr1,
                                             const int8_t compact_sz1,
                                             const int8_t* ptr2,
                                             const int8_t compact_sz2,
                                             const TargetInfo& target_info,
                                             const size_t target_logical_idx,
                                             const size_t entry_buff_idx) const {
  auto varlen_ptr = read_int_from_buff(ptr1, compact_sz1);
  if (none_encoded_strings_valid_) {
    if (varlen_ptr < 0) {
      CHECK_EQ(-1, varlen_ptr);
      return TargetValue(nullptr);
    }
    const auto storage_idx = getStorageIndex(entry_buff_idx);
    CHECK_LT(static_cast<size_t>(storage_idx.first), none_encoded_strings_.size());
    const auto none_encoded_strings = none_encoded_strings_[storage_idx.first];
    CHECK_LT(static_cast<size_t>(varlen_ptr), none_encoded_strings.size());
    return none_encoded_strings[varlen_ptr];
  }
  if (!lazy_fetch_info_.empty()) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    const auto& col_lazy_fetch = lazy_fetch_info_[target_logical_idx];
    if (col_lazy_fetch.is_lazily_fetched) {
      const auto storage_idx = getStorageIndex(entry_buff_idx);
      CHECK_LT(static_cast<size_t>(storage_idx.first), col_buffers_.size());
      auto& frag_col_buffers = getColumnFrag(storage_idx.first, target_logical_idx, varlen_ptr);
      bool is_end{false};
      if (target_info.sql_type.is_string()) {
        VarlenDatum vd;
        ChunkIter_get_nth(
            reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffers[col_lazy_fetch.local_col_id])),
            varlen_ptr,
            false,
            &vd,
            &is_end);
        CHECK(!is_end);
        if (vd.is_null) {
          return TargetValue(nullptr);
        }
        CHECK(vd.pointer);
        CHECK_GT(vd.length, 0);
        std::string fetched_str(reinterpret_cast<char*>(vd.pointer), vd.length);
        return fetched_str;
      } else {
        CHECK(target_info.sql_type.is_array());
        ArrayDatum ad;
        ChunkIter_get_nth(
            reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffers[col_lazy_fetch.local_col_id])),
            varlen_ptr,
            &ad,
            &is_end);
        CHECK(!is_end);
        if (ad.is_null) {
          std::vector<ScalarTargetValue> empty_array;
          return TargetValue(empty_array);
        }
        CHECK(ad.pointer);
        CHECK_GT(ad.length, 0);
        return build_array_target_value(target_info.sql_type, ad.pointer, ad.length, row_set_mem_owner_, executor_);
      }
    }
  }
  if (!varlen_ptr) {
    return TargetValue(nullptr);
  }
  auto length = read_int_from_buff(ptr2, compact_sz2);
  if (target_info.sql_type.is_array()) {
    const auto& elem_ti = target_info.sql_type.get_elem_type();
    length *= elem_ti.get_logical_size();
  }
  std::vector<int8_t> cpu_buffer;
  if (varlen_ptr && device_type_ == ExecutorDeviceType::GPU) {
    cpu_buffer.resize(length);
    auto& data_mgr = query_mem_desc_.executor_->catalog_->get_dataMgr();
    copy_from_gpu(&data_mgr, &cpu_buffer[0], static_cast<CUdeviceptr>(varlen_ptr), length, device_id_);
    varlen_ptr = reinterpret_cast<int64_t>(&cpu_buffer[0]);
  }
  if (target_info.sql_type.is_array()) {
    return build_array_target_value(
        target_info.sql_type, reinterpret_cast<const int8_t*>(varlen_ptr), length, row_set_mem_owner_, executor_);
  }
  return std::string(reinterpret_cast<char*>(varlen_ptr), length);
}

// Reads an integer or a float from ptr based on the type and the byte width.
TargetValue ResultSet::makeTargetValue(const int8_t* ptr,
                                       const int8_t compact_sz,
                                       const TargetInfo& target_info,
                                       const size_t target_logical_idx,
                                       const bool translate_strings,
                                       const bool decimal_to_double,
                                       const size_t entry_buff_idx) const {
  const bool float_argument_input = takes_float_argument(target_info);
  const auto actual_compact_sz = float_argument_input ? sizeof(float) : compact_sz;

  auto ival = read_int_from_buff(ptr, actual_compact_sz);
  const auto& chosen_type = get_compact_type(target_info);
  if (!lazy_fetch_info_.empty()) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    const auto& col_lazy_fetch = lazy_fetch_info_[target_logical_idx];
    if (col_lazy_fetch.is_lazily_fetched) {
      const auto storage_idx = getStorageIndex(entry_buff_idx);
      CHECK_LT(static_cast<size_t>(storage_idx.first), col_buffers_.size());
      auto& frag_col_buffers = getColumnFrag(storage_idx.first, target_logical_idx, ival);
      ival = lazy_decode(col_lazy_fetch, frag_col_buffers[col_lazy_fetch.local_col_id], ival);
      if (chosen_type.is_fp()) {
        const auto dval = *reinterpret_cast<const double*>(may_alias_ptr(&ival));
        if (chosen_type.get_type() == kFLOAT) {
          return ScalarTargetValue(static_cast<float>(dval));
        } else {
          return ScalarTargetValue(dval);
        }
      }
    }
  }
  if (chosen_type.is_fp()) {
    switch (actual_compact_sz) {
      case 8: {
        const auto dval = *reinterpret_cast<const double*>(ptr);
        return chosen_type.get_type() == kFLOAT ? ScalarTargetValue(static_cast<const float>(dval))
                                                : ScalarTargetValue(dval);
      }
      case 4: {
        CHECK_EQ(kFLOAT, chosen_type.get_type());
        return *reinterpret_cast<const float*>(ptr);
      }
      default:
        CHECK(false);
    }
  }
  if (chosen_type.is_integer() | chosen_type.is_boolean() || chosen_type.is_time() || chosen_type.is_timeinterval()) {
    if (is_distinct_target(target_info)) {
      return TargetValue(
          count_distinct_set_size(ival, target_logical_idx, query_mem_desc_.count_distinct_descriptors_));
    }
    // TODO(alex): remove int_resize_cast, make read_int_from_buff return the right type instead
    if (inline_int_null_val(chosen_type) == int_resize_cast(ival, chosen_type.get_logical_size())) {
      return inline_int_null_val(target_info.sql_type);
    }
    return ival;
  }
  if (chosen_type.is_string() && chosen_type.get_compression() == kENCODING_DICT) {
    if (translate_strings) {
      if (static_cast<int32_t>(ival) == NULL_INT) {  // TODO(alex): this isn't nice, fix it
        return NullableString(nullptr);
      }
      StringDictionaryProxy* sdp{nullptr};
      if (!chosen_type.get_comp_param()) {
        sdp = row_set_mem_owner_->getLiteralStringDictProxy();
      } else {
        sdp = executor_ ? executor_->getStringDictionaryProxy(chosen_type.get_comp_param(), row_set_mem_owner_, false)
                        : row_set_mem_owner_->getStringDictProxy(chosen_type.get_comp_param());
      }
      return NullableString(sdp->getString(ival));
    } else {
      return static_cast<int64_t>(static_cast<int32_t>(ival));
    }
  }
  if (chosen_type.is_decimal()) {
    if (decimal_to_double) {
      if (ival == inline_int_null_val(SQLTypeInfo(decimal_to_int_type(chosen_type), false))) {
        return NULL_DOUBLE;
      }
      return static_cast<double>(ival) / exp_to_scale(chosen_type.get_scale());
    }
    return ival;
  }
  CHECK(false);
  return TargetValue(int64_t(0));
}

// Gets the TargetValue stored at position entry_idx in the col1_ptr and col2_ptr
// column buffers. The second column is only used for AVG.
TargetValue ResultSet::getTargetValueFromBufferColwise(const int8_t* col1_ptr,
                                                       const int8_t compact_sz1,
                                                       const int8_t* col2_ptr,
                                                       const int8_t compact_sz2,
                                                       const size_t entry_idx,
                                                       const TargetInfo& target_info,
                                                       const size_t target_logical_idx,
                                                       const bool translate_strings,
                                                       const bool decimal_to_double) const {
  CHECK(query_mem_desc_.output_columnar);
  const auto ptr1 = columnar_elem_ptr(entry_idx, col1_ptr, compact_sz1);
  if (target_info.agg_kind == kAVG || is_real_str_or_array(target_info)) {
    CHECK(col2_ptr);
    CHECK(compact_sz2);
    const auto ptr2 = columnar_elem_ptr(entry_idx, col2_ptr, compact_sz2);
    return target_info.agg_kind == kAVG
               ? make_avg_target_value(ptr1, compact_sz1, ptr2, compact_sz2, target_info)
               : makeVarlenTargetValue(
                     ptr1, compact_sz1, ptr2, compact_sz2, target_info, target_logical_idx, entry_idx);
  }
  return makeTargetValue(
      ptr1, compact_sz1, target_info, target_logical_idx, translate_strings, decimal_to_double, entry_idx);
}

// Gets the TargetValue stored in slot_idx (and slot_idx for AVG) of rowwise_target_ptr.
TargetValue ResultSet::getTargetValueFromBufferRowwise(int8_t* rowwise_target_ptr,
                                                       int8_t* keys_ptr,
                                                       const size_t entry_buff_idx,
                                                       const TargetInfo& target_info,
                                                       const size_t target_logical_idx,
                                                       const size_t slot_idx,
                                                       const bool translate_strings,
                                                       const bool decimal_to_double,
                                                       const bool fixup_count_distinct_pointers) const {
  if (UNLIKELY(fixup_count_distinct_pointers) && is_distinct_target(target_info)) {
    auto count_distinct_ptr_ptr = reinterpret_cast<int64_t*>(rowwise_target_ptr);
    const auto remote_ptr = *count_distinct_ptr_ptr;
    if (remote_ptr) {
      const auto ptr = storage_->mappedPtr(remote_ptr);
      if (ptr) {
        *count_distinct_ptr_ptr = ptr;
      } else {
        // need to create a zero filled buffer for this remote_ptr
        const auto& count_distinct_desc = query_mem_desc_.count_distinct_descriptors_[target_logical_idx];
        const auto bitmap_byte_sz = count_distinct_desc.bitmapSizeBytes();

        auto count_distinct_buffer = static_cast<int8_t*>(checked_malloc(bitmap_byte_sz));
        memset(count_distinct_buffer, 0, bitmap_byte_sz);
        row_set_mem_owner_->addCountDistinctBuffer(count_distinct_buffer, bitmap_byte_sz, true);
        *count_distinct_ptr_ptr = reinterpret_cast<int64_t>(count_distinct_buffer);
      }
    }
    return int64_t(0);
  }
  auto ptr1 = rowwise_target_ptr;
  CHECK_LT(slot_idx, query_mem_desc_.agg_col_widths.size());
  auto compact_sz1 = query_mem_desc_.agg_col_widths[slot_idx].compact;
  if (target_info.agg_kind == kAVG || is_real_str_or_array(target_info)) {
    const auto ptr2 = rowwise_target_ptr + query_mem_desc_.agg_col_widths[slot_idx].compact;
    const auto compact_sz2 = query_mem_desc_.agg_col_widths[slot_idx + 1].compact;
    CHECK(ptr2);
    return target_info.agg_kind == kAVG
               ? make_avg_target_value(ptr1, compact_sz1, ptr2, compact_sz2, target_info)
               : makeVarlenTargetValue(
                     ptr1, compact_sz1, ptr2, compact_sz2, target_info, target_logical_idx, entry_buff_idx);
  }
  if (query_mem_desc_.target_groupby_indices.empty()) {
    return makeTargetValue(
        ptr1, compact_sz1, target_info, target_logical_idx, translate_strings, decimal_to_double, entry_buff_idx);
  }
  CHECK_LT(target_logical_idx, query_mem_desc_.target_groupby_indices.size());
  if (query_mem_desc_.target_groupby_indices[target_logical_idx] < 0) {
    return makeTargetValue(
        ptr1, compact_sz1, target_info, target_logical_idx, translate_strings, decimal_to_double, entry_buff_idx);
  }
  const auto key_width = query_mem_desc_.getEffectiveKeyWidth();
  ptr1 = keys_ptr + query_mem_desc_.target_groupby_indices[target_logical_idx] * key_width;
  return makeTargetValue(
      ptr1, key_width, target_info, target_logical_idx, translate_strings, decimal_to_double, entry_buff_idx);
}

// Returns true iff the entry at position entry_idx in buff contains a valid row.
bool ResultSetStorage::isEmptyEntry(const size_t entry_idx, const int8_t* buff) const {
  if (GroupByColRangeType::Scan == query_mem_desc_.hash_type) {
    return false;
  }
  if (query_mem_desc_.keyless_hash) {
    CHECK(query_mem_desc_.hash_type == GroupByColRangeType::OneColKnownRange ||
          query_mem_desc_.hash_type == GroupByColRangeType::MultiColPerfectHash);
    CHECK_GE(query_mem_desc_.idx_target_as_key, 0);
    CHECK_LT(static_cast<size_t>(query_mem_desc_.idx_target_as_key), target_init_vals_.size());
    if (query_mem_desc_.output_columnar) {
      const auto col_buff =
          advance_col_buff_to_slot(buff, query_mem_desc_, targets_, query_mem_desc_.idx_target_as_key, false);
      const auto entry_buff =
          col_buff + entry_idx * query_mem_desc_.agg_col_widths[query_mem_desc_.idx_target_as_key].compact;
      return read_int_from_buff(entry_buff,
                                query_mem_desc_.agg_col_widths[query_mem_desc_.idx_target_as_key].compact) ==
             target_init_vals_[query_mem_desc_.idx_target_as_key];
    }
    const auto key_bytes_with_padding = align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
    const auto rowwise_target_ptr = row_ptr_rowwise(buff, query_mem_desc_, entry_idx) + key_bytes_with_padding;
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
  switch (query_mem_desc_.getEffectiveKeyWidth()) {
    case 4:
      CHECK(GroupByColRangeType::MultiCol == query_mem_desc_.hash_type);
      return *reinterpret_cast<const int32_t*>(keys_ptr) == EMPTY_KEY_32;
    case 8:
      return *reinterpret_cast<const int64_t*>(keys_ptr) == EMPTY_KEY_64;
    default:
      CHECK(false);
      return true;
  }
}

bool ResultSetStorage::isEmptyEntry(const size_t entry_idx) const {
  return isEmptyEntry(entry_idx, buff_);
}

bool ResultSet::isNull(const SQLTypeInfo& ti, const InternalTargetValue& val, const bool float_argument_input) {
  if (ti.get_notnull()) {
    return false;
  }
  if (val.isInt()) {
    return val.i1 == null_val_bit_pattern(ti, float_argument_input);
  }
  if (val.isPair()) {
    return !val.i2 || pair_to_double({val.i1, val.i2}, ti, float_argument_input) == NULL_DOUBLE;
  }
  if (val.isStr()) {
    return !val.i1;
  }
  CHECK(val.isNull());
  return true;
}
