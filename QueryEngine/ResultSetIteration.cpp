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

#include "../Shared/geo_types.h"
#include "../Shared/likely.h"
#include "Execute.h"
#include "ResultRows.h"
#include "ResultSet.h"
#include "ResultSetGeoSerialization.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"
#include "TypePunning.h"

#include <utility>

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
size_t get_byteoff_of_slot(const size_t slot_idx,
                           const QueryMemoryDescriptor& query_mem_desc) {
  size_t result = 0;
  for (size_t i = 0; i < slot_idx; ++i) {
    result += query_mem_desc.getColumnWidth(i).compact;
  }
  return result;
}

// Given the entire buffer for the result set, buff, finds the beginning of the
// column for slot_idx. Only makes sense for column-wise representation.
const int8_t* advance_col_buff_to_slot(const int8_t* buff,
                                       const QueryMemoryDescriptor& query_mem_desc,
                                       const std::vector<TargetInfo>& targets,
                                       const size_t slot_idx,
                                       const bool separate_varlen_storage) {
  auto crt_col_ptr = get_cols_ptr(buff, query_mem_desc);
  const auto buffer_col_count = query_mem_desc.getBufferColSlotCount();
  size_t agg_col_idx{0};
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    if (agg_col_idx == slot_idx) {
      return crt_col_ptr;
    }
    CHECK_LT(agg_col_idx, buffer_col_count);
    const auto& agg_info = targets[target_idx];
    crt_col_ptr =
        advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc, agg_col_idx);
    if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
      if (agg_col_idx + 1 == slot_idx) {
        return crt_col_ptr;
      }
      crt_col_ptr = advance_to_next_columnar_target_buff(
          crt_col_ptr, query_mem_desc, agg_col_idx + 1);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info, separate_varlen_storage);
  }
  CHECK(false);
  return nullptr;
}

}  // namespace

std::vector<TargetValue> ResultSet::getRowAt(
    const size_t global_entry_idx,
    const bool translate_strings,
    const bool decimal_to_double,
    const bool fixup_count_distinct_pointers,
    const bool skip_non_lazy_columns /* = false*/) const {
  const auto storage_lookup_result =
      fixup_count_distinct_pointers
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
  if (query_mem_desc_.didOutputColumnar()) {
    keys_ptr = buff;
    crt_col_ptr = get_cols_ptr(buff, storage->query_mem_desc_);
  } else {
    keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, local_entry_idx);
    const auto key_bytes_with_padding =
        align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
    rowwise_target_ptr = keys_ptr + key_bytes_with_padding;
  }
  for (size_t target_idx = 0; target_idx < storage_->targets_.size(); ++target_idx) {
    const auto& agg_info = storage_->targets_[target_idx];
    if (query_mem_desc_.didOutputColumnar()) {
      if (skip_non_lazy_columns) {
        row.push_back(!lazy_fetch_info_.empty() &&
                              lazy_fetch_info_[target_idx].is_lazily_fetched
                          ? getTargetValueFromBufferColwise(crt_col_ptr,
                                                            keys_ptr,
                                                            storage->query_mem_desc_,
                                                            local_entry_idx,
                                                            global_entry_idx,
                                                            agg_info,
                                                            target_idx,
                                                            agg_col_idx,
                                                            translate_strings,
                                                            decimal_to_double)
                          : nullptr);
      } else {
        row.push_back(getTargetValueFromBufferColwise(crt_col_ptr,
                                                      keys_ptr,
                                                      storage->query_mem_desc_,
                                                      local_entry_idx,
                                                      global_entry_idx,
                                                      agg_info,
                                                      target_idx,
                                                      agg_col_idx,
                                                      translate_strings,
                                                      decimal_to_double));
      }
      crt_col_ptr = advance_target_ptr_col_wise(crt_col_ptr,
                                                agg_info,
                                                agg_col_idx,
                                                storage->query_mem_desc_,
                                                separate_varlen_storage_valid_);
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
      rowwise_target_ptr = advance_target_ptr_row_wise(rowwise_target_ptr,
                                                       agg_info,
                                                       agg_col_idx,
                                                       query_mem_desc_,
                                                       separate_varlen_storage_valid_);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info, separate_varlen_storage_valid_);
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
  CHECK(!query_mem_desc_.didOutputColumnar());
  const auto keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, local_entry_idx);
  const auto key_bytes_with_padding =
      align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
  const auto rowwise_target_ptr = keys_ptr + key_bytes_with_padding;
  const auto tv = getTargetValueFromBufferRowwise(rowwise_target_ptr,
                                                  keys_ptr,
                                                  global_entry_idx,
                                                  targets_.front(),
                                                  0,
                                                  0,
                                                  false,
                                                  false,
                                                  false);
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
  const auto entry_idx =
      permutation_.empty() ? logical_index : permutation_[logical_index];
  return getRowAt(entry_idx, true, false, false);
}

std::vector<TargetValue> ResultSet::getRowAtNoTranslations(
    const size_t logical_index,
    const bool skip_non_lazy_columns /* = false*/) const {
  if (logical_index >= entryCount()) {
    return {};
  }
  const auto entry_idx =
      permutation_.empty() ? logical_index : permutation_[logical_index];
  return getRowAt(entry_idx, false, false, false, skip_non_lazy_columns);
}

bool ResultSet::isRowAtEmpty(const size_t logical_index) const {
  if (logical_index >= entryCount()) {
    return true;
  }
  const auto entry_idx =
      permutation_.empty() ? logical_index : permutation_[logical_index];
  const auto storage_lookup_result = findStorage(entry_idx);
  const auto storage = storage_lookup_result.storage_ptr;
  const auto local_entry_idx = storage_lookup_result.fixedup_entry_idx;
  return storage->isEmptyEntry(local_entry_idx);
}

std::vector<TargetValue> ResultSet::getNextRow(const bool translate_strings,
                                               const bool decimal_to_double) const {
  std::lock_guard<std::mutex> lock(row_iteration_mutex_);
  if (!storage_ && !just_explain_) {
    return {};
  }
  return getNextRowUnlocked(translate_strings, decimal_to_double);
}

std::vector<TargetValue> ResultSet::getNextRowUnlocked(
    const bool translate_strings,
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

std::vector<TargetValue> ResultSet::getNextRowImpl(const bool translate_strings,
                                                   const bool decimal_to_double) const {
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

const int8_t* columnar_elem_ptr(const size_t entry_idx,
                                const int8_t* col1_ptr,
                                const int8_t compact_sz1) {
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

void ResultSet::RowWiseTargetAccessor::initializeOffsetsForStorage() {
  // Compute offsets for base storage and all appended storage
  for (size_t storage_idx = 0; storage_idx < result_set_->appended_storage_.size() + 1;
       ++storage_idx) {
    offsets_for_storage_.emplace_back();

    const int8_t* rowwise_target_ptr{0};

    size_t agg_col_idx = 0;
    for (size_t target_idx = 0; target_idx < result_set_->storage_->targets_.size();
         ++target_idx) {
      const auto& agg_info = result_set_->storage_->targets_[target_idx];

      auto ptr1 = rowwise_target_ptr;
      const auto compact_sz1 =
          result_set_->query_mem_desc_.getColumnWidth(agg_col_idx).compact
              ? result_set_->query_mem_desc_.getColumnWidth(agg_col_idx).compact
              : key_width_;

      const int8_t* ptr2{nullptr};
      int8_t compact_sz2{0};
      if ((agg_info.is_agg && agg_info.agg_kind == kAVG)) {
        ptr2 = ptr1 + compact_sz1;
        compact_sz2 =
            result_set_->query_mem_desc_.getColumnWidth(agg_col_idx + 1).compact;
      } else if (is_real_str_or_array(agg_info)) {
        ptr2 = ptr1 + compact_sz1;
        if (!result_set_->separate_varlen_storage_valid_) {
          // None encoded strings explicitly attached to ResultSetStorage do not have a
          // second slot in the QueryMemoryDescriptor col width vector
          compact_sz2 =
              result_set_->query_mem_desc_.getColumnWidth(agg_col_idx + 1).compact;
        }
      }
      offsets_for_storage_[storage_idx].push_back(
          TargetOffsets{ptr1,
                        static_cast<size_t>(compact_sz1),
                        ptr2,
                        static_cast<size_t>(compact_sz2)});
      rowwise_target_ptr =
          advance_target_ptr_row_wise(rowwise_target_ptr,
                                      agg_info,
                                      agg_col_idx,
                                      result_set_->query_mem_desc_,
                                      result_set_->separate_varlen_storage_valid_);

      agg_col_idx = advance_slot(
          agg_col_idx, agg_info, result_set_->separate_varlen_storage_valid_);
    }
    CHECK_EQ(offsets_for_storage_[storage_idx].size(),
             result_set_->storage_->targets_.size());
  }
}

InternalTargetValue ResultSet::RowWiseTargetAccessor::getColumnInternal(
    const int8_t* buff,
    const size_t entry_idx,
    const size_t target_logical_idx,
    const StorageLookupResult& storage_lookup_result) const {
  CHECK(buff);
  const int8_t* rowwise_target_ptr{nullptr};
  const int8_t* keys_ptr{nullptr};

  const size_t storage_idx = storage_lookup_result.storage_idx;

  CHECK_LT(storage_idx, offsets_for_storage_.size());
  CHECK_LT(target_logical_idx, offsets_for_storage_[storage_idx].size());

  const auto& offsets_for_target = offsets_for_storage_[storage_idx][target_logical_idx];
  const auto& agg_info = result_set_->storage_->targets_[target_logical_idx];

  keys_ptr = get_rowwise_ptr(buff, entry_idx);
  rowwise_target_ptr = keys_ptr + key_bytes_with_padding_;
  auto ptr1 = rowwise_target_ptr + reinterpret_cast<size_t>(offsets_for_target.ptr1);
  if (result_set_->query_mem_desc_.targetGroupbyIndicesSize() > 0) {
    if (result_set_->query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) >= 0) {
      ptr1 = keys_ptr +
             result_set_->query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) *
                 key_width_;
    }
  }
  const auto i1 =
      result_set_->lazyReadInt(read_int_from_buff(ptr1, offsets_for_target.compact_sz1),
                               target_logical_idx,
                               storage_lookup_result);

  if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
    CHECK(offsets_for_target.ptr2);
    const auto ptr2 =
        rowwise_target_ptr + reinterpret_cast<size_t>(offsets_for_target.ptr2);
    const auto i2 = read_int_from_buff(ptr2, offsets_for_target.compact_sz2);
    return InternalTargetValue(i1, i2);
  } else {
    if (agg_info.sql_type.is_string() &&
        agg_info.sql_type.get_compression() == kENCODING_NONE) {
      CHECK(!agg_info.is_agg);
      if (!result_set_->lazy_fetch_info_.empty()) {
        CHECK_LT(target_logical_idx, result_set_->lazy_fetch_info_.size());
        const auto& col_lazy_fetch = result_set_->lazy_fetch_info_[target_logical_idx];
        if (col_lazy_fetch.is_lazily_fetched) {
          return InternalTargetValue(reinterpret_cast<const std::string*>(i1));
        }
      }
      if (result_set_->separate_varlen_storage_valid_) {
        if (i1 < 0) {
          CHECK_EQ(-1, i1);
          return InternalTargetValue(static_cast<const std::string*>(nullptr));
        }
        CHECK_LT(storage_lookup_result.storage_idx,
                 result_set_->serialized_varlen_buffer_.size());
        const auto& varlen_buffer_for_fragment =
            result_set_->serialized_varlen_buffer_[storage_lookup_result.storage_idx];
        CHECK_LT(i1, varlen_buffer_for_fragment.size());
        return InternalTargetValue(&varlen_buffer_for_fragment[i1]);
      }
      CHECK(offsets_for_target.ptr2);
      const auto ptr2 =
          rowwise_target_ptr + reinterpret_cast<size_t>(offsets_for_target.ptr2);
      const auto str_len = read_int_from_buff(ptr2, offsets_for_target.compact_sz2);
      CHECK_GE(str_len, 0);
      return result_set_->getVarlenOrderEntry(i1, str_len);
    }
    return InternalTargetValue(
        agg_info.sql_type.is_fp()
            ? i1
            : int_resize_cast(i1, agg_info.sql_type.get_logical_size()));
  }
}

void ResultSet::ColumnWiseTargetAccessor::initializeOffsetsForStorage() {
  // Compute offsets for base storage and all appended storage
  const auto key_width = result_set_->query_mem_desc_.getEffectiveKeyWidth();
  for (size_t storage_idx = 0; storage_idx < result_set_->appended_storage_.size() + 1;
       ++storage_idx) {
    offsets_for_storage_.emplace_back();

    const int8_t* buff = storage_idx == 0
                             ? result_set_->storage_->buff_
                             : result_set_->appended_storage_[storage_idx - 1]->buff_;
    CHECK(buff);

    const auto& crt_query_mem_desc =
        storage_idx == 0
            ? result_set_->storage_->query_mem_desc_
            : result_set_->appended_storage_[storage_idx - 1]->query_mem_desc_;
    const int8_t* crt_col_ptr = get_cols_ptr(buff, crt_query_mem_desc);

    size_t agg_col_idx = 0;
    for (size_t target_idx = 0; target_idx < result_set_->storage_->targets_.size();
         ++target_idx) {
      const auto& agg_info = result_set_->storage_->targets_[target_idx];

      const auto compact_sz1 =
          crt_query_mem_desc.getColumnWidth(agg_col_idx).compact
              ? crt_query_mem_desc.getPaddedColumnWidthBytes(agg_col_idx)
              : key_width;

      const auto next_col_ptr = advance_to_next_columnar_target_buff(
          crt_col_ptr, crt_query_mem_desc, agg_col_idx);
      const bool uses_two_slots = (agg_info.is_agg && agg_info.agg_kind == kAVG) ||
                                  is_real_str_or_array(agg_info);
      const auto col2_ptr = uses_two_slots ? next_col_ptr : nullptr;
      const auto compact_sz2 =
          (agg_info.is_agg && agg_info.agg_kind == kAVG) || is_real_str_or_array(agg_info)
              ? crt_query_mem_desc.getPaddedColumnWidthBytes(agg_col_idx + 1)
              : 0;

      offsets_for_storage_[storage_idx].push_back(
          TargetOffsets{crt_col_ptr,
                        static_cast<size_t>(compact_sz1),
                        col2_ptr,
                        static_cast<size_t>(compact_sz2)});

      crt_col_ptr = next_col_ptr;
      if (uses_two_slots) {
        crt_col_ptr = advance_to_next_columnar_target_buff(
            crt_col_ptr, crt_query_mem_desc, agg_col_idx + 1);
      }
      agg_col_idx = advance_slot(
          agg_col_idx, agg_info, result_set_->separate_varlen_storage_valid_);
    }
    CHECK_EQ(offsets_for_storage_[storage_idx].size(),
             result_set_->storage_->targets_.size());
  }
}

InternalTargetValue ResultSet::ColumnWiseTargetAccessor::getColumnInternal(
    const int8_t* buff,
    const size_t entry_idx,
    const size_t target_logical_idx,
    const StorageLookupResult& storage_lookup_result) const {
  const size_t storage_idx = storage_lookup_result.storage_idx;

  CHECK_LT(storage_idx, offsets_for_storage_.size());
  CHECK_LT(target_logical_idx, offsets_for_storage_[storage_idx].size());

  const auto& offsets_for_target = offsets_for_storage_[storage_idx][target_logical_idx];
  const auto& agg_info = result_set_->storage_->targets_[target_logical_idx];
  auto ptr1 = offsets_for_target.ptr1;
  if (result_set_->query_mem_desc_.targetGroupbyIndicesSize() > 0) {
    if (result_set_->query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) >= 0) {
      ptr1 =
          buff + result_set_->query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) *
                     result_set_->query_mem_desc_.getEffectiveKeyWidth() *
                     result_set_->query_mem_desc_.entry_count_;
    }
  }

  const auto i1 = result_set_->lazyReadInt(
      read_int_from_buff(
          columnar_elem_ptr(entry_idx, ptr1, offsets_for_target.compact_sz1),
          offsets_for_target.compact_sz1),
      target_logical_idx,
      storage_lookup_result);
  if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
    CHECK(offsets_for_target.ptr2);
    const auto i2 = read_int_from_buff(
        columnar_elem_ptr(
            entry_idx, offsets_for_target.ptr2, offsets_for_target.compact_sz2),
        offsets_for_target.compact_sz2);
    return InternalTargetValue(i1, i2);
  } else {
    // for TEXT ENCODING NONE:
    if (agg_info.sql_type.is_string() &&
        agg_info.sql_type.get_compression() == kENCODING_NONE) {
      CHECK(!agg_info.is_agg);
      if (!result_set_->lazy_fetch_info_.empty()) {
        CHECK_LT(target_logical_idx, result_set_->lazy_fetch_info_.size());
        const auto& col_lazy_fetch = result_set_->lazy_fetch_info_[target_logical_idx];
        if (col_lazy_fetch.is_lazily_fetched) {
          return InternalTargetValue(reinterpret_cast<const std::string*>(i1));
        }
      }
      if (result_set_->separate_varlen_storage_valid_) {
        if (i1 < 0) {
          CHECK_EQ(-1, i1);
          return InternalTargetValue(static_cast<const std::string*>(nullptr));
        }
        CHECK_LT(storage_lookup_result.storage_idx,
                 result_set_->serialized_varlen_buffer_.size());
        const auto& varlen_buffer_for_fragment =
            result_set_->serialized_varlen_buffer_[storage_lookup_result.storage_idx];
        CHECK_LT(i1, varlen_buffer_for_fragment.size());
        return InternalTargetValue(&varlen_buffer_for_fragment[i1]);
      }
      CHECK(offsets_for_target.ptr2);
      const auto i2 = read_int_from_buff(
          columnar_elem_ptr(
              entry_idx, offsets_for_target.ptr2, offsets_for_target.compact_sz2),
          offsets_for_target.compact_sz2);
      CHECK_GE(i2, 0);
      return result_set_->getVarlenOrderEntry(i1, i2);
    }
    return InternalTargetValue(
        agg_info.sql_type.is_fp()
            ? i1
            : int_resize_cast(i1, agg_info.sql_type.get_logical_size()));
  }
}

InternalTargetValue ResultSet::getVarlenOrderEntry(const int64_t str_ptr,
                                                   const size_t str_len) const {
  char* host_str_ptr{nullptr};
  std::vector<int8_t> cpu_buffer;
  if (device_type_ == ExecutorDeviceType::GPU) {
    cpu_buffer.resize(str_len);
    const auto executor = query_mem_desc_.getExecutor();
    CHECK(executor);
    auto& data_mgr = executor->catalog_->getDataMgr();
    copy_from_gpu(&data_mgr,
                  &cpu_buffer[0],
                  static_cast<CUdeviceptr>(str_ptr),
                  str_len,
                  device_id_);
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
      CHECK_LT(static_cast<size_t>(storage_lookup_result.storage_idx),
               col_buffers_.size());
      int64_t ival_copy = ival;
      auto& frag_col_buffers =
          getColumnFrag(static_cast<size_t>(storage_lookup_result.storage_idx),
                        target_logical_idx,
                        ival_copy);
      auto& frag_col_buffer = frag_col_buffers[col_lazy_fetch.local_col_id];
      CHECK_LT(target_logical_idx, targets_.size());
      const TargetInfo& target_info = targets_[target_logical_idx];
      CHECK(!target_info.is_agg);
      if (target_info.sql_type.is_string() &&
          target_info.sql_type.get_compression() == kENCODING_NONE) {
        VarlenDatum vd;
        bool is_end{false};
        ChunkIter_get_nth(
            reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffer)),
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
void ResultSet::advanceCursorToNextEntry(ResultSetRowIterator& iter) const {
  if (keep_first_ && iter.fetched_so_far_ >= drop_first_ + keep_first_) {
    iter.global_entry_idx_valid_ = false;
    return;
  }

  while (iter.crt_row_buff_idx_ < entryCount()) {
    const auto entry_idx = permutation_.empty() ? iter.crt_row_buff_idx_
                                                : permutation_[iter.crt_row_buff_idx_];
    const auto storage_lookup_result = findStorage(entry_idx);
    const auto storage = storage_lookup_result.storage_ptr;
    const auto fixedup_entry_idx = storage_lookup_result.fixedup_entry_idx;
    if (!storage->isEmptyEntry(fixedup_entry_idx)) {
      if (iter.fetched_so_far_ < drop_first_) {
        ++iter.fetched_so_far_;
      } else {
        break;
      }
    }
    ++iter.crt_row_buff_idx_;
  }
  if (permutation_.empty()) {
    iter.global_entry_idx_ = iter.crt_row_buff_idx_;
  } else {
    CHECK_LE(iter.crt_row_buff_idx_, permutation_.size());
    iter.global_entry_idx_ = iter.crt_row_buff_idx_ == permutation_.size()
                                 ? iter.crt_row_buff_idx_
                                 : permutation_[iter.crt_row_buff_idx_];
  }

  iter.global_entry_idx_valid_ = iter.crt_row_buff_idx_ < entryCount();

  if (iter.global_entry_idx_valid_) {
    ++iter.crt_row_buff_idx_;
    ++iter.fetched_so_far_;
  }
}

// Not all entries in the buffer represent a valid row. Advance the internal cursor
// used for the getNextRow method to the next row which is valid.
size_t ResultSet::advanceCursorToNextEntry() const {
  while (crt_row_buff_idx_ < entryCount()) {
    const auto entry_idx =
        permutation_.empty() ? crt_row_buff_idx_ : permutation_[crt_row_buff_idx_];
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
  return crt_row_buff_idx_ == permutation_.size() ? crt_row_buff_idx_
                                                  : permutation_[crt_row_buff_idx_];
}

size_t ResultSet::entryCount() const {
  return permutation_.empty() ? query_mem_desc_.getEntryCount() : permutation_.size();
}

size_t ResultSet::getBufferSizeBytes(const ExecutorDeviceType device_type) const {
  CHECK(storage_);
  return storage_->query_mem_desc_.getBufferSizeBytes(device_type);
}

int64_t lazy_decode(const ColumnLazyFetchInfo& col_lazy_fetch,
                    const int8_t* byte_stream,
                    const int64_t pos) {
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
  CHECK(type_info.is_integer() || type_info.is_decimal() || type_info.is_time() ||
        type_info.is_boolean() || type_info.is_string() || type_info.is_array());
  size_t type_bitwidth = get_bit_width(type_info);
  if (type_info.get_compression() == kENCODING_FIXED) {
    type_bitwidth = type_info.get_comp_param();
  } else if (type_info.get_compression() == kENCODING_DICT) {
    type_bitwidth = 8 * type_info.get_size();
  }
  CHECK_EQ(size_t(0), type_bitwidth % 8);
  int64_t val;
  if (type_info.is_date_in_days()) {
    val = type_info.get_comp_param() == 16
              ? fixed_width_small_date_decode_noinline(
                    byte_stream, 2, NULL_SMALLINT, NULL_BIGINT, pos)
              : fixed_width_small_date_decode_noinline(
                    byte_stream, 4, NULL_INT, NULL_BIGINT, pos);
  } else {
    val = (type_info.get_compression() == kENCODING_DICT &&
           type_info.get_size() < type_info.get_logical_size() &&
           type_info.get_comp_param())
              ? fixed_width_unsigned_decode_noinline(byte_stream, type_bitwidth / 8, pos)
              : fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
  }
  if (type_info.get_compression() != kENCODING_NONE &&
      type_info.get_compression() != kENCODING_DATE_IN_DAYS) {
    CHECK(type_info.get_compression() == kENCODING_FIXED ||
          type_info.get_compression() == kENCODING_DICT);
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
TargetValue build_array_target_value(
    const int8_t* buff,
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

TargetValue build_string_array_target_value(
    const int32_t* buff,
    const size_t buff_sz,
    const int dict_id,
    const bool translate_strings,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const Executor* executor) {
  std::vector<ScalarTargetValue> values;
  CHECK_EQ(size_t(0), buff_sz % sizeof(int32_t));
  const size_t num_elems = buff_sz / sizeof(int32_t);
  if (translate_strings) {
    for (size_t i = 0; i < num_elems; ++i) {
      const auto string_id = buff[i];

      if (string_id == NULL_INT) {
        values.emplace_back(NullableString(nullptr));
      } else {
        if (dict_id == 0) {
          StringDictionaryProxy* sdp = row_set_mem_owner->getLiteralStringDictProxy();
          values.emplace_back(sdp->getString(string_id));
        } else {
          values.emplace_back(NullableString(
              executor->getStringDictionaryProxy(dict_id, row_set_mem_owner, false)
                  ->getString(string_id)));
        }
      }
    }
  } else {
    for (size_t i = 0; i < num_elems; i++) {
      values.emplace_back(static_cast<int64_t>(buff[i]));
    }
  }
  return values;
}

TargetValue build_array_target_value(const SQLTypeInfo& array_ti,
                                     const int8_t* buff,
                                     const size_t buff_sz,
                                     const bool translate_strings,
                                     std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                     const Executor* executor) {
  CHECK(array_ti.is_array());
  const auto& elem_ti = array_ti.get_elem_type();
  if (elem_ti.is_string()) {
    return build_string_array_target_value(reinterpret_cast<const int32_t*>(buff),
                                           buff_sz,
                                           elem_ti.get_comp_param(),
                                           translate_strings,
                                           row_set_mem_owner,
                                           executor);
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

template <class Tuple, size_t... indices>
inline std::vector<std::pair<const int8_t*, const int64_t>> make_vals_vector(
    std::index_sequence<indices...>,
    const Tuple& tuple) {
  return std::vector<std::pair<const int8_t*, const int64_t>>{
      std::make_pair(std::get<2 * indices>(tuple), std::get<2 * indices + 1>(tuple))...};
}

inline std::unique_ptr<ArrayDatum> lazy_fetch_chunk(const int8_t* ptr,
                                                    const int64_t varlen_ptr) {
  auto ad = std::make_unique<ArrayDatum>();
  bool is_end;
  ChunkIter_get_nth(reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(ptr)),
                    varlen_ptr,
                    ad.get(),
                    &is_end);
  CHECK(!is_end);
  return ad;
}

struct GeoLazyFetchHandler {
  template <typename... T>
  static inline auto fetch(T&&... vals) {
    constexpr int num_vals = sizeof...(vals);
    static_assert(
        num_vals % 2 == 0,
        "Must have consistent pointer/size pairs for lazy fetch of geo target values.");
    const auto vals_vector = make_vals_vector(std::make_index_sequence<num_vals / 2>{},
                                              std::make_tuple(vals...));
    std::array<VarlenDatumPtr, num_vals / 2> ad_arr;
    size_t ctr = 0;
    for (const auto& col_pair : vals_vector) {
      ad_arr[ctr++] = lazy_fetch_chunk(col_pair.first, col_pair.second);
    }
    return ad_arr;
  }
};

inline std::unique_ptr<ArrayDatum> fetch_data_from_gpu(int64_t varlen_ptr,
                                                       const int64_t length,
                                                       Data_Namespace::DataMgr* data_mgr,
                                                       const int device_id) {
  int8_t* cpu_buf = new int8_t[length];
  copy_from_gpu(
      data_mgr, cpu_buf, static_cast<CUdeviceptr>(varlen_ptr), length, device_id);
  return std::make_unique<ArrayDatum>(length, cpu_buf, false);
}

struct GeoQueryOutputFetchHandler {
  static inline auto yieldGpuDatumFetcher(Data_Namespace::DataMgr* data_mgr_ptr,
                                          const int device_id) {
    return [data_mgr_ptr, device_id](const int64_t ptr,
                                     const int64_t length) -> VarlenDatumPtr {
      return fetch_data_from_gpu(ptr, length, data_mgr_ptr, device_id);
    };
  }

  static inline auto yieldCpuDatumFetcher() {
    return [](const int64_t ptr, const int64_t length) -> VarlenDatumPtr {
      return std::make_unique<VarlenDatum>(length, reinterpret_cast<int8_t*>(ptr), false);
    };
  }

  template <typename... T>
  static inline auto fetch(Data_Namespace::DataMgr* data_mgr,
                           const bool fetch_data_from_gpu,
                           const int device_id,
                           T&&... vals) {
    auto ad_arr_generator = [&](auto datum_fetcher) {
      constexpr int num_vals = sizeof...(vals);
      static_assert(
          num_vals % 2 == 0,
          "Must have consistent pointer/size pairs for lazy fetch of geo target values.");
      const auto vals_vector = std::vector<int64_t>{vals...};

      std::array<VarlenDatumPtr, num_vals / 2> ad_arr;
      size_t ctr = 0;
      for (size_t i = 0; i < vals_vector.size(); i += 2) {
        ad_arr[ctr++] = datum_fetcher(vals_vector[i], vals_vector[i + 1]);
      }
      return ad_arr;
    };

    if (fetch_data_from_gpu) {
      auto datum_fetcher = yieldGpuDatumFetcher(data_mgr, device_id);
      return ad_arr_generator(datum_fetcher);
    } else {
      auto datum_fetcher = yieldCpuDatumFetcher();
      return ad_arr_generator(datum_fetcher);
    }
  }
};

template <SQLTypes GEO_SOURCE_TYPE, typename GeoTargetFetcher>
struct GeoTargetValueBuilder {
  template <typename... T>
  static inline TargetValue build(const SQLTypeInfo& geo_ti,
                                  const ResultSet::GeoReturnType return_type,
                                  T&&... vals) {
    auto ad_arr = GeoTargetFetcher::fetch(std::forward<T>(vals)...);
    static_assert(std::tuple_size<decltype(ad_arr)>::value > 0,
                  "ArrayDatum array for Geo Target must contain at least one value.");

    switch (return_type) {
      case ResultSet::GeoReturnType::GeoTargetValue: {
        if (ad_arr[0]->is_null) {
          return TargetValue(std::vector<ScalarTargetValue>{});
        }
        return GeoReturnTypeTraits<ResultSet::GeoReturnType::GeoTargetValue,
                                   GEO_SOURCE_TYPE>::GeoSerializerType::serialize(geo_ti,
                                                                                  ad_arr);
      }
      case ResultSet::GeoReturnType::WktString: {
        return GeoReturnTypeTraits<ResultSet::GeoReturnType::WktString,
                                   GEO_SOURCE_TYPE>::GeoSerializerType::serialize(geo_ti,
                                                                                  ad_arr);
      }
      default: {
        UNREACHABLE();
        return TargetValue(nullptr);
      }
    }
  }
};

}  // namespace

const std::vector<const int8_t*>& ResultSet::getColumnFrag(const size_t storage_idx,
                                                           const size_t col_logical_idx,
                                                           int64_t& global_idx) const {
  CHECK_LT(static_cast<size_t>(storage_idx), col_buffers_.size());
  if (col_buffers_[storage_idx].size() > 1) {
    int64_t frag_id = 0;
    int64_t local_idx = global_idx;
    if (consistent_frag_sizes_[storage_idx][col_logical_idx] != -1) {
      frag_id = global_idx / consistent_frag_sizes_[storage_idx][col_logical_idx];
      local_idx = global_idx % consistent_frag_sizes_[storage_idx][col_logical_idx];
    } else {
      std::tie(frag_id, local_idx) = get_frag_id_and_local_idx(
          frag_offsets_[storage_idx], col_logical_idx, global_idx);
      CHECK_LE(local_idx, global_idx);
    }
    CHECK_GE(frag_id, int64_t(0));
    CHECK_LT(frag_id, col_buffers_[storage_idx].size());
    global_idx = local_idx;
    return col_buffers_[storage_idx][frag_id];
  } else {
    CHECK_EQ(size_t(1), col_buffers_[storage_idx].size());
    return col_buffers_[storage_idx][0];
  }
}

/**
 * For each specified column, this function goes through all available storages and copy
 * its content into a contiguous output_buffer
 */
void ResultSet::copyColumnIntoBuffer(const size_t column_idx,
                                     int8_t* output_buffer,
                                     const size_t output_buffer_size) const {
  CHECK(isFastColumnarConversionPossible());
  CHECK_LT(column_idx, query_mem_desc_.getColCount());
  CHECK(output_buffer_size > 0);
  CHECK(output_buffer);
  const auto column_width_size = query_mem_desc_.getPaddedColumnWidthBytes(column_idx);
  size_t out_buff_offset = 0;

  // the main storage:
  const size_t crt_storage_row_count = storage_->query_mem_desc_.getEntryCount();
  const size_t crt_buffer_size = crt_storage_row_count * column_width_size;
  const size_t column_offset = storage_->query_mem_desc_.getColOffInBytes(column_idx);
  const int8_t* storage_buffer = storage_->getUnderlyingBuffer() + column_offset;
  CHECK(crt_buffer_size <= output_buffer_size);
  std::memcpy(output_buffer, storage_buffer, crt_buffer_size);

  out_buff_offset += crt_buffer_size;

  // the appended storages:
  for (size_t i = 0; i < appended_storage_.size(); i++) {
    CHECK_LT(out_buff_offset, output_buffer_size);
    const size_t crt_storage_row_count =
        appended_storage_[i]->query_mem_desc_.getEntryCount();
    const size_t crt_buffer_size = crt_storage_row_count * column_width_size;
    const size_t column_offset =
        appended_storage_[i]->query_mem_desc_.getColOffInBytes(column_idx);
    const int8_t* storage_buffer =
        appended_storage_[i]->getUnderlyingBuffer() + column_offset;
    CHECK(out_buff_offset + crt_buffer_size <= output_buffer_size);
    std::memcpy(output_buffer + out_buff_offset, storage_buffer, crt_buffer_size);

    out_buff_offset += crt_buffer_size;
  }
}

// Interprets ptr1, ptr2 as the ptr and len pair used for variable length data.
TargetValue ResultSet::makeVarlenTargetValue(const int8_t* ptr1,
                                             const int8_t compact_sz1,
                                             const int8_t* ptr2,
                                             const int8_t compact_sz2,
                                             const TargetInfo& target_info,
                                             const size_t target_logical_idx,
                                             const bool translate_strings,
                                             const size_t entry_buff_idx) const {
  auto varlen_ptr = read_int_from_buff(ptr1, compact_sz1);
  if (separate_varlen_storage_valid_ && !target_info.is_agg) {
    if (varlen_ptr < 0) {
      CHECK_EQ(-1, varlen_ptr);
      return TargetValue(nullptr);
    }
    const auto storage_idx = getStorageIndex(entry_buff_idx);
    if (target_info.sql_type.is_string()) {
      CHECK(target_info.sql_type.get_compression() == kENCODING_NONE);
      CHECK_LT(static_cast<size_t>(storage_idx.first), serialized_varlen_buffer_.size());
      const auto varlen_buffer_for_storage = serialized_varlen_buffer_[storage_idx.first];
      CHECK_LT(static_cast<size_t>(varlen_ptr), varlen_buffer_for_storage.size());
      return varlen_buffer_for_storage[varlen_ptr];
    } else if (target_info.sql_type.get_type() == kARRAY) {
      CHECK_LT(static_cast<size_t>(storage_idx.first), serialized_varlen_buffer_.size());
      const auto varlen_buffer = serialized_varlen_buffer_[storage_idx.first];
      CHECK_LT(static_cast<size_t>(varlen_ptr), varlen_buffer.size());

      return build_array_target_value(
          target_info.sql_type,
          reinterpret_cast<const int8_t*>(varlen_buffer[varlen_ptr].data()),
          varlen_buffer[varlen_ptr].size(),
          translate_strings,
          row_set_mem_owner_,
          executor_);
    } else {
      CHECK(false);
    }
  }
  if (!lazy_fetch_info_.empty()) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    const auto& col_lazy_fetch = lazy_fetch_info_[target_logical_idx];
    if (col_lazy_fetch.is_lazily_fetched) {
      const auto storage_idx = getStorageIndex(entry_buff_idx);
      CHECK_LT(static_cast<size_t>(storage_idx.first), col_buffers_.size());
      auto& frag_col_buffers =
          getColumnFrag(storage_idx.first, target_logical_idx, varlen_ptr);
      bool is_end{false};
      if (target_info.sql_type.is_string()) {
        VarlenDatum vd;
        ChunkIter_get_nth(reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(
                              frag_col_buffers[col_lazy_fetch.local_col_id])),
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
        ChunkIter_get_nth(reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(
                              frag_col_buffers[col_lazy_fetch.local_col_id])),
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
        return build_array_target_value(target_info.sql_type,
                                        ad.pointer,
                                        ad.length,
                                        translate_strings,
                                        row_set_mem_owner_,
                                        executor_);
      }
    }
  }
  if (!varlen_ptr) {
    return TargetValue(nullptr);
  }
  auto length = read_int_from_buff(ptr2, compact_sz2);
  if (target_info.sql_type.is_array()) {
    const auto& elem_ti = target_info.sql_type.get_elem_type();
    length *= elem_ti.get_array_context_logical_size();
  }
  std::vector<int8_t> cpu_buffer;
  if (varlen_ptr && device_type_ == ExecutorDeviceType::GPU) {
    cpu_buffer.resize(length);
    const auto executor = query_mem_desc_.getExecutor();
    CHECK(executor);
    auto& data_mgr = executor->catalog_->getDataMgr();
    copy_from_gpu(&data_mgr,
                  &cpu_buffer[0],
                  static_cast<CUdeviceptr>(varlen_ptr),
                  length,
                  device_id_);
    varlen_ptr = reinterpret_cast<int64_t>(&cpu_buffer[0]);
  }
  if (target_info.sql_type.is_array()) {
    return build_array_target_value(target_info.sql_type,
                                    reinterpret_cast<const int8_t*>(varlen_ptr),
                                    length,
                                    translate_strings,
                                    row_set_mem_owner_,
                                    executor_);
  }
  return std::string(reinterpret_cast<char*>(varlen_ptr), length);
}

// Reads a geo value from a series of ptrs to var len types
// In Columnar format, geo_target_ptr is the geo column ptr (a pointer to the beginning of
// that specific geo column) and should be appropriately adjusted with the entry_buff_idx
TargetValue ResultSet::makeGeoTargetValue(const int8_t* geo_target_ptr,
                                          const size_t slot_idx,
                                          const TargetInfo& target_info,
                                          const size_t target_logical_idx,
                                          const size_t entry_buff_idx) const {
  CHECK(target_info.sql_type.is_geometry());

  auto getNextTargetBufferRowWise = [&](const size_t slot_idx, const size_t range) {
    return geo_target_ptr + query_mem_desc_.getPaddedColWidthForRange(slot_idx, range);
  };

  auto getNextTargetBufferColWise = [&](const size_t slot_idx, const size_t range) {
    const auto storage_info = findStorage(entry_buff_idx);
    auto crt_geo_col_ptr = geo_target_ptr;
    for (size_t i = slot_idx; i < slot_idx + range; i++) {
      crt_geo_col_ptr = advance_to_next_columnar_target_buff(
          crt_geo_col_ptr, storage_info.storage_ptr->query_mem_desc_, i);
    }
    // adjusting the column pointer to represent a pointer to the geo target value
    return crt_geo_col_ptr +
           storage_info.fixedup_entry_idx *
               storage_info.storage_ptr->query_mem_desc_.getPaddedColumnWidthBytes(
                   slot_idx + range);
  };

  auto getNextTargetBuffer = [&](const size_t slot_idx, const size_t range) {
    return query_mem_desc_.didOutputColumnar()
               ? getNextTargetBufferColWise(slot_idx, range)
               : getNextTargetBufferRowWise(slot_idx, range);
  };

  auto getCoordsDataPtr = [&](const int8_t* geo_target_ptr) {
    return read_int_from_buff(getNextTargetBuffer(slot_idx, 0),
                              query_mem_desc_.getPaddedColumnWidthBytes(slot_idx));
  };

  auto getCoordsLength = [&](const int8_t* geo_target_ptr) {
    return read_int_from_buff(getNextTargetBuffer(slot_idx, 1),
                              query_mem_desc_.getPaddedColumnWidthBytes(slot_idx + 1));
  };

  auto getRingSizesPtr = [&](const int8_t* geo_target_ptr) {
    return read_int_from_buff(getNextTargetBuffer(slot_idx, 2),
                              query_mem_desc_.getPaddedColumnWidthBytes(slot_idx + 2));
  };

  auto getRingSizesLength = [&](const int8_t* geo_target_ptr) {
    return read_int_from_buff(getNextTargetBuffer(slot_idx, 3),
                              query_mem_desc_.getPaddedColumnWidthBytes(slot_idx + 3));
  };

  auto getPolyRingsPtr = [&](const int8_t* geo_target_ptr) {
    return read_int_from_buff(getNextTargetBuffer(slot_idx, 4),
                              query_mem_desc_.getPaddedColumnWidthBytes(slot_idx + 4));
  };

  auto getPolyRingsLength = [&](const int8_t* geo_target_ptr) {
    return read_int_from_buff(getNextTargetBuffer(slot_idx, 5),
                              query_mem_desc_.getPaddedColumnWidthBytes(slot_idx + 5));
  };

  auto getFragColBuffers = [&]() {
    const auto storage_idx = getStorageIndex(entry_buff_idx);
    CHECK_LT(static_cast<size_t>(storage_idx.first), col_buffers_.size());
    auto global_idx = getCoordsDataPtr(geo_target_ptr);
    return getColumnFrag(storage_idx.first, target_logical_idx, global_idx);
  };

  const bool is_gpu_fetch = device_type_ == ExecutorDeviceType::GPU;

  auto getDataMgr = [&]() {
    auto executor = query_mem_desc_.getExecutor();
    CHECK(executor);
    auto& data_mgr = executor->catalog_->getDataMgr();
    return &data_mgr;
  };

  auto getSeparateVarlenStorage = [&]() {
    const auto storage_idx = getStorageIndex(entry_buff_idx);
    CHECK_LT(static_cast<size_t>(storage_idx.first), serialized_varlen_buffer_.size());
    const auto& varlen_buffer = serialized_varlen_buffer_[storage_idx.first];
    return varlen_buffer;
  };

  auto removeCompressionFromGeoType = [](const SQLTypeInfo& ti) {
    return SQLTypeInfo(ti.get_type(),
                       ti.get_dimension(),
                       ti.get_scale(),
                       ti.get_notnull(),
                       kENCODING_NONE,
                       0,
                       ti.get_subtype());
  };

  if (separate_varlen_storage_valid_ && getCoordsDataPtr(geo_target_ptr) < 0) {
    CHECK_EQ(-1, getCoordsDataPtr(geo_target_ptr));
    return TargetValue(nullptr);
  }

  const ColumnLazyFetchInfo* col_lazy_fetch = nullptr;
  if (!lazy_fetch_info_.empty()) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    col_lazy_fetch = &lazy_fetch_info_[target_logical_idx];
  }

  switch (target_info.sql_type.get_type()) {
    case kPOINT: {
      if (separate_varlen_storage_valid_ && !target_info.is_agg) {
        auto varlen_buffer = getSeparateVarlenStorage();
        CHECK_LT(static_cast<size_t>(getCoordsDataPtr(geo_target_ptr)),
                 varlen_buffer.size());

        return GeoTargetValueBuilder<kPOINT, GeoQueryOutputFetchHandler>::build(
            removeCompressionFromGeoType(target_info.sql_type),
            geo_return_type_,
            nullptr,
            false,
            device_id_,
            reinterpret_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr)].data()),
            static_cast<int64_t>(varlen_buffer[getCoordsDataPtr(geo_target_ptr)].size()));
      } else if (col_lazy_fetch && col_lazy_fetch->is_lazily_fetched) {
        auto frag_col_buffers = getFragColBuffers();
        return GeoTargetValueBuilder<kPOINT, GeoLazyFetchHandler>::build(
            target_info.sql_type,
            geo_return_type_,
            frag_col_buffers[col_lazy_fetch->local_col_id],
            getCoordsDataPtr(geo_target_ptr));
      } else {
        return GeoTargetValueBuilder<kPOINT, GeoQueryOutputFetchHandler>::build(
            target_info.sql_type,
            geo_return_type_,
            is_gpu_fetch ? getDataMgr() : nullptr,
            is_gpu_fetch,
            device_id_,
            getCoordsDataPtr(geo_target_ptr),
            getCoordsLength(geo_target_ptr));
      }
    } break;
    case kLINESTRING: {
      if (separate_varlen_storage_valid_ && !target_info.is_agg) {
        auto varlen_buffer = getSeparateVarlenStorage();
        CHECK_LT(static_cast<size_t>(getCoordsDataPtr(geo_target_ptr)),
                 varlen_buffer.size());

        return GeoTargetValueBuilder<kLINESTRING, GeoQueryOutputFetchHandler>::build(
            removeCompressionFromGeoType(target_info.sql_type),
            geo_return_type_,
            nullptr,
            false,
            device_id_,
            reinterpret_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr)].data()),
            static_cast<int64_t>(varlen_buffer[getCoordsDataPtr(geo_target_ptr)].size()));
      } else if (col_lazy_fetch && col_lazy_fetch->is_lazily_fetched) {
        auto frag_col_buffers = getFragColBuffers();
        return GeoTargetValueBuilder<kLINESTRING, GeoLazyFetchHandler>::build(
            target_info.sql_type,
            geo_return_type_,
            frag_col_buffers[col_lazy_fetch->local_col_id],
            getCoordsDataPtr(geo_target_ptr));
      } else {
        return GeoTargetValueBuilder<kLINESTRING, GeoQueryOutputFetchHandler>::build(
            target_info.sql_type,
            geo_return_type_,
            is_gpu_fetch ? getDataMgr() : nullptr,
            is_gpu_fetch,
            device_id_,
            getCoordsDataPtr(geo_target_ptr),
            getCoordsLength(geo_target_ptr));
      }
    } break;
    case kPOLYGON: {
      if (separate_varlen_storage_valid_ && !target_info.is_agg) {
        auto varlen_buffer = getSeparateVarlenStorage();
        CHECK_LT(static_cast<size_t>(getCoordsDataPtr(geo_target_ptr) + 1),
                 varlen_buffer.size());

        return GeoTargetValueBuilder<kPOLYGON, GeoQueryOutputFetchHandler>::build(
            removeCompressionFromGeoType(target_info.sql_type),
            geo_return_type_,
            nullptr,
            false,
            device_id_,
            reinterpret_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr)].data()),
            static_cast<int64_t>(varlen_buffer[getCoordsDataPtr(geo_target_ptr)].size()),
            reinterpret_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr) + 1].data()),
            static_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr) + 1].size()));
      } else if (col_lazy_fetch && col_lazy_fetch->is_lazily_fetched) {
        auto frag_col_buffers = getFragColBuffers();

        return GeoTargetValueBuilder<kPOLYGON, GeoLazyFetchHandler>::build(
            target_info.sql_type,
            geo_return_type_,
            frag_col_buffers[col_lazy_fetch->local_col_id],
            getCoordsDataPtr(geo_target_ptr),
            frag_col_buffers[col_lazy_fetch->local_col_id + 1],
            getCoordsDataPtr(geo_target_ptr));
      } else {
        return GeoTargetValueBuilder<kPOLYGON, GeoQueryOutputFetchHandler>::build(
            target_info.sql_type,
            geo_return_type_,
            is_gpu_fetch ? getDataMgr() : nullptr,
            is_gpu_fetch,
            device_id_,
            getCoordsDataPtr(geo_target_ptr),
            getCoordsLength(geo_target_ptr),
            getRingSizesPtr(geo_target_ptr),
            getRingSizesLength(geo_target_ptr) * 4);
      }
    } break;
    case kMULTIPOLYGON: {
      if (separate_varlen_storage_valid_ && !target_info.is_agg) {
        auto varlen_buffer = getSeparateVarlenStorage();
        CHECK_LT(static_cast<size_t>(getCoordsDataPtr(geo_target_ptr) + 2),
                 varlen_buffer.size());

        return GeoTargetValueBuilder<kMULTIPOLYGON, GeoQueryOutputFetchHandler>::build(
            removeCompressionFromGeoType(target_info.sql_type),
            geo_return_type_,
            nullptr,
            false,
            device_id_,
            reinterpret_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr)].data()),
            static_cast<int64_t>(varlen_buffer[getCoordsDataPtr(geo_target_ptr)].size()),
            reinterpret_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr) + 1].data()),
            static_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr) + 1].size()),
            reinterpret_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr) + 2].data()),
            static_cast<int64_t>(
                varlen_buffer[getCoordsDataPtr(geo_target_ptr) + 2].size()));
      } else if (col_lazy_fetch && col_lazy_fetch->is_lazily_fetched) {
        auto frag_col_buffers = getFragColBuffers();

        return GeoTargetValueBuilder<kMULTIPOLYGON, GeoLazyFetchHandler>::build(
            target_info.sql_type,
            geo_return_type_,
            frag_col_buffers[col_lazy_fetch->local_col_id],
            getCoordsDataPtr(geo_target_ptr),
            frag_col_buffers[col_lazy_fetch->local_col_id + 1],
            getCoordsDataPtr(geo_target_ptr),
            frag_col_buffers[col_lazy_fetch->local_col_id + 2],
            getCoordsDataPtr(geo_target_ptr));
      } else {
        return GeoTargetValueBuilder<kMULTIPOLYGON, GeoQueryOutputFetchHandler>::build(
            target_info.sql_type,
            geo_return_type_,
            is_gpu_fetch ? getDataMgr() : nullptr,
            is_gpu_fetch,
            device_id_,
            getCoordsDataPtr(geo_target_ptr),
            getCoordsLength(geo_target_ptr),
            getRingSizesPtr(geo_target_ptr),
            getRingSizesLength(geo_target_ptr) * 4,
            getPolyRingsPtr(geo_target_ptr),
            getPolyRingsLength(geo_target_ptr) * 4);
      }
    } break;
    default:
      throw std::runtime_error("Unknown Geometry type encountered: " +
                               target_info.sql_type.get_type_name());
  }
  UNREACHABLE();
  return TargetValue(nullptr);
}

// Reads an integer or a float from ptr based on the type and the byte width.
TargetValue ResultSet::makeTargetValue(const int8_t* ptr,
                                       const int8_t compact_sz,
                                       const TargetInfo& target_info,
                                       const size_t target_logical_idx,
                                       const bool translate_strings,
                                       const bool decimal_to_double,
                                       const size_t entry_buff_idx) const {
  auto actual_compact_sz = compact_sz;
  if (target_info.sql_type.get_type() == kFLOAT &&
      !query_mem_desc_.forceFourByteFloat()) {
    // TODO(Saman): this condition should eventually just be didOutputColumnar(), remove
    // others once we can
    if (query_mem_desc_.didOutputColumnar() && !g_cluster &&
        query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
      actual_compact_sz = sizeof(float);
    } else {
      actual_compact_sz = sizeof(double);
    }
    if (target_info.is_agg &&
        (target_info.agg_kind == kAVG || target_info.agg_kind == kSUM ||
         target_info.agg_kind == kMIN || target_info.agg_kind == kMAX)) {
      // The above listed aggregates use two floats in a single 8-byte slot. Set the
      // padded size to 4 bytes to properly read each value.
      actual_compact_sz = sizeof(float);
    }
  }
  if (get_compact_type(target_info).is_date_in_days()) {
    // Dates encoded in days are converted to 8 byte values on read.
    actual_compact_sz = sizeof(int64_t);
  }

  // String dictionary keys are read as 32-bit values regardless of encoding
  if (target_info.sql_type.is_string() &&
      target_info.sql_type.get_compression() == kENCODING_DICT &&
      target_info.sql_type.get_comp_param()) {
    actual_compact_sz = sizeof(int32_t);
  }

  auto ival = read_int_from_buff(ptr, actual_compact_sz);
  const auto& chosen_type = get_compact_type(target_info);
  if (!lazy_fetch_info_.empty()) {
    CHECK_LT(target_logical_idx, lazy_fetch_info_.size());
    const auto& col_lazy_fetch = lazy_fetch_info_[target_logical_idx];
    if (col_lazy_fetch.is_lazily_fetched) {
      CHECK_GE(ival, 0);
      const auto storage_idx = getStorageIndex(entry_buff_idx);
      CHECK_LT(static_cast<size_t>(storage_idx.first), col_buffers_.size());
      auto& frag_col_buffers = getColumnFrag(storage_idx.first, target_logical_idx, ival);
      ival = lazy_decode(
          col_lazy_fetch, frag_col_buffers[col_lazy_fetch.local_col_id], ival);
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
        return chosen_type.get_type() == kFLOAT
                   ? ScalarTargetValue(static_cast<const float>(dval))
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
  if (chosen_type.is_integer() | chosen_type.is_boolean() || chosen_type.is_time() ||
      chosen_type.is_timeinterval()) {
    if (is_distinct_target(target_info)) {
      return TargetValue(count_distinct_set_size(
          ival, query_mem_desc_.getCountDistinctDescriptor(target_logical_idx)));
    }
    // TODO(alex): remove int_resize_cast, make read_int_from_buff return the
    // right type instead
    if (inline_int_null_val(chosen_type) ==
        int_resize_cast(ival, chosen_type.get_logical_size())) {
      return inline_int_null_val(target_info.sql_type);
    }
    return ival;
  }
  if (chosen_type.is_string() && chosen_type.get_compression() == kENCODING_DICT) {
    if (translate_strings) {
      if (static_cast<int32_t>(ival) ==
          NULL_INT) {  // TODO(alex): this isn't nice, fix it
        return NullableString(nullptr);
      }
      StringDictionaryProxy* sdp{nullptr};
      if (!chosen_type.get_comp_param()) {
        sdp = row_set_mem_owner_->getLiteralStringDictProxy();
      } else {
        sdp = executor_
                  ? executor_->getStringDictionaryProxy(
                        chosen_type.get_comp_param(), row_set_mem_owner_, false)
                  : row_set_mem_owner_->getStringDictProxy(chosen_type.get_comp_param());
      }
      return NullableString(sdp->getString(ival));
    } else {
      return static_cast<int64_t>(static_cast<int32_t>(ival));
    }
  }
  if (chosen_type.is_decimal()) {
    if (decimal_to_double) {
      if (ival ==
          inline_int_null_val(SQLTypeInfo(decimal_to_int_type(chosen_type), false))) {
        return NULL_DOUBLE;
      }
      return static_cast<double>(ival) / exp_to_scale(chosen_type.get_scale());
    }
    return ival;
  }
  CHECK(false);
  return TargetValue(int64_t(0));
}

// Gets the TargetValue stored at position local_entry_idx in the col1_ptr and col2_ptr
// column buffers. The second column is only used for AVG.
// the global_entry_idx is passed to makeTargetValue to be used for
// final lazy fetch (if there's any).
TargetValue ResultSet::getTargetValueFromBufferColwise(
    const int8_t* col_ptr,
    const int8_t* keys_ptr,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t local_entry_idx,
    const size_t global_entry_idx,
    const TargetInfo& target_info,
    const size_t target_logical_idx,
    const size_t slot_idx,
    const bool translate_strings,
    const bool decimal_to_double) const {
  CHECK(query_mem_desc_.didOutputColumnar());
  const auto col1_ptr = col_ptr;
  const auto compact_sz1 = query_mem_desc.getPaddedColumnWidthBytes(slot_idx);
  const auto next_col_ptr =
      advance_to_next_columnar_target_buff(col1_ptr, query_mem_desc, slot_idx);
  const auto col2_ptr = ((target_info.is_agg && target_info.agg_kind == kAVG) ||
                         is_real_str_or_array(target_info))
                            ? next_col_ptr
                            : nullptr;
  const auto compact_sz2 = ((target_info.is_agg && target_info.agg_kind == kAVG) ||
                            is_real_str_or_array(target_info))
                               ? query_mem_desc.getPaddedColumnWidthBytes(slot_idx + 1)
                               : 0;

  // TODO(Saman): add required logics for count distinct
  // geospatial target values:
  if (target_info.sql_type.is_geometry()) {
    return makeGeoTargetValue(
        col1_ptr, slot_idx, target_info, target_logical_idx, global_entry_idx);
  }

  const auto ptr1 = columnar_elem_ptr(local_entry_idx, col1_ptr, compact_sz1);
  if (target_info.agg_kind == kAVG || is_real_str_or_array(target_info)) {
    CHECK(col2_ptr);
    CHECK(compact_sz2);
    const auto ptr2 = columnar_elem_ptr(local_entry_idx, col2_ptr, compact_sz2);
    return target_info.agg_kind == kAVG
               ? make_avg_target_value(ptr1, compact_sz1, ptr2, compact_sz2, target_info)
               : makeVarlenTargetValue(ptr1,
                                       compact_sz1,
                                       ptr2,
                                       compact_sz2,
                                       target_info,
                                       target_logical_idx,
                                       translate_strings,
                                       global_entry_idx);
  }
  if (query_mem_desc_.targetGroupbyIndicesSize() == 0 ||
      query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) < 0) {
    return makeTargetValue(ptr1,
                           compact_sz1,
                           target_info,
                           target_logical_idx,
                           translate_strings,
                           decimal_to_double,
                           global_entry_idx);
  }
  const auto key_width = query_mem_desc_.getEffectiveKeyWidth();
  const auto key_idx = query_mem_desc_.getTargetGroupbyIndex(target_logical_idx);
  CHECK_GE(key_idx, 0);
  auto key_col_ptr = keys_ptr + key_idx * query_mem_desc_.getEntryCount() * key_width;
  return makeTargetValue(columnar_elem_ptr(local_entry_idx, key_col_ptr, key_width),
                         key_width,
                         target_info,
                         target_logical_idx,
                         translate_strings,
                         decimal_to_double,
                         global_entry_idx);
}

// Gets the TargetValue stored in slot_idx (and slot_idx for AVG) of
// rowwise_target_ptr.
TargetValue ResultSet::getTargetValueFromBufferRowwise(
    int8_t* rowwise_target_ptr,
    int8_t* keys_ptr,
    const size_t entry_buff_idx,
    const TargetInfo& target_info,
    const size_t target_logical_idx,
    const size_t slot_idx,
    const bool translate_strings,
    const bool decimal_to_double,
    const bool fixup_count_distinct_pointers) const {
  if (UNLIKELY(fixup_count_distinct_pointers)) {
    if (is_distinct_target(target_info)) {
      auto count_distinct_ptr_ptr = reinterpret_cast<int64_t*>(rowwise_target_ptr);
      const auto remote_ptr = *count_distinct_ptr_ptr;
      if (remote_ptr) {
        const auto ptr = storage_->mappedPtr(remote_ptr);
        if (ptr) {
          *count_distinct_ptr_ptr = ptr;
        } else {
          // need to create a zero filled buffer for this remote_ptr
          const auto& count_distinct_desc =
              query_mem_desc_.count_distinct_descriptors_[target_logical_idx];
          const auto bitmap_byte_sz = count_distinct_desc.sub_bitmap_count == 1
                                          ? count_distinct_desc.bitmapSizeBytes()
                                          : count_distinct_desc.bitmapPaddedSizeBytes();
          auto count_distinct_buffer =
              static_cast<int8_t*>(checked_malloc(bitmap_byte_sz));
          memset(count_distinct_buffer, 0, bitmap_byte_sz);
          row_set_mem_owner_->addCountDistinctBuffer(
              count_distinct_buffer, bitmap_byte_sz, true);
          *count_distinct_ptr_ptr = reinterpret_cast<int64_t>(count_distinct_buffer);
        }
      }
    }
    return int64_t(0);
  }
  if (target_info.sql_type.is_geometry()) {
    return makeGeoTargetValue(
        rowwise_target_ptr, slot_idx, target_info, target_logical_idx, entry_buff_idx);
  }

  auto ptr1 = rowwise_target_ptr;
  int8_t compact_sz1 = query_mem_desc_.getColumnWidth(slot_idx).compact;
  if (query_mem_desc_.isSingleColumnGroupByWithPerfectHash() &&
      !query_mem_desc_.hasKeylessHash() && !target_info.is_agg) {
    // Single column perfect hash group by can utilize one slot for both the key and the
    // target value if both values fit in 8 bytes. Use the target value actual size for
    // this case. If they don't, the target value should be 8 bytes, so we can still use
    // the actual size rather than the compact size.
    compact_sz1 = query_mem_desc_.getColumnWidth(slot_idx).actual;
  }

  // logic for deciding width of column
  if (target_info.agg_kind == kAVG || is_real_str_or_array(target_info)) {
    const auto ptr2 =
        rowwise_target_ptr + query_mem_desc_.getColumnWidth(slot_idx).compact;
    int8_t compact_sz2 = 0;
    // Skip reading the second slot if we have a none encoded string and are using
    // the none encoded strings buffer attached to ResultSetStorage
    if (!(separate_varlen_storage_valid_ &&
          (target_info.sql_type.is_array() ||
           (target_info.sql_type.is_string() &&
            target_info.sql_type.get_compression() == kENCODING_NONE)))) {
      compact_sz2 = query_mem_desc_.getColumnWidth(slot_idx + 1).compact;
    }
    if (separate_varlen_storage_valid_ && target_info.is_agg) {
      compact_sz2 = 8;  // TODO(adb): is there a better way to do this?
    }
    CHECK(ptr2);
    return target_info.agg_kind == kAVG
               ? make_avg_target_value(ptr1, compact_sz1, ptr2, compact_sz2, target_info)
               : makeVarlenTargetValue(ptr1,
                                       compact_sz1,
                                       ptr2,
                                       compact_sz2,
                                       target_info,
                                       target_logical_idx,
                                       translate_strings,
                                       entry_buff_idx);
  }
  if (query_mem_desc_.targetGroupbyIndicesSize() == 0 ||
      query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) < 0) {
    return makeTargetValue(ptr1,
                           compact_sz1,
                           target_info,
                           target_logical_idx,
                           translate_strings,
                           decimal_to_double,
                           entry_buff_idx);
  }
  const auto key_width = query_mem_desc_.getEffectiveKeyWidth();
  ptr1 = keys_ptr + query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) * key_width;
  return makeTargetValue(ptr1,
                         key_width,
                         target_info,
                         target_logical_idx,
                         translate_strings,
                         decimal_to_double,
                         entry_buff_idx);
}

// Returns true iff the entry at position entry_idx in buff contains a valid row.
bool ResultSetStorage::isEmptyEntry(const size_t entry_idx, const int8_t* buff) const {
  if (QueryDescriptionType::NonGroupedAggregate ==
      query_mem_desc_.getQueryDescriptionType()) {
    return false;
  }
  if (query_mem_desc_.didOutputColumnar()) {
    return isEmptyEntryColumnar(entry_idx, buff);
  }
  if (query_mem_desc_.hasKeylessHash()) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash);
    CHECK_GE(query_mem_desc_.getTargetIdxForKey(), 0);
    CHECK_LT(static_cast<size_t>(query_mem_desc_.getTargetIdxForKey()),
             target_init_vals_.size());
    const auto key_bytes_with_padding =
        align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
    const auto rowwise_target_ptr =
        row_ptr_rowwise(buff, query_mem_desc_, entry_idx) + key_bytes_with_padding;
    const auto target_slot_off =
        get_byteoff_of_slot(query_mem_desc_.getTargetIdxForKey(), query_mem_desc_);
    return read_int_from_buff(
               rowwise_target_ptr + target_slot_off,
               query_mem_desc_.getColumnWidth(query_mem_desc_.getTargetIdxForKey())
                   .compact) == target_init_vals_[query_mem_desc_.getTargetIdxForKey()];
  } else {
    const auto keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, entry_idx);
    switch (query_mem_desc_.getEffectiveKeyWidth()) {
      case 4:
        CHECK(QueryDescriptionType::GroupByBaselineHash ==
              query_mem_desc_.getQueryDescriptionType());
        return *reinterpret_cast<const int32_t*>(keys_ptr) == EMPTY_KEY_32;
      case 8:
        return *reinterpret_cast<const int64_t*>(keys_ptr) == EMPTY_KEY_64;
      default:
        CHECK(false);
        return true;
    }
  }
}

/*
 * Returns true if the entry contain empty keys
 * This function should only be used with columanr format.
 */
bool ResultSetStorage::isEmptyEntryColumnar(const size_t entry_idx,
                                            const int8_t* buff) const {
  CHECK(query_mem_desc_.didOutputColumnar());
  if (query_mem_desc_.getQueryDescriptionType() ==
      QueryDescriptionType::NonGroupedAggregate) {
    return false;
  }
  if (query_mem_desc_.hasKeylessHash()) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash);
    CHECK_GE(query_mem_desc_.getTargetIdxForKey(), 0);
    CHECK_LT(static_cast<size_t>(query_mem_desc_.getTargetIdxForKey()),
             target_init_vals_.size());
    const auto col_buff = advance_col_buff_to_slot(
        buff, query_mem_desc_, targets_, query_mem_desc_.getTargetIdxForKey(), false);
    const auto entry_buff =
        col_buff + entry_idx * query_mem_desc_.getPaddedColumnWidthBytes(
                                   query_mem_desc_.getTargetIdxForKey());
    return read_int_from_buff(entry_buff,
                              query_mem_desc_.getPaddedColumnWidthBytes(
                                  query_mem_desc_.getTargetIdxForKey())) ==
           target_init_vals_[query_mem_desc_.getTargetIdxForKey()];
  } else {
    // it's enough to find the first group key which is empty
    if (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
      return reinterpret_cast<const int64_t*>(buff)[entry_idx] == EMPTY_KEY_64;
    } else {
      CHECK(query_mem_desc_.groupColWidthsSize() > 0);
      const auto target_buff = buff + query_mem_desc_.getPrependedGroupColOffInBytes(0);
      switch (query_mem_desc_.groupColWidth(0)) {
        case 8:
          return reinterpret_cast<const int64_t*>(target_buff)[entry_idx] == EMPTY_KEY_64;
        case 4:
          return reinterpret_cast<const int32_t*>(target_buff)[entry_idx] == EMPTY_KEY_32;
        case 2:
          return reinterpret_cast<const int16_t*>(target_buff)[entry_idx] == EMPTY_KEY_16;
        case 1:
          return reinterpret_cast<const int8_t*>(target_buff)[entry_idx] == EMPTY_KEY_8;
        default:
          CHECK(false);
      }
    }
    return false;
  }
  return false;
}

bool ResultSetStorage::isEmptyEntry(const size_t entry_idx) const {
  return isEmptyEntry(entry_idx, buff_);
}

bool ResultSet::isNull(const SQLTypeInfo& ti,
                       const InternalTargetValue& val,
                       const bool float_argument_input) {
  if (ti.get_notnull()) {
    return false;
  }
  if (val.isInt()) {
    return val.i1 == null_val_bit_pattern(ti, float_argument_input);
  }
  if (val.isPair()) {
    return !val.i2 ||
           pair_to_double({val.i1, val.i2}, ti, float_argument_input) == NULL_DOUBLE;
  }
  if (val.isStr()) {
    return !val.i1;
  }
  CHECK(val.isNull());
  return true;
}
