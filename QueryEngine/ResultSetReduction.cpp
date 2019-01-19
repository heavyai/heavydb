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
 * @file    ResultSetReduction.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Reduction part of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "DynamicWatchdog.h"
#include "ResultRows.h"
#include "ResultSet.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"

#include "Shared/likely.h"
#include "Shared/thread_count.h"

#include <algorithm>
#include <future>
#include <numeric>

extern bool g_enable_dynamic_watchdog;

namespace {

bool use_multithreaded_reduction(const size_t entry_count) {
  return entry_count > 100000;
}

size_t get_row_qw_count(const QueryMemoryDescriptor& query_mem_desc) {
  const auto row_bytes = get_row_bytes(query_mem_desc);
  CHECK_EQ(size_t(0), row_bytes % 8);
  return row_bytes / 8;
}

std::vector<int64_t> make_key(const int64_t* buff,
                              const size_t entry_count,
                              const size_t key_count) {
  std::vector<int64_t> key;
  size_t off = 0;
  for (size_t i = 0; i < key_count; ++i) {
    key.push_back(buff[off]);
    off += entry_count;
  }
  return key;
}

void fill_slots(int64_t* dst_entry,
                const size_t dst_entry_count,
                const int64_t* src_buff,
                const size_t src_entry_idx,
                const size_t src_entry_count,
                const QueryMemoryDescriptor& query_mem_desc) {
  const auto slot_count = query_mem_desc.getBufferColSlotCount();
  const auto key_count = query_mem_desc.getGroupbyColCount();
  if (query_mem_desc.didOutputColumnar()) {
    for (size_t i = 0, dst_slot_off = 0; i < slot_count;
         ++i, dst_slot_off += dst_entry_count) {
      dst_entry[dst_slot_off] =
          src_buff[slot_offset_colwise(src_entry_idx, i, key_count, src_entry_count)];
    }
  } else {
    const auto row_ptr = src_buff + get_row_qw_count(query_mem_desc) * src_entry_idx;
    const auto slot_off_quad = get_slot_off_quad(query_mem_desc);
    for (size_t i = 0; i < slot_count; ++i) {
      dst_entry[i] = row_ptr[slot_off_quad + i];
    }
  }
}

ALWAYS_INLINE
void fill_empty_key_32(int32_t* key_ptr_i32, const size_t key_count) {
  for (size_t i = 0; i < key_count; ++i) {
    key_ptr_i32[i] = EMPTY_KEY_32;
  }
}

ALWAYS_INLINE
void fill_empty_key_64(int64_t* key_ptr_i64, const size_t key_count) {
  for (size_t i = 0; i < key_count; ++i) {
    key_ptr_i64[i] = EMPTY_KEY_64;
  }
}

}  // namespace

void fill_empty_key(void* key_ptr, const size_t key_count, const size_t key_width) {
  switch (key_width) {
    case 4: {
      auto key_ptr_i32 = reinterpret_cast<int32_t*>(key_ptr);
      fill_empty_key_32(key_ptr_i32, key_count);
    } break;
    case 8: {
      auto key_ptr_i64 = reinterpret_cast<int64_t*>(key_ptr);
      fill_empty_key_64(key_ptr_i64, key_count);
    } break;
    default:
      CHECK(false);
  }
}

// Driver method for various buffer layouts, actual work is done by reduceOne* methods.
// Reduces the entries of `that` into the buffer of this ResultSetStorage object.
void ResultSetStorage::reduce(
    const ResultSetStorage& that,
    const std::vector<std::string>& serialized_varlen_buffer) const {
  auto entry_count = query_mem_desc_.getEntryCount();
  CHECK_GT(entry_count, size_t(0));
  if (query_mem_desc_.didOutputColumnar()) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByPerfectHash ||
          query_mem_desc_.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByBaselineHash);
  }
  switch (query_mem_desc_.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByBaselineHash:
      CHECK_GE(entry_count, that.query_mem_desc_.getEntryCount());
      break;
    default:
      CHECK_EQ(entry_count, that.query_mem_desc_.getEntryCount());
  }
  auto this_buff = buff_;
  CHECK(this_buff);
  auto that_buff = that.buff_;
  CHECK(that_buff);
  if (query_mem_desc_.getQueryDescriptionType() ==
      QueryDescriptionType::GroupByBaselineHash) {
    if (!serialized_varlen_buffer.empty()) {
      throw std::runtime_error(
          "Projection of variable length targets with baseline hash group by is not yet "
          "supported in Distributed mode");
    }
    if (use_multithreaded_reduction(that.query_mem_desc_.getEntryCount())) {
      const size_t thread_count = cpu_threads();
      std::vector<std::future<void>> reduction_threads;
      for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        const auto thread_entry_count =
            (that.query_mem_desc_.getEntryCount() + thread_count - 1) / thread_count;
        const auto start_index = thread_idx * thread_entry_count;
        const auto end_index = std::min(start_index + thread_entry_count,
                                        that.query_mem_desc_.getEntryCount());
        reduction_threads.emplace_back(std::async(
            std::launch::async,
            [this, this_buff, that_buff, start_index, end_index, &that] {
              for (size_t entry_idx = start_index; entry_idx < end_index; ++entry_idx) {
                reduceOneEntryBaseline(this_buff,
                                       that_buff,
                                       entry_idx,
                                       that.query_mem_desc_.getEntryCount(),
                                       that);
              }
            }));
      }
      for (auto& reduction_thread : reduction_threads) {
        reduction_thread.wait();
      }
      for (auto& reduction_thread : reduction_threads) {
        reduction_thread.get();
      }
    } else {
      for (size_t i = 0; i < that.query_mem_desc_.getEntryCount(); ++i) {
        reduceOneEntryBaseline(
            this_buff, that_buff, i, that.query_mem_desc_.getEntryCount(), that);
      }
    }
    return;
  }
  if (use_multithreaded_reduction(entry_count)) {
    const size_t thread_count = cpu_threads();
    std::vector<std::future<void>> reduction_threads;
    for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      const auto thread_entry_count = (entry_count + thread_count - 1) / thread_count;
      const auto start_index = thread_idx * thread_entry_count;
      const auto end_index = std::min(start_index + thread_entry_count, entry_count);
      if (query_mem_desc_.didOutputColumnar()) {
        reduction_threads.emplace_back(std::async(std::launch::async,
                                                  [this,
                                                   this_buff,
                                                   that_buff,
                                                   start_index,
                                                   end_index,
                                                   &that,
                                                   &serialized_varlen_buffer] {
                                                    reduceEntriesNoCollisionsColWise(
                                                        this_buff,
                                                        that_buff,
                                                        that,
                                                        start_index,
                                                        end_index,
                                                        serialized_varlen_buffer);
                                                  }));
      } else {
        reduction_threads.emplace_back(std::async(
            std::launch::async,
            [this,
             this_buff,
             that_buff,
             start_index,
             end_index,
             &that,
             &serialized_varlen_buffer] {
              for (size_t entry_idx = start_index; entry_idx < end_index; ++entry_idx) {
                reduceOneEntryNoCollisionsRowWise(
                    entry_idx, this_buff, that_buff, that, serialized_varlen_buffer);
              }
            }));
      }
    }
    for (auto& reduction_thread : reduction_threads) {
      reduction_thread.wait();
    }
    for (auto& reduction_thread : reduction_threads) {
      reduction_thread.get();
    }
  } else {
    if (query_mem_desc_.didOutputColumnar()) {
      reduceEntriesNoCollisionsColWise(this_buff,
                                       that_buff,
                                       that,
                                       0,
                                       query_mem_desc_.getEntryCount(),
                                       serialized_varlen_buffer);
    } else {
      for (size_t i = 0; i < entry_count; ++i) {
        reduceOneEntryNoCollisionsRowWise(
            i, this_buff, that_buff, that, serialized_varlen_buffer);
      }
    }
  }
}

namespace {

ALWAYS_INLINE void check_watchdog(const size_t sample_seed) {
  if (UNLIKELY(g_enable_dynamic_watchdog && (sample_seed & 0x3F) == 0 &&
               dynamic_watchdog())) {
    // TODO(alex): distinguish between the deadline and interrupt
    throw std::runtime_error(
        "Query execution has exceeded the time limit or was interrupted during result "
        "set reduction");
  }
}

}  // namespace

void ResultSetStorage::reduceEntriesNoCollisionsColWise(
    int8_t* this_buff,
    const int8_t* that_buff,
    const ResultSetStorage& that,
    const size_t start_index,
    const size_t end_index,
    const std::vector<std::string>& serialized_varlen_buffer) const {
  // TODO(adb / saman): Support column wise output when serializing distributed agg
  // functions
  CHECK(serialized_varlen_buffer.empty());

  auto this_crt_col_ptr = get_cols_ptr(this_buff, query_mem_desc_);
  auto that_crt_col_ptr = get_cols_ptr(that_buff, query_mem_desc_);
  size_t agg_col_idx = 0;
  for (size_t target_idx = 0; target_idx < targets_.size(); ++target_idx) {
    const auto& agg_info = targets_[target_idx];
    const auto this_next_col_ptr = advance_to_next_columnar_target_buff(
        this_crt_col_ptr, query_mem_desc_, agg_col_idx);
    const auto that_next_col_ptr = advance_to_next_columnar_target_buff(
        that_crt_col_ptr, query_mem_desc_, agg_col_idx);
    for (size_t entry_idx = start_index; entry_idx < end_index; ++entry_idx) {
      check_watchdog(entry_idx);
      if (isEmptyEntryColumnar(entry_idx, that_buff)) {
        continue;
      }
      if (LIKELY(!query_mem_desc_.hasKeylessHash())) {
        // copy the key from right hand side
        copyKeyColWise(entry_idx, this_buff, that_buff);
      }
      auto this_ptr1 = this_crt_col_ptr +
                       entry_idx * query_mem_desc_.getPaddedColumnWidthBytes(agg_col_idx);
      auto that_ptr1 = that_crt_col_ptr +
                       entry_idx * query_mem_desc_.getPaddedColumnWidthBytes(agg_col_idx);
      int8_t* this_ptr2{nullptr};
      const int8_t* that_ptr2{nullptr};
      if (agg_info.is_agg &&
          (agg_info.agg_kind == kAVG ||
           (agg_info.agg_kind == kSAMPLE && agg_info.sql_type.is_varlen()))) {
        this_ptr2 =
            this_next_col_ptr +
            entry_idx * query_mem_desc_.getPaddedColumnWidthBytes(agg_col_idx + 1);
        that_ptr2 =
            that_next_col_ptr +
            entry_idx * query_mem_desc_.getPaddedColumnWidthBytes(agg_col_idx + 1);
      }
      reduceOneSlot(this_ptr1,
                    this_ptr2,
                    that_ptr1,
                    that_ptr2,
                    agg_info,
                    target_idx,
                    agg_col_idx,
                    agg_col_idx,
                    that,
                    serialized_varlen_buffer);
    }

    this_crt_col_ptr = this_next_col_ptr;
    that_crt_col_ptr = that_next_col_ptr;
    if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
      this_crt_col_ptr = advance_to_next_columnar_target_buff(
          this_crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
      that_crt_col_ptr = advance_to_next_columnar_target_buff(
          that_crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info, false);
  }
}

/*
 * copy all keys from the columnar prepended group buffer of "that_buff" into
 * "this_buff"
 */
void ResultSetStorage::copyKeyColWise(const size_t entry_idx,
                                      int8_t* this_buff,
                                      const int8_t* that_buff) const {
  CHECK(query_mem_desc_.didOutputColumnar());
  for (size_t group_idx = 0; group_idx < query_mem_desc_.getGroupbyColCount();
       group_idx++) {
    // if the column corresponds to a group key
    const auto column_offset_bytes =
        query_mem_desc_.getPrependedGroupColOffInBytes(group_idx);
    auto lhs_key_ptr = this_buff + column_offset_bytes;
    auto rhs_key_ptr = that_buff + column_offset_bytes;
    switch (query_mem_desc_.groupColWidth(group_idx)) {
      case 8:
        *(reinterpret_cast<int64_t*>(lhs_key_ptr) + entry_idx) =
            *(reinterpret_cast<const int64_t*>(rhs_key_ptr) + entry_idx);
        break;
      case 4:
        *(reinterpret_cast<int32_t*>(lhs_key_ptr) + entry_idx) =
            *(reinterpret_cast<const int32_t*>(rhs_key_ptr) + entry_idx);
        break;
      case 2:
        *(reinterpret_cast<int16_t*>(lhs_key_ptr) + entry_idx) =
            *(reinterpret_cast<const int16_t*>(rhs_key_ptr) + entry_idx);
        break;
      case 1:
        *(reinterpret_cast<int8_t*>(lhs_key_ptr) + entry_idx) =
            *(reinterpret_cast<const int8_t*>(rhs_key_ptr) + entry_idx);
        break;
      default:
        CHECK(false);
        break;
    }
  }
}

// Rewrites the entries of this ResultSetStorage object to point directly into the
// serialized_varlen_buffer rather than using offsets.
void ResultSetStorage::rewriteAggregateBufferOffsets(
    const std::vector<std::string>& serialized_varlen_buffer) const {
  if (serialized_varlen_buffer.empty()) {
    return;
  }

  CHECK(!query_mem_desc_.didOutputColumnar());
  auto entry_count = query_mem_desc_.getEntryCount();
  CHECK_GT(entry_count, size_t(0));
  CHECK(buff_);

  // Row-wise iteration, consider moving to separate function
  for (size_t i = 0; i < entry_count; ++i) {
    if (isEmptyEntry(i, buff_)) {
      continue;
    }
    const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
    const auto key_bytes_with_padding = align_to_int64(key_bytes);
    auto rowwise_targets_ptr =
        row_ptr_rowwise(buff_, query_mem_desc_, i) + key_bytes_with_padding;
    size_t target_slot_idx = 0;
    for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
         ++target_logical_idx) {
      const auto& target_info = targets_[target_logical_idx];
      if (target_info.sql_type.is_varlen() && target_info.is_agg) {
        CHECK(target_info.agg_kind == kSAMPLE);
        auto ptr1 = rowwise_targets_ptr;
        auto slot_idx = target_slot_idx;
        auto ptr2 = ptr1 + query_mem_desc_.getColumnWidth(slot_idx).compact;
        auto offset = *reinterpret_cast<const int64_t*>(ptr1);

        const auto& elem_ti = target_info.sql_type.get_elem_type();
        size_t length_to_elems =
            target_info.sql_type.is_string() || target_info.sql_type.is_geometry()
                ? 1
                : elem_ti.get_size();
        if (target_info.sql_type.is_geometry()) {
          for (int j = 0; j < target_info.sql_type.get_physical_coord_cols(); j++) {
            if (j > 0) {
              ptr1 = ptr2 + query_mem_desc_.getColumnWidth(slot_idx + 1).compact;
              ptr2 = ptr1 + query_mem_desc_.getColumnWidth(slot_idx + 2).compact;
              slot_idx += 2;
              length_to_elems = 4;
            }
            CHECK_LT(offset, serialized_varlen_buffer.size());
            const auto& varlen_bytes_str = serialized_varlen_buffer[offset++];
            const auto str_ptr =
                reinterpret_cast<const int8_t*>(varlen_bytes_str.c_str());
            CHECK(ptr1);
            *reinterpret_cast<int64_t*>(ptr1) = reinterpret_cast<const int64_t>(str_ptr);
            CHECK(ptr2);
            *reinterpret_cast<int64_t*>(ptr2) =
                static_cast<int64_t>(varlen_bytes_str.size() / length_to_elems);
          }
        } else {
          CHECK_LT(offset, serialized_varlen_buffer.size());
          const auto& varlen_bytes_str = serialized_varlen_buffer[offset];
          const auto str_ptr = reinterpret_cast<const int8_t*>(varlen_bytes_str.c_str());
          CHECK(ptr1);
          *reinterpret_cast<int64_t*>(ptr1) = reinterpret_cast<const int64_t>(str_ptr);
          CHECK(ptr2);
          *reinterpret_cast<int64_t*>(ptr2) =
              static_cast<int64_t>(varlen_bytes_str.size() / length_to_elems);
        }
      }

      rowwise_targets_ptr = advance_target_ptr_row_wise(
          rowwise_targets_ptr, target_info, target_slot_idx, query_mem_desc_, false);
      target_slot_idx = advance_slot(target_slot_idx, target_info, false);
    }
  }

  return;
}

// Reduces entry at position entry_idx in that_buff into the same position in this_buff,
// row-wise format.
void ResultSetStorage::reduceOneEntryNoCollisionsRowWise(
    const size_t entry_idx,
    int8_t* this_buff,
    const int8_t* that_buff,
    const ResultSetStorage& that,
    const std::vector<std::string>& serialized_varlen_buffer) const {
  check_watchdog(entry_idx);
  CHECK(!query_mem_desc_.didOutputColumnar());
  if (isEmptyEntry(entry_idx, that_buff)) {
    return;
  }
  const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
  const auto key_bytes_with_padding = align_to_int64(key_bytes);
  auto this_targets_ptr =
      row_ptr_rowwise(this_buff, query_mem_desc_, entry_idx) + key_bytes_with_padding;
  auto that_targets_ptr =
      row_ptr_rowwise(that_buff, query_mem_desc_, entry_idx) + key_bytes_with_padding;
  if (key_bytes) {  // copy the key from right hand side
    memcpy(this_targets_ptr - key_bytes_with_padding,
           that_targets_ptr - key_bytes_with_padding,
           key_bytes);
  }
  size_t target_slot_idx = 0;
  size_t init_agg_val_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
       ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    int8_t* this_ptr2{nullptr};
    const int8_t* that_ptr2{nullptr};
    if (target_info.is_agg &&
        (target_info.agg_kind == kAVG ||
         (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()))) {
      this_ptr2 =
          this_targets_ptr + query_mem_desc_.getColumnWidth(target_slot_idx).compact;
      that_ptr2 =
          that_targets_ptr + query_mem_desc_.getColumnWidth(target_slot_idx).compact;
    }
    reduceOneSlot(this_targets_ptr,
                  this_ptr2,
                  that_targets_ptr,
                  that_ptr2,
                  target_info,
                  target_logical_idx,
                  target_slot_idx,
                  init_agg_val_idx,
                  that,
                  serialized_varlen_buffer);
    this_targets_ptr = advance_target_ptr_row_wise(
        this_targets_ptr, target_info, target_slot_idx, query_mem_desc_, false);
    that_targets_ptr = advance_target_ptr_row_wise(
        that_targets_ptr, target_info, target_slot_idx, query_mem_desc_, false);
    target_slot_idx = advance_slot(target_slot_idx, target_info, false);
    if (query_mem_desc_.targetGroupbyIndicesSize() == 0) {
      init_agg_val_idx = advance_slot(init_agg_val_idx, target_info, false);
    } else {
      if (query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) < 0) {
        init_agg_val_idx = advance_slot(init_agg_val_idx, target_info, false);
      }
    }
  }
}

namespace {

using GroupValueInfo = std::pair<int64_t*, bool>;

GroupValueInfo get_matching_group_value_columnar_reduction(int64_t* groups_buffer,
                                                           const uint32_t h,
                                                           const int64_t* key,
                                                           const uint32_t key_qw_count,
                                                           const size_t entry_count) {
  auto off = h;
  const auto old_key =
      __sync_val_compare_and_swap(&groups_buffer[off], EMPTY_KEY_64, *key);
  if (old_key == EMPTY_KEY_64) {
    for (size_t i = 0; i < key_qw_count; ++i) {
      groups_buffer[off] = key[i];
      off += entry_count;
    }
    return {&groups_buffer[off], true};
  }
  off = h;
  for (size_t i = 0; i < key_qw_count; ++i) {
    if (groups_buffer[off] != key[i]) {
      return {nullptr, true};
    }
    off += entry_count;
  }
  return {&groups_buffer[off], false};
}

// TODO(alex): fix synchronization when we enable it
GroupValueInfo get_group_value_columnar_reduction(
    int64_t* groups_buffer,
    const uint32_t groups_buffer_entry_count,
    const int64_t* key,
    const uint32_t key_qw_count) {
  uint32_t h = key_hash(key, key_qw_count, sizeof(int64_t)) % groups_buffer_entry_count;
  auto matching_gvi = get_matching_group_value_columnar_reduction(
      groups_buffer, h, key, key_qw_count, groups_buffer_entry_count);
  if (matching_gvi.first) {
    return matching_gvi;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_gvi = get_matching_group_value_columnar_reduction(
        groups_buffer, h_probe, key, key_qw_count, groups_buffer_entry_count);
    if (matching_gvi.first) {
      return matching_gvi;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return {nullptr, true};
}

#define cas_cst(ptr, expected, desired) \
  __atomic_compare_exchange_n(          \
      ptr, expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define store_cst(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST)
#define load_cst(ptr) __atomic_load_n(ptr, __ATOMIC_SEQ_CST)

template <typename T = int64_t>
GroupValueInfo get_matching_group_value_reduction(
    int64_t* groups_buffer,
    const uint32_t h,
    const T* key,
    const uint32_t key_count,
    const QueryMemoryDescriptor& query_mem_desc,
    const int64_t* that_buff_i64,
    const size_t that_entry_idx,
    const size_t that_entry_count,
    const uint32_t row_size_quad) {
  auto off = h * row_size_quad;
  T empty_key = get_empty_key<T>();
  T write_pending = get_empty_key<T>() - 1;
  auto row_ptr = reinterpret_cast<T*>(groups_buffer + off);
  const auto slot_off_quad = get_slot_off_quad(query_mem_desc);
  const bool success = cas_cst(row_ptr, &empty_key, write_pending);
  if (success) {
    fill_slots(groups_buffer + off + slot_off_quad,
               query_mem_desc.getEntryCount(),
               that_buff_i64,
               that_entry_idx,
               that_entry_count,
               query_mem_desc);
    if (key_count > 1) {
      memcpy(row_ptr + 1, key + 1, (key_count - 1) * sizeof(T));
    }
    store_cst(row_ptr, *key);
    return {groups_buffer + off + slot_off_quad, true};
  }
  while (load_cst(row_ptr) == write_pending) {
    // spin until the winning thread has finished writing the entire key and the init
    // value
  }
  for (size_t i = 0; i < key_count; ++i) {
    if (load_cst(row_ptr + i) != key[i]) {
      return {nullptr, true};
    }
  }
  return {groups_buffer + off + slot_off_quad, false};
}

#undef load_cst
#undef store_cst
#undef cas_cst

inline GroupValueInfo get_matching_group_value_reduction(
    int64_t* groups_buffer,
    const uint32_t h,
    const int64_t* key,
    const uint32_t key_count,
    const size_t key_width,
    const QueryMemoryDescriptor& query_mem_desc,
    const int64_t* that_buff_i64,
    const size_t that_entry_idx,
    const size_t that_entry_count,
    const uint32_t row_size_quad) {
  switch (key_width) {
    case 4:
      return get_matching_group_value_reduction(groups_buffer,
                                                h,
                                                reinterpret_cast<const int32_t*>(key),
                                                key_count,
                                                query_mem_desc,
                                                that_buff_i64,
                                                that_entry_idx,
                                                that_entry_count,
                                                row_size_quad);
    case 8:
      return get_matching_group_value_reduction(groups_buffer,
                                                h,
                                                key,
                                                key_count,
                                                query_mem_desc,
                                                that_buff_i64,
                                                that_entry_idx,
                                                that_entry_count,
                                                row_size_quad);
    default:
      CHECK(false);
      return {nullptr, true};
  }
}

GroupValueInfo get_group_value_reduction(int64_t* groups_buffer,
                                         const uint32_t groups_buffer_entry_count,
                                         const int64_t* key,
                                         const uint32_t key_count,
                                         const size_t key_width,
                                         const QueryMemoryDescriptor& query_mem_desc,
                                         const int64_t* that_buff_i64,
                                         const size_t that_entry_idx,
                                         const size_t that_entry_count,
                                         const uint32_t row_size_quad) {
  uint32_t h = key_hash(key, key_count, key_width) % groups_buffer_entry_count;
  auto matching_gvi = get_matching_group_value_reduction(groups_buffer,
                                                         h,
                                                         key,
                                                         key_count,
                                                         key_width,
                                                         query_mem_desc,
                                                         that_buff_i64,
                                                         that_entry_idx,
                                                         that_entry_count,
                                                         row_size_quad);
  if (matching_gvi.first) {
    return matching_gvi;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_gvi = get_matching_group_value_reduction(groups_buffer,
                                                      h_probe,
                                                      key,
                                                      key_count,
                                                      key_width,
                                                      query_mem_desc,
                                                      that_buff_i64,
                                                      that_entry_idx,
                                                      that_entry_count,
                                                      row_size_quad);
    if (matching_gvi.first) {
      return matching_gvi;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return {nullptr, true};
}

}  // namespace

// Reduces entry at position that_entry_idx in that_buff into this_buff. This is
// the baseline layout, so the position in this_buff isn't known to be that_entry_idx.
void ResultSetStorage::reduceOneEntryBaseline(int8_t* this_buff,
                                              const int8_t* that_buff,
                                              const size_t that_entry_idx,
                                              const size_t that_entry_count,
                                              const ResultSetStorage& that) const {
  check_watchdog(that_entry_idx);
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByBaselineHash);
  CHECK(!query_mem_desc_.hasKeylessHash());
  const auto key_off =
      query_mem_desc_.didOutputColumnar()
          ? key_offset_colwise(that_entry_idx, 0, query_mem_desc_.didOutputColumnar())
          : get_row_qw_count(query_mem_desc_) * that_entry_idx;
  if (isEmptyEntry(that_entry_idx, that_buff)) {
    return;
  }
  int64_t* this_entry_slots{nullptr};
  auto this_buff_i64 = reinterpret_cast<int64_t*>(this_buff);
  auto that_buff_i64 = reinterpret_cast<const int64_t*>(that_buff);
  bool empty_entry = false;
  if (query_mem_desc_.didOutputColumnar()) {
    const auto key = make_key(&that_buff_i64[key_off], that_entry_count, key_count);
    std::tie(this_entry_slots, empty_entry) = get_group_value_columnar_reduction(
        this_buff_i64, query_mem_desc_.getEntryCount(), &key[0], key_count);
  } else {
    const uint32_t row_size_quad = get_row_qw_count(query_mem_desc_);
    std::tie(this_entry_slots, empty_entry) =
        get_group_value_reduction(this_buff_i64,
                                  query_mem_desc_.getEntryCount(),
                                  &that_buff_i64[key_off],
                                  key_count,
                                  query_mem_desc_.getEffectiveKeyWidth(),
                                  query_mem_desc_,
                                  that_buff_i64,
                                  that_entry_idx,
                                  that_entry_count,
                                  row_size_quad);
  }
  CHECK(this_entry_slots);
  if (empty_entry) {
    if (query_mem_desc_.didOutputColumnar()) {
      fill_slots(this_entry_slots,
                 query_mem_desc_.getEntryCount(),
                 that_buff_i64,
                 that_entry_idx,
                 that_entry_count,
                 query_mem_desc_);
    }
    return;
  }
  reduceOneEntrySlotsBaseline(
      this_entry_slots, that_buff_i64, that_entry_idx, that_entry_count, that);
}

void ResultSetStorage::reduceOneEntrySlotsBaseline(int64_t* this_entry_slots,
                                                   const int64_t* that_buff,
                                                   const size_t that_entry_idx,
                                                   const size_t that_entry_count,
                                                   const ResultSetStorage& that) const {
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  size_t j = 0;
  size_t init_agg_val_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
       ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    const auto that_slot_off =
        query_mem_desc_.didOutputColumnar()
            ? slot_offset_colwise(
                  that_entry_idx, init_agg_val_idx, key_count, that_entry_count)
            : get_row_qw_count(query_mem_desc_) * that_entry_idx +
                  get_slot_off_quad(query_mem_desc_) + init_agg_val_idx;
    const auto this_slot_off = query_mem_desc_.didOutputColumnar()
                                   ? init_agg_val_idx * query_mem_desc_.getEntryCount()
                                   : init_agg_val_idx;
    reduceOneSlotBaseline(this_entry_slots,
                          this_slot_off,
                          that_buff,
                          that_entry_count,
                          that_slot_off,
                          target_info,
                          target_logical_idx,
                          j,
                          init_agg_val_idx,
                          that);
    if (query_mem_desc_.targetGroupbyIndicesSize() == 0) {
      init_agg_val_idx = advance_slot(init_agg_val_idx, target_info, false);
    } else {
      if (query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) < 0) {
        init_agg_val_idx = advance_slot(init_agg_val_idx, target_info, false);
      }
    }
    j = advance_slot(j, target_info, false);
  }
}

void ResultSetStorage::reduceOneSlotBaseline(int64_t* this_buff,
                                             const size_t this_slot,
                                             const int64_t* that_buff,
                                             const size_t that_entry_count,
                                             const size_t that_slot,
                                             const TargetInfo& target_info,
                                             const size_t target_logical_idx,
                                             const size_t target_slot_idx,
                                             const size_t init_agg_val_idx,
                                             const ResultSetStorage& that) const {
  int8_t* this_ptr2{nullptr};
  const int8_t* that_ptr2{nullptr};
  if (target_info.is_agg &&
      (target_info.agg_kind == kAVG ||
       (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()))) {
    const auto this_count_off =
        query_mem_desc_.didOutputColumnar() ? query_mem_desc_.getEntryCount() : 1;
    const auto that_count_off =
        query_mem_desc_.didOutputColumnar() ? that_entry_count : 1;
    this_ptr2 = reinterpret_cast<int8_t*>(&this_buff[this_slot + this_count_off]);
    that_ptr2 = reinterpret_cast<const int8_t*>(&that_buff[that_slot + that_count_off]);
  }
  reduceOneSlot(reinterpret_cast<int8_t*>(&this_buff[this_slot]),
                this_ptr2,
                reinterpret_cast<const int8_t*>(&that_buff[that_slot]),
                that_ptr2,
                target_info,
                target_logical_idx,
                target_slot_idx,
                init_agg_val_idx,
                that,
                {});
}

// During the reduction of two result sets using the baseline strategy, we first create a
// big enough buffer to hold the entries for both and we move the entries from the first
// into it before doing the reduction as usual (into the first buffer).
template <class KeyType>
void ResultSetStorage::moveEntriesToBuffer(int8_t* new_buff,
                                           const size_t new_entry_count) const {
  CHECK(!query_mem_desc_.hasKeylessHash());
  CHECK_GT(new_entry_count, query_mem_desc_.getEntryCount());
  auto new_buff_i64 = reinterpret_cast<int64_t*>(new_buff);
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  CHECK(QueryDescriptionType::GroupByBaselineHash ==
        query_mem_desc_.getQueryDescriptionType());
  const auto this_buff = reinterpret_cast<const int64_t*>(buff_);
  const auto row_qw_count = get_row_qw_count(query_mem_desc_);
  const auto key_byte_width = query_mem_desc_.getEffectiveKeyWidth();
  for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
    const auto key_off = query_mem_desc_.didOutputColumnar()
                             ? key_offset_colwise(i, 0, query_mem_desc_.getEntryCount())
                             : row_qw_count * i;
    const auto key_ptr = reinterpret_cast<const KeyType*>(&this_buff[key_off]);
    if (*key_ptr == get_empty_key<KeyType>()) {
      continue;
    }
    int64_t* new_entries_ptr{nullptr};
    if (query_mem_desc_.didOutputColumnar()) {
      const auto key =
          make_key(&this_buff[key_off], query_mem_desc_.getEntryCount(), key_count);
      new_entries_ptr =
          get_group_value_columnar(new_buff_i64, new_entry_count, &key[0], key_count);
    } else {
      new_entries_ptr = get_group_value(new_buff_i64,
                                        new_entry_count,
                                        &this_buff[key_off],
                                        key_count,
                                        key_byte_width,
                                        row_qw_count,
                                        nullptr);
    }
    CHECK(new_entries_ptr);
    fill_slots(new_entries_ptr,
               new_entry_count,
               this_buff,
               i,
               query_mem_desc_.getEntryCount(),
               query_mem_desc_);
  }
}

void ResultSet::initializeStorage() const {
  if (query_mem_desc_.didOutputColumnar()) {
    storage_->initializeColWise();
  } else {
    storage_->initializeRowWise();
  }
}

// Driver for reductions. Needed because the result of a reduction on the baseline
// layout, which can have collisions, cannot be done in place and something needs
// to take the ownership of the new result set with the bigger underlying buffer.
ResultSet* ResultSetManager::reduce(std::vector<ResultSet*>& result_sets) {
  CHECK(!result_sets.empty());
  auto result_rs = result_sets.front();
  CHECK(result_rs->storage_);
  auto& first_result = *result_rs->storage_;
  auto result = &first_result;
  const auto row_set_mem_owner = result_rs->row_set_mem_owner_;
  for (const auto result_set : result_sets) {
    CHECK_EQ(row_set_mem_owner, result_set->row_set_mem_owner_);
  }
  const auto executor = result_rs->executor_;
  for (const auto result_set : result_sets) {
    CHECK_EQ(executor, result_set->executor_);
  }
  if (first_result.query_mem_desc_.getQueryDescriptionType() ==
      QueryDescriptionType::GroupByBaselineHash) {
    const auto total_entry_count =
        std::accumulate(result_sets.begin(),
                        result_sets.end(),
                        size_t(0),
                        [](const size_t init, const ResultSet* rs) {
                          return init + rs->query_mem_desc_.getEntryCount();
                        });
    CHECK(total_entry_count);
    auto query_mem_desc = first_result.query_mem_desc_;
    query_mem_desc.setEntryCount(total_entry_count);
    rs_.reset(new ResultSet(first_result.targets_,
                            ExecutorDeviceType::CPU,
                            query_mem_desc,
                            row_set_mem_owner,
                            executor));
    auto result_storage = rs_->allocateStorage(first_result.target_init_vals_);
    rs_->initializeStorage();
    switch (query_mem_desc.getEffectiveKeyWidth()) {
      case 4:
        first_result.moveEntriesToBuffer<int32_t>(result_storage->getUnderlyingBuffer(),
                                                  query_mem_desc.getEntryCount());
        break;
      case 8:
        first_result.moveEntriesToBuffer<int64_t>(result_storage->getUnderlyingBuffer(),
                                                  query_mem_desc.getEntryCount());
        break;
      default:
        CHECK(false);
    }
    result = rs_->storage_.get();
    result_rs = rs_.get();
  }

  auto& serialized_varlen_buffer = result_sets.front()->serialized_varlen_buffer_;
  if (!serialized_varlen_buffer.empty()) {
    result->rewriteAggregateBufferOffsets(serialized_varlen_buffer.front());
    for (auto result_it = result_sets.begin() + 1; result_it != result_sets.end();
         ++result_it) {
      auto& result_serialized_varlen_buffer = (*result_it)->serialized_varlen_buffer_;
      CHECK_EQ(result_serialized_varlen_buffer.size(), size_t(1));
      serialized_varlen_buffer.emplace_back(
          std::move(result_serialized_varlen_buffer.front()));
    }
  }

  size_t ctr = 1;
  for (auto result_it = result_sets.begin() + 1; result_it != result_sets.end();
       ++result_it) {
    if (!serialized_varlen_buffer.empty()) {
      result->reduce(*((*result_it)->storage_), serialized_varlen_buffer[ctr++]);
    } else {
      result->reduce(*((*result_it)->storage_), {});
    }
  }
  return result_rs;
}

std::shared_ptr<ResultSet> ResultSetManager::getOwnResultSet() {
  return rs_;
}

void ResultSetManager::rewriteVarlenAggregates(ResultSet* result_rs) {
  auto& result_storage = result_rs->storage_;
  result_storage->rewriteAggregateBufferOffsets(
      result_rs->serialized_varlen_buffer_.front());
}

void ResultSetStorage::fillOneEntryRowWise(const std::vector<int64_t>& entry) {
  const auto slot_count = query_mem_desc_.getBufferColSlotCount();
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  CHECK_EQ(slot_count + key_count, entry.size());
  auto this_buff = reinterpret_cast<int64_t*>(buff_);
  CHECK(!query_mem_desc_.didOutputColumnar());
  CHECK_EQ(size_t(1), query_mem_desc_.getEntryCount());
  const auto key_off = key_offset_rowwise(0, key_count, slot_count);
  CHECK_EQ(query_mem_desc_.getEffectiveKeyWidth(), sizeof(int64_t));
  for (size_t i = 0; i < key_count; ++i) {
    this_buff[key_off + i] = entry[i];
  }
  const auto first_slot_off = slot_offset_rowwise(0, 0, key_count, slot_count);
  for (size_t i = 0; i < target_init_vals_.size(); ++i) {
    this_buff[first_slot_off + i] = entry[key_count + i];
  }
}

void ResultSetStorage::initializeRowWise() const {
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  const auto row_size = get_row_bytes(query_mem_desc_);
  CHECK_EQ(row_size % 8, 0);
  const auto key_bytes_with_padding =
      align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
  CHECK(!query_mem_desc_.hasKeylessHash());
  switch (query_mem_desc_.getEffectiveKeyWidth()) {
    case 4: {
      for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
        auto row_ptr = buff_ + i * row_size;
        fill_empty_key_32(reinterpret_cast<int32_t*>(row_ptr), key_count);
        auto slot_ptr = reinterpret_cast<int64_t*>(row_ptr + key_bytes_with_padding);
        for (size_t j = 0; j < target_init_vals_.size(); ++j) {
          slot_ptr[j] = target_init_vals_[j];
        }
      }
      break;
    }
    case 8: {
      for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
        auto row_ptr = buff_ + i * row_size;
        fill_empty_key_64(reinterpret_cast<int64_t*>(row_ptr), key_count);
        auto slot_ptr = reinterpret_cast<int64_t*>(row_ptr + key_bytes_with_padding);
        for (size_t j = 0; j < target_init_vals_.size(); ++j) {
          slot_ptr[j] = target_init_vals_[j];
        }
      }
      break;
    }
    default:
      CHECK(false);
  }
}

void ResultSetStorage::fillOneEntryColWise(const std::vector<int64_t>& entry) {
  CHECK(query_mem_desc_.didOutputColumnar());
  CHECK_EQ(size_t(1), query_mem_desc_.getEntryCount());
  const auto slot_count = query_mem_desc_.getBufferColSlotCount();
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  CHECK_EQ(slot_count + key_count, entry.size());
  auto this_buff = reinterpret_cast<int64_t*>(buff_);

  for (size_t i = 0; i < key_count; i++) {
    const auto key_offset = key_offset_colwise(0, i, 1);
    this_buff[key_offset] = entry[i];
  }

  for (size_t i = 0; i < target_init_vals_.size(); i++) {
    const auto slot_offset = slot_offset_colwise(0, i, key_count, 1);
    this_buff[slot_offset] = entry[key_count + i];
  }
}

void ResultSetStorage::initializeColWise() const {
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  auto this_buff = reinterpret_cast<int64_t*>(buff_);
  CHECK(!query_mem_desc_.hasKeylessHash());
  for (size_t key_idx = 0; key_idx < key_count; ++key_idx) {
    const auto first_key_off =
        key_offset_colwise(0, key_idx, query_mem_desc_.getEntryCount());
    for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
      this_buff[first_key_off + i] = EMPTY_KEY_64;
    }
  }
  for (size_t target_idx = 0; target_idx < target_init_vals_.size(); ++target_idx) {
    const auto first_val_off =
        slot_offset_colwise(0, target_idx, key_count, query_mem_desc_.getEntryCount());
    for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
      this_buff[first_val_off + i] = target_init_vals_[target_idx];
    }
  }
}

void ResultSetStorage::initializeBaselineValueSlots(int64_t* entry_slots) const {
  CHECK(entry_slots);
  if (query_mem_desc_.didOutputColumnar()) {
    size_t slot_off = 0;
    for (size_t j = 0; j < target_init_vals_.size(); ++j) {
      entry_slots[slot_off] = target_init_vals_[j];
      slot_off += query_mem_desc_.getEntryCount();
    }
  } else {
    for (size_t j = 0; j < target_init_vals_.size(); ++j) {
      entry_slots[j] = target_init_vals_[j];
    }
  }
}

#define AGGREGATE_ONE_VALUE(                                                      \
    agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__)               \
  do {                                                                            \
    const auto sql_type = get_compact_type(agg_info__);                           \
    if (sql_type.is_fp()) {                                                       \
      if (chosen_bytes__ == sizeof(float)) {                                      \
        agg_##agg_kind__##_float(reinterpret_cast<int32_t*>(val_ptr__),           \
                                 *reinterpret_cast<const float*>(other_ptr__));   \
      } else {                                                                    \
        agg_##agg_kind__##_double(reinterpret_cast<int64_t*>(val_ptr__),          \
                                  *reinterpret_cast<const double*>(other_ptr__)); \
      }                                                                           \
    } else {                                                                      \
      if (chosen_bytes__ == sizeof(int32_t)) {                                    \
        auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                     \
        auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);           \
        agg_##agg_kind__##_int32(val_ptr, *other_ptr);                            \
      } else {                                                                    \
        auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                     \
        auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);           \
        agg_##agg_kind__(val_ptr, *other_ptr);                                    \
      }                                                                           \
    }                                                                             \
  } while (0)

#define AGGREGATE_ONE_NULLABLE_VALUE(                                               \
    agg_kind__, val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__)     \
  do {                                                                              \
    if (agg_info__.skip_null_val) {                                                 \
      const auto sql_type = get_compact_type(agg_info__);                           \
      if (sql_type.is_fp()) {                                                       \
        if (chosen_bytes__ == sizeof(float)) {                                      \
          agg_##agg_kind__##_float_skip_val(                                        \
              reinterpret_cast<int32_t*>(val_ptr__),                                \
              *reinterpret_cast<const float*>(other_ptr__),                         \
              *reinterpret_cast<const float*>(may_alias_ptr(&init_val__)));         \
        } else {                                                                    \
          agg_##agg_kind__##_double_skip_val(                                       \
              reinterpret_cast<int64_t*>(val_ptr__),                                \
              *reinterpret_cast<const double*>(other_ptr__),                        \
              *reinterpret_cast<const double*>(may_alias_ptr(&init_val__)));        \
        }                                                                           \
      } else {                                                                      \
        if (chosen_bytes__ == sizeof(int32_t)) {                                    \
          int32_t* val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                 \
          const int32_t* other_ptr = reinterpret_cast<const int32_t*>(other_ptr__); \
          const auto null_val = static_cast<int32_t>(init_val__);                   \
          agg_##agg_kind__##_int32_skip_val(val_ptr, *other_ptr, null_val);         \
        } else {                                                                    \
          int64_t* val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                 \
          const int64_t* other_ptr = reinterpret_cast<const int64_t*>(other_ptr__); \
          const auto null_val = static_cast<int64_t>(init_val__);                   \
          agg_##agg_kind__##_skip_val(val_ptr, *other_ptr, null_val);               \
        }                                                                           \
      }                                                                             \
    } else {                                                                        \
      AGGREGATE_ONE_VALUE(                                                          \
          agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__);          \
    }                                                                               \
  } while (0)

#define AGGREGATE_ONE_COUNT(val_ptr__, other_ptr__, chosen_bytes__)   \
  do {                                                                \
    if (chosen_bytes__ == sizeof(int32_t)) {                          \
      auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);           \
      auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__); \
      agg_sum_int32(val_ptr, *other_ptr);                             \
    } else {                                                          \
      auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);           \
      auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__); \
      agg_sum(val_ptr, *other_ptr);                                   \
    }                                                                 \
  } while (0)

#define AGGREGATE_ONE_NULLABLE_COUNT(                                        \
    val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__)          \
  {                                                                          \
    if (agg_info__.skip_null_val) {                                          \
      const auto sql_type = get_compact_type(agg_info__);                    \
      if (sql_type.is_fp()) {                                                \
        if (chosen_bytes__ == sizeof(float)) {                               \
          agg_sum_float_skip_val(                                            \
              reinterpret_cast<int32_t*>(val_ptr__),                         \
              *reinterpret_cast<const float*>(other_ptr__),                  \
              *reinterpret_cast<const float*>(may_alias_ptr(&init_val__)));  \
        } else {                                                             \
          agg_sum_double_skip_val(                                           \
              reinterpret_cast<int64_t*>(val_ptr__),                         \
              *reinterpret_cast<const double*>(other_ptr__),                 \
              *reinterpret_cast<const double*>(may_alias_ptr(&init_val__))); \
        }                                                                    \
      } else {                                                               \
        if (chosen_bytes__ == sizeof(int32_t)) {                             \
          auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);              \
          auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);    \
          const auto null_val = static_cast<int32_t>(init_val__);            \
          agg_sum_int32_skip_val(val_ptr, *other_ptr, null_val);             \
        } else {                                                             \
          auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);              \
          auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);    \
          const auto null_val = static_cast<int64_t>(init_val__);            \
          agg_sum_skip_val(val_ptr, *other_ptr, null_val);                   \
        }                                                                    \
      }                                                                      \
    } else {                                                                 \
      AGGREGATE_ONE_COUNT(val_ptr__, other_ptr__, chosen_bytes__);           \
    }                                                                        \
  }

void ResultSetStorage::reduceOneSlot(
    int8_t* this_ptr1,
    int8_t* this_ptr2,
    const int8_t* that_ptr1,
    const int8_t* that_ptr2,
    const TargetInfo& target_info,
    const size_t target_logical_idx,
    const size_t target_slot_idx,
    const size_t init_agg_val_idx,
    const ResultSetStorage& that,
    const std::vector<std::string>& serialized_varlen_buffer) const {
  if (query_mem_desc_.targetGroupbyIndicesSize() > 0) {
    if (query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) >= 0) {
      return;
    }
  }
  CHECK_LT(init_agg_val_idx, target_init_vals_.size());
  const bool float_argument_input = takes_float_argument(target_info);
  const auto chosen_bytes = float_argument_input
                                ? static_cast<int8_t>(sizeof(float))
                                : query_mem_desc_.getColumnWidth(target_slot_idx).compact;
  auto init_val = target_init_vals_[init_agg_val_idx];
  if (target_info.is_agg && target_info.agg_kind != kSAMPLE) {
    switch (target_info.agg_kind) {
      case kCOUNT:
      case kAPPROX_COUNT_DISTINCT: {
        if (is_distinct_target(target_info)) {
          CHECK_EQ(static_cast<size_t>(chosen_bytes), sizeof(int64_t));
          reduceOneCountDistinctSlot(this_ptr1, that_ptr1, target_logical_idx, that);
          break;
        }
        CHECK_EQ(int64_t(0), init_val);
        AGGREGATE_ONE_COUNT(this_ptr1, that_ptr1, chosen_bytes);
        break;
      }
      case kAVG: {
        // Ignore float argument compaction for count component for fear of its overflow
        AGGREGATE_ONE_COUNT(this_ptr2,
                            that_ptr2,
                            query_mem_desc_.getColumnWidth(target_slot_idx).compact);
      }
      // fall thru
      case kSUM: {
        AGGREGATE_ONE_NULLABLE_VALUE(
            sum, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        break;
      }
      case kMIN: {
        AGGREGATE_ONE_NULLABLE_VALUE(
            min, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        break;
      }
      case kMAX: {
        AGGREGATE_ONE_NULLABLE_VALUE(
            max, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        break;
      }
      default:
        CHECK(false);
    }
  } else {
    switch (chosen_bytes) {
      case 4: {
        CHECK(target_info.agg_kind != kSAMPLE);
        const auto rhs_proj_col = *reinterpret_cast<const int32_t*>(that_ptr1);
        if (rhs_proj_col != init_val) {
          *reinterpret_cast<int32_t*>(this_ptr1) = rhs_proj_col;
        }
        break;
      }
      case 8: {
        auto rhs_proj_col = *reinterpret_cast<const int64_t*>(that_ptr1);
        if ((target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()) &&
            !serialized_varlen_buffer.empty()) {
          auto ptr1 = this_ptr1;
          auto ptr2 = this_ptr2;
          auto slot_idx = target_slot_idx;

          const auto& elem_ti = target_info.sql_type.get_elem_type();
          size_t length_to_elems =
              target_info.sql_type.is_string() || target_info.sql_type.is_geometry()
                  ? 1
                  : elem_ti.get_size();
          if (target_info.sql_type.is_geometry()) {
            for (int j = 0; j < target_info.sql_type.get_physical_coord_cols(); j++) {
              if (j > 0) {
                ptr1 = ptr2 + query_mem_desc_.getColumnWidth(++slot_idx).compact;
                ptr2 = ptr1 + query_mem_desc_.getColumnWidth(++slot_idx).compact;
                length_to_elems = 4;
              }
              CHECK_LT(rhs_proj_col, serialized_varlen_buffer.size());
              const auto& varlen_bytes_str = serialized_varlen_buffer[rhs_proj_col++];
              const auto str_ptr =
                  reinterpret_cast<const int8_t*>(varlen_bytes_str.c_str());
              CHECK(ptr1);
              *reinterpret_cast<int64_t*>(ptr1) =
                  reinterpret_cast<const int64_t>(str_ptr);
              CHECK(ptr2);
              *reinterpret_cast<int64_t*>(ptr2) =
                  static_cast<int64_t>(varlen_bytes_str.size() / length_to_elems);
            }
          } else {
            CHECK_LT(rhs_proj_col, serialized_varlen_buffer.size());
            const auto& varlen_bytes_str = serialized_varlen_buffer[rhs_proj_col];
            const auto str_ptr =
                reinterpret_cast<const int8_t*>(varlen_bytes_str.c_str());
            CHECK(ptr1);
            *reinterpret_cast<int64_t*>(ptr1) = reinterpret_cast<const int64_t>(str_ptr);
            CHECK(ptr2);
            *reinterpret_cast<int64_t*>(ptr2) =
                static_cast<int64_t>(varlen_bytes_str.size() / length_to_elems);
          }
        } else {
          if (rhs_proj_col != init_val) {
            *reinterpret_cast<int64_t*>(this_ptr1) = rhs_proj_col;
          }
          if ((target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen())) {
            CHECK(this_ptr2 && that_ptr2);
            *reinterpret_cast<int64_t*>(this_ptr2) =
                *reinterpret_cast<const int64_t*>(that_ptr2);
          }
        }

        break;
      }
      case 0: {
        break;
      }
      default:
        CHECK(false);
    }
  }
}

void ResultSetStorage::reduceOneCountDistinctSlot(int8_t* this_ptr1,
                                                  const int8_t* that_ptr1,
                                                  const size_t target_logical_idx,
                                                  const ResultSetStorage& that) const {
  CHECK_LT(target_logical_idx, query_mem_desc_.getCountDistinctDescriptorsSize());
  const auto& old_count_distinct_desc =
      query_mem_desc_.getCountDistinctDescriptor(target_logical_idx);
  CHECK(old_count_distinct_desc.impl_type_ != CountDistinctImplType::Invalid);
  const auto& new_count_distinct_desc =
      that.query_mem_desc_.getCountDistinctDescriptor(target_logical_idx);
  CHECK(old_count_distinct_desc.impl_type_ == new_count_distinct_desc.impl_type_);
  CHECK(this_ptr1 && that_ptr1);
  auto old_set_ptr = reinterpret_cast<const int64_t*>(this_ptr1);
  auto new_set_ptr = reinterpret_cast<const int64_t*>(that_ptr1);
  count_distinct_set_union(
      *new_set_ptr, *old_set_ptr, new_count_distinct_desc, old_count_distinct_desc);
}

bool ResultRows::reduceSingleRow(const int8_t* row_ptr,
                                 const int8_t warp_count,
                                 const bool is_columnar,
                                 const bool replace_bitmap_ptr_with_bitmap_sz,
                                 std::vector<int64_t>& agg_vals,
                                 const QueryMemoryDescriptor& query_mem_desc,
                                 const std::vector<TargetInfo>& targets,
                                 const std::vector<int64_t>& agg_init_vals) {
  const size_t agg_col_count{agg_vals.size()};
  const auto row_size = query_mem_desc.getRowSize();
  CHECK_EQ(agg_col_count, query_mem_desc.getColCount());
  CHECK_GE(agg_col_count, targets.size());
  CHECK_EQ(is_columnar, query_mem_desc.didOutputColumnar());
  CHECK(query_mem_desc.hasKeylessHash());
  std::vector<int64_t> partial_agg_vals(agg_col_count, 0);
  bool discard_row = true;
  for (int8_t warp_idx = 0; warp_idx < warp_count; ++warp_idx) {
    bool discard_partial_result = true;
    for (size_t target_idx = 0, agg_col_idx = 0;
         target_idx < targets.size() && agg_col_idx < agg_col_count;
         ++target_idx, ++agg_col_idx) {
      const auto& agg_info = targets[target_idx];
      const bool float_argument_input = takes_float_argument(agg_info);
      const auto chosen_bytes = float_argument_input
                                    ? sizeof(float)
                                    : query_mem_desc.getColumnWidth(agg_col_idx).compact;
      auto partial_bin_val = get_component(
          row_ptr + query_mem_desc.getColOnlyOffInBytes(agg_col_idx), chosen_bytes);
      partial_agg_vals[agg_col_idx] = partial_bin_val;
      if (is_distinct_target(agg_info)) {
        CHECK_EQ(int8_t(1), warp_count);
        CHECK(agg_info.is_agg && (agg_info.agg_kind == kCOUNT ||
                                  agg_info.agg_kind == kAPPROX_COUNT_DISTINCT));
        partial_bin_val = count_distinct_set_size(
            partial_bin_val, query_mem_desc.getCountDistinctDescriptor(target_idx));
        if (replace_bitmap_ptr_with_bitmap_sz) {
          partial_agg_vals[agg_col_idx] = partial_bin_val;
        }
      }
      if (kAVG == agg_info.agg_kind) {
        CHECK(agg_info.is_agg && !agg_info.is_distinct);
        ++agg_col_idx;
        partial_bin_val = partial_agg_vals[agg_col_idx] =
            get_component(row_ptr + query_mem_desc.getColOnlyOffInBytes(agg_col_idx),
                          query_mem_desc.getColumnWidth(agg_col_idx).compact);
      }
      if (agg_col_idx == static_cast<size_t>(query_mem_desc.getTargetIdxForKey()) &&
          partial_bin_val != agg_init_vals[query_mem_desc.getTargetIdxForKey()]) {
        CHECK(agg_info.is_agg);
        discard_partial_result = false;
      }
    }
    row_ptr += row_size;
    if (discard_partial_result) {
      continue;
    }
    discard_row = false;
    for (size_t target_idx = 0, agg_col_idx = 0;
         target_idx < targets.size() && agg_col_idx < agg_col_count;
         ++target_idx, ++agg_col_idx) {
      auto partial_bin_val = partial_agg_vals[agg_col_idx];
      const auto& agg_info = targets[target_idx];
      const bool float_argument_input = takes_float_argument(agg_info);
      const auto chosen_bytes = float_argument_input
                                    ? sizeof(float)
                                    : query_mem_desc.getColumnWidth(agg_col_idx).compact;
      const auto& chosen_type = get_compact_type(agg_info);
      if (agg_info.is_agg && agg_info.agg_kind != kSAMPLE) {
        try {
          switch (agg_info.agg_kind) {
            case kCOUNT:
            case kAPPROX_COUNT_DISTINCT:
              AGGREGATE_ONE_NULLABLE_COUNT(
                  reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                  agg_init_vals[agg_col_idx],
                  chosen_bytes,
                  agg_info);
              break;
            case kAVG:
              // Ignore float argument compaction for count component for fear of its
              // overflow
              AGGREGATE_ONE_COUNT(
                  reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx + 1]),
                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx + 1]),
                  query_mem_desc.getColumnWidth(agg_col_idx).compact);
            // fall thru
            case kSUM:
              AGGREGATE_ONE_NULLABLE_VALUE(
                  sum,
                  reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                  agg_init_vals[agg_col_idx],
                  chosen_bytes,
                  agg_info);
              break;
            case kMIN:
              AGGREGATE_ONE_NULLABLE_VALUE(
                  min,
                  reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                  agg_init_vals[agg_col_idx],
                  chosen_bytes,
                  agg_info);
              break;
            case kMAX:
              AGGREGATE_ONE_NULLABLE_VALUE(
                  max,
                  reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                  agg_init_vals[agg_col_idx],
                  chosen_bytes,
                  agg_info);
              break;
            default:
              CHECK(false);
              break;
          }
        } catch (std::runtime_error& e) {
          // TODO(miyu): handle the case where chosen_bytes < 8
          LOG(ERROR) << e.what();
        }
        if (chosen_type.is_integer() || chosen_type.is_decimal()) {
          switch (chosen_bytes) {
            case 8:
              break;
            case 4: {
              int32_t ret = *reinterpret_cast<const int32_t*>(&agg_vals[agg_col_idx]);
              if (!(agg_info.agg_kind == kCOUNT && ret != agg_init_vals[agg_col_idx])) {
                agg_vals[agg_col_idx] = static_cast<int64_t>(ret);
              }
              break;
            }
            default:
              CHECK(false);
          }
        }
        if (kAVG == agg_info.agg_kind) {
          ++agg_col_idx;
        }
      } else {
        if (agg_info.agg_kind == kSAMPLE) {
          CHECK(!agg_info.sql_type.is_varlen())
              << "Interleaved bins reduction not supported for variable length arguments "
                 "to SAMPLE";
        }
        if (agg_vals[agg_col_idx]) {
          if (agg_info.agg_kind == kSAMPLE) {
            continue;
          }
          CHECK_EQ(agg_vals[agg_col_idx], partial_bin_val);
        } else {
          agg_vals[agg_col_idx] = partial_bin_val;
        }
      }
    }
  }
  return discard_row;
}
