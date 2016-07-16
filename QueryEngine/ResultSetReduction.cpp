/**
 * @file    ResultSetReduction.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Reduction part of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultSet.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"

#include "Shared/thread_count.h"

#include <future>
#include <numeric>

namespace {

bool use_multithreaded_reduction(const size_t entry_count) {
  return entry_count > 2;
}

size_t get_row_qw_count(const QueryMemoryDescriptor& query_mem_desc) {
  const auto row_bytes = get_row_bytes(query_mem_desc);
  CHECK_EQ(size_t(0), row_bytes % 8);
  return row_bytes / 8;
}

std::vector<int64_t> make_key(const int64_t* buff, const size_t entry_count, const size_t key_count) {
  std::vector<int64_t> key;
  size_t off = 0;
  for (size_t i = 0; i < key_count; ++i) {
    key.push_back(buff[off]);
    off += entry_count;
  }
  return key;
}

}  // namespace

// Driver method for various buffer layouts, actual work is done by reduceOne* methods.
// Reduces the entries of `that` into the buffer of this ResultSetStorage object.
void ResultSetStorage::reduce(const ResultSetStorage& that) const {
  const auto entry_count = query_mem_desc_.entry_count;
  CHECK_GT(entry_count, size_t(0));
  CHECK_EQ(size_t(0), query_mem_desc_.entry_count_small);
  if (query_mem_desc_.hash_type == GroupByColRangeType::MultiCol) {
    CHECK_GE(entry_count, that.query_mem_desc_.entry_count);
  } else {
    CHECK_EQ(entry_count, that.query_mem_desc_.entry_count);
  }
  const auto this_buff = buff_;
  CHECK(this_buff);
  const auto that_buff = that.buff_;
  CHECK(that_buff);
  if (query_mem_desc_.hash_type == GroupByColRangeType::MultiCol) {
    for (size_t i = 0; i < that.query_mem_desc_.entry_count; ++i) {
      reduceOneEntryBaseline(this_buff, that_buff, i, that.query_mem_desc_.entry_count);
    }
    return;
  }
  if (use_multithreaded_reduction(entry_count)) {
    const size_t thread_count = cpu_threads();
    std::vector<std::future<void>> reduction_threads;
    for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      reduction_threads.emplace_back(std::async([this, thread_idx, entry_count, this_buff, that_buff, thread_count] {
        for (size_t i = thread_idx; i < entry_count; i += thread_count) {
          if (query_mem_desc_.output_columnar) {
            reduceOneEntryNoCollisionsColWise(i, this_buff, that_buff);
          } else {
            reduceOneEntryNoCollisionsRowWise(i, this_buff, that_buff);
          }
        }
      }));
    }
    for (auto& reduction_thread : reduction_threads) {
      reduction_thread.get();
    }
  } else {
    for (size_t i = 0; i < entry_count; ++i) {
      if (query_mem_desc_.output_columnar) {
        reduceOneEntryNoCollisionsColWise(i, this_buff, that_buff);
      } else {
        reduceOneEntryNoCollisionsRowWise(i, this_buff, that_buff);
      }
    }
  }
}

// Reduces entry at position entry_idx in that_buff into the same position in this_buff, columnar format.
void ResultSetStorage::reduceOneEntryNoCollisionsColWise(const size_t entry_idx,
                                                         int8_t* this_buff,
                                                         const int8_t* that_buff) const {
  CHECK(query_mem_desc_.output_columnar);
  CHECK(query_mem_desc_.hash_type == GroupByColRangeType::OneColKnownRange ||
        query_mem_desc_.hash_type == GroupByColRangeType::MultiColPerfectHash ||
        query_mem_desc_.hash_type == GroupByColRangeType::MultiCol);
  auto this_crt_col_ptr = get_cols_ptr(this_buff, query_mem_desc_);
  auto that_crt_col_ptr = get_cols_ptr(that_buff, query_mem_desc_);
  size_t agg_col_idx = 0;
  const auto buffer_col_count = get_buffer_col_slot_count(query_mem_desc_);
  for (size_t target_idx = 0; target_idx < targets_.size(); ++target_idx) {
    CHECK_LT(agg_col_idx, buffer_col_count);
    const auto& agg_info = targets_[target_idx];
    const auto this_next_col_ptr = advance_to_next_columnar_target_buff(this_crt_col_ptr, query_mem_desc_, agg_col_idx);
    const auto that_next_col_ptr = advance_to_next_columnar_target_buff(that_crt_col_ptr, query_mem_desc_, agg_col_idx);
    auto this_ptr1 = this_crt_col_ptr + entry_idx * query_mem_desc_.agg_col_widths[agg_col_idx].compact;
    auto that_ptr1 = that_crt_col_ptr + entry_idx * query_mem_desc_.agg_col_widths[agg_col_idx].compact;
    int8_t* this_ptr2{nullptr};
    const int8_t* that_ptr2{nullptr};
    if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
      this_ptr2 = this_next_col_ptr + entry_idx * query_mem_desc_.agg_col_widths[agg_col_idx + 1].compact;
      that_ptr2 = that_next_col_ptr + entry_idx * query_mem_desc_.agg_col_widths[agg_col_idx + 1].compact;
    }
    reduceOneSlot(this_ptr1, this_ptr2, that_ptr1, that_ptr2, agg_info, agg_col_idx);
    this_crt_col_ptr = this_next_col_ptr;
    that_crt_col_ptr = that_next_col_ptr;
    if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
      this_crt_col_ptr = advance_to_next_columnar_target_buff(this_crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
      that_crt_col_ptr = advance_to_next_columnar_target_buff(that_crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info);
  }
}

// Reduces entry at position entry_idx in that_buff into the same position in this_buff, row-wise format.
void ResultSetStorage::reduceOneEntryNoCollisionsRowWise(const size_t entry_idx,
                                                         int8_t* this_buff,
                                                         const int8_t* that_buff) const {
  CHECK(!query_mem_desc_.output_columnar);
  CHECK(query_mem_desc_.hash_type == GroupByColRangeType::OneColKnownRange ||
        query_mem_desc_.hash_type == GroupByColRangeType::MultiColPerfectHash ||
        query_mem_desc_.hash_type == GroupByColRangeType::MultiCol);
  const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
  auto this_targets_ptr = row_ptr_rowwise(this_buff, query_mem_desc_, entry_idx) + key_bytes;
  auto that_targets_ptr = row_ptr_rowwise(that_buff, query_mem_desc_, entry_idx) + key_bytes;
  size_t target_slot_idx = 0;
  for (const auto& target_info : targets_) {
    int8_t* this_ptr2{nullptr};
    const int8_t* that_ptr2{nullptr};
    if (target_info.is_agg && target_info.agg_kind == kAVG) {
      this_ptr2 = this_targets_ptr + query_mem_desc_.agg_col_widths[target_slot_idx].compact;
      that_ptr2 = that_targets_ptr + query_mem_desc_.agg_col_widths[target_slot_idx].compact;
    }
    reduceOneSlot(this_targets_ptr, this_ptr2, that_targets_ptr, that_ptr2, target_info, target_slot_idx);
    this_targets_ptr = advance_target_ptr(this_targets_ptr, target_info, target_slot_idx, query_mem_desc_);
    that_targets_ptr = advance_target_ptr(that_targets_ptr, target_info, target_slot_idx, query_mem_desc_);
    target_slot_idx = advance_slot(target_slot_idx, target_info);
  }
}

// Reduces entry at position that_entry_idx in that_buff into this_buff. This is
// the baseline layout, so the position in this_buff isn't known to be that_entry_idx.
void ResultSetStorage::reduceOneEntryBaseline(int8_t* this_buff,
                                              const int8_t* that_buff,
                                              const size_t that_entry_idx,
                                              const size_t that_entry_count) const {
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc_);
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  CHECK(GroupByColRangeType::MultiCol == query_mem_desc_.hash_type);
  CHECK(!query_mem_desc_.keyless_hash);
  const auto key_off = query_mem_desc_.output_columnar
                           ? key_offset_colwise(that_entry_idx, 0, query_mem_desc_.output_columnar)
                           : key_offset_rowwise(that_entry_idx, key_count, slot_count);
  if (isEmptyEntry(that_entry_idx, that_buff)) {
    return;
  }
  int64_t* this_entry_slots{nullptr};
  auto this_buff_i64 = reinterpret_cast<int64_t*>(this_buff);
  auto that_buff_i64 = reinterpret_cast<const int64_t*>(that_buff);
  if (query_mem_desc_.output_columnar) {
    const auto key = make_key(&that_buff_i64[key_off], that_entry_count, key_count);
    this_entry_slots = get_group_value_columnar(this_buff_i64, query_mem_desc_.entry_count, &key[0], key_count);
  } else {
    this_entry_slots = get_group_value(this_buff_i64,
                                       query_mem_desc_.entry_count,
                                       &that_buff_i64[key_off],
                                       key_count,
                                       get_row_qw_count(query_mem_desc_),
                                       nullptr);
  }
  initializeBaselineValueSlots(this_entry_slots);
  reduceOneEntrySlotsBaseline(this_entry_slots, that_buff_i64, that_entry_idx, that_entry_count);
}

void ResultSetStorage::reduceOneEntrySlotsBaseline(int64_t* this_entry_slots,
                                                   const int64_t* that_buff,
                                                   const size_t that_entry_idx,
                                                   const size_t that_entry_count) const {
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc_);
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  size_t j = 0;
  for (const auto& target_info : targets_) {
    const auto that_slot_off = query_mem_desc_.output_columnar
                                   ? slot_offset_colwise(that_entry_idx, j, key_count, that_entry_count)
                                   : slot_offset_rowwise(that_entry_idx, j, key_count, slot_count);
    const auto this_slot_off = query_mem_desc_.output_columnar ? j * query_mem_desc_.entry_count : j;
    reduceOneSlotBaseline(this_entry_slots, this_slot_off, that_buff, that_entry_count, that_slot_off, target_info, j);
    j = advance_slot(j, target_info);
  }
}

void ResultSetStorage::reduceOneSlotBaseline(int64_t* this_buff,
                                             const size_t this_slot,
                                             const int64_t* that_buff,
                                             const size_t that_entry_count,
                                             const size_t that_slot,
                                             const TargetInfo& target_info,
                                             const size_t target_slot_idx) const {
  int8_t* this_ptr2{nullptr};
  const int8_t* that_ptr2{nullptr};
  if (target_info.is_agg && target_info.agg_kind == kAVG) {
    const auto this_count_off = query_mem_desc_.output_columnar ? query_mem_desc_.entry_count : 1;
    const auto that_count_off = query_mem_desc_.output_columnar ? that_entry_count : 1;
    this_ptr2 = reinterpret_cast<int8_t*>(&this_buff[this_slot + this_count_off]);
    that_ptr2 = reinterpret_cast<const int8_t*>(&that_buff[that_slot + that_count_off]);
  }
  reduceOneSlot(reinterpret_cast<int8_t*>(&this_buff[this_slot]),
                this_ptr2,
                reinterpret_cast<const int8_t*>(&that_buff[that_slot]),
                that_ptr2,
                target_info,
                target_slot_idx);
}

// During the reduction of two result sets using the baseline strategy, we first create a big
// enough buffer to hold the entries for both and we move the entries from the first into it
// before doing the reduction as usual (into the first buffer).
void ResultSetStorage::moveEntriesToBuffer(int8_t* new_buff, const size_t new_entry_count) const {
  CHECK(!query_mem_desc_.keyless_hash);
  CHECK_GT(new_entry_count, query_mem_desc_.entry_count);
  auto new_buff_i64 = reinterpret_cast<int64_t*>(new_buff);
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc_);
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  CHECK(GroupByColRangeType::MultiCol == query_mem_desc_.hash_type);
  const auto this_buff = reinterpret_cast<const int64_t*>(buff_);
  for (size_t i = 0; i < query_mem_desc_.entry_count; ++i) {
    const auto key_off = query_mem_desc_.output_columnar ? key_offset_colwise(i, 0, query_mem_desc_.entry_count)
                                                         : key_offset_rowwise(i, key_count, slot_count);
    if (this_buff[key_off] == EMPTY_KEY_64) {
      continue;
    }
    int64_t* new_entries_ptr{nullptr};
    if (query_mem_desc_.output_columnar) {
      const auto key = make_key(&this_buff[key_off], query_mem_desc_.entry_count, key_count);
      new_entries_ptr = get_group_value_columnar(new_buff_i64, new_entry_count, &key[0], key_count);
    } else {
      new_entries_ptr = get_group_value(
          new_buff_i64, new_entry_count, &this_buff[key_off], key_count, get_row_qw_count(query_mem_desc_), nullptr);
    }
    CHECK(new_entries_ptr);
    if (query_mem_desc_.output_columnar) {
      size_t new_slot_off = 0;
      for (size_t j = 0; j < slot_count; ++j) {
        new_entries_ptr[new_slot_off] = this_buff[slot_offset_colwise(i, j, key_count, query_mem_desc_.entry_count)];
        new_slot_off += new_entry_count;
      }
    } else {
      for (size_t j = 0; j < slot_count; ++j) {
        new_entries_ptr[j] = this_buff[slot_offset_rowwise(i, j, key_count, slot_count)];
      }
    }
  }
}

void ResultSet::initializeStorage() const {
  if (query_mem_desc_.output_columnar) {
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
  auto& first_result = *result_sets.front()->storage_;
  auto result = &first_result;
  if (first_result.query_mem_desc_.hash_type == GroupByColRangeType::MultiCol) {
    const auto total_entry_count =
        std::accumulate(result_sets.begin(), result_sets.end(), size_t(0), [](const size_t init, const ResultSet* rs) {
          return init + rs->query_mem_desc_.entry_count;
        });
    CHECK(total_entry_count);
    auto query_mem_desc = first_result.query_mem_desc_;
    query_mem_desc.entry_count = total_entry_count * 2;
    rs_.reset(new ResultSet(first_result.targets_, ExecutorDeviceType::CPU, query_mem_desc));
    auto result_storage = rs_->allocateStorage();
    rs_->initializeStorage();
    first_result.moveEntriesToBuffer(result_storage->getUnderlyingBuffer(), query_mem_desc.entry_count);
    result = rs_->storage_.get();
    result_rs = rs_.get();
  }
  for (auto result_it = result_sets.begin() + 1; result_it != result_sets.end(); ++result_it) {
    result->reduce(*((*result_it)->storage_));
  }
  return result_rs;
}

void ResultSetStorage::initializeRowWise() const {
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc_);
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  auto this_buff = reinterpret_cast<int64_t*>(buff_);
  CHECK(!query_mem_desc_.keyless_hash);
  for (size_t i = 0; i < query_mem_desc_.entry_count; ++i) {
    const auto key_off = key_offset_rowwise(i, key_count, slot_count);
    for (size_t j = 0; j < key_count; ++j) {
      this_buff[key_off + j] = EMPTY_KEY_64;
    }
    const auto first_slot_off = slot_offset_rowwise(i, 0, key_count, slot_count);
    for (size_t j = 0; j < target_init_vals_.size(); ++j) {
      this_buff[first_slot_off + j] = target_init_vals_[j];
    }
  }
}

void ResultSetStorage::initializeColWise() const {
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  auto this_buff = reinterpret_cast<int64_t*>(buff_);
  CHECK(!query_mem_desc_.keyless_hash);
  for (size_t key_idx = 0; key_idx < key_count; ++key_idx) {
    const auto first_key_off = key_offset_colwise(0, key_idx, query_mem_desc_.entry_count);
    for (size_t i = 0; i < query_mem_desc_.entry_count; ++i) {
      this_buff[first_key_off + i] = EMPTY_KEY_64;
    }
  }
  for (size_t target_idx = 0; target_idx < target_init_vals_.size(); ++target_idx) {
    const auto first_val_off = slot_offset_colwise(0, target_idx, key_count, query_mem_desc_.entry_count);
    for (size_t i = 0; i < query_mem_desc_.entry_count; ++i) {
      this_buff[first_val_off + i] = target_init_vals_[target_idx];
    }
  }
}

void ResultSetStorage::initializeBaselineValueSlots(int64_t* entry_slots) const {
  CHECK(entry_slots);
  if (query_mem_desc_.output_columnar) {
    size_t slot_off = 0;
    for (size_t j = 0; j < target_init_vals_.size(); ++j) {
      entry_slots[slot_off] = target_init_vals_[j];
      slot_off += query_mem_desc_.entry_count;
    }
  } else {
    for (size_t j = 0; j < target_init_vals_.size(); ++j) {
      entry_slots[j] = target_init_vals_[j];
    }
  }
}

namespace {

const bool min_check_flag = false;
const bool max_check_flag = false;
const bool sum_check_flag = true;

}  // namespace

#define AGGREGATE_ONE_VALUE(agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__)                            \
  {                                                                                                                    \
    const auto sql_type = get_compact_type(agg_info__);                                                                \
    if (sql_type.is_fp()) {                                                                                            \
      if (chosen_bytes__ == sizeof(float)) {                                                                           \
        agg_##agg_kind__##_float(reinterpret_cast<int32_t*>(val_ptr__), *reinterpret_cast<const float*>(other_ptr__)); \
      } else {                                                                                                         \
        agg_##agg_kind__##_double(reinterpret_cast<int64_t*>(val_ptr__),                                               \
                                  *reinterpret_cast<const double*>(other_ptr__));                                      \
      }                                                                                                                \
    } else {                                                                                                           \
      if (chosen_bytes__ == sizeof(int32_t)) {                                                                         \
        auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                                                          \
        auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);                                                \
        if (agg_kind__##_check_flag &&                                                                                 \
            detect_overflow_and_underflow(*val_ptr, *other_ptr, false, int32_t(0), sql_type)) {                        \
          throw OverflowOrUnderflow();                                                                                 \
        }                                                                                                              \
        agg_##agg_kind__##_int32(val_ptr, *other_ptr);                                                                 \
      } else {                                                                                                         \
        auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                                                          \
        auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);                                                \
        if (agg_kind__##_check_flag &&                                                                                 \
            detect_overflow_and_underflow(*val_ptr, *other_ptr, false, int64_t(0), sql_type)) {                        \
          throw OverflowOrUnderflow();                                                                                 \
        }                                                                                                              \
        agg_##agg_kind__(val_ptr, *other_ptr);                                                                         \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define AGGREGATE_ONE_NULLABLE_VALUE(agg_kind__, val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__) \
  {                                                                                                              \
    if (agg_info__.skip_null_val) {                                                                              \
      const auto sql_type = get_compact_type(agg_info__);                                                        \
      if (sql_type.is_fp()) {                                                                                    \
        if (chosen_bytes__ == sizeof(float)) {                                                                   \
          agg_##agg_kind__##_float_skip_val(reinterpret_cast<int32_t*>(val_ptr__),                               \
                                            *reinterpret_cast<const float*>(other_ptr__),                        \
                                            *reinterpret_cast<const float*>(&init_val__));                       \
        } else {                                                                                                 \
          agg_##agg_kind__##_double_skip_val(reinterpret_cast<int64_t*>(val_ptr__),                              \
                                             *reinterpret_cast<const double*>(other_ptr__),                      \
                                             *reinterpret_cast<const double*>(&init_val__));                     \
        }                                                                                                        \
      } else {                                                                                                   \
        if (chosen_bytes__ == sizeof(int32_t)) {                                                                 \
          int32_t* val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                                              \
          const int32_t* other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);                              \
          const auto null_val = static_cast<int32_t>(init_val__);                                                \
          if (agg_kind__##_check_flag &&                                                                         \
              detect_overflow_and_underflow(*val_ptr, *other_ptr, true, null_val, sql_type)) {                   \
            throw OverflowOrUnderflow();                                                                         \
          }                                                                                                      \
          agg_##agg_kind__##_int32_skip_val(val_ptr, *other_ptr, null_val);                                      \
        } else {                                                                                                 \
          int64_t* val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                                              \
          const int64_t* other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);                              \
          const auto null_val = static_cast<int64_t>(init_val__);                                                \
          if (agg_kind__##_check_flag &&                                                                         \
              detect_overflow_and_underflow(*val_ptr, *other_ptr, true, null_val, sql_type)) {                   \
            throw OverflowOrUnderflow();                                                                         \
          }                                                                                                      \
          agg_##agg_kind__##_skip_val(val_ptr, *other_ptr, null_val);                                            \
        }                                                                                                        \
      }                                                                                                          \
    } else {                                                                                                     \
      AGGREGATE_ONE_VALUE(agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__);                       \
    }                                                                                                            \
  }

void ResultSetStorage::reduceOneSlot(int8_t* this_ptr1,
                                     int8_t* this_ptr2,
                                     const int8_t* that_ptr1,
                                     const int8_t* that_ptr2,
                                     const TargetInfo& target_info,
                                     const size_t target_slot_idx) const {
  CHECK_LT(target_slot_idx, target_init_vals_.size());
  CHECK_LT(target_slot_idx, query_mem_desc_.agg_col_widths.size());
  const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_slot_idx].compact;
  const auto init_val = target_init_vals_[target_slot_idx];
  if (target_info.is_agg) {
    switch (target_info.agg_kind) {
      case kAVG: {
        AGGREGATE_ONE_VALUE(sum, this_ptr2, that_ptr2, chosen_bytes, target_info);
      }
      // fall thru
      case kCOUNT: {
        if (target_info.is_distinct) {
          CHECK(false);
          break;
        }
      }
      case kSUM: {
        AGGREGATE_ONE_NULLABLE_VALUE(sum, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        break;
      }
      case kMIN: {
        AGGREGATE_ONE_NULLABLE_VALUE(min, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        break;
      }
      case kMAX: {
        AGGREGATE_ONE_NULLABLE_VALUE(max, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        break;
      }
      default:
        CHECK(false);
    }
  } else {
    if (query_mem_desc_.hash_type == GroupByColRangeType::MultiCol) {
      memcpy(this_ptr1, that_ptr1, chosen_bytes);
    } else {
      CHECK_EQ(0, memcmp(this_ptr1, that_ptr1, chosen_bytes));
    }
  }
}
