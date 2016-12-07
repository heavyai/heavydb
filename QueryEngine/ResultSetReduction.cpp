/**
 * @file    ResultSetReduction.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Reduction part of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultRows.h"
#include "ResultSet.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"

#include "Shared/thread_count.h"

#include <future>
#include <numeric>

namespace {

bool use_multithreaded_reduction(const size_t entry_count) {
  return entry_count > 100000;
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

void fill_slots(int64_t* dst_entry,
                const size_t dst_entry_count,
                const int64_t* src_buff,
                const size_t src_entry_idx,
                const size_t src_entry_count,
                const QueryMemoryDescriptor& query_mem_desc) {
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc);
  const auto key_count = get_groupby_col_count(query_mem_desc);
  if (query_mem_desc.output_columnar) {
    for (size_t i = 0, dst_slot_off = 0; i < slot_count; ++i, dst_slot_off += dst_entry_count) {
      dst_entry[dst_slot_off] = src_buff[slot_offset_colwise(src_entry_idx, i, key_count, src_entry_count)];
    }
  } else {
    for (size_t i = 0; i < slot_count; ++i) {
      dst_entry[i] = src_buff[slot_offset_rowwise(src_entry_idx, i, key_count, slot_count)];
    }
  }
}

}  // namespace

// Driver method for various buffer layouts, actual work is done by reduceOne* methods.
// Reduces the entries of `that` into the buffer of this ResultSetStorage object.
void ResultSetStorage::reduce(const ResultSetStorage& that) const {
  auto entry_count = query_mem_desc_.entry_count;
  CHECK_GT(entry_count, size_t(0));
  switch (query_mem_desc_.hash_type) {
    case GroupByColRangeType::MultiCol:
      CHECK_EQ(size_t(0), query_mem_desc_.entry_count_small);
      CHECK_GE(entry_count, that.query_mem_desc_.entry_count);
      break;
    case GroupByColRangeType::OneColGuessedRange:
      CHECK_NE(size_t(0), query_mem_desc_.entry_count_small);
      CHECK_EQ(query_mem_desc_.entry_count_small, that.query_mem_desc_.entry_count_small);
      CHECK_GE(entry_count, that.query_mem_desc_.entry_count);
      break;
    default:
      CHECK_EQ(entry_count, that.query_mem_desc_.entry_count);
  }
  auto this_buff = buff_;
  CHECK(this_buff);
  auto that_buff = that.buff_;
  CHECK(that_buff);
  if (query_mem_desc_.hash_type == GroupByColRangeType::MultiCol) {
    if (use_multithreaded_reduction(that.query_mem_desc_.entry_count)) {
      const size_t thread_count = cpu_threads();
      std::vector<std::future<void>> reduction_threads;
      for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        reduction_threads.emplace_back(
            std::async(std::launch::async, [this, thread_idx, thread_count, this_buff, that_buff, &that] {
              for (size_t i = thread_idx; i < that.query_mem_desc_.entry_count; i += thread_count) {
                reduceOneEntryBaseline(this_buff, that_buff, i, that.query_mem_desc_.entry_count);
              }
            }));
      }
      for (auto& reduction_thread : reduction_threads) {
        reduction_thread.get();
      }
    } else {
      for (size_t i = 0; i < that.query_mem_desc_.entry_count; ++i) {
        reduceOneEntryBaseline(this_buff, that_buff, i, that.query_mem_desc_.entry_count);
      }
    }
    return;
  }
  if (query_mem_desc_.hash_type == GroupByColRangeType::OneColGuessedRange) {
    CHECK(!query_mem_desc_.output_columnar);
    for (size_t i = 0; i < that.query_mem_desc_.entry_count; ++i) {
      reduceOneEntryBaseline(this_buff, that_buff, i, that.query_mem_desc_.entry_count);
    }
    entry_count = query_mem_desc_.entry_count_small;
    const auto row_bytes = get_row_bytes(query_mem_desc_);
    CHECK_EQ(get_row_bytes(that.query_mem_desc_), row_bytes);
    this_buff += query_mem_desc_.entry_count * row_bytes;
    that_buff += that.query_mem_desc_.entry_count * row_bytes;
  }
  if (use_multithreaded_reduction(entry_count)) {
    const size_t thread_count = cpu_threads();
    std::vector<std::future<void>> reduction_threads;
    for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      reduction_threads.emplace_back(
          std::async(std::launch::async, [this, thread_idx, entry_count, this_buff, that_buff, thread_count] {
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
  if (isEmptyEntry(entry_idx, that_buff)) {
    return;
  }
  // copy the key from right hand side
  if (!query_mem_desc_.keyless_hash) {
    copyKeyColWise(entry_idx, this_buff, that_buff);
  }
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
    reduceOneSlot(this_ptr1, this_ptr2, that_ptr1, that_ptr2, agg_info, target_idx, agg_col_idx, agg_col_idx);
    this_crt_col_ptr = this_next_col_ptr;
    that_crt_col_ptr = that_next_col_ptr;
    if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
      this_crt_col_ptr = advance_to_next_columnar_target_buff(this_crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
      that_crt_col_ptr = advance_to_next_columnar_target_buff(that_crt_col_ptr, query_mem_desc_, agg_col_idx + 1);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info);
  }
}

void ResultSetStorage::copyKeyColWise(const size_t entry_idx, int8_t* this_buff, const int8_t* that_buff) const {
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  // TODO(alex): we might want to support keys smaller than 64 bits at some point
  auto lhs_key_buff = reinterpret_cast<int64_t*>(this_buff) + entry_idx;
  auto rhs_key_buff = reinterpret_cast<const int64_t*>(that_buff) + entry_idx;
  for (size_t key_comp_idx = 0; key_comp_idx < key_count; ++key_comp_idx) {
    *lhs_key_buff = *rhs_key_buff;
    lhs_key_buff += query_mem_desc_.entry_count;
    rhs_key_buff += query_mem_desc_.entry_count;
  }
}

// Reduces entry at position entry_idx in that_buff into the same position in this_buff, row-wise format.
void ResultSetStorage::reduceOneEntryNoCollisionsRowWise(const size_t entry_idx,
                                                         int8_t* this_buff,
                                                         const int8_t* that_buff) const {
  CHECK(!query_mem_desc_.output_columnar);
  if (isEmptyEntry(entry_idx, that_buff)) {
    return;
  }
  const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
  auto this_targets_ptr = row_ptr_rowwise(this_buff, query_mem_desc_, entry_idx) + key_bytes;
  auto that_targets_ptr = row_ptr_rowwise(that_buff, query_mem_desc_, entry_idx) + key_bytes;
  if (key_bytes) {  // copy the key from right hand side
    memcpy(this_targets_ptr - key_bytes, that_targets_ptr - key_bytes, key_bytes);
  }
  size_t target_slot_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size(); ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    int8_t* this_ptr2{nullptr};
    const int8_t* that_ptr2{nullptr};
    if (target_info.is_agg && target_info.agg_kind == kAVG) {
      this_ptr2 = this_targets_ptr + query_mem_desc_.agg_col_widths[target_slot_idx].compact;
      that_ptr2 = that_targets_ptr + query_mem_desc_.agg_col_widths[target_slot_idx].compact;
    }
    reduceOneSlot(this_targets_ptr,
                  this_ptr2,
                  that_targets_ptr,
                  that_ptr2,
                  target_info,
                  target_logical_idx,
                  target_slot_idx,
                  target_slot_idx);
    this_targets_ptr = advance_target_ptr(this_targets_ptr, target_info, target_slot_idx, query_mem_desc_);
    that_targets_ptr = advance_target_ptr(that_targets_ptr, target_info, target_slot_idx, query_mem_desc_);
    target_slot_idx = advance_slot(target_slot_idx, target_info);
  }
}

namespace {

typedef std::pair<int64_t*, bool> GroupValueInfo;

GroupValueInfo get_matching_group_value_columnar_reduction(int64_t* groups_buffer,
                                                           const uint32_t h,
                                                           const int64_t* key,
                                                           const uint32_t key_qw_count,
                                                           const size_t entry_count) {
  auto off = h;
  const auto old_key = __sync_val_compare_and_swap(&groups_buffer[off], EMPTY_KEY_64, *key);
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
GroupValueInfo get_group_value_columnar_reduction(int64_t* groups_buffer,
                                                  const uint32_t groups_buffer_entry_count,
                                                  const int64_t* key,
                                                  const uint32_t key_qw_count) {
  uint32_t h = key_hash(key, key_qw_count) % groups_buffer_entry_count;
  auto matching_gvi =
      get_matching_group_value_columnar_reduction(groups_buffer, h, key, key_qw_count, groups_buffer_entry_count);
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
  __atomic_compare_exchange_n(ptr, expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define store_cst(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST)
#define load_cst(ptr) __atomic_load_n(ptr, __ATOMIC_SEQ_CST)

GroupValueInfo get_matching_group_value_reduction(int64_t* groups_buffer,
                                                  const uint32_t h,
                                                  const int64_t* key,
                                                  const uint32_t key_qw_count,
                                                  const uint32_t row_size_quad,
                                                  const int64_t* init_vals) {
  auto off = h * row_size_quad;
  int64_t empty_key = EMPTY_KEY_64;
  const bool success = cas_cst(&groups_buffer[off], &empty_key, *key);
  if (success && key_qw_count > 1) {
    for (size_t i = 1; i <= key_qw_count - 1; ++i) {
      store_cst(groups_buffer + off + i, key[i]);
    }
    return {groups_buffer + off + key_qw_count, true};
  }
  if (key_qw_count > 1) {
    while (load_cst(groups_buffer + off + key_qw_count - 1) == EMPTY_KEY_64) {
      // spin until the winning thread has finished writing the entire key and the init value
    }
  }
  if (memcmp(groups_buffer + off, key, key_qw_count * sizeof(*key)) == 0) {
    return {groups_buffer + off + key_qw_count, false};
  }
  return {nullptr, true};
}

#undef load_cst
#undef store_cst
#undef cas_cst

GroupValueInfo get_group_value_reduction(int64_t* groups_buffer,
                                         const uint32_t groups_buffer_entry_count,
                                         const int64_t* key,
                                         const uint32_t key_qw_count,
                                         const uint32_t row_size_quad,
                                         const int64_t* init_vals) {
  uint32_t h = key_hash(key, key_qw_count) % groups_buffer_entry_count;
  auto matching_gvi = get_matching_group_value_reduction(groups_buffer, h, key, key_qw_count, row_size_quad, init_vals);
  if (matching_gvi.first) {
    return matching_gvi;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_gvi =
        get_matching_group_value_reduction(groups_buffer, h_probe, key, key_qw_count, row_size_quad, init_vals);
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
                                              const size_t that_entry_count) const {
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc_);
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  CHECK(GroupByColRangeType::MultiCol == query_mem_desc_.hash_type ||
        GroupByColRangeType::OneColGuessedRange == query_mem_desc_.hash_type);
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
  bool empty_entry = false;
  if (query_mem_desc_.output_columnar) {
    const auto key = make_key(&that_buff_i64[key_off], that_entry_count, key_count);
    std::tie(this_entry_slots, empty_entry) =
        get_group_value_columnar_reduction(this_buff_i64, query_mem_desc_.entry_count, &key[0], key_count);
  } else {
    std::tie(this_entry_slots, empty_entry) = get_group_value_reduction(this_buff_i64,
                                                                        query_mem_desc_.entry_count,
                                                                        &that_buff_i64[key_off],
                                                                        key_count,
                                                                        get_row_qw_count(query_mem_desc_),
                                                                        nullptr);
  }
  CHECK(this_entry_slots);
  if (empty_entry) {
    fill_slots(this_entry_slots,
               query_mem_desc_.entry_count,
               that_buff_i64,
               that_entry_idx,
               that_entry_count,
               query_mem_desc_);
    return;
  }
  reduceOneEntrySlotsBaseline(this_entry_slots, that_buff_i64, that_entry_idx, that_entry_count);
}

void ResultSetStorage::reduceOneEntrySlotsBaseline(int64_t* this_entry_slots,
                                                   const int64_t* that_buff,
                                                   const size_t that_entry_idx,
                                                   const size_t that_entry_count) const {
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc_);
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  size_t j = 0;
  size_t init_agg_val_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size(); ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    const auto that_slot_off = query_mem_desc_.output_columnar
                                   ? slot_offset_colwise(that_entry_idx, j, key_count, that_entry_count)
                                   : slot_offset_rowwise(that_entry_idx, init_agg_val_idx, key_count, slot_count);
    const auto this_slot_off = query_mem_desc_.output_columnar ? j * query_mem_desc_.entry_count : init_agg_val_idx;
    reduceOneSlotBaseline(this_entry_slots,
                          this_slot_off,
                          that_buff,
                          that_entry_count,
                          that_slot_off,
                          target_info,
                          target_logical_idx,
                          j,
                          init_agg_val_idx);
    if (query_mem_desc_.target_groupby_indices.empty()) {
      init_agg_val_idx = advance_slot(init_agg_val_idx, target_info);
    } else {
      CHECK_LT(target_logical_idx, query_mem_desc_.target_groupby_indices.size());
      if (query_mem_desc_.target_groupby_indices[target_logical_idx] < 0) {
        init_agg_val_idx = advance_slot(init_agg_val_idx, target_info);
      }
    }
    j = advance_slot(j, target_info);
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
                                             const size_t init_agg_val_idx) const {
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
                target_logical_idx,
                target_slot_idx,
                init_agg_val_idx);
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
    fill_slots(new_entries_ptr, new_entry_count, this_buff, i, query_mem_desc_.entry_count, query_mem_desc_);
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
  if (first_result.query_mem_desc_.hash_type == GroupByColRangeType::MultiCol) {
    const auto total_entry_count =
        std::accumulate(result_sets.begin(), result_sets.end(), size_t(0), [](const size_t init, const ResultSet* rs) {
          return init + rs->query_mem_desc_.entry_count;
        });
    CHECK(total_entry_count);
    auto query_mem_desc = first_result.query_mem_desc_;
    query_mem_desc.entry_count = total_entry_count * 2;
    rs_.reset(
        new ResultSet(first_result.targets_, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, nullptr));
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

std::unique_ptr<ResultSet>& ResultSetManager::getOwnResultSet() {
  return rs_;
}

void ResultSetStorage::fillOneEntryRowWise(const std::vector<int64_t>& entry) {
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc_);
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  CHECK_EQ(slot_count + key_count, entry.size());
  auto this_buff = reinterpret_cast<int64_t*>(buff_);
  CHECK(!query_mem_desc_.output_columnar);
  CHECK_EQ(size_t(1), query_mem_desc_.entry_count);
  const auto key_off = key_offset_rowwise(0, key_count, slot_count);
  for (size_t i = 0; i < key_count; ++i) {
    this_buff[key_off + i] = entry[i];
  }
  const auto first_slot_off = slot_offset_rowwise(0, 0, key_count, slot_count);
  for (size_t i = 0; i < target_init_vals_.size(); ++i) {
    this_buff[first_slot_off + i] = entry[key_count + i];
  }
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

#define AGGREGATE_ONE_COUNT(val_ptr__, other_ptr__, chosen_bytes__, agg_info__)                                      \
  {                                                                                                                  \
    const auto sql_type = get_compact_type(agg_info__);                                                              \
    if (sql_type.is_fp()) {                                                                                          \
      if (chosen_bytes__ == sizeof(float)) {                                                                         \
        agg_sum_float(reinterpret_cast<int32_t*>(val_ptr__), *reinterpret_cast<const float*>(other_ptr__));          \
      } else {                                                                                                       \
        agg_sum_double(reinterpret_cast<int64_t*>(val_ptr__), *reinterpret_cast<const double*>(other_ptr__));        \
      }                                                                                                              \
    } else {                                                                                                         \
      if (chosen_bytes__ == sizeof(int32_t)) {                                                                       \
        auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                                                        \
        auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);                                              \
        if (detect_overflow_and_underflow(                                                                           \
                static_cast<uint32_t>(*val_ptr), static_cast<uint32_t>(*other_ptr), false, uint32_t(0), sql_type)) { \
          throw OverflowOrUnderflow();                                                                               \
        }                                                                                                            \
        agg_sum_int32(val_ptr, *other_ptr);                                                                          \
      } else {                                                                                                       \
        auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                                                        \
        auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);                                              \
        if (detect_overflow_and_underflow(                                                                           \
                static_cast<uint64_t>(*val_ptr), static_cast<uint64_t>(*other_ptr), false, uint64_t(0), sql_type)) { \
          throw OverflowOrUnderflow();                                                                               \
        }                                                                                                            \
        agg_sum(val_ptr, *other_ptr);                                                                                \
      }                                                                                                              \
    }                                                                                                                \
  }

#define AGGREGATE_ONE_NULLABLE_COUNT(val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__) \
  {                                                                                                  \
    if (agg_info__.skip_null_val) {                                                                  \
      const auto sql_type = get_compact_type(agg_info__);                                            \
      if (sql_type.is_fp()) {                                                                        \
        if (chosen_bytes__ == sizeof(float)) {                                                       \
          agg_sum_float_skip_val(reinterpret_cast<int32_t*>(val_ptr__),                              \
                                 *reinterpret_cast<const float*>(other_ptr__),                       \
                                 *reinterpret_cast<const float*>(&init_val__));                      \
        } else {                                                                                     \
          agg_sum_double_skip_val(reinterpret_cast<int64_t*>(val_ptr__),                             \
                                  *reinterpret_cast<const double*>(other_ptr__),                     \
                                  *reinterpret_cast<const double*>(&init_val__));                    \
        }                                                                                            \
      } else {                                                                                       \
        if (chosen_bytes__ == sizeof(int32_t)) {                                                     \
          auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                                      \
          auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);                            \
          const auto null_val = static_cast<int32_t>(init_val__);                                    \
          if (detect_overflow_and_underflow(static_cast<uint32_t>(*val_ptr),                         \
                                            static_cast<uint32_t>(*other_ptr),                       \
                                            true,                                                    \
                                            static_cast<uint32_t>(null_val),                         \
                                            sql_type)) {                                             \
            throw OverflowOrUnderflow();                                                             \
          }                                                                                          \
          agg_sum_int32_skip_val(val_ptr, *other_ptr, null_val);                                     \
        } else {                                                                                     \
          auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                                      \
          auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);                            \
          const auto null_val = static_cast<int64_t>(init_val__);                                    \
          if (detect_overflow_and_underflow(static_cast<uint64_t>(*val_ptr),                         \
                                            static_cast<uint64_t>(*other_ptr),                       \
                                            true,                                                    \
                                            static_cast<uint64_t>(null_val),                         \
                                            sql_type)) {                                             \
            throw OverflowOrUnderflow();                                                             \
          }                                                                                          \
          agg_sum_skip_val(val_ptr, *other_ptr, null_val);                                           \
        }                                                                                            \
      }                                                                                              \
    } else {                                                                                         \
      AGGREGATE_ONE_COUNT(val_ptr__, other_ptr__, chosen_bytes__, agg_info__);                       \
    }                                                                                                \
  }

void ResultSetStorage::reduceOneSlot(int8_t* this_ptr1,
                                     int8_t* this_ptr2,
                                     const int8_t* that_ptr1,
                                     const int8_t* that_ptr2,
                                     const TargetInfo& target_info,
                                     const size_t target_logical_idx,
                                     const size_t target_slot_idx,
                                     const size_t init_agg_val_idx) const {
  if (!query_mem_desc_.target_groupby_indices.empty()) {
    CHECK_LT(target_logical_idx, query_mem_desc_.target_groupby_indices.size());
    if (query_mem_desc_.target_groupby_indices[target_logical_idx] >= 0) {
      return;
    }
  }
  CHECK_LT(init_agg_val_idx, target_init_vals_.size());
  CHECK_LT(target_slot_idx, query_mem_desc_.agg_col_widths.size());
  const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_slot_idx].compact;
  auto init_val = target_init_vals_[init_agg_val_idx];
  if (target_info.is_agg) {
    switch (target_info.agg_kind) {
      case kCOUNT: {
        if (target_info.is_distinct) {
          CHECK(target_info.is_agg);
          CHECK_EQ(kCOUNT, target_info.agg_kind);
          CHECK_EQ(static_cast<size_t>(chosen_bytes), sizeof(int64_t));
          reduceOneCountDistinctSlot(this_ptr1, that_ptr1, target_info, target_logical_idx);
          break;
        }
        AGGREGATE_ONE_NULLABLE_COUNT(this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        break;
      }
      case kAVG: {
        AGGREGATE_ONE_COUNT(this_ptr2, that_ptr2, chosen_bytes, target_info);
      }
      // fall thru
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
    switch (chosen_bytes) {
      case 4: {
        const auto rhs_proj_col = *reinterpret_cast<const int32_t*>(that_ptr1);
        if (rhs_proj_col) {
          *reinterpret_cast<int32_t*>(this_ptr1) = rhs_proj_col;
        }
        break;
      }
      case 8: {
        const auto rhs_proj_col = *reinterpret_cast<const int64_t*>(that_ptr1);
        if (rhs_proj_col) {
          *reinterpret_cast<int64_t*>(this_ptr1) = rhs_proj_col;
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
                                                  const TargetInfo& target_info,
                                                  const size_t target_logical_idx) const {
  auto count_distinct_desc_it = query_mem_desc_.count_distinct_descriptors_.find(target_logical_idx);
  CHECK(count_distinct_desc_it != query_mem_desc_.count_distinct_descriptors_.end());
  auto old_set_ptr = reinterpret_cast<const int64_t*>(this_ptr1);
  auto new_set_ptr = reinterpret_cast<const int64_t*>(that_ptr1);
  if (count_distinct_desc_it->second.impl_type_ == CountDistinctImplType::Bitmap) {
    auto old_set = reinterpret_cast<int8_t*>(*old_set_ptr);
    auto new_set = reinterpret_cast<int8_t*>(*new_set_ptr);
    bitmap_set_unify(new_set, old_set, count_distinct_desc_it->second.bitmapSizeBytes());
  } else {
    CHECK(count_distinct_desc_it->second.impl_type_ == CountDistinctImplType::StdSet);
    auto old_set = reinterpret_cast<std::set<int64_t>*>(*old_set_ptr);
    auto new_set = reinterpret_cast<std::set<int64_t>*>(*new_set_ptr);
    old_set->insert(new_set->begin(), new_set->end());
    new_set->insert(old_set->begin(), old_set->end());
  }
}
