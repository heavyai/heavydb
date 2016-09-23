/**
 * @file    ResultSet.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultSet.h"
#include "Execute.h"
#include "InPlaceSort.h"
#include "OutputBufferInitialization.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Shared/checked_alloc.h"
#include "Shared/thread_count.h"

#include <algorithm>
#include <future>
#include <numeric>

ResultSetStorage::ResultSetStorage(const std::vector<TargetInfo>& targets,
                                   const ExecutorDeviceType device_type,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   int8_t* buff)
    : targets_(targets), query_mem_desc_(query_mem_desc), buff_(buff) {
  for (const auto& target_info : targets_) {
    target_init_vals_.push_back(target_info.is_agg ? 0xdeadbeef : 0);
    if (target_info.agg_kind == kAVG) {
      target_init_vals_.push_back(0);
    }
  }
}

int8_t* ResultSetStorage::getUnderlyingBuffer() const {
  return buff_;
}

void ResultSet::keepFirstN(const size_t n) {
  keep_first_ = n;
}

void ResultSet::dropFirstN(const size_t n) {
  drop_first_ = n;
}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const ExecutorDeviceType device_type,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const Executor* executor)
    : targets_(targets),
      device_type_(device_type),
      query_mem_desc_(query_mem_desc),
      crt_row_buff_idx_(0),
      fetched_so_far_(0),
      drop_first_(0),
      keep_first_(0),
      row_set_mem_owner_(row_set_mem_owner),
      executor_(executor) {
}

ResultSet::ResultSet() : device_type_(ExecutorDeviceType::CPU), query_mem_desc_{}, crt_row_buff_idx_(0) {
}

ResultSet::~ResultSet() {
  if (storage_) {
    CHECK(storage_->buff_);
    free(storage_->buff_);
  }
}

const ResultSetStorage* ResultSet::allocateStorage() const {
  CHECK(!storage_);
  auto buff = static_cast<int8_t*>(checked_malloc(query_mem_desc_.getBufferSizeBytes(device_type_)));
  return allocateStorage(buff);
}

const ResultSetStorage* ResultSet::allocateStorage(int8_t* buff) const {
  CHECK(buff);
  storage_.reset(new ResultSetStorage(targets_, device_type_, query_mem_desc_, buff));
  return storage_.get();
}

void ResultSet::append(ResultSet& that) {
  CHECK(that.appended_storage_.empty());
  appended_storage_.push_back(std::move(that.storage_));
  CHECK(false);
}

void ResultSet::sort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n) {
  if (isEmptyInitializer()) {
    return;
  }
  // This check isn't strictly required, but allows the index buffer to be 32-bit.
  if (query_mem_desc_.entry_count > std::numeric_limits<uint32_t>::max()) {
    throw RowSortException("Sorting more than 4B elements not supported");
  }
  CHECK_EQ(size_t(0), query_mem_desc_.entry_count_small);  // TODO(alex)
  CHECK(permutation_.empty());
  permutation_.resize(query_mem_desc_.entry_count);
  std::iota(permutation_.begin(), permutation_.end(), 0);
  CHECK(!query_mem_desc_.sortOnGpu());  // TODO(alex)
  const bool use_heap{order_entries.size() == 1 && top_n};
  auto compare = createComparator(order_entries, use_heap);
  if (g_enable_watchdog && (query_mem_desc_.entry_count + query_mem_desc_.entry_count_small > 100000)) {
    throw WatchdogException("Sorting the result would be too slow");
  }
  if (use_heap) {
    topPermutation(top_n, compare);
  } else {
    sortPermutation(compare);
  }
}

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

std::function<bool(const uint32_t, const uint32_t)> ResultSet::createComparator(
    const std::list<Analyzer::OrderEntry>& order_entries,
    const bool use_heap) const {
  return [this, &order_entries, use_heap](const uint32_t lhs, const uint32_t rhs) {
    // NB: The compare function must define a strict weak ordering, otherwise
    // std::sort will trigger a segmentation fault (or corrupt memory).
    for (const auto order_entry : order_entries) {
      CHECK_GE(order_entry.tle_no, 1);
      const auto& entry_ti = get_compact_type(targets_[order_entry.tle_no - 1]);
      const auto is_dict = entry_ti.is_string() && entry_ti.get_compression() == kENCODING_DICT;
      if (storage_->isEmptyEntry(lhs, storage_->buff_) && storage_->isEmptyEntry(rhs, storage_->buff_)) {
        return false;
      }
      if (storage_->isEmptyEntry(lhs, storage_->buff_) && !storage_->isEmptyEntry(rhs, storage_->buff_)) {
        return use_heap;
      }
      if (storage_->isEmptyEntry(rhs, storage_->buff_) && !storage_->isEmptyEntry(lhs, storage_->buff_)) {
        return !use_heap;
      }
      const auto lhs_v = getColumnInternal(lhs, order_entry.tle_no - 1);
      const auto rhs_v = getColumnInternal(rhs, order_entry.tle_no - 1);
      if (UNLIKELY(isNull(entry_ti, lhs_v) && isNull(entry_ti, rhs_v))) {
        return false;
      }
      if (UNLIKELY(isNull(entry_ti, lhs_v) && !isNull(entry_ti, rhs_v))) {
        return use_heap ? !order_entry.nulls_first : order_entry.nulls_first;
      }
      if (UNLIKELY(isNull(entry_ti, rhs_v) && !isNull(entry_ti, lhs_v))) {
        return use_heap ? order_entry.nulls_first : !order_entry.nulls_first;
      }
      const bool use_desc_cmp = use_heap ? !order_entry.is_desc : order_entry.is_desc;
      if (LIKELY(lhs_v.isInt())) {
        CHECK(rhs_v.isInt());
        if (UNLIKELY(is_dict)) {
          CHECK_EQ(4, entry_ti.get_logical_size());
          const auto string_dict = executor_->getStringDictionary(entry_ti.get_comp_param(), row_set_mem_owner_);
          auto lhs_str = string_dict->getString(lhs_v.i1);
          auto rhs_str = string_dict->getString(rhs_v.i1);
          if (lhs_str == rhs_str) {
            continue;
          }
          return use_desc_cmp ? lhs_str > rhs_str : lhs_str < rhs_str;
        }
        if (UNLIKELY(targets_[order_entry.tle_no - 1].is_distinct)) {
          const auto lhs_sz =
              bitmap_set_size(lhs_v.i1, order_entry.tle_no - 1, row_set_mem_owner_->getCountDistinctDescriptors());
          const auto rhs_sz =
              bitmap_set_size(rhs_v.i1, order_entry.tle_no - 1, row_set_mem_owner_->getCountDistinctDescriptors());
          if (lhs_sz == rhs_sz) {
            continue;
          }
          return use_desc_cmp ? lhs_sz > rhs_sz : lhs_sz < rhs_sz;
        }
        if (lhs_v.i1 == rhs_v.i1) {
          continue;
        }
        return use_desc_cmp ? lhs_v.i1 > rhs_v.i1 : lhs_v.i1 < rhs_v.i1;
      } else {
        if (lhs_v.isPair()) {
          CHECK(rhs_v.isPair());
          const auto lhs = pair_to_double({lhs_v.i1, lhs_v.i2}, entry_ti);
          const auto rhs = pair_to_double({rhs_v.i1, rhs_v.i2}, entry_ti);
          if (lhs == rhs) {
            continue;
          }
          return use_desc_cmp ? lhs > rhs : lhs < rhs;
        } else {
          CHECK(lhs_v.isStr() && rhs_v.isStr());
          const auto lhs = lhs_v.strVal();
          const auto rhs = rhs_v.strVal();
          if (lhs == rhs) {
            continue;
          }
          return use_desc_cmp ? lhs > rhs : lhs < rhs;
        }
      }
    }
    return false;
  };
}

#undef UNLIKELY
#undef LIKELY

void ResultSet::topPermutation(const size_t n, const std::function<bool(const uint32_t, const uint32_t)> compare) {
  std::make_heap(permutation_.begin(), permutation_.end(), compare);
  decltype(permutation_) permutation_top;
  permutation_top.reserve(n);
  for (size_t i = 0; i < n && !permutation_.empty(); ++i) {
    permutation_top.push_back(permutation_.front());
    std::pop_heap(permutation_.begin(), permutation_.end(), compare);
    permutation_.pop_back();
  }
  permutation_.swap(permutation_top);
}

void ResultSet::sortPermutation(const std::function<bool(const uint32_t, const uint32_t)> compare) {
  std::sort(permutation_.begin(), permutation_.end(), compare);
}

bool ResultSet::isEmptyInitializer() const {
  return targets_.empty();
}

void ResultSet::sortOnGpu(const std::list<Analyzer::OrderEntry>& order_entries) const {
  auto data_mgr = &query_mem_desc_.executor_->catalog_->get_dataMgr();
  const int device_id{0};
  std::vector<int64_t*> group_by_buffers(query_mem_desc_.executor_->blockSize());
  group_by_buffers[0] = reinterpret_cast<int64_t*>(storage_->buff_);
  auto gpu_query_mem = create_dev_group_by_buffers(data_mgr,
                                                   group_by_buffers,
                                                   {},
                                                   query_mem_desc_,
                                                   query_mem_desc_.executor_->blockSize(),
                                                   query_mem_desc_.executor_->gridSize(),
                                                   device_id,
                                                   true,
                                                   true,
                                                   nullptr);
  size_t max_entry_size{0};
  for (const auto& wid : query_mem_desc_.agg_col_widths) {
    max_entry_size = std::max(max_entry_size, size_t(wid.compact));
  }
  ScopedScratchBuffer scratch_buff(query_mem_desc_.entry_count * max_entry_size, data_mgr, device_id);
  auto tmp_buff = reinterpret_cast<int64_t*>(scratch_buff.getPtr());
  CHECK_EQ(size_t(1), order_entries.size());
  const auto idx_buff =
      gpu_query_mem.group_by_buffers.second - align_to_int64(query_mem_desc_.entry_count * sizeof(int32_t));
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc_.getColOffInBytes(0, target_idx);
    const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
    sort_groups_gpu(reinterpret_cast<int64_t*>(val_buff),
                    reinterpret_cast<int32_t*>(idx_buff),
                    query_mem_desc_.entry_count,
                    order_entry.is_desc,
                    chosen_bytes);
    if (!query_mem_desc_.keyless_hash) {
      apply_permutation_gpu(reinterpret_cast<int64_t*>(gpu_query_mem.group_by_buffers.second),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc_.entry_count,
                            tmp_buff,
                            sizeof(int64_t));
    }
    for (size_t target_idx = 0; target_idx < query_mem_desc_.agg_col_widths.size(); ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
      const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc_.getColOffInBytes(0, target_idx);
      apply_permutation_gpu(reinterpret_cast<int64_t*>(val_buff),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc_.entry_count,
                            tmp_buff,
                            chosen_bytes);
    }
  }
  copy_group_by_buffers_from_gpu(data_mgr,
                                 group_by_buffers,
                                 query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU),
                                 gpu_query_mem.group_by_buffers.second,
                                 query_mem_desc_,
                                 query_mem_desc_.executor_->blockSize(),
                                 query_mem_desc_.executor_->gridSize(),
                                 device_id,
                                 false);
}

void ResultSet::sortOnCpu(const std::list<Analyzer::OrderEntry>& order_entries) const {
  CHECK(!query_mem_desc_.keyless_hash);
  CHECK(storage_->buff_);
  std::vector<int64_t> tmp_buff(query_mem_desc_.entry_count);
  std::vector<int32_t> idx_buff(query_mem_desc_.entry_count);
  CHECK_EQ(size_t(1), order_entries.size());
  auto i64_buff = reinterpret_cast<int64_t*>(storage_->buff_);
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto sortkey_val_buff = i64_buff + order_entry.tle_no * query_mem_desc_.entry_count;
    const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
    sort_groups_cpu(sortkey_val_buff, &idx_buff[0], query_mem_desc_.entry_count, order_entry.is_desc, chosen_bytes);
    apply_permutation_cpu(i64_buff, &idx_buff[0], query_mem_desc_.entry_count, &tmp_buff[0], sizeof(int64_t));
    for (size_t target_idx = 0; target_idx < query_mem_desc_.agg_col_widths.size(); ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
      const auto satellite_val_buff = i64_buff + (target_idx + 1) * query_mem_desc_.entry_count;
      apply_permutation_cpu(satellite_val_buff, &idx_buff[0], query_mem_desc_.entry_count, &tmp_buff[0], chosen_bytes);
    }
  }
  CHECK(false);
}

namespace {

int64_t lazy_decode(const SQLTypeInfo& type_info, const int8_t* byte_stream, const int64_t pos) {
  const auto enc_type = type_info.get_compression();
  if (type_info.is_fp()) {
    if (type_info.get_type() == kFLOAT) {
      float fval = fixed_width_float_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<int32_t*>(&fval);
    } else {
      double fval = fixed_width_double_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<int64_t*>(&fval);
    }
  }
  CHECK(type_info.is_integer() || type_info.is_decimal() || type_info.is_time() || type_info.is_boolean() ||
        (type_info.is_string() && enc_type == kENCODING_DICT));
  size_t type_bitwidth = get_bit_width(type_info);
  if (type_info.get_compression() == kENCODING_FIXED) {
    type_bitwidth = type_info.get_comp_param();
  }
  CHECK_EQ(size_t(0), type_bitwidth % 8);
  return fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
}

}  // namespace

void ResultSet::fetchLazy(const std::vector<ssize_t> lazy_col_local_ids,
                          const std::vector<std::vector<const int8_t*>>& col_buffers) const {
  const auto target_count = targets_.size();
  CHECK_EQ(target_count, lazy_col_local_ids.size());
  for (size_t i = 0; i < target_count; ++i) {
    const auto& target_info = targets_[i];
    CHECK_EQ(size_t(1), col_buffers.size());
    auto& frag_col_buffers = col_buffers.front();
    const auto local_col_id = lazy_col_local_ids[i];
    if (local_col_id >= 0) {
      lazy_decode(target_info.sql_type, frag_col_buffers[local_col_id], -1);
      CHECK(false);
    }
  }
  CHECK(false);
}
