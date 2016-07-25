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

ResultSetStorage::ResultSetStorage(const std::vector<TargetInfo>& targets,
                                   const ExecutorDeviceType device_type,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   int8_t* buff)
    : targets_(targets), query_mem_desc_(query_mem_desc), buff_(buff) {
  for (const auto& target_info : targets_) {
    target_init_vals_.push_back(
        target_info.is_agg ? get_agg_initial_val(target_info.agg_kind, target_info.sql_type, false, 8) : 0);
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
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
    : targets_(targets),
      device_type_(device_type),
      query_mem_desc_(query_mem_desc),
      crt_row_buff_idx_(0),
      drop_first_(0),
      keep_first_(0),
      row_set_mem_owner_(row_set_mem_owner) {
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
  CHECK(buff);
  storage_.reset(new ResultSetStorage(targets_, device_type_, query_mem_desc_, buff));
  return storage_.get();
}

void ResultSet::append(ResultSet& that) {
  CHECK(that.appended_storage_.empty());
  appended_storage_.push_back(std::move(that.storage_));
  CHECK(false);
}

void ResultSet::sort(const std::list<Analyzer::OrderEntry>& order_entries,
                     const bool /* remove_duplicates */,
                     const int64_t /* top_n */) const {
  if (isEmptyInitializer()) {
    return;
  }
  if (query_mem_desc_.sortOnGpu()) {
    try {
      sortOnGpu(order_entries);
    } catch (const OutOfMemory&) {
      sortOnCpu(order_entries);
    }
    return;
  }
  CHECK(false);
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
  ScopedScratchBuffer scratch_buff(query_mem_desc_.entry_count * sizeof(int64_t), data_mgr, device_id);
  auto tmp_buff = reinterpret_cast<int64_t*>(scratch_buff.getPtr());
  CHECK_EQ(size_t(1), order_entries.size());
  const auto idx_buff = gpu_query_mem.group_by_buffers.second - query_mem_desc_.entry_count * sizeof(int64_t);
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc_.getColOffInBytes(0, target_idx);
    const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
    sort_groups_gpu(reinterpret_cast<int64_t*>(val_buff),
                    reinterpret_cast<int64_t*>(idx_buff),
                    query_mem_desc_.entry_count,
                    order_entry.is_desc,
                    chosen_bytes);
    if (!query_mem_desc_.keyless_hash) {
      apply_permutation_gpu(reinterpret_cast<int64_t*>(gpu_query_mem.group_by_buffers.second),
                            reinterpret_cast<int64_t*>(idx_buff),
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
                            reinterpret_cast<int64_t*>(idx_buff),
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
  std::vector<int64_t> idx_buff(query_mem_desc_.entry_count);
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
