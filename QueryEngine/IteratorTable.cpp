/**
 * @file    IteratorTable.cpp
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Basic constructors and methods of the iterator table interface.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#include "IteratorTable.h"

#include "AggregateUtils.h"
#include "Execute.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"

IteratorTable::IteratorTable(const QueryMemoryDescriptor& query_mem_desc,
                             const std::vector<Analyzer::Expr*>& targets,
                             const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                             int64_t* group_by_buffer,
                             const size_t groups_buffer_entry_count,
                             const bool output_columnar,
                             const std::vector<std::vector<const int8_t*>>& col_buffers,
                             const ExecutorDeviceType device_type,
                             const int device_id)
    : targets_([&targets]() {
        std::vector<TargetInfo> info;
        for (size_t col_idx = 0; col_idx < targets.size(); ++col_idx) {
          info.push_back(target_info(targets[col_idx]));
        }
        return info;
      }()),
      query_mem_desc_(query_mem_desc),
      row_set_mem_owner_(row_set_mem_owner),
      entry_count_per_frag_(groups_buffer_entry_count),
      device_type_(device_type),
      just_explain_(false) {
  bool has_lazy_columns = false;
  std::unordered_map<size_t, ssize_t> lazy_col_local_ids;
  for (size_t col_idx = 0; col_idx < targets.size(); ++col_idx) {
    if (query_mem_desc.executor_->plan_state_->isLazyFetchColumn(targets[col_idx])) {
      has_lazy_columns = true;
      lazy_col_local_ids.insert(std::make_pair(col_idx,
                                               query_mem_desc.executor_->getLocalColumnId(
                                                   static_cast<const Analyzer::ColumnVar*>(targets[col_idx]), false)));
    }
  }
  if (group_by_buffer) {
    group_by_buffer_frags_.push_back(group_by_buffer);
    if (has_lazy_columns) {
      fetchLazy(lazy_col_local_ids, col_buffers, 0);
    }
  }
}

size_t IteratorTable::rowCount() const {
  size_t row_count{0};
  for (auto frag : group_by_buffer_frags_) {
    if (frag) {
      for (size_t bin_base_off = 0, crt_row_buff_idx = 0; crt_row_buff_idx < static_cast<size_t>(entry_count_per_frag_);
           ++crt_row_buff_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
        auto key_off = query_mem_desc_.getKeyOffInBytes(crt_row_buff_idx) / sizeof(int64_t);
        if (frag[key_off] == EMPTY_KEY_64) {
          continue;
        }
        ++row_count;
      }
    }
  }
  return row_count;
}

namespace {

int64_t lazy_decode(const SQLTypeInfo& type_info, const int8_t* byte_stream, const int64_t pos) {
  CHECK(kENCODING_DICT != type_info.get_compression() && kENCODING_FIXED != type_info.get_compression());
  CHECK(type_info.is_integer());
  size_t type_bitwidth = get_bit_width(type_info);
  CHECK_EQ(size_t(0), type_bitwidth % 8);
  return fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
}

}  // namespace

void IteratorTable::fetchLazy(const std::unordered_map<size_t, ssize_t>& lazy_col_local_ids,
                              const std::vector<std::vector<const int8_t*>>& col_buffers,
                              const ssize_t frag_id) {
  const auto target_count = targets_.size();
  CHECK_LT(frag_id, group_by_buffer_frags_.size());
  CHECK_EQ(col_buffers.size(), group_by_buffer_frags_.size());
  CHECK_EQ(target_count, lazy_col_local_ids.size());

  auto group_by_buffer = group_by_buffer_frags_[frag_id];
  CHECK(group_by_buffer);
  for (size_t bin_base_off = 0, crt_row_buff_idx = 0; crt_row_buff_idx < static_cast<size_t>(entry_count_per_frag_);
       ++crt_row_buff_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
    auto key_off = query_mem_desc_.getKeyOffInBytes(crt_row_buff_idx) / sizeof(int64_t);
    if (group_by_buffer[key_off] == EMPTY_KEY_64) {
      continue;
    }

    auto col_ptr = reinterpret_cast<int8_t*>(group_by_buffer) + bin_base_off;
    for (size_t col_idx = 0; col_idx < colCount();
         col_ptr += query_mem_desc_.getNextColOffInBytes(col_ptr, crt_row_buff_idx, col_idx++)) {
      auto chosen_bytes = query_mem_desc_.agg_col_widths[col_idx].compact;
      CHECK_EQ(sizeof(int64_t), chosen_bytes);
      auto val = get_component(col_ptr, chosen_bytes);
      if (lazy_col_local_ids.count(col_idx)) {
        auto it = lazy_col_local_ids.find(col_idx);
        CHECK(it != lazy_col_local_ids.end());
        auto col_id = it->second;
        auto& frag_col_buffers = col_buffers[frag_id];
        set_component(col_ptr, chosen_bytes, lazy_decode(targets_[col_idx].sql_type, frag_col_buffers[col_id], val));
      }
    }
  }
}
