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
                             const std::vector<std::vector<const int8_t*>>& iter_buffers,
                             const ssize_t frag_id,
                             const ExecutorDeviceType device_type)
    : targets_([&targets]() {
        std::vector<TargetInfo> info;
        for (size_t col_idx = 0; col_idx < targets.size(); ++col_idx) {
          info.push_back(target_info(targets[col_idx]));
        }
        return info;
      }()),
      query_mem_desc_([&query_mem_desc, &targets]() {
        CHECK_EQ(size_t(2), query_mem_desc.agg_col_widths.size());
        auto desc = query_mem_desc;
        desc.output_columnar = true;
        desc.group_col_widths.clear();
        desc.agg_col_widths.clear();
        auto width = query_mem_desc.agg_col_widths[0];
        for (size_t i = 0; i < targets.size(); ++i) {
          desc.agg_col_widths.push_back(width);
        }
        return desc;
      }()),
      row_set_mem_owner_(row_set_mem_owner),
      device_type_(device_type),
      just_explain_(false) {
  if (group_by_buffer) {
    buffer_frags_.push_back(transformGroupByBuffer(group_by_buffer, groups_buffer_entry_count, query_mem_desc));
    if (!iter_buffers.empty()) {
      fetchLazy(iter_buffers, frag_id);
    }
  }
}

IteratorTable::IteratorTable(const std::vector<TargetInfo>& targets,
                             const QueryMemoryDescriptor& query_mem_desc,
                             const ExecutorDeviceType device_type,
                             const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
    : targets_(targets),
      query_mem_desc_(query_mem_desc),
      row_set_mem_owner_(row_set_mem_owner),
      device_type_(device_type),
      just_explain_(false) {}

IteratorTable::IteratorTable()
    : targets_{},
      query_mem_desc_{},
      row_set_mem_owner_(nullptr),
      device_type_(ExecutorDeviceType::CPU),
      just_explain_(false) {}

size_t IteratorTable::rowCount() const {
  size_t row_count{0};
  for (const auto& frag : buffer_frags_) {
    if (frag.data) {
      row_count += frag.row_count;
    }
  }
  return row_count;
}

BufferFragment IteratorTable::transformGroupByBuffer(const int64_t* group_by_buffer,
                                                     const size_t groups_buffer_entry_count,
                                                     const QueryMemoryDescriptor& query_mem_desc) {
  CHECK(group_by_buffer);
  CHECK_LT(size_t(0), groups_buffer_entry_count);
  CHECK_EQ(size_t(2), query_mem_desc.agg_col_widths.size());
  const auto col_count = colCount();
  std::vector<int64_t> scratch_buffer;
  scratch_buffer.reserve(groups_buffer_entry_count * col_count);

  for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
    const auto src_col_idx = (col_idx == col_count - 1 ? 1 : 0);
    for (size_t bin_base_off = query_mem_desc.getColOffInBytes(0, src_col_idx), crt_row_buff_idx = 0;
         crt_row_buff_idx < static_cast<size_t>(groups_buffer_entry_count);
         ++crt_row_buff_idx, bin_base_off += query_mem_desc.getColOffInBytesInNextBin(src_col_idx)) {
      auto key_off = query_mem_desc.getKeyOffInBytes(crt_row_buff_idx) / sizeof(int64_t);
      if (group_by_buffer[key_off] == EMPTY_KEY_64) {
        continue;
      }
      auto col_ptr = reinterpret_cast<const int8_t*>(group_by_buffer) + bin_base_off;
      auto chosen_bytes = query_mem_desc.agg_col_widths[src_col_idx].compact;
      CHECK_EQ(sizeof(int64_t), static_cast<size_t>(chosen_bytes));
      scratch_buffer.push_back(get_component(col_ptr, chosen_bytes));
    }
  }
  CHECK_EQ(size_t(0), scratch_buffer.size() % col_count);

  if (scratch_buffer.empty()) {
    return {nullptr, size_t(0)};
  }
  CHECK(row_set_mem_owner_);
  OOM_TRACE_PUSH(+": scratch_buffer " + std::to_string(scratch_buffer.size() * sizeof(int64_t)));
  auto table_frag = reinterpret_cast<int64_t*>(checked_malloc(scratch_buffer.size() * sizeof(int64_t)));
  memcpy(table_frag, &scratch_buffer[0], scratch_buffer.size() * sizeof(int64_t));
  // TODO(miyu): free old buffer held by row_set_mem_owner_ if proved safe.
  CHECK(row_set_mem_owner_);
  row_set_mem_owner_->addGroupByBuffer(table_frag);
  return {table_frag, (scratch_buffer.size() / col_count)};
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

void IteratorTable::fetchLazy(const std::vector<std::vector<const int8_t*>>& iter_buffers, const ssize_t frag_id) {
  if (iter_buffers.empty()) {
    return;
  }
  CHECK_EQ(size_t(1), iter_buffers.size());
  CHECK_EQ(size_t(1), buffer_frags_.size());
  auto& buffer_frag = buffer_frags_[0];
  if (!buffer_frag.row_count) {
    CHECK(!buffer_frag.data);
    return;
  }

  CHECK(buffer_frag.data);
  CHECK_LT(size_t(0), buffer_frag.row_count);
  for (size_t col_idx = 0, col_cnt = colCount() - 1, col_base_off = 0; col_idx < col_cnt;
       ++col_idx, col_base_off += buffer_frag.row_count) {
    auto chosen_bytes = query_mem_desc_.agg_col_widths[col_idx].compact;
    CHECK_EQ(sizeof(int64_t), static_cast<size_t>(chosen_bytes));
    for (size_t row_idx = 0; row_idx < buffer_frag.row_count; ++row_idx) {
      auto col_ptr = reinterpret_cast<int8_t*>(&buffer_frag.data[col_base_off + row_idx]);
      auto frag_iter_buffer = iter_buffers.front();
      CHECK_LT(col_idx, frag_iter_buffer.size());
      auto val = get_component(col_ptr, chosen_bytes);
      set_component(col_ptr, chosen_bytes, lazy_decode(targets_[col_idx].sql_type, frag_iter_buffer[col_idx], val));
    }
  }
}

void IteratorTable::fuse(const IteratorTable& that) {
  CHECK(!query_mem_desc_.keyless_hash && !that.query_mem_desc_.keyless_hash);
  CHECK_EQ(size_t(0), query_mem_desc_.group_col_widths.size());
  CHECK_EQ(size_t(0), that.query_mem_desc_.group_col_widths.size());
  CHECK_EQ(query_mem_desc_.agg_col_widths.size(), that.query_mem_desc_.agg_col_widths.size());
  CHECK_EQ(query_mem_desc_.output_columnar, that.query_mem_desc_.output_columnar);
  CHECK_EQ(size_t(1), buffer_frags_.size());
  CHECK_EQ(size_t(1), that.buffer_frags_.size());
  auto& this_buffer = buffer_frags_[0].data;
  auto& this_entry_count = buffer_frags_[0].row_count;
  const auto that_buffer = that.buffer_frags_[0].data;
  const auto that_entry_count = that.buffer_frags_[0].row_count;
  if (that.definitelyHasNoRows()) {
    return;
  }
  if (definitelyHasNoRows()) {
    this_buffer = that_buffer;
    return;
  }

  const auto col_count = query_mem_desc_.agg_col_widths.size();
  OOM_TRACE_PUSH(+": fused_buffer " +
                 std::to_string(((this_entry_count + that_entry_count) * col_count) * sizeof(int64_t)));
  auto fused_buffer =
      reinterpret_cast<int64_t*>(checked_malloc(((this_entry_count + that_entry_count) * col_count) * sizeof(int64_t)));
  auto write_ptr = fused_buffer;
  auto this_read_ptr = this_buffer;
  auto that_read_ptr = that_buffer;
  for (size_t col_idx = 0; col_idx < col_count;
       ++col_idx, this_read_ptr += this_entry_count, that_read_ptr += that_entry_count) {
    memcpy(write_ptr, this_read_ptr, this_entry_count * sizeof(int64_t));
    write_ptr += this_entry_count;
    memcpy(write_ptr, that_read_ptr, that_entry_count * sizeof(int64_t));
    write_ptr += that_entry_count;
  }
  this_entry_count += that_entry_count;
  // TODO(miyu): free old buffer held by row_set_mem_owner_ if proved safe.
  CHECK(row_set_mem_owner_);
  row_set_mem_owner_->addGroupByBuffer(fused_buffer);
  this_buffer = fused_buffer;
}

IterTabPtr QueryExecutionContext::groupBufferToTab(const size_t buf_idx,
                                                   const ssize_t frag_idx,
                                                   const std::vector<Analyzer::Expr*>& targets) const {
  CHECK_EQ(size_t(2), query_mem_desc_.agg_col_widths.size());
  const size_t group_by_col_count{query_mem_desc_.group_col_widths.size()};
  CHECK(!output_columnar_ || group_by_col_count == 1);
  auto impl = [this, &targets](
      const size_t groups_buffer_entry_count, int64_t* group_by_buffer, const ssize_t frag_idx) {
    IterTabPtr table = boost::make_unique<IteratorTable>(query_mem_desc_,
                                                         targets,
                                                         row_set_mem_owner_,
                                                         group_by_buffer,
                                                         groups_buffer_entry_count,
                                                         iter_buffers_,
                                                         frag_idx,
                                                         device_type_);
    CHECK(table);
    return table;
  };
  IterTabPtr table{nullptr};
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
    CHECK(!sort_on_gpu_);
    table = impl(query_mem_desc_.entry_count_small, small_group_by_buffers_[buf_idx], frag_idx);
    CHECK(table);
  }
  CHECK_LT(buf_idx, group_by_buffers_.size());
  auto other_table = impl(query_mem_desc_.entry_count, group_by_buffers_[buf_idx], frag_idx);
  CHECK(other_table);
  if (table) {
    table->fuse(*other_table);
    return table;
  } else {
    return other_table;
  }
}

IterTabPtr QueryExecutionContext::getIterTab(const std::vector<Analyzer::Expr*>& targets,
                                             const ssize_t frag_idx) const {
  CHECK_EQ(num_buffers_, group_by_buffers_.size());
  if (device_type_ == ExecutorDeviceType::CPU) {
    CHECK_EQ(size_t(1), num_buffers_);
    return groupBufferToTab(0, frag_idx, targets);
  }

  CHECK(device_type_ == ExecutorDeviceType::GPU);
  IterTabPtr table{nullptr};
  size_t step{query_mem_desc_.threadsShareMemory() ? executor_->blockSize() : 1};
  for (size_t i = 0; i < group_by_buffers_.size(); i += step) {
    if (!table) {
      table = groupBufferToTab(i, frag_idx, targets);
      continue;
    }
    table->fuse(*groupBufferToTab(i, frag_idx, targets));
  }

  return table;
}

ResultPtr QueryExecutionContext::getResult(const RelAlgExecutionUnit& ra_exe_unit,
                                           const std::vector<size_t>& outer_tab_frag_ids,
                                           const bool was_auto_device) const {
  if (contains_iter_expr(ra_exe_unit.target_exprs)) {
    CHECK_EQ(size_t(1), outer_tab_frag_ids.size());
    return getIterTab(ra_exe_unit.orig_target_exprs, outer_tab_frag_ids[0]);
  } else {
    return getRowSet(ra_exe_unit, query_mem_desc_, was_auto_device);
  }
}
