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

#include "ColumnarResults.h"
#include "ResultRows.h"

#include "../Shared/thread_count.h"

#include <atomic>
#include <future>
#include <numeric>

namespace {

int64_t fixed_encoding_nullable_val(const int64_t val, const SQLTypeInfo& type_info) {
  if (type_info.get_compression() != kENCODING_NONE) {
    CHECK(type_info.get_compression() == kENCODING_FIXED ||
          type_info.get_compression() == kENCODING_DICT);
    auto logical_ti = get_logical_type_info(type_info);
    if (val == inline_int_null_val(logical_ti)) {
      return inline_fixed_encoding_null_val(type_info);
    }
  }
  return val;
}

}  // namespace

ColumnarResults::ColumnarResults(
    const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const ResultSet& rows,
    const size_t num_columns,
    const std::vector<SQLTypeInfo>& target_types)
    : column_buffers_(num_columns)
    , num_rows_(use_parallel_algorithms(rows) ? rows.entryCount() : rows.rowCount())
    , target_types_(target_types) {
  column_buffers_.resize(num_columns);
  for (size_t i = 0; i < num_columns; ++i) {
    const bool is_varlen = target_types[i].is_array() ||
                           (target_types[i].is_string() &&
                            target_types[i].get_compression() == kENCODING_NONE) ||
                           target_types[i].is_geometry();
    if (is_varlen) {
      throw ColumnarConversionNotSupported();
    }
    column_buffers_[i] = reinterpret_cast<const int8_t*>(
        checked_malloc(num_rows_ * target_types[i].get_size()));
    row_set_mem_owner->addColBuffer(column_buffers_[i]);
  }
  std::atomic<size_t> row_idx{0};
  const auto do_work = [num_columns, &target_types, this](
                           const std::vector<TargetValue>& crt_row,
                           const size_t row_idx) {
    for (size_t i = 0; i < num_columns; ++i) {
      const auto col_val = crt_row[i];
      const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
      CHECK(scalar_col_val);
      auto i64_p = boost::get<int64_t>(scalar_col_val);
      const auto& type_info = target_types[i];
      if (i64_p) {
        const auto val = fixed_encoding_nullable_val(*i64_p, type_info);
        switch (target_types[i].get_size()) {
          case 1:
            ((int8_t*)column_buffers_[i])[row_idx] = static_cast<int8_t>(val);
            break;
          case 2:
            ((int16_t*)column_buffers_[i])[row_idx] = static_cast<int16_t>(val);
            break;
          case 4:
            ((int32_t*)column_buffers_[i])[row_idx] = static_cast<int32_t>(val);
            break;
          case 8:
            ((int64_t*)column_buffers_[i])[row_idx] = val;
            break;
          default:
            CHECK(false);
        }
      } else {
        CHECK(target_types[i].is_fp());
        switch (target_types[i].get_type()) {
          case kFLOAT: {
            auto float_p = boost::get<float>(scalar_col_val);
            ((float*)column_buffers_[i])[row_idx] = static_cast<float>(*float_p);
            break;
          }
          case kDOUBLE: {
            auto double_p = boost::get<double>(scalar_col_val);
            ((double*)column_buffers_[i])[row_idx] = static_cast<double>(*double_p);
            break;
          }
          default:
            CHECK(false);
        }
      }
    }
  };
  if (use_parallel_algorithms(rows)) {
    const size_t worker_count = cpu_threads();
    std::vector<std::future<void>> conversion_threads;
    const auto entry_count = rows.entryCount();
    for (size_t i = 0,
                start_entry = 0,
                stride = (entry_count + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entry_count;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      conversion_threads.push_back(
          std::async(std::launch::async,
                     [&rows, &do_work, &row_idx](const size_t start, const size_t end) {
                       for (size_t i = start; i < end; ++i) {
                         const auto crt_row = rows.getRowAtNoTranslations(i);
                         if (!crt_row.empty()) {
                           do_work(crt_row, row_idx.fetch_add(1));
                         }
                       }
                     },
                     start_entry,
                     end_entry));
    }
    for (auto& child : conversion_threads) {
      child.wait();
    }
    for (auto& child : conversion_threads) {
      child.get();
    }
    num_rows_ = row_idx;
    rows.setCachedRowCount(num_rows_);
    return;
  }
  while (true) {
    const auto crt_row = rows.getNextRow(false, false);
    if (crt_row.empty()) {
      break;
    }
    do_work(crt_row, row_idx);
    ++row_idx;
  }
  rows.moveToBegin();
}

ColumnarResults::ColumnarResults(
    const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const int8_t* one_col_buffer,
    const size_t num_rows,
    const SQLTypeInfo& target_type)
    : column_buffers_(1), num_rows_(num_rows), target_types_{target_type} {
  const bool is_varlen =
      target_type.is_array() ||
      (target_type.is_string() && target_type.get_compression() == kENCODING_NONE) ||
      target_type.is_geometry();
  if (is_varlen) {
    throw ColumnarConversionNotSupported();
  }
  const auto buf_size = num_rows * target_type.get_size();
  column_buffers_[0] = reinterpret_cast<const int8_t*>(checked_malloc(buf_size));
  memcpy(((void*)column_buffers_[0]), one_col_buffer, buf_size);
  row_set_mem_owner->addColBuffer(column_buffers_[0]);
}

std::unique_ptr<ColumnarResults> ColumnarResults::mergeResults(
    const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const std::vector<std::unique_ptr<ColumnarResults>>& sub_results) {
  if (sub_results.empty()) {
    return nullptr;
  }
  const auto total_row_count = std::accumulate(
      sub_results.begin(),
      sub_results.end(),
      size_t(0),
      [](const size_t init, const std::unique_ptr<ColumnarResults>& result) {
        return init + result->size();
      });
  std::unique_ptr<ColumnarResults> merged_results(
      new ColumnarResults(total_row_count, sub_results[0]->target_types_));
  const auto col_count = sub_results[0]->column_buffers_.size();
  const auto nonempty_it = std::find_if(
      sub_results.begin(),
      sub_results.end(),
      [](const std::unique_ptr<ColumnarResults>& needle) { return needle->size(); });
  if (nonempty_it == sub_results.end()) {
    return nullptr;
  }
  for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
    const auto byte_width = (*nonempty_it)->getColumnType(col_idx).get_size();
    auto write_ptr =
        reinterpret_cast<int8_t*>(checked_malloc(byte_width * total_row_count));
    merged_results->column_buffers_.push_back(write_ptr);
    row_set_mem_owner->addColBuffer(write_ptr);
    for (auto& rs : sub_results) {
      CHECK_EQ(col_count, rs->column_buffers_.size());
      if (!rs->size()) {
        continue;
      }
      CHECK_EQ(byte_width, rs->getColumnType(col_idx).get_size());
      memcpy(write_ptr, rs->column_buffers_[col_idx], rs->size() * byte_width);
      write_ptr += rs->size() * byte_width;
    }
  }
  return merged_results;
}
