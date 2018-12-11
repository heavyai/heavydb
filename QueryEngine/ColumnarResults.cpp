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
    , num_rows_(use_parallel_algorithms(rows) || rows.isFastColumnarConversionPossible()
                    ? rows.entryCount()
                    : rows.rowCount())
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
  const auto do_work = [num_columns, this](const std::vector<TargetValue>& crt_row,
                                           const size_t row_idx) {
    for (size_t i = 0; i < num_columns; ++i) {
      writeBackCell(crt_row[i], row_idx, i);
    }
  };

  if (rows.isFastColumnarConversionPossible() && rows.entryCount() > 0) {
    materializeAllColumns(rows, num_columns);
  } else {
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

/*
 * This function processes and decodes its input TargetValue
 * and write it into its corresponding column buffer's cell (with corresponding
 * row and column indices)
 *
 * NOTE: this is not supposed to be processing varlen types, and they should be
 * handled differently outside this function.
 */
inline void ColumnarResults::writeBackCell(const TargetValue& col_val,
                                           const size_t row_idx,
                                           const size_t column_idx) {
  const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
  CHECK(scalar_col_val);
  auto i64_p = boost::get<int64_t>(scalar_col_val);
  const auto& type_info = target_types_[column_idx];
  if (i64_p) {
    const auto val = fixed_encoding_nullable_val(*i64_p, type_info);
    switch (target_types_[column_idx].get_size()) {
      case 1:
        ((int8_t*)column_buffers_[column_idx])[row_idx] = static_cast<int8_t>(val);
        break;
      case 2:
        ((int16_t*)column_buffers_[column_idx])[row_idx] = static_cast<int16_t>(val);
        break;
      case 4:
        ((int32_t*)column_buffers_[column_idx])[row_idx] = static_cast<int32_t>(val);
        break;
      case 8:
        ((int64_t*)column_buffers_[column_idx])[row_idx] = val;
        break;
      default:
        CHECK(false);
    }
  } else {
    CHECK(target_types_[column_idx].is_fp());
    switch (target_types_[column_idx].get_type()) {
      case kFLOAT: {
        auto float_p = boost::get<float>(scalar_col_val);
        ((float*)column_buffers_[column_idx])[row_idx] = static_cast<float>(*float_p);
        break;
      }
      case kDOUBLE: {
        auto double_p = boost::get<double>(scalar_col_val);
        ((double*)column_buffers_[column_idx])[row_idx] = static_cast<double>(*double_p);
        break;
      }
      default:
        CHECK(false);
    }
  }
}

/*
 * This function materializes all columns from the main storage and all appended storages
 * and form a single continguous column for each output column. Depending on whether the
 * column is lazily fetched or not, it will treat them differently.
 *
 * NOTE: this function should
 * only be used when the result set is columnar and completely compacted (e.g., in
 * columnar projections).
 */
void ColumnarResults::materializeAllColumns(const ResultSet& rows,
                                            const size_t num_columns) {
  CHECK(rows.isFastColumnarConversionPossible());
  const auto& lazy_fetch_info = rows.getLazyFetchInfo();

  // We can directly copy each non-lazy column's content
  copyAllNonLazyColumns(lazy_fetch_info, rows, num_columns);

  // Only lazy columns are iterated through first and then materialized
  materializeAllLazyColumns(lazy_fetch_info, rows, num_columns);
}

/*
 * For all non-lazy columns, we can directly copy back the results of each column's
 * contents from different storages and put them into the corresponding output buffer.
 *
 * This function is parallelized through assigning each column to a CPU thread.
 */
void ColumnarResults::copyAllNonLazyColumns(
    const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
    const ResultSet& rows,
    const size_t num_columns) {
  CHECK(rows.isFastColumnarConversionPossible());
  const auto is_column_non_lazily_fetched = [&lazy_fetch_info](const size_t col_idx) {
    // Saman: make sure when this lazy_fetch_info is empty
    if (lazy_fetch_info.empty()) {
      return true;
    } else {
      return !lazy_fetch_info[col_idx].is_lazily_fetched;
    }
  };

  // parallelized by assigning each column to a thread
  std::vector<std::future<void>> direct_copy_threads;
  for (size_t col_idx = 0; col_idx < num_columns; col_idx++) {
    if (is_column_non_lazily_fetched(col_idx)) {
      direct_copy_threads.push_back(std::async(
          std::launch::async,
          [&rows, this](const size_t column_index) {
            const size_t column_size = num_rows_ * target_types_[column_index].get_size();
            rows.copyColumnIntoBuffer(column_index,
                                      const_cast<int8_t*>(column_buffers_[column_index]),
                                      column_size);
          },
          col_idx));
    }
  }

  for (auto& child : direct_copy_threads) {
    child.wait();
  }
  for (auto& child : direct_copy_threads) {
    child.get();
  }
}

/*
 * For all lazy fetched columns, we should iterate through the column's content and
 * properly materialize it.
 *
 * This function is parallelized through dividing total rows among all existing threads.
 * Since there's no invalid element in the result set (e.g., columnar projections), the
 * output buffer will have as many rows as there are in the result set, removing the need
 * for atomicly incrementing the output buffer position.
 */
void ColumnarResults::materializeAllLazyColumns(
    const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
    const ResultSet& rows,
    const size_t num_columns) {
  CHECK(rows.isFastColumnarConversionPossible());
  const auto do_work_just_lazy_columns =
      [num_columns, this](const std::vector<TargetValue>& crt_row,
                          const size_t row_idx,
                          const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info) {
        for (size_t i = 0; i < num_columns; ++i) {
          if (!lazy_fetch_info.empty() && lazy_fetch_info[i].is_lazily_fetched) {
            writeBackCell(crt_row[i], row_idx, i);
          }
        }
      };

  const auto contains_lazy_fetched_column =
      [](const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info) {
        for (auto& col_info : lazy_fetch_info) {
          if (col_info.is_lazily_fetched) {
            return true;
          }
        }
        return false;
      };

  // parallelized by assigning a chunk of rows to each thread)
  const bool skip_non_lazy_columns = rows.isPermutationBufferEmpty() ? true : false;
  if (contains_lazy_fetched_column(lazy_fetch_info)) {
    const size_t worker_count = use_parallel_algorithms(rows) ? cpu_threads() : 1;
    std::vector<std::future<void>> conversion_threads;
    const auto entry_count = rows.entryCount();
    for (size_t i = 0,
                start_entry = 0,
                stride = (entry_count + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entry_count;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      conversion_threads.push_back(std::async(
          std::launch::async,
          [&rows, &do_work_just_lazy_columns, &lazy_fetch_info](
              const size_t start, const size_t end, const bool skip_non_lazy_columns) {
            for (size_t i = start; i < end; ++i) {
              const auto crt_row = rows.getRowAtNoTranslations(i, skip_non_lazy_columns);
              do_work_just_lazy_columns(crt_row, i, lazy_fetch_info);
            }
          },
          start_entry,
          end_entry,
          skip_non_lazy_columns));
    }

    for (auto& child : conversion_threads) {
      child.wait();
    }
    for (auto& child : conversion_threads) {
      child.get();
    }
  }
}
