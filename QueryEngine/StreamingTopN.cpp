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

#include "StreamingTopN.h"
#include "Shared/checked_alloc.h"

namespace streaming_top_n {

size_t get_heap_size(const size_t row_size, const size_t n, const size_t thread_count) {
  const auto row_size_quad = row_size / sizeof(int64_t);
  return (1 + n + row_size_quad * n) * thread_count * sizeof(int64_t);
}

size_t get_rows_offset_of_heaps(const size_t n, const size_t thread_count) {
  return (1 + n) * thread_count * sizeof(int64_t);
}

std::vector<int8_t> get_rows_copy_from_heaps(const int64_t* heaps,
                                             const size_t heaps_size,
                                             const size_t n,
                                             const size_t thread_count) {
  const auto rows_offset = streaming_top_n::get_rows_offset_of_heaps(n, thread_count);
  const auto row_buff_size = heaps_size - rows_offset;
  std::vector<int8_t> rows_copy(row_buff_size);
  const auto rows_ptr = reinterpret_cast<const int8_t*>(heaps) + rows_offset;
  std::memcpy(&rows_copy[0], rows_ptr, row_buff_size);
  return rows_copy;
}

}  // namespace streaming_top_n

bool use_streaming_top_n(const RelAlgExecutionUnit& ra_exe_unit, const QueryMemoryDescriptor& query_mem_desc) {
  if (g_cluster) {
    return false;  // TODO(miyu)
  }

  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      return false;
    }
  }

  if (!query_mem_desc.canOutputColumnar() &&  // TODO(miyu): relax this limitation
      ra_exe_unit.sort_info.order_entries.size() == 1 &&
      ra_exe_unit.sort_info.limit && ra_exe_unit.sort_info.algorithm == SortAlgorithm::StreamingTopN) {
    const auto only_order_entry = ra_exe_unit.sort_info.order_entries.front();
    CHECK_GT(only_order_entry.tle_no, int(0));
    CHECK_LE(static_cast<size_t>(only_order_entry.tle_no), ra_exe_unit.target_exprs.size());
    const auto order_entry_expr = ra_exe_unit.target_exprs[only_order_entry.tle_no - 1];
    const auto n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
    if ((order_entry_expr->get_type_info().is_number() || order_entry_expr->get_type_info().is_time()) &&
        n <= 100000) {  // TODO(miyu): relax?
      return true;
    }
  }

  return false;
}
