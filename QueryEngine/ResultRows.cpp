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

#include "ResultRows.h"
#include "Execute.h"
#include "GpuMemUtils.h"
#include "GroupByAndAggregate.h"
#include "InPlaceSort.h"
#include "ResultSet.h"
#include "ThrustAllocator.h"

void ResultRows::inplaceSortGpuImpl(const std::list<Analyzer::OrderEntry>& order_entries,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const GpuQueryMemory& group_by_buffers_struct,
                                    Data_Namespace::DataMgr* data_mgr,
                                    const int device_id) {
  // TODO(adb): fix this
  auto group_by_buffers = group_by_buffers_struct.group_by_buffers;

  ThrustAllocator alloc(data_mgr, device_id);
  CHECK_EQ(size_t(1), order_entries.size());
  const auto idx_buff = group_by_buffers.second -
                        align_to_int64(query_mem_desc.getEntryCount() * sizeof(int32_t));
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto val_buff =
        group_by_buffers.second + query_mem_desc.getColOffInBytes(target_idx);
    const auto chosen_bytes = query_mem_desc.getColumnWidth(target_idx).compact;
    sort_groups_gpu(reinterpret_cast<int64_t*>(val_buff),
                    reinterpret_cast<int32_t*>(idx_buff),
                    query_mem_desc.getEntryCount(),
                    order_entry.is_desc,
                    chosen_bytes,
                    alloc);
    if (!query_mem_desc.hasKeylessHash()) {
      apply_permutation_gpu(reinterpret_cast<int64_t*>(group_by_buffers.second),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc.getEntryCount(),
                            sizeof(int64_t),
                            alloc);
    }
    for (size_t target_idx = 0; target_idx < query_mem_desc.getColCount(); ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc.getColumnWidth(target_idx).compact;
      const auto val_buff =
          group_by_buffers.second + query_mem_desc.getColOffInBytes(target_idx);
      apply_permutation_gpu(reinterpret_cast<int64_t*>(val_buff),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc.getEntryCount(),
                            chosen_bytes,
                            alloc);
    }
  }
}
