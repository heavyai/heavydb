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
#include "ThrustAllocator.h"

void ResultRows::inplaceSortGpuImpl(const std::list<Analyzer::OrderEntry>& order_entries,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const GpuQueryMemory& gpu_query_mem,
                                    Data_Namespace::DataMgr* data_mgr,
                                    const int device_id) {
  ThrustAllocator alloc(data_mgr, device_id);
  CHECK_EQ(size_t(1), order_entries.size());
  const auto idx_buff =
      gpu_query_mem.group_by_buffers.second - align_to_int64(query_mem_desc.entry_count * sizeof(int32_t));
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc.getColOffInBytes(0, target_idx);
    const auto chosen_bytes = query_mem_desc.agg_col_widths[target_idx].compact;
    sort_groups_gpu(reinterpret_cast<int64_t*>(val_buff),
                    reinterpret_cast<int32_t*>(idx_buff),
                    query_mem_desc.entry_count,
                    order_entry.is_desc,
                    chosen_bytes,
                    alloc);
    if (!query_mem_desc.keyless_hash) {
      apply_permutation_gpu(reinterpret_cast<int64_t*>(gpu_query_mem.group_by_buffers.second),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc.entry_count,
                            sizeof(int64_t),
                            alloc);
    }
    for (size_t target_idx = 0; target_idx < query_mem_desc.agg_col_widths.size(); ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc.agg_col_widths[target_idx].compact;
      const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc.getColOffInBytes(0, target_idx);
      apply_permutation_gpu(reinterpret_cast<int64_t*>(val_buff),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc.entry_count,
                            chosen_bytes,
                            alloc);
    }
  }
}

const std::vector<const int8_t*>& QueryExecutionContext::getColumnFrag(const size_t table_idx,
                                                                       int64_t& global_idx) const {
#ifdef ENABLE_MULTIFRAG_JOIN
  if (col_buffers_.size() > 1) {
    int64_t frag_id = 0;
    int64_t local_idx = global_idx;
    if (consistent_frag_sizes_[table_idx] != -1) {
      frag_id = global_idx / consistent_frag_sizes_[table_idx];
      local_idx = global_idx % consistent_frag_sizes_[table_idx];
    } else {
      std::tie(frag_id, local_idx) = get_frag_id_and_local_idx(frag_offsets_, table_idx, global_idx);
    }
    CHECK_GE(frag_id, int64_t(0));
    CHECK_LT(frag_id, col_buffers_.size());
    global_idx = local_idx;
    return col_buffers_[frag_id];
  } else
#endif
  {
    CHECK_EQ(size_t(1), col_buffers_.size());
    return col_buffers_.front();
  }
}

RowSetPtr QueryExecutionContext::groupBufferToResults(const size_t i,
                                                      const std::vector<Analyzer::Expr*>& targets,
                                                      const bool was_auto_device) const {
  if (query_mem_desc_.interleavedBins(device_type_)) {
    return groupBufferToDeinterleavedResults(i);
  }
  CHECK_LT(i, result_sets_.size());
  return std::unique_ptr<ResultSet>(result_sets_[i].release());
}

RowSetPtr QueryExecutionContext::groupBufferToDeinterleavedResults(const size_t i) const {
  CHECK(!output_columnar_);
  const auto& result_set = result_sets_[i];
  auto deinterleaved_query_mem_desc = ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_);
  deinterleaved_query_mem_desc.interleaved_bins_on_gpu = false;
  for (auto& col_widths : deinterleaved_query_mem_desc.agg_col_widths) {
    col_widths.actual = col_widths.compact = 8;
  }
  auto deinterleaved_result_set = std::make_shared<ResultSet>(result_set->getTargetInfos(),
                                                              std::vector<ColumnLazyFetchInfo>{},
                                                              std::vector<std::vector<const int8_t*>>{},
#ifdef ENABLE_MULTIFRAG_JOIN
                                                              std::vector<std::vector<int64_t>>{},
                                                              std::vector<int64_t>{},
#endif
                                                              ExecutorDeviceType::CPU,
                                                              -1,
                                                              deinterleaved_query_mem_desc,
                                                              row_set_mem_owner_,
                                                              executor_);
  auto deinterleaved_storage = deinterleaved_result_set->allocateStorage(executor_->plan_state_->init_agg_vals_);
  auto deinterleaved_buffer = reinterpret_cast<int64_t*>(deinterleaved_storage->getUnderlyingBuffer());
  const auto rows_ptr = result_set->getStorage()->getUnderlyingBuffer();
  size_t deinterleaved_buffer_idx = 0;
  const size_t agg_col_count{query_mem_desc_.agg_col_widths.size()};
  for (size_t bin_base_off = query_mem_desc_.getColOffInBytes(0, 0), bin_idx = 0; bin_idx < result_set->entryCount();
       ++bin_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
    std::vector<int64_t> agg_vals(agg_col_count, 0);
    memcpy(&agg_vals[0], &executor_->plan_state_->init_agg_vals_[0], agg_col_count * sizeof(agg_vals[0]));
    ResultRows::reduceSingleRow(rows_ptr + bin_base_off,
                                executor_->warpSize(),
                                false,
                                true,
                                agg_vals,
                                query_mem_desc_,
                                result_set->getTargetInfos(),
                                executor_->plan_state_->init_agg_vals_);
    for (size_t agg_idx = 0; agg_idx < agg_col_count; ++agg_idx, ++deinterleaved_buffer_idx) {
      deinterleaved_buffer[deinterleaved_buffer_idx] = agg_vals[agg_idx];
    }
  }
  result_sets_[i].reset();
  return deinterleaved_result_set;
}
