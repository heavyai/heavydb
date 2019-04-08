/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "QueryExecutionContext.h"
#include "AggregateUtils.h"
#include "Execute.h"
#include "GpuInitGroups.h"
#include "QueryMemoryDescriptor.h"
#include "QueryMemoryInitializer.h"
#include "RelAlgExecutionUnit.h"
#include "SpeculativeTopN.h"
#include "StreamingTopN.h"

extern bool g_enable_columnar_output;

QueryExecutionContext::QueryExecutionContext(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const int device_id,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<std::vector<const int8_t*>>& iter_buffers,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool output_columnar,
    const bool sort_on_gpu,
    RenderInfo* render_info)
    : query_mem_desc_(query_mem_desc)
    , executor_(executor)
    , device_type_(device_type)
    , device_id_(device_id)
    , col_buffers_(col_buffers)
    , iter_buffers_(iter_buffers)
    , frag_offsets_(frag_offsets)
    , consistent_frag_sizes_(get_consistent_frags_sizes(frag_offsets))
    , row_set_mem_owner_(row_set_mem_owner)
    , output_columnar_(output_columnar)
    , sort_on_gpu_(sort_on_gpu) {
  auto render_allocator_map = render_info && render_info->isPotentialInSituRender()
                                  ? render_info->render_allocator_map_ptr.get()
                                  : nullptr;
  query_buffers_ = std::make_unique<QueryMemoryInitializer>(ra_exe_unit,
                                                            query_mem_desc,
                                                            device_id,
                                                            device_type,
                                                            output_columnar,
                                                            sort_on_gpu,
                                                            col_buffers,
                                                            consistent_frag_sizes_,
                                                            frag_offsets,
                                                            render_allocator_map,
                                                            render_info,
                                                            row_set_mem_owner,
                                                            executor);
}

ResultSetPtr QueryExecutionContext::groupBufferToDeinterleavedResults(
    const size_t i) const {
  CHECK(!output_columnar_);
  const auto& result_set = query_buffers_->getResultSet(i);
  auto deinterleaved_query_mem_desc =
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_);
  deinterleaved_query_mem_desc.setHasInterleavedBinsOnGpu(false);
  for (auto& col_widths : deinterleaved_query_mem_desc.agg_col_widths_) {
    col_widths.actual = col_widths.compact = 8;
  }
  deinterleaved_query_mem_desc.recomputePaddedColumnWidthBytes();

  auto deinterleaved_result_set =
      std::make_shared<ResultSet>(result_set->getTargetInfos(),
                                  std::vector<ColumnLazyFetchInfo>{},
                                  std::vector<std::vector<const int8_t*>>{},
                                  std::vector<std::vector<int64_t>>{},
                                  std::vector<int64_t>{},
                                  ExecutorDeviceType::CPU,
                                  -1,
                                  deinterleaved_query_mem_desc,
                                  row_set_mem_owner_,
                                  executor_);
  auto deinterleaved_storage =
      deinterleaved_result_set->allocateStorage(executor_->plan_state_->init_agg_vals_);
  auto deinterleaved_buffer =
      reinterpret_cast<int64_t*>(deinterleaved_storage->getUnderlyingBuffer());
  const auto rows_ptr = result_set->getStorage()->getUnderlyingBuffer();
  size_t deinterleaved_buffer_idx = 0;
  const size_t agg_col_count{query_mem_desc_.getColCount()};
  for (size_t bin_base_off = query_mem_desc_.getColOffInBytes(0), bin_idx = 0;
       bin_idx < result_set->entryCount();
       ++bin_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
    std::vector<int64_t> agg_vals(agg_col_count, 0);
    memcpy(&agg_vals[0],
           &executor_->plan_state_->init_agg_vals_[0],
           agg_col_count * sizeof(agg_vals[0]));
    ResultRows::reduceSingleRow(rows_ptr + bin_base_off,
                                executor_->warpSize(),
                                false,
                                true,
                                agg_vals,
                                query_mem_desc_,
                                result_set->getTargetInfos(),
                                executor_->plan_state_->init_agg_vals_);
    for (size_t agg_idx = 0; agg_idx < agg_col_count;
         ++agg_idx, ++deinterleaved_buffer_idx) {
      deinterleaved_buffer[deinterleaved_buffer_idx] = agg_vals[agg_idx];
    }
  }
  query_buffers_->resetResultSet(i);
  return deinterleaved_result_set;
}

int64_t QueryExecutionContext::getAggInitValForIndex(const size_t index) const {
  CHECK(query_buffers_);
  return query_buffers_->getAggInitValForIndex(index);
}

const std::vector<const int8_t*>& QueryExecutionContext::getColumnFrag(
    const size_t table_idx,
    int64_t& global_idx) const {
  if (col_buffers_.size() > 1) {
    int64_t frag_id = 0;
    int64_t local_idx = global_idx;
    if (consistent_frag_sizes_[table_idx] != -1) {
      frag_id = global_idx / consistent_frag_sizes_[table_idx];
      local_idx = global_idx % consistent_frag_sizes_[table_idx];
    } else {
      std::tie(frag_id, local_idx) =
          get_frag_id_and_local_idx(frag_offsets_, table_idx, global_idx);
    }
    CHECK_GE(frag_id, int64_t(0));
    CHECK_LT(frag_id, col_buffers_.size());
    global_idx = local_idx;
    return col_buffers_[frag_id];
  } else {
    CHECK_EQ(size_t(1), col_buffers_.size());
    return col_buffers_.front();
  }
}

ResultSetPtr QueryExecutionContext::getRowSet(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc) const {
  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>> results_per_sm;
  CHECK(query_buffers_);
  const auto group_by_buffers_size = query_buffers_->getNumBuffers();
  if (device_type_ == ExecutorDeviceType::CPU) {
    CHECK_EQ(size_t(1), group_by_buffers_size);
    return groupBufferToResults(0, ra_exe_unit.target_exprs);
  }
  size_t step{query_mem_desc_.threadsShareMemory() ? executor_->blockSize() : 1};
  for (size_t i = 0; i < group_by_buffers_size; i += step) {
    results_per_sm.emplace_back(groupBufferToResults(i, ra_exe_unit.target_exprs),
                                std::vector<size_t>{});
  }
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return executor_->reduceMultiDeviceResults(
      ra_exe_unit, results_per_sm, row_set_mem_owner_, query_mem_desc);
}

ResultSetPtr QueryExecutionContext::groupBufferToResults(
    const size_t i,
    const std::vector<Analyzer::Expr*>& targets) const {
  if (query_mem_desc_.interleavedBins(device_type_)) {
    return groupBufferToDeinterleavedResults(i);
  }
  return query_buffers_->getResultSetOwned(i);
}

#ifdef HAVE_CUDA
namespace {

int32_t aggregate_error_codes(const std::vector<int32_t>& error_codes) {
  // Check overflow / division by zero / interrupt first
  for (const auto err : error_codes) {
    if (err > 0) {
      return err;
    }
  }
  for (const auto err : error_codes) {
    if (err) {
      return err;
    }
  }
  return 0;
}

}  // namespace
#endif

std::vector<int64_t*> QueryExecutionContext::launchGpuCode(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<std::pair<void*, void*>>& cu_functions,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<std::vector<const int8_t*>> col_buffers,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const uint32_t frag_stride,
    const int32_t scan_limit,
    Data_Namespace::DataMgr* data_mgr,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    int32_t* error_code,
    const uint32_t num_tables,
    const std::vector<int64_t>& join_hash_tables,
    RenderAllocatorMap* render_allocator_map) {
  INJECT_TIMER(lauchGpuCode);
#ifdef HAVE_CUDA
  CHECK(query_buffers_);
  const auto& init_agg_vals = query_buffers_->init_agg_vals_;

  bool is_group_by{query_mem_desc_.isGroupBy()};

  RenderAllocator* render_allocator = nullptr;
  if (render_allocator_map) {
    render_allocator = render_allocator_map->getRenderAllocator(device_id);
  }
  CudaAllocator cuda_allocator(data_mgr, device_id);

  auto cu_func = static_cast<CUfunction>(cu_functions[device_id].first);
  std::vector<int64_t*> out_vec;
  uint32_t num_fragments = col_buffers.size();
  std::vector<int32_t> error_codes(grid_size_x * block_size_x);

  CUevent start0, stop0;  // preparation
  cuEventCreate(&start0, 0);
  cuEventCreate(&stop0, 0);
  CUevent start1, stop1;  // cuLaunchKernel
  cuEventCreate(&start1, 0);
  cuEventCreate(&stop1, 0);
  CUevent start2, stop2;  // finish
  cuEventCreate(&start2, 0);
  cuEventCreate(&stop2, 0);

  if (g_enable_dynamic_watchdog) {
    cuEventRecord(start0, 0);
  }

  if (g_enable_dynamic_watchdog) {
    initializeDynamicWatchdog(cu_functions[device_id].second, device_id);
  }

  auto kernel_params = prepareKernelParams(col_buffers,
                                           literal_buff,
                                           num_rows,
                                           frag_offsets,
                                           frag_stride,
                                           scan_limit,
                                           init_agg_vals,
                                           error_codes,
                                           num_tables,
                                           join_hash_tables,
                                           data_mgr,
                                           device_id,
                                           hoist_literals,
                                           is_group_by);

  CHECK_EQ(static_cast<size_t>(KERN_PARAM_COUNT), kernel_params.size());
  CHECK_EQ(CUdeviceptr(0), kernel_params[GROUPBY_BUF]);
  CHECK_EQ(CUdeviceptr(0), kernel_params[SMALL_BUF]);

  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  const unsigned grid_size_y = 1;
  const unsigned grid_size_z = 1;
  const auto total_thread_count = block_size_x * grid_size_x;
  const auto err_desc = kernel_params[ERROR_CODE];

  if (is_group_by) {
    CHECK(!(query_buffers_->getGroupByBuffersSize() == 0) || render_allocator);
    bool can_sort_on_gpu = query_mem_desc_.sortOnGpu();
    auto gpu_query_mem = prepareGroupByDevBuffer(cuda_allocator,
                                                 render_allocator,
                                                 ra_exe_unit,
                                                 kernel_params[INIT_AGG_VALS],
                                                 device_id,
                                                 block_size_x,
                                                 grid_size_x,
                                                 can_sort_on_gpu);

    kernel_params[GROUPBY_BUF] = gpu_query_mem.group_by_buffers.first;
    kernel_params[SMALL_BUF] = CUdeviceptr(0);  // TODO(adb): remove
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }

    if (g_enable_dynamic_watchdog) {
      cuEventRecord(stop0, 0);
      cuEventSynchronize(stop0);
      float milliseconds0 = 0;
      cuEventElapsedTime(&milliseconds0, start0, stop0);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: group-by prepare: " << std::to_string(milliseconds0)
              << " ms";
      cuEventRecord(start1, 0);
    }

    if (hoist_literals) {
      OOM_TRACE_PUSH();
      checkCudaErrors(
          cuLaunchKernel(cu_func,
                         grid_size_x,
                         grid_size_y,
                         grid_size_z,
                         block_size_x,
                         block_size_y,
                         block_size_z,
                         query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                         nullptr,
                         &param_ptrs[0],
                         nullptr));
    } else {
      OOM_TRACE_PUSH();
      param_ptrs.erase(param_ptrs.begin() + LITERALS);  // TODO(alex): remove
      checkCudaErrors(
          cuLaunchKernel(cu_func,
                         grid_size_x,
                         grid_size_y,
                         grid_size_z,
                         block_size_x,
                         block_size_y,
                         block_size_z,
                         query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                         nullptr,
                         &param_ptrs[0],
                         nullptr));
    }
    if (g_enable_dynamic_watchdog) {
      executor_->registerActiveModule(cu_functions[device_id].second, device_id);
      cuEventRecord(stop1, 0);
      cuEventSynchronize(stop1);
      executor_->unregisterActiveModule(cu_functions[device_id].second, device_id);
      float milliseconds1 = 0;
      cuEventElapsedTime(&milliseconds1, start1, stop1);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: group-by cuLaunchKernel: "
              << std::to_string(milliseconds1) << " ms";
      cuEventRecord(start2, 0);
    }

    cuda_allocator.copyFromDevice(&error_codes[0],
                                  err_desc,
                                  error_codes.size() * sizeof(error_codes[0]),
                                  device_id);
    *error_code = aggregate_error_codes(error_codes);
    if (*error_code > 0) {
      return {};
    }

    if (!render_allocator) {
      if (use_streaming_top_n(ra_exe_unit, query_mem_desc_)) {
        query_buffers_->applyStreamingTopNOffsetGpu(data_mgr,
                                                    query_mem_desc_,
                                                    gpu_query_mem,
                                                    ra_exe_unit,
                                                    total_thread_count,
                                                    device_id);
      } else {
        if (use_speculative_top_n(ra_exe_unit, query_mem_desc_)) {
          ResultRows::inplaceSortGpuImpl(ra_exe_unit.sort_info.order_entries,
                                         query_mem_desc_,
                                         gpu_query_mem,
                                         data_mgr,
                                         device_id);
        }
        if (query_mem_desc_.didOutputColumnar() &&
            query_mem_desc_.getQueryDescriptionType() ==
                QueryDescriptionType::Projection) {
          query_buffers_->compactProjectionBuffersGpu(
              query_mem_desc_,
              data_mgr,
              gpu_query_mem,
              get_num_allocated_rows_from_gpu(
                  data_mgr, kernel_params[TOTAL_MATCHED], device_id),
              device_id);
        } else {
          query_buffers_->copyGroupByBuffersFromGpu(
              data_mgr,
              query_mem_desc_,
              gpu_query_mem,
              ra_exe_unit,
              block_size_x,
              grid_size_x,
              device_id,
              can_sort_on_gpu && query_mem_desc_.hasKeylessHash());
        }
      }
    }
  } else {
    CHECK_EQ(num_fragments % frag_stride, 0u);
    const auto num_out_frags = num_fragments / frag_stride;
    std::vector<CUdeviceptr> out_vec_dev_buffers;
    const size_t agg_col_count{ra_exe_unit.estimator ? size_t(1) : init_agg_vals.size()};
    if (ra_exe_unit.estimator) {
      estimator_result_set_.reset(new ResultSet(
          ra_exe_unit.estimator, ExecutorDeviceType::GPU, device_id, data_mgr));
      out_vec_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(
          estimator_result_set_->getDeviceEstimatorBuffer()));
    } else {
      OOM_TRACE_PUSH();
      for (size_t i = 0; i < agg_col_count; ++i) {
        auto out_vec_dev_buffer = num_out_frags
                                      ? alloc_gpu_mem(data_mgr,
                                                      block_size_x * grid_size_x *
                                                          sizeof(int64_t) * num_out_frags,
                                                      device_id,
                                                      nullptr)
                                      : 0;
        out_vec_dev_buffers.push_back(out_vec_dev_buffer);
      }
    }
    auto out_vec_dev_ptr =
        alloc_gpu_mem(data_mgr, agg_col_count * sizeof(CUdeviceptr), device_id, nullptr);
    copy_to_gpu(data_mgr,
                out_vec_dev_ptr,
                &out_vec_dev_buffers[0],
                agg_col_count * sizeof(CUdeviceptr),
                device_id);
    CUdeviceptr unused_dev_ptr{0};
    kernel_params[GROUPBY_BUF] = out_vec_dev_ptr;
    kernel_params[SMALL_BUF] = unused_dev_ptr;
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }

    if (g_enable_dynamic_watchdog) {
      cuEventRecord(stop0, 0);
      cuEventSynchronize(stop0);
      float milliseconds0 = 0;
      cuEventElapsedTime(&milliseconds0, start0, stop0);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: prepare: " << std::to_string(milliseconds0) << " ms";
      cuEventRecord(start1, 0);
    }

    if (hoist_literals) {
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     0,
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    } else {
      param_ptrs.erase(param_ptrs.begin() + LITERALS);  // TODO(alex): remove
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     0,
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    }

    if (g_enable_dynamic_watchdog) {
      executor_->registerActiveModule(cu_functions[device_id].second, device_id);
      cuEventRecord(stop1, 0);
      cuEventSynchronize(stop1);
      executor_->unregisterActiveModule(cu_functions[device_id].second, device_id);
      float milliseconds1 = 0;
      cuEventElapsedTime(&milliseconds1, start1, stop1);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: cuLaunchKernel: " << std::to_string(milliseconds1)
              << " ms";
      cuEventRecord(start2, 0);
    }

    copy_from_gpu(data_mgr,
                  &error_codes[0],
                  err_desc,
                  error_codes.size() * sizeof(error_codes[0]),
                  device_id);
    *error_code = aggregate_error_codes(error_codes);
    if (*error_code > 0) {
      return {};
    }
    if (ra_exe_unit.estimator) {
      CHECK(estimator_result_set_);
      estimator_result_set_->syncEstimatorBuffer();
      return {};
    }
    for (size_t i = 0; i < agg_col_count; ++i) {
      int64_t* host_out_vec =
          new int64_t[block_size_x * grid_size_x * sizeof(int64_t) * num_out_frags];
      copy_from_gpu(data_mgr,
                    host_out_vec,
                    out_vec_dev_buffers[i],
                    block_size_x * grid_size_x * sizeof(int64_t) * num_out_frags,
                    device_id);
      out_vec.push_back(host_out_vec);
    }
  }
  const auto count_distinct_bitmap_mem = query_buffers_->getCountDistinctBitmapPtr();
  if (count_distinct_bitmap_mem) {
    copy_from_gpu(data_mgr,
                  query_buffers_->getCountDistinctHostPtr(),
                  count_distinct_bitmap_mem,
                  query_buffers_->getCountDistinctBitmapBytes(),
                  device_id);
  }

  if (g_enable_dynamic_watchdog) {
    cuEventRecord(stop2, 0);
    cuEventSynchronize(stop2);
    float milliseconds2 = 0;
    cuEventElapsedTime(&milliseconds2, start2, stop2);
    VLOG(1) << "Device " << std::to_string(device_id)
            << ": launchGpuCode: finish: " << std::to_string(milliseconds2) << " ms";
  }

  return out_vec;
#else
  return {};
#endif
}

std::vector<int64_t*> QueryExecutionContext::launchCpuCode(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<std::pair<void*, void*>>& fn_ptrs,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<std::vector<const int8_t*>> col_buffers,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const uint32_t frag_stride,
    const int32_t scan_limit,
    int32_t* error_code,
    const uint32_t num_tables,
    const std::vector<int64_t>& join_hash_tables) {
  INJECT_TIMER(lauchCpuCode);

  CHECK(query_buffers_);
  const auto& init_agg_vals = query_buffers_->init_agg_vals_;

  std::vector<const int8_t**> multifrag_col_buffers;
  for (auto& col_buffer : col_buffers) {
    multifrag_col_buffers.push_back(&col_buffer[0]);
  }
  const int8_t*** multifrag_cols_ptr{
      multifrag_col_buffers.empty() ? nullptr : &multifrag_col_buffers[0]};
  // TODO(adb): remove
  int64_t** small_group_by_buffers_ptr{nullptr};
  const uint32_t num_fragments = multifrag_cols_ptr
                                     ? static_cast<uint32_t>(col_buffers.size())
                                     : 0u;  // TODO(miyu): check 0
  CHECK_EQ(num_fragments % frag_stride, 0u);
  const auto num_out_frags = multifrag_cols_ptr ? num_fragments / frag_stride : 0u;

  const bool is_group_by{query_mem_desc_.isGroupBy()};
  std::vector<int64_t*> out_vec;
  if (ra_exe_unit.estimator) {
    estimator_result_set_.reset(
        new ResultSet(ra_exe_unit.estimator, ExecutorDeviceType::CPU, 0, nullptr));
    out_vec.push_back(
        reinterpret_cast<int64_t*>(estimator_result_set_->getHostEstimatorBuffer()));
  } else {
    if (!is_group_by) {
      for (size_t i = 0; i < init_agg_vals.size(); ++i) {
        auto buff = new int64_t[num_out_frags];
        out_vec.push_back(static_cast<int64_t*>(buff));
      }
    }
  }

  CHECK_EQ(num_rows.size(), col_buffers.size());
  std::vector<int64_t> flatened_num_rows;
  OOM_TRACE_PUSH();
  for (auto& nums : num_rows) {
    flatened_num_rows.insert(flatened_num_rows.end(), nums.begin(), nums.end());
  }
  std::vector<uint64_t> flatened_frag_offsets;
  for (auto& offsets : frag_offsets) {
    flatened_frag_offsets.insert(
        flatened_frag_offsets.end(), offsets.begin(), offsets.end());
  }
  int64_t rowid_lookup_num_rows{*error_code ? *error_code + 1 : 0};
  auto num_rows_ptr =
      rowid_lookup_num_rows ? &rowid_lookup_num_rows : &flatened_num_rows[0];
  int32_t total_matched_init{0};

  std::vector<int64_t> cmpt_val_buff;
  if (is_group_by) {
    cmpt_val_buff =
        compact_init_vals(align_to_int64(query_mem_desc_.getColsSize()) / sizeof(int64_t),
                          init_agg_vals,
                          query_mem_desc_);
  }

  const int64_t* join_hash_tables_ptr =
      join_hash_tables.size() == 1
          ? reinterpret_cast<int64_t*>(join_hash_tables[0])
          : (join_hash_tables.size() > 1 ? &join_hash_tables[0] : nullptr);
  if (hoist_literals) {
    using agg_query = void (*)(const int8_t***,  // col_buffers
                               const uint32_t*,  // num_fragments
                               const uint32_t*,  // frag_stride
                               const int8_t*,    // literals
                               const int64_t*,   // num_rows
                               const uint64_t*,  // frag_row_offsets
                               const int32_t*,   // max_matched
                               int32_t*,         // total_matched
                               const int64_t*,   // init_agg_value
                               int64_t**,        // out
                               int64_t**,        // out2
                               int32_t*,         // error_code
                               const uint32_t*,  // num_tables
                               const int64_t*);  // join_hash_tables_ptr
    if (is_group_by) {
      OOM_TRACE_PUSH();
      reinterpret_cast<agg_query>(fn_ptrs[0].first)(
          multifrag_cols_ptr,
          &num_fragments,
          &frag_stride,
          &literal_buff[0],
          num_rows_ptr,
          &flatened_frag_offsets[0],
          &scan_limit,
          &total_matched_init,
          &cmpt_val_buff[0],
          query_buffers_->getGroupByBuffersPtr(),
          small_group_by_buffers_ptr,
          error_code,
          &num_tables,
          join_hash_tables_ptr);
    } else {
      OOM_TRACE_PUSH();
      reinterpret_cast<agg_query>(fn_ptrs[0].first)(multifrag_cols_ptr,
                                                    &num_fragments,
                                                    &frag_stride,
                                                    &literal_buff[0],
                                                    num_rows_ptr,
                                                    &flatened_frag_offsets[0],
                                                    &scan_limit,
                                                    &total_matched_init,
                                                    &init_agg_vals[0],
                                                    &out_vec[0],
                                                    nullptr,
                                                    error_code,
                                                    &num_tables,
                                                    join_hash_tables_ptr);
    }
  } else {
    using agg_query = void (*)(const int8_t***,  // col_buffers
                               const uint32_t*,  // num_fragments
                               const uint32_t*,  // frag_stride
                               const int64_t*,   // num_rows
                               const uint64_t*,  // frag_row_offsets
                               const int32_t*,   // max_matched
                               int32_t*,         // total_matched
                               const int64_t*,   // init_agg_value
                               int64_t**,        // out
                               int64_t**,        // out2
                               int32_t*,         // error_code
                               const uint32_t*,  // num_tables
                               const int64_t*);  // join_hash_tables_ptr
    if (is_group_by) {
      OOM_TRACE_PUSH();
      reinterpret_cast<agg_query>(fn_ptrs[0].first)(
          multifrag_cols_ptr,
          &num_fragments,
          &frag_stride,
          num_rows_ptr,
          &flatened_frag_offsets[0],
          &scan_limit,
          &total_matched_init,
          &cmpt_val_buff[0],
          query_buffers_->getGroupByBuffersPtr(),
          small_group_by_buffers_ptr,
          error_code,
          &num_tables,
          join_hash_tables_ptr);
    } else {
      OOM_TRACE_PUSH();
      reinterpret_cast<agg_query>(fn_ptrs[0].first)(multifrag_cols_ptr,
                                                    &num_fragments,
                                                    &frag_stride,
                                                    num_rows_ptr,
                                                    &flatened_frag_offsets[0],
                                                    &scan_limit,
                                                    &total_matched_init,
                                                    &init_agg_vals[0],
                                                    &out_vec[0],
                                                    nullptr,
                                                    error_code,
                                                    &num_tables,
                                                    join_hash_tables_ptr);
    }
  }

  if (ra_exe_unit.estimator) {
    return {};
  }

  if (rowid_lookup_num_rows && *error_code < 0) {
    *error_code = 0;
  }

  if (use_streaming_top_n(ra_exe_unit, query_mem_desc_)) {
    query_buffers_->applyStreamingTopNOffsetCpu(query_mem_desc_, ra_exe_unit);
  }

  if (query_mem_desc_.didOutputColumnar() &&
      query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    query_buffers_->compactProjectionBuffersCpu(query_mem_desc_, total_matched_init);
  }

  return out_vec;
}

#ifdef HAVE_CUDA
void QueryExecutionContext::initializeDynamicWatchdog(void* native_module,
                                                      const int device_id) const {
  auto cu_module = static_cast<CUmodule>(native_module);
  CHECK(cu_module);
  CUevent start, stop;
  cuEventCreate(&start, 0);
  cuEventCreate(&stop, 0);
  cuEventRecord(start, 0);

  CUdeviceptr dw_cycle_budget;
  size_t dw_cycle_budget_size;
  // Translate milliseconds to device cycles
  uint64_t cycle_budget = executor_->deviceCycles(g_dynamic_watchdog_time_limit);
  if (device_id == 0) {
    LOG(INFO) << "Dynamic Watchdog budget: GPU: "
              << std::to_string(g_dynamic_watchdog_time_limit) << "ms, "
              << std::to_string(cycle_budget) << " cycles";
  }
  checkCudaErrors(cuModuleGetGlobal(
      &dw_cycle_budget, &dw_cycle_budget_size, cu_module, "dw_cycle_budget"));
  CHECK_EQ(dw_cycle_budget_size, sizeof(uint64_t));
  checkCudaErrors(cuMemcpyHtoD(
      dw_cycle_budget, reinterpret_cast<void*>(&cycle_budget), sizeof(uint64_t)));

  CUdeviceptr dw_sm_cycle_start;
  size_t dw_sm_cycle_start_size;
  checkCudaErrors(cuModuleGetGlobal(
      &dw_sm_cycle_start, &dw_sm_cycle_start_size, cu_module, "dw_sm_cycle_start"));
  CHECK_EQ(dw_sm_cycle_start_size, 128 * sizeof(uint64_t));
  checkCudaErrors(cuMemsetD32(dw_sm_cycle_start, 0, 128 * 2));

  if (!executor_->interrupted_) {
    // Executor is not marked as interrupted, make sure dynamic watchdog doesn't block
    // execution
    CUdeviceptr dw_abort;
    size_t dw_abort_size;
    checkCudaErrors(cuModuleGetGlobal(&dw_abort, &dw_abort_size, cu_module, "dw_abort"));
    CHECK_EQ(dw_abort_size, sizeof(uint32_t));
    checkCudaErrors(cuMemsetD32(dw_abort, 0, 1));
  }

  cuEventRecord(stop, 0);
  cuEventSynchronize(stop);
  float milliseconds = 0;
  cuEventElapsedTime(&milliseconds, start, stop);
  VLOG(1) << "Device " << std::to_string(device_id)
          << ": launchGpuCode: dynamic watchdog init: " << std::to_string(milliseconds)
          << " ms\n";
}

std::vector<CUdeviceptr> QueryExecutionContext::prepareKernelParams(
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<int8_t>& literal_buff,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const uint32_t frag_stride,
    const int32_t scan_limit,
    const std::vector<int64_t>& init_agg_vals,
    const std::vector<int32_t>& error_codes,
    const uint32_t num_tables,
    const std::vector<int64_t>& join_hash_tables,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const bool hoist_literals,
    const bool is_group_by) const {
  std::vector<CUdeviceptr> params(KERN_PARAM_COUNT, 0);
  const uint32_t num_fragments = col_buffers.size();
  const size_t col_count{num_fragments > 0 ? col_buffers.front().size() : 0};
  if (col_count) {
    std::vector<CUdeviceptr> multifrag_col_dev_buffers;
    for (auto frag_col_buffers : col_buffers) {
      std::vector<CUdeviceptr> col_dev_buffers;
      for (auto col_buffer : frag_col_buffers) {
        col_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(col_buffer));
      }
      auto col_buffers_dev_ptr =
          alloc_gpu_mem(data_mgr, col_count * sizeof(CUdeviceptr), device_id, nullptr);
      copy_to_gpu(data_mgr,
                  col_buffers_dev_ptr,
                  &col_dev_buffers[0],
                  col_count * sizeof(CUdeviceptr),
                  device_id);
      multifrag_col_dev_buffers.push_back(col_buffers_dev_ptr);
    }
    params[COL_BUFFERS] =
        alloc_gpu_mem(data_mgr, num_fragments * sizeof(CUdeviceptr), device_id, nullptr);
    copy_to_gpu(data_mgr,
                params[COL_BUFFERS],
                &multifrag_col_dev_buffers[0],
                num_fragments * sizeof(CUdeviceptr),
                device_id);
  }
  params[NUM_FRAGMENTS] = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id, nullptr);
  copy_to_gpu(
      data_mgr, params[NUM_FRAGMENTS], &num_fragments, sizeof(uint32_t), device_id);

  params[FRAG_STRIDE] = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id, nullptr);
  copy_to_gpu(data_mgr, params[FRAG_STRIDE], &frag_stride, sizeof(uint32_t), device_id);

  CUdeviceptr literals_and_addr_mapping = alloc_gpu_mem(
      data_mgr, literal_buff.size() + 2 * sizeof(int64_t), device_id, nullptr);
  CHECK_EQ(0, literals_and_addr_mapping % 8);
  std::vector<int64_t> additional_literal_bytes;
  const auto count_distinct_bitmap_mem = query_buffers_->getCountDistinctBitmapPtr();
  if (count_distinct_bitmap_mem) {
    // Store host and device addresses
    const auto count_distinct_bitmap_host_mem = query_buffers_->getCountDistinctHostPtr();
    CHECK(count_distinct_bitmap_host_mem);
    additional_literal_bytes.push_back(
        reinterpret_cast<int64_t>(count_distinct_bitmap_host_mem));
    additional_literal_bytes.push_back(static_cast<int64_t>(count_distinct_bitmap_mem));
    copy_to_gpu(data_mgr,
                literals_and_addr_mapping,
                &additional_literal_bytes[0],
                additional_literal_bytes.size() * sizeof(additional_literal_bytes[0]),
                device_id);
  }
  params[LITERALS] = literals_and_addr_mapping + additional_literal_bytes.size() *
                                                     sizeof(additional_literal_bytes[0]);
  if (!literal_buff.empty()) {
    CHECK(hoist_literals);
    copy_to_gpu(
        data_mgr, params[LITERALS], &literal_buff[0], literal_buff.size(), device_id);
  }
  CHECK_EQ(num_rows.size(), col_buffers.size());
  std::vector<int64_t> flatened_num_rows;
  for (auto& nums : num_rows) {
    CHECK_EQ(nums.size(), num_tables);
    flatened_num_rows.insert(flatened_num_rows.end(), nums.begin(), nums.end());
  }
  params[NUM_ROWS] = alloc_gpu_mem(
      data_mgr, sizeof(int64_t) * flatened_num_rows.size(), device_id, nullptr);
  copy_to_gpu(data_mgr,
              params[NUM_ROWS],
              &flatened_num_rows[0],
              sizeof(int64_t) * flatened_num_rows.size(),
              device_id);

  CHECK_EQ(frag_offsets.size(), col_buffers.size());
  std::vector<int64_t> flatened_frag_offsets;
  for (auto& offsets : frag_offsets) {
    CHECK_EQ(offsets.size(), num_tables);
    flatened_frag_offsets.insert(
        flatened_frag_offsets.end(), offsets.begin(), offsets.end());
  }
  params[FRAG_ROW_OFFSETS] = alloc_gpu_mem(
      data_mgr, sizeof(int64_t) * flatened_frag_offsets.size(), device_id, nullptr);
  copy_to_gpu(data_mgr,
              params[FRAG_ROW_OFFSETS],
              &flatened_frag_offsets[0],
              sizeof(int64_t) * flatened_frag_offsets.size(),
              device_id);
  int32_t max_matched{scan_limit};
  params[MAX_MATCHED] = alloc_gpu_mem(data_mgr, sizeof(max_matched), device_id, nullptr);
  copy_to_gpu(
      data_mgr, params[MAX_MATCHED], &max_matched, sizeof(max_matched), device_id);

  int32_t total_matched{0};
  params[TOTAL_MATCHED] =
      alloc_gpu_mem(data_mgr, sizeof(total_matched), device_id, nullptr);
  copy_to_gpu(
      data_mgr, params[TOTAL_MATCHED], &total_matched, sizeof(total_matched), device_id);

  if (is_group_by && !output_columnar_) {
    auto cmpt_sz = align_to_int64(query_mem_desc_.getColsSize()) / sizeof(int64_t);
    auto cmpt_val_buff = compact_init_vals(cmpt_sz, init_agg_vals, query_mem_desc_);
    params[INIT_AGG_VALS] =
        alloc_gpu_mem(data_mgr, cmpt_sz * sizeof(int64_t), device_id, nullptr);
    copy_to_gpu(data_mgr,
                params[INIT_AGG_VALS],
                &cmpt_val_buff[0],
                cmpt_sz * sizeof(int64_t),
                device_id);
  } else {
    params[INIT_AGG_VALS] = alloc_gpu_mem(
        data_mgr, init_agg_vals.size() * sizeof(int64_t), device_id, nullptr);
    copy_to_gpu(data_mgr,
                params[INIT_AGG_VALS],
                &init_agg_vals[0],
                init_agg_vals.size() * sizeof(int64_t),
                device_id);
  }

  params[ERROR_CODE] = alloc_gpu_mem(
      data_mgr, error_codes.size() * sizeof(error_codes[0]), device_id, nullptr);
  copy_to_gpu(data_mgr,
              params[ERROR_CODE],
              &error_codes[0],
              error_codes.size() * sizeof(error_codes[0]),
              device_id);

  params[NUM_TABLES] = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id, nullptr);
  copy_to_gpu(data_mgr, params[NUM_TABLES], &num_tables, sizeof(uint32_t), device_id);

  const auto hash_table_count = join_hash_tables.size();
  switch (hash_table_count) {
    case 0: {
      params[JOIN_HASH_TABLES] = CUdeviceptr(0);
    } break;
    case 1:
      params[JOIN_HASH_TABLES] = static_cast<CUdeviceptr>(join_hash_tables[0]);
      break;
    default: {
      params[JOIN_HASH_TABLES] =
          alloc_gpu_mem(data_mgr, hash_table_count * sizeof(int64_t), device_id, nullptr);
      copy_to_gpu(data_mgr,
                  params[JOIN_HASH_TABLES],
                  &join_hash_tables[0],
                  hash_table_count * sizeof(int64_t),
                  device_id);
    } break;
  }

  return params;
}

std::pair<CUdeviceptr, CUdeviceptr> QueryExecutionContext::prepareTopNHeapsDevBuffer(
    const CudaAllocator& cuda_allocator,
    const CUdeviceptr init_agg_vals_dev_ptr,
    const size_t n,
    const int device_id,
    const unsigned block_size_x,
    const unsigned grid_size_x) const {
  const auto thread_count = block_size_x * grid_size_x;
  const auto total_buff_size =
      streaming_top_n::get_heap_size(query_mem_desc_.getRowSize(), n, thread_count);
  OOM_TRACE_PUSH();
  CUdeviceptr dev_buffer = cuda_allocator.alloc(total_buff_size, device_id, nullptr);

  std::vector<CUdeviceptr> dev_buffers(thread_count);

  for (size_t i = 0; i < thread_count; ++i) {
    dev_buffers[i] = dev_buffer;
  }

  auto dev_ptr =
      cuda_allocator.alloc(thread_count * sizeof(CUdeviceptr), device_id, nullptr);
  cuda_allocator.copyToDevice(
      dev_ptr, &dev_buffers[0], thread_count * sizeof(CUdeviceptr), device_id);

  CHECK(query_mem_desc_.lazyInitGroups(ExecutorDeviceType::GPU));
  CHECK(!output_columnar_);

  cuda_allocator.zeroDeviceMem(
      reinterpret_cast<int8_t*>(dev_buffer), thread_count * sizeof(int64_t), device_id);

  cuda_allocator.setDeviceMem(
      reinterpret_cast<int8_t*>(dev_buffer + thread_count * sizeof(int64_t)),
      (unsigned char)-1,
      thread_count * n * sizeof(int64_t),
      device_id);

  init_group_by_buffer_on_device(
      reinterpret_cast<int64_t*>(
          dev_buffer + streaming_top_n::get_rows_offset_of_heaps(n, thread_count)),
      reinterpret_cast<int64_t*>(init_agg_vals_dev_ptr),
      n * thread_count,
      query_mem_desc_.groupColWidthsSize(),
      query_mem_desc_.getEffectiveKeyWidth(),
      query_mem_desc_.getRowSize() / sizeof(int64_t),
      query_mem_desc_.hasKeylessHash(),
      1,
      block_size_x,
      grid_size_x);

  return {dev_ptr, dev_buffer};
}

GpuQueryMemory QueryExecutionContext::prepareGroupByDevBuffer(
    const CudaAllocator& cuda_allocator,
    RenderAllocator* render_allocator,
    const RelAlgExecutionUnit& ra_exe_unit,
    const CUdeviceptr init_agg_vals_dev_ptr,
    const int device_id,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const bool can_sort_on_gpu) const {
  CHECK(query_buffers_);
  if (use_streaming_top_n(ra_exe_unit, query_mem_desc_)) {
    if (render_allocator) {
      throw StreamingTopNNotSupportedInRenderQuery();
    }
    const auto n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
    auto heap_buffers = prepareTopNHeapsDevBuffer(
        cuda_allocator, init_agg_vals_dev_ptr, n, device_id, block_size_x, grid_size_x);
    return GpuQueryMemory{heap_buffers};
  }
  auto dev_group_by_buffers = query_buffers_->createGroupByBuffersOnGpu(cuda_allocator,
                                                                        render_allocator,
                                                                        query_mem_desc_,
                                                                        device_id,
                                                                        block_size_x,
                                                                        grid_size_x,
                                                                        can_sort_on_gpu);
  if (render_allocator) {
    CHECK_EQ(size_t(0), render_allocator->getAllocatedSize() % 8);
  }
  if (query_mem_desc_.lazyInitGroups(ExecutorDeviceType::GPU)) {
    CHECK(!render_allocator);
    const size_t step{query_mem_desc_.threadsShareMemory() ? block_size_x : 1};
    size_t groups_buffer_size{
        query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU)};
    auto group_by_dev_buffer = dev_group_by_buffers.second;
    const size_t col_count = query_mem_desc_.getColCount();
    CUdeviceptr col_widths_dev_ptr{0};
    if (output_columnar_) {
      std::vector<int8_t> compact_col_widths(col_count);
      for (size_t idx = 0; idx < col_count; ++idx) {
        compact_col_widths[idx] = query_mem_desc_.getPaddedColumnWidthBytes(idx);
      }
      col_widths_dev_ptr =
          cuda_allocator.alloc(col_count * sizeof(int8_t), device_id, nullptr);
      cuda_allocator.copyToDevice(col_widths_dev_ptr,
                                  &compact_col_widths[0],
                                  col_count * sizeof(int8_t),
                                  device_id);
    }
    const int8_t warp_count = query_mem_desc_.interleavedBins(ExecutorDeviceType::GPU)
                                  ? executor_->warpSize()
                                  : 1;
    OOM_TRACE_PUSH();
    for (size_t i = 0; i < query_buffers_->getGroupByBuffersSize(); i += step) {
      if (output_columnar_) {
        init_columnar_group_by_buffer_on_device(
            reinterpret_cast<int64_t*>(group_by_dev_buffer),
            reinterpret_cast<const int64_t*>(init_agg_vals_dev_ptr),
            query_mem_desc_.getEntryCount(),
            query_mem_desc_.groupColWidthsSize(),
            col_count,
            reinterpret_cast<int8_t*>(col_widths_dev_ptr),
            /*need_padding = */ true,
            query_mem_desc_.hasKeylessHash(),
            sizeof(int64_t),
            block_size_x,
            grid_size_x);
      } else {
        init_group_by_buffer_on_device(reinterpret_cast<int64_t*>(group_by_dev_buffer),
                                       reinterpret_cast<int64_t*>(init_agg_vals_dev_ptr),
                                       query_mem_desc_.getEntryCount(),
                                       query_mem_desc_.groupColWidthsSize(),
                                       query_mem_desc_.getEffectiveKeyWidth(),
                                       query_mem_desc_.getRowSize() / sizeof(int64_t),
                                       query_mem_desc_.hasKeylessHash(),
                                       warp_count,
                                       block_size_x,
                                       grid_size_x);
      }
      group_by_dev_buffer += groups_buffer_size;
    }
  }
  return GpuQueryMemory{dev_group_by_buffers};
}
#endif
