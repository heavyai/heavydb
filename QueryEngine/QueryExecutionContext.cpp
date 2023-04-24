/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include "Descriptors/QueryMemoryDescriptor.h"
#include "DeviceKernel.h"
#include "Execute.h"
#include "GpuInitGroups.h"
#include "InPlaceSort.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/RowFunctionManager.h"
#include "QueryMemoryInitializer.h"
#include "RelAlgExecutionUnit.h"
#include "ResultSet.h"
#include "Shared/likely.h"
#include "SpeculativeTopN.h"
#include "StreamingTopN.h"

QueryExecutionContext::QueryExecutionContext(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const ExecutorDispatchMode dispatch_mode,
    const int device_id,
    const shared::TableKey& outer_table_key,
    const int64_t num_rows,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool output_columnar,
    const bool sort_on_gpu,
    const size_t thread_idx,
    RenderInfo* render_info)
    : query_mem_desc_(query_mem_desc)
    , executor_(executor)
    , device_type_(device_type)
    , dispatch_mode_(dispatch_mode)
    , row_set_mem_owner_(row_set_mem_owner)
    , output_columnar_(output_columnar) {
  CHECK(executor);
  auto data_mgr = executor->getDataMgr();
  if (device_type == ExecutorDeviceType::GPU) {
    gpu_allocator_ = std::make_unique<CudaAllocator>(
        data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id));
  }

  auto render_allocator_map = render_info && render_info->isInSitu()
                                  ? render_info->render_allocator_map_ptr.get()
                                  : nullptr;
  query_buffers_ = std::make_unique<QueryMemoryInitializer>(ra_exe_unit,
                                                            query_mem_desc,
                                                            device_id,
                                                            device_type,
                                                            dispatch_mode,
                                                            output_columnar,
                                                            sort_on_gpu,
                                                            outer_table_key,
                                                            num_rows,
                                                            col_buffers,
                                                            frag_offsets,
                                                            render_allocator_map,
                                                            render_info,
                                                            row_set_mem_owner,
                                                            gpu_allocator_.get(),
                                                            thread_idx,
                                                            executor);
}

ResultSetPtr QueryExecutionContext::groupBufferToDeinterleavedResults(
    const size_t i) const {
  CHECK(!output_columnar_);
  const auto& result_set = query_buffers_->getResultSet(i);
  auto deinterleaved_query_mem_desc =
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_);
  deinterleaved_query_mem_desc.setHasInterleavedBinsOnGpu(false);
  deinterleaved_query_mem_desc.useConsistentSlotWidthSize(8);

  auto deinterleaved_result_set =
      std::make_shared<ResultSet>(result_set->getTargetInfos(),
                                  std::vector<ColumnLazyFetchInfo>{},
                                  std::vector<std::vector<const int8_t*>>{},
                                  std::vector<std::vector<int64_t>>{},
                                  std::vector<int64_t>{},
                                  ExecutorDeviceType::CPU,
                                  -1,
                                  -1,
                                  deinterleaved_query_mem_desc,
                                  row_set_mem_owner_,
                                  executor_->blockSize(),
                                  executor_->gridSize());
  auto deinterleaved_storage =
      deinterleaved_result_set->allocateStorage(executor_->plan_state_->init_agg_vals_);
  auto deinterleaved_buffer =
      reinterpret_cast<int64_t*>(deinterleaved_storage->getUnderlyingBuffer());
  const auto rows_ptr = result_set->getStorage()->getUnderlyingBuffer();
  size_t deinterleaved_buffer_idx = 0;
  const size_t agg_col_count{query_mem_desc_.getSlotCount()};
  auto do_work = [&](const size_t bin_base_off) {
    std::vector<int64_t> agg_vals(agg_col_count, 0);
    memcpy(&agg_vals[0],
           &executor_->plan_state_->init_agg_vals_[0],
           agg_col_count * sizeof(agg_vals[0]));
    ResultSetStorage::reduceSingleRow(rows_ptr + bin_base_off,
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
  };
  if (g_enable_non_kernel_time_query_interrupt) {
    for (size_t bin_base_off = query_mem_desc_.getColOffInBytes(0), bin_idx = 0;
         bin_idx < result_set->entryCount();
         ++bin_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
      if (UNLIKELY((bin_idx & 0xFFFF) == 0 &&
                   executor_->checkNonKernelTimeInterrupted())) {
        throw std::runtime_error(
            "Query execution has interrupted during result set reduction");
      }
      do_work(bin_base_off);
    }
  } else {
    for (size_t bin_base_off = query_mem_desc_.getColOffInBytes(0), bin_idx = 0;
         bin_idx < result_set->entryCount();
         ++bin_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
      do_work(bin_base_off);
    }
  }
  query_buffers_->resetResultSet(i);
  return deinterleaved_result_set;
}

int64_t QueryExecutionContext::getAggInitValForIndex(const size_t index) const {
  CHECK(query_buffers_);
  return query_buffers_->getAggInitValForIndex(index);
}

ResultSetPtr QueryExecutionContext::getRowSet(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc) const {
  auto timer = DEBUG_TIMER(__func__);
  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>> results_per_sm;
  CHECK(query_buffers_);
  const auto group_by_buffers_size = query_buffers_->getNumBuffers();
  if (device_type_ == ExecutorDeviceType::CPU) {
    const size_t expected_num_buffers = query_mem_desc.hasVarlenOutput() ? 2 : 1;
    CHECK_EQ(expected_num_buffers, group_by_buffers_size);
    return groupBufferToResults(0);
  }
  const size_t step{query_mem_desc_.threadsShareMemory() ? executor_->blockSize() : 1};
  const size_t group_by_output_buffers_size =
      group_by_buffers_size - (query_mem_desc.hasVarlenOutput() ? 1 : 0);
  for (size_t i = 0; i < group_by_output_buffers_size; i += step) {
    results_per_sm.emplace_back(groupBufferToResults(i), std::vector<size_t>{});
  }
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return executor_->reduceMultiDeviceResults(
      ra_exe_unit, results_per_sm, row_set_mem_owner_, query_mem_desc);
}

ResultSetPtr QueryExecutionContext::groupBufferToResults(const size_t i) const {
  if (query_mem_desc_.interleavedBins(device_type_)) {
    return groupBufferToDeinterleavedResults(i);
  }
  return query_buffers_->getResultSetOwned(i);
}

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

std::vector<int64_t*> QueryExecutionContext::launchGpuCode(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CompilationContext* compilation_context,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<std::vector<const int8_t*>> col_buffers,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const int32_t scan_limit,
    Data_Namespace::DataMgr* data_mgr,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const size_t shared_memory_size,
    int32_t* error_code,
    const uint32_t num_tables,
    const bool allow_runtime_interrupt,
    const std::vector<int8_t*>& join_hash_tables,
    RenderAllocatorMap* render_allocator_map,
    bool optimize_cuda_block_and_grid_sizes) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(lauchGpuCode);
  CHECK(gpu_allocator_);
  CHECK(query_buffers_);
  CHECK(compilation_context);
  const auto& init_agg_vals = query_buffers_->init_agg_vals_;

  bool is_group_by{query_mem_desc_.isGroupBy()};

  RenderAllocator* render_allocator = nullptr;
  if (render_allocator_map) {
    render_allocator = render_allocator_map->getRenderAllocator(device_id);
  }

  auto kernel = create_device_kernel(compilation_context, device_id);

  std::vector<int64_t*> out_vec;
  uint32_t num_fragments = col_buffers.size();
  std::vector<int32_t> error_codes(grid_size_x * block_size_x);

  auto prepareClock = kernel->make_clock();
  auto launchClock = kernel->make_clock();
  auto finishClock = kernel->make_clock();

  if (g_enable_dynamic_watchdog || (allow_runtime_interrupt && !render_allocator)) {
    prepareClock->start();
  }

  if (g_enable_dynamic_watchdog) {
    kernel->initializeDynamicWatchdog(
        executor_->interrupted_.load(),
        executor_->deviceCycles(g_dynamic_watchdog_time_limit));
  }

  if (allow_runtime_interrupt && !render_allocator) {
    kernel->initializeRuntimeInterrupter(device_id);
  }

  auto kernel_params = prepareKernelParams(col_buffers,
                                           literal_buff,
                                           num_rows,
                                           frag_offsets,
                                           scan_limit,
                                           init_agg_vals,
                                           error_codes,
                                           num_tables,
                                           join_hash_tables,
                                           data_mgr,
                                           device_id,
                                           hoist_literals,
                                           is_group_by);

  static_assert(size_t(KERN_PARAM_COUNT) == kernel_params.size());
  CHECK(!kernel_params[GROUPBY_BUF]);

  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  const unsigned grid_size_y = 1;
  const unsigned grid_size_z = 1;
  const auto total_thread_count = block_size_x * grid_size_x;
  const auto err_desc = kernel_params[ERROR_CODE];
  if (is_group_by) {
    CHECK(!(query_buffers_->getGroupByBuffersSize() == 0) || render_allocator);
    bool can_sort_on_gpu = query_mem_desc_.sortOnGpu();
    auto gpu_group_by_buffers =
        query_buffers_->createAndInitializeGroupByBufferGpu(ra_exe_unit,
                                                            query_mem_desc_,
                                                            kernel_params[INIT_AGG_VALS],
                                                            device_id,
                                                            dispatch_mode_,
                                                            block_size_x,
                                                            grid_size_x,
                                                            executor_->warpSize(),
                                                            can_sort_on_gpu,
                                                            output_columnar_,
                                                            render_allocator);
    const auto max_matched = static_cast<int32_t>(gpu_group_by_buffers.entry_count);
    gpu_allocator_->copyToDevice(
        kernel_params[MAX_MATCHED], &max_matched, sizeof(max_matched));

    kernel_params[GROUPBY_BUF] = gpu_group_by_buffers.ptrs;
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }

    if (g_enable_dynamic_watchdog || (allow_runtime_interrupt && !render_allocator)) {
      auto prepareTime = prepareClock->stop();
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: group-by prepare: " << std::to_string(prepareTime)
              << " ms";
      launchClock->start();
    }

    if (hoist_literals) {
      VLOG(1) << "Launching(" << kernel->name() << ") on device_id(" << device_id << ')';
      kernel->launch(grid_size_x,
                     grid_size_y,
                     grid_size_z,
                     block_size_x,
                     block_size_y,
                     block_size_z,
                     shared_memory_size,
                     &param_ptrs[0],
                     optimize_cuda_block_and_grid_sizes);
    } else {
      param_ptrs.erase(param_ptrs.begin() + LITERALS);  // TODO(alex): remove
      VLOG(1) << "Launching(" << kernel->name() << ") on device_id(" << device_id << ')';
      kernel->launch(grid_size_x,
                     grid_size_y,
                     grid_size_z,
                     block_size_x,
                     block_size_y,
                     block_size_z,
                     shared_memory_size,
                     &param_ptrs[0],
                     optimize_cuda_block_and_grid_sizes);
    }
    if (g_enable_dynamic_watchdog || (allow_runtime_interrupt && !render_allocator)) {
      auto launchTime = launchClock->stop();
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: group-by cuLaunchKernel: "
              << std::to_string(launchTime) << " ms";
      finishClock->start();
    }

    gpu_allocator_->copyFromDevice(reinterpret_cast<int8_t*>(error_codes.data()),
                                   reinterpret_cast<int8_t*>(err_desc),
                                   error_codes.size() * sizeof(error_codes[0]));
    *error_code = aggregate_error_codes(error_codes);
    if (*error_code > 0) {
      return {};
    }

    if (!render_allocator) {
      if (query_mem_desc_.useStreamingTopN()) {
        query_buffers_->applyStreamingTopNOffsetGpu(data_mgr,
                                                    query_mem_desc_,
                                                    gpu_group_by_buffers,
                                                    ra_exe_unit,
                                                    total_thread_count,
                                                    device_id);
      } else {
        if (use_speculative_top_n(ra_exe_unit, query_mem_desc_)) {
          try {
            inplace_sort_gpu(ra_exe_unit.sort_info.order_entries,
                             query_mem_desc_,
                             gpu_group_by_buffers,
                             data_mgr,
                             device_id);
          } catch (const std::bad_alloc&) {
            throw SpeculativeTopNFailed("Failed during in-place GPU sort.");
          }
        }
        if (query_mem_desc_.getQueryDescriptionType() ==
            QueryDescriptionType::Projection) {
          if (query_mem_desc_.didOutputColumnar()) {
            query_buffers_->compactProjectionBuffersGpu(
                query_mem_desc_,
                data_mgr,
                gpu_group_by_buffers,
                get_num_allocated_rows_from_gpu(
                    *gpu_allocator_, kernel_params[TOTAL_MATCHED], device_id),
                device_id);
          } else {
            size_t num_allocated_rows{0};
            if (ra_exe_unit.use_bump_allocator) {
              num_allocated_rows = get_num_allocated_rows_from_gpu(
                  *gpu_allocator_, kernel_params[TOTAL_MATCHED], device_id);
              // First, check the error code. If we ran out of slots, don't copy data back
              // into the ResultSet or update ResultSet entry count
              if (*error_code < 0) {
                return {};
              }
            }
            query_buffers_->copyGroupByBuffersFromGpu(
                *gpu_allocator_,
                query_mem_desc_,
                ra_exe_unit.use_bump_allocator ? num_allocated_rows
                                               : query_mem_desc_.getEntryCount(),
                gpu_group_by_buffers,
                &ra_exe_unit,
                block_size_x,
                grid_size_x,
                device_id,
                can_sort_on_gpu && query_mem_desc_.hasKeylessHash());
            if (num_allocated_rows) {
              CHECK(ra_exe_unit.use_bump_allocator);
              CHECK(!query_buffers_->result_sets_.empty());
              query_buffers_->result_sets_.front()->updateStorageEntryCount(
                  num_allocated_rows);
            }
          }
        } else {
          query_buffers_->copyGroupByBuffersFromGpu(
              *gpu_allocator_,
              query_mem_desc_,
              query_mem_desc_.getEntryCount(),
              gpu_group_by_buffers,
              &ra_exe_unit,
              block_size_x,
              grid_size_x,
              device_id,
              can_sort_on_gpu && query_mem_desc_.hasKeylessHash());
        }
      }
    }
  } else {
    std::vector<int8_t*> out_vec_dev_buffers;
    const size_t agg_col_count{ra_exe_unit.estimator ? size_t(1) : init_agg_vals.size()};
    // by default, non-grouped aggregate queries generate one result per available thread
    // in the lifetime of (potentially multi-fragment) kernel execution.
    // We can reduce these intermediate results internally in the device and hence have
    // only one result per device, if GPU shared memory optimizations are enabled.
    const auto num_results_per_agg_col =
        shared_memory_size ? 1 : block_size_x * grid_size_x * num_fragments;
    const auto output_buffer_size_per_agg = num_results_per_agg_col * sizeof(int64_t);
    if (ra_exe_unit.estimator) {
      estimator_result_set_.reset(new ResultSet(
          ra_exe_unit.estimator, ExecutorDeviceType::GPU, device_id, data_mgr));
      out_vec_dev_buffers.push_back(estimator_result_set_->getDeviceEstimatorBuffer());
    } else {
      for (size_t i = 0; i < agg_col_count; ++i) {
        int8_t* out_vec_dev_buffer =
            num_fragments ? gpu_allocator_->alloc(output_buffer_size_per_agg) : nullptr;
        out_vec_dev_buffers.push_back(out_vec_dev_buffer);
        if (shared_memory_size) {
          CHECK_EQ(output_buffer_size_per_agg, size_t(8));
          gpu_allocator_->copyToDevice(reinterpret_cast<int8_t*>(out_vec_dev_buffer),
                                       reinterpret_cast<const int8_t*>(&init_agg_vals[i]),
                                       output_buffer_size_per_agg);
        }
      }
    }
    auto out_vec_dev_ptr = gpu_allocator_->alloc(agg_col_count * sizeof(int8_t*));
    gpu_allocator_->copyToDevice(out_vec_dev_ptr,
                                 reinterpret_cast<int8_t*>(out_vec_dev_buffers.data()),
                                 agg_col_count * sizeof(int8_t*));
    kernel_params[GROUPBY_BUF] = out_vec_dev_ptr;
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }

    if (g_enable_dynamic_watchdog || (allow_runtime_interrupt && !render_allocator)) {
      auto prepareTime = prepareClock->stop();

      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: prepare: " << std::to_string(prepareTime) << " ms";
      launchClock->start();
    }

    if (hoist_literals) {
      VLOG(1) << "Launching(" << kernel->name() << ") on device_id(" << device_id << ')';
      kernel->launch(grid_size_x,
                     grid_size_y,
                     grid_size_z,
                     block_size_x,
                     block_size_y,
                     block_size_z,
                     shared_memory_size,
                     &param_ptrs[0],
                     optimize_cuda_block_and_grid_sizes);
    } else {
      param_ptrs.erase(param_ptrs.begin() + LITERALS);  // TODO(alex): remove
      VLOG(1) << "Launching(" << kernel->name() << ") on device_id(" << device_id << ')';
      kernel->launch(grid_size_x,
                     grid_size_y,
                     grid_size_z,
                     block_size_x,
                     block_size_y,
                     block_size_z,
                     shared_memory_size,
                     &param_ptrs[0],
                     optimize_cuda_block_and_grid_sizes);
    }

    if (g_enable_dynamic_watchdog || (allow_runtime_interrupt && !render_allocator)) {
      auto launchTime = launchClock->stop();
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: cuLaunchKernel: " << std::to_string(launchTime)
              << " ms";
      finishClock->start();
    }

    gpu_allocator_->copyFromDevice(
        &error_codes[0], err_desc, error_codes.size() * sizeof(error_codes[0]));
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
      int64_t* host_out_vec = new int64_t[output_buffer_size_per_agg];
      gpu_allocator_->copyFromDevice(
          host_out_vec, out_vec_dev_buffers[i], output_buffer_size_per_agg);
      out_vec.push_back(host_out_vec);
    }
  }
  const auto count_distinct_bitmap_mem = query_buffers_->getCountDistinctBitmapPtr();
  if (count_distinct_bitmap_mem) {
    gpu_allocator_->copyFromDevice(query_buffers_->getCountDistinctHostPtr(),
                                   reinterpret_cast<void*>(count_distinct_bitmap_mem),
                                   query_buffers_->getCountDistinctBitmapBytes());
  }

  const auto varlen_output_gpu_buf = query_buffers_->getVarlenOutputPtr();
  if (varlen_output_gpu_buf) {
    CHECK(query_mem_desc_.varlenOutputBufferElemSize());
    const size_t varlen_output_buf_bytes =
        query_mem_desc_.getEntryCount() *
        query_mem_desc_.varlenOutputBufferElemSize().value();
    CHECK(query_buffers_->getVarlenOutputHostPtr());
    gpu_allocator_->copyFromDevice(query_buffers_->getVarlenOutputHostPtr(),
                                   reinterpret_cast<void*>(varlen_output_gpu_buf),
                                   varlen_output_buf_bytes);
  }

  if (g_enable_dynamic_watchdog || (allow_runtime_interrupt && !render_allocator)) {
    if (allow_runtime_interrupt) {
      kernel->resetRuntimeInterrupter(device_id);
    }
    auto finishTime = finishClock->stop();
    VLOG(1) << "Device " << std::to_string(device_id)
            << ": launchGpuCode: finish: " << std::to_string(finishTime) << " ms";
  }

  return out_vec;
}

std::vector<int64_t*> QueryExecutionContext::launchCpuCode(
    const RelAlgExecutionUnit& ra_exe_unit,
    const CpuCompilationContext* native_code,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<std::vector<const int8_t*>> col_buffers,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const int32_t scan_limit,
    int32_t* error_code,
    const uint32_t start_rowid,
    const uint32_t num_tables,
    const std::vector<int8_t*>& join_hash_tables,
    const int64_t num_rows_to_process) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(lauchCpuCode);

  CHECK(query_buffers_);
  const auto& init_agg_vals = query_buffers_->init_agg_vals_;

  std::vector<const int8_t**> multifrag_col_buffers;
  for (auto& col_buffer : col_buffers) {
    multifrag_col_buffers.push_back(col_buffer.empty() ? nullptr : col_buffer.data());
  }
  const int8_t*** multifrag_cols_ptr{
      multifrag_col_buffers.empty() ? nullptr : &multifrag_col_buffers[0]};
  const uint32_t num_fragments =
      multifrag_cols_ptr ? static_cast<uint32_t>(col_buffers.size()) : uint32_t(0);
  const auto num_out_frags = multifrag_cols_ptr ? num_fragments : uint32_t(0);

  const bool is_group_by{query_mem_desc_.isGroupBy()};
  std::vector<int64_t*> out_vec;
  if (ra_exe_unit.estimator) {
    // Subfragments collect the result from multiple runs in a single
    // result set.
    if (!estimator_result_set_) {
      estimator_result_set_.reset(
          new ResultSet(ra_exe_unit.estimator, ExecutorDeviceType::CPU, 0, nullptr));
    }
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
  for (auto& nums : num_rows) {
    flatened_num_rows.insert(flatened_num_rows.end(), nums.begin(), nums.end());
  }
  std::vector<uint64_t> flatened_frag_offsets;
  for (auto& offsets : frag_offsets) {
    flatened_frag_offsets.insert(
        flatened_frag_offsets.end(), offsets.begin(), offsets.end());
  }
  const int64_t rowid_lookup_num_rows =
      start_rowid ? static_cast<int64_t>(start_rowid) + 1 : 0;
  int64_t const* num_rows_ptr;
  if (num_rows_to_process > 0) {
    flatened_num_rows[0] = num_rows_to_process;
    num_rows_ptr = flatened_num_rows.data();
  } else {
    num_rows_ptr =
        rowid_lookup_num_rows ? &rowid_lookup_num_rows : flatened_num_rows.data();
  }
  int32_t total_matched_init{0};

  std::vector<int64_t> cmpt_val_buff;
  if (is_group_by) {
    cmpt_val_buff =
        compact_init_vals(align_to_int64(query_mem_desc_.getColsSize()) / sizeof(int64_t),
                          init_agg_vals,
                          query_mem_desc_);
  }

  RowFunctionManager mgr(executor_, ra_exe_unit);
  int8_t* row_func_mgr_ptr = reinterpret_cast<int8_t*>(&mgr);

  CHECK(native_code);
  const int64_t* join_hash_tables_ptr =
      join_hash_tables.size() == 1
          ? reinterpret_cast<const int64_t*>(join_hash_tables[0])
          : (join_hash_tables.size() > 1
                 ? reinterpret_cast<const int64_t*>(&join_hash_tables[0])
                 : nullptr);
  VLOG(1) << "Calling " << native_code->name() << " hoist_literals(" << hoist_literals
          << ')';
  const int64_t* const init_agg_value =
      is_group_by ? cmpt_val_buff.data() : init_agg_vals.data();
  int64_t** const out =
      is_group_by ? query_buffers_->getGroupByBuffersPtr() : out_vec.data();
  if (hoist_literals) {
    native_code->call(
        error_code,           // int32_t*,         // error_code
        &total_matched_init,  // int32_t*,         // total_matched
        out,                  // int64_t**,        // out
        &num_fragments,       // const uint32_t*,  // num_fragments
        &num_tables,          // const uint32_t*,  // num_tables
        &start_rowid,         // const uint32_t*,  // start_rowid aka row_index_resume
        multifrag_cols_ptr,   // const int8_t***,  // col_buffers
        literal_buff.data(),  // const int8_t*,    // literals
        num_rows_ptr,         // const int64_t*,   // num_rows
        flatened_frag_offsets.data(),  // const uint64_t*,  // frag_row_offsets
        &scan_limit,                   // const int32_t*,   // max_matched
        init_agg_value,                // const int64_t*,   // init_agg_value
        join_hash_tables_ptr,          // const int64_t*,   // join_hash_tables_ptr
        row_func_mgr_ptr);             // const int8_t*);   // row_func_mgr
  } else {
    native_code->call(
        error_code,           // int32_t*,         // error_code
        &total_matched_init,  // int32_t*,         // total_matched
        out,                  // int64_t**,        // out
        &num_fragments,       // const uint32_t*,  // num_fragments
        &num_tables,          // const uint32_t*,  // num_tables
        &start_rowid,         // const uint32_t*,  // start_rowid aka row_index_resume
        multifrag_cols_ptr,   // const int8_t***,  // col_buffers
        num_rows_ptr,         // const int64_t*,   // num_rows
        flatened_frag_offsets.data(),  // const uint64_t*,  // frag_row_offsets
        &scan_limit,                   // const int32_t*,   // max_matched
        init_agg_value,                // const int64_t*,   // init_agg_value
        join_hash_tables_ptr,          // const int64_t*,   // join_hash_tables_ptr
        row_func_mgr_ptr);             // const int8_t*);   // row_func_mgr
  }

  if (ra_exe_unit.estimator) {
    return {};
  }

  if (rowid_lookup_num_rows && *error_code < 0) {
    *error_code = 0;
  }

  if (query_mem_desc_.useStreamingTopN()) {
    query_buffers_->applyStreamingTopNOffsetCpu(query_mem_desc_, ra_exe_unit);
  }

  if (query_mem_desc_.didOutputColumnar() &&
      query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    query_buffers_->compactProjectionBuffersCpu(query_mem_desc_, total_matched_init);
  }
  return out_vec;
}

size_t QueryExecutionContext::sizeofColBuffers(
    std::vector<std::vector<int8_t const*>> const& col_buffers) const {
  if (size_t const num_fragments = col_buffers.size()) {
    size_t const col_bytes = col_buffers.front().size() * sizeof(int8_t const*);
    // num_fragments pointers, each pointing to a chunk of size col_bytes
    return num_fragments * sizeof(int8_t*) + num_fragments * col_bytes;
  }
  return 0;
}
// Assumes all vectors in col_buffers are the same size.
// Copy 2d vector to device without flattening.
void QueryExecutionContext::copyColBuffersToDevice(
    int8_t* device_ptr,
    std::vector<std::vector<int8_t const*>> const& col_buffers) const {
  if (size_t const num_fragments = col_buffers.size()) {
    size_t const col_bytes = col_buffers.front().size() * sizeof(int8_t const*);
    int8_t* col_buffer_ptr = device_ptr + num_fragments * sizeof(int8_t*);
    // The code could be shorter w/ one for loop, but the memory access is linear w/ two.
    for (size_t i = 0; i < num_fragments; ++i) {
      gpu_allocator_->copyToDevice(device_ptr, &col_buffer_ptr, sizeof(int8_t*));
      device_ptr += sizeof(int8_t*);
      col_buffer_ptr += col_bytes;
    }
    col_buffer_ptr = device_ptr;
    for (size_t i = 0; i < num_fragments; ++i) {
      CHECK_EQ(col_buffers.front().size(), col_buffers[i].size()) << i;
      gpu_allocator_->copyToDevice(col_buffer_ptr, col_buffers[i].data(), col_bytes);
      col_buffer_ptr += col_bytes;
    }
  }
}

template <typename T>
size_t QueryExecutionContext::sizeofFlattened2dVec(
    uint32_t const expected_subvector_size,
    std::vector<std::vector<T>> const& vec2d) const {
  return expected_subvector_size * vec2d.size() * sizeof(T);
}
template <typename T>
void QueryExecutionContext::copyFlattened2dVecToDevice(
    int8_t* device_ptr,
    uint32_t const expected_subvector_size,
    std::vector<std::vector<T>> const& vec2d) const {
  size_t const bytes_per_subvector = expected_subvector_size * sizeof(T);
  for (size_t i = 0; i < vec2d.size(); ++i) {
    CHECK_EQ(expected_subvector_size, vec2d[i].size()) << i << '/' << vec2d.size();
    gpu_allocator_->copyToDevice(device_ptr, vec2d[i].data(), bytes_per_subvector);
    device_ptr += bytes_per_subvector;
  }
}

size_t QueryExecutionContext::sizeofInitAggVals(
    bool const is_group_by,
    std::vector<int64_t> const& init_agg_vals) const {
  if (is_group_by && !output_columnar_) {
    auto cmpt_sz = align_to<8>(query_mem_desc_.getColsSize()) / sizeof(int64_t);
    return cmpt_sz * sizeof(int64_t);
  } else {
    return init_agg_vals.size() * sizeof(int64_t);
  }
}
void QueryExecutionContext::copyInitAggValsToDevice(
    int8_t* device_ptr,
    bool const is_group_by,
    std::vector<int64_t> const& init_agg_vals) const {
  if (is_group_by && !output_columnar_) {
    auto cmpt_sz = align_to<8>(query_mem_desc_.getColsSize()) / sizeof(int64_t);
    auto cmpt_val_buff = compact_init_vals(cmpt_sz, init_agg_vals, query_mem_desc_);
    copyVectorToDevice(device_ptr, cmpt_val_buff);
  } else {
    copyVectorToDevice(device_ptr, init_agg_vals);
  }
}

size_t QueryExecutionContext::sizeofJoinHashTables(
    std::vector<int8_t*> const& join_hash_tables) const {
  return join_hash_tables.size() < 2u ? 0u : join_hash_tables.size() * sizeof(int8_t*);
}
int8_t* QueryExecutionContext::copyJoinHashTablesToDevice(
    int8_t* device_ptr,
    std::vector<int8_t*> const& join_hash_tables) const {
  switch (join_hash_tables.size()) {
    case 0u:
      return nullptr;
    case 1u:
      return join_hash_tables[0];
    default:
      copyVectorToDevice(device_ptr, join_hash_tables);
      return device_ptr;
  }
}

size_t QueryExecutionContext::sizeofLiterals(
    std::vector<int8_t> const& literal_buff) const {
  size_t const count_distinct_bytes =
      query_buffers_->getCountDistinctBitmapPtr() ? 2 * sizeof(int64_t) : 0u;
  return count_distinct_bytes + literal_buff.size();
}
// The count_distinct_addresses are considered "additional literals"
// and are retrieved as negative offsets relative to the "literals" pointer
// via GroupByAndAggregate::getAdditionalLiteral().
int8_t* QueryExecutionContext::copyLiteralsToDevice(
    int8_t* device_ptr,
    std::vector<int8_t> const& literal_buff) const {
  // Calculate additional space
  //  * Count Distinct Bitmap
  int64_t count_distinct_addresses[2];
  size_t const count_distinct_bytes =
      query_buffers_->getCountDistinctBitmapPtr() ? sizeof count_distinct_addresses : 0u;
  CHECK_EQ(0u, uint64_t(device_ptr) % 8);
  // Copy to device, literals last.
  if (count_distinct_bytes) {
    // Store host and device addresses
    auto const count_distinct_bitmap_host_mem = query_buffers_->getCountDistinctHostPtr();
    CHECK(count_distinct_bitmap_host_mem);
    count_distinct_addresses[0] =  // getAdditionalLiteral(-2) in codegenCountDistinct()
        reinterpret_cast<int64_t>(count_distinct_bitmap_host_mem);
    count_distinct_addresses[1] =  // getAdditionalLiteral(-1) in codegenCountDistinct()
        static_cast<int64_t>(query_buffers_->getCountDistinctBitmapPtr());
    gpu_allocator_->copyToDevice(
        device_ptr, count_distinct_addresses, count_distinct_bytes);
    device_ptr += count_distinct_bytes;
  }
  if (!literal_buff.empty()) {
    gpu_allocator_->copyToDevice(device_ptr, literal_buff.data(), literal_buff.size());
  }
  return device_ptr;
}

template <typename T>
void QueryExecutionContext::copyValueToDevice(int8_t* device_ptr, T const value) const {
  gpu_allocator_->copyToDevice(device_ptr, &value, sizeof(T));
}

template <typename T>
size_t QueryExecutionContext::sizeofVector(std::vector<T> const& vec) const {
  return vec.size() * sizeof(T);
}
template <typename T>
void QueryExecutionContext::copyVectorToDevice(int8_t* device_ptr,
                                               std::vector<T> const& vec) const {
  gpu_allocator_->copyToDevice(device_ptr, vec.data(), vec.size() * sizeof(T));
}

QueryExecutionContext::KernelParams QueryExecutionContext::prepareKernelParams(
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<int8_t>& literal_buff,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const int32_t scan_limit,
    const std::vector<int64_t>& init_agg_vals,
    const std::vector<int32_t>& error_codes,
    const uint32_t num_tables,
    const std::vector<int8_t*>& join_hash_tables,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const bool hoist_literals,
    const bool is_group_by) const {
  CHECK(gpu_allocator_);
  CHECK(literal_buff.empty() || hoist_literals) << literal_buff.size();
  CHECK_EQ(num_rows.size(), col_buffers.size());
  CHECK_EQ(frag_offsets.size(), col_buffers.size());

  // All sizes are in number of bytes and divisible by 8 (int64_t-aligned).
  KernelParamSizes param_sizes;
  param_sizes[ERROR_CODE] = align_to<8>(sizeofVector(error_codes));
  param_sizes[TOTAL_MATCHED] = align_to<8>(sizeof(uint32_t));
  param_sizes[GROUPBY_BUF] = 0u;
  param_sizes[NUM_FRAGMENTS] = align_to<8>(sizeof(uint32_t));
  param_sizes[NUM_TABLES] = align_to<8>(sizeof(num_tables));
  param_sizes[ROW_INDEX_RESUME] = align_to<8>(sizeof(uint32_t));
  param_sizes[COL_BUFFERS] = sizeofColBuffers(col_buffers);
  param_sizes[LITERALS] = align_to<8>(sizeofLiterals(literal_buff));
  param_sizes[NUM_ROWS] = sizeofFlattened2dVec(num_tables, num_rows);
  param_sizes[FRAG_ROW_OFFSETS] = sizeofFlattened2dVec(num_tables, frag_offsets);
  param_sizes[MAX_MATCHED] = align_to<8>(sizeof(scan_limit));
  param_sizes[INIT_AGG_VALS] = sizeofInitAggVals(is_group_by, init_agg_vals);
  param_sizes[JOIN_HASH_TABLES] = sizeofJoinHashTables(join_hash_tables);
  param_sizes[ROW_FUNC_MGR] = 0u;
  auto const nbytes = std::accumulate(param_sizes.begin(), param_sizes.end(), size_t(0));

  KernelParams params;
  // Allocate one block for all kernel params and set pointers based on param_sizes.
  params[ERROR_CODE] = gpu_allocator_->alloc(nbytes);
  static_assert(ERROR_CODE == 0);
  for (size_t i = 1; i < params.size(); ++i) {
    params[i] = params[i - 1] + param_sizes[i - 1];
  }
  // Copy data to device based on params w/ adjustments to LITERALS and JOIN_HASH_TABLES.
  copyVectorToDevice(params[ERROR_CODE], error_codes);
  copyValueToDevice(params[TOTAL_MATCHED], int32_t(0));
  params[GROUPBY_BUF] = nullptr;
  copyValueToDevice(params[NUM_FRAGMENTS], uint32_t(col_buffers.size()));
  copyValueToDevice(params[NUM_TABLES], num_tables);
  copyValueToDevice(params[ROW_INDEX_RESUME], uint32_t(0));
  copyColBuffersToDevice(params[COL_BUFFERS], col_buffers);
  params[LITERALS] = copyLiteralsToDevice(params[LITERALS], literal_buff);
  copyFlattened2dVecToDevice(params[NUM_ROWS], num_tables, num_rows);
  copyFlattened2dVecToDevice(params[FRAG_ROW_OFFSETS], num_tables, frag_offsets);
  // Note that this will be overwritten if we are setting the entry count during group by
  // buffer allocation and initialization
  copyValueToDevice(params[MAX_MATCHED], scan_limit);
  copyInitAggValsToDevice(params[INIT_AGG_VALS], is_group_by, init_agg_vals);
  params[JOIN_HASH_TABLES] =
      copyJoinHashTablesToDevice(params[JOIN_HASH_TABLES], join_hash_tables);

  // RowFunctionManager is not supported in GPU. We just keep the argument
  // to avoid diverging from CPU generated code
  params[ROW_FUNC_MGR] = nullptr;

  return params;
}
