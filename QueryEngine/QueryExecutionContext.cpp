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

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

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
  if (device_type == ExecutorDeviceType::GPU) {
    device_allocator_ = executor->getCudaAllocator(device_id);
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
                                                            device_allocator_,
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
    const CompilationResult& compilation_result,
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
    int32_t* error_code,
    const uint32_t num_tables,
    const bool allow_runtime_interrupt,
    const std::vector<int8_t*>& join_hash_tables,
    RenderAllocatorMap* render_allocator_map,
    bool optimize_cuda_block_and_grid_sizes) {
  auto timer = DEBUG_TIMER(__func__);
  INJECT_TIMER(lauchGpuCode);
  CHECK(device_allocator_);
  CHECK(query_buffers_);
  CompilationContext const* compilation_context = compilation_result.generated_code.get();
  CHECK(compilation_context);
  const auto& init_agg_vals = query_buffers_->init_agg_vals_;
  const size_t shared_memory_size =
      compilation_result.gpu_smem_context.getSharedMemorySize();
  bool is_group_by{query_mem_desc_.isGroupBy()};
  std::vector<int64_t*> out_vec;
  uint32_t num_fragments = col_buffers.size();
  std::vector<int32_t> error_codes(grid_size_x * block_size_x);

  auto prepare_start = timer_start();
  auto query_resultset_copy_start = timer_start();

  RenderAllocator* render_allocator = nullptr;
  if (render_allocator_map) {
    render_allocator = render_allocator_map->getRenderAllocator(device_id);
  }
  // cast device_allocator to get cuda stream of the current device
  auto cuda_allocator = dynamic_cast<CudaAllocator*>(device_allocator_);
  CHECK(cuda_allocator);
  CHECK_EQ(cuda_allocator->getDeviceId(), device_id);
  auto cuda_stream = executor_->getCudaStream(device_id);
  auto kernel = create_device_kernel(compilation_context, device_id, cuda_stream);
#if HAVE_CUDA
  auto nvidia_kernel = dynamic_cast<NvidiaKernel*>(kernel.get());
  CHECK(nvidia_kernel);
  auto cuda_mgr = executor_->getDataMgr()->getCudaMgr();
  CHECK(cuda_mgr);
  if (g_enable_dynamic_watchdog) {
    executor_->initializeDynamicWatchdog(
        nvidia_kernel->getModulePtr(),
        device_id,
        cuda_stream,
        executor_->interrupted_.load(),
        executor_->deviceCycles(g_dynamic_watchdog_time_limit),
        g_dynamic_watchdog_time_limit);
  }
  if (allow_runtime_interrupt && !render_allocator) {
    executor_->initializeRuntimeInterrupter(
        nvidia_kernel->getModulePtr(), device_id, cuda_stream);
  }
#endif
  auto [kernel_params, kernel_params_log] = prepareKernelParams(col_buffers,
                                                                literal_buff,
                                                                num_rows,
                                                                frag_offsets,
                                                                scan_limit,
                                                                init_agg_vals,
                                                                error_codes,
                                                                num_tables,
                                                                join_hash_tables,
                                                                hoist_literals,
                                                                is_group_by);

  using KP = heavyai::KernelParam;
  CHECK_EQ(size_t(KP::N_), kernel_params.size());
  CHECK(!kernel_params[int(KP::GROUPBY_BUF)]);

  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  const unsigned grid_size_y = 1;
  const unsigned grid_size_z = 1;
  const auto total_thread_count = block_size_x * grid_size_x;
  const auto err_desc = kernel_params[int(KP::ERROR_CODE)];
  if (is_group_by) {
    CHECK(!(query_buffers_->getGroupByBuffersSize() == 0) || render_allocator);
    bool can_sort_on_gpu = query_mem_desc_.sortOnGpu();
    auto gpu_group_by_buffers = query_buffers_->createAndInitializeGroupByBufferGpu(
        ra_exe_unit,
        query_mem_desc_,
        kernel_params[int(KP::INIT_AGG_VALS)],
        device_id,
        cuda_stream,
        dispatch_mode_,
        block_size_x,
        grid_size_x,
        executor_->warpSize(),
        can_sort_on_gpu,
        output_columnar_,
        render_allocator);
    const auto max_matched = static_cast<int32_t>(gpu_group_by_buffers.entry_count);
    device_allocator_->copyToDevice(kernel_params[int(KP::MAX_MATCHED)],
                                    &max_matched,
                                    sizeof(max_matched),
                                    "GPU kernel param[MAX_MATCHED]");

    kernel_params[int(KP::GROUPBY_BUF)] = gpu_group_by_buffers.ptrs;
    kernel_params_log.ptrs[int(KP::GROUPBY_BUF)] = gpu_group_by_buffers.ptrs;
    kernel_params_log.values[int(KP::GROUPBY_BUF)] =
        KernelParamsLog::NamedSize{"EntryCount", gpu_group_by_buffers.entry_count};
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }
    VLOG(1) << "Device " << device_id
            << ": launchGpuCode: prepare query execution: " << timer_stop(prepare_start)
            << " ms";
    auto kernel_start = timer_start();
    if (hoist_literals) {
      VLOG(1) << "Launching(" << kernel->name() << ") on device_id(" << device_id << ')';
      kernel->launch(grid_size_x,
                     grid_size_y,
                     grid_size_z,
                     block_size_x,
                     block_size_y,
                     block_size_z,
                     compilation_result,
                     &param_ptrs[0],
                     kernel_params_log,
                     optimize_cuda_block_and_grid_sizes);
    } else {
      param_ptrs.erase(param_ptrs.begin() + int(KP::LITERALS));  // TODO(alex): remove
      VLOG(1) << "Launching(" << kernel->name() << ") on device_id(" << device_id << ')';
      kernel->launch(grid_size_x,
                     grid_size_y,
                     grid_size_z,
                     block_size_x,
                     block_size_y,
                     block_size_z,
                     compilation_result,
                     &param_ptrs[0],
                     kernel_params_log,
                     optimize_cuda_block_and_grid_sizes);
    }
    VLOG(1) << "Device " << device_id
            << ": launchGpuCode: query execution: " << timer_stop(kernel_start) << " ms";
    query_resultset_copy_start = timer_start();
    device_allocator_->copyFromDevice(reinterpret_cast<int8_t*>(error_codes.data()),
                                      reinterpret_cast<int8_t*>(err_desc),
                                      error_codes.size() * sizeof(error_codes[0]),
                                      "Query error code buffer");
    *error_code = aggregate_error_codes(error_codes);
    if (*error_code > 0) {
      return {};
    }

    if (!render_allocator) {
      if (query_mem_desc_.useStreamingTopN()) {
        query_buffers_->applyStreamingTopNOffsetGpu(data_mgr,
                                                    cuda_allocator,
                                                    query_mem_desc_,
                                                    gpu_group_by_buffers,
                                                    ra_exe_unit,
                                                    total_thread_count,
                                                    device_id,
                                                    executor_->getCudaStream(device_id));
      } else {
        if (use_speculative_top_n(ra_exe_unit, query_mem_desc_)) {
          try {
            inplace_sort_gpu(ra_exe_unit.sort_info.order_entries,
                             query_mem_desc_,
                             gpu_group_by_buffers,
                             cuda_allocator,
                             executor_->getCudaStream(device_id));
          } catch (const std::bad_alloc&) {
            throw SpeculativeTopNFailed("Failed during in-place GPU sort.");
          }
        }
        if (query_mem_desc_.getQueryDescriptionType() ==
            QueryDescriptionType::Projection) {
          if (query_mem_desc_.didOutputColumnar()) {
            query_buffers_->compactProjectionBuffersGpu(
                query_mem_desc_,
                cuda_allocator,
                gpu_group_by_buffers,
                get_num_allocated_rows_from_gpu(
                    *device_allocator_, kernel_params[int(KP::TOTAL_MATCHED)], device_id),
                device_id);
          } else {
            size_t num_allocated_rows{0};
            if (ra_exe_unit.use_bump_allocator) {
              num_allocated_rows = get_num_allocated_rows_from_gpu(
                  *device_allocator_, kernel_params[int(KP::TOTAL_MATCHED)], device_id);
              // First, check the error code. If we ran out of slots, don't copy data back
              // into the ResultSet or update ResultSet entry count
              if (*error_code < 0) {
                return {};
              }
            }
            query_buffers_->copyGroupByBuffersFromGpu(
                *device_allocator_,
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
              *device_allocator_,
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
      estimator_result_set_.reset(
          new ResultSet(ra_exe_unit.estimator,
                        ExecutorDeviceType::GPU,
                        device_id,
                        data_mgr,
                        executor_->getCudaAllocatorShared(device_id)));
      out_vec_dev_buffers.push_back(estimator_result_set_->getDeviceEstimatorBuffer());
    } else {
      for (size_t i = 0; i < agg_col_count; ++i) {
        int8_t* out_vec_dev_buffer = nullptr;
        if (num_fragments) {
          out_vec_dev_buffer = device_allocator_->alloc(output_buffer_size_per_agg);
        }
        out_vec_dev_buffers.push_back(out_vec_dev_buffer);
        if (shared_memory_size) {
          CHECK_EQ(output_buffer_size_per_agg, size_t(8));
          device_allocator_->copyToDevice(
              reinterpret_cast<int8_t*>(out_vec_dev_buffer),
              reinterpret_cast<const int8_t*>(&init_agg_vals[i]),
              output_buffer_size_per_agg,
              "Query metadata: output_buffer_size_per_agg");
        }
      }
    }
    auto const out_vec_buf_size = agg_col_count * sizeof(int8_t*);
    auto out_vec_dev_ptr = device_allocator_->alloc(out_vec_buf_size);
    device_allocator_->copyToDevice(out_vec_dev_ptr,
                                    reinterpret_cast<int8_t*>(out_vec_dev_buffers.data()),
                                    out_vec_buf_size,
                                    "Query output buffer ptrs");
    kernel_params[int(KP::GROUPBY_BUF)] = out_vec_dev_ptr;
    kernel_params_log.ptrs[int(KP::GROUPBY_BUF)] = out_vec_dev_ptr;
    kernel_params_log.values[int(KP::GROUPBY_BUF)] =
        KernelParamsLog::NamedSize{"AggColCount", agg_col_count};
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }

    VLOG(1) << "Device " << device_id
            << ": launchGpuCode: prepare query execution: " << timer_stop(prepare_start)
            << " ms";
    auto kernel_start = timer_start();
    if (hoist_literals) {
      VLOG(1) << "Launching(" << kernel->name() << ") on device_id(" << device_id << ')';
      kernel->launch(grid_size_x,
                     grid_size_y,
                     grid_size_z,
                     block_size_x,
                     block_size_y,
                     block_size_z,
                     compilation_result,
                     &param_ptrs[0],
                     kernel_params_log,
                     optimize_cuda_block_and_grid_sizes);
    } else {
      param_ptrs.erase(param_ptrs.begin() + int(KP::LITERALS));  // TODO(alex): remove
      VLOG(1) << "Launching(" << kernel->name() << ") on device_id(" << device_id << ')';
      kernel->launch(grid_size_x,
                     grid_size_y,
                     grid_size_z,
                     block_size_x,
                     block_size_y,
                     block_size_z,
                     compilation_result,
                     &param_ptrs[0],
                     kernel_params_log,
                     optimize_cuda_block_and_grid_sizes);
    }

    VLOG(1) << "Device " << device_id
            << ": launchGpuCode: query execution: " << timer_stop(kernel_start) << " ms";
    query_resultset_copy_start = timer_start();
    device_allocator_->copyFromDevice(&error_codes[0],
                                      err_desc,
                                      error_codes.size() * sizeof(error_codes[0]),
                                      "Query error code buffer");
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
      device_allocator_->copyFromDevice(host_out_vec,
                                        out_vec_dev_buffers[i],
                                        output_buffer_size_per_agg,
                                        "Query output buffer device ptrs");
      out_vec.push_back(host_out_vec);
    }
  }
  const auto count_distinct_bitmap_device_mem =
      query_buffers_->getCountDistinctBitmapDevicePtr();
  if (count_distinct_bitmap_device_mem) {
    device_allocator_->copyFromDevice(
        query_buffers_->getCountDistinctBitmapHostPtr(),
        reinterpret_cast<void*>(count_distinct_bitmap_device_mem),
        query_buffers_->getCountDistinctBitmapBytes(),
        "Count-distinct bitmap");
  }
  query_buffers_->copyFromDeviceForAggMode();

  const auto varlen_output_gpu_buf = query_buffers_->getVarlenOutputPtr();
  if (varlen_output_gpu_buf) {
    CHECK(query_mem_desc_.varlenOutputBufferElemSize());
    const size_t varlen_output_buf_bytes =
        query_mem_desc_.getEntryCount() *
        query_mem_desc_.varlenOutputBufferElemSize().value();
    CHECK(query_buffers_->getVarlenOutputHostPtr());
    device_allocator_->copyFromDevice(query_buffers_->getVarlenOutputHostPtr(),
                                      reinterpret_cast<void*>(varlen_output_gpu_buf),
                                      varlen_output_buf_bytes,
                                      "Varlen output buffer");
  }
  VLOG(1) << "Device " << device_id << ": launchGpuCode: copy query resultset: "
          << timer_stop(query_resultset_copy_start) << " ms";
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
      estimator_result_set_.reset(new ResultSet(ra_exe_unit.estimator,
                                                ExecutorDeviceType::CPU,
                                                0 /*=device_id*/,
                                                nullptr /*=data_mgr*/,
                                                nullptr /*=cuda_allocator*/));
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
// n = num_fragments
// m = col_buffers.front().size()
// Memory layout for column buffers:
// d_ptr           -> d_ptr + n*8
// d_ptr +     1*8 -> d_ptr + n 8 +     1*m*8
// d_ptr +     2*8 -> d_ptr + n*8 +     2*m*8
// d_ptr +     3*8 -> d_ptr + n*8 +     3*m*8
// ...
// d_ptr + (n-1)*8 -> d_ptr + n*8 + (n-1)*m*8
//
// d_ptr + n*8             -> col_buffer[0] (size m*8 bytes)
// d_ptr + n*8 +     1*m*8 -> col_buffer[1] (size m*8 bytes)
// d_ptr + n*8 +     2*m*8 -> col_buffer[2] (size m*8 bytes)
// d_ptr + n*8 +     3*m*8 -> col_buffer[3] (size m*8 bytes)
// ...
// d_ptr + n*8 + (n-1)*m*8 -> col_buffer[n-1] (size m*8 bytes)
//
// In other words, two arrays are allocated and set:
//  * n pointers pointing to the col_buffers.
//  * n col_buffers.
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
      device_allocator_->copyToDevice(
          device_ptr, &col_buffer_ptr, sizeof(int8_t*), "Input column buffer ptr");
      device_ptr += sizeof(int8_t*);
      col_buffer_ptr += col_bytes;
    }
    col_buffer_ptr = device_ptr;
    for (size_t i = 0; i < num_fragments; ++i) {
      CHECK_EQ(col_buffers.front().size(), col_buffers[i].size()) << i;
      device_allocator_->copyToDevice(
          col_buffer_ptr, col_buffers[i].data(), col_bytes, "Input column buffer data");
      col_buffer_ptr += col_bytes;
    }
  }
}

namespace {
KernelParamsLog::ColBuffer col_buffers_log(
    int8_t const* const d_ptr,
    std::vector<std::vector<int8_t const*>> const& col_buffers) {
  KernelParamsLog::ColBuffer cb{};
  if (size_t const n = col_buffers.size()) {
    size_t const m = col_buffers.front().size();
    cb.ptrs.reserve(n);
    cb.size = m;
    for (size_t i = 0; i < n; ++i) {
      cb.ptrs.push_back(static_cast<void const*>(d_ptr + (n + i * m) * sizeof(int8_t*)));
    }
  }
  return cb;
}
}  // namespace

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
    std::vector<std::vector<T>> const& vec2d,
    std::string_view tag) const {
  size_t const bytes_per_subvector = expected_subvector_size * sizeof(T);
  for (size_t i = 0; i < vec2d.size(); ++i) {
    CHECK_EQ(expected_subvector_size, vec2d[i].size()) << i << '/' << vec2d.size();
    device_allocator_->copyToDevice(
        device_ptr, vec2d[i].data(), bytes_per_subvector, tag);
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
    copyVectorToDevice(device_ptr, cmpt_val_buff, "Compact init agg values");
  } else if (init_agg_vals.size()) {
    copyVectorToDevice(device_ptr, init_agg_vals, "Init agg values");
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
      copyVectorToDevice(device_ptr, join_hash_tables, "Join hashtable(s) ptr");
      return device_ptr;
  }
}

namespace {
// If there is 1 element, return its address. If more, then return the cardinality.
// The multiple addresses can be inferred from the device address.
KernelParamsLog::Value join_hash_tables_log(
    std::vector<int8_t*> const& join_hash_tables) {
  if (join_hash_tables.empty()) {
    return nullptr;
  }
  std::vector<void const*> ptrs;
  ptrs.reserve(join_hash_tables.size());
  std::transform(join_hash_tables.begin(),
                 join_hash_tables.end(),
                 std::back_inserter(ptrs),
                 [](int8_t* ptr) { return static_cast<void const*>(ptr); });
  return ptrs;
}

using AggModeAddrs = int64_t[1];
using CountDistinctAddrs = int64_t[2];
}  // namespace

size_t QueryExecutionContext::sizeofLiterals(
    std::vector<int8_t> const& literal_buff) const {
  // If substantially more "additional literals" are added in the future,
  // consider a way that only allocates memory when it is needed.
  // It will have to work w/ calls to getAdditionalLiteral(i) in codegen.
  return sizeof(AggModeAddrs) + sizeof(CountDistinctAddrs) + literal_buff.size();
}

// if device_ptr==nullptr then just return size in bytes without copying.
size_t QueryExecutionContext::copyAggModeLiteralsToDevice(int8_t* device_ptr) const {
  AggModeAddrs agg_mode_addrs;
  std::vector<int8_t> const serialized_hash_tables =
      query_buffers_->getAggModeHashTablesGpu();
  if (size_t const size = serialized_hash_tables.size()) {
    CHECK_EQ(0u, size % 8u) << size;
    // These are container objects to the hash tables (~128 bytes each) not the content.
    int8_t* const hash_tables_gpu = device_allocator_->alloc(size);
    device_allocator_->copyToDevice(hash_tables_gpu,
                                    serialized_hash_tables.data(),
                                    size,
                                    "Hashtable container for agg. mode");
    // getAdditionalLiteral(-3) in codegenMode()
    agg_mode_addrs[0] = reinterpret_cast<int64_t>(hash_tables_gpu);
    device_allocator_->copyToDevice(device_ptr,
                                    agg_mode_addrs,
                                    sizeof agg_mode_addrs,
                                    "Hashtable ptrs for agg. mode");
  }
  return sizeof agg_mode_addrs;
}

// if device_ptr==nullptr then just return size in bytes without copying.
size_t QueryExecutionContext::copyCountDistinctBitmapLiteralsToDevice(
    int8_t* device_ptr) const {
  CountDistinctAddrs count_distinct_addrs;
  if (query_buffers_->getCountDistinctBitmapDevicePtr()) {
    // Store host and device addresses
    // getAdditionalLiteral(-2) and getAdditionalLiteral(-1) in codegenCountDistinct()
    count_distinct_addrs[0] = int64_t(query_buffers_->getCountDistinctBitmapHostPtr());
    count_distinct_addrs[1] = int64_t(query_buffers_->getCountDistinctBitmapDevicePtr());
    CHECK(count_distinct_addrs[0]);
    device_allocator_->copyToDevice(device_ptr,
                                    count_distinct_addrs,
                                    sizeof count_distinct_addrs,
                                    "Count-distinct bitmap ptrs");
  }
  return sizeof count_distinct_addrs;
}

// The AggModeLiterals and CountDistinctBitmapLiterals are considered
// "additional literals" and are retrieved as negative offsets relative to the
// "literals" pointer via GroupByAndAggregate::getAdditionalLiteral().
int8_t* QueryExecutionContext::copyLiteralsToDevice(
    int8_t* device_ptr,
    std::vector<int8_t> const& literal_buff) const {
  CHECK_EQ(0u, uint64_t(device_ptr) % 8);
  device_ptr += copyAggModeLiteralsToDevice(device_ptr);
  device_ptr += copyCountDistinctBitmapLiteralsToDevice(device_ptr);
  if (!literal_buff.empty()) {
    device_allocator_->copyToDevice(device_ptr,
                                    literal_buff.data(),
                                    literal_buff.size(),
                                    "params[LITERALS] and additional input literals");
  }
  return device_ptr;
}

template <typename T>
void QueryExecutionContext::copyValueToDevice(int8_t* device_ptr,
                                              T const value,
                                              std::string_view tag) const {
  device_allocator_->copyToDevice(device_ptr, &value, sizeof(T), tag);
}

template <typename T>
size_t QueryExecutionContext::sizeofVector(std::vector<T> const& vec) const {
  return vec.size() * sizeof(T);
}
template <typename T>
void QueryExecutionContext::copyVectorToDevice(int8_t* device_ptr,
                                               std::vector<T> const& vec,
                                               std::string_view tag) const {
  device_allocator_->copyToDevice(device_ptr, vec.data(), vec.size() * sizeof(T), tag);
}

std::pair<QueryExecutionContext::KernelParams, KernelParamsLog>
QueryExecutionContext::prepareKernelParams(
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<int8_t>& literal_buff,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const int32_t scan_limit,
    const std::vector<int64_t>& init_agg_vals,
    const std::vector<int32_t>& error_codes,
    const uint32_t num_tables,
    const std::vector<int8_t*>& join_hash_tables,
    const bool hoist_literals,
    const bool is_group_by) const {
  CHECK(device_allocator_);
  CHECK(literal_buff.empty() || hoist_literals) << literal_buff.size();
  CHECK_EQ(num_rows.size(), col_buffers.size());
  CHECK_EQ(frag_offsets.size(), col_buffers.size());
  KernelParamsLog params_log{};  // Log values to report in case of cuda error.
  params_log.hoist_literals = hoist_literals;

  // All sizes are in number of bytes and divisible by 8 (int64_t-aligned).
  using KP = heavyai::KernelParam;
  KernelParamSizes param_sizes;
  param_sizes[int(KP::ERROR_CODE)] = align_to<8>(sizeofVector(error_codes));
  param_sizes[int(KP::TOTAL_MATCHED)] = align_to<8>(sizeof(uint32_t));
  param_sizes[int(KP::GROUPBY_BUF)] = 0u;
  param_sizes[int(KP::NUM_FRAGMENTS)] = align_to<8>(sizeof(uint32_t));
  param_sizes[int(KP::NUM_TABLES)] = align_to<8>(sizeof(num_tables));
  param_sizes[int(KP::ROW_INDEX_RESUME)] = align_to<8>(sizeof(uint32_t));
  param_sizes[int(KP::COL_BUFFERS)] = sizeofColBuffers(col_buffers);
  param_sizes[int(KP::LITERALS)] = align_to<8>(sizeofLiterals(literal_buff));
  param_sizes[int(KP::NUM_ROWS)] = sizeofFlattened2dVec(num_tables, num_rows);
  param_sizes[int(KP::FRAG_ROW_OFFSETS)] = sizeofFlattened2dVec(num_tables, frag_offsets);
  param_sizes[int(KP::MAX_MATCHED)] = align_to<8>(sizeof(scan_limit));
  param_sizes[int(KP::INIT_AGG_VALS)] = sizeofInitAggVals(is_group_by, init_agg_vals);
  param_sizes[int(KP::JOIN_HASH_TABLES)] = sizeofJoinHashTables(join_hash_tables);
  param_sizes[int(KP::ROW_FUNC_MGR)] = 0u;
  auto const nbytes = std::accumulate(param_sizes.begin(), param_sizes.end(), size_t(0));

  KernelParams params;
  // Allocate one block for all kernel params and set pointers based on param_sizes.
  VLOG(1) << "Prepare GPU kernel parameters: " << nbytes << " bytes";
  params[int(KP::ERROR_CODE)] = device_allocator_->alloc(nbytes);
  params_log.ptrs[int(KP::ERROR_CODE)] =
      static_cast<void const*>(params[int(KP::ERROR_CODE)]);
  static_assert(int(KP::ERROR_CODE) == 0);

  for (size_t i = 1; i < params.size(); ++i) {
    params[i] = params[i - 1] + param_sizes[i - 1];
    params_log.ptrs[i] = static_cast<void const*>(params[i]);
  }

  using KPL = KernelParamsLog;

  // Copy data to device based on params w/ adjustments to LITERALS and JOIN_HASH_TABLES.
  copyVectorToDevice(params[int(KP::ERROR_CODE)], error_codes, "params[ERROR_CODE]");
  params_log.values[int(KP::ERROR_CODE)] = KPL::NamedSize{"Size", error_codes.size()};

  copyValueToDevice(params[int(KP::TOTAL_MATCHED)], int32_t(0), "params[TOTAL_MATCHED]");
  params_log.values[int(KP::TOTAL_MATCHED)] = KPL::NamedSize{"Value", size_t(int32_t(0))};

  params[int(KP::GROUPBY_BUF)] = nullptr;
  params_log.ptrs[int(KP::GROUPBY_BUF)] = nullptr;

  copyValueToDevice(params[int(KP::NUM_FRAGMENTS)],
                    uint32_t(col_buffers.size()),
                    "params[NUM_FRAGMENTS]");
  params_log.values[int(KP::NUM_FRAGMENTS)] = KPL::NamedSize{"Value", col_buffers.size()};

  copyValueToDevice(params[int(KP::NUM_TABLES)], num_tables, "params[NUM_TABLES]");
  params_log.values[int(KP::NUM_TABLES)] = KPL::NamedSize{"Value", num_tables};

  copyValueToDevice(
      params[int(KP::ROW_INDEX_RESUME)], uint32_t(0), "params[ROW_INDEX_RESUME]");
  params_log.values[int(KP::ROW_INDEX_RESUME)] = KPL::NamedSize{"Value", uint32_t(0)};

  copyColBuffersToDevice(params[int(KP::COL_BUFFERS)], col_buffers);
  params_log.values[int(KP::COL_BUFFERS)] =
      col_buffers_log(params[int(KP::COL_BUFFERS)], col_buffers);

  params[int(KP::LITERALS)] =
      copyLiteralsToDevice(params[int(KP::LITERALS)], literal_buff);
  params_log.ptrs[int(KP::LITERALS)] =
      static_cast<void const*>(params[int(KP::LITERALS)]);
  params_log.values[int(KP::LITERALS)] =
      KPL::NamedSize{"Size", sizeofLiterals(literal_buff)};

  copyFlattened2dVecToDevice(
      params[int(KP::NUM_ROWS)], num_tables, num_rows, "params[NUM_ROWS]");
  params_log.values[int(KP::NUM_ROWS)] = num_rows;

  copyFlattened2dVecToDevice(params[int(KP::FRAG_ROW_OFFSETS)],
                             num_tables,
                             frag_offsets,
                             "params[FRAG_ROW_OFFSETS]");
  params_log.values[int(KP::FRAG_ROW_OFFSETS)] = frag_offsets;

  // Note that this will be overwritten if we are setting the entry count during group by
  // buffer allocation and initialization
  copyValueToDevice(params[int(KP::MAX_MATCHED)], scan_limit, "params[MAX_MATCHED]");
  params_log.values[int(KP::MAX_MATCHED)] =
      KPL::NamedSize{"ScanLimit", size_t(scan_limit)};

  copyInitAggValsToDevice(params[int(KP::INIT_AGG_VALS)], is_group_by, init_agg_vals);
  params_log.values[int(KP::INIT_AGG_VALS)] =
      KPL::NamedSize{"Size", param_sizes[int(KP::INIT_AGG_VALS)] / sizeof(int64_t)};

  params[int(KP::JOIN_HASH_TABLES)] =
      copyJoinHashTablesToDevice(params[int(KP::JOIN_HASH_TABLES)], join_hash_tables);
  params_log.ptrs[int(KP::JOIN_HASH_TABLES)] =
      static_cast<void const*>(params[int(KP::JOIN_HASH_TABLES)]);
  params_log.values[int(KP::JOIN_HASH_TABLES)] = join_hash_tables_log(join_hash_tables);

  // RowFunctionManager is not supported in GPU. We just keep the argument
  // to avoid diverging from CPU generated code
  params[int(KP::ROW_FUNC_MGR)] = nullptr;
  params_log.ptrs[int(KP::ROW_FUNC_MGR)] = nullptr;

  return std::make_pair(std::move(params), std::move(params_log));
}

namespace {
std::ostream& operator<<(std::ostream& os, KernelParamsLog::ColBuffer const& cb) {
  if (cb.ptrs.empty()) {
    os << "empty";
  } else {
    os << "NumFragments(" << cb.ptrs.size() << ") NumColumns(" << cb.size << ") "
       << shared::printContainer(cb.ptrs);
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, KernelParamsLog::NamedSize const ns) {
  return os << ns.name << '(' << ns.size << ')';
}

std::ostream& operator<<(std::ostream& os, std::vector<void const*> const& ptrs) {
  return os << ptrs.size() << ": " << shared::printContainer(ptrs);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<std::vector<T>> const& vec2d) {
  if (vec2d.empty()) {
    os << "empty";
  } else {
    os << vec2d.size() << " x " << vec2d.front().size() << ": "
       << shared::printContainer(vec2d);
  }
  return os;
}
}  // namespace

std::ostream& operator<<(std::ostream& os, KernelParamsLog const& kpl) {
  using KP = heavyai::KernelParam;
  constexpr unsigned w = 17u;          // constant width to align pointer addresses
  bool const c = !kpl.hoist_literals;  // correction to indices after excluded LITERALS
  os << "hoist_literals=" << kpl.hoist_literals << ", " << (int(KP::N_) - c)
     << " kernel parameters:" << std::left;
  // Iterate through the heavyai::KernelParam enums, possibly skipping LITERALS.
  for (int i = 0u; i < int(KP::N_); ++i) {
    if (kpl.hoist_literals || i != int(KP::LITERALS)) {
      os << '\n' << std::setw(w) << static_cast<KP>(i) << kpl.ptrs[i] << ' ';
      std::visit([&](auto const& val) { os << val; }, kpl.values[i]);
    }
  }
  return os << std::right;
}
