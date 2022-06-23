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

#include "GpuMemUtils.h"
#include "DataMgr/Allocators/GpuAllocator.h"
#include "GpuInitGroups.h"
#include "Logger/Logger.h"
#include "StreamingTopN.h"

#include "../CudaMgr/CudaMgr.h"
#include "GroupByAndAggregate.h"

#include <vector>

extern double g_bump_allocator_step_reduction;

void copy_to_nvidia_gpu(Data_Namespace::DataMgr* data_mgr,

                        CUdeviceptr dst,

                        const void* src,

                        const size_t num_bytes,

                        const int device_id) {
#ifdef HAVE_CUDA
  if (!data_mgr) {  // only for unit tests
    cuMemcpyHtoD(dst, src, num_bytes);
    return;
  }
#endif  // HAVE_CUDA
  const auto cuda_mgr = data_mgr->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyHostToDevice(reinterpret_cast<int8_t*>(dst),
                             static_cast<const int8_t*>(src),
                             num_bytes,
                             device_id);
}

namespace {

inline size_t coalesced_size(const QueryMemoryDescriptor& query_mem_desc,
                             const size_t group_by_one_buffer_size,
                             const unsigned grid_size_x) {
  CHECK(query_mem_desc.threadsShareMemory());
  return grid_size_x * group_by_one_buffer_size;
}

}  // namespace

GpuGroupByBuffers create_dev_group_by_buffers(
    DeviceAllocator* cuda_allocator,
    const Config& config,
    const std::vector<int64_t*>& group_by_buffers,
    const QueryMemoryDescriptor& query_mem_desc,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const ExecutorDispatchMode dispatch_mode,
    const int64_t num_input_rows,
    const bool prepend_index_buffer,
    const bool always_init_group_by_on_host,
    const bool use_bump_allocator,
    const bool has_varlen_output,
    Allocator* insitu_allocator) {
  if (group_by_buffers.empty() && !insitu_allocator) {
    return {0, 0, 0, 0};
  }
  CHECK(cuda_allocator);

  size_t groups_buffer_size{0};
  CUdeviceptr group_by_dev_buffers_mem{0};
  size_t mem_size{0};
  size_t entry_count{0};

  if (use_bump_allocator) {
    CHECK(!prepend_index_buffer);
    CHECK(!insitu_allocator);

    if (dispatch_mode == ExecutorDispatchMode::KernelPerFragment) {
      // Allocate an output buffer equal to the size of the number of rows in the
      // fragment. The kernel per fragment path is only used for projections with lazy
      // fetched outputs. Therefore, the resulting output buffer should be relatively
      // narrow compared to the width of an input row, offsetting the larger allocation.

      CHECK_GT(num_input_rows, int64_t(0));
      entry_count = num_input_rows;
      groups_buffer_size =
          query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU, entry_count);
      mem_size = coalesced_size(query_mem_desc,
                                groups_buffer_size,
                                query_mem_desc.blocksShareMemory() ? 1 : grid_size_x);
      group_by_dev_buffers_mem =
          reinterpret_cast<CUdeviceptr>(cuda_allocator->alloc(mem_size));
    } else {
      // Attempt to allocate increasingly small buffers until we have less than 256B of
      // memory remaining on the device. This may have the side effect of evicting
      // memory allocated for previous queries. However, at current maximum slab sizes
      // (2GB) we expect these effects to be minimal.
      size_t max_memory_size{config.mem.gpu.max_memory_allocation_size};
      while (true) {
        entry_count = max_memory_size / query_mem_desc.getRowSize();
        groups_buffer_size =
            query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU, entry_count);

        try {
          mem_size = coalesced_size(query_mem_desc,
                                    groups_buffer_size,
                                    query_mem_desc.blocksShareMemory() ? 1 : grid_size_x);
          CHECK_LE(entry_count, std::numeric_limits<uint32_t>::max());

          group_by_dev_buffers_mem =
              reinterpret_cast<CUdeviceptr>(cuda_allocator->alloc(mem_size));
        } catch (const OutOfMemory& e) {
          LOG(WARNING) << e.what();
          max_memory_size = max_memory_size * g_bump_allocator_step_reduction;
          if (max_memory_size < config.mem.gpu.min_memory_allocation_size) {
            throw;
          }

          LOG(WARNING) << "Ran out of memory for projection query output. Retrying with "
                       << std::to_string(max_memory_size) << " bytes";

          continue;
        }
        break;
      }
    }
    LOG(INFO) << "Projection query allocation succeeded with " << groups_buffer_size
              << " bytes allocated (max entry count " << entry_count << ")";
  } else {
    entry_count = query_mem_desc.getEntryCount();
    CHECK_GT(entry_count, size_t(0));
    groups_buffer_size =
        query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU, entry_count);
    mem_size = coalesced_size(query_mem_desc,
                              groups_buffer_size,
                              query_mem_desc.blocksShareMemory() ? 1 : grid_size_x);
    const size_t prepended_buff_size{
        prepend_index_buffer ? align_to_int64(entry_count * sizeof(int32_t)) : 0};

    int8_t* group_by_dev_buffers_allocation{nullptr};
    if (insitu_allocator) {
      group_by_dev_buffers_allocation =
          insitu_allocator->alloc(mem_size + prepended_buff_size);
    } else {
      group_by_dev_buffers_allocation =
          cuda_allocator->alloc(mem_size + prepended_buff_size);
    }
    CHECK(group_by_dev_buffers_allocation);

    group_by_dev_buffers_mem =
        reinterpret_cast<CUdeviceptr>(group_by_dev_buffers_allocation) +
        prepended_buff_size;
  }
  CHECK_GT(groups_buffer_size, size_t(0));
  CHECK(group_by_dev_buffers_mem);

  CHECK(query_mem_desc.threadsShareMemory());
  const size_t step{block_size_x};

  if (!insitu_allocator && (always_init_group_by_on_host ||
                            !query_mem_desc.lazyInitGroups(ExecutorDeviceType::GPU))) {
    std::vector<int8_t> buff_to_gpu(mem_size);
    auto buff_to_gpu_ptr = buff_to_gpu.data();

    const size_t start = has_varlen_output ? 1 : 0;
    for (size_t i = start; i < group_by_buffers.size(); i += step) {
      memcpy(buff_to_gpu_ptr, group_by_buffers[i], groups_buffer_size);
      buff_to_gpu_ptr += groups_buffer_size;
    }
    cuda_allocator->copyToDevice(reinterpret_cast<int8_t*>(group_by_dev_buffers_mem),
                                 buff_to_gpu.data(),
                                 buff_to_gpu.size());
  }

  auto group_by_dev_buffer = group_by_dev_buffers_mem;

  const size_t num_ptrs =
      (block_size_x * grid_size_x) + (has_varlen_output ? size_t(1) : size_t(0));

  std::vector<CUdeviceptr> group_by_dev_buffers(num_ptrs);

  const size_t start_index = has_varlen_output ? 1 : 0;
  for (size_t i = start_index; i < num_ptrs; i += step) {
    for (size_t j = 0; j < step; ++j) {
      group_by_dev_buffers[i + j] = group_by_dev_buffer;
    }
    if (!query_mem_desc.blocksShareMemory()) {
      group_by_dev_buffer += groups_buffer_size;
    }
  }

  CUdeviceptr varlen_output_buffer{0};
  if (has_varlen_output) {
    const auto varlen_buffer_elem_size_opt = query_mem_desc.varlenOutputBufferElemSize();
    CHECK(varlen_buffer_elem_size_opt);  // TODO(adb): relax

    group_by_dev_buffers[0] = reinterpret_cast<CUdeviceptr>(cuda_allocator->alloc(
        query_mem_desc.getEntryCount() * varlen_buffer_elem_size_opt.value()));
    varlen_output_buffer = group_by_dev_buffers[0];
  }

  auto group_by_dev_ptr = cuda_allocator->alloc(num_ptrs * sizeof(CUdeviceptr));
  cuda_allocator->copyToDevice(group_by_dev_ptr,
                               reinterpret_cast<int8_t*>(group_by_dev_buffers.data()),
                               num_ptrs * sizeof(CUdeviceptr));

  return {reinterpret_cast<CUdeviceptr>(group_by_dev_ptr),
          group_by_dev_buffers_mem,
          entry_count,
          varlen_output_buffer};
}

void copy_group_by_buffers_from_gpu(BufferProvider* buffer_provider,
                                    const std::vector<int64_t*>& group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const CUdeviceptr group_by_dev_buffers_mem,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id,
                                    const bool prepend_index_buffer,
                                    const bool has_varlen_output) {
  if (group_by_buffers.empty()) {
    return;
  }
  const size_t first_group_buffer_idx = has_varlen_output ? 1 : 0;

  const unsigned block_buffer_count{query_mem_desc.blocksShareMemory() ? 1 : grid_size_x};
  if (block_buffer_count == 1 && !prepend_index_buffer) {
    CHECK_EQ(coalesced_size(query_mem_desc, groups_buffer_size, block_buffer_count),
             groups_buffer_size);
    buffer_provider->copyFromDevice(
        reinterpret_cast<int8_t*>(group_by_buffers[first_group_buffer_idx]),
        reinterpret_cast<const int8_t*>(group_by_dev_buffers_mem),
        groups_buffer_size,
        device_id);
    return;
  }
  const size_t index_buffer_sz{
      prepend_index_buffer ? query_mem_desc.getEntryCount() * sizeof(int64_t) : 0};
  std::vector<int8_t> buff_from_gpu(
      coalesced_size(query_mem_desc, groups_buffer_size, block_buffer_count) +
      index_buffer_sz);
  buffer_provider->copyFromDevice(
      reinterpret_cast<int8_t*>(&buff_from_gpu[0]),
      reinterpret_cast<const int8_t*>(group_by_dev_buffers_mem - index_buffer_sz),
      buff_from_gpu.size(),
      device_id);
  auto buff_from_gpu_ptr = &buff_from_gpu[0];
  for (size_t i = 0; i < block_buffer_count; ++i) {
    const size_t buffer_idx = (i * block_size_x) + first_group_buffer_idx;
    CHECK_LT(buffer_idx, group_by_buffers.size());
    memcpy(group_by_buffers[buffer_idx],
           buff_from_gpu_ptr,
           groups_buffer_size + index_buffer_sz);
    buff_from_gpu_ptr += groups_buffer_size;
  }
}

/**
 * Returns back total number of allocated rows per device (i.e., number of matched
 * elements in projections).
 *
 * TODO(Saman): revisit this for bump allocators
 */
size_t get_num_allocated_rows_from_gpu(BufferProvider* buffer_provider,
                                       CUdeviceptr projection_size_gpu,
                                       const int device_id) {
  int32_t num_rows{0};
  buffer_provider->copyFromDevice(reinterpret_cast<int8_t*>(&num_rows),
                                  reinterpret_cast<const int8_t*>(projection_size_gpu),
                                  sizeof(num_rows),
                                  device_id);
  CHECK(num_rows >= 0);
  return static_cast<size_t>(num_rows);
}

/**
 * For projection queries we only copy back as many elements as necessary, not the whole
 * output buffer. The goal is to be able to build a compact ResultSet, particularly useful
 * for columnar outputs.
 *
 * NOTE: Saman: we should revisit this function when we have a bump allocator
 */
void copy_projection_buffer_from_gpu_columnar(
    BufferProvider* buffer_provider,
    const GpuGroupByBuffers& gpu_group_by_buffers,
    const QueryMemoryDescriptor& query_mem_desc,
    int8_t* projection_buffer,
    const size_t projection_count,
    const int device_id) {
  CHECK(query_mem_desc.didOutputColumnar());
  CHECK(query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection);
  constexpr size_t row_index_width = sizeof(int64_t);
  // copy all the row indices back to the host
  buffer_provider->copyFromDevice(
      projection_buffer,
      reinterpret_cast<const int8_t*>(gpu_group_by_buffers.data),
      projection_count * row_index_width,
      device_id);
  size_t buffer_offset_cpu{projection_count * row_index_width};
  // other columns are actual non-lazy columns for the projection:
  for (size_t i = 0; i < query_mem_desc.getSlotCount(); i++) {
    if (query_mem_desc.getPaddedSlotWidthBytes(i) > 0) {
      const auto column_proj_size =
          projection_count * query_mem_desc.getPaddedSlotWidthBytes(i);
      buffer_provider->copyFromDevice(
          projection_buffer + buffer_offset_cpu,
          reinterpret_cast<const int8_t*>(gpu_group_by_buffers.data +
                                          query_mem_desc.getColOffInBytes(i)),
          column_proj_size,
          device_id);
      buffer_offset_cpu += align_to_int64(column_proj_size);
    }
  }
}
