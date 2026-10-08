/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_GPUMEMUTILS_H
#define QUERYENGINE_GPUMEMUTILS_H

#include "CompilationOptions.h"
#include "ResultSetEntryFilter.h"
#include "Shared/TargetInfo.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include "../Shared/nocuda.h"
#endif  // HAVE_CUDA

namespace CudaMgr_Namespace {

class CudaMgr;

}  // namespace CudaMgr_Namespace

namespace Data_Namespace {

class AbstractBuffer;
class DataMgr;

}  // namespace Data_Namespace

struct GpuGroupByBuffers {
  int8_t* ptrs;  // ptrs for individual outputs
  int8_t* data;  // ptr to data allocation
  size_t entry_count;
  int8_t* varlen_output_buffer;
};

class QueryMemoryDescriptor;
class DeviceAllocator;
class Allocator;

inline constexpr size_t kMinGpuPerfectHashReductionInputBytes = size_t{8} << 20;

void copy_to_nvidia_gpu(Data_Namespace::DataMgr* data_mgr,
                        CUstream cuda_stream,
                        CUdeviceptr dst,
                        const void* src,
                        const size_t num_bytes,
                        const int device_id,
                        std::string_view tag);

GpuGroupByBuffers create_dev_group_by_buffers(
    DeviceAllocator* device_allocator,
    const std::vector<int64_t*>& group_by_buffers,
    const QueryMemoryDescriptor&,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const ExecutorDispatchMode dispatch_mode,
    const int64_t num_input_rows,
    const bool prepend_index_buffer,
    const bool always_init_group_by_on_host,
    const bool use_bump_allocator,
    const bool has_varlen_output,
    Allocator* insitu_allocator);

bool can_defer_keyless_perfect_hash_rowwise_result(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t entry_count);

std::optional<size_t> copy_group_by_buffers_from_gpu(
    DeviceAllocator& device_allocator,
    const std::vector<int64_t*>& group_by_buffers,
    const size_t groups_buffer_size,
    const int8_t* group_by_dev_buffers_mem,
    const QueryMemoryDescriptor& query_mem_desc,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    CUstream cuda_stream,
    const bool prepend_index_buffer,
    const bool has_varlen_output,
    const std::vector<TargetInfo>* target_infos = nullptr,
    const std::vector<int64_t>* init_vals = nullptr,
    int8_t** compacted_device_buffer = nullptr,
    bool skip_host_copy_for_compacted = false,
    bool skip_host_copy_for_perfect_hash = false,
    bool* host_copy_performed = nullptr,
    const ResultSetEntryFilter* entry_filter = nullptr,
    const std::vector<int64_t>* preserved_keys = nullptr,
    bool* entry_filter_applied = nullptr);

size_t get_num_allocated_rows_from_gpu(DeviceAllocator& device_allocator,
                                       int8_t* projection_size_gpu,
                                       const int device_id);

void copy_projection_buffer_from_gpu_columnar(DeviceAllocator* device_allocator,
                                              const GpuGroupByBuffers& gpu_query_buffers,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              int8_t* projection_buffer,
                                              const size_t projection_count,
                                              const int device_id);

#endif  // QUERYENGINE_GPUMEMUTILS_H
