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

#ifndef QUERYENGINE_GPUMEMUTILS_H
#define QUERYENGINE_GPUMEMUTILS_H

#include "CompilationOptions.h"
#include "Rendering/RenderAllocator.h"
#include "ThrustAllocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace CudaMgr_Namespace {

class CudaMgr;

}  // namespace CudaMgr_Namespace

namespace Data_Namespace {

class AbstractBuffer;
class DataMgr;

}  // namespace Data_Namespace

CUdeviceptr alloc_gpu_mem(Data_Namespace::DataMgr* data_mgr,
                          const size_t num_bytes,
                          const int device_id,
                          RenderAllocator* render_allocator);

Data_Namespace::AbstractBuffer* alloc_gpu_abstract_buffer(
    Data_Namespace::DataMgr* data_mgr,
    const size_t num_bytes,
    const int device_id);

void free_gpu_abstract_buffer(Data_Namespace::DataMgr* data_mgr,
                              Data_Namespace::AbstractBuffer* ab);

void copy_to_gpu(Data_Namespace::DataMgr* data_mgr,
                 CUdeviceptr dst,
                 const void* src,
                 const size_t num_bytes,
                 const int device_id);

void copy_from_gpu(Data_Namespace::DataMgr* data_mgr,
                   void* dst,
                   const CUdeviceptr src,
                   const size_t num_bytes,
                   const int device_id);

struct GpuQueryMemory {
  std::pair<CUdeviceptr, CUdeviceptr> group_by_buffers;
  std::pair<CUdeviceptr, CUdeviceptr> small_group_by_buffers;
};

using GpuGroupByBuffers = std::pair<CUdeviceptr, CUdeviceptr>;

class QueryMemoryDescriptor;
class CudaAllocator;

GpuGroupByBuffers create_dev_group_by_buffers(
    const CudaAllocator& cuda_allocator,
    const std::vector<int64_t*>& group_by_buffers,
    const QueryMemoryDescriptor&,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const bool prepend_index_buffer,
    const bool always_init_group_by_on_host,
    RenderAllocator* render_allocator);

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const std::vector<int64_t*>& group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const CUdeviceptr group_by_dev_buffers_mem,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id,
                                    const bool prepend_index_buffer);

size_t get_num_allocated_rows_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                       CUdeviceptr projection_size_gpu,
                                       const int device_id);

void copy_projection_buffer_from_gpu_columnar(Data_Namespace::DataMgr* data_mgr,
                                              const GpuQueryMemory& gpu_query_mem,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              int8_t* projection_buffer,
                                              const size_t projection_count,
                                              const int device_id);

#endif  // QUERYENGINE_GPUMEMUTILS_H
