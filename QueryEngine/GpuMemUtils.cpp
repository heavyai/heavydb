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
#include "CudaAllocator.h"
#include "GpuInitGroups.h"
#include "StreamingTopN.h"

#include "../CudaMgr/CudaMgr.h"
#include "GroupByAndAggregate.h"

#include <glog/logging.h>

CUdeviceptr alloc_gpu_mem(Data_Namespace::DataMgr* data_mgr,
                          const size_t num_bytes,
                          const int device_id,
                          RenderAllocator* render_allocator) {
  if (render_allocator) {
    return reinterpret_cast<CUdeviceptr>(render_allocator->alloc(num_bytes));
  }
  OOM_TRACE_PUSH(+": device_id " + std::to_string(device_id) + ", num_bytes " +
                 std::to_string(num_bytes));
  auto ab = alloc_gpu_abstract_buffer(data_mgr, num_bytes, device_id);
  return reinterpret_cast<CUdeviceptr>(ab->getMemoryPtr());
}

Data_Namespace::AbstractBuffer* alloc_gpu_abstract_buffer(
    Data_Namespace::DataMgr* data_mgr,
    const size_t num_bytes,
    const int device_id) {
  auto ab = data_mgr->alloc(Data_Namespace::GPU_LEVEL, device_id, num_bytes);
  CHECK_EQ(ab->getPinCount(), 1);
  return ab;
}

void free_gpu_abstract_buffer(Data_Namespace::DataMgr* data_mgr,
                              Data_Namespace::AbstractBuffer* ab) {
  data_mgr->free(ab);
}

void copy_to_gpu(Data_Namespace::DataMgr* data_mgr,
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
    const CudaAllocator& cuda_allocator,
    const std::vector<int64_t*>& group_by_buffers,
    const QueryMemoryDescriptor& query_mem_desc,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const bool prepend_index_buffer,
    const bool always_init_group_by_on_host,
    RenderAllocator* render_allocator) {
  if (group_by_buffers.empty() && !render_allocator) {
    return std::make_pair(0, 0);
  }

  size_t groups_buffer_size{query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU)};

  CHECK_GT(groups_buffer_size, size_t(0));

  const size_t mem_size{
      coalesced_size(query_mem_desc,
                     groups_buffer_size,
                     query_mem_desc.blocksShareMemory() ? 1 : grid_size_x)};

  CHECK_LE(query_mem_desc.getEntryCount(), std::numeric_limits<uint32_t>::max());
  const size_t prepended_buff_size{
      prepend_index_buffer
          ? align_to_int64(query_mem_desc.getEntryCount() * sizeof(int32_t))
          : 0};

  auto group_by_dev_buffers_mem =
      cuda_allocator.alloc(mem_size + prepended_buff_size, device_id, render_allocator) +
      prepended_buff_size;
  if (query_mem_desc.getCompactByteWidth() < 8) {
    // TODO(miyu): Compaction assumes the base ptr to be aligned to int64_t, otherwise
    // offsetting is wrong.
    //            Remove this assumption amd make offsetting work in all cases.
    CHECK_EQ(uint64_t(0),
             static_cast<int64_t>(group_by_dev_buffers_mem) % sizeof(int64_t));
  }
  CHECK(query_mem_desc.threadsShareMemory());
  const size_t step{block_size_x};

  if (!render_allocator && (always_init_group_by_on_host ||
                            !query_mem_desc.lazyInitGroups(ExecutorDeviceType::GPU))) {
    std::vector<int8_t> buff_to_gpu(mem_size);
    auto buff_to_gpu_ptr = &buff_to_gpu[0];

    for (size_t i = 0; i < group_by_buffers.size(); i += step) {
      memcpy(buff_to_gpu_ptr, group_by_buffers[i], groups_buffer_size);
      buff_to_gpu_ptr += groups_buffer_size;
    }
    cuda_allocator.copyToDevice(
        group_by_dev_buffers_mem, &buff_to_gpu[0], buff_to_gpu.size(), device_id);
  }

  auto group_by_dev_buffer = group_by_dev_buffers_mem;

  const size_t num_ptrs{block_size_x * grid_size_x};

  std::vector<CUdeviceptr> group_by_dev_buffers(num_ptrs);

  for (size_t i = 0; i < num_ptrs; i += step) {
    for (size_t j = 0; j < step; ++j) {
      group_by_dev_buffers[i + j] = group_by_dev_buffer;
    }
    if (!query_mem_desc.blocksShareMemory()) {
      group_by_dev_buffer += groups_buffer_size;
    }
  }

  auto group_by_dev_ptr =
      cuda_allocator.alloc(num_ptrs * sizeof(CUdeviceptr), device_id, nullptr);
  cuda_allocator.copyToDevice(group_by_dev_ptr,
                              &group_by_dev_buffers[0],
                              num_ptrs * sizeof(CUdeviceptr),
                              device_id);

  return std::make_pair(group_by_dev_ptr, group_by_dev_buffers_mem);
}

void copy_from_gpu(Data_Namespace::DataMgr* data_mgr,
                   void* dst,
                   const CUdeviceptr src,
                   const size_t num_bytes,
                   const int device_id) {
  const auto cuda_mgr = data_mgr->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyDeviceToHost(static_cast<int8_t*>(dst),
                             reinterpret_cast<const int8_t*>(src),
                             num_bytes,
                             device_id);
}

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const std::vector<int64_t*>& group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const CUdeviceptr group_by_dev_buffers_mem,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id,
                                    const bool prepend_index_buffer) {
  if (group_by_buffers.empty()) {
    return;
  }
  const unsigned block_buffer_count{query_mem_desc.blocksShareMemory() ? 1 : grid_size_x};
  if (block_buffer_count == 1 && !prepend_index_buffer) {
    CHECK_EQ(block_size_x, group_by_buffers.size());
    CHECK_EQ(coalesced_size(query_mem_desc, groups_buffer_size, block_buffer_count),
             groups_buffer_size);
    copy_from_gpu(data_mgr,
                  group_by_buffers[0],
                  group_by_dev_buffers_mem,
                  groups_buffer_size,
                  device_id);
    return;
  }
  const size_t index_buffer_sz{
      prepend_index_buffer ? query_mem_desc.getEntryCount() * sizeof(int64_t) : 0};
  std::vector<int8_t> buff_from_gpu(
      coalesced_size(query_mem_desc, groups_buffer_size, block_buffer_count) +
      index_buffer_sz);
  copy_from_gpu(data_mgr,
                &buff_from_gpu[0],
                group_by_dev_buffers_mem - index_buffer_sz,
                buff_from_gpu.size(),
                device_id);
  auto buff_from_gpu_ptr = &buff_from_gpu[0];
  for (size_t i = 0; i < block_buffer_count; ++i) {
    CHECK_LT(i * block_size_x, group_by_buffers.size());
    memcpy(group_by_buffers[i * block_size_x],
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
size_t get_num_allocated_rows_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                       CUdeviceptr projection_size_gpu,
                                       const int device_id) {
  int32_t num_rows{0};
  copy_from_gpu(data_mgr, &num_rows, projection_size_gpu, sizeof(num_rows), device_id);
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
void copy_projection_buffer_from_gpu_columnar(Data_Namespace::DataMgr* data_mgr,
                                              const GpuQueryMemory& gpu_query_mem,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              int8_t* projection_buffer,
                                              const size_t projection_count,
                                              const int device_id) {
  CHECK(query_mem_desc.didOutputColumnar());
  CHECK(query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection);
  constexpr size_t row_index_width = sizeof(int64_t);
  // copy all the row indices back to the host
  copy_from_gpu(data_mgr,
                reinterpret_cast<int64_t*>(projection_buffer),
                gpu_query_mem.group_by_buffers.second,
                projection_count * row_index_width,
                device_id);
  size_t buffer_offset_cpu{projection_count * row_index_width};
  // other columns are actual non-lazy columns for the projection:
  for (size_t i = 0; i < query_mem_desc.getColCount(); i++) {
    if (query_mem_desc.getPaddedColumnWidthBytes(i) > 0) {
      const auto column_proj_size =
          projection_count * query_mem_desc.getPaddedColumnWidthBytes(i);
      copy_from_gpu(
          data_mgr,
          projection_buffer + buffer_offset_cpu,
          gpu_query_mem.group_by_buffers.second + query_mem_desc.getColOffInBytes(i),
          column_proj_size,
          device_id);
      buffer_offset_cpu += align_to_int64(column_proj_size);
    }
  }
}

int8_t* ThrustAllocator::allocate(std::ptrdiff_t num_bytes) {
#ifdef HAVE_CUDA
  if (!data_mgr_) {  // only for unit tests
    CUdeviceptr ptr;
    const auto err = cuMemAlloc(&ptr, num_bytes);
    CHECK_EQ(CUDA_SUCCESS, err);
    return reinterpret_cast<int8_t*>(ptr);
  }
#endif  // HAVE_CUDA
  OOM_TRACE_PUSH(+": device_id " + std::to_string(device_id_) + ", num_bytes " +
                 std::to_string(num_bytes));
  Data_Namespace::AbstractBuffer* ab =
      alloc_gpu_abstract_buffer(data_mgr_, num_bytes, device_id_);
  int8_t* raw_ptr = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
  CHECK(!raw_to_ab_ptr_.count(raw_ptr));
  raw_to_ab_ptr_.insert(std::make_pair(raw_ptr, ab));
  return raw_ptr;
}

void ThrustAllocator::deallocate(int8_t* ptr, size_t num_bytes) {
#ifdef HAVE_CUDA
  if (!data_mgr_) {  // only for unit tests
    const auto err = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
    CHECK_EQ(CUDA_SUCCESS, err);
    return;
  }
#endif  // HAVE_CUDA
  PtrMapperType::iterator ab_it = raw_to_ab_ptr_.find(ptr);
  CHECK(ab_it != raw_to_ab_ptr_.end());
  data_mgr_->free(ab_it->second);
  raw_to_ab_ptr_.erase(ab_it);
}

int8_t* ThrustAllocator::allocateScopedBuffer(std::ptrdiff_t num_bytes) {
#ifdef HAVE_CUDA
  if (!data_mgr_) {  // only for unit tests
    CUdeviceptr ptr;
    const auto err = cuMemAlloc(&ptr, num_bytes);
    CHECK_EQ(CUDA_SUCCESS, err);
    default_alloc_scoped_buffers_.push_back(reinterpret_cast<int8_t*>(ptr));
    return reinterpret_cast<int8_t*>(ptr);
  }
#endif  // HAVE_CUDA
  OOM_TRACE_PUSH(+": device_id " + std::to_string(device_id_) + ", num_bytes " +
                 std::to_string(num_bytes));
  Data_Namespace::AbstractBuffer* ab =
      alloc_gpu_abstract_buffer(data_mgr_, num_bytes, device_id_);
  scoped_buffers_.push_back(ab);
  return reinterpret_cast<int8_t*>(ab->getMemoryPtr());
}

ThrustAllocator::~ThrustAllocator() {
  for (auto ab : scoped_buffers_) {
    data_mgr_->free(ab);
  }
#ifdef HAVE_CUDA
  for (auto ptr : default_alloc_scoped_buffers_) {
    const auto err = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
    CHECK_EQ(CUDA_SUCCESS, err);
  }
#endif  // HAVE_CUDA
}
