/*
 * Copyright 2019 OmniSci, Inc.
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

#pragma once

#include "GpuMemUtils.h"
#include "QueryMemoryDescriptor.h"
#include "ResultSet.h"

#include "Rendering/RenderAllocator.h"

#include <memory>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

class QueryMemoryInitializer {
 public:
  QueryMemoryInitializer(const RelAlgExecutionUnit& ra_exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc,
                         const int device_id,
                         const ExecutorDeviceType device_type,
                         const bool output_columnar,
                         const bool sort_on_gpu,
                         const std::vector<std::vector<const int8_t*>>& col_buffers,
                         const std::vector<int64_t>& consistent_frag_sizes,
                         const std::vector<std::vector<uint64_t>>& frag_offsets,
                         RenderAllocatorMap* render_allocator_map,
                         RenderInfo* render_info,
                         std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                         const Executor* executor);

  const auto getCountDistinctBitmapPtr() const { return count_distinct_bitmap_mem_; }

  const auto getCountDistinctHostPtr() const { return count_distinct_bitmap_host_mem_; }

  const auto getCountDistinctBitmapBytes() const {
    return count_distinct_bitmap_mem_bytes_;
  }

  ResultSet* getResultSet(const size_t index) const {
    CHECK_LT(index, result_sets_.size());
    return result_sets_[index].get();
  }

  std::unique_ptr<ResultSet> getResultSetOwned(const size_t index) {
    CHECK_LT(index, result_sets_.size());
    return std::unique_ptr<ResultSet>(result_sets_[index].release());
  }

  void resetResultSet(const size_t index) {
    CHECK_LT(index, result_sets_.size());
    result_sets_[index].reset();
  }

  int64_t getAggInitValForIndex(const size_t index) const {
    CHECK_LT(index, init_agg_vals_.size());
    return init_agg_vals_[index];
  }

  const auto getGroupByBuffersPtr() {
    return reinterpret_cast<int64_t**>(group_by_buffers_.data());
  }

  const auto getGroupByBuffersSize() const { return group_by_buffers_.size(); }

  const auto getNumBuffers() const {
    CHECK_EQ(num_buffers_, group_by_buffers_.size());
    return num_buffers_;
  }

 private:
  void initGroups(const QueryMemoryDescriptor& query_mem_desc,
                  int64_t* groups_buffer,
                  const std::vector<int64_t>& init_vals,
                  const int32_t groups_buffer_entry_count,
                  const size_t warp_size,
                  const Executor* executor);

  void initColumnarGroups(const QueryMemoryDescriptor& query_mem_desc,
                          int64_t* groups_buffer,
                          const std::vector<int64_t>& init_vals,
                          const Executor* executor);

  void initColumnPerRow(const QueryMemoryDescriptor& query_mem_desc,
                        int8_t* row_ptr,
                        const size_t bin,
                        const std::vector<int64_t>& init_vals,
                        const std::vector<ssize_t>& bitmap_sizes);

  void allocateCountDistinctGpuMem(const int device_id,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const Executor* executor);

  std::vector<ssize_t> allocateCountDistinctBuffers(
      const QueryMemoryDescriptor& query_mem_desc,
      const bool deferred,
      const Executor* executor);

  int64_t allocateCountDistinctBitmap(const size_t bitmap_byte_sz);

  int64_t allocateCountDistinctSet();

  std::vector<ColumnLazyFetchInfo> getColLazyFetchInfo(
      const std::vector<Analyzer::Expr*>& target_exprs,
      const Executor* executor) const;

  size_t getNumBuffers(const QueryMemoryDescriptor& query_mem_desc,
                       const ExecutorDeviceType device_type,
                       const Executor* executor) const;

  void compactProjectionBuffersCpu(const QueryMemoryDescriptor& query_mem_desc,
                                   const size_t projection_count);
  void compactProjectionBuffersGpu(const QueryMemoryDescriptor& query_mem_desc,
                                   Data_Namespace::DataMgr* data_mgr,
                                   const GpuQueryMemory& gpu_query_mem,
                                   const size_t projection_count,
                                   const int device_id);

  GpuGroupByBuffers createGroupByBuffersOnGpu(const CudaAllocator& cuda_allocator,
                                              RenderAllocator* render_allocator,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              const int device_id,
                                              const unsigned block_size_x,
                                              const unsigned grid_size_x,
                                              const bool can_sort_on_gpu);

  void copyGroupByBuffersFromGpu(Data_Namespace::DataMgr* data_mgr,
                                 const QueryMemoryDescriptor& query_mem_desc,
                                 const GpuQueryMemory& gpu_query_mem,
                                 const RelAlgExecutionUnit& ra_exe_unit,
                                 const unsigned block_size_x,
                                 const unsigned grid_size_x,
                                 const int device_id,
                                 const bool prepend_index_buffer) const;

  void applyStreamingTopNOffsetCpu(const QueryMemoryDescriptor& query_mem_desc,
                                   const RelAlgExecutionUnit& ra_exe_unit);

  void applyStreamingTopNOffsetGpu(Data_Namespace::DataMgr* data_mgr,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const GpuQueryMemory& gpu_query_mem,
                                   const RelAlgExecutionUnit& ra_exe_unit,
                                   const unsigned total_thread_count,
                                   const int device_id);

  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  std::vector<std::unique_ptr<ResultSet>> result_sets_;

  std::vector<int64_t> init_agg_vals_;
  const std::vector<int64_t> consistent_frag_sizes_;

  const size_t num_buffers_;
  std::vector<int64_t*> group_by_buffers_;

  CUdeviceptr count_distinct_bitmap_mem_;
  size_t count_distinct_bitmap_mem_bytes_;
  int8_t* count_distinct_bitmap_crt_ptr_;
  int8_t* count_distinct_bitmap_host_mem_;

  friend class Executor;  // Accesses result_sets_
  friend class QueryExecutionContext;
};
