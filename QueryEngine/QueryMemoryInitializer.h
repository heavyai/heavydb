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

#pragma once

#include "Aggregate/AggModeHashTableGpu.h"
#include "DataMgr/Allocators/DeviceAllocator.h"
#include "Descriptors/QueryMemoryDescriptor.h"
#include "GpuMemUtils.h"
#include "Rendering/RenderAllocator.h"
#include "ResultSet.h"

#include "ThirdParty/robin_hood/robin_hood.h"

#include <memory>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

class AggMode;

class QueryMemoryInitializer {
 public:
  using ModeIndexSet = robin_hood::unordered_set<size_t>;
  using QuantileParam = std::optional<double>;
  struct TargetAggOpsMetadata {
    bool has_count_distinct{false};
    bool has_mode{false};
    bool has_tdigest{false};
    std::vector<int64_t> count_distinct_buf_size;
    ModeIndexSet mode_index_set;
    std::vector<QuantileParam> quantile_params;
  };

  // Row-based execution constructor
  QueryMemoryInitializer(const RelAlgExecutionUnit& ra_exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc,
                         const int device_id,
                         const ExecutorDeviceType device_type,
                         const ExecutorDispatchMode dispatch_mode,
                         const bool output_columnar,
                         const bool sort_on_gpu,
                         const shared::TableKey& outer_table_key,
                         const int64_t num_rows,
                         const std::vector<std::vector<const int8_t*>>& col_buffers,
                         const std::vector<std::vector<uint64_t>>& frag_offsets,
                         RenderAllocatorMap* render_allocator_map,
                         RenderInfo* render_info,
                         std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                         DeviceAllocator* gpu_allocator,
                         const size_t thread_idx,
                         const Executor* executor);

  // Table functions execution constructor
  QueryMemoryInitializer(const TableFunctionExecutionUnit& exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc,
                         const int device_id,
                         const ExecutorDeviceType device_type,
                         const int64_t num_rows,
                         const std::vector<std::vector<const int8_t*>>& col_buffers,
                         const std::vector<std::vector<uint64_t>>& frag_offsets,
                         std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                         DeviceAllocator* device_allocator,
                         const Executor* executor);

  const auto getCountDistinctBitmapDevicePtr() const {
    return count_distinct_bitmap_device_mem_ptr_;
  }

  const auto getCountDistinctBitmapHostPtr() const {
    return count_distinct_bitmap_host_mem_ptr_;
  }

  const auto getCountDistinctBitmapBytes() const {
    return count_distinct_bitmap_mem_size_;
  }

  std::vector<int8_t> getAggModeHashTablesGpu() const;

  // Number of mode hash table gpu objects
  size_t getNumAggModeHashTablesGpu() const { return agg_mode_hash_tables_gpu_.size(); }

  // TODO: lazy init (maybe lazy init count distinct above, too?)
  const auto getVarlenOutputHostPtr() const { return varlen_output_buffer_host_ptr_; }

  const auto getVarlenOutputPtr() const { return varlen_output_buffer_; }

  ResultSet* getResultSet(const size_t index) const {
    CHECK_LT(index, result_sets_.size());
    return result_sets_[index].get();
  }

  std::unique_ptr<ResultSet> getResultSetOwned(const size_t index) {
    CHECK_LT(index, result_sets_.size());
    return std::move(result_sets_[index]);
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

  GpuGroupByBuffers setupTableFunctionGpuBuffers(
      const QueryMemoryDescriptor& query_mem_desc,
      const int device_id,
      const unsigned block_size_x,
      const unsigned grid_size_x,
      const bool zero_initialize_buffers);

  void copyFromTableFunctionGpuBuffers(DeviceAllocator* device_allocator,
                                       const QueryMemoryDescriptor& query_mem_desc,
                                       const size_t entry_count,
                                       const GpuGroupByBuffers& gpu_group_by_buffers,
                                       const int device_id,
                                       const unsigned block_size_x,
                                       const unsigned grid_size_x);

  void copyGroupByBuffersFromGpu(DeviceAllocator& device_allocator,
                                 const QueryMemoryDescriptor& query_mem_desc,
                                 const size_t entry_count,
                                 const GpuGroupByBuffers& gpu_group_by_buffers,
                                 const RelAlgExecutionUnit* ra_exe_unit,
                                 const unsigned block_size_x,
                                 const unsigned grid_size_x,
                                 const int device_id,
                                 const bool prepend_index_buffer) const;

  void copyFromDeviceForAggMode();

 private:
  void initGroupByBuffer(int64_t* buffer,
                         const RelAlgExecutionUnit& ra_exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc,
                         TargetAggOpsMetadata& agg_expr_metadata,
                         const ExecutorDeviceType device_type,
                         const bool output_columnar,
                         const Executor* executor);

  void initRowGroups(const QueryMemoryDescriptor& query_mem_desc,
                     int64_t* groups_buffer,
                     const std::vector<int64_t>& init_vals,
                     TargetAggOpsMetadata& agg_expr_metadata,
                     const int32_t groups_buffer_entry_count,
                     const size_t warp_size,
                     const Executor* executor,
                     const RelAlgExecutionUnit& ra_exe_unit);

  void initColumnarGroups(const QueryMemoryDescriptor& query_mem_desc,
                          int64_t* groups_buffer,
                          const std::vector<int64_t>& init_vals,
                          const Executor* executor,
                          const RelAlgExecutionUnit& ra_exe_unit);

  void initColumnsPerRow(const QueryMemoryDescriptor& query_mem_desc,
                         int8_t* row_ptr,
                         const std::vector<int64_t>& init_vals,
                         const TargetAggOpsMetadata& agg_op_metadata);

  void allocateCountDistinctGpuMem(const QueryMemoryDescriptor& query_mem_desc);

  std::vector<int64_t> calculateCountDistinctBufferSize(
      const QueryMemoryDescriptor& query_mem_desc,
      const RelAlgExecutionUnit& ra_exe_unit) const;

  void fastAllocateCountDistinctBuffers(const QueryMemoryDescriptor& query_mem_desc,
                                        const RelAlgExecutionUnit& ra_exe_unit);

  int64_t allocateCountDistinctBitmap(const size_t bitmap_byte_sz);

  int64_t allocateCountDistinctSet();

  ModeIndexSet initializeModeIndexSet(const QueryMemoryDescriptor& query_mem_desc,
                                      const RelAlgExecutionUnit& ra_exe_unit);
  void allocateModeMem(ExecutorDeviceType,
                       const QueryMemoryDescriptor&,
                       const Executor*,
                       int);

  // Return CPU: AggMode* or GPU: (1<<63 | i<<32 | j+1)
  // where i, j are agg_mode_hash_tables_cpu_, row_set_mem_owner_ indexes.
  int64_t allocateAggMode();

  void allocateModeBuffer(const QueryMemoryDescriptor& query_mem_desc,
                          const RelAlgExecutionUnit& ra_exe_unit);

  std::vector<QuantileParam> initializeQuantileParams(
      const QueryMemoryDescriptor& query_mem_desc,
      const RelAlgExecutionUnit& ra_exe_unit);

  void allocateTDigestsBuffer(const QueryMemoryDescriptor& query_mem_desc,
                              const RelAlgExecutionUnit& ra_exe_unit);

  GpuGroupByBuffers prepareTopNHeapsDevBuffer(const QueryMemoryDescriptor& query_mem_desc,
                                              const int8_t* init_agg_vals_dev_ptr,
                                              const size_t n,
                                              const int device_id,
                                              const unsigned block_size_x,
                                              const unsigned grid_size_x,
                                              CUstream cuda_stream);

  GpuGroupByBuffers createAndInitializeGroupByBufferGpu(
      const RelAlgExecutionUnit& ra_exe_unit,
      const QueryMemoryDescriptor& query_mem_desc,
      const int8_t* init_agg_vals_dev_ptr,
      const int device_id,
      CUstream cuda_stream,
      const ExecutorDispatchMode dispatch_mode,
      const unsigned block_size_x,
      const unsigned grid_size_x,
      const int8_t warp_size,
      const bool can_sort_on_gpu,
      const bool output_columnar,
      RenderAllocator* render_allocator);

  size_t computeNumberOfBuffers(const QueryMemoryDescriptor& query_mem_desc,
                                const ExecutorDeviceType device_type,
                                const Executor* executor) const;

  void compactProjectionBuffersCpu(const QueryMemoryDescriptor& query_mem_desc,
                                   const size_t projection_count);
  void compactProjectionBuffersGpu(const QueryMemoryDescriptor& query_mem_desc,
                                   DeviceAllocator* device_allocator,
                                   const GpuGroupByBuffers& gpu_group_by_buffers,
                                   const size_t projection_count,
                                   const int device_id);

  void applyStreamingTopNOffsetCpu(const QueryMemoryDescriptor& query_mem_desc,
                                   const RelAlgExecutionUnit& ra_exe_unit);

  void applyStreamingTopNOffsetGpu(Data_Namespace::DataMgr* data_mgr,
                                   CudaAllocator* cuda_allocator,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const GpuGroupByBuffers& gpu_group_by_buffers,
                                   const RelAlgExecutionUnit& ra_exe_unit,
                                   const unsigned total_thread_count,
                                   const int device_id,
                                   CUstream cuda_stream);

  std::shared_ptr<VarlenOutputInfo> getVarlenOutputInfo();

  const int64_t num_rows_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  std::vector<std::unique_ptr<ResultSet>> result_sets_;

  std::vector<int64_t> init_agg_vals_;

  size_t num_buffers_;
  std::vector<int64_t*> group_by_buffers_;
  std::shared_ptr<VarlenOutputInfo> varlen_output_info_;
  CUdeviceptr varlen_output_buffer_;
  int8_t* varlen_output_buffer_host_ptr_;

  // Contains inter-device hash table objects, one per target (i.e. column).
  // Each object is sizeof(AggModeHashTableGpu)=104 bytes and copies lightly.
  AggModeHashTablesGpu agg_mode_hash_tables_gpu_;
  // CPU hash tables owned by RowSetMemoryOwner.
  std::vector<AggMode*> agg_mode_hash_tables_cpu_;

  CUdeviceptr count_distinct_bitmap_device_mem_ptr_;
  size_t count_distinct_bitmap_mem_size_;
  int8_t* count_distinct_bitmap_host_crt_ptr_;
  int8_t* count_distinct_bitmap_host_mem_ptr_;

  DeviceAllocator* device_allocator_{nullptr};
  std::vector<Data_Namespace::AbstractBuffer*> temporary_buffers_;

  const size_t thread_idx_;

  friend class Executor;  // Accesses result_sets_
  friend class QueryExecutionContext;
};
