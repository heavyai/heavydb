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

#ifndef QUERYENGINE_QUERYEXECUTIONCONTEXT_H
#define QUERYENGINE_QUERYEXECUTIONCONTEXT_H

#include "CompilationOptions.h"
#include "CudaAllocator.h"
#include "GpuMemUtils.h"
#include "Rendering/RenderInfo.h"
#include "ResultSet.h"

#include "QueryMemoryInitializer.h"

#include <boost/core/noncopyable.hpp>
#include <vector>

struct RelAlgExecutionUnit;
class QueryMemoryDescriptor;
class Executor;

class QueryExecutionContext : boost::noncopyable {
 public:
  // TODO(alex): remove device_type
  QueryExecutionContext(const RelAlgExecutionUnit& ra_exe_unit,
                        const QueryMemoryDescriptor&,
                        const Executor* executor,
                        const ExecutorDeviceType device_type,
                        const int device_id,
                        const std::vector<std::vector<const int8_t*>>& col_buffers,
                        const std::vector<std::vector<const int8_t*>>& iter_buffers,
                        const std::vector<std::vector<uint64_t>>& frag_offsets,
                        std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                        const bool output_columnar,
                        const bool sort_on_gpu,
                        RenderInfo*);

  ResultSetPtr getRowSet(const RelAlgExecutionUnit& ra_exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc) const;

  ResultSetPtr groupBufferToResults(const size_t i,
                                    const std::vector<Analyzer::Expr*>& targets) const;

  std::vector<int64_t*> launchGpuCode(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<std::pair<void*, void*>>& cu_functions,
      const bool hoist_literals,
      const std::vector<int8_t>& literal_buff,
      std::vector<std::vector<const int8_t*>> col_buffers,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_row_offsets,
      const uint32_t frag_stride,
      const int32_t scan_limit,
      Data_Namespace::DataMgr* data_mgr,
      const unsigned block_size_x,
      const unsigned grid_size_x,
      const int device_id,
      int32_t* error_code,
      const uint32_t num_tables,
      const std::vector<int64_t>& join_hash_tables,
      RenderAllocatorMap* render_allocator_map);

  std::vector<int64_t*> launchCpuCode(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<std::pair<void*, void*>>& fn_ptrs,
      const bool hoist_literals,
      const std::vector<int8_t>& literal_buff,
      std::vector<std::vector<const int8_t*>> col_buffers,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_row_offsets,
      const uint32_t frag_stride,
      const int32_t scan_limit,
      int32_t* error_code,
      const uint32_t num_tables,
      const std::vector<int64_t>& join_hash_tables);

  bool hasNoFragments() const { return consistent_frag_sizes_.empty(); }

  int64_t getAggInitValForIndex(const size_t index) const;

 private:
  const std::vector<const int8_t*>& getColumnFrag(const size_t table_idx,
                                                  int64_t& global_idx) const;

  uint32_t getFragmentStride(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids) const;

#ifdef HAVE_CUDA
  enum {
    COL_BUFFERS,
    NUM_FRAGMENTS,
    FRAG_STRIDE,
    LITERALS,
    NUM_ROWS,
    FRAG_ROW_OFFSETS,
    MAX_MATCHED,
    TOTAL_MATCHED,
    INIT_AGG_VALS,
    GROUPBY_BUF,
    SMALL_BUF,
    ERROR_CODE,
    NUM_TABLES,
    JOIN_HASH_TABLES,
    KERN_PARAM_COUNT,
  };

  void initializeDynamicWatchdog(void* native_module, const int device_id) const;

  std::vector<CUdeviceptr> prepareKernelParams(
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
      const bool is_group_by) const;

  std::pair<CUdeviceptr, CUdeviceptr> prepareTopNHeapsDevBuffer(
      const CudaAllocator& cuda_allocator,
      const CUdeviceptr init_agg_vals_dev_ptr,
      const size_t n,
      const int device_id,
      const unsigned block_size_x,
      const unsigned grid_size_x) const;

  GpuQueryMemory prepareGroupByDevBuffer(const CudaAllocator& cuda_allocator,
                                         RenderAllocator* render_allocator,
                                         const RelAlgExecutionUnit& ra_exe_unit,
                                         const CUdeviceptr init_agg_vals_dev_ptr,
                                         const int device_id,
                                         const unsigned block_size_x,
                                         const unsigned grid_size_x,
                                         const bool can_sort_on_gpu) const;
#endif  // HAVE_CUDA

  ResultSetPtr groupBufferToDeinterleavedResults(const size_t i) const;

  const QueryMemoryDescriptor& query_mem_desc_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  const std::vector<std::vector<const int8_t*>>& col_buffers_;
  const std::vector<std::vector<const int8_t*>>& iter_buffers_;
  const std::vector<std::vector<uint64_t>>& frag_offsets_;
  const std::vector<int64_t> consistent_frag_sizes_;

  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  const bool output_columnar_;
  const bool sort_on_gpu_;

  std::unique_ptr<QueryMemoryInitializer> query_buffers_;

  mutable std::unique_ptr<ResultSet> estimator_result_set_;

  friend class Executor;

  // Temporary; Reduction egress needs to become part of executor
  template <typename META_CLASS_TYPE>
  friend class AggregateReductionEgress;

  friend void copy_group_by_buffers_from_gpu(
      Data_Namespace::DataMgr* data_mgr,
      const QueryExecutionContext* query_exe_context,
      const GpuQueryMemory& gpu_query_mem,
      const RelAlgExecutionUnit& ra_exe_unit,
      const unsigned block_size_x,
      const unsigned grid_size_x,
      const int device_id,
      const bool prepend_index_buffer);
};

#endif  // QUERYENGINE_QUERYEXECUTIONCONTEXT_H
