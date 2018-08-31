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
#include "IteratorTable.h"
#include "Rendering/RenderInfo.h"

#include <boost/core/noncopyable.hpp>
#include <vector>

struct RelAlgExecutionUnit;
class QueryMemoryDescriptor;
class Executor;

class QueryExecutionContext : boost::noncopyable {
 public:
  // TODO(alex): move init_agg_vals to GroupByBufferDescriptor, remove device_type
  QueryExecutionContext(const RelAlgExecutionUnit& ra_exe_unit,
                        const QueryMemoryDescriptor&,
                        const std::vector<int64_t>& init_agg_vals,
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

  ResultPtr getResult(const RelAlgExecutionUnit& ra_exe_unit,
                      const std::vector<size_t>& outer_tab_frag_ids) const;

  // TOOD(alex): get rid of targets parameter
  RowSetPtr getRowSet(const RelAlgExecutionUnit& ra_exe_unit,
                      const QueryMemoryDescriptor& query_mem_desc) const;
  RowSetPtr groupBufferToResults(const size_t i,
                                 const std::vector<Analyzer::Expr*>& targets) const;

  IterTabPtr getIterTab(const std::vector<Analyzer::Expr*>& targets,
                        const ssize_t frag_idx) const;

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
      const std::vector<int64_t>& init_agg_vals,
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
      const std::vector<int64_t>& init_agg_vals,
      int32_t* error_code,
      const uint32_t num_tables,
      const std::vector<int64_t>& join_hash_tables);

  bool hasNoFragments() const { return consistent_frag_sizes_.empty(); }

 private:
  const std::vector<const int8_t*>& getColumnFrag(const size_t table_idx,
                                                  int64_t& global_idx) const;
  bool isEmptyBin(const int64_t* group_by_buffer,
                  const size_t bin,
                  const size_t key_idx) const;

  void initColumnPerRow(const QueryMemoryDescriptor& query_mem_desc,
                        int8_t* row_ptr,
                        const size_t bin,
                        const int64_t* init_vals,
                        const std::vector<ssize_t>& bitmap_sizes);

  void initGroups(int64_t* groups_buffer,
                  const int64_t* init_vals,
                  const int32_t groups_buffer_entry_count,
                  const bool keyless,
                  const size_t warp_size);

  template <typename T>
  int8_t* initColumnarBuffer(T* buffer_ptr, const T init_val, const uint32_t entry_count);

  void initColumnarGroups(int64_t* groups_buffer,
                          const int64_t* init_vals,
                          const int32_t groups_buffer_entry_count,
                          const bool keyless);

  IterTabPtr groupBufferToTab(const size_t buf_idx,
                              const ssize_t frag_idx,
                              const std::vector<Analyzer::Expr*>& targets) const;

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
      Data_Namespace::DataMgr* data_mgr,
      const CUdeviceptr init_agg_vals_dev_ptr,
      const size_t n,
      const int device_id,
      const unsigned block_size_x,
      const unsigned grid_size_x) const;

  GpuQueryMemory prepareGroupByDevBuffer(Data_Namespace::DataMgr* data_mgr,
                                         RenderAllocator* render_allocator,
                                         const RelAlgExecutionUnit& ra_exe_unit,
                                         const CUdeviceptr init_agg_vals_dev_ptr,
                                         const int device_id,
                                         const unsigned block_size_x,
                                         const unsigned grid_size_x,
                                         const bool can_sort_on_gpu) const;
#endif  // HAVE_CUDA

  std::vector<ssize_t> allocateCountDistinctBuffers(const bool deferred);
  int64_t allocateCountDistinctBitmap(const size_t bitmap_byte_sz);
  int64_t allocateCountDistinctSet();

  std::vector<ColumnLazyFetchInfo> getColLazyFetchInfo(
      const std::vector<Analyzer::Expr*>& target_exprs) const;

  void allocateCountDistinctGpuMem();

  RowSetPtr groupBufferToDeinterleavedResults(const size_t i) const;

  const QueryMemoryDescriptor& query_mem_desc_;
  std::vector<int64_t> init_agg_vals_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  const std::vector<std::vector<const int8_t*>>& col_buffers_;
  const std::vector<std::vector<const int8_t*>>& iter_buffers_;
  const std::vector<std::vector<uint64_t>>& frag_offsets_;
  const std::vector<int64_t> consistent_frag_sizes_;
  const size_t num_buffers_;

  std::vector<int64_t*> group_by_buffers_;
  std::vector<int64_t*> small_group_by_buffers_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  const bool output_columnar_;
  const bool sort_on_gpu_;

  mutable std::vector<std::unique_ptr<ResultSet>> result_sets_;
  mutable std::unique_ptr<ResultSet> estimator_result_set_;
  CUdeviceptr count_distinct_bitmap_mem_;
  int8_t* count_distinct_bitmap_host_mem_;
  int8_t* count_distinct_bitmap_crt_ptr_;
  size_t count_distinct_bitmap_mem_bytes_;

  friend class Executor;
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