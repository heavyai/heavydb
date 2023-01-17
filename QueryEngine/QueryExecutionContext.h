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

#ifndef QUERYENGINE_QUERYEXECUTIONCONTEXT_H
#define QUERYENGINE_QUERYEXECUTIONCONTEXT_H

#include "CompilationOptions.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "GpuMemUtils.h"
#include "QueryMemoryInitializer.h"
#include "Rendering/RenderInfo.h"
#include "ResultSet.h"

#include <boost/core/noncopyable.hpp>
#include <vector>

class CompilationContext;
class GpuCompilationContext;
class CpuCompilationContext;

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
                        RenderInfo*);

  ResultSetPtr getRowSet(const RelAlgExecutionUnit& ra_exe_unit,
                         const QueryMemoryDescriptor& query_mem_desc) const;

  ResultSetPtr groupBufferToResults(const size_t i) const;

  std::vector<int64_t*> launchGpuCode(
      const RelAlgExecutionUnit& ra_exe_unit,
      const CompilationContext* compilation_context,
      const bool hoist_literals,
      const std::vector<int8_t>& literal_buff,
      std::vector<std::vector<const int8_t*>> col_buffers,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_row_offsets,
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
      bool optimize_cuda_block_and_grid_sizes);

  std::vector<int64_t*> launchCpuCode(
      const RelAlgExecutionUnit& ra_exe_unit,
      const CpuCompilationContext* fn_ptrs,
      const bool hoist_literals,
      const std::vector<int8_t>& literal_buff,
      std::vector<std::vector<const int8_t*>> col_buffers,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_row_offsets,
      const int32_t scan_limit,
      int32_t* error_code,
      const uint32_t start_rowid,
      const uint32_t num_tables,
      const std::vector<int8_t*>& join_hash_tables,
      const int64_t num_rows_to_process = -1);

  int64_t getAggInitValForIndex(const size_t index) const;

 private:
  // enum must be kept in sync w/ prepareKernelParams().
  enum {
    ERROR_CODE,
    TOTAL_MATCHED,
    GROUPBY_BUF,
    NUM_FRAGMENTS,
    NUM_TABLES,
    ROW_INDEX_RESUME,
    COL_BUFFERS,
    LITERALS,
    NUM_ROWS,
    FRAG_ROW_OFFSETS,
    MAX_MATCHED,
    INIT_AGG_VALS,
    JOIN_HASH_TABLES,
    ROW_FUNC_MGR,
    KERN_PARAM_COUNT
  };
  using KernelParamSizes = std::array<size_t, KERN_PARAM_COUNT>;
  using KernelParams = std::array<int8_t*, KERN_PARAM_COUNT>;

  size_t sizeofColBuffers(
      std::vector<std::vector<int8_t const*>> const& col_buffers) const;
  void copyColBuffersToDevice(
      int8_t* device_ptr,
      std::vector<std::vector<int8_t const*>> const& col_buffers) const;

  template <typename T>
  size_t sizeofFlattened2dVec(uint32_t const expected_subvector_size,
                              std::vector<std::vector<T>> const& vec2d) const;
  template <typename T>
  void copyFlattened2dVecToDevice(int8_t* device_ptr,
                                  uint32_t const expected_subvector_size,
                                  std::vector<std::vector<T>> const& vec2d) const;

  size_t sizeofInitAggVals(bool const is_group_by,
                           std::vector<int64_t> const& init_agg_vals) const;
  void copyInitAggValsToDevice(int8_t* device_ptr,
                               bool const is_group_by,
                               std::vector<int64_t> const& init_agg_vals) const;

  size_t sizeofJoinHashTables(std::vector<int8_t*> const& join_hash_tables) const;
  int8_t* copyJoinHashTablesToDevice(int8_t* device_ptr,
                                     std::vector<int8_t*> const& join_hash_tables) const;

  size_t sizeofLiterals(std::vector<int8_t> const& literal_buff) const;
  int8_t* copyLiteralsToDevice(int8_t* device_ptr,
                               std::vector<int8_t> const& literal_buff) const;

  template <typename T>
  void copyValueToDevice(int8_t* device_ptr, T const value) const;

  template <typename T>
  size_t sizeofVector(std::vector<T> const& vec) const;
  template <typename T>
  void copyVectorToDevice(int8_t* device_ptr, std::vector<T> const& vec) const;

  KernelParams prepareKernelParams(
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
      const bool is_group_by) const;

  ResultSetPtr groupBufferToDeinterleavedResults(const size_t i) const;

  std::unique_ptr<DeviceAllocator> gpu_allocator_;

  // TODO(adb): convert to shared_ptr
  QueryMemoryDescriptor query_mem_desc_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const ExecutorDispatchMode dispatch_mode_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  const bool output_columnar_;
  std::unique_ptr<QueryMemoryInitializer> query_buffers_;
  mutable std::unique_ptr<ResultSet> estimator_result_set_;

  friend class Executor;
};

#endif  // QUERYENGINE_QUERYEXECUTIONCONTEXT_H
