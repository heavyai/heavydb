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
#include <variant>
#include <vector>

class CompilationContext;
struct CompilationResult;
class GpuCompilationContext;
class CpuCompilationContext;

struct RelAlgExecutionUnit;
class QueryMemoryDescriptor;
class Executor;

// For logging the KernelParams device addresses and data attributes.
struct KernelParamsLog {
  struct ColBuffer {
    size_t size;                    // Size of each column buffer
    std::vector<void const*> ptrs;  // Pointers to each column buffer
  };
  struct NamedSize {
    char const* name;
    size_t size;
  };
  using Value = std::variant<std::nullptr_t,  // Default to nullptr if not set
                             ColBuffer,
                             NamedSize,
                             std::vector<void const*>,
                             std::vector<std::vector<int64_t>>,
                             std::vector<std::vector<uint64_t>>>;
  std::array<void const*, size_t(heavyai::KernelParam::N_)> ptrs;
  std::array<Value, size_t(heavyai::KernelParam::N_)> values;
  bool hoist_literals;
};
std::ostream& operator<<(std::ostream&, KernelParamsLog const&);

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
      const CompilationResult& compilation_result,
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
  using KernelParamSizes = std::array<size_t, size_t(heavyai::KernelParam::N_)>;
  using KernelParams = std::array<int8_t*, size_t(heavyai::KernelParam::N_)>;

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
                                  std::vector<std::vector<T>> const& vec2d,
                                  std::string_view tag) const;

  size_t sizeofInitAggVals(bool const is_group_by,
                           std::vector<int64_t> const& init_agg_vals) const;
  void copyInitAggValsToDevice(int8_t* device_ptr,
                               bool const is_group_by,
                               std::vector<int64_t> const& init_agg_vals) const;

  size_t sizeofJoinHashTables(std::vector<int8_t*> const& join_hash_tables) const;
  int8_t* copyJoinHashTablesToDevice(int8_t* device_ptr,
                                     std::vector<int8_t*> const& join_hash_tables) const;

  size_t sizeofLiterals(std::vector<int8_t> const& literal_buff) const;
  size_t copyCountDistinctBitmapLiteralsToDevice(int8_t* device_ptr) const;
  size_t copyAggModeLiteralsToDevice(int8_t* device_ptr) const;
  int8_t* copyLiteralsToDevice(int8_t* device_ptr,
                               std::vector<int8_t> const& literal_buff) const;

  template <typename T>
  void copyValueToDevice(int8_t* device_ptr, T const value, std::string_view tag) const;

  template <typename T>
  size_t sizeofVector(std::vector<T> const& vec) const;
  template <typename T>
  void copyVectorToDevice(int8_t* device_ptr,
                          std::vector<T> const& vec,
                          std::string_view tag) const;

  std::pair<KernelParams, KernelParamsLog> prepareKernelParams(
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
      const bool is_group_by) const;

  ResultSetPtr groupBufferToDeinterleavedResults(const size_t i) const;

  DeviceAllocator* device_allocator_;

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
  friend std::ostream& operator<<(std::ostream&, KernelParamsLog const&);
};

#endif  // QUERYENGINE_QUERYEXECUTIONCONTEXT_H
