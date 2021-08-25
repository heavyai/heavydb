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

#include "QueryEngine/TableFunctions/TableFunctionExecutionContext.h"

#include "Analyzer/Analyzer.h"
#include "Logger/Logger.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/TableFunctions/QueryOutputBufferMemoryManager.h"
#include "QueryEngine/TableFunctions/TableFunctionCompilationContext.h"
#include "Shared/funcannotations.h"

namespace {

template <typename T>
const int8_t* create_literal_buffer(T literal,
                                    const ExecutorDeviceType device_type,
                                    std::vector<std::unique_ptr<char[]>>& literals_owner,
                                    CudaAllocator* gpu_allocator) {
  CHECK_LE(sizeof(T), sizeof(int64_t));  // pad to 8 bytes
  switch (device_type) {
    case ExecutorDeviceType::CPU: {
      literals_owner.emplace_back(std::make_unique<char[]>(sizeof(int64_t)));
      std::memcpy(literals_owner.back().get(), &literal, sizeof(T));
      return reinterpret_cast<const int8_t*>(literals_owner.back().get());
    }
    case ExecutorDeviceType::GPU: {
      CHECK(gpu_allocator);
      const auto gpu_literal_buf_ptr = gpu_allocator->alloc(sizeof(int64_t));
      gpu_allocator->copyToDevice(
          gpu_literal_buf_ptr, reinterpret_cast<int8_t*>(&literal), sizeof(T));
      return gpu_literal_buf_ptr;
    }
  }
  UNREACHABLE();
  return nullptr;
}

size_t get_output_row_count(const TableFunctionExecutionUnit& exe_unit,
                            size_t input_element_count) {
  size_t allocated_output_row_count = 0;
  switch (exe_unit.table_func.getOutputRowSizeType()) {
    case table_functions::OutputBufferSizeType::kConstant:
    case table_functions::OutputBufferSizeType::kUserSpecifiedConstantParameter: {
      allocated_output_row_count = exe_unit.output_buffer_size_param;
      break;
    }
    case table_functions::OutputBufferSizeType::kUserSpecifiedRowMultiplier: {
      allocated_output_row_count =
          exe_unit.output_buffer_size_param * input_element_count;
      break;
    }
    case table_functions::OutputBufferSizeType::kTableFunctionSpecifiedParameter: {
      allocated_output_row_count = input_element_count;
      break;
    }
    default: {
      UNREACHABLE();
    }
  }
  return allocated_output_row_count;
}

}  // namespace

ResultSetPtr TableFunctionExecutionContext::execute(
    const TableFunctionExecutionUnit& exe_unit,
    const std::vector<InputTableInfo>& table_infos,
    const TableFunctionCompilationContext* compilation_context,
    const ColumnFetcher& column_fetcher,
    const ExecutorDeviceType device_type,
    Executor* executor) {
  CHECK(compilation_context);
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<std::unique_ptr<char[]>> literals_owner;

  const int device_id = 0;  // TODO(adb): support multi-gpu table functions
  std::unique_ptr<CudaAllocator> device_allocator;
  if (device_type == ExecutorDeviceType::GPU) {
    auto data_mgr = executor->getDataMgr();
    device_allocator.reset(new CudaAllocator(data_mgr, device_id));
  }
  std::vector<const int8_t*> col_buf_ptrs;
  std::vector<int64_t> col_sizes;
  std::optional<size_t> output_column_size;

  int col_index = -1;
  // TODO: col_list_bufs are allocated on CPU memory, so UDTFs with column_list
  // arguments are not supported on GPU atm.
  std::vector<std::vector<const int8_t*>> col_list_bufs;
  for (const auto& input_expr : exe_unit.input_exprs) {
    auto ti = input_expr->get_type_info();
    if (!ti.is_column_list()) {
      CHECK_EQ(col_index, -1);
    }
    if (auto col_var = dynamic_cast<Analyzer::ColumnVar*>(input_expr)) {
      auto table_id = col_var->get_table_id();
      auto table_info_it = std::find_if(
          table_infos.begin(), table_infos.end(), [&table_id](const auto& table_info) {
            return table_info.table_id == table_id;
          });
      CHECK(table_info_it != table_infos.end());
      auto [col_buf, buf_elem_count] = ColumnFetcher::getOneColumnFragment(
          executor,
          *col_var,
          table_info_it->info.fragments.front(),
          device_type == ExecutorDeviceType::CPU ? Data_Namespace::MemoryLevel::CPU_LEVEL
                                                 : Data_Namespace::MemoryLevel::GPU_LEVEL,
          device_id,
          device_allocator.get(),
          /*thread_idx=*/0,
          chunks_owner,
          column_fetcher.columnarized_table_cache_);
      // We use the size of the first column to be the size of the output column
      if (!output_column_size) {
        output_column_size = (buf_elem_count ? buf_elem_count : 1);
      }
      if (ti.is_column_list()) {
        if (col_index == -1) {
          col_list_bufs.push_back({});
          col_list_bufs.back().reserve(ti.get_dimension());
        } else {
          CHECK_EQ(col_sizes.back(), buf_elem_count);
        }
        col_index++;
        col_list_bufs.back().push_back(col_buf);
        // append col_buf to column_list col_buf
        if (col_index + 1 == ti.get_dimension()) {
          col_index = -1;
        }
        // columns in the same column_list point to column_list data
        col_buf_ptrs.push_back((const int8_t*)col_list_bufs.back().data());
      } else {
        col_buf_ptrs.push_back(col_buf);
      }
      col_sizes.push_back(buf_elem_count);
    } else if (const auto& constant_val = dynamic_cast<Analyzer::Constant*>(input_expr)) {
      // TODO(adb): Unify literal handling with rest of system, either in Codegen or as a
      // separate serialization component
      col_sizes.push_back(0);
      const auto const_val_datum = constant_val->get_constval();
      const auto& ti = constant_val->get_type_info();
      if (ti.is_fp()) {
        switch (get_bit_width(ti)) {
          case 32:
            col_buf_ptrs.push_back(create_literal_buffer(const_val_datum.floatval,
                                                         device_type,
                                                         literals_owner,
                                                         device_allocator.get()));
            break;
          case 64:
            col_buf_ptrs.push_back(create_literal_buffer(const_val_datum.doubleval,
                                                         device_type,
                                                         literals_owner,
                                                         device_allocator.get()));
            break;
          default:
            UNREACHABLE();
        }
      } else if (ti.is_integer()) {
        switch (get_bit_width(ti)) {
          case 8:
            col_buf_ptrs.push_back(create_literal_buffer(const_val_datum.tinyintval,
                                                         device_type,
                                                         literals_owner,
                                                         device_allocator.get()));
            break;
          case 16:
            col_buf_ptrs.push_back(create_literal_buffer(const_val_datum.smallintval,
                                                         device_type,
                                                         literals_owner,
                                                         device_allocator.get()));
            break;
          case 32:
            col_buf_ptrs.push_back(create_literal_buffer(const_val_datum.intval,
                                                         device_type,
                                                         literals_owner,
                                                         device_allocator.get()));
            break;
          case 64:
            col_buf_ptrs.push_back(create_literal_buffer(const_val_datum.bigintval,
                                                         device_type,
                                                         literals_owner,
                                                         device_allocator.get()));
            break;
          default:
            UNREACHABLE();
        }
      } else if (ti.is_boolean()) {
        col_buf_ptrs.push_back(create_literal_buffer(const_val_datum.boolval,
                                                     device_type,
                                                     literals_owner,
                                                     device_allocator.get()));
      } else {
        throw std::runtime_error("Literal value " + constant_val->toString() +
                                 " is not yet supported.");
      }
    }
  }
  CHECK_EQ(col_buf_ptrs.size(), exe_unit.input_exprs.size());
  CHECK_EQ(col_sizes.size(), exe_unit.input_exprs.size());
  if (!exe_unit.table_func
           .hasNonUserSpecifiedOutputSizeConstant()) {  // includes compile-time constants
                                                        // and runtime table funtion
                                                        // specified sizing
    CHECK(output_column_size);
  }
  switch (device_type) {
    case ExecutorDeviceType::CPU:
      return launchCpuCode(exe_unit,
                           compilation_context,
                           col_buf_ptrs,
                           col_sizes,
                           *output_column_size,
                           executor);
    case ExecutorDeviceType::GPU:
      return launchGpuCode(exe_unit,
                           compilation_context,
                           col_buf_ptrs,
                           col_sizes,
                           *output_column_size,
                           /*device_id=*/0,
                           executor);
  }
  UNREACHABLE();
  return nullptr;
}

ResultSetPtr TableFunctionExecutionContext::launchCpuCode(
    const TableFunctionExecutionUnit& exe_unit,
    const TableFunctionCompilationContext* compilation_context,
    std::vector<const int8_t*>& col_buf_ptrs,
    std::vector<int64_t>& col_sizes,
    const size_t elem_count,
    Executor* executor) {
  int64_t output_row_count = 0;

  // mgr will allocate output buffers on output column resize
  auto mgr = std::make_unique<QueryOutputBufferMemoryManager>(
      exe_unit, executor, col_buf_ptrs, row_set_mem_owner_);

  if (!exe_unit.table_func.hasTableFunctionSpecifiedParameter()) {
    // allocate output buffers because the size is defined by the size
    // of input buffers
    output_row_count = get_output_row_count(exe_unit, elem_count);
  }

  // setup the inputs
  // We can have an empty col_buf_ptrs vector if there are no arguments to the function
  const auto byte_stream_ptr = !col_buf_ptrs.empty()
                                   ? reinterpret_cast<const int8_t**>(col_buf_ptrs.data())
                                   : nullptr;
  if (!col_buf_ptrs.empty()) {
    CHECK(byte_stream_ptr);
  }
  const auto col_sizes_ptr = !col_sizes.empty() ? col_sizes.data() : nullptr;
  if (!col_sizes.empty()) {
    CHECK(col_sizes_ptr);
  }

  // execute
  auto timer = DEBUG_TIMER(__func__);
  const auto err =
      compilation_context->getFuncPtr()(byte_stream_ptr,  // input columns buffer
                                        col_sizes_ptr,    // input column sizes
                                        nullptr,
                                        &output_row_count);
  if (err) {
    throw std::runtime_error("Error executing table function: " + std::to_string(err));
  }
  if (exe_unit.table_func.hasCompileTimeOutputSizeConstant()) {
    if (static_cast<size_t>(output_row_count) != mgr->get_nrows()) {
      throw std::runtime_error(
          "Table function with constant sizing parameter must return " +
          std::to_string(mgr->get_nrows()) + " (got " + std::to_string(output_row_count) +
          ")");
    }
  } else {
    if (output_row_count < 0 || (size_t)output_row_count > mgr->get_nrows()) {
      output_row_count = mgr->get_nrows();
    }
  }
  // Update entry count, it may differ from allocated mem size
  if (exe_unit.table_func.hasTableFunctionSpecifiedParameter() && !mgr->query_buffers) {
    // set_output_row_size has not been called
    if (output_row_count == 0) {
      // allocate for empty output columns
      mgr->allocate_output_buffers(0);
    } else {
      throw std::runtime_error("Table function must call set_output_row_size");
    }
  }

  mgr->query_buffers->getResultSet(0)->updateStorageEntryCount(output_row_count);

  const size_t column_size = output_row_count * sizeof(int64_t);
  const size_t allocated_column_size = mgr->get_nrows() * sizeof(int64_t);
  auto group_by_buffers_ptr = mgr->query_buffers->getGroupByBuffersPtr();
  CHECK(group_by_buffers_ptr);
  auto output_buffers_ptr = reinterpret_cast<int64_t*>(group_by_buffers_ptr[0]);

  auto num_out_columns = exe_unit.target_exprs.size();
  int8_t* src = reinterpret_cast<int8_t*>(output_buffers_ptr);
  int8_t* dst = reinterpret_cast<int8_t*>(output_buffers_ptr);
  for (size_t i = 0; i < num_out_columns; i++) {
    if (src != dst) {
      auto t = memmove(dst, src, column_size);
      CHECK_EQ(dst, t);
    }
    src += allocated_column_size;
    dst += column_size;
  }

  return mgr->query_buffers->getResultSetOwned(0);
}

namespace {
enum {
  ERROR_BUFFER,
  COL_BUFFERS,
  COL_SIZES,
  OUTPUT_BUFFERS,
  OUTPUT_ROW_COUNT,
  KERNEL_PARAM_COUNT,
};
}

ResultSetPtr TableFunctionExecutionContext::launchGpuCode(
    const TableFunctionExecutionUnit& exe_unit,
    const TableFunctionCompilationContext* compilation_context,
    std::vector<const int8_t*>& col_buf_ptrs,
    std::vector<int64_t>& col_sizes,
    const size_t elem_count,
    const int device_id,
    Executor* executor) {
#ifdef HAVE_CUDA
  if (exe_unit.table_func.hasTableFunctionSpecifiedParameter()) {
    throw QueryMustRunOnCpu();
  }

  auto num_out_columns = exe_unit.target_exprs.size();
  auto data_mgr = executor->getDataMgr();
  auto gpu_allocator = std::make_unique<CudaAllocator>(data_mgr, device_id);
  CHECK(gpu_allocator);
  std::vector<CUdeviceptr> kernel_params(KERNEL_PARAM_COUNT, 0);

  // setup the inputs
  auto byte_stream_ptr = !(col_buf_ptrs.empty())
                             ? gpu_allocator->alloc(col_buf_ptrs.size() * sizeof(int64_t))
                             : nullptr;
  if (byte_stream_ptr) {
    gpu_allocator->copyToDevice(byte_stream_ptr,
                                reinterpret_cast<int8_t*>(col_buf_ptrs.data()),
                                col_buf_ptrs.size() * sizeof(int64_t));
  }
  kernel_params[COL_BUFFERS] = reinterpret_cast<CUdeviceptr>(byte_stream_ptr);

  auto col_sizes_ptr = !(col_sizes.empty())
                           ? gpu_allocator->alloc(col_sizes.size() * sizeof(int64_t))
                           : nullptr;
  if (col_sizes_ptr) {
    gpu_allocator->copyToDevice(col_sizes_ptr,
                                reinterpret_cast<int8_t*>(col_sizes.data()),
                                col_sizes.size() * sizeof(int64_t));
  }
  kernel_params[COL_SIZES] = reinterpret_cast<CUdeviceptr>(col_sizes_ptr);

  kernel_params[ERROR_BUFFER] =
      reinterpret_cast<CUdeviceptr>(gpu_allocator->alloc(sizeof(int32_t)));
  // initialize output memory
  QueryMemoryDescriptor query_mem_desc(
      executor, elem_count, QueryDescriptionType::Projection, /*is_table_function=*/true);
  query_mem_desc.setOutputColumnar(true);

  for (size_t i = 0; i < num_out_columns; i++) {
    // All outputs padded to 8 bytes
    query_mem_desc.addColSlotInfo({std::make_tuple(8, 8)});
  }
  const auto allocated_output_row_count = get_output_row_count(exe_unit, elem_count);
  auto query_buffers = std::make_unique<QueryMemoryInitializer>(
      exe_unit,
      query_mem_desc,
      device_id,
      ExecutorDeviceType::GPU,
      (allocated_output_row_count == 0 ? 1 : allocated_output_row_count),
      std::vector<std::vector<const int8_t*>>{col_buf_ptrs},
      std::vector<std::vector<uint64_t>>{{0}},  // frag offsets
      row_set_mem_owner_,
      gpu_allocator.get(),
      executor);

  // setup the output
  int64_t output_row_count = allocated_output_row_count;

  kernel_params[OUTPUT_ROW_COUNT] =
      reinterpret_cast<CUdeviceptr>(gpu_allocator->alloc(sizeof(int64_t*)));
  gpu_allocator->copyToDevice(reinterpret_cast<int8_t*>(kernel_params[OUTPUT_ROW_COUNT]),
                              reinterpret_cast<int8_t*>(&output_row_count),
                              sizeof(output_row_count));

  // const unsigned block_size_x = executor->blockSize();
  const unsigned block_size_x = 1;
  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  // const unsigned grid_size_x = executor->gridSize();
  const unsigned grid_size_x = 1;
  const unsigned grid_size_y = 1;
  const unsigned grid_size_z = 1;

  auto gpu_output_buffers = query_buffers->setupTableFunctionGpuBuffers(
      query_mem_desc, device_id, block_size_x, grid_size_x);

  kernel_params[OUTPUT_BUFFERS] = reinterpret_cast<CUdeviceptr>(gpu_output_buffers.ptrs);

  // execute
  CHECK_EQ(static_cast<size_t>(KERNEL_PARAM_COUNT), kernel_params.size());

  std::vector<void*> param_ptrs;
  for (auto& param : kernel_params) {
    param_ptrs.push_back(&param);
  }

  // Get cu func
  const auto gpu_context = compilation_context->getGpuCode();
  CHECK(gpu_context);
  const auto native_code = gpu_context->getNativeCode(device_id);
  auto cu_func = static_cast<CUfunction>(native_code.first);
  checkCudaErrors(cuLaunchKernel(cu_func,
                                 grid_size_x,
                                 grid_size_y,
                                 grid_size_z,
                                 block_size_x,
                                 block_size_y,
                                 block_size_z,
                                 0,  // shared mem bytes
                                 nullptr,
                                 &param_ptrs[0],
                                 nullptr));
  // TODO(adb): read errors

  // read output row count from GPU
  gpu_allocator->copyFromDevice(
      reinterpret_cast<int8_t*>(&output_row_count),
      reinterpret_cast<int8_t*>(kernel_params[OUTPUT_ROW_COUNT]),
      sizeof(int64_t));
  if (exe_unit.table_func.hasNonUserSpecifiedOutputSizeConstant()) {
    if (static_cast<size_t>(output_row_count) != allocated_output_row_count) {
      throw std::runtime_error(
          "Table function with constant sizing parameter must return " +
          std::to_string(allocated_output_row_count) + " (got " +
          std::to_string(output_row_count) + ")");
    }
  } else {
    if (output_row_count < 0 || (size_t)output_row_count > allocated_output_row_count) {
      output_row_count = allocated_output_row_count;
    }
  }

  // Update entry count, it may differ from allocated mem size
  query_buffers->getResultSet(0)->updateStorageEntryCount(output_row_count);

  // Copy back to CPU storage
  query_buffers->copyFromTableFunctionGpuBuffers(data_mgr,
                                                 query_mem_desc,
                                                 output_row_count,
                                                 gpu_output_buffers,
                                                 device_id,
                                                 block_size_x,
                                                 grid_size_x);

  return query_buffers->getResultSetOwned(0);
#else
  UNREACHABLE();
  return nullptr;
#endif
}
