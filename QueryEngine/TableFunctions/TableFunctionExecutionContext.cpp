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

#include "QueryEngine/TableFunctions/TableFunctionExecutionContext.h"

#include "Analyzer/Analyzer.h"
#include "Logger/Logger.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/TableFunctions/TableFunctionCompilationContext.h"
#include "QueryEngine/TableFunctions/TableFunctionManager.h"
#include "QueryEngine/Utils/FlatBuffer.h"
#include "Shared/funcannotations.h"

namespace {

template <typename T>
const int8_t* create_literal_buffer(const T literal,
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
          gpu_literal_buf_ptr, reinterpret_cast<const int8_t*>(&literal), sizeof(T));
      return gpu_literal_buf_ptr;
    }
  }
  UNREACHABLE();
  return nullptr;
}

// Specialization for std::string. Currently we simply hand the UDTF a char* to the
// first char of a c-style null-terminated string we copy out of the std::string.
// May want to evaluate moving to sending in the ptr and size
template <>
const int8_t* create_literal_buffer(std::string* const literal,
                                    const ExecutorDeviceType device_type,
                                    std::vector<std::unique_ptr<char[]>>& literals_owner,
                                    CudaAllocator* gpu_allocator) {
  const int64_t string_size = literal->size();
  const int64_t padded_string_size =
      (string_size + 7) / 8 * 8;  // round up to the next multiple of 8
  switch (device_type) {
    case ExecutorDeviceType::CPU: {
      literals_owner.emplace_back(
          std::make_unique<char[]>(sizeof(int64_t) + padded_string_size));
      std::memcpy(literals_owner.back().get(), &string_size, sizeof(int64_t));
      std::memcpy(
          literals_owner.back().get() + sizeof(int64_t), literal->data(), string_size);
      return reinterpret_cast<const int8_t*>(literals_owner.back().get());
    }
    case ExecutorDeviceType::GPU: {
      CHECK(gpu_allocator);
      const auto gpu_literal_buf_ptr =
          gpu_allocator->alloc(sizeof(int64_t) + padded_string_size);
      gpu_allocator->copyToDevice(gpu_literal_buf_ptr,
                                  reinterpret_cast<const int8_t*>(&string_size),
                                  sizeof(int64_t));
      gpu_allocator->copyToDevice(gpu_literal_buf_ptr + sizeof(int64_t),
                                  reinterpret_cast<const int8_t*>(literal->data()),
                                  string_size);
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
    case table_functions::OutputBufferSizeType::kUserSpecifiedConstantParameter:
    case table_functions::OutputBufferSizeType::kPreFlightParameter: {
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
    const std::shared_ptr<CompilationContext>& compilation_context,
    const ColumnFetcher& column_fetcher,
    const ExecutorDeviceType device_type,
    Executor* executor,
    bool is_pre_launch_udtf) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(compilation_context);
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<std::unique_ptr<char[]>> literals_owner;

  const int device_id = 0;  // TODO(adb): support multi-gpu table functions
  std::unique_ptr<CudaAllocator> device_allocator;
  if (device_type == ExecutorDeviceType::GPU) {
    auto data_mgr = executor->getDataMgr();
    device_allocator.reset(new CudaAllocator(
        data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id)));
  }
  std::vector<const int8_t*> col_buf_ptrs;
  std::vector<int64_t> col_sizes;
  std::vector<const int8_t*> input_str_dict_proxy_ptrs;
  std::optional<size_t> input_num_rows;

  int col_index = -1;
  // TODO: col_list_bufs are allocated on CPU memory, so UDTFs with column_list
  // arguments are not supported on GPU atm.
  std::vector<std::vector<const int8_t*>> col_list_bufs;
  std::vector<std::vector<const int8_t*>> input_col_list_str_dict_proxy_ptrs;
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
      // We use the number of entries in the first column to be the number of rows to base
      // the output off of (optionally depending on the sizing parameter)
      if (!input_num_rows) {
        input_num_rows = (buf_elem_count > 0 ? buf_elem_count : 1);
      }

      int8_t* input_str_dict_proxy_ptr = nullptr;
      if (ti.is_subtype_dict_encoded_string()) {
        const auto input_string_dictionary_proxy = executor->getStringDictionaryProxy(
            ti.get_comp_param(), executor->getRowSetMemoryOwner(), true);
        input_str_dict_proxy_ptr =
            reinterpret_cast<int8_t*>(input_string_dictionary_proxy);
      }
      if (ti.is_column_list()) {
        if (col_index == -1) {
          col_list_bufs.push_back({});
          input_col_list_str_dict_proxy_ptrs.push_back({});
          col_list_bufs.back().reserve(ti.get_dimension());
          input_col_list_str_dict_proxy_ptrs.back().reserve(ti.get_dimension());
        } else {
          CHECK_EQ(col_sizes.back(), buf_elem_count);
        }
        col_index++;
        col_list_bufs.back().push_back(col_buf);
        input_col_list_str_dict_proxy_ptrs.back().push_back(input_str_dict_proxy_ptr);
        // append col_buf to column_list col_buf
        if (col_index + 1 == ti.get_dimension()) {
          col_index = -1;
        }
        // columns in the same column_list point to column_list data
        col_buf_ptrs.push_back((const int8_t*)col_list_bufs.back().data());
        input_str_dict_proxy_ptrs.push_back(
            (const int8_t*)input_col_list_str_dict_proxy_ptrs.back().data());
      } else {
        col_buf_ptrs.push_back(col_buf);
        input_str_dict_proxy_ptrs.push_back(input_str_dict_proxy_ptr);
      }
      col_sizes.push_back(buf_elem_count);
    } else if (const auto& constant_val = dynamic_cast<Analyzer::Constant*>(input_expr)) {
      // TODO(adb): Unify literal handling with rest of system, either in Codegen or as a
      // separate serialization component
      col_sizes.push_back(0);
      input_str_dict_proxy_ptrs.push_back(nullptr);
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
      } else if (ti.is_integer() || ti.is_timestamp() || ti.is_timeinterval()) {
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
      } else if (ti.is_bytes()) {  // text encoding none string
        col_buf_ptrs.push_back(create_literal_buffer(const_val_datum.stringval,
                                                     device_type,
                                                     literals_owner,
                                                     device_allocator.get()));
      } else {
        throw TableFunctionError("Literal value " + constant_val->toString() +
                                 " is not yet supported.");
      }
    } else {
      throw TableFunctionError(
          "Unsupported expression as input to table function: " + input_expr->toString() +
          "\n Only literal constants and columns are supported!");
    }
  }
  CHECK_EQ(col_buf_ptrs.size(), exe_unit.input_exprs.size());
  CHECK_EQ(col_sizes.size(), exe_unit.input_exprs.size());
  if (!exe_unit.table_func
           .hasOutputSizeIndependentOfInputSize()) {  // includes compile-time constants,
                                                      // user-specified constants,
                                                      // and runtime table funtion
                                                      // specified sizing, only
                                                      // user-specified row-multipliers
                                                      // currently take into account input
                                                      // row size
    CHECK(input_num_rows);
  }
  std::vector<int8_t*> output_str_dict_proxy_ptrs;
  for (const auto& output_expr : exe_unit.target_exprs) {
    int8_t* output_str_dict_proxy_ptr = nullptr;
    auto ti = output_expr->get_type_info();
    if (ti.is_dict_encoded_string()) {
      const auto output_string_dictionary_proxy = executor->getStringDictionaryProxy(
          ti.get_comp_param(), executor->getRowSetMemoryOwner(), true);
      output_str_dict_proxy_ptr =
          reinterpret_cast<int8_t*>(output_string_dictionary_proxy);
    }
    output_str_dict_proxy_ptrs.emplace_back(output_str_dict_proxy_ptr);
  }

  if (is_pre_launch_udtf) {
    CHECK(exe_unit.table_func.containsPreFlightFn());
    launchPreCodeOnCpu(
        exe_unit,
        std::dynamic_pointer_cast<CpuCompilationContext>(compilation_context),
        col_buf_ptrs,
        col_sizes,
        *input_num_rows,
        executor);
    return nullptr;
  } else {
    switch (device_type) {
      case ExecutorDeviceType::CPU:
        return launchCpuCode(
            exe_unit,
            std::dynamic_pointer_cast<CpuCompilationContext>(compilation_context),
            col_buf_ptrs,
            col_sizes,
            input_str_dict_proxy_ptrs,
            *input_num_rows,
            output_str_dict_proxy_ptrs,
            executor);
      case ExecutorDeviceType::GPU:
        return launchGpuCode(
            exe_unit,
            std::dynamic_pointer_cast<GpuCompilationContext>(compilation_context),
            col_buf_ptrs,
            col_sizes,
            input_str_dict_proxy_ptrs,
            *input_num_rows,
            output_str_dict_proxy_ptrs,
            /*device_id=*/0,
            executor);
    }
  }
  UNREACHABLE();
  return nullptr;
}

std::mutex TableFunctionManager_singleton_mutex;

void TableFunctionExecutionContext::launchPreCodeOnCpu(
    const TableFunctionExecutionUnit& exe_unit,
    const std::shared_ptr<CpuCompilationContext>& compilation_context,
    std::vector<const int8_t*>& col_buf_ptrs,
    std::vector<int64_t>& col_sizes,
    const size_t elem_count,  // taken from first source only currently
    Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  int64_t output_row_count = 0;

  // If TableFunctionManager must be a singleton but it has been
  // initialized from another thread, TableFunctionManager constructor
  // blocks via TableFunctionManager_singleton_mutex until the
  // existing singleton is deconstructed.
  auto mgr = std::make_unique<TableFunctionManager>(
      exe_unit,
      executor,
      col_buf_ptrs,
      row_set_mem_owner_,
      /*is_singleton=*/!exe_unit.table_func.usesManager());

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
  const auto err = compilation_context->table_function_entry_point()(
      reinterpret_cast<const int8_t*>(mgr.get()),
      byte_stream_ptr,  // input columns buffer
      col_sizes_ptr,    // input column sizes
      nullptr,  // input string dictionary proxy ptrs - not supported for pre-flights yet
      nullptr,
      nullptr,  // output string dictionary proxy ptrs - not supported for pre-flights yet
      &output_row_count);

  if (exe_unit.table_func.hasPreFlightOutputSizer()) {
    exe_unit.output_buffer_size_param = output_row_count;
  }

  if (err == TableFunctionErrorCode::GenericError) {
    throw UserTableFunctionError("Error executing table function pre flight check: " +
                                 std::string(mgr->get_error_message()));
  } else if (err) {
    throw UserTableFunctionError("Error executing table function pre flight check: " +
                                 std::to_string(err));
  }
}

// clang-format off
/*
  Managing the output buffers from table functions
  ------------------------------------------------

  In general, the results of a query (a list of columns) is hold by a
  ResultSet instance. While ResultSet is a rather complicated
  structure, its most important members are

    std::vector<TargetInfo> targets_ that holds the type of output
      columns (recall: `struct TargetInfo {..., SQLTypeInfo sql_type,
      ...};`)

    std::unique_ptr<ResultSetStorage> storage_ that stores the
      underlying buffer for a result set (recall: `struct
      ResultSetStorage {..., int8_t* buff_, ...};`)

    QueryMemoryDescriptor query_mem_desc_ that describes the format of
      the storage for a result set.

  QueryMemoryDescriptor structure contains the following relevant
  members:

    QueryDescriptionType query_desc_type_ is equal to one of
      GroupByPerfectHash, GroupByBaselineHash, Projection,
      TableFunction, NonGroupedAggregate, Estimator. In the following,
      we assume query_desc_type_ == TableFunction.

    bool output_columnar_ is always true for table function result
      sets.

    size_t entry_count_ is the number of entries in the storage
      buffer. This typically corresponds to the number of output rows.

    ColSlotContext col_slot_context_ describes the internal structure
      of the storage buffer using the following members:

        std::vector<SlotSize> slot_sizes_ where we have `struct SlotSize
          { int8_t padded_size; int8_t logical_size; };`

        std::vector<std::vector<size_t>> col_to_slot_map_ describes the
          mapping of a column to possibly multiple slots.

        std::unordered_map<SlotIndex, ArraySize> varlen_output_slot_map_

  In the case of table function result sets, the QueryMemoryDescriptor
  instance is created in TableFunctionManager::allocate_output_buffers
  method and we have query_desc_type_ == TableFunction.

  Depending on the target info of an output column, the internal
  structure of the storage buffer has two variants:

    - traditional where the buffer size of a particular column is
      described by entry_count_ and
      col_slot_context_.slot_sizes_. This variant is used for output
      columns of fixed-width scalar types such as integers, floats,
      boolean, text encoded dicts, etc. For the corresponding column
      with col_idx, we have

        col_to_slot_map_[col_idx] == {slot_idx}
        slot_sizes_[slot_idx] == {column_width, column_width}

      where column_width is targets_[col_idx].sql_type.get_size().

    - flatbuffer where the buffer size of a particular column is
      described by varlen_output_slot_map_. This variant is used for
      output columns of variable length composite types such as arrays
      of ints, floats, etc. For the corresponding column with col_idx,
      we have

        col_to_slot_map_[col_idx] == {slot_idx}
        slot_sizes_[slot_idx] == {0, 0}
        varlen_output_slot_map_ contains an item col_idx:flatbuffer_size

  Only table functions produce result sets that may contain both
  variants. The variants can be distinguished via
  `getPaddedSlotWidthBytes(slot_idx) == 0` test.

  In the case of table function result sets, col_idx == slot_idx holds.

*/
// clang-format on

ResultSetPtr TableFunctionExecutionContext::launchCpuCode(
    const TableFunctionExecutionUnit& exe_unit,
    const std::shared_ptr<CpuCompilationContext>& compilation_context,
    std::vector<const int8_t*>& col_buf_ptrs,
    std::vector<int64_t>& col_sizes,
    std::vector<const int8_t*>& input_str_dict_proxy_ptrs,
    const size_t elem_count,  // taken from first source only currently
    std::vector<int8_t*>& output_str_dict_proxy_ptrs,
    Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  int64_t output_row_count = 0;

  // If TableFunctionManager must be a singleton but it has been
  // initialized from another thread, TableFunctionManager constructor
  // blocks via TableFunctionManager_singleton_mutex until the
  // existing singleton is deconstructed.
  auto mgr = std::make_unique<TableFunctionManager>(
      exe_unit,
      executor,
      col_buf_ptrs,
      row_set_mem_owner_,
      /*is_singleton=*/!exe_unit.table_func.usesManager());

  if (exe_unit.table_func.hasOutputSizeKnownPreLaunch()) {
    // allocate output buffers because the size is known up front, from
    // user specified parameters (and table size in the case of a user
    // specified row multiplier)
    output_row_count = get_output_row_count(exe_unit, elem_count);
  } else if (exe_unit.table_func.hasPreFlightOutputSizer()) {
    output_row_count = exe_unit.output_buffer_size_param;
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
  const auto input_str_dict_proxy_byte_stream_ptr =
      !input_str_dict_proxy_ptrs.empty()
          ? reinterpret_cast<const int8_t**>(input_str_dict_proxy_ptrs.data())
          : nullptr;

  const auto output_str_dict_proxy_byte_stream_ptr =
      !output_str_dict_proxy_ptrs.empty()
          ? reinterpret_cast<int8_t**>(output_str_dict_proxy_ptrs.data())
          : nullptr;

  // execute
  const auto err = compilation_context->table_function_entry_point()(
      reinterpret_cast<const int8_t*>(mgr.get()),
      byte_stream_ptr,                       // input columns buffer
      col_sizes_ptr,                         // input column sizes
      input_str_dict_proxy_byte_stream_ptr,  // input str dictionary proxies
      nullptr,
      output_str_dict_proxy_byte_stream_ptr,
      &output_row_count);

  if (err == TableFunctionErrorCode::GenericError) {
    throw UserTableFunctionError("Error executing table function: " +
                                 std::string(mgr->get_error_message()));
  }

  else if (err) {
    throw UserTableFunctionError("Error executing table function: " +
                                 std::to_string(err));
  }

  if (exe_unit.table_func.hasCompileTimeOutputSizeConstant()) {
    if (static_cast<size_t>(output_row_count) != mgr->get_nrows()) {
      throw TableFunctionError(
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
      throw TableFunctionError("Table function must call set_output_row_size");
    }
  }

  mgr->query_buffers->getResultSet(0)->updateStorageEntryCount(output_row_count);

  auto group_by_buffers_ptr = mgr->query_buffers->getGroupByBuffersPtr();
  CHECK(group_by_buffers_ptr);
  auto output_buffers_ptr = reinterpret_cast<int64_t*>(group_by_buffers_ptr[0]);

  auto num_out_columns = exe_unit.target_exprs.size();
  int8_t* src = reinterpret_cast<int8_t*>(output_buffers_ptr);
  int8_t* dst = reinterpret_cast<int8_t*>(output_buffers_ptr);
  // Todo (todd): Consolidate this column byte offset logic that occurs in at least 4
  // places
  for (size_t col_idx = 0; col_idx < num_out_columns; col_idx++) {
    auto ti = exe_unit.target_exprs[col_idx]->get_type_info();
    if (ti.is_array()) {
      // TODO: implement FlatBuffer normalization when the
      // max_nof_values is larger than the nof specified values.
      //
      // TODO: implement flatbuffer resize when output_row_count < mgr->get_nrows()
      CHECK_EQ(mgr->get_nrows(), output_row_count);
      FlatBufferManager m{src};
      const size_t allocated_column_size = m.flatbufferSize();
      const size_t actual_column_size = allocated_column_size;
      src = align_to_int64(src + allocated_column_size);
      dst = align_to_int64(dst + actual_column_size);
      if (ti.is_text_encoding_dict_array()) {
        CHECK_EQ(m.getDTypeMetadataDictId(),
                 ti.get_comp_param());  // ensure that dict_id is preserved
      }
    } else {
      const size_t target_width = ti.get_size();
      const size_t allocated_column_size = target_width * mgr->get_nrows();
      const size_t actual_column_size = target_width * output_row_count;
      if (src != dst) {
        auto t = memmove(dst, src, actual_column_size);
        CHECK_EQ(dst, t);
      }
      src = align_to_int64(src + allocated_column_size);
      dst = align_to_int64(dst + actual_column_size);
    }
  }
  return mgr->query_buffers->getResultSetOwned(0);
}

namespace {
enum {
  MANAGER,
  ERROR_BUFFER,
  COL_BUFFERS,
  COL_SIZES,
  INPUT_STR_DICT_PROXIES,
  OUTPUT_BUFFERS,
  OUTPUT_STR_DICT_PROXIES,
  OUTPUT_ROW_COUNT,
  KERNEL_PARAM_COUNT,
};
}

ResultSetPtr TableFunctionExecutionContext::launchGpuCode(
    const TableFunctionExecutionUnit& exe_unit,
    const std::shared_ptr<GpuCompilationContext>& compilation_context,
    std::vector<const int8_t*>& col_buf_ptrs,
    std::vector<int64_t>& col_sizes,
    std::vector<const int8_t*>& input_str_dict_proxy_ptrs,
    const size_t elem_count,
    std::vector<int8_t*>& output_str_dict_proxy_ptrs,
    const int device_id,
    Executor* executor) {
#ifdef HAVE_CUDA
  auto timer = DEBUG_TIMER(__func__);
  if (exe_unit.table_func.hasTableFunctionSpecifiedParameter()) {
    throw QueryMustRunOnCpu();
  }

  auto num_out_columns = exe_unit.target_exprs.size();
  auto data_mgr = executor->getDataMgr();
  auto gpu_allocator = std::make_unique<CudaAllocator>(
      data_mgr, device_id, getQueryEngineCudaStreamForDevice(device_id));
  CHECK(gpu_allocator);
  std::vector<CUdeviceptr> kernel_params(KERNEL_PARAM_COUNT, 0);

  // TODO: implement table function manager for CUDA
  // kernels. kernel_params[MANAGER] ought to contain a device pointer
  // to a struct that a table function kernel with a
  // TableFunctionManager argument can access from the device.
  kernel_params[MANAGER] =
      reinterpret_cast<CUdeviceptr>(gpu_allocator->alloc(sizeof(int8_t*)));

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
  QueryMemoryDescriptor query_mem_desc(executor,
                                       elem_count,
                                       QueryDescriptionType::TableFunction,
                                       /*is_table_function=*/true);
  query_mem_desc.setOutputColumnar(true);

  for (size_t i = 0; i < num_out_columns; i++) {
    const size_t col_width = exe_unit.target_exprs[i]->get_type_info().get_size();
    query_mem_desc.addColSlotInfo({std::make_tuple(col_width, col_width)});
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
  /*
 ￼ TODO: RBC generated runtime table functions do not support
   concurrent execution on a CUDA device. Hence, we'll force 1 as
   block/grid size in the case of runtime table functions.  To support
   this, in RBC, we'll need to expose threadIdx/blockIdx/blockDim to
   runtime table functions and these must do something sensible with
   this information..
 */
  const unsigned block_size_x =
      (exe_unit.table_func.isRuntime() ? 1 : executor->blockSize());
  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  const unsigned grid_size_x =
      (exe_unit.table_func.isRuntime() ? 1 : executor->gridSize());
  const unsigned grid_size_y = 1;
  const unsigned grid_size_z = 1;

  auto gpu_output_buffers =
      query_buffers->setupTableFunctionGpuBuffers(query_mem_desc,
                                                  device_id,
                                                  block_size_x,
                                                  grid_size_x,
                                                  true /* zero_initialize_buffers */);

  kernel_params[OUTPUT_BUFFERS] = reinterpret_cast<CUdeviceptr>(gpu_output_buffers.ptrs);

  // execute
  CHECK_EQ(static_cast<size_t>(KERNEL_PARAM_COUNT), kernel_params.size());

  std::vector<void*> param_ptrs;
  for (auto& param : kernel_params) {
    param_ptrs.push_back(&param);
  }

  // Get cu func

  CHECK(compilation_context);
  const auto native_code = compilation_context->getNativeCode(device_id);
  auto cu_func = static_cast<CUfunction>(native_code.first);
  auto qe_cuda_stream = getQueryEngineCudaStreamForDevice(device_id);
  VLOG(1) << "Launch GPU table function kernel compiled with the following block and "
             "grid sizes: "
          << block_size_x << " and " << grid_size_x;
  checkCudaErrors(cuLaunchKernel(cu_func,
                                 grid_size_x,
                                 grid_size_y,
                                 grid_size_z,
                                 block_size_x,
                                 block_size_y,
                                 block_size_z,
                                 0,  // shared mem bytes
                                 qe_cuda_stream,
                                 &param_ptrs[0],
                                 nullptr));
  checkCudaErrors(cuStreamSynchronize(qe_cuda_stream));

  // read output row count from GPU
  gpu_allocator->copyFromDevice(
      reinterpret_cast<int8_t*>(&output_row_count),
      reinterpret_cast<int8_t*>(kernel_params[OUTPUT_ROW_COUNT]),
      sizeof(int64_t));
  if (exe_unit.table_func.hasNonUserSpecifiedOutputSize()) {
    if (static_cast<size_t>(output_row_count) != allocated_output_row_count) {
      throw TableFunctionError(
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
