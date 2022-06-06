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

#include "QueryEngine/QueryMemoryInitializer.h"
#include "QueryEngine/Utils/FlatBuffer.h"

/*
  The TableFunctionManager implements the following features:

  - Manage the memory of output column buffers.

  - Allow table functions to communicate error/exception messages up
    to the execution context. Table functions can return with a call
    to `table_function_error` with an error message. This will
    indicate to the execution context that an error ocurred within the
    table function, and the error will be propagated as an exception.
*/

// TableFunctionError encapsulates any runtime errors caused by table function execution.
class TableFunctionError : public std::runtime_error {
 public:
  TableFunctionError(const std::string& message) : std::runtime_error(message) {}
};

// UserTableFunctionErrors represent errors thrown explicitly by user code within table
// functions, i.e. through calling table_function_error()
class UserTableFunctionError : public TableFunctionError {
 public:
  UserTableFunctionError(const std::string& message) : TableFunctionError(message) {}
};

// Use a set negative value to distinguish from already-existing
// negative return values
enum TableFunctionErrorCode : int32_t {
  GenericError = -0x75BCD15,
};

extern std::mutex TableFunctionManager_singleton_mutex;

struct TableFunctionManager {
  std::unique_ptr<QueryMemoryInitializer> query_buffers;

  TableFunctionManager(const TableFunctionExecutionUnit& exe_unit,
                       Executor* executor,
                       std::vector<const int8_t*>& col_buf_ptrs,
                       std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                       bool is_singleton)
      : exe_unit_(exe_unit)
      , executor_(executor)
      , col_buf_ptrs_(col_buf_ptrs)
      , row_set_mem_owner_(row_set_mem_owner)
      , output_num_rows_(-1)
      , is_singleton_(is_singleton)
      , thread_id_(std::this_thread::get_id()) {
    if (isSingleton()) {
      set_singleton(this);  // start of singleton life
    }
    auto num_out_columns = get_ncols();
    output_col_buf_ptrs.reserve(num_out_columns);
    output_column_ptrs.reserve(num_out_columns);
    output_array_values_total_number_.reserve(num_out_columns);
    for (size_t i = 0; i < num_out_columns; i++) {
      output_col_buf_ptrs.emplace_back(nullptr);
      output_column_ptrs.emplace_back(nullptr);
      output_array_values_total_number_.emplace_back(-1);
    }
  }

  // Return the number of output columns
  size_t get_ncols() const { return exe_unit_.target_exprs.size(); }

  // Return the number of rows of output columns.
  size_t get_nrows() const { return output_num_rows_; }

  void check_thread_id() const {
    if (std::this_thread::get_id() != thread_id_) {
      throw std::runtime_error(
          "TableFunctionManager instance accessed from an alien thread!");
    }
  }

  // Store the pointer to output Column instance
  void set_output_column(int32_t index, int8_t* ptr) {
    check_thread_id();
    CHECK(index >= 0 && index < static_cast<int32_t>(get_ncols()));
    CHECK(ptr);
    output_column_ptrs[index] = ptr;
  }

  // Set the total number of array values in a column of arrays
  void set_output_array_values_total_number(int32_t index,
                                            int64_t output_array_values_total_number) {
    CHECK_EQ(output_num_rows_,
             size_t(-1));  // set_output_array_size must be called
                           // before set_output_row_size because
                           // set_output_row_size allocates the output
                           // buffers
    auto num_out_columns = get_ncols();
    CHECK_LE(0, index);
    CHECK_LT(index, num_out_columns);
    output_array_values_total_number_[index] = output_array_values_total_number;
  }

  void allocate_output_buffers(int64_t output_num_rows) {
    check_thread_id();
    CHECK_EQ(output_num_rows_,
             size_t(-1));  // re-allocation of output buffers is not supported

    output_num_rows_ = output_num_rows;
    auto num_out_columns = get_ncols();
    QueryMemoryDescriptor query_mem_desc(executor_,
                                         output_num_rows,  // divide by row multiplier???
                                         QueryDescriptionType::TableFunction,
                                         /*is_table_function=*/true);
    query_mem_desc.setOutputColumnar(true);

    for (size_t i = 0; i < num_out_columns; i++) {
      // All outputs have padded width set to logical column width
      auto ti = exe_unit_.target_exprs[i]->get_type_info();
      if (ti.is_array()) {
        const size_t array_item_width = ti.get_elem_type().get_size();
        CHECK_NE(output_array_values_total_number_[i],
                 -1);  // set_output_array_values_total_number(i, ...) is not called
        /*
          Here we compute the byte size of flatbuffer and store it in
          query memory descriptor's ColSlotContext instance. The
          flatbuffer memory will be allocated in
          QueryMemoryInitializer constructor and the memory will be
          initialized below.
         */
        const int64_t flatbuffer_size =
            FlatBufferManager::get_VarlenArray_flatbuffer_size(
                output_num_rows_, output_array_values_total_number_[i], array_item_width);
        query_mem_desc.addColSlotInfoFlatBuffer(
            flatbuffer_size);  // used by QueryMemoryInitializer
      } else {
        const size_t col_width = ti.get_size();
        query_mem_desc.addColSlotInfo({std::make_tuple(col_width, col_width)});
      }
    }

    // The members layout of Column must match with Column defined in
    // heavydbTypes.h
    struct Column {
      int8_t* ptr;
      int64_t size;
      // just for debugging:
      std::string toString() const {
        return "Column{" + ::toString(ptr) + ", " + ::toString(size) + "}";
      }
    };
    // We do not init output buffers for CPU currently, so CPU
    // table functions are expected to handle their own initialization
    query_buffers = std::make_unique<QueryMemoryInitializer>(
        exe_unit_,
        query_mem_desc,
        /*device_id=*/0,
        ExecutorDeviceType::CPU,
        (output_num_rows_ == 0 ? 1 : output_num_rows_),
        std::vector<std::vector<const int8_t*>>{col_buf_ptrs_},
        std::vector<std::vector<uint64_t>>{{0}},  // frag offsets
        row_set_mem_owner_,
        nullptr,
        executor_);
    if (output_num_rows_ != 0) {
      auto group_by_buffers_ptr = query_buffers->getGroupByBuffersPtr();
      CHECK(group_by_buffers_ptr);
      auto output_buffers_ptr = reinterpret_cast<int8_t*>(group_by_buffers_ptr[0]);
      for (size_t i = 0; i < num_out_columns; i++) {
        Column* col = reinterpret_cast<Column*>(output_column_ptrs[i]);
        CHECK(col);
        // set the members of output Column instances:
        output_col_buf_ptrs[i] = reinterpret_cast<int64_t*>(output_buffers_ptr);
        col->ptr = output_buffers_ptr;
        col->size = output_num_rows_;

        auto ti = exe_unit_.target_exprs[i]->get_type_info();
        if (ti.is_array()) {
          const size_t array_item_width = ti.get_elem_type().get_size();
          FlatBufferManager m{output_buffers_ptr};
          m.initializeVarlenArray(
              output_num_rows_, output_array_values_total_number_[i], array_item_width);
          output_buffers_ptr = align_to_int64(output_buffers_ptr + m.flatbufferSize());
        } else {
          const size_t col_width = ti.get_size();
          output_buffers_ptr =
              align_to_int64(output_buffers_ptr + col_width * output_num_rows_);
        }
      }
    }
  }

  const char* get_error_message() const {
    check_thread_id();
    return error_message_.c_str();
  }

  void set_error_message(const char* msg) {
    check_thread_id();
    error_message_ = std::string(msg);
  }

  // Methods for managing singleton instance of TableFunctionManager:

  bool isSingleton() const { return is_singleton_; }

  ~TableFunctionManager() {
    if (isSingleton()) {
      set_singleton(nullptr);  // end of singleton life
    }
  }

  static TableFunctionManager*& get_singleton() {
    static TableFunctionManager* instance = nullptr;
    return instance;
  }

 private:
  void lock() { TableFunctionManager_singleton_mutex.lock(); }
  void unlock() { TableFunctionManager_singleton_mutex.unlock(); }

  static void set_singleton(TableFunctionManager* instance) {
    auto& instance_ = get_singleton();
    // ensure being singleton and lock/unlock
    if (instance) {
      instance->lock();
      CHECK(instance_ == nullptr);
    } else {
      CHECK(instance_ != nullptr);
      instance_->unlock();
    }
    instance_ = instance;
  }

  const TableFunctionExecutionUnit& exe_unit_;
  Executor* executor_;
  // Pointers to the buffers of input Columns
  std::vector<const int8_t*>& col_buf_ptrs_;
  //
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  // Pointers to the buffers of output Columns
  std::vector<int64_t*> output_col_buf_ptrs;
  // Number of rows of output Columns
  size_t output_num_rows_;
  // Total number of array values in the output columns of arrays
  std::vector<int64_t> output_array_values_total_number_;
  // Pointers to output Column instances
  std::vector<int8_t*> output_column_ptrs;
  // If TableFunctionManager is global
  bool is_singleton_;
  // Store thread id for sanity check
  std::thread::id thread_id_;
  // Error message
  std::string error_message_;
};
