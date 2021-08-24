/*
 * Copyright 2021 OmniSci, Inc.
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

#include "QueryEngine/QueryMemoryInitializer.h"

/*
  QueryOutputBufferMemoryManager manages the memory of output column
  buffers using the following approach:

  1. Before executing a table function, a memory manager of output
  columns buffers is initiated (when entering launchCpuCode). This
  step includes creating output Column instances that will passed as
  arguments to the table functions. The pointers to the output Column
  instances are saved within the memory manager so that the instances
  ptr and size members can be updated when the number of output
  columns rows is set and the buffers of output columns are allocated.

  2. If the number of rows of output columns is known,
  `set_output_row_size` is called before executing the table
  function. Otherwise (when
  exe_unit.table_func.hasTableFunctionSpecifiedParameter() evaluates
  to true), it is expected that the table function will call
  `set_output_row_size` to set the number of rows of output columns.
  Calling `set_output_row_size` will also update the output Column
  instances members.

  3. The table function is expected to return the final number of rows
  of output columns. It must not be greater than the count specified
  in the `set_output_row_size` call but may be a smaller value.

  4. After returning the table function, one can access the output
  Column instances until the memory manager of output columns is
  destroyed (when leaving launchCpuCode). The buffers of output
  columns are now owned by the ResultSet instance.

*/

struct QueryOutputBufferMemoryManager {
  std::unique_ptr<QueryMemoryInitializer> query_buffers;

  /*
    QueryOutputBufferMemoryManager is a dynamic singleton:
    `get_singleton()` returns a pointer to the singleton instance when
    in the scope of QueryOutputBufferMemoryManager life-time,
    otherwise returns nullptr. For internal usage.
  */
  static QueryOutputBufferMemoryManager*& get_singleton() {
    static QueryOutputBufferMemoryManager* instance = nullptr;
    return instance;
  }

  QueryOutputBufferMemoryManager(const TableFunctionExecutionUnit& exe_unit,
                                 Executor* executor,
                                 std::vector<const int8_t*>& col_buf_ptrs,
                                 std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
      : exe_unit_(exe_unit)
      , executor_(executor)
      , col_buf_ptrs_(col_buf_ptrs)
      , row_set_mem_owner_(row_set_mem_owner)
      , output_num_rows_(-1) {
    set_singleton(this);  // start of singleton life
    auto num_out_columns = get_ncols();
    output_col_buf_ptrs.reserve(num_out_columns);
    output_column_ptrs.reserve(num_out_columns);
    for (size_t i = 0; i < num_out_columns; i++) {
      output_col_buf_ptrs.emplace_back(nullptr);
      output_column_ptrs.emplace_back(nullptr);
    }
  }

  ~QueryOutputBufferMemoryManager() {
    set_singleton(nullptr);  // end of singleton life
  }

  // Return the number of output columns
  size_t get_ncols() const { return exe_unit_.target_exprs.size(); }

  // Return the number of rows of output columns.
  size_t get_nrows() const { return output_num_rows_; }

  // Store the pointer to output Column instance
  void set_output_column(int32_t index, int8_t* ptr) {
    CHECK(index >= 0 && index < static_cast<int32_t>(get_ncols()));
    CHECK(ptr);
    output_column_ptrs[index] = ptr;
  }

  void allocate_output_buffers(int64_t output_num_rows) {
    CHECK_EQ(output_num_rows_,
             size_t(-1));  // re-allocation of output buffers is not supported
    output_num_rows_ = output_num_rows;
    auto num_out_columns = get_ncols();
    QueryMemoryDescriptor query_mem_desc(executor_,
                                         output_num_rows,  // divide by row multiplier???
                                         QueryDescriptionType::Projection,
                                         /*is_table_function=*/true);
    query_mem_desc.setOutputColumnar(true);

    for (size_t i = 0; i < num_out_columns; i++) {
      // All outputs padded to 8 bytes
      query_mem_desc.addColSlotInfo({std::make_tuple(8, 8)});
    }

    // The members layout of Column must match with Column defined in
    // OmniSciTypes.h
    struct Column {
      int8_t* ptr;
      int64_t size;
      // just for debugging:
      std::string toString() const {
        return "Column{" + ::toString(ptr) + ", " + ::toString(size) + "}";
      }
    };

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
      auto output_buffers_ptr = reinterpret_cast<int64_t*>(group_by_buffers_ptr[0]);
      for (size_t i = 0; i < num_out_columns; i++) {
        Column* col = reinterpret_cast<Column*>(output_column_ptrs[i]);
        CHECK(col);
        // set the members of output Column instances:
        output_col_buf_ptrs[i] = output_buffers_ptr + i * output_num_rows_;
        col->ptr = reinterpret_cast<int8_t*>(output_col_buf_ptrs[i]);
        col->size = output_num_rows_;
      }
    }
  }

 private:
  static void set_singleton(QueryOutputBufferMemoryManager* instance) {
    auto& instance_ = get_singleton();
    // ensure being singleton
    if (instance) {
      CHECK(instance_ == nullptr);
    } else {
      CHECK(instance_ != nullptr);
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
  // Pointers to output Column instances
  std::vector<int8_t*> output_column_ptrs;
};
