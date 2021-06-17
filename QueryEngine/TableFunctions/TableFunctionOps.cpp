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

#ifdef EXECUTE_INCLUDE

#include "QueryEngine/TableFunctions/QueryOutputBufferMemoryManager.h"

/*
  set_output_row_size sets the row size of output Columns and
  allocates the corresponding column buffers.

  `set_output_row_size` can be called exactly one time when entering a
  table function (when not using TableFunctionSpecifiedParameter
  sizer) or within a table function (when using
  TableFunctionSpecifiedParameter sizer).
*/
extern "C" DEVICE RUNTIME_EXPORT void set_output_row_size(int64_t num_rows) {
  auto& mgr = QueryOutputBufferMemoryManager::get_singleton();
  CHECK(mgr != nullptr);  // failure means that set_output_row_size
                          // is called out-of-scope of
                          // QueryOutputBufferMemoryManager usage
  CHECK_GE(num_rows, 0);
  mgr->allocate_output_buffers(num_rows);
}

/*
  register_output_column stores the pointer of output Column instance
  so that when the buffers of output columns are allocated, the Column
  instance members ptr_ and size_ can be updated.

  register_output_column is used internally when creating output
  Column instances prior entering the table function.
 */
extern "C" DEVICE RUNTIME_EXPORT void register_output_column(int32_t index, int8_t* ptr) {
  auto& mgr = QueryOutputBufferMemoryManager::get_singleton();
  CHECK(mgr);
  mgr->set_output_column(index, ptr);
}

#endif  // EXECUTE_INCLUDE
