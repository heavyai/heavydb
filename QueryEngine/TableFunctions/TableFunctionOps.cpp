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

#include "QueryEngine/TableFunctions/TableFunctionManager.h"

/*
  set_output_row_size sets the row size of output Columns and
  allocates the corresponding column buffers.

  `set_output_row_size` can be called exactly one time when entering a
  table function (when not using TableFunctionSpecifiedParameter
  sizer) or within a table function (when using
  TableFunctionSpecifiedParameter sizer).

  For thread-safety, it is recommended to have `TableFunctionManager
  mgr` as a first argument in a UDTF definition and then use
  `mgr.set_output_row_size(..)` to set the row size of output columns,
*/

extern "C" DEVICE RUNTIME_EXPORT void TableFunctionManager_set_output_row_size(
    int8_t* mgr_ptr,
    int64_t num_rows) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  if (num_rows < 0) {
    throw std::runtime_error(
        "set_output_row_size: expected non-negative row size but got " +
        std::to_string(num_rows));
  }
  mgr->allocate_output_buffers(num_rows);
}

extern "C" DEVICE RUNTIME_EXPORT void set_output_row_size(int64_t num_rows) {
  auto& mgr = TableFunctionManager::get_singleton();
  TableFunctionManager_set_output_row_size(reinterpret_cast<int8_t*>(mgr), num_rows);
}

/*
  TableFunctionManager_register_output_column stores the pointer of
  output Column instance so that when the buffers of output columns
  are allocated, the Column instance members ptr_ and size_ can be
  updated.

  TableFunctionManager_register_output_column is used internally when
  creating output Column instances prior entering the table function.
 */

extern "C" DEVICE RUNTIME_EXPORT void
TableFunctionManager_register_output_column(int8_t* mgr_ptr, int32_t index, int8_t* ptr) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  mgr->set_output_column(index, ptr);
}

/*
  table_function_error allows code from within a table function to
  set an error message that can be accessed from the execution context.
  This allows the error message to be propagated as an exception message.
 */
extern "C" DEVICE RUNTIME_EXPORT int32_t
TableFunctionManager_error_message(int8_t* mgr_ptr, const char* message) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  if (message != nullptr) {
    mgr->set_error_message(message);
  } else {
    mgr->set_error_message("no error message set");
  }
  return TableFunctionError::GenericError;
}

extern "C" DEVICE RUNTIME_EXPORT int32_t table_function_error(const char* message) {
  auto& mgr = TableFunctionManager::get_singleton();
  return TableFunctionManager_error_message(reinterpret_cast<int8_t*>(mgr), message);
}
/*
  TableFunctionManager_get_singleton is used internally to get the
  pointer to global singleton of TableFunctionManager, if initialized,
  otherwise throws runtime error.
*/
extern "C" DEVICE RUNTIME_EXPORT int8_t* TableFunctionManager_get_singleton() {
  auto& mgr = TableFunctionManager::get_singleton();
  if (!mgr) {
    throw std::runtime_error("uninitialized TableFunctionManager singleton");
  }
  return reinterpret_cast<int8_t*>(mgr);
}

#endif  // EXECUTE_INCLUDE
