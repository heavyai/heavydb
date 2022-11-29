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
    throw TableFunctionError(
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
  set_output_array_values_total_number sets the total number of array
  values in the index-th output Column of arrays.

  set_output_array_values_total_number must be called before
  set_output_row_size.
*/
extern "C" DEVICE RUNTIME_EXPORT void
TableFunctionManager_set_output_array_values_total_number(
    int8_t* mgr_ptr,
    int32_t index,
    int64_t output_array_values_total_number) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  if (output_array_values_total_number < 0) {
    throw TableFunctionError(
        "set_output_array_values_total_number: expected non-negative array size but "
        "got " +
        std::to_string(output_array_values_total_number));
  }
  mgr->set_output_array_values_total_number(index, output_array_values_total_number);
}

extern "C" DEVICE RUNTIME_EXPORT void set_output_array_values_total_number(
    int32_t index,
    int64_t output_array_values_total_number) {
  auto& mgr = TableFunctionManager::get_singleton();
  TableFunctionManager_set_output_array_values_total_number(
      reinterpret_cast<int8_t*>(mgr), index, output_array_values_total_number);
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
  return TableFunctionErrorCode::GenericError;
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
    throw TableFunctionError("uninitialized TableFunctionManager singleton");
  }
  return reinterpret_cast<int8_t*>(mgr);
}

extern "C" DEVICE RUNTIME_EXPORT void TableFunctionManager_set_metadata(
    int8_t* mgr_ptr,
    const char* key,
    const uint8_t* raw_bytes,
    const size_t num_bytes,
    const TableFunctionMetadataType value_type) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  mgr->set_metadata(key, raw_bytes, num_bytes, value_type);
}

extern "C" DEVICE RUNTIME_EXPORT void TableFunctionManager_get_metadata(
    int8_t* mgr_ptr,
    const char* key,
    const uint8_t*& raw_bytes,
    size_t& num_bytes,
    TableFunctionMetadataType& value_type) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  mgr->get_metadata(key, raw_bytes, num_bytes, value_type);
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
TableFunctionManager_getNewDictDbId(int8_t* mgr_ptr) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getNewDictDbId();
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
TableFunctionManager_getNewDictId(int8_t* mgr_ptr) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getNewDictId();
}

DEVICE RUNTIME_EXPORT std::string TableFunctionManager_getString(int8_t* mgr_ptr,
                                                                 int32_t db_id,
                                                                 int32_t dict_id,
                                                                 int32_t string_id) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getString(db_id, dict_id, string_id);
}

extern "C" DEVICE RUNTIME_EXPORT const char* TableFunctionManager_getCString(
    int8_t* mgr_ptr,
    int32_t db_id,
    int32_t dict_id,
    int32_t string_id) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getString(db_id, dict_id, string_id).c_str();
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
TableFunctionManager_getOrAddTransient(int8_t* mgr_ptr,
                                       int32_t db_id,
                                       int32_t dict_id,
                                       std::string str) {
  auto mgr = reinterpret_cast<TableFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getOrAddTransient(db_id, dict_id, str);
}

#endif  // EXECUTE_INCLUDE
