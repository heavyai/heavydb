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

#include "QueryEngine/RowFunctionManager.h"

DEVICE RUNTIME_EXPORT std::string RowFunctionManager_getString(int8_t* mgr_ptr,
                                                               int32_t db_id,
                                                               int32_t dict_id,
                                                               int32_t string_id) {
  auto mgr = reinterpret_cast<RowFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getString(db_id, dict_id, string_id);
}

extern "C" DEVICE RUNTIME_EXPORT int8_t* RowFunctionManager_getStringDictionaryProxy(
    int8_t* mgr_ptr,
    int32_t db_id,
    int32_t dict_id) {
  auto mgr = reinterpret_cast<RowFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getStringDictionaryProxy(db_id, dict_id);
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
RowFunctionManager_getDictDbId(int8_t* mgr_ptr, const char* func_name, size_t index) {
  auto mgr = reinterpret_cast<RowFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getDictDbId(std::string(func_name), index);
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
RowFunctionManager_getDictId(int8_t* mgr_ptr, const char* func_name, size_t index) {
  auto mgr = reinterpret_cast<RowFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getDictId(std::string(func_name), index);
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
RowFunctionManager_getOrAddTransient(int8_t* mgr_ptr,
                                     int32_t db_id,
                                     int32_t dict_id,
                                     std::string str) {
  auto mgr = reinterpret_cast<RowFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->getOrAddTransient(db_id, dict_id, str);
}

#endif  // #ifdef EXECUTE_INCLUDE
