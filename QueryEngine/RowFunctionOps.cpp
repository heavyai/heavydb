/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

extern "C" DEVICE RUNTIME_EXPORT int8_t* RowFunctionManager_makeBuffer(
    int8_t* mgr_ptr,
    int64_t element_count,
    int64_t element_size) {
  auto mgr = reinterpret_cast<RowFunctionManager*>(mgr_ptr);
  CHECK(mgr);
  return mgr->makeBuffer(element_count, element_size);
}

#endif  // #ifdef EXECUTE_INCLUDE
