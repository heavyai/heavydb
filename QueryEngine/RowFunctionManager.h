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

#include "QueryEngine/Execute.h"

struct RowFunctionManager {
  RowFunctionManager(const Executor* executor, const RelAlgExecutionUnit& ra_exe_unit)
      : executor_(executor), ra_exe_unit_(ra_exe_unit) {}

  inline std::string getString(int32_t dict_id, int32_t string_id) {
    const auto proxy = executor_->getStringDictionaryProxy(
        dict_id, executor_->getRowSetMemoryOwner(), true);
    return proxy->getString(string_id);
  }

  inline int32_t getDictId(size_t arg_idx) {
    const Analyzer::FunctionOper* function_oper =
        dynamic_cast<Analyzer::FunctionOper*>(ra_exe_unit_.target_exprs[0]);

    CHECK(function_oper);
    CHECK_LT(arg_idx, function_oper->getArity());
    const SQLTypeInfo typ = function_oper->getArg(arg_idx)->get_type_info();
    CHECK(typ.is_text_encoding_dict());
    return typ.get_comp_param();
  }

  inline int32_t getOrAddTransient(int32_t dict_id, std::string str) {
    const auto proxy = executor_->getStringDictionaryProxy(
        dict_id, executor_->getRowSetMemoryOwner(), true);
    return proxy->getOrAddTransient(str);
  }

  // Executor
  const Executor* executor_;
  const RelAlgExecutionUnit& ra_exe_unit_;
};
