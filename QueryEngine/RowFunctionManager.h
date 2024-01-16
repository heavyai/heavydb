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

#include <boost/algorithm/string/predicate.hpp>

#include "QueryEngine/Execute.h"
#include "Shared/toString.h"

// copied from TableFunctionsFactory.cpp
namespace {

std::string drop_suffix_impl(const std::string& str) {
  const auto idx = str.find("__");
  if (idx == std::string::npos) {
    return str;
  }
  CHECK_GT(idx, std::string::size_type(0));
  return str.substr(0, idx);
}

std::list<const Analyzer::Expr*> find_function_oper(
    const Analyzer::Expr* expr,
    const std::string& func_name_wo_suffix) {
  auto is_func_oper = [&func_name_wo_suffix](const Analyzer::Expr* e) -> bool {
    auto function_oper = dynamic_cast<const Analyzer::FunctionOper*>(e);

    if (function_oper) {
      std::string func_oper_name = drop_suffix_impl(function_oper->getName());
      boost::algorithm::to_lower(func_oper_name);
      if (func_name_wo_suffix == func_oper_name) {
        return true;
      }
    }
    return false;
  };
  std::list<const Analyzer::Expr*> funcoper_list;
  expr->find_expr(is_func_oper, funcoper_list);
  return funcoper_list;
}

}  // namespace

struct RowFunctionManager {
  RowFunctionManager(const Executor* executor, const RelAlgExecutionUnit& ra_exe_unit)
      : executor_(executor) {
    target_exprs_.reserve(ra_exe_unit.target_exprs.size() + ra_exe_unit.quals.size());
    std::copy(ra_exe_unit.target_exprs.cbegin(),
              ra_exe_unit.target_exprs.cend(),
              std::back_inserter(target_exprs_));
    std::transform(ra_exe_unit.quals.cbegin(),
                   ra_exe_unit.quals.cend(),
                   std::back_inserter(target_exprs_),
                   [](auto& ptr) { return ptr.get(); });
  }

  inline std::string getString(int32_t db_id, int32_t dict_id, int32_t string_id) {
    const auto proxy = executor_->getStringDictionaryProxy(
        {db_id, dict_id}, executor_->getRowSetMemoryOwner(), true);
    return proxy->getString(string_id);
  }

  inline int32_t getDictDbId(const std::string& func_name, size_t arg_idx) {
    std::string func_name_wo_suffix =
        boost::algorithm::to_lower_copy(drop_suffix_impl(func_name));

    for (const auto& expr : target_exprs_) {
      for (const auto* op : find_function_oper(expr, func_name_wo_suffix)) {
        const Analyzer::FunctionOper* function_oper =
            dynamic_cast<const Analyzer::FunctionOper*>(op);
        CHECK_LT(arg_idx, function_oper->getArity());
        const SQLTypeInfo typ = function_oper->getArg(arg_idx)->get_type_info();
        CHECK(typ.is_text_encoding_dict() || typ.is_text_encoding_dict_array());
        return typ.getStringDictKey().db_id;
      }
    }
    UNREACHABLE();
    return 0;
  }

  inline int32_t getDictId(const std::string& func_name, size_t arg_idx) {
    std::string func_name_wo_suffix =
        boost::algorithm::to_lower_copy(drop_suffix_impl(func_name));

    for (const auto& expr : target_exprs_) {
      for (const auto* op : find_function_oper(expr, func_name_wo_suffix)) {
        const Analyzer::FunctionOper* function_oper =
            dynamic_cast<const Analyzer::FunctionOper*>(op);
        CHECK_LT(arg_idx, function_oper->getArity());
        const SQLTypeInfo typ = function_oper->getArg(arg_idx)->get_type_info();
        CHECK(typ.is_text_encoding_dict() || typ.is_text_encoding_dict_array());
        return typ.getStringDictKey().dict_id;
      }
    }
    UNREACHABLE();
    return 0;
  }

  inline int32_t getOrAddTransient(int32_t db_id, int32_t dict_id, std::string str) {
    const auto proxy = executor_->getStringDictionaryProxy(
        {db_id, dict_id}, executor_->getRowSetMemoryOwner(), true);
    return proxy->getOrAddTransient(str);
  }

  inline int8_t* getStringDictionaryProxy(int32_t db_id, int32_t dict_id) {
    auto* proxy = executor_->getStringDictionaryProxy(
        {db_id, dict_id}, executor_->getRowSetMemoryOwner(), true);
    return reinterpret_cast<int8_t*>(proxy);
  }

  inline int8_t* makeBuffer(int64_t element_count, int64_t element_size) {
    int8_t* buffer =
        reinterpret_cast<int8_t*>(checked_malloc((element_count + 1) * element_size));
    executor_->getRowSetMemoryOwner()->addVarlenBuffer(buffer);
    return buffer;
  }

  // Executor
  const Executor* executor_;
  std::vector<const Analyzer::Expr*> target_exprs_;
};
