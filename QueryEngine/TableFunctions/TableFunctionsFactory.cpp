/*
 * Copyright 2019 OmniSci, Inc.
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

#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"

#include <boost/algorithm/string.hpp>
#include <mutex>

extern bool g_enable_table_functions;

namespace table_functions {

namespace {

SQLTypeInfo ext_arg_pointer_type_to_type_info(const ExtArgumentType ext_arg_type) {
  auto generate_column_type = [](const auto subtype) {
    auto ti = SQLTypeInfo(kCOLUMN, false);
    ti.set_subtype(subtype);
    return ti;
  };
  switch (ext_arg_type) {
    case ExtArgumentType::PInt8:
      return SQLTypeInfo(kTINYINT, false);
    case ExtArgumentType::PInt16:
      return SQLTypeInfo(kSMALLINT, false);
    case ExtArgumentType::PInt32:
      return SQLTypeInfo(kINT, false);
    case ExtArgumentType::PInt64:
      return SQLTypeInfo(kBIGINT, false);
    case ExtArgumentType::PFloat:
      return SQLTypeInfo(kFLOAT, false);
    case ExtArgumentType::PDouble:
      return SQLTypeInfo(kDOUBLE, false);
    case ExtArgumentType::PBool:
      return SQLTypeInfo(kBOOLEAN, false);
    case ExtArgumentType::ColumnInt8:
      return generate_column_type(kTINYINT);
    case ExtArgumentType::ColumnInt16:
      return generate_column_type(kSMALLINT);
    case ExtArgumentType::ColumnInt32:
      return generate_column_type(kINT);
    case ExtArgumentType::ColumnInt64:
      return generate_column_type(kBIGINT);
    case ExtArgumentType::ColumnFloat:
      return generate_column_type(kFLOAT);
    case ExtArgumentType::ColumnDouble:
      return generate_column_type(kDOUBLE);
    case ExtArgumentType::ColumnBool:
      return generate_column_type(kBOOLEAN);
    default:
      LOG(WARNING) << "ext_arg_pointer_type_to_type_info: ExtArgumentType `"
                   << ExtensionFunctionsWhitelist::toString(ext_arg_type)
                   << "` conversion to SQLTypeInfo not implemented.";
      UNREACHABLE();
  }
  UNREACHABLE();
  return SQLTypeInfo(kNULLT, false);
}

}  // namespace

SQLTypeInfo TableFunction::getInputSQLType(const size_t idx) const {
  CHECK_LT(idx, input_args_.size());
  return ext_arg_pointer_type_to_type_info(input_args_[idx]);
}

SQLTypeInfo TableFunction::getOutputSQLType(const size_t idx) const {
  CHECK_LT(idx, output_args_.size());
  // TODO(adb): conditionally handle nulls
  return ext_arg_pointer_type_to_type_info(output_args_[idx]);
}

void TableFunctionsFactory::add(const std::string& name,
                                const TableFunctionOutputRowSizer sizer,
                                const std::vector<ExtArgumentType>& input_args,
                                const std::vector<ExtArgumentType>& output_args,
                                bool is_runtime) {
  for (auto it = functions_.begin(); it != functions_.end();) {
    if (it->second.getName() == name) {
      if (it->second.isRuntime()) {
        VLOG(1) << "Overriding existing run-time table function (reset not called?): "
                << name;
        it = functions_.erase(it);
      } else {
        throw std::runtime_error("Will not override existing load-time table function: " +
                                 name);
      }
    } else {
      ++it;
    }
  }
  auto tf = TableFunction(name, sizer, input_args, output_args, is_runtime);
  functions_.emplace(name, tf);
}

std::once_flag init_flag;

void TableFunctionsFactory::init() {
  if (!g_enable_table_functions) {
    return;
  }
  std::call_once(init_flag, []() {
    TableFunctionsFactory::add(
        "row_copier",
        TableFunctionOutputRowSizer{OutputBufferSizeType::kUserSpecifiedRowMultiplier, 2},
        std::vector<ExtArgumentType>{ExtArgumentType::ColumnDouble,
                                     ExtArgumentType::Int32},
        std::vector<ExtArgumentType>{ExtArgumentType::ColumnDouble});
  });
}

// removes existing runtime table functions
void TableFunctionsFactory::reset() {
  if (!g_enable_table_functions) {
    return;
  }
  for (auto it = functions_.begin(); it != functions_.end();) {
    if (it->second.isRuntime()) {
      it = functions_.erase(it);
    } else {
      ++it;
    }
  }
}

namespace {

std::string drop_suffix(const std::string& str) {
  const auto idx = str.find("__");
  if (idx == std::string::npos) {
    return str;
  }
  CHECK_GT(idx, std::string::size_type(0));
  return str.substr(0, idx);
}

}  // namespace

std::vector<TableFunction> TableFunctionsFactory::get_table_funcs(
    const std::string& name) {
  std::vector<TableFunction> table_funcs;
  auto table_func_name = name;
  boost::algorithm::to_lower(table_func_name);
  for (const auto& pair : functions_) {
    auto fname = drop_suffix(pair.first);
    if (fname == table_func_name) {
      table_funcs.push_back(pair.second);
    }
  }
  return table_funcs;
}

std::unordered_map<std::string, TableFunction> TableFunctionsFactory::functions_;

}  // namespace table_functions
