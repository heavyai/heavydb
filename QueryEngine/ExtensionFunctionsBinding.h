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

/**
 * @file    ExtensionFunctionsBinding.h
 * @brief   Argument type based extension function binding.
 *
 */

#ifndef QUERYENGINE_EXTENSIONFUNCTIONSBINDING_H
#define QUERYENGINE_EXTENSIONFUNCTIONSBINDING_H

#include "ExtensionFunctionsWhitelist.h"
#include "TableFunctions/TableFunctionsFactory.h"

#include "../Analyzer/Analyzer.h"
#include "../Shared/sqltypes.h"

#include <tuple>
#include <vector>

namespace Analyzer {
class FunctionOper;
}  // namespace Analyzer

class ExtensionFunctionBindingError : public std::runtime_error {
 public:
  ExtensionFunctionBindingError(const std::string message)
      : std::runtime_error(message) {}
};

ExtensionFunction bind_function(std::string name,
                                Analyzer::ExpressionPtrVector func_args,
                                const bool is_gpu);

ExtensionFunction bind_function(std::string name,
                                Analyzer::ExpressionPtrVector func_args);

ExtensionFunction bind_function(const Analyzer::FunctionOper* function_oper,
                                const bool is_gpu);

const std::tuple<table_functions::TableFunction, std::vector<SQLTypeInfo>>
bind_table_function(std::string name,
                    Analyzer::ExpressionPtrVector input_args,
                    const bool is_gpu);

#endif  // QUERYENGINE_EXTENSIONFUNCTIONSBINDING_H
