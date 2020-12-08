/*
 * Copyright 2017 MapD Technologies, Inc.
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

/*
 * @file    ExtensionFunctionsBinding.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Argument type based extension function binding.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_EXTENSIONFUNCTIONSBINDING_H
#define QUERYENGINE_EXTENSIONFUNCTIONSBINDING_H

#include "ExtensionFunctionsWhitelist.h"
#include "TableFunctions/TableFunctionsFactory.h"

#include "../Analyzer/Analyzer.h"
#include "../Shared/sqltypes.h"

#include <vector>

namespace Analyzer {
class FunctionOper;
}  // namespace Analyzer

ExtensionFunction bind_function(std::string name,
                                Analyzer::ExpressionPtrVector func_args,
                                const bool is_gpu);

ExtensionFunction bind_function(std::string name,
                                Analyzer::ExpressionPtrVector func_args);

ExtensionFunction bind_function(const Analyzer::FunctionOper* function_oper,
                                const bool is_gpu);

const table_functions::TableFunction bind_table_function(
    std::string name,
    Analyzer::ExpressionPtrVector input_args,
    const bool is_gpu);

bool is_ext_arg_type_column(const ExtArgumentType ext_arg_type);
bool is_ext_arg_type_array(const ExtArgumentType ext_arg_type);
bool is_ext_arg_type_geo(const ExtArgumentType ext_arg_type);
bool is_ext_arg_type_pointer(const ExtArgumentType ext_arg_type);
bool is_ext_arg_type_scalar(const ExtArgumentType ext_arg_type);
#endif  // QUERYENGINE_EXTENSIONFUNCTIONSBINDING_H
