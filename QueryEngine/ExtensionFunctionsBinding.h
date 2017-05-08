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

#include "../Shared/sqltypes.h"

#include <vector>

namespace Analyzer {
class FunctionOper;
}  // Analyzer

SQLTypeInfo ext_arg_type_to_type_info(const ExtArgumentType ext_arg_type);

const ExtensionFunction& bind_function(const Analyzer::FunctionOper* function_oper,
                                       const std::vector<ExtensionFunction>& ext_func_sigs);

#endif  // QUERYENGINE_EXTENSIONFUNCTIONSBINDING_H
