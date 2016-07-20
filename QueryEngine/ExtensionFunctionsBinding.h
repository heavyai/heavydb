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
