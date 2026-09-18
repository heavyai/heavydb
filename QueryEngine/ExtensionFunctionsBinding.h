/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
