/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ResultSetReductionJIT.h"
#include "ResultSetReductionOps.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

// Convert an IR type to the corresponding LLVM one.
llvm::Type* llvm_type(const Type type, llvm::LLVMContext& ctx);

// Translate a function to a LLVM function provided as llvm_function (initially empty).
// The mapping to LLVM for the reduction functions is also provided as input f.
void translate_function(const Function* function,
                        llvm::Function* llvm_function,
                        const ReductionCode& reduction_code,
                        const std::unordered_map<const Function*, llvm::Function*>& f);
