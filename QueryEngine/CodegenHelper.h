/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "CodeGenerator.h"
#include "CompilationOptions.h"

#include <optional>
#include <string_view>

#include <llvm/IR/IRBuilder.h>

namespace CodegenUtil {

// todo (yoonmin) : locate more utility functions used during codegen here
llvm::Function* findCalledFunction(llvm::CallInst& call_inst);
std::optional<std::string_view> getCalledFunctionName(llvm::CallInst& call_inst);
std::unordered_map<int, llvm::Value*> createPtrWithHoistedMemoryAddr(
    CgenState* cgen_state,
    CodeGenerator* code_generator,
    CompilationOptions const& co,
    llvm::ConstantInt* ptr,
    llvm::Type* type,
    std::set<int> const& target_device_ids);
std::unordered_map<int, llvm::Value*> hoistLiteral(
    CodeGenerator* code_generator,
    CompilationOptions const& co,
    Datum d,
    SQLTypeInfo type,
    std::set<int> const& target_device_ids);

}  // namespace CodegenUtil
