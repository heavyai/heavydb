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
