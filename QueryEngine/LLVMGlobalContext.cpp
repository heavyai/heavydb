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

#include "LLVMGlobalContext.h"

#define MAPD_LLVM_VERSION (LLVM_VERSION_MAJOR * 10000 + LLVM_VERSION_MINOR * 100 + LLVM_VERSION_PATCH)

#if MAPD_LLVM_VERSION >= 30900

#include <llvm/Support/ManagedStatic.h>

namespace {

llvm::ManagedStatic<llvm::LLVMContext> g_global_context;

}  // namespace

llvm::LLVMContext& getGlobalLLVMContext() {
  return *g_global_context;
}

#else

llvm::LLVMContext& getGlobalLLVMContext() {
  return llvm::getGlobalContext();
}

#endif
