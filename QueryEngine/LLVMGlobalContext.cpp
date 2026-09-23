/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "LLVMGlobalContext.h"
#include <llvm/Support/ManagedStatic.h>

namespace {

llvm::ManagedStatic<llvm::LLVMContext> g_global_context;

}  // namespace

llvm::LLVMContext& getGlobalLLVMContext() {
  return *g_global_context;
}
