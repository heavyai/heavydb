/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "LLVMFunctionAttributesUtil.h"

void mark_function_always_inline(llvm::Function* func) {
  func->addFnAttr(llvm::Attribute::AlwaysInline);
}

void mark_function_never_inline(llvm::Function* func) {
  clear_function_attributes(func);
  func->addFnAttr(llvm::Attribute::NoInline);
}

void clear_function_attributes(llvm::Function* func) {
  llvm::AttributeList no_attributes;
  func->setAttributes(no_attributes);
}
