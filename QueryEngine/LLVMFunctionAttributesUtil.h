/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>

void mark_function_always_inline(llvm::Function* func);
void mark_function_never_inline(llvm::Function* func);
void clear_function_attributes(llvm::Function* func);
