/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/ExecutionEngine/ExecutionEngine.h>

#include <QueryEngine/CgenState.h>
#include <QueryEngine/CodeCache.h>
#include <QueryEngine/CodeGenerator.h>
#include <QueryEngine/CompilationOptions.h>
#include <QueryEngine/RelAlgExecutionUnit.h>

class Executor;

class TableFunctionCompilationContext {
 public:
  TableFunctionCompilationContext(Executor* executor, const CompilationOptions& co)
      : executor_(executor), co_(co) {}

  // non-copyable
  TableFunctionCompilationContext(const TableFunctionCompilationContext&) = delete;
  TableFunctionCompilationContext& operator=(const TableFunctionCompilationContext&) =
      delete;

  std::shared_ptr<CompilationContext> compile(const TableFunctionExecutionUnit& exe_unit,
                                              bool emit_only_preflight_fn);

 private:
  void generateEntryPoint(const TableFunctionExecutionUnit& exe_unit,
                          bool emit_only_preflight_fn);
  void generateTableFunctionCall(const TableFunctionExecutionUnit& exe_unit,
                                 const std::vector<llvm::Value*>& func_args,
                                 llvm::BasicBlock* bb_exit,
                                 llvm::Value* output_row_count_ptr,
                                 bool emit_only_preflight_fn);
  void generateCastsForInputTypes(
      const TableFunctionExecutionUnit& exe_unit,
      const std::vector<std::pair<llvm::Value*, const SQLTypeInfo>>& columns_to_cast,
      llvm::Value* mgr_ptr);
  void generateGpuKernel();
  bool passColumnsByValue(const TableFunctionExecutionUnit& exe_unit);

  std::shared_ptr<CompilationContext> finalize(
      bool emit_only_preflight_fn,
      std::chrono::steady_clock::time_point& compile_start_timer);

  llvm::Function* entry_point_func_;
  llvm::Function* kernel_func_;
  Executor* executor_;
  const CompilationOptions& co_;
};
