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
  void generateCastsForInputTypes(const TableFunctionExecutionUnit& exe_unit,
                                  const std::vector<llvm::Value*>& func_args);
  void generateGpuKernel();
  bool passColumnsByValue(const TableFunctionExecutionUnit& exe_unit);

  std::shared_ptr<CompilationContext> finalize(bool emit_only_preflight_fn);

  llvm::Function* entry_point_func_;
  llvm::Function* kernel_func_;
  Executor* executor_;
  const CompilationOptions& co_;
};
