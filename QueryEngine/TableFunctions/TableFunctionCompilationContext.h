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
  TableFunctionCompilationContext();

  // non-copyable
  TableFunctionCompilationContext(const TableFunctionCompilationContext&) = delete;
  TableFunctionCompilationContext& operator=(const TableFunctionCompilationContext&) =
      delete;

  void compile(const TableFunctionExecutionUnit& exe_unit,
               const CompilationOptions& co,
               Executor* executor);

  using FuncPtr = int32_t (*)(const int8_t** input_cols,
                              const int64_t* input_row_count,
                              int64_t** out,
                              int64_t* output_row_count);
  TableFunctionCompilationContext::FuncPtr getFuncPtr() const { return func_ptr; };

  CodeGenerator::GPUCode* getGpuCode() const { return gpu_code_.get(); }

 private:
  void generateEntryPoint(const size_t in_col_count);
  void generateGpuKernel();
  void finalize(const CompilationOptions& co, Executor* executor);

  std::unique_ptr<CgenState> cgen_state_;
  std::unique_ptr<llvm::Module> module_;
  ExecutionEngineWrapper own_execution_engine_;  // TODO: remove and replace with cache
  std::unique_ptr<CodeGenerator::GPUCode> gpu_code_;
  llvm::Function* entry_point_func_;
  llvm::Function* kernel_func_;
  FuncPtr func_ptr;
};
