/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/IR/Module.h>

#include <memory>

class CompilationContext {
 public:
  virtual ~CompilationContext() {}
  virtual size_t getMemSize() const = 0;
  size_t size() const { return getMemSize(); }
};

struct CompilationOptions;

class ExecutionEngineWrapper {
 public:
  ExecutionEngineWrapper();
  ExecutionEngineWrapper(llvm::ExecutionEngine* execution_engine);
  ExecutionEngineWrapper(llvm::ExecutionEngine* execution_engine,
                         const CompilationOptions& co);

  ExecutionEngineWrapper(const ExecutionEngineWrapper& other) = delete;
  ExecutionEngineWrapper(ExecutionEngineWrapper&& other) = default;

  ExecutionEngineWrapper& operator=(const ExecutionEngineWrapper& other) = delete;
  ExecutionEngineWrapper& operator=(ExecutionEngineWrapper&& other) = default;

  ExecutionEngineWrapper& operator=(llvm::ExecutionEngine* execution_engine);

  llvm::ExecutionEngine* get() { return execution_engine_.get(); }
  const llvm::ExecutionEngine* get() const { return execution_engine_.get(); }

  llvm::ExecutionEngine& operator*() { return *execution_engine_; }
  const llvm::ExecutionEngine& operator*() const { return *execution_engine_; }

  llvm::ExecutionEngine* operator->() { return execution_engine_.get(); }
  const llvm::ExecutionEngine* operator->() const { return execution_engine_.get(); }

 private:
  std::unique_ptr<llvm::ExecutionEngine> execution_engine_;
  std::unique_ptr<llvm::JITEventListener> intel_jit_listener_;
};

class CpuCompilationContext : public CompilationContext {
 public:
  CpuCompilationContext(ExecutionEngineWrapper&& execution_engine)
      : execution_engine_(std::move(execution_engine)) {}

  template <typename... Ts>
  void call(Ts... args) const {
    reinterpret_cast<void (*)(Ts...)>(func_)(args...);
  }

  const std::string& name() const { return name_; }

  void setFunctionPointer(llvm::Function* function) {
    func_ = execution_engine_->getPointerToFunction(function);
    CHECK(func_);
    name_ = function->getName().str();
    execution_engine_->removeModule(function->getParent());
  }

  void* func() const { return func_; }

  using TableFunctionEntryPointPtr = int32_t (*)(const int8_t* mgr_ptr,
                                                 const int8_t** input_cols,
                                                 const int64_t* input_row_count,
                                                 const int8_t** input_str_dict_proxy_ptrs,
                                                 int64_t** out,
                                                 int8_t** output_str_dict_proxy_ptrs,
                                                 int64_t* output_row_count);
  TableFunctionEntryPointPtr table_function_entry_point() const {
    return (TableFunctionEntryPointPtr)func_;
  }

  size_t getMemSize() const { return 0; }

 private:
  void* func_{nullptr};
  std::string name_;
  ExecutionEngineWrapper execution_engine_;
};
