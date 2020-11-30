/*
 * Copyright 2020 OmniSci, Inc.
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
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/IR/Module.h>

struct CompilationOptions;

#ifdef ENABLE_ORCJIT

#include <llvm/ExecutionEngine/Orc/LLJIT.h>

class ORCJITExecutionEngineWrapper {
 public:
  ORCJITExecutionEngineWrapper();
  ORCJITExecutionEngineWrapper(std::unique_ptr<llvm::orc::LLJIT> lljit)
      : lljit_(std::move(lljit)) {}

  ORCJITExecutionEngineWrapper(const ORCJITExecutionEngineWrapper& other) = delete;
  ORCJITExecutionEngineWrapper(ORCJITExecutionEngineWrapper&& other) = default;

  void* getPointerToFunction(llvm::Function* function) {
    CHECK(function);
    CHECK(lljit_);
    auto symbol = lljit_->lookup(function->getName());
    if (!symbol) {
      auto llvm_err_to_str = [](const llvm::Error& err) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << err;
        return msg;
      };
      LOG(FATAL) << "Failed to find function " << std::string(function->getName())
                 << "\nError: " << llvm_err_to_str(symbol.takeError());
    }
    return reinterpret_cast<void*>(symbol->getAddress());
  }

  bool exists() const { return !(lljit_ == nullptr); }

  ORCJITExecutionEngineWrapper& operator=(const ORCJITExecutionEngineWrapper& other) =
      delete;
  ORCJITExecutionEngineWrapper& operator=(ORCJITExecutionEngineWrapper&& other) = default;

 private:
  std::unique_ptr<llvm::orc::LLJIT> lljit_;
  std::unique_ptr<llvm::JITEventListener> intel_jit_listener_;
};

using ExecutionEngineWrapper = ORCJITExecutionEngineWrapper;
#else
class MCJITExecutionEngineWrapper {
 public:
  MCJITExecutionEngineWrapper();
  MCJITExecutionEngineWrapper(llvm::ExecutionEngine* execution_engine,
                              const CompilationOptions& co);

  MCJITExecutionEngineWrapper(const MCJITExecutionEngineWrapper& other) = delete;
  MCJITExecutionEngineWrapper(MCJITExecutionEngineWrapper&& other) = default;

  void* getPointerToFunction(llvm::Function* function) {
    CHECK(execution_engine_);
    return execution_engine_->getPointerToFunction(function);
  }

  void finalize() {
    CHECK(execution_engine_);
    execution_engine_->finalizeObject();
  }

  bool exists() const { return !(execution_engine_ == nullptr); }

  llvm::ExecutionEngine* operator->() { return execution_engine_.get(); }
  const llvm::ExecutionEngine* operator->() const { return execution_engine_.get(); }

  MCJITExecutionEngineWrapper& operator=(const MCJITExecutionEngineWrapper& other) =
      delete;
  MCJITExecutionEngineWrapper& operator=(MCJITExecutionEngineWrapper&& other) = default;

  MCJITExecutionEngineWrapper& operator=(llvm::ExecutionEngine* execution_engine);

 private:
  std::unique_ptr<llvm::ExecutionEngine> execution_engine_;
  std::unique_ptr<llvm::JITEventListener> intel_jit_listener_;
};

using ExecutionEngineWrapper = MCJITExecutionEngineWrapper;
#endif
