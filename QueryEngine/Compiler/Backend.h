#pragma once

#include <llvm/IR/Value.h>
#include <memory>

// todo: remove
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Execute.h"

namespace compiler {

class Backend {
 public:
  virtual ~Backend(){};
  virtual std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) = 0;
};

class CPUBackend : public Backend {
 public:
  CPUBackend() = default;
  std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func /*ignored*/,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) override;
};

class CUDABackend : public Backend {
 public:
  CUDABackend(Executor* executor,
              bool is_gpu_smem_used,
              CodeGenerator::GPUTarget& gpu_target)
      : executor_(executor)
      , is_gpu_smem_used_(is_gpu_smem_used)
      , gpu_target_(gpu_target) {}

  std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) override;

 private:
  Executor* executor_;
  bool is_gpu_smem_used_;
  CodeGenerator::GPUTarget& gpu_target_;
};

std::shared_ptr<Backend> getBackend(ExecutorDeviceType dt,
                                    Executor* executor,
                                    bool is_gpu_smem_used_,
                                    CodeGenerator::GPUTarget& gpu_target);

}  // namespace compiler
