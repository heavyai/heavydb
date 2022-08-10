/*
    Copyright 2021 OmniSci, Inc.
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <llvm/IR/Value.h>
#include <memory>

#include "QueryEngine/ExtensionModules.h"
#include "QueryEngine/L0Kernel.h"
#include "QueryEngine/LLVMFunctionAttributesUtil.h"
#include "QueryEngine/NvidiaKernel.h"
#include "QueryEngine/Target.h"

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

  static std::shared_ptr<CpuCompilationContext> generateNativeCPUCode(
      llvm::Function* func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co);
};

class CUDABackend : public Backend {
 public:
  CUDABackend(const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
              bool is_gpu_smem_used,
              GPUTarget& gpu_target);

  std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) override;

  static std::string generatePTX(const std::string& cuda_llir,
                                 llvm::TargetMachine* nvptx_target_machine,
                                 llvm::LLVMContext& context);

  static void linkModuleWithLibdevice(const std::unique_ptr<llvm::Module>& ext,
                                      llvm::Module& module,
                                      llvm::PassManagerBuilder& pass_manager_builder,
                                      const GPUTarget& gpu_target,
                                      llvm::TargetMachine* nvptx_target_machine);

  static std::unique_ptr<llvm::TargetMachine> initializeNVPTXBackend(
      const CudaMgr_Namespace::NvidiaDeviceArch arch);

  static std::shared_ptr<CudaCompilationContext> generateNativeGPUCode(
      const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const bool is_gpu_smem_used,
      const CompilationOptions& co,
      const GPUTarget& gpu_target,
      llvm::TargetMachine* nvptx_target_machine);

 private:
  const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts_;
  bool is_gpu_smem_used_;
  GPUTarget& gpu_target_;

  mutable std::unique_ptr<llvm::TargetMachine> nvptx_target_machine_;
};

class L0Backend : public Backend {
 public:
  L0Backend(GPUTarget& gpu_target) : gpu_target_(gpu_target) {}

  std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) override;

  static std::shared_ptr<L0CompilationContext> generateNativeGPUCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co,
      const GPUTarget& gpu_target);

 private:
  GPUTarget& gpu_target_;
};

std::shared_ptr<Backend> getBackend(
    ExecutorDeviceType dt,
    const std::map<ExtModuleKinds, std::unique_ptr<llvm::Module>>& exts,
    bool is_gpu_smem_used_,
    GPUTarget& gpu_target);

}  // namespace compiler
