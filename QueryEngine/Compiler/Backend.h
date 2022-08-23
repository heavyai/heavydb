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

class CodegenTraits {
  explicit CodegenTraits(unsigned local_addr_space, unsigned global_addr_space)
      : local_addr_space_(local_addr_space), global_addr_space_(global_addr_space) {}

  const unsigned local_addr_space_;
  const unsigned global_addr_space_;

 public:
  CodegenTraits(const CodegenTraits&) = delete;
  CodegenTraits& operator=(const CodegenTraits&) = delete;

  static CodegenTraits get(unsigned local_addr_space, unsigned global_addr_space) {
    return CodegenTraits(local_addr_space, global_addr_space);
  }

  llvm::PointerType* localPointerType(llvm::Type* ElementType) const {
    return llvm::PointerType::get(ElementType, local_addr_space_);
  }
  llvm::PointerType* globalPointerType(llvm::Type* ElementType) const {
    return llvm::PointerType::get(ElementType, global_addr_space_);
  }
};

class Backend {
 public:
  virtual ~Backend(){};
  virtual std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) = 0;
  virtual CodegenTraits traits() const = 0;
};

class CPUBackend : public Backend {
 public:
  CPUBackend() = default;
  std::shared_ptr<CompilationContext> generateNativeCode(
      llvm::Function* func,
      llvm::Function* wrapper_func /*ignored*/,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co) override;

  CodegenTraits traits() const { return CodegenTraits::get(0, 0); };

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

  CodegenTraits traits() const { return CodegenTraits::get(0, 0); };

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

  CodegenTraits traits() const { return CodegenTraits::get(4, 1); };

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
