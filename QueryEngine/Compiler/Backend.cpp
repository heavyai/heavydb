#include "Backend.h"

#include "QueryEngine/CodeGenerator.h"

namespace compiler {
std::shared_ptr<CompilationContext> CPUBackend::generateNativeCode(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  return std::dynamic_pointer_cast<CpuCompilationContext>(
      CodeGenerator::generateNativeCPUCode(func, live_funcs, co));
}

CUDABackend::CUDABackend(Executor* executor,
                         bool is_gpu_smem_used,
                         CodeGenerator::GPUTarget& gpu_target)
    : executor_(executor), is_gpu_smem_used_(is_gpu_smem_used), gpu_target_(gpu_target) {
  const auto arch = gpu_target_.cuda_mgr->getDeviceArch();
  nvptx_target_machine_ = CodeGenerator::initializeNVPTXBackend(arch);
  gpu_target_.nvptx_target_machine = nvptx_target_machine_.get();
}

std::shared_ptr<CompilationContext> CUDABackend::generateNativeCode(
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const std::unordered_set<llvm::Function*>& live_funcs,
    const CompilationOptions& co) {
  return std::dynamic_pointer_cast<GpuCompilationContext>(
      CodeGenerator::generateNativeGPUCode(
          executor_, func, wrapper_func, live_funcs, is_gpu_smem_used_, co, gpu_target_));
}

std::shared_ptr<Backend> getBackend(ExecutorDeviceType dt,
                                    Executor* executor,
                                    bool is_gpu_smem_used_,
                                    CodeGenerator::GPUTarget& gpu_target) {
  switch (dt) {
    case ExecutorDeviceType::CPU:
      return std::make_shared<CPUBackend>();
    case ExecutorDeviceType::GPU:
      return std::make_shared<CUDABackend>(executor, is_gpu_smem_used_, gpu_target);
    default:
      CHECK(false);
      return {};
  };
}
}  // namespace compiler