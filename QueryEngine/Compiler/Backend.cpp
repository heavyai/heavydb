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