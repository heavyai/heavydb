#include "LLVMGlobalContext.h"

#define MAPD_LLVM_VERSION (LLVM_VERSION_MAJOR * 10000 + LLVM_VERSION_MINOR * 100 + LLVM_VERSION_PATCH)

#if MAPD_LLVM_VERSION >= 30900

#include <llvm/Support/ManagedStatic.h>

namespace {

llvm::ManagedStatic<llvm::LLVMContext> g_global_context;

}  // namespace

llvm::LLVMContext& getGlobalLLVMContext() {
  return *g_global_context;
}

#else

llvm::LLVMContext& getGlobalLLVMContext() {
  return llvm::getGlobalContext();
}

#endif
