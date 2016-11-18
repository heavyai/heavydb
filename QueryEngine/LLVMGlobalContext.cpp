#include "LLVMGlobalContext.h"

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 9

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
