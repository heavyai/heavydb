#include "LLVMGlobalContext.h"
#include <llvm/Support/ManagedStatic.h>

namespace {

llvm::ManagedStatic<llvm::LLVMContext> g_global_context;

}  // namespace

llvm::LLVMContext& getGlobalLLVMContext() {
  return *g_global_context;
}
