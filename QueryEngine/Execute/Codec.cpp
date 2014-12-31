#include "Codec.h"
#include "Translator.h"

#include <glog/logging.h>
#include <llvm/IR/LLVMContext.h>


FixedWidthInt::FixedWidthInt(const size_t byte_width) : byte_width_{byte_width} {}

llvm::Instruction* FixedWidthInt::codegenDecode(
    llvm::Value* byte_stream,
    llvm::Value* pos,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto& context = llvm::getGlobalContext();
  auto f = module->getFunction("fixed_width_int_decode");
  llvm::Value *args[] = {
    byte_stream,
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), byte_width_),
    pos
  };
  return llvm::CallInst::Create(f, args);
}
