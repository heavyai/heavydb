#include "Codec.h"
#include "Translator.h"

#include <glog/logging.h>
#include <llvm/IR/LLVMContext.h>


FixedWidthInt64::FixedWidthInt64(const size_t byte_width) : byte_width_{byte_width} {}

llvm::Value* FixedWidthInt64::codegenDecode(
    llvm::Value* byte_stream,
    llvm::Value* col_id,
    llvm::Value* pos,
    llvm::IRBuilder<>& ir_builder,
    llvm::Module* module) {
  auto& context = llvm::getGlobalContext();
  auto f = module->getFunction("fixed_width_int64_decode");
  return ir_builder.CreateCall4(f,
    byte_stream,
    col_id,
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), byte_width_),
    pos
  );
}
