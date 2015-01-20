#include "Codec.h"
#include "Translator.h"

#include <glog/logging.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Module.h>


FixedWidthInt::FixedWidthInt(const size_t byte_width) : byte_width_{byte_width} {}

llvm::Instruction* FixedWidthInt::codegenDecode(
    llvm::Value* byte_stream,
    llvm::Value* pos,
    llvm::Module* module) const {
  auto& context = llvm::getGlobalContext();
  auto f = module->getFunction("fixed_width_int_decode");
  CHECK(f);
  llvm::Value *args[] = {
    byte_stream,
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), byte_width_),
    pos
  };
  return llvm::CallInst::Create(f, args);
}

DiffFixedWidthInt::DiffFixedWidthInt(const size_t byte_width, const int64_t baseline)
  : byte_width_{byte_width}, baseline_{baseline} {}

llvm::Instruction* DiffFixedWidthInt::codegenDecode(
    llvm::Value* byte_stream,
    llvm::Value* pos,
    llvm::Module* module) const {
  auto& context = llvm::getGlobalContext();
  auto f = module->getFunction("diff_fixed_width_int_decode");
  CHECK(f);
  llvm::Value *args[] = {
    byte_stream,
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), byte_width_),
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), baseline_),
    pos
  };
  return llvm::CallInst::Create(f, args);
}
