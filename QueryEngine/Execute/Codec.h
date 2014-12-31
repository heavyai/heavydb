#ifndef QUERYENGINE_EXECUTE_CODEC_H
#define QUERYENGINE_EXECUTE_CODEC_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>


class Decoder {
public:
  virtual llvm::Instruction* codegenDecode(
      llvm::Value* byte_stream,
      llvm::Value* pos,
      llvm::IRBuilder<>&,
      llvm::Module* module) = 0;
};

class FixedWidthInt : public Decoder {
public:
  FixedWidthInt(const size_t byte_width);
  llvm::Instruction* codegenDecode(
      llvm::Value* byte_stream,
      llvm::Value* pos,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
private:
  const size_t byte_width_;
};

class DiffFixedWidthInt : public Decoder {
public:
  DiffFixedWidthInt(const size_t byte_width, const int64_t baseline);
  llvm::Instruction* codegenDecode(
      llvm::Value* byte_stream,
      llvm::Value* pos,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
private:
  const size_t byte_width_;
  const int64_t baseline_;
};

#endif  // QUERYENGINE_EXECUTE_CODEC_H
