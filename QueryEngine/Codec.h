#ifndef QUERYENGINE_CODEC_H
#define QUERYENGINE_CODEC_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

class Decoder {
 public:
  virtual llvm::Instruction* codegenDecode(llvm::Value* byte_stream, llvm::Value* pos, llvm::Module* module) const = 0;
};

class FixedWidthInt : public Decoder {
 public:
  FixedWidthInt(const size_t byte_width);
  llvm::Instruction* codegenDecode(llvm::Value* byte_stream, llvm::Value* pos, llvm::Module* module) const override;

 private:
  const size_t byte_width_;
};

class FixedWidthUnsigned : public Decoder {
 public:
  FixedWidthUnsigned(const size_t byte_width);
  llvm::Instruction* codegenDecode(llvm::Value* byte_stream, llvm::Value* pos, llvm::Module* module) const override;

 private:
  const size_t byte_width_;
};

class DiffFixedWidthInt : public Decoder {
 public:
  DiffFixedWidthInt(const size_t byte_width, const int64_t baseline);
  llvm::Instruction* codegenDecode(llvm::Value* byte_stream, llvm::Value* pos, llvm::Module* module) const override;

 private:
  const size_t byte_width_;
  const int64_t baseline_;
};

class FixedWidthReal : public Decoder {
 public:
  FixedWidthReal(const bool is_double);
  llvm::Instruction* codegenDecode(llvm::Value* byte_stream, llvm::Value* pos, llvm::Module* module) const override;

 private:
  const bool is_double_;
};

#endif  // QUERYENGINE_CODEC_H
