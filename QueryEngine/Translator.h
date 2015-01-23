#ifndef QUERYENGINE_TRANSLATOR_H
#define QUERYENGINE_TRANSLATOR_H

#include "Codec.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

class AstNode {
public:
  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module) = 0;
  virtual void collectUsedColumns(std::unordered_set<int>& columns) = 0;
};

class FetchIntCol : public AstNode {
public:
  FetchIntCol(const int col_id, const int width, const std::shared_ptr<Decoder> decoder);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  const int col_id_;
  const int width_;
  const std::shared_ptr<Decoder> decoder_;
  static std::unordered_map<int, llvm::Value*> fetch_cache_;
};

class PromoteToReal : public AstNode {
public:
  PromoteToReal(std::shared_ptr<AstNode> from, const bool double_precision);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

private:
  std::shared_ptr<AstNode> from_;
  const bool double_precision_;
};

class PromoteToWiderInt : public AstNode {
public:
  PromoteToWiderInt(std::shared_ptr<AstNode> from, const int target_width);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

private:
  std::shared_ptr<AstNode> from_;
  const int target_width_;
};

class ImmInt : public AstNode {
public:
  ImmInt(const int64_t val, const int width);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  const int64_t val_;
  const int width_;
};

class OpBinary : public AstNode {
public:
  OpBinary(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  void collectUsedColumns(std::unordered_set<int>& columns);

protected:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpICmp : public OpBinary {
public:
  OpICmp(std::shared_ptr<AstNode> lhs,
         std::shared_ptr<AstNode> rhs,
         llvm::ICmpInst::Predicate op);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

private:
  llvm::ICmpInst::Predicate op_;
};

class OpIGt : public OpICmp {
public:
  OpIGt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpILt : public OpICmp {
public:
  OpILt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpIGe : public OpICmp {
  OpIGe(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpILe : public OpICmp {
public:
  OpILe(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpINe : public OpICmp {
public:
  OpINe(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpIEq : public OpICmp {
public:
  OpIEq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpFCmp : public OpBinary {
public:
  OpFCmp(std::shared_ptr<AstNode> lhs,
         std::shared_ptr<AstNode> rhs,
         llvm::FCmpInst::Predicate op);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

private:
  llvm::FCmpInst::Predicate op_;
};

class OpFGt : public OpFCmp {
public:
  OpFGt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpFLt : public OpFCmp {
public:
  OpFLt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpFGe : public OpFCmp {
  OpFGe(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpFLe : public OpFCmp {
public:
  OpFLe(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpFNe : public OpFCmp {
public:
  OpFNe(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpFEq : public OpFCmp {
public:
  OpFEq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpIAdd : public OpBinary {
public:
  OpIAdd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpISub : public OpBinary {
public:
  OpISub(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpIMul : public OpBinary {
public:
  OpIMul(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpIDiv : public OpBinary {
public:
  OpIDiv(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpFAdd : public OpBinary {
public:
  OpFAdd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpFSub : public OpBinary {
public:
  OpFSub(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpFMul : public OpBinary {
public:
  OpFMul(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpFDiv : public OpBinary {
public:
  OpFDiv(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpAnd : public OpBinary {
public:
  OpAnd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpOr : public OpBinary {
public:
  OpOr(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);
};

class OpNot : public AstNode {
public:
  OpNot(std::shared_ptr<AstNode> op);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> op_;
};

class AggQueryCodeGenerator {
public:
  AggQueryCodeGenerator(
      std::shared_ptr<AstNode> filter,
      std::shared_ptr<AstNode> aggr_col,
      const std::vector<std::shared_ptr<AstNode>>& group_by_cols,
      const int32_t groups_buffer_entry_count,
      const std::string& agg_name,
      const std::string& query_template_name,
      const std::string& row_process_name,
      const std::string& pos_start_name,
      const std::string& pos_step_name);

  ~AggQueryCodeGenerator();

  void* getNativeCode() {
    return query_native_code_;
  }

private:
  llvm::ExecutionEngine* execution_engine_;
  void* query_native_code_;
};

#endif  // QUERYENGINE_TRANSLATOR_H
