#ifndef QUERYENGINE_EXECUTE_TRANSLATOR_H
#define QUERYENGINE_EXECUTE_TRANSLATOR_H

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

class OpICmp : public AstNode {
public:
  OpICmp(std::shared_ptr<AstNode> lhs,
         std::shared_ptr<AstNode> rhs,
         llvm::ICmpInst::Predicate op);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
  llvm::ICmpInst::Predicate op_;
};

class OpGt : public OpICmp {
public:
  OpGt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpLt : public OpICmp {
public:
  OpLt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpGte : public OpICmp {
  OpGte(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpLte : public OpICmp {
public:
  OpLte(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpNeq : public OpICmp {
public:
  OpNeq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpEq : public OpICmp {
public:
  OpEq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);
};

class OpAdd : public AstNode {
public:
  OpAdd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpSub : public AstNode {
public:
  OpSub(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpMul : public AstNode {
public:
  OpMul(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpDiv : public AstNode {
public:
  OpDiv(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpAnd : public AstNode {
public:
  OpAnd(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpOr : public AstNode {
public:
  OpOr(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
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

#endif  // QUERYENGINE_EXECUTE_TRANSLATOR_H
