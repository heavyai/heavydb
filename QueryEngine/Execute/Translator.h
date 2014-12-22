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

class FetchInt64Col : public AstNode {
public:
  FetchInt64Col(const int col_id, const std::shared_ptr<Decoder> decoder);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  const int col_id_;
  const std::shared_ptr<Decoder> decoder_;
  static std::unordered_map<int64_t, llvm::Value*> fetch_cache_;
};

class ImmInt64 : public AstNode {
public:
  ImmInt64(const int64_t val);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  const int64_t val_;
};

class OpGt : public AstNode {
public:
  OpGt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpLt : public AstNode {
public:
  OpLt(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpGte : public AstNode {
public:
  OpGte(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpLte : public AstNode {
public:
  OpLte(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpNeq : public AstNode {
public:
  OpNeq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
};

class OpEq : public AstNode {
public:
  OpEq(std::shared_ptr<AstNode> lhs, std::shared_ptr<AstNode> rhs);

  virtual llvm::Value* codegen(
      llvm::Function* func,
      llvm::IRBuilder<>& ir_builder,
      llvm::Module* module);

  void collectUsedColumns(std::unordered_set<int>& columns);

private:
  std::shared_ptr<AstNode> lhs_;
  std::shared_ptr<AstNode> rhs_;
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
      const std::string& query_template_name,
      const std::string& filter_placeholder_name,
      const std::string& agg_placeholder_name,
      const std::string& pos_start_name,
      const std::string& pos_step_name,
      const int agg_col_id,
      const std::shared_ptr<Decoder> agg_col_decoder,
      const std::string& agg_name);

  ~AggQueryCodeGenerator();

  void* getNativeCode() {
    return query_native_code_;
  }

private:
  llvm::ExecutionEngine* execution_engine_;
  void* query_native_code_;
};

#endif  // QUERYENGINE_EXECUTE_TRANSLATOR_H
