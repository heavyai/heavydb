#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include "../Analyzer/Analyzer.h"
#include "../Planner/Planner.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <glog/logging.h>

#include <unordered_map>


enum class ExecutorOptLevel {
  Default,
  LoopStrengthReduction
};

class Executor {
public:
  Executor(const Planner::RootPlan* root_plan);
  ~Executor();
  int64_t execute(const ExecutorOptLevel = ExecutorOptLevel::Default);
private:
  llvm::Value* codegen(const Analyzer::Expr*) const;
  llvm::Value* codegen(const Analyzer::BinOper*) const;
  llvm::Value* codegen(const Analyzer::UOper*) const;
  llvm::Value* codegen(const Analyzer::ColumnVar*) const;
  llvm::Value* codegen(const Analyzer::Constant*) const;
  llvm::Value* codegenCmp(const Analyzer::BinOper*) const;
  llvm::Value* codegenLogical(const Analyzer::BinOper*) const;
  llvm::Value* codegenArith(const Analyzer::BinOper*) const;
  llvm::Value* codegenLogical(const Analyzer::UOper*) const;
  int64_t executeAggScanPlan(const Planner::AggPlan* agg_plan, const ExecutorOptLevel);
  void executeScanPlan(const Planner::Scan* scan_plan);
  void compileAggScanPlan(const Planner::AggPlan* agg_plan, const ExecutorOptLevel);
  void* optimizeAndCodegen(llvm::Function*, const ExecutorOptLevel, llvm::Module*);
  void call_aggregator(
    const std::string& agg_name,
    const Analyzer::Expr* aggr_col,
    llvm::Value* filter_result,
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    llvm::Module* module);
  void allocateLocalColumnIds(const Planner::Scan* scan_plan);
  int getLocalColumnId(const int global_col_id) const;

  const Planner::RootPlan* root_plan_;
  llvm::LLVMContext& context_;
  llvm::Module* module_;
  mutable llvm::IRBuilder<> ir_builder_;
  llvm::ExecutionEngine* execution_engine_;
  void* query_native_code_;
  mutable std::unordered_map<int, llvm::Value*> fetch_cache_;
  llvm::Function* row_func_;
  int64_t init_agg_val_;
  std::unordered_map<int, int> global_to_local_col_ids_;
  std::vector<int> local_to_global_col_ids_;
};

#endif // QUERYENGINE_EXECUTE_H
