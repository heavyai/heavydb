#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include "../Analyzer/Analyzer.h"
#include "../Planner/Planner.h"
#include "NvidiaKernel.h"

#include <boost/variant.hpp>
#include <boost/thread.hpp>
#include <glog/logging.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <cuda.h>

#include <atomic>
#include <chrono>
#include <unordered_map>


enum class ExecutorOptLevel {
  Default,
  LoopStrengthReduction
};

enum class ExecutorDeviceType {
  CPU,
  GPU
};

typedef boost::variant<int64_t, double> AggResult;

class ResultRow {
public:
  AggResult agg_result(const size_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, agg_types_.size());
    CHECK_EQ(agg_results_idx_.size(), agg_types_.size());
    if (agg_types_[idx] == kAVG) {
      CHECK_LT(idx, agg_results_.size() - 1);
      auto actual_idx = agg_results_idx_[idx];
      return AggResult(
        static_cast<double>(agg_results_[actual_idx]) /
        static_cast<double>(agg_results_[actual_idx + 1]));
    } else {
      CHECK_LT(idx, agg_results_.size());
      auto actual_idx = agg_results_idx_[idx];
      return AggResult(agg_results_[actual_idx]);
    }
    return agg_results_[idx];
  }
  size_t size() const {
    return agg_results_idx_.size();
  }
  std::vector<int64_t> value_tuple() const {
    return value_tuple_;
  }
private:
  // TODO(alex): support for strings
  std::vector<int64_t> value_tuple_;
  std::vector<int64_t> agg_results_;
  std::vector<size_t> agg_results_idx_;
  std::vector<SQLAgg> agg_types_;

  friend class Executor;
};

class Executor {
public:
  Executor(const Planner::RootPlan* root_plan);
  ~Executor();

  typedef std::tuple<std::string, const Analyzer::Expr*, int64_t, void*> AggInfo;
  typedef std::vector<ResultRow> ResultRows;

  std::vector<ResultRow> execute(
    const ExecutorDeviceType device_type = ExecutorDeviceType::CPU,
    const ExecutorOptLevel = ExecutorOptLevel::Default);
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
  llvm::Value* codegenCast(const Analyzer::UOper*) const;
  llvm::Value* codegenUMinus(const Analyzer::UOper*) const;
  llvm::Value* codegenIsNull(const Analyzer::UOper*) const;
  std::vector<ResultRow> executeSelectPlan(
    const Planner::Plan* plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel);
  std::vector<ResultRow> executeAggScanPlan(
    const Planner::Plan* plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&);
  std::vector<ResultRow> executeResultPlan(
    const Planner::Result* result_plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&);
  void executePlanWithGroupBy(
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const size_t group_by_col_count,
    const ExecutorDeviceType device_type,
    const Catalog_Namespace::Catalog&,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows);
  void executePlanWithoutGroupBy(
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows);
  static ResultRows reduceMultiDeviceResults(const std::vector<ResultRows>&);
  ResultRows groupBufferToResults(
    const int64_t* group_by_buffer,
    const size_t group_by_col_count,
    const size_t agg_col_count,
    const std::list<Analyzer::Expr*>& target_exprs);
  void executeSimpleInsert();
  void compilePlan(
    const std::vector<Executor::AggInfo>& agg_infos,
    const std::list<Analyzer::Expr*>& groupby_list,
    const std::list<int>& scan_cols,
    const std::list<Analyzer::Expr*>& simple_quals,
    const std::list<Analyzer::Expr*>& quals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const size_t groups_buffer_entry_count);
  void nukeOldState();
  void* optimizeAndCodegenCPU(llvm::Function*, const ExecutorOptLevel, llvm::Module*);
  CUfunction optimizeAndCodegenGPU(llvm::Function*, const ExecutorOptLevel, llvm::Module*);
  void call_aggregators(
    const std::vector<AggInfo>& agg_infos,
    llvm::Value* filter_result,
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    llvm::Module* module);
  void allocateLocalColumnIds(const std::list<int>& global_col_ids);
  int getLocalColumnId(const int global_col_id) const;

  const Planner::RootPlan* root_plan_;
  llvm::LLVMContext& context_;
  llvm::Module* module_;
  mutable llvm::IRBuilder<> ir_builder_;
  llvm::ExecutionEngine* execution_engine_;
  union {
    void* query_cpu_code_;
    CUfunction query_gpu_code_;
  };
  std::unique_ptr<GpuExecutionContext> gpu_context_;
  mutable std::unordered_map<int, llvm::Value*> fetch_cache_;
  llvm::Function* row_func_;
  std::vector<int64_t> init_agg_vals_;
  std::unordered_map<int, int> global_to_local_col_ids_;
  std::vector<int> local_to_global_col_ids_;
  const size_t groups_buffer_entry_count_ { 2048 };
  boost::mutex reduce_mutex_;
  std::atomic<int32_t> query_id_;
};

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
  template<typename F, typename ...Args>
  static typename TimeT::rep execution(F func, Args&&... args)
  {
    auto start = std::chrono::system_clock::now();
    func(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);
    return duration.count();
  }
};

#endif // QUERYENGINE_EXECUTE_H
