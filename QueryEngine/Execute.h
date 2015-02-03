#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include "../Analyzer/Analyzer.h"
#include "../Planner/Planner.h"
#include "NvidiaKernel.h"

#include <boost/variant.hpp>
#include <glog/logging.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <cuda.h>

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

struct ResultRow {
  // TODO(alex): support for strings
  std::vector<int64_t> value_tuple;
  std::vector<AggResult> agg_results;
};

class Executor {
public:
  Executor(const Planner::RootPlan* root_plan);
  ~Executor();

  typedef std::tuple<std::string, const Analyzer::Expr*, int64_t> AggInfo;

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
  std::vector<ResultRow> executeAggScanPlan(
    const Planner::AggPlan* agg_plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&);
  void executeAggScanPlanWithGroupBy(
    std::vector<ResultRow>& results,
    const Planner::AggPlan* agg_plan,
    const ExecutorDeviceType device_type,
    const Catalog_Namespace::Catalog&,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows);
  void executeAggScanPlanWithoutGroupBy(
    std::vector<ResultRow>& results,
    const Planner::AggPlan* agg_plan,
    const ExecutorDeviceType device_type,
    const Catalog_Namespace::Catalog&,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows);
  void executeSimpleInsert();
  void executeScanPlan(const Planner::Scan* scan_plan);
  void compileAggScanPlan(
    const Planner::AggPlan* agg_plan,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const size_t groups_buffer_entry_count);
  void* optimizeAndCodegenCPU(llvm::Function*, const ExecutorOptLevel, llvm::Module*);
  CUfunction optimizeAndCodegenGPU(llvm::Function*, const ExecutorOptLevel, llvm::Module*);
  void call_aggregators(
    const std::vector<AggInfo>& agg_infos,
    llvm::Value* filter_result,
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    llvm::Module* module);
  void allocateLocalColumnIds(const Planner::Scan* scan_plan);
  int getLocalColumnId(const int global_col_id) const;

  typedef void (*agg_query)(
    const int8_t** col_buffers,
    const int64_t* num_rows,
    const int64_t* init_agg_value,
    int64_t** out);

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
