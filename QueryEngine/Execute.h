#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include "../Analyzer/Analyzer.h"
#include "../Planner/Planner.h"
#include "../StringDictionary/StringDictionary.h"
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
#include <map>
#include <unordered_map>


enum class ExecutorOptLevel {
  Default,
  LoopStrengthReduction
};

enum class ExecutorDeviceType {
  CPU,
  GPU
};

typedef boost::variant<int64_t, double, std::string> AggResult;

class Executor;

inline bool approx_eq(const double v, const double target, const double eps = 0.01) {
  return target - eps < v && v < target + eps;
}

class ResultRow {
public:
  ResultRow(const Executor* executor) : executor_(executor) {}

  AggResult agg_result(const size_t idx, const bool translate_strings = true) const;

  SQLTypes agg_type(const size_t idx) const;

  size_t size() const {
    return agg_results_idx_.size();
  }

  std::vector<int64_t> value_tuple() const {
    return value_tuple_;
  }

  bool operator==(const ResultRow& r) const {
    if (size() != r.size()) {
      return false;
    }
    for (size_t idx = 0; idx < size(); ++idx) {
      const auto lhs_val = agg_result(idx);
      const auto rhs_val = agg_result(idx);
      {
        const auto lhs_pd = boost::get<double>(&lhs_val);
        if (lhs_pd) {
          const auto rhs_pd = boost::get<double>(&rhs_val);
          if (!rhs_pd) {
            return false;
          }
          if (!approx_eq(*lhs_pd, *rhs_pd)) {
            return false;
          }
        } else {
          if (lhs_val < rhs_val || rhs_val < lhs_val) {
            return false;
          }
        }
      }
    }
    return true;
  }

private:
  // TODO(alex): support for strings
  std::vector<int64_t> value_tuple_;
  std::vector<int64_t> agg_results_;
  std::vector<size_t> agg_results_idx_;
  std::vector<SQLAgg> agg_kinds_;
  std::vector<SQLTypes> agg_types_;
  const Executor* executor_;

  friend class Executor;
};

class Executor {
  static_assert(sizeof(float) == 4 && sizeof(double) == 8,
    "Host hardware not supported, unexpected size of float / double.");
public:
  Executor(const int db_id);

  static std::shared_ptr<Executor> getExecutor(const int db_id);

  typedef std::tuple<std::string, const Analyzer::Expr*, int64_t, void*> AggInfo;
  typedef std::vector<ResultRow> ResultRows;

  std::vector<ResultRow> execute(
    const Planner::RootPlan* root_plan,
    const bool hoist_literals = true,
    const ExecutorDeviceType device_type = ExecutorDeviceType::CPU,
    const ExecutorOptLevel = ExecutorOptLevel::Default);

  StringDictionary* getStringDictionary() const;

  typedef boost::variant<bool, int16_t, int32_t, int64_t, float, double, std::string> LiteralValue;
  typedef std::vector<Executor::LiteralValue> LiteralValues;

private:
  llvm::Value* codegen(const Analyzer::Expr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::ColumnVar*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::Constant*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::CaseExpr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::ExtractExpr*, const bool hoist_literals);
  llvm::Value* codegenCmp(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegenLogical(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegenArith(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegenLogical(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenCast(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenUMinus(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenIsNull(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* inlineIntNull(const SQLTypes);
  std::vector<ResultRow> executeSelectPlan(
    const Planner::Plan* plan,
    const Planner::RootPlan* root_plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel);
  std::vector<ResultRow> executeAggScanPlan(
    const Planner::Plan* plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&);
  std::vector<ResultRow> executeResultPlan(
    const Planner::Result* result_plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&);
  std::vector<ResultRow> executeSortPlan(
    const Planner::Sort* sort_plan,
    const Planner::RootPlan* root_plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&);
  void executePlanWithGroupBy(
    void* query_native_code,
    const bool hoist_literals,
    const LiteralValues& hoisted_literals,
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const size_t group_by_col_count,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows,
    Data_Namespace::DataMgr*,
    const int32_t db_id);
  void executePlanWithoutGroupBy(
    void* query_native_code,
    const bool hoist_literals,
    const LiteralValues& hoisted_literals,
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const int64_t num_rows,
    Data_Namespace::DataMgr* data_mgr);
  ResultRows reduceMultiDeviceResults(const std::vector<ResultRows>&);
  ResultRows groupBufferToResults(
    const int64_t* group_by_buffer,
    const size_t group_by_col_count,
    const size_t agg_col_count,
    const std::list<Analyzer::Expr*>& target_exprs,
    const int32_t db_id);
  void executeSimpleInsert(const Planner::RootPlan* root_plan);

  std::pair<void*, LiteralValues> compilePlan(
    const std::vector<Executor::AggInfo>& agg_infos,
    const std::list<Analyzer::Expr*>& groupby_list,
    const std::list<int>& scan_cols,
    const std::list<Analyzer::Expr*>& simple_quals,
    const std::list<Analyzer::Expr*>& quals,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const size_t groups_buffer_entry_count);

  void nukeOldState();
  void* optimizeAndCodegenCPU(llvm::Function*,
                              const bool hoist_literals,
                              const ExecutorOptLevel,
                              llvm::Module*);
  CUfunction optimizeAndCodegenGPU(llvm::Function*,
                                   const bool hoist_literals,
                                   const ExecutorOptLevel,
                                   llvm::Module*,
                                   const bool is_group_by);
  void call_aggregators(
    const std::vector<AggInfo>& agg_infos,
    llvm::Value* filter_result,
    const std::list<Analyzer::Expr*>& group_by_cols,
    const int32_t groups_buffer_entry_count,
    llvm::Module* module,
    const bool hoist_literals);
  void allocateLocalColumnIds(const std::list<int>& global_col_ids);
  int getLocalColumnId(const int global_col_id) const;

  typedef std::pair<std::string, std::string> CodeCacheKey;
  typedef std::tuple<void*, std::unique_ptr<llvm::ExecutionEngine>, std::unique_ptr<GpuExecutionContext>> CodeCacheVal;
  void* getCodeFromCache(
    const CodeCacheKey&,
    const std::map<CodeCacheKey, CodeCacheVal>&);
  void addCodeToCache(
    const CodeCacheKey&,
    void* native_code,
    std::map<CodeCacheKey, CodeCacheVal>&,
    llvm::ExecutionEngine*,
    GpuExecutionContext*);

  std::vector<int8_t> serializeLiterals(const Executor::LiteralValues& literals);

  static size_t literalBytes(const LiteralValue& lit) {
    switch (lit.which()) {
      case 0:
        return 1;
      case 1:
        return 2;
      case 2:
        return 4;
      case 3:
        return 8;
      case 4:
        return 4;
      case 5:
        return 8;
      case 6:
        return 4;
      default:
        CHECK(false);
    }
  }

  static size_t addAligned(const size_t off_in, const size_t alignment) {
    size_t off = off_in;
    if (off % alignment != 0) {
      off += (alignment - off % alignment);
    }
    return off + alignment;
  }

  struct CgenState {
  public:
    CgenState()
      : module_(nullptr)
      , row_func_(nullptr)
      , context_(llvm::getGlobalContext())
      , ir_builder_(context_)
      , literal_bytes_(0) {}

    size_t getOrAddLiteral(const Analyzer::Constant* constant) {
      const auto& type_info = constant->get_type_info();
      switch (type_info.type) {
      case kBOOLEAN:
        return getOrAddLiteral(constant->get_constval().boolval);
      case kSMALLINT:
        return getOrAddLiteral(constant->get_constval().smallintval);
      case kINT:
        return getOrAddLiteral(constant->get_constval().intval);
      case kBIGINT:
        return getOrAddLiteral(constant->get_constval().bigintval);
      case kFLOAT:
        return getOrAddLiteral(constant->get_constval().floatval);
      case kDOUBLE:
        return getOrAddLiteral(constant->get_constval().doubleval);
      case kVARCHAR:
        return getOrAddLiteral(*constant->get_constval().stringval);
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
        return getOrAddLiteral(static_cast<int64_t>(constant->get_constval().timeval));
      default:
        CHECK(false);
      }
    }

    const LiteralValues& getLiterals() const {
      return literals_;
    }

    llvm::Module* module_;
    llvm::Function* row_func_;
    llvm::LLVMContext& context_;
    llvm::IRBuilder<> ir_builder_;
    std::unordered_map<int, llvm::Value*> fetch_cache_;
    std::vector<llvm::Value*> group_by_expr_cache_;
  private:
    template<class T>
    size_t getOrAddLiteral(const T& val) {
      const Executor::LiteralValue var_val(val);
      size_t literal_found_off { 0 };
      for (const auto& literal : literals_) {
        const auto lit_bytes = literalBytes(literal);
        literal_found_off = addAligned(literal_found_off, lit_bytes);
        if (literal == var_val) {
          return literal_found_off - lit_bytes;
        }
      }
      literals_.emplace_back(val);
      const auto lit_bytes = literalBytes(var_val);
      literal_bytes_ = addAligned(literal_bytes_, lit_bytes);
      return literal_bytes_ - lit_bytes;
    }

    LiteralValues literals_;
    size_t literal_bytes_;
  };
  std::unique_ptr<CgenState> cgen_state_;

  struct PlanState {
    std::vector<int64_t> init_agg_vals_;
    std::unordered_map<int, int> global_to_local_col_ids_;
    std::vector<int> local_to_global_col_ids_;
  };
  std::unique_ptr<PlanState> plan_state_;

  bool is_nested_;

  boost::mutex reduce_mutex_;

  mutable std::unique_ptr<StringDictionary> str_dict_;

  std::map<CodeCacheKey, CodeCacheVal> cpu_code_cache_;
  std::map<CodeCacheKey, CodeCacheVal> gpu_code_cache_;

  const size_t groups_buffer_entry_count_ { 2048 };
  const unsigned block_size_x_ { 16 };
  const unsigned grid_size_x_ { 16 };

  const int db_id_;

  static std::unordered_map<int, std::shared_ptr<Executor>> executors_;
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
