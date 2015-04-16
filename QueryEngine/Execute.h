#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include "GroupByAndAggregate.h"
#include "../Analyzer/Analyzer.h"
#include "../Planner/Planner.h"
#include "../StringDictionary/StringDictionary.h"
#include "NvidiaKernel.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <cuda.h>

#include <map>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include "../Shared/measure.h"


enum class ExecutorOptLevel {
  Default,
  LoopStrengthReduction
};

class Executor;

inline llvm::Type* get_int_type(const int width, llvm::LLVMContext& context) {
  switch (width) {
  case 64:
    return llvm::Type::getInt64Ty(context);
  case 32:
    return llvm::Type::getInt32Ty(context);
    break;
  case 16:
    return llvm::Type::getInt16Ty(context);
    break;
  case 8:
    return llvm::Type::getInt8Ty(context);
    break;
  case 1:
    return llvm::Type::getInt1Ty(context);
    break;
  default:
    LOG(FATAL) << "Unsupported integer width: " << width;
  }
}

class Executor {
  static_assert(sizeof(float) == 4 && sizeof(double) == 8,
    "Host hardware not supported, unexpected size of float / double.");
public:
  Executor(const int db_id, const size_t block_size_x, const size_t grid_size_x);

  static std::shared_ptr<Executor> getExecutor(
    const int db_id,
    const size_t block_size_x = 1024,
    const size_t grid_size_x = 12);

  typedef std::tuple<std::string, const Analyzer::Expr*, int64_t, const size_t> AggInfo;
  typedef std::vector<ResultRow> ResultRows;

  std::vector<ResultRow> execute(
    const Planner::RootPlan* root_plan,
    const bool hoist_literals = true,
    const ExecutorDeviceType device_type = ExecutorDeviceType::CPU,
    const ExecutorOptLevel = ExecutorOptLevel::Default);

  StringDictionary* getStringDictionary(const int dictId) const;

  typedef boost::variant<bool, int16_t, int32_t, int64_t, float, double, std::pair<std::string, int>> LiteralValue;
  typedef std::vector<Executor::LiteralValue> LiteralValues;

private:
  template<class T>
  llvm::ConstantInt* ll_int(const T v) {
    return static_cast<llvm::ConstantInt*>(llvm::ConstantInt::get(
      get_int_type(sizeof(v) * 8, cgen_state_->context_), v));
  }
  std::vector<llvm::Value*> codegen(const Analyzer::Expr*, const bool fetch_columns, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::UOper*, const bool hoist_literals);
  std::vector<llvm::Value*> codegen(const Analyzer::ColumnVar*, const bool fetch_column, const bool hoist_literals);
  std::vector<llvm::Value*> codegen(const Analyzer::Constant*, const int dict_id, const bool hoist_literals);
  std::vector<llvm::Value*> codegen(const Analyzer::CaseExpr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::ExtractExpr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::LikeExpr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::InValues*, const bool hoist_literals);
  llvm::Value* codegenCmp(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegenCmp(const SQLOps, const Analyzer::Expr*, const Analyzer::Expr*, const bool hoist_literals);
  llvm::Value* codegenLogical(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegenArith(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegenDiv(llvm::Value*, llvm::Value*);
  llvm::Value* codegenLogical(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenCast(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenUMinus(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenIsNull(const Analyzer::UOper*, const bool hoist_literals);
  llvm::ConstantInt* codegenIntConst(const Analyzer::Constant* constant);
  std::pair<llvm::Value*, llvm::Value*>
  colByteStream(const int col_id, const bool fetch_column, const bool hoist_literals);
  llvm::ConstantInt* inlineIntNull(const SQLTypeInfo&);
  std::vector<ResultRow> executeSelectPlan(
    const Planner::Plan* plan,
    const int64_t limit,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&,
    const size_t max_groups_buffer_entry_count,
    int32_t* error_code);
  std::vector<ResultRow> executeAggScanPlan(
    const Planner::Plan* plan,
    const int64_t limit,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&,
    std::shared_ptr<RowSetMemoryOwner>,
    const size_t max_groups_buffer_entry_count,
    int32_t* error_code);
  std::vector<ResultRow> executeResultPlan(
    const Planner::Result* result_plan,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&,
    const size_t max_groups_buffer_entry_count,
    int32_t* error_code);
  std::vector<ResultRow> executeSortPlan(
    const Planner::Sort* sort_plan,
    const int64_t limit,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const Catalog_Namespace::Catalog&,
    const size_t max_groups_buffer_entry_count,
    int32_t* error_code);

  struct CompilationResult {
    std::vector<void*> native_functions;
    LiteralValues literal_values;
    QueryMemoryDescriptor query_mem_desc;
  };

  int32_t executePlanWithGroupBy(
    const CompilationResult&,
    const bool hoist_literals,
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const size_t group_by_col_count,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const QueryExecutionContext*,
    const int64_t num_rows,
    Data_Namespace::DataMgr*,
    const int device_id,
    const int64_t limit);
  int32_t executePlanWithoutGroupBy(
    const CompilationResult&,
    const bool hoist_literals,
    std::vector<ResultRow>& results,
    const std::vector<Analyzer::Expr*>& target_exprs,
    const ExecutorDeviceType device_type,
    std::vector<const int8_t*>& col_buffers,
    const QueryExecutionContext* query_exe_context,
    const int64_t num_rows,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id);
  ResultRows reduceMultiDeviceResults(
    const std::vector<ResultRows>&,
    const QueryMemoryDescriptor&,
    std::shared_ptr<RowSetMemoryOwner>) const;
  void executeSimpleInsert(const Planner::RootPlan* root_plan);

  CompilationResult compilePlan(
    const Planner::Plan* plan,
    const Fragmenter_Namespace::QueryInfo& query_info,
    const std::vector<Executor::AggInfo>& agg_infos,
    const std::list<int>& scan_cols,
    const std::list<Analyzer::Expr*>& simple_quals,
    const std::list<Analyzer::Expr*>& quals,
    const bool hoist_literals,
    const ExecutorDeviceType device_type,
    const ExecutorOptLevel,
    const CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const bool allow_lazy_fetch,
    std::shared_ptr<RowSetMemoryOwner>,
    const size_t max_groups_buffer_entry_count);

  void nukeOldState(const bool allow_lazy_fetch);
  std::vector<void*> optimizeAndCodegenCPU(llvm::Function*,
                                           const bool hoist_literals,
                                           const ExecutorOptLevel,
                                           llvm::Module*);
  std::vector<void*> optimizeAndCodegenGPU(llvm::Function*,
                                           const bool hoist_literals,
                                           const ExecutorOptLevel,
                                           llvm::Module*,
                                           const bool no_inline,
                                           const CudaMgr_Namespace::CudaMgr* cuda_mgr);

  int8_t warpSize() const;

  llvm::Value* groupByColumnCodegen(Analyzer::Expr* group_by_col, const bool hoist_literals);

  llvm::Value* toDoublePrecision(llvm::Value* val);

  void allocateLocalColumnIds(const std::list<int>& global_col_ids);
  int getLocalColumnId(const int global_col_id, const bool fetch_column) const;

  bool skipFragment(
    const Fragmenter_Namespace::FragmentInfo& frag_info,
    const std::list<Analyzer::Expr*>& simple_quals);

  typedef std::pair<std::string, std::string> CodeCacheKey;
  typedef std::vector<std::tuple<void*,
                                 std::unique_ptr<llvm::ExecutionEngine>,
                                 std::unique_ptr<GpuCompilationContext>>> CodeCacheVal;
  std::vector<void*> getCodeFromCache(
    const CodeCacheKey&,
    const std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>&);
  void addCodeToCache(
    const CodeCacheKey&,
    const std::vector<std::tuple<void*, llvm::ExecutionEngine*, GpuCompilationContext*>>&,
    llvm::Module*,
    std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>&);

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
      , must_run_on_cpu_(false)
      , uses_div_(false)
      , literal_bytes_(0) {}

    size_t getOrAddLiteral(const Analyzer::Constant* constant, const int dict_id) {
      const auto& type_info = constant->get_type_info();
      switch (type_info.get_type()) {
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
      case kCHAR:
      case kTEXT:
      case kVARCHAR:
        return getOrAddLiteral(std::make_pair(*constant->get_constval().stringval, dict_id));
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

    llvm::Value* getCurrentRowIndex() const {
      llvm::Value* pos_arg { nullptr };
      auto& in_arg_list = row_func_->getArgumentList();
      for (auto& arg : in_arg_list) {
        if (arg.getType()->isIntegerTy()) {
          pos_arg = &arg;
          break;
        }
      }
      CHECK(pos_arg);
      CHECK_EQ(static_cast<llvm::IntegerType*>(pos_arg->getType())->getBitWidth(), 64);
      return pos_arg;
    }

    llvm::Value* addStringConstant(const std::string& str) {
      auto str_lv = ir_builder_.CreateGlobalString(
        str, "str_const_" + std::to_string(std::hash<std::string>()(str)));
      auto i8_ptr = llvm::PointerType::get(get_int_type(8, context_), 0);
      str_constants_.push_back(str_lv);
      str_lv = ir_builder_.CreateBitCast(str_lv, i8_ptr);
      return str_lv;
    }

    // look up a runtime function based on the name, return type and type of
    // the arguments and call it; x64 only, don't call from GPU codegen
    llvm::Value* emitExternalCall(const std::string& fname,
                                  llvm::Type* ret_type,
                                  const std::vector<llvm::Value*> args) {
      std::vector<llvm::Type*> arg_types;
      for (const auto arg : args) {
        arg_types.push_back(arg->getType());
      }
      auto func_ty = llvm::FunctionType::get(ret_type, arg_types, false);
      auto func_p = module_->getOrInsertFunction(fname, func_ty);
      CHECK(func_p);
      llvm::Value* result = ir_builder_.CreateCall(func_p, args);
      // check the assumed type
      CHECK_EQ(result->getType(), ret_type);
      return result;
    }

    llvm::Value* emitCall(const std::string& fname,
                          const std::vector<llvm::Value*>& args) {
      auto f = module_->getFunction(fname);
      CHECK(f);
      return ir_builder_.CreateCall(f, args);
    }

    llvm::Module* module_;
    llvm::Function* row_func_;
    llvm::LLVMContext& context_;
    llvm::IRBuilder<> ir_builder_;
    std::unordered_map<int, std::vector<llvm::Value*>> fetch_cache_;
    std::vector<llvm::Value*> group_by_expr_cache_;
    std::vector<llvm::Value*> str_constants_;
    bool must_run_on_cpu_;
    bool uses_div_;
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
    PlanState(const bool allow_lazy_fetch) : allow_lazy_fetch_(allow_lazy_fetch) {}

    std::vector<int64_t> init_agg_vals_;
    std::vector<Analyzer::Expr*> target_exprs_;
    std::unordered_map<int, int> global_to_local_col_ids_;
    std::vector<int> local_to_global_col_ids_;
    std::unordered_set<int> columns_to_fetch_;
    std::unordered_set<int> columns_to_not_fetch_;
    bool allow_lazy_fetch_;

    bool isLazyFetchColumn(const Analyzer::Expr* target_expr) {
      if (!allow_lazy_fetch_) {
        return false;
      }
      const auto do_not_fetch_column = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      if (!do_not_fetch_column || dynamic_cast<const Analyzer::Var*>(do_not_fetch_column)) {
        return false;
      }
      std::unordered_set<int> intersect;
      set_intersection(columns_to_fetch_.begin(), columns_to_fetch_.end(),
          columns_to_not_fetch_.begin(), columns_to_not_fetch_.end(),
          std::inserter(intersect, intersect.begin()));
      if (!intersect.empty()) {
        throw std::exception();
      }
      return columns_to_fetch_.find(do_not_fetch_column->get_column_id()) == columns_to_fetch_.end();
    }
  };
  std::unique_ptr<PlanState> plan_state_;

  bool is_nested_;

  static const int max_gpu_count { 8 };
  std::mutex gpu_exec_mutex_[max_gpu_count];

  mutable std::unordered_map<int, std::unique_ptr<StringDictionary>> str_dicts_;
  mutable std::mutex str_dicts_mutex_;

  std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>> cpu_code_cache_;
  std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>> gpu_code_cache_;

  const size_t small_groups_buffer_entry_count_ { 512 };
  const unsigned block_size_x_;
  const unsigned grid_size_x_;

  const int db_id_;
  const Catalog_Namespace::Catalog* catalog_;

  static std::map<std::tuple<int, size_t, size_t>, std::shared_ptr<Executor>> executors_;
  static std::mutex execute_mutex_;

  static const int32_t ERR_DIV_BY_ZERO { 1 };
  friend class GroupByAndAggregate;
  friend struct QueryMemoryDescriptor;
  friend class QueryExecutionContext;
  friend class ResultRow;
};

#endif // QUERYENGINE_EXECUTE_H
