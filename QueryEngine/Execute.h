#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include "GroupByAndAggregate.h"
#include "JoinHashTable.h"
#include "../Analyzer/Analyzer.h"
#include "../Chunk/Chunk.h"
#include "../Fragmenter/Fragmenter.h"
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

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>
#include "../Shared/measure.h"

enum class NVVMBackend { CUDA, NVPTX };

enum class ExecutorOptLevel { Default, LoopStrengthReduction };

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

inline uint32_t log2_bytes(const uint32_t bytes) {
  switch (bytes) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 4:
      return 2;
    case 8:
      return 3;
    default:
      CHECK(false);
  }
}

namespace {

const ColumnDescriptor* get_column_descriptor(const int col_id,
                                              const int table_id,
                                              const Catalog_Namespace::Catalog& cat) {
  const auto col_desc = cat.getMetadataForColumn(table_id, col_id);
  CHECK(col_desc);
  return col_desc;
}

}  // namespace

struct ScanColDescriptor {
  ScanColDescriptor(const int col_id, const TableDescriptor* td, const int scan_idx)
      : col_id_(col_id), td_(td), scan_idx_(scan_idx) {}

  bool operator==(const ScanColDescriptor& that) const {
    return col_id_ == that.col_id_ && !!td_ == !!that.td_ && (!td_ || td_->tableId == that.td_->tableId) &&
           scan_idx_ == that.scan_idx_;
  }

  const int col_id_;
  const TableDescriptor* td_;
  const int scan_idx_;
};

namespace std {
template <>
struct hash<ScanColDescriptor> {
  size_t operator()(const ScanColDescriptor& scan_col_desc) const {
    return static_cast<size_t>(scan_col_desc.col_id_) ^ reinterpret_cast<size_t>(scan_col_desc.td_) ^
           scan_col_desc.scan_idx_;
  }
};
}

struct ScanId {
  ScanId(const int table_id, const int scan_idx) : table_id_(table_id), scan_idx_(scan_idx) {}

  bool operator==(const ScanId& that) const { return table_id_ == that.table_id_ && scan_idx_ == that.scan_idx_; }

  const int table_id_;
  const int scan_idx_;
};

namespace std {
template <>
struct hash<ScanId> {
  size_t operator()(const ScanId& scan_id) const { return scan_id.table_id_ ^ scan_id.scan_idx_; }
};
}

class Executor {
  static_assert(sizeof(float) == 4 && sizeof(double) == 8,
                "Host hardware not supported, unexpected size of float / double.");

 public:
  Executor(const int db_id,
           const size_t block_size_x,
           const size_t grid_size_x,
           const std::string& debug_dir,
           const std::string& debug_file);

  static std::shared_ptr<Executor> getExecutor(const int db_id,
                                               const std::string& debug_dir = "",
                                               const std::string& debug_file = "",
                                               const size_t block_size_x = 0,
                                               const size_t grid_size_x = 0);

  static void nukeCacheOfExecutors() {
    std::lock_guard<std::mutex> flush_lock(execute_mutex_);  // don't want native code to vanish while executing
    mapd_unique_lock<mapd_shared_mutex> lock(executors_cache_mutex_);
    (decltype(executors_){}).swap(executors_);
  }

  typedef std::tuple<std::string, const Analyzer::Expr*, int64_t, const size_t> AggInfo;

  ResultRows execute(const Planner::RootPlan* root_plan,
                     const bool hoist_literals,
                     const ExecutorDeviceType device_type,
                     const NVVMBackend nvvm_backend,
                     const ExecutorOptLevel,
                     const bool allow_multifrag,
                     const bool allow_joins);

  StringDictionary* getStringDictionary(const int dictId,
                                        const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const;

  bool isCPUOnly() const;

  typedef boost::variant<int8_t, int16_t, int32_t, int64_t, float, double, std::pair<std::string, int>, std::string>
      LiteralValue;
  typedef std::vector<LiteralValue> LiteralValues;

 private:
  template <class T>
  llvm::ConstantInt* ll_int(const T v) const {
    return static_cast<llvm::ConstantInt*>(
        llvm::ConstantInt::get(get_int_type(sizeof(v) * 8, cgen_state_->context_), v));
  }
  llvm::ConstantFP* ll_fp(const float v) const {
    return static_cast<llvm::ConstantFP*>(llvm::ConstantFP::get(llvm::Type::getFloatTy(cgen_state_->context_), v));
  }
  llvm::ConstantFP* ll_fp(const double v) const {
    return static_cast<llvm::ConstantFP*>(llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state_->context_), v));
  }
  std::vector<llvm::Value*> codegen(const Analyzer::Expr*, const bool fetch_columns, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::UOper*, const bool hoist_literals);
  std::vector<llvm::Value*> codegen(const Analyzer::ColumnVar*, const bool fetch_column, const bool hoist_literals);
  std::vector<llvm::Value*> codegen(const Analyzer::Constant*,
                                    const EncodingType enc_type,
                                    const int dict_id,
                                    const bool hoist_literals);
  std::vector<llvm::Value*> codegen(const Analyzer::CaseExpr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::ExtractExpr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::CharLengthExpr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::LikeExpr*, const bool hoist_literals);
  llvm::Value* codegen(const Analyzer::InValues*, const bool hoist_literals);
  llvm::Value* codegenCmp(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegenCmp(const SQLOps,
                          const SQLQualifier,
                          const Analyzer::Expr*,
                          const Analyzer::Expr*,
                          const bool hoist_literals);
  llvm::Value* codegenLogical(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* toBool(llvm::Value*);
  llvm::Value* codegenArith(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::Value* codegenDiv(llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&);
  llvm::Value* codegenMod(llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&);
  llvm::Value* codegenLogical(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenCast(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenUMinus(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenIsNull(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenUnnest(const Analyzer::UOper*, const bool hoist_literals);
  llvm::Value* codegenArrayAt(const Analyzer::BinOper*, const bool hoist_literals);
  llvm::ConstantInt* codegenIntConst(const Analyzer::Constant* constant);
  llvm::Value* colByteStream(const Analyzer::ColumnVar* col_var, const bool fetch_column, const bool hoist_literals);
  llvm::Value* posArg(const Analyzer::Expr*) const;
  const Analyzer::ColumnVar* hashJoinLhs(const Analyzer::ColumnVar* rhs) const;
  llvm::Value* fragRowOff() const;
  llvm::Value* rowsPerScan() const;
  llvm::ConstantInt* inlineIntNull(const SQLTypeInfo&);
  llvm::ConstantFP* inlineFpNull(const SQLTypeInfo&);

  ResultRows executeSelectPlan(const Planner::Plan* plan,
                               const int64_t limit,
                               const int64_t offset,
                               const bool hoist_literals,
                               const ExecutorDeviceType device_type,
                               const NVVMBackend,
                               const ExecutorOptLevel,
                               const Catalog_Namespace::Catalog&,
                               size_t& max_groups_buffer_entry_guess,
                               int32_t* error_code,
                               const Planner::Sort* sort_plan,
                               const bool allow_multifrag,
                               const bool just_explain,
                               const bool allow_joins);
  ResultRows executeAggScanPlan(const Planner::Plan* plan,
                                const int64_t limit,
                                const bool hoist_literals,
                                const ExecutorDeviceType device_type,
                                const NVVMBackend,
                                const ExecutorOptLevel,
                                const Catalog_Namespace::Catalog&,
                                std::shared_ptr<RowSetMemoryOwner>,
                                size_t& max_groups_buffer_entry_guess,
                                int32_t* error_code,
                                const Planner::Sort* sort_plan,
                                const bool output_columnar_hint,
                                const bool allow_multifrag,
                                const bool just_explain,
                                const bool allow_joins);
  ResultRows collectAllDeviceResults(std::vector<std::pair<ResultRows, std::vector<size_t>>>& all_fragment_results,
                                     const Planner::Plan* plan,
                                     const QueryMemoryDescriptor& query_mem_desc,
                                     const ExecutorDeviceType device_type,
                                     const std::vector<std::unique_ptr<QueryExecutionContext>>& query_contexts,
                                     std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                     const bool output_columnar);

  typedef std::deque<Fragmenter_Namespace::FragmentInfo> TableFragments;

  void dispatchFragments(const std::function<void(const ExecutorDeviceType chosen_device_type,
                                                  int chosen_device_id,
                                                  const std::map<int, std::vector<size_t>>& frag_ids,
                                                  const size_t ctx_idx,
                                                  const int64_t rowid_lookup_key)> dispatch,
                         const ExecutorDeviceType device_type,
                         const bool allow_multifrag,
                         const Planner::AggPlan* agg_plan,
                         const std::vector<ScanId>& scan_ids,
                         const std::map<int, const TableFragments*>& all_tables_fragments,
                         const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                         const std::vector<uint64_t>& all_frag_row_offsets,
                         const size_t context_count,
                         std::condition_variable& scheduler_cv,
                         std::mutex& scheduler_mutex,
                         std::unordered_set<int>& available_gpus,
                         int& available_cpus);

  std::vector<std::vector<const int8_t*>> fetchChunks(const std::list<ScanColDescriptor>&,
                                                      const int device_id,
                                                      const Data_Namespace::MemoryLevel,
                                                      const std::vector<ScanId>& scan_ids,
                                                      const std::map<int, const TableFragments*>&,
                                                      const std::map<int, std::vector<size_t>>& selected_fragments,
                                                      const Catalog_Namespace::Catalog&,
                                                      std::list<ChunkIter>&,
                                                      std::list<std::shared_ptr<Chunk_NS::Chunk>>&);

  void buildSelectedFragsMapping(std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
                                 std::vector<size_t>& local_col_to_frag_pos,
                                 const std::list<ScanColDescriptor>& col_global_ids,
                                 const std::map<int, std::vector<size_t>>& selected_fragments,
                                 const std::vector<ScanId>& scan_ids);

  ResultRows executeResultPlan(const Planner::Result* result_plan,
                               const bool hoist_literals,
                               const ExecutorDeviceType device_type,
                               const NVVMBackend,
                               const ExecutorOptLevel,
                               const Catalog_Namespace::Catalog&,
                               size_t& max_groups_buffer_entry_guess,
                               int32_t* error_code,
                               const Planner::Sort* sort_plan,
                               const bool allow_multifrag,
                               const bool just_explain,
                               const bool allow_joins);
  ResultRows executeSortPlan(const Planner::Sort* sort_plan,
                             const int64_t limit,
                             const int64_t offset,
                             const bool hoist_literals,
                             const ExecutorDeviceType device_type,
                             const NVVMBackend nvvm_backend,
                             const ExecutorOptLevel,
                             const Catalog_Namespace::Catalog&,
                             size_t& max_groups_buffer_entry_guess,
                             int32_t* error_code,
                             const bool allow_multifrag,
                             const bool just_explain,
                             const bool allow_joins);

  struct CompilationResult {
    std::vector<void*> native_functions;
    LiteralValues literal_values;
    QueryMemoryDescriptor query_mem_desc;
    bool output_columnar;
  };

  int32_t executePlanWithGroupBy(const CompilationResult&,
                                 const bool hoist_literals,
                                 ResultRows& results,
                                 const std::vector<Analyzer::Expr*>& target_exprs,
                                 const size_t group_by_col_count,
                                 const ExecutorDeviceType device_type,
                                 std::vector<std::vector<const int8_t*>>& col_buffers,
                                 const QueryExecutionContext*,
                                 const std::vector<int64_t>& num_rows,
                                 const std::vector<uint64_t>& dev_frag_row_offsets,
                                 Data_Namespace::DataMgr*,
                                 const int device_id,
                                 const int64_t limit,
                                 const bool was_auto_device,
                                 const uint32_t start_rowid,
                                 const uint32_t num_tables);
  int32_t executePlanWithoutGroupBy(const CompilationResult&,
                                    const bool hoist_literals,
                                    ResultRows& results,
                                    const std::vector<Analyzer::Expr*>& target_exprs,
                                    const ExecutorDeviceType device_type,
                                    std::vector<std::vector<const int8_t*>>& col_buffers,
                                    const QueryExecutionContext* query_exe_context,
                                    const std::vector<int64_t>& num_rows,
                                    const std::vector<uint64_t>& dev_frag_row_offsets,
                                    Data_Namespace::DataMgr* data_mgr,
                                    const int device_id,
                                    const uint32_t start_rowid,
                                    const uint32_t num_tables);
  int64_t getJoinHashTablePtr(const ExecutorDeviceType device_type, const int device_id);
  ResultRows reduceMultiDeviceResults(std::vector<std::pair<ResultRows, std::vector<size_t>>>& all_fragment_results,
                                      std::shared_ptr<RowSetMemoryOwner>,
                                      const QueryMemoryDescriptor&,
                                      const bool output_columnar) const;
  void executeSimpleInsert(const Planner::RootPlan* root_plan);

  enum class JoinImplType { Invalid, Loop, HashOneToOne };

  struct JoinInfo {
    JoinInfo(const JoinImplType join_impl_type,
             const std::vector<std::shared_ptr<Analyzer::BinOper>>& equi_join_tautologies,
             std::shared_ptr<JoinHashTable> join_hash_table)
        : join_impl_type_(join_impl_type),
          equi_join_tautologies_(equi_join_tautologies),
          join_hash_table_(join_hash_table) {}

    JoinImplType join_impl_type_;
    std::vector<std::shared_ptr<Analyzer::BinOper>> equi_join_tautologies_;  // expressions we equi-join on are true by
                                                                             // definition when using a hash join; we'll
                                                                             // fold them to true during code generation
    std::shared_ptr<JoinHashTable> join_hash_table_;
  };

  CompilationResult compilePlan(const Planner::Plan* plan,
                                const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
                                const std::vector<Executor::AggInfo>& agg_infos,
                                const std::vector<ScanId>& scan_ids,
                                const std::list<ScanColDescriptor>& scan_cols,
                                const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                                const std::list<std::shared_ptr<Analyzer::Expr>>& quals,
                                const bool hoist_literals,
                                const bool allow_multifrag,
                                const ExecutorDeviceType device_type,
                                const NVVMBackend nvvm_backend,
                                const ExecutorOptLevel,
                                const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                const bool allow_lazy_fetch,
                                std::shared_ptr<RowSetMemoryOwner>,
                                const size_t max_groups_buffer_entry_count,
                                const int64_t scan_limit,
                                const Planner::Sort* sort_plan,
                                const bool output_columnar_hint,
                                const bool serialize_llvm_ir,
                                std::string& llvm_ir,
                                const JoinInfo& join_info,
                                const bool allow_joins);

  void codegenInnerScanNextRow();

  void allocateInnerScansIterators(const std::vector<ScanId>& scan_ids, const bool allow_joins);

  JoinInfo chooseJoinType(const Planner::Join*,
                          const std::vector<Fragmenter_Namespace::QueryInfo>&,
                          const ExecutorDeviceType device_type);

  void bindInitGroupByBuffer(llvm::Function* query_func,
                             const QueryMemoryDescriptor& query_mem_desc,
                             const ExecutorDeviceType device_type);

  void nukeOldState(const bool allow_lazy_fetch, const JoinInfo& join_info);
  std::vector<void*> optimizeAndCodegenCPU(llvm::Function*,
                                           llvm::Function*,
                                           const bool hoist_literals,
                                           const ExecutorOptLevel,
                                           llvm::Module*);
  std::vector<void*> optimizeAndCodegenGPU(llvm::Function*,
                                           llvm::Function*,
                                           const bool hoist_literals,
                                           const NVVMBackend,
                                           const ExecutorOptLevel,
                                           llvm::Module*,
                                           const bool no_inline,
                                           const CudaMgr_Namespace::CudaMgr* cuda_mgr);
  std::string generatePTX(const std::string&) const;
  void initializeNVPTXBackend() const;

  int8_t warpSize() const;
  unsigned gridSize() const;
  unsigned blockSize() const;

  llvm::Value* groupByColumnCodegen(Analyzer::Expr* group_by_col,
                                    const bool hoist_literals,
                                    const bool translate_null_val,
                                    const int64_t translated_null_val,
                                    GroupByAndAggregate::DiamondCodegen&,
                                    std::stack<llvm::BasicBlock*>&);

  llvm::Value* toDoublePrecision(llvm::Value* val);

  void allocateLocalColumnIds(const std::list<ScanColDescriptor>& global_col_ids);
  int getLocalColumnId(const Analyzer::ColumnVar* col_var, const bool fetch_column) const;

  std::pair<bool, int64_t> skipFragment(const int table_id,
                                        const Fragmenter_Namespace::FragmentInfo& frag_info,
                                        const std::list<std::shared_ptr<Analyzer::Expr>>& simple_quals,
                                        const std::vector<uint64_t>& all_frag_row_offsets,
                                        const size_t frag_idx);

  typedef std::vector<std::string> CodeCacheKey;
  typedef std::vector<std::tuple<void*, std::unique_ptr<llvm::ExecutionEngine>, std::unique_ptr<GpuCompilationContext>>>
      CodeCacheVal;
  std::vector<void*> getCodeFromCache(const CodeCacheKey&,
                                      const std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>>&);
  void addCodeToCache(const CodeCacheKey&,
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
      case 7:
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
        : module_(nullptr),
          row_func_(nullptr),
          context_(llvm::getGlobalContext()),
          ir_builder_(context_),
          must_run_on_cpu_(false),
          uses_div_(false),
          literal_bytes_(0) {}

    size_t getOrAddLiteral(const Analyzer::Constant* constant, const EncodingType enc_type, const int dict_id) {
      const auto& ti = constant->get_type_info();
      const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
      switch (type) {
        case kBOOLEAN:
          return getOrAddLiteral(constant->get_is_null() ? int8_t(inline_int_null_val(ti))
                                                         : int8_t(constant->get_constval().boolval ? 1 : 0));
        case kSMALLINT:
          return getOrAddLiteral(constant->get_is_null() ? int16_t(inline_int_null_val(ti))
                                                         : constant->get_constval().smallintval);
        case kINT:
          return getOrAddLiteral(constant->get_is_null() ? int32_t(inline_int_null_val(ti))
                                                         : constant->get_constval().intval);
        case kBIGINT:
          return getOrAddLiteral(constant->get_is_null() ? int64_t(inline_int_null_val(ti))
                                                         : constant->get_constval().bigintval);
        case kFLOAT:
          return getOrAddLiteral(constant->get_is_null() ? float(inline_fp_null_val(ti))
                                                         : constant->get_constval().floatval);
        case kDOUBLE:
          return getOrAddLiteral(constant->get_is_null() ? inline_fp_null_val(ti) : constant->get_constval().doubleval);
        case kCHAR:
        case kTEXT:
        case kVARCHAR:
          CHECK(constant->get_constval().stringval);  // TODO(alex): support null
          if (enc_type == kENCODING_DICT) {
            return getOrAddLiteral(std::make_pair(*constant->get_constval().stringval, dict_id));
          }
          CHECK_EQ(kENCODING_NONE, enc_type);
          return getOrAddLiteral(*constant->get_constval().stringval);
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          // TODO(alex): support null
          return getOrAddLiteral(static_cast<int64_t>(constant->get_constval().timeval));
        default:
          CHECK(false);
      }
    }

    const LiteralValues& getLiterals() const { return literals_; }

    llvm::Value* addStringConstant(const std::string& str) {
      auto str_lv = ir_builder_.CreateGlobalString(str, "str_const_" + std::to_string(std::hash<std::string>()(str)));
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

    llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args) {
      auto f = module_->getFunction(fname);
      CHECK(f);
      return ir_builder_.CreateCall(f, args);
    }

    llvm::Module* module_;
    llvm::Function* row_func_;
    std::vector<llvm::Function*> helper_functions_;
    llvm::LLVMContext& context_;
    llvm::IRBuilder<> ir_builder_;
    std::unordered_map<int, std::vector<llvm::Value*>> fetch_cache_;
    std::vector<llvm::Value*> group_by_expr_cache_;
    std::vector<llvm::Value*> str_constants_;
    std::unordered_map<ScanId, std::pair<llvm::Value*, llvm::Value*>> scan_to_iterator_;
    std::vector<llvm::BasicBlock*> inner_scan_labels_;
    std::unordered_map<int, llvm::Value*> scan_idx_to_hash_pos_;
    bool must_run_on_cpu_;
    bool uses_div_;

   private:
    template <class T>
    size_t getOrAddLiteral(const T& val) {
      const Executor::LiteralValue var_val(val);
      size_t literal_found_off{0};
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
    PlanState(const bool allow_lazy_fetch, const JoinInfo& join_info, const Executor* executor)
        : allow_lazy_fetch_(allow_lazy_fetch), join_info_(join_info), executor_(executor) {}

    std::vector<int64_t> init_agg_vals_;
    std::vector<Analyzer::Expr*> target_exprs_;
    std::unordered_map<ScanColDescriptor, int> global_to_local_col_ids_;
    std::vector<int> local_to_global_col_ids_;
    std::unordered_set<int> columns_to_fetch_;
    std::unordered_set<int> columns_to_not_fetch_;
    bool allow_lazy_fetch_;
    JoinInfo join_info_;
    const Executor* executor_;

    bool isLazyFetchColumn(const Analyzer::Expr* target_expr) {
      if (!allow_lazy_fetch_) {
        return false;
      }
      const auto do_not_fetch_column = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      if (!do_not_fetch_column || dynamic_cast<const Analyzer::Var*>(do_not_fetch_column)) {
        return false;
      }
      auto cd = get_column_descriptor(
          do_not_fetch_column->get_column_id(), do_not_fetch_column->get_table_id(), *executor_->catalog_);
      if (cd->isVirtualCol) {
        return false;
      }
      std::unordered_set<int> intersect;
      std::set_intersection(columns_to_fetch_.begin(),
                            columns_to_fetch_.end(),
                            columns_to_not_fetch_.begin(),
                            columns_to_not_fetch_.end(),
                            std::inserter(intersect, intersect.begin()));
      if (!intersect.empty()) {
        throw std::exception();
      }
      return columns_to_fetch_.find(do_not_fetch_column->get_column_id()) == columns_to_fetch_.end();
    }
  };

  struct RowSetHolder {
    RowSetHolder(Executor* executor) : executor_(executor) {}

    ~RowSetHolder() { executor_->row_set_mem_owner_ = nullptr; }
    Executor* executor_;
  };

  std::unique_ptr<PlanState> plan_state_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  bool is_nested_;

  static const int max_gpu_count{16};
  std::mutex gpu_exec_mutex_[max_gpu_count];

  mutable std::shared_ptr<StringDictionary> lit_str_dict_;
  mutable std::mutex str_dict_mutex_;

  mutable std::unique_ptr<llvm::TargetMachine> nvptx_target_machine_;

  std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>> cpu_code_cache_;
  std::map<CodeCacheKey, std::pair<CodeCacheVal, llvm::Module*>> gpu_code_cache_;

  const size_t small_groups_buffer_entry_count_{512};
  const unsigned block_size_x_;
  const unsigned grid_size_x_;
  const std::string debug_dir_;
  const std::string debug_file_;

  const int db_id_;
  const Catalog_Namespace::Catalog* catalog_;

  static std::map<std::tuple<int, size_t, size_t>, std::shared_ptr<Executor>> executors_;
  static std::mutex execute_mutex_;
  static mapd_shared_mutex executors_cache_mutex_;

  static const int32_t ERR_DIV_BY_ZERO{1};
  static const int32_t ERR_OUT_OF_GPU_MEM{2};
  static const int32_t ERR_OUT_OF_SLOTS{3};
  static const int32_t ERR_UNSUPPORTED_SELF_JOIN{4};
  friend class GroupByAndAggregate;
  friend struct QueryMemoryDescriptor;
  friend class QueryExecutionContext;
  friend class ResultRows;
  friend class JoinHashTable;
};

#endif  // QUERYENGINE_EXECUTE_H
