/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <llvm/IR/Value.h>

#include "../Analyzer/Analyzer.h"
#include "Execute.h"
#include "Shared/DbObjectKeys.h"

// Code generation utility to be used for queries and scalar expressions.
class CodeGenerator {
 public:
  CodeGenerator(Executor* executor)
      : executor_(executor)
      , cgen_state_(executor->cgen_state_.get())
      , plan_state_(executor->plan_state_.get()) {}

  // Overload which can be used without an executor, for SQL scalar expression code
  // generation.
  CodeGenerator(CgenState* cgen_state, PlanState* plan_state)
      : executor_(nullptr), cgen_state_(cgen_state), plan_state_(plan_state) {}

  // Generates IR value(s) for the given analyzer expression.
  std::vector<llvm::Value*> codegen(const Analyzer::Expr*,
                                    const bool fetch_columns,
                                    const CompilationOptions&);

  llvm::Value* codegenPerRowStringOper(const Analyzer::StringOper* string_oper,
                                       const CompilationOptions& co);

  llvm::Value* codegenPseudoStringOper(
      const Analyzer::ColumnVar*,
      const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
      const CompilationOptions&);

  // Generates constant values in the literal buffer of a query.
  std::vector<llvm::Value*> codegenHoistedConstants(
      const std::vector<const Analyzer::Constant*>& constants,
      const EncodingType enc_type,
      const shared::StringDictKey& dict_id);

  static llvm::ConstantInt* codegenIntConst(const Analyzer::Constant* constant,
                                            CgenState* cgen_state);

  llvm::Value* codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                          const SQLTypeInfo& operand_ti,
                                          const SQLTypeInfo& ti,
                                          bool upscale = true);

  void codegenCastBetweenIntTypesOverflowChecks(llvm::Value* operand_lv,
                                                const SQLTypeInfo& operand_ti,
                                                const SQLTypeInfo& ti,
                                                const int64_t scale);

  // Generates the index of the current row in the context of query execution.
  llvm::Value* posArg(const Analyzer::Expr*) const;

  llvm::Value* toBool(llvm::Value*);

  llvm::Value* castArrayPointer(llvm::Value* ptr, const SQLTypeInfo& elem_ti);

  static std::unordered_set<llvm::Function*> markDeadRuntimeFuncs(
      llvm::Module& module,
      const std::vector<llvm::Function*>& roots,
      const std::vector<llvm::Function*>& leaves);

  static ExecutionEngineWrapper generateNativeCPUCode(
      llvm::Function* func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const CompilationOptions& co);

  static std::string generatePTX(const std::string& cuda_llir,
                                 llvm::TargetMachine* nvptx_target_machine,
                                 llvm::LLVMContext& context);

  static std::unique_ptr<llvm::TargetMachine> initializeNVPTXBackend(
      const CudaMgr_Namespace::NvidiaDeviceArch arch);

  static bool alwaysCloneRuntimeFunction(const llvm::Function* func);

  struct GPUTarget {
    llvm::TargetMachine* nvptx_target_machine;
    const CudaMgr_Namespace::CudaMgr* cuda_mgr;
    CgenState* cgen_state;
    bool row_func_not_inlined;
  };

  static void linkModuleWithLibdevice(Executor* executor,
                                      llvm::Module& module,
                                      llvm::PassManagerBuilder& pass_manager_builder,
                                      const GPUTarget& gpu_target);

  static std::shared_ptr<GpuCompilationContext> generateNativeGPUCode(
      Executor* executor,
      llvm::Function* func,
      llvm::Function* wrapper_func,
      const std::unordered_set<llvm::Function*>& live_funcs,
      const bool is_gpu_smem_used,
      const CompilationOptions& co,
      const GPUTarget& gpu_target);

  static void link_udf_module(const std::unique_ptr<llvm::Module>& udf_module,
                              llvm::Module& module,
                              CgenState* cgen_state,
                              llvm::Linker::Flags flags = llvm::Linker::Flags::None);

  static bool prioritizeQuals(const RelAlgExecutionUnit& ra_exe_unit,
                              std::vector<Analyzer::Expr*>& primary_quals,
                              std::vector<Analyzer::Expr*>& deferred_quals,
                              const PlanState::HoistedFiltersSet& hoisted_quals);

  struct ExecutorRequired : public std::runtime_error {
    ExecutorRequired()
        : std::runtime_error("Executor required to generate this expression") {}
  };

  struct NullCheckCodegen {
    NullCheckCodegen(CgenState* cgen_state,
                     Executor* executor,
                     llvm::Value* nullable_lv,
                     const SQLTypeInfo& nullable_ti,
                     const std::string& name = "");

    llvm::Value* finalize(llvm::Value* null_lv, llvm::Value* notnull_lv);

    CgenState* cgen_state{nullptr};
    std::string name;
    llvm::BasicBlock* nullcheck_bb{nullptr};
    llvm::PHINode* nullcheck_value{nullptr};
    std::unique_ptr<DiamondCodegen> null_check;
  };

  static ArrayLoadCodegen codegenGeoArrayLoadAndNullcheck(llvm::Value* byte_stream,
                                                          llvm::Value* pos,
                                                          const SQLTypeInfo& ti,
                                                          CgenState* cgen_state);

  llvm::Value* codegenCastBetweenTimestamps(llvm::Value* ts_lv,
                                            const SQLTypeInfo& operand_dimen,
                                            const SQLTypeInfo& target_dimen,
                                            const bool nullable);

  // Generate the position for the given window function and the query iteration position.
  llvm::Value* codegenWindowPosition(const WindowFunctionContext* window_func_context,
                                     llvm::Value* pos_arg);

 private:
  std::vector<llvm::Value*> codegen(const Analyzer::Constant*,
                                    const EncodingType enc_type,
                                    const shared::StringDictKey& dict_id,
                                    const CompilationOptions&);

  virtual std::vector<llvm::Value*> codegenColumn(const Analyzer::ColumnVar*,
                                                  const bool fetch_column,
                                                  const CompilationOptions&);

  llvm::Value* codegenArith(const Analyzer::BinOper*, const CompilationOptions&);

  llvm::Value* codegenUMinus(const Analyzer::UOper*, const CompilationOptions&);

  llvm::Value* codegenCmp(const Analyzer::BinOper*, const CompilationOptions&);

  llvm::Value* codegenCmp(const SQLOps,
                          const SQLQualifier,
                          std::vector<llvm::Value*>,
                          const SQLTypeInfo&,
                          const Analyzer::Expr*,
                          const CompilationOptions&);

  llvm::Value* codegenIsNull(const Analyzer::UOper*, const CompilationOptions&);

  llvm::Value* codegenIsNullNumber(llvm::Value*, const SQLTypeInfo&);

  llvm::Value* codegenLogical(const Analyzer::BinOper*, const CompilationOptions&);

  llvm::Value* codegenLogical(const Analyzer::UOper*, const CompilationOptions&);

  llvm::Value* codegenCast(const Analyzer::UOper*, const CompilationOptions&);

  llvm::Value* codegenCast(llvm::Value* operand_lv,
                           const SQLTypeInfo& operand_ti,
                           const SQLTypeInfo& ti,
                           const bool operand_is_const,
                           const CompilationOptions& co);

  llvm::Value* codegen(const Analyzer::InValues*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::InIntegerSet* expr, const CompilationOptions& co);

  std::vector<llvm::Value*> codegen(const Analyzer::CaseExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::ExtractExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::DateaddExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::DatediffExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::DatetruncExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::CharLengthExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::KeyForStringExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::SampleRatioExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::WidthBucketExpr*, const CompilationOptions&);

  llvm::Value* codegenConstantWidthBucketExpr(const Analyzer::WidthBucketExpr*,
                                              const CompilationOptions&);

  llvm::Value* codegenWidthBucketExpr(const Analyzer::WidthBucketExpr*,
                                      const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::StringOper*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::LikeExpr*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::RegexpExpr*, const CompilationOptions&);

  llvm::Value* codegenUnnest(const Analyzer::UOper*, const CompilationOptions&);

  llvm::Value* codegenArrayAt(const Analyzer::BinOper*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::CardinalityExpr*, const CompilationOptions&);

  std::vector<llvm::Value*> codegenArrayExpr(const Analyzer::ArrayExpr*,
                                             const CompilationOptions&);

  std::vector<llvm::Value*> codegenGeoColumnVar(const Analyzer::GeoColumnVar*,
                                                const bool fetch_columns,
                                                const CompilationOptions& co);

  std::vector<llvm::Value*> codegenGeoExpr(const Analyzer::GeoExpr*,
                                           const CompilationOptions&);

  std::vector<llvm::Value*> codegenGeoConstant(const Analyzer::GeoConstant*,
                                               const CompilationOptions&);

  std::vector<llvm::Value*> codegenGeoOperator(const Analyzer::GeoOperator*,
                                               const CompilationOptions&);

  std::vector<llvm::Value*> codegenGeoUOper(const Analyzer::GeoUOper*,
                                            const CompilationOptions&);

  std::vector<llvm::Value*> codegenGeoBinOper(const Analyzer::GeoBinOper*,
                                              const CompilationOptions&);

  std::vector<llvm::Value*> codegenGeosPredicateCall(const std::string&,
                                                     std::vector<llvm::Value*>,
                                                     const CompilationOptions&);

  std::vector<llvm::Value*> codegenGeosConstructorCall(const std::string&,
                                                       std::vector<llvm::Value*>,
                                                       llvm::Value*,
                                                       const CompilationOptions&);

  std::vector<llvm::Value*> codegenGeoArgs(
      const std::vector<std::shared_ptr<Analyzer::Expr>>&,
      const CompilationOptions&);

  llvm::Value* codegenFunctionOper(const Analyzer::FunctionOper*,
                                   const CompilationOptions&);

  llvm::Value* codegenFunctionOperWithCustomTypeHandling(
      const Analyzer::FunctionOperWithCustomTypeHandling*,
      const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::BinOper*, const CompilationOptions&);

  llvm::Value* codegen(const Analyzer::UOper*, const CompilationOptions&);

  std::vector<llvm::Value*> codegenHoistedConstantsLoads(
      const SQLTypeInfo& type_info,
      const EncodingType enc_type,
      const shared::StringDictKey& dict_id,
      const int16_t lit_off,
      const size_t lit_bytes);

  std::vector<llvm::Value*> codegenHoistedConstantsPlaceholders(
      const SQLTypeInfo& type_info,
      const EncodingType enc_type,
      const int16_t lit_off,
      const std::vector<llvm::Value*>& literal_loads);

  std::vector<llvm::Value*> codegenColVar(const Analyzer::ColumnVar*,
                                          const bool fetch_column,
                                          const bool update_query_plan,
                                          const CompilationOptions&);

  llvm::Value* codegenFixedLengthColVar(
      const Analyzer::ColumnVar* col_var,
      llvm::Value* col_byte_stream,
      llvm::Value* pos_arg,
      const WindowFunctionContext* window_function_context = nullptr);

  // Generates code for a fixed length column when a window function is active.
  llvm::Value* codegenFixedLengthColVarInWindow(
      const Analyzer::ColumnVar* col_var,
      llvm::Value* col_byte_stream,
      llvm::Value* pos_arg,
      const WindowFunctionContext* window_function_context = nullptr);

  std::vector<llvm::Value*> codegenVariableLengthStringColVar(
      llvm::Value* col_byte_stream,
      llvm::Value* pos_arg);

  llvm::Value* codegenRowId(const Analyzer::ColumnVar* col_var,
                            const CompilationOptions& co);

  llvm::Value* codgenAdjustFixedEncNull(llvm::Value*, const SQLTypeInfo&);

  std::vector<llvm::Value*> codegenOuterJoinNullPlaceholder(
      const Analyzer::ColumnVar* col_var,
      const bool fetch_column,
      const CompilationOptions& co);

  llvm::Value* codegenIntArith(const Analyzer::BinOper*,
                               llvm::Value*,
                               llvm::Value*,
                               const CompilationOptions&);

  llvm::Value* codegenFpArith(const Analyzer::BinOper*, llvm::Value*, llvm::Value*);

  llvm::Value* codegenCastTimestampToTime(llvm::Value* ts_lv,
                                          const int dimen,
                                          const bool nullable);

  llvm::Value* codegenCastTimestampToDate(llvm::Value* ts_lv,
                                          const int dimen,
                                          const bool nullable);

  llvm::Value* codegenCastFromString(llvm::Value* operand_lv,
                                     const SQLTypeInfo& operand_ti,
                                     const SQLTypeInfo& ti,
                                     const bool operand_is_const,
                                     const CompilationOptions& co);

  llvm::Value* codegenCastNonStringToString(llvm::Value* operand_lv,
                                            const SQLTypeInfo& operand_ti,
                                            const SQLTypeInfo& ti,
                                            const bool operand_is_const,
                                            const CompilationOptions& co);

  llvm::Value* codegenCastToFp(llvm::Value* operand_lv,
                               const SQLTypeInfo& operand_ti,
                               const SQLTypeInfo& ti);

  llvm::Value* codegenCastFromFp(llvm::Value* operand_lv,
                                 const SQLTypeInfo& operand_ti,
                                 const SQLTypeInfo& ti);

  llvm::Value* codegenAdd(const Analyzer::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&,
                          const CompilationOptions&);

  llvm::Value* codegenSub(const Analyzer::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&,
                          const CompilationOptions&);

  void codegenSkipOverflowCheckForNull(llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       llvm::BasicBlock* no_overflow_bb,
                                       const SQLTypeInfo& ti);

  llvm::Value* codegenMul(const Analyzer::BinOper*,
                          llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&,
                          const CompilationOptions&,
                          bool downscale = true);

  llvm::Value* codegenDiv(llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&,
                          bool upscale = true);

  llvm::Value* codegenDeciDiv(const Analyzer::BinOper*, const CompilationOptions&);

  llvm::Value* codegenMod(llvm::Value*,
                          llvm::Value*,
                          const std::string& null_typename,
                          const std::string& null_check_suffix,
                          const SQLTypeInfo&);

  llvm::Value* codegenCase(const Analyzer::CaseExpr*,
                           llvm::Type* case_llvm_type,
                           const bool is_real_str,
                           const CompilationOptions&);

  llvm::Value* codegenExtractHighPrecisionTimestamps(llvm::Value*,
                                                     const SQLTypeInfo&,
                                                     const ExtractField&);

  llvm::Value* codegenDateTruncHighPrecisionTimestamps(llvm::Value*,
                                                       const SQLTypeInfo&,
                                                       const DatetruncField&);

  llvm::Value* codegenCmpDecimalConst(const SQLOps,
                                      const SQLQualifier,
                                      const Analyzer::Expr*,
                                      const SQLTypeInfo&,
                                      const Analyzer::Expr*,
                                      const CompilationOptions&);

  llvm::Value* codegenOverlaps(const SQLOps,
                               const SQLQualifier,
                               const std::shared_ptr<Analyzer::Expr>,
                               const std::shared_ptr<Analyzer::Expr>,
                               const CompilationOptions&);

  llvm::Value* codegenStrCmp(const SQLOps,
                             const SQLQualifier,
                             const std::shared_ptr<Analyzer::Expr>,
                             const std::shared_ptr<Analyzer::Expr>,
                             const CompilationOptions&);

  llvm::Value* codegenQualifierCmp(const SQLOps,
                                   const SQLQualifier,
                                   std::vector<llvm::Value*>,
                                   const Analyzer::Expr*,
                                   const CompilationOptions&);

  llvm::Value* codegenLogicalShortCircuit(const Analyzer::BinOper*,
                                          const CompilationOptions&);

  llvm::Value* codegenDictLike(const std::shared_ptr<Analyzer::Expr> arg,
                               const Analyzer::Constant* pattern,
                               const bool ilike,
                               const bool is_simple,
                               const char escape_char,
                               const CompilationOptions&);

  llvm::Value* codegenDictStrCmp(const std::shared_ptr<Analyzer::Expr>,
                                 const std::shared_ptr<Analyzer::Expr>,
                                 const SQLOps,
                                 const CompilationOptions& co);

  llvm::Value* codegenDictRegexp(const std::shared_ptr<Analyzer::Expr> arg,
                                 const Analyzer::Constant* pattern,
                                 const char escape_char,
                                 const CompilationOptions&);

  // Returns the IR value which holds true iff at least one match has been found for outer
  // join, null if there's no outer join condition on the given nesting level.
  llvm::Value* foundOuterJoinMatch(const size_t nesting_level) const;

  llvm::Value* resolveGroupedColumnReference(const Analyzer::ColumnVar*);

  llvm::Value* colByteStream(const Analyzer::ColumnVar* col_var,
                             const bool fetch_column,
                             const bool hoist_literals);

  std::shared_ptr<const Analyzer::Expr> hashJoinLhs(const Analyzer::ColumnVar* rhs) const;

  std::shared_ptr<const Analyzer::ColumnVar> hashJoinLhsTuple(
      const Analyzer::ColumnVar* rhs,
      const Analyzer::BinOper* tautological_eq) const;

  bool needCastForHashJoinLhs(const Analyzer::ColumnVar* rhs) const;

  std::unique_ptr<InValuesBitmap> createInValuesBitmap(const Analyzer::InValues*,
                                                       const CompilationOptions&);

  bool checkExpressionRanges(const Analyzer::UOper*, int64_t, int64_t);

  bool checkExpressionRanges(const Analyzer::BinOper*, int64_t, int64_t);

  struct ArgNullcheckBBs {
    llvm::BasicBlock* args_null_bb;
    llvm::BasicBlock* args_notnull_bb;
    llvm::BasicBlock* orig_bb;
  };

  std::tuple<ArgNullcheckBBs, llvm::Value*> beginArgsNullcheck(
      const Analyzer::FunctionOper* function_oper,
      const std::vector<llvm::Value*>& orig_arg_lvs);

  llvm::Value* endArgsNullcheck(const ArgNullcheckBBs&,
                                llvm::Value*,
                                llvm::Value*,
                                const Analyzer::FunctionOper*);

  llvm::Value* codegenFunctionOperNullArg(const Analyzer::FunctionOper*,
                                          const std::vector<llvm::Value*>&);

  llvm::Value* codegenCompression(const SQLTypeInfo& type_info);

  std::pair<llvm::Value*, llvm::Value*> codegenArrayBuff(llvm::Value* chunk,
                                                         llvm::Value* row_pos,
                                                         SQLTypes array_type,
                                                         bool cast_and_extend);

  void codegenBufferArgs(const std::string& udf_func_name,
                         size_t param_num,
                         llvm::Value* buffer_buf,
                         llvm::Value* buffer_size,
                         llvm::Value* buffer_is_null,
                         std::vector<llvm::Value*>& output_args);

  llvm::StructType* createPointStructType(const std::string& udf_func_name,
                                          size_t param_num);

  void codegenGeoPointArgs(const std::string& udf_func_name,
                           size_t param_num,
                           llvm::Value* point_buf,
                           llvm::Value* point_size,
                           llvm::Value* compression,
                           llvm::Value* input_srid,
                           llvm::Value* output_srid,
                           std::vector<llvm::Value*>& output_args);

  llvm::StructType* createMultiPointStructType(const std::string& udf_func_name,
                                               size_t param_num);

  void codegenGeoMultiPointArgs(const std::string& udf_func_name,
                                size_t param_num,
                                llvm::Value* multi_point_buf,
                                llvm::Value* multi_point_size,
                                // TODO: bounds?
                                llvm::Value* compression,
                                llvm::Value* input_srid,
                                llvm::Value* output_srid,
                                std::vector<llvm::Value*>& output_args);

  llvm::StructType* createLineStringStructType(const std::string& udf_func_name,
                                               size_t param_num);

  void codegenGeoLineStringArgs(const std::string& udf_func_name,
                                size_t param_num,
                                llvm::Value* line_string_buf,
                                llvm::Value* line_string_size,
                                llvm::Value* compression,
                                llvm::Value* input_srid,
                                llvm::Value* output_srid,
                                std::vector<llvm::Value*>& output_args);

  llvm::StructType* createMultiLineStringStructType(const std::string& udf_func_name,
                                                    size_t param_num);

  void codegenGeoMultiLineStringArgs(const std::string& udf_func_name,
                                     size_t param_num,
                                     llvm::Value* multi_linestring_coords,
                                     llvm::Value* multi_linestring_size,
                                     llvm::Value* linestring_sizes,
                                     llvm::Value* linestring_sizes_size,
                                     // TODO: bounds?
                                     llvm::Value* compression,
                                     llvm::Value* input_srid,
                                     llvm::Value* output_srid,
                                     std::vector<llvm::Value*>& output_args);

  llvm::StructType* createPolygonStructType(const std::string& udf_func_name,
                                            size_t param_num);

  void codegenGeoPolygonArgs(const std::string& udf_func_name,
                             size_t param_num,
                             llvm::Value* polygon_buf,
                             llvm::Value* polygon_size,
                             llvm::Value* ring_sizes_buf,
                             llvm::Value* num_rings,
                             llvm::Value* compression,
                             llvm::Value* input_srid,
                             llvm::Value* output_srid,
                             std::vector<llvm::Value*>& output_args);

  llvm::StructType* createMultiPolygonStructType(const std::string& udf_func_name,
                                                 size_t param_num);

  void codegenGeoMultiPolygonArgs(const std::string& udf_func_name,
                                  size_t param_num,
                                  llvm::Value* polygon_coords,
                                  llvm::Value* polygon_coords_size,
                                  llvm::Value* ring_sizes_buf,
                                  llvm::Value* ring_sizes,
                                  llvm::Value* polygon_bounds,
                                  llvm::Value* polygon_bounds_sizes,
                                  llvm::Value* compression,
                                  llvm::Value* input_srid,
                                  llvm::Value* output_srid,
                                  std::vector<llvm::Value*>& output_args);

  std::vector<llvm::Value*> codegenFunctionOperCastArgs(
      const Analyzer::FunctionOper*,
      const ExtensionFunction*,
      const std::vector<llvm::Value*>&,
      const std::vector<size_t>&,
      const std::unordered_map<llvm::Value*, llvm::Value*>&,
      const CompilationOptions&);

  llvm::StructType* createStringViewStructType();

  // Return LLVM intrinsic providing fast arithmetic with overflow check
  // for the given binary operation.
  llvm::Function* getArithWithOverflowIntrinsic(const Analyzer::BinOper* bin_oper,
                                                llvm::Type* type);

  // Generate code for the given binary operation with overflow check.
  // Signed integer add, sub and mul operations are supported. Overflow
  // check is performed using LLVM arithmetic intrinsics which are not
  // supported for GPU. Return the IR value which holds operation result.
  llvm::Value* codegenBinOpWithOverflowForCPU(const Analyzer::BinOper* bin_oper,
                                              llvm::Value* lhs_lv,
                                              llvm::Value* rhs_lv,
                                              const std::string& null_check_suffix,
                                              const SQLTypeInfo& ti);

  std::pair<std::vector<llvm::Value*>, std::unique_ptr<CodeGenerator::NullCheckCodegen>>
  codegenStringFetchAndEncode(const Analyzer::StringOper* expr,
                              const CompilationOptions& co,
                              const size_t arg_idx,
                              const bool codegen_nullcheck);

  Executor* executor_;

 protected:
  Executor* executor() const {
    if (!executor_) {
      throw ExecutorRequired();
    }
    return executor_;
  }

  CgenState* cgen_state_;
  PlanState* plan_state_;

  friend class GroupByAndAggregate;

 private:
  // mutex to protect the critical section in initializeNVPTXBackend
  static std::mutex initialize_nvptx_mutex_;
  // mutex to protect the critical sections in
  // generateNativeCPUCode/create_execution_engine
  static std::mutex initialize_cpu_backend_mutex_;
};

// Code generator specialized for scalar expressions which doesn't require an executor.
class ScalarCodeGenerator : public CodeGenerator {
 public:
  // Constructor which takes the runtime module.
  ScalarCodeGenerator(std::unique_ptr<llvm::Module> llvm_module)
      : CodeGenerator(nullptr, nullptr), module_(std::move(llvm_module)) {}

  // Function generated for a given analyzer expression. For GPU, a wrapper which meets
  // the kernel signature constraints (returns void, takes all arguments as pointers) is
  // generated. Also returns the list of column expressions for which compatible input
  // parameters must be passed to the input of the generated function.
  struct CompiledExpression {
    llvm::Function* func;
    llvm::Function* wrapper_func;
    std::vector<std::shared_ptr<Analyzer::ColumnVar>> inputs;
  };

  // Compiles the given scalar expression to IR and the list of columns in the expression,
  // needed to provide inputs to the generated function.
  CompiledExpression compile(const Analyzer::Expr* expr,
                             const bool fetch_columns,
                             const CompilationOptions& co);

  // Generates the native function pointers for each device.
  // NB: this is separated from the compile method to allow building higher level code
  // generators which can inline the IR for evaluating a single expression (for example
  // loops).
  std::vector<void*> generateNativeCode(Executor* executor,
                                        const CompiledExpression& compiled_expression,
                                        const CompilationOptions& co);

  CudaMgr_Namespace::CudaMgr* getCudaMgr() const { return cuda_mgr_.get(); }

  using ColumnMap =
      std::unordered_map<InputColDescriptor, std::shared_ptr<Analyzer::ColumnVar>>;

 private:
  std::vector<llvm::Value*> codegenColumn(const Analyzer::ColumnVar*,
                                          const bool fetch_column,
                                          const CompilationOptions&) override;

  // Collect the columns used by the given analyzer expressions and fills in the column
  // map to be used during code generation.
  ColumnMap prepare(const Analyzer::Expr*);

  std::vector<void*> generateNativeGPUCode(Executor* executor,
                                           llvm::Function* func,
                                           llvm::Function* wrapper_func,
                                           const CompilationOptions& co);

  std::unique_ptr<llvm::Module> module_;
  ExecutionEngineWrapper execution_engine_;
  std::unique_ptr<CgenState> own_cgen_state_;
  std::unique_ptr<PlanState> own_plan_state_;
  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr_;
  std::shared_ptr<GpuCompilationContext> gpu_compilation_context_;
  std::unique_ptr<llvm::TargetMachine> nvptx_target_machine_;
};

/**
 * Makes a shallow copy (just declarations) of the runtime module. Function definitions
 * are cloned only if they're used from the generated code.
 */
std::unique_ptr<llvm::Module> runtime_module_shallow_copy(CgenState* cgen_state);

/**
 *  Loads individual columns from a single, packed pointers buffer (the byte stream arg)
 */
std::vector<llvm::Value*> generate_column_heads_load(const int num_columns,
                                                     llvm::Value* byte_stream_arg,
                                                     llvm::IRBuilder<>& ir_builder,
                                                     llvm::LLVMContext& ctx);
