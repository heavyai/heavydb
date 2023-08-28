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

#include "../Parser/ParserNode.h"
#include "CodeGenerator.h"
#include "Execute.h"
#include "ExternalExecutor.h"
#include "MaxwellCodegenPatch.h"
#include "RelAlgTranslator.h"

#include "QueryEngine/JoinHashTable/RangeJoinHashTable.h"

// Driver methods for the IR generation.

extern bool g_enable_left_join_filter_hoisting;

std::vector<llvm::Value*> CodeGenerator::codegen(const Analyzer::Expr* expr,
                                                 const bool fetch_columns,
                                                 const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (!expr) {
    return {posArg(expr)};
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    return {codegen(bin_oper, co)};
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    return {codegen(u_oper, co)};
  }
  auto geo_col_var = dynamic_cast<const Analyzer::GeoColumnVar*>(expr);
  if (geo_col_var) {
    // inherits from ColumnVar, so it is important we check this first
    return codegenGeoColumnVar(geo_col_var, fetch_columns, co);
  }
  auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (col_var) {
    return codegenColumn(col_var, fetch_columns, co);
  }
  auto constant = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant) {
    const auto& ti = constant->get_type_info();
    if (ti.get_type() == kNULLT) {
      throw std::runtime_error(
          "NULL type literals are not currently supported in this context.");
    }
    if (constant->get_is_null()) {
      const auto i8p_ty =
          llvm::PointerType::get(get_int_type(8, executor_->cgen_state_->context_), 0);
      if (ti.is_string() && ti.get_compression() == kENCODING_NONE) {
        std::vector<llvm::Value*> null_target_lvs;
        llvm::StructType* str_view_ty = createStringViewStructType();
        auto null_str_view_struct_lv = cgen_state_->ir_builder_.CreateAlloca(str_view_ty);
        // we do not need to fill the values of the string_view struct representing null
        // string
        null_target_lvs.push_back(
            cgen_state_->ir_builder_.CreateLoad(str_view_ty, null_str_view_struct_lv));
        null_target_lvs.push_back(llvm::ConstantPointerNull::get(
            llvm::PointerType::get(llvm::IntegerType::get(cgen_state_->context_, 8), 0)));
        null_target_lvs.push_back(cgen_state_->llInt((int32_t)0));
        return null_target_lvs;
      } else if (ti.is_geometry()) {
        std::vector<llvm::Value*> ret_lvs;
        // we classify whether the geo col value is null from the rhs of the left-join
        // qual i.e., table S in the qual: R left join S on R.v = S.v by checking 1)
        // nulled chunk_iter and 2) its col buffer w/ the size 0 for instance, we consider
        // the size of col buffer for point as 16 and for other types we use `array_size`
        // function to determine it, and they all assume valid geos have their buf size >
        // 0 thus, by returning all bufs' ptr as nullptr we can return 0 as the length of
        // the nulled geos' col val, and remaining logic can exploit this to finally
        // return "null" as the col value of the corresponding row in the resultset
        switch (ti.get_type()) {
          // outer join's phi node in the generated code expects the # returned lvs'
          // is the same as `col_ti.get_physical_coord_cols()`
          // now we have necessary logic to deal with these nullptr and can conclude
          // it as nulled geo val, i.e., see ResultSetIteration::991
          case kPOINT:
          case kMULTIPOINT:
          case kLINESTRING:
            return {llvm::ConstantPointerNull::get(i8p_ty)};
          case kMULTILINESTRING:
          case kPOLYGON:
            return {llvm::ConstantPointerNull::get(i8p_ty),
                    llvm::ConstantPointerNull::get(i8p_ty)};
          case kMULTIPOLYGON:
            return {llvm::ConstantPointerNull::get(i8p_ty),
                    llvm::ConstantPointerNull::get(i8p_ty),
                    llvm::ConstantPointerNull::get(i8p_ty)};
          default:
            CHECK(false);
            return {nullptr};
        }
      } else if (ti.is_array()) {
        // similar to above nulled geo case, we can use this `nullptr` to guide
        // `array_buff` and `array_size` functions representing nulled array value
        return {llvm::ConstantPointerNull::get(i8p_ty)};
      }
      return {ti.is_fp()
                  ? static_cast<llvm::Value*>(executor_->cgen_state_->inlineFpNull(ti))
                  : static_cast<llvm::Value*>(executor_->cgen_state_->inlineIntNull(ti))};
    }
    if (ti.get_compression() == kENCODING_DICT) {
      // The dictionary encoding case should be handled by the parent expression
      // (cast, for now), here is too late to know the dictionary id if not already set
      CHECK_NE(ti.getStringDictKey().dict_id, 0);
      return {codegen(constant, ti.get_compression(), ti.getStringDictKey(), co)};
    }
    return {codegen(constant, ti.get_compression(), {}, co)};
  }
  auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (case_expr) {
    return {codegen(case_expr, co)};
  }
  auto extract_expr = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
  if (extract_expr) {
    return {codegen(extract_expr, co)};
  }
  auto dateadd_expr = dynamic_cast<const Analyzer::DateaddExpr*>(expr);
  if (dateadd_expr) {
    return {codegen(dateadd_expr, co)};
  }
  auto datediff_expr = dynamic_cast<const Analyzer::DatediffExpr*>(expr);
  if (datediff_expr) {
    return {codegen(datediff_expr, co)};
  }
  auto datetrunc_expr = dynamic_cast<const Analyzer::DatetruncExpr*>(expr);
  if (datetrunc_expr) {
    return {codegen(datetrunc_expr, co)};
  }
  auto charlength_expr = dynamic_cast<const Analyzer::CharLengthExpr*>(expr);
  if (charlength_expr) {
    return {codegen(charlength_expr, co)};
  }
  auto keyforstring_expr = dynamic_cast<const Analyzer::KeyForStringExpr*>(expr);
  if (keyforstring_expr) {
    return {codegen(keyforstring_expr, co)};
  }
  auto sample_ratio_expr = dynamic_cast<const Analyzer::SampleRatioExpr*>(expr);
  if (sample_ratio_expr) {
    return {codegen(sample_ratio_expr, co)};
  }
  auto string_oper_expr = dynamic_cast<const Analyzer::StringOper*>(expr);
  if (string_oper_expr) {
    return {codegen(string_oper_expr, co)};
  }
  auto cardinality_expr = dynamic_cast<const Analyzer::CardinalityExpr*>(expr);
  if (cardinality_expr) {
    return {codegen(cardinality_expr, co)};
  }
  auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
  if (like_expr) {
    return {codegen(like_expr, co)};
  }
  auto regexp_expr = dynamic_cast<const Analyzer::RegexpExpr*>(expr);
  if (regexp_expr) {
    return {codegen(regexp_expr, co)};
  }
  auto width_bucket_expr = dynamic_cast<const Analyzer::WidthBucketExpr*>(expr);
  if (width_bucket_expr) {
    return {codegen(width_bucket_expr, co)};
  }

  auto ml_predict_expr = dynamic_cast<const Analyzer::MLPredictExpr*>(expr);
  if (ml_predict_expr) {
    return {codegen(ml_predict_expr, co)};
  }

  auto pca_project_expr = dynamic_cast<const Analyzer::PCAProjectExpr*>(expr);
  if (pca_project_expr) {
    return {codegen(pca_project_expr, co)};
  }

  auto likelihood_expr = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (likelihood_expr) {
    return {codegen(likelihood_expr->get_arg(), fetch_columns, co)};
  }
  auto in_expr = dynamic_cast<const Analyzer::InValues*>(expr);
  if (in_expr) {
    return {codegen(in_expr, co)};
  }
  auto in_integer_set_expr = dynamic_cast<const Analyzer::InIntegerSet*>(expr);
  if (in_integer_set_expr) {
    return {codegen(in_integer_set_expr, co)};
  }
  auto function_oper_with_custom_type_handling_expr =
      dynamic_cast<const Analyzer::FunctionOperWithCustomTypeHandling*>(expr);
  if (function_oper_with_custom_type_handling_expr) {
    return {codegenFunctionOperWithCustomTypeHandling(
        function_oper_with_custom_type_handling_expr, co)};
  }
  auto array_oper_expr = dynamic_cast<const Analyzer::ArrayExpr*>(expr);
  if (array_oper_expr) {
    return {codegenArrayExpr(array_oper_expr, co)};
  }
  auto geo_uop = dynamic_cast<const Analyzer::GeoUOper*>(expr);
  if (geo_uop) {
    return {codegenGeoUOper(geo_uop, co)};
  }
  auto geo_binop = dynamic_cast<const Analyzer::GeoBinOper*>(expr);
  if (geo_binop) {
    return {codegenGeoBinOper(geo_binop, co)};
  }
  auto function_oper_expr = dynamic_cast<const Analyzer::FunctionOper*>(expr);
  if (function_oper_expr) {
    return {codegenFunctionOper(function_oper_expr, co)};
  }
  auto geo_expr = dynamic_cast<const Analyzer::GeoExpr*>(expr);
  if (geo_expr) {
    return codegenGeoExpr(geo_expr, co);
  }
  if (dynamic_cast<const Analyzer::OffsetInFragment*>(expr)) {
    return {posArg(nullptr)};
  }
  if (dynamic_cast<const Analyzer::WindowFunction*>(expr)) {
    throw NativeExecutionError("Window expression not supported in this context");
  }
  abort();
}

llvm::Value* CodeGenerator::codegen(const Analyzer::BinOper* bin_oper,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto optype = bin_oper->get_optype();
  if (IS_ARITHMETIC(optype)) {
    return codegenArith(bin_oper, co);
  }
  if (IS_COMPARISON(optype)) {
    return codegenCmp(bin_oper, co);
  }
  if (IS_LOGIC(optype)) {
    return codegenLogical(bin_oper, co);
  }
  if (optype == kARRAY_AT) {
    return codegenArrayAt(bin_oper, co);
  }
  abort();
}

llvm::Value* CodeGenerator::codegen(const Analyzer::UOper* u_oper,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto optype = u_oper->get_optype();
  switch (optype) {
    case kNOT: {
      return codegenLogical(u_oper, co);
    }
    case kCAST: {
      return codegenCast(u_oper, co);
    }
    case kUMINUS: {
      return codegenUMinus(u_oper, co);
    }
    case kISNULL: {
      return codegenIsNull(u_oper, co);
    }
    case kUNNEST:
      return codegenUnnest(u_oper, co);
    default:
      UNREACHABLE();
  }
  return nullptr;
}

llvm::Value* CodeGenerator::codegen(const Analyzer::SampleRatioExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto input_expr = expr->get_arg();
  CHECK(input_expr);

  auto double_lv = codegen(input_expr, true, co);
  CHECK_EQ(size_t(1), double_lv.size());

  std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
  const bool is_nullable = !input_expr->get_type_info().get_notnull();
  if (is_nullable) {
    nullcheck_codegen = std::make_unique<NullCheckCodegen>(cgen_state_,
                                                           executor(),
                                                           double_lv.front(),
                                                           input_expr->get_type_info(),
                                                           "sample_ratio_nullcheck");
  }
  CHECK_EQ(input_expr->get_type_info().get_type(), kDOUBLE);
  std::vector<llvm::Value*> args{double_lv[0], posArg(nullptr)};
  auto ret = cgen_state_->emitCall("sample_ratio", args);
  if (nullcheck_codegen) {
    ret = nullcheck_codegen->finalize(ll_bool(false, cgen_state_->context_), ret);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegen(const Analyzer::WidthBucketExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto target_value_expr = expr->get_target_value();
  auto lower_bound_expr = expr->get_lower_bound();
  auto upper_bound_expr = expr->get_upper_bound();
  auto partition_count_expr = expr->get_partition_count();
  CHECK(target_value_expr);
  CHECK(lower_bound_expr);
  CHECK(upper_bound_expr);
  CHECK(partition_count_expr);

  auto is_constant_expr = [](const Analyzer::Expr* expr) {
    auto target_expr = expr;
    if (auto cast_expr = dynamic_cast<const Analyzer::UOper*>(expr)) {
      if (cast_expr->get_optype() == SQLOps::kCAST) {
        target_expr = cast_expr->get_operand();
      }
    }
    // there are more complex constant expr like 1+2, 1/2*3, and so on
    // but when considering a typical usage of width_bucket function
    // it is sufficient to consider a singleton constant expr
    auto constant_expr = dynamic_cast<const Analyzer::Constant*>(target_expr);
    if (constant_expr) {
      return true;
    }
    return false;
  };
  if (is_constant_expr(lower_bound_expr) && is_constant_expr(upper_bound_expr) &&
      is_constant_expr(partition_count_expr)) {
    expr->set_constant_expr();
  }
  // compute width_bucket's expresn range and check the possibility of avoiding oob check
  auto col_range =
      getExpressionRange(expr,
                         plan_state_->query_infos_,
                         executor_,
                         boost::make_optional(plan_state_->getSimpleQuals()));
  // check whether target_expr is valid
  if (col_range.getType() == ExpressionRangeType::Integer &&
      !expr->can_skip_out_of_bound_check() && col_range.getIntMin() > 0 &&
      col_range.getIntMax() <= expr->get_partition_count_val()) {
    // check whether target_col is not-nullable or has filter expr on it
    if (!col_range.hasNulls()) {
      // Even if the target_expr has its filter expression, target_col_range may exactly
      // the same with the col_range of the target_expr's operand col,
      // i.e., SELECT WIDTH_BUCKET(v1, 1, 10, 10) FROM T WHERE v1 != 1;
      // In that query, col_range of v1 with/without considering the filter expression
      // v1 != 1 have exactly the same col ranges, so we cannot recognize the existence
      // of the filter expression based on them. Also, is (not) null is located in
      // FilterNode, so we cannot trace it in here.
      // todo (yoonmin): relax this to allow skipping oob check more cases
      expr->skip_out_of_bound_check();
    }
  }
  return expr->is_constant_expr() ? codegenConstantWidthBucketExpr(expr, co)
                                  : codegenWidthBucketExpr(expr, co);
}

llvm::Value* CodeGenerator::codegenConstantWidthBucketExpr(
    const Analyzer::WidthBucketExpr* expr,
    const CompilationOptions& co) {
  auto target_value_expr = expr->get_target_value();
  auto lower_bound_expr = expr->get_lower_bound();
  auto upper_bound_expr = expr->get_upper_bound();
  auto partition_count_expr = expr->get_partition_count();

  auto num_partitions = expr->get_partition_count_val();
  if (num_partitions < 1 || num_partitions > INT32_MAX) {
    throw std::runtime_error(
        "PARTITION_COUNT expression of width_bucket function should be in a valid "
        "range: 0 < PARTITION_COUNT <= 2147483647");
  }
  double lower = expr->get_bound_val(lower_bound_expr);
  double upper = expr->get_bound_val(upper_bound_expr);
  if (lower == upper) {
    throw std::runtime_error(
        "LOWER_BOUND and UPPER_BOUND expressions of width_bucket function cannot have "
        "the same constant value");
  }
  if (lower == NULL_DOUBLE || upper == NULL_DOUBLE) {
    throw std::runtime_error(
        "Both LOWER_BOUND and UPPER_BOUND of width_bucket function should be finite "
        "numeric constants.");
  }

  bool const reversed = lower > upper;
  double const scale_factor = num_partitions / (reversed ? lower - upper : upper - lower);
  std::string func_name = reversed ? "width_bucket_reversed" : "width_bucket";

  auto get_double_constant_lvs = [this, &co](double const_val) {
    Datum d;
    d.doubleval = const_val;
    auto double_const_expr =
        makeExpr<Analyzer::Constant>(SQLTypeInfo(kDOUBLE, false), false, d);
    return codegen(double_const_expr.get(), false, co);
  };

  auto target_value_ti = target_value_expr->get_type_info();
  auto target_value_expr_lvs = codegen(target_value_expr, true, co);
  CHECK_EQ(size_t(1), target_value_expr_lvs.size());
  auto lower_expr_lvs = codegen(lower_bound_expr, true, co);
  CHECK_EQ(size_t(1), lower_expr_lvs.size());
  auto scale_factor_lvs = get_double_constant_lvs(scale_factor);
  CHECK_EQ(size_t(1), scale_factor_lvs.size());

  std::vector<llvm::Value*> width_bucket_args{target_value_expr_lvs[0],
                                              lower_expr_lvs[0]};
  if (expr->can_skip_out_of_bound_check()) {
    func_name += "_no_oob_check";
    width_bucket_args.push_back(scale_factor_lvs[0]);
  } else {
    auto upper_expr_lvs = codegen(upper_bound_expr, true, co);
    CHECK_EQ(size_t(1), upper_expr_lvs.size());
    auto partition_count_expr_lvs = codegen(partition_count_expr, true, co);
    CHECK_EQ(size_t(1), partition_count_expr_lvs.size());
    width_bucket_args.push_back(upper_expr_lvs[0]);
    width_bucket_args.push_back(scale_factor_lvs[0]);
    width_bucket_args.push_back(partition_count_expr_lvs[0]);
    if (!target_value_ti.get_notnull()) {
      func_name += "_nullable";
      auto translated_null_value = target_value_ti.is_fp()
                                       ? inline_fp_null_val(target_value_ti)
                                       : inline_int_null_val(target_value_ti);
      auto null_value_lvs = get_double_constant_lvs(translated_null_value);
      CHECK_EQ(size_t(1), null_value_lvs.size());
      width_bucket_args.push_back(null_value_lvs[0]);
    }
  }
  return cgen_state_->emitCall(func_name, width_bucket_args);
}

llvm::Value* CodeGenerator::codegenWidthBucketExpr(const Analyzer::WidthBucketExpr* expr,
                                                   const CompilationOptions& co) {
  auto target_value_expr = expr->get_target_value();
  auto lower_bound_expr = expr->get_lower_bound();
  auto upper_bound_expr = expr->get_upper_bound();
  auto partition_count_expr = expr->get_partition_count();

  std::string func_name = "width_bucket_expr";
  bool nullable_expr = false;
  if (expr->can_skip_out_of_bound_check()) {
    func_name += "_no_oob_check";
  } else if (!target_value_expr->get_type_info().get_notnull()) {
    func_name += "_nullable";
    nullable_expr = true;
  }

  auto target_value_expr_lvs = codegen(target_value_expr, true, co);
  CHECK_EQ(size_t(1), target_value_expr_lvs.size());
  auto lower_bound_expr_lvs = codegen(lower_bound_expr, true, co);
  CHECK_EQ(size_t(1), lower_bound_expr_lvs.size());
  auto upper_bound_expr_lvs = codegen(upper_bound_expr, true, co);
  CHECK_EQ(size_t(1), upper_bound_expr_lvs.size());
  auto partition_count_expr_lvs = codegen(partition_count_expr, true, co);
  CHECK_EQ(size_t(1), partition_count_expr_lvs.size());
  auto target_value_ti = target_value_expr->get_type_info();
  auto null_value_lv = cgen_state_->inlineFpNull(target_value_ti);

  // check partition count : 1 ~ INT32_MAX
  // INT32_MAX will be checked during casting by OVERFLOW checking step
  auto partition_count_ti = partition_count_expr->get_type_info();
  CHECK(partition_count_ti.is_integer());
  auto int32_ti = SQLTypeInfo(kINT, partition_count_ti.get_notnull());
  auto partition_count_expr_lv =
      codegenCastBetweenIntTypes(partition_count_expr_lvs[0],
                                 partition_count_ti,
                                 int32_ti,
                                 partition_count_ti.get_size() < int32_ti.get_size());
  llvm::Value* chosen_min = cgen_state_->llInt(static_cast<int32_t>(0));
  llvm::Value* partition_count_min =
      cgen_state_->ir_builder_.CreateICmpSLE(partition_count_expr_lv, chosen_min);
  llvm::BasicBlock* width_bucket_partition_count_ok_bb =
      llvm::BasicBlock::Create(cgen_state_->context_,
                               "width_bucket_partition_count_ok_bb",
                               cgen_state_->current_func_);
  llvm::BasicBlock* width_bucket_argument_check_fail_bb =
      llvm::BasicBlock::Create(cgen_state_->context_,
                               "width_bucket_argument_check_fail_bb",
                               cgen_state_->current_func_);
  cgen_state_->ir_builder_.CreateCondBr(partition_count_min,
                                        width_bucket_argument_check_fail_bb,
                                        width_bucket_partition_count_ok_bb);
  cgen_state_->ir_builder_.SetInsertPoint(width_bucket_argument_check_fail_bb);
  cgen_state_->ir_builder_.CreateRet(
      cgen_state_->llInt(Executor::ERR_WIDTH_BUCKET_INVALID_ARGUMENT));
  cgen_state_->ir_builder_.SetInsertPoint(width_bucket_partition_count_ok_bb);

  llvm::BasicBlock* width_bucket_bound_check_ok_bb =
      llvm::BasicBlock::Create(cgen_state_->context_,
                               "width_bucket_bound_check_ok_bb",
                               cgen_state_->current_func_);
  llvm::Value* bound_check{nullptr};
  if (lower_bound_expr->get_type_info().get_notnull() &&
      upper_bound_expr->get_type_info().get_notnull()) {
    bound_check = cgen_state_->ir_builder_.CreateFCmpOEQ(
        lower_bound_expr_lvs[0], upper_bound_expr_lvs[0], "bound_check");
  } else {
    std::vector<llvm::Value*> bound_check_args{
        lower_bound_expr_lvs[0],
        upper_bound_expr_lvs[0],
        null_value_lv,
        cgen_state_->llInt(static_cast<int8_t>(1))};
    bound_check = toBool(cgen_state_->emitCall("eq_double_nullable", bound_check_args));
  }
  cgen_state_->ir_builder_.CreateCondBr(
      bound_check, width_bucket_argument_check_fail_bb, width_bucket_bound_check_ok_bb);
  cgen_state_->ir_builder_.SetInsertPoint(width_bucket_bound_check_ok_bb);
  cgen_state_->needs_error_check_ = true;
  auto reversed_expr = toBool(codegenCmp(SQLOps::kGT,
                                         kONE,
                                         lower_bound_expr_lvs,
                                         lower_bound_expr->get_type_info(),
                                         upper_bound_expr,
                                         co));
  auto lower_bound_expr_lv = lower_bound_expr_lvs[0];
  auto upper_bound_expr_lv = upper_bound_expr_lvs[0];
  std::vector<llvm::Value*> width_bucket_args{target_value_expr_lvs[0],
                                              reversed_expr,
                                              lower_bound_expr_lv,
                                              upper_bound_expr_lv,
                                              partition_count_expr_lv};
  if (nullable_expr) {
    width_bucket_args.push_back(null_value_lv);
  }
  return cgen_state_->emitCall(func_name, width_bucket_args);
}

namespace {

void add_qualifier_to_execution_unit(RelAlgExecutionUnit& ra_exe_unit,
                                     const std::shared_ptr<Analyzer::Expr>& qual) {
  const auto qual_cf = qual_to_conjunctive_form(qual);
  ra_exe_unit.simple_quals.insert(ra_exe_unit.simple_quals.end(),
                                  qual_cf.simple_quals.begin(),
                                  qual_cf.simple_quals.end());
  ra_exe_unit.quals.insert(
      ra_exe_unit.quals.end(), qual_cf.quals.begin(), qual_cf.quals.end());
}

void check_if_loop_join_is_allowed(RelAlgExecutionUnit& ra_exe_unit,
                                   const ExecutionOptions& eo,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const size_t level_idx,
                                   const std::string& fail_reason) {
  if (ra_exe_unit.input_descs.size() < 2) {
    // We only support loop join at the end of folded joins
    // where ra_exe_unit.input_descs.size() > 2 for now.
    throw std::runtime_error("Hash join failed, reason(s): " + fail_reason +
                             " | Incorrect # tables for executing loop join");
  }
  const auto loop_join_size = get_loop_join_size(query_infos, ra_exe_unit);
  const bool has_loop_size_hint =
      ra_exe_unit.query_hint.isHintRegistered(QueryHint::kLoopJoinInnerTableMaxNumRows);
  const size_t loop_join_size_threshold =
      has_loop_size_hint ? ra_exe_unit.query_hint.loop_join_inner_table_max_num_rows
                         : g_trivial_loop_join_threshold;
  if (eo.allow_loop_joins) {
    if (has_loop_size_hint && loop_join_size_threshold < loop_join_size) {
      throw std::runtime_error(
          "Hash join failed, reason(s): " + fail_reason +
          " | Cannot fall back to loop join for non-trivial inner table size");
    }
    return;
  }
  if (level_idx + 1 != ra_exe_unit.join_quals.size()) {
    throw std::runtime_error(
        "Hash join failed, reason(s): " + fail_reason +
        " | Cannot fall back to loop join for intermediate join quals");
  }
  if (loop_join_size_threshold < loop_join_size) {
    throw std::runtime_error(
        "Hash join failed, reason(s): " + fail_reason +
        " | Cannot fall back to loop join for non-trivial inner table size");
  }
  if (ra_exe_unit.query_hint.isHintRegistered(kDisableLoopJoin)) {
    throw std::runtime_error("Hash join failed, reason(s): " + fail_reason +
                             " | Loop join is disabled by user");
  }
}

void check_valid_join_qual(std::shared_ptr<Analyzer::BinOper>& bin_oper) {
  // check whether a join qual is valid before entering the hashtable build and codegen

  auto lhs_cv = dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_left_operand());
  auto rhs_cv = dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_right_operand());
  if (lhs_cv && rhs_cv && !bin_oper->is_bbox_intersect_oper()) {
    auto lhs_type = lhs_cv->get_type_info().get_type();
    auto rhs_type = rhs_cv->get_type_info().get_type();
    // check #1. avoid a join btw full array columns
    if (lhs_type == SQLTypes::kARRAY && rhs_type == SQLTypes::kARRAY) {
      throw std::runtime_error(
          "Join operation between full array columns (i.e., R.arr = S.arr) instead of "
          "indexed array columns (i.e., R.arr[1] = S.arr[2]) is not supported yet.");
    }
  }
}

}  // namespace

std::vector<JoinLoop> Executor::buildJoinLoops(
    RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const std::vector<InputTableInfo>& query_infos,
    ColumnCacheMap& column_cache) {
  INJECT_TIMER(buildJoinLoops);
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  std::vector<JoinLoop> join_loops;
  for (size_t level_idx = 0, current_hash_table_idx = 0;
       level_idx < ra_exe_unit.join_quals.size();
       ++level_idx) {
    const auto& current_level_join_conditions = ra_exe_unit.join_quals[level_idx];
    std::vector<std::string> fail_reasons;
    const auto current_level_hash_table =
        buildCurrentLevelHashTable(current_level_join_conditions,
                                   level_idx,
                                   ra_exe_unit,
                                   co,
                                   query_infos,
                                   column_cache,
                                   fail_reasons);
    const auto found_outer_join_matches_cb =
        [this, level_idx](llvm::Value* found_outer_join_matches) {
          CHECK_LT(level_idx, cgen_state_->outer_join_match_found_per_level_.size());
          CHECK(!cgen_state_->outer_join_match_found_per_level_[level_idx]);
          cgen_state_->outer_join_match_found_per_level_[level_idx] =
              found_outer_join_matches;
        };
    const auto is_deleted_cb = buildIsDeletedCb(ra_exe_unit, level_idx, co);
    auto rem_left_join_quals_it =
        plan_state_->left_join_non_hashtable_quals_.find(level_idx);
    bool has_remaining_left_join_quals =
        rem_left_join_quals_it != plan_state_->left_join_non_hashtable_quals_.end() &&
        !rem_left_join_quals_it->second.empty();
    const auto outer_join_condition_remaining_quals_cb =
        [this, level_idx, &co](const std::vector<llvm::Value*>& prev_iters) {
          // when we have multiple quals for the left join in the current join level
          // we first try to build a hashtable by using one of the possible qual,
          // and deal with remaining quals as extra join conditions
          FetchCacheAnchor anchor(cgen_state_.get());
          addJoinLoopIterator(prev_iters, level_idx + 1);
          llvm::Value* left_join_cond = cgen_state_->llBool(true);
          CodeGenerator code_generator(this);
          auto it = plan_state_->left_join_non_hashtable_quals_.find(level_idx);
          if (it != plan_state_->left_join_non_hashtable_quals_.end()) {
            for (auto expr : it->second) {
              left_join_cond = cgen_state_->ir_builder_.CreateAnd(
                  left_join_cond,
                  code_generator.toBool(
                      code_generator.codegen(expr.get(), true, co).front()));
            }
          }
          return left_join_cond;
        };
    if (current_level_hash_table) {
      const auto hoisted_filters_cb = buildHoistLeftHandSideFiltersCb(
          ra_exe_unit, level_idx, current_level_hash_table->getInnerTableId(), co);
      if (current_level_hash_table->getHashType() == HashType::OneToOne) {
        join_loops.emplace_back(
            /*kind=*/JoinLoopKind::Singleton,
            /*type=*/current_level_join_conditions.type,
            /*iteration_domain_codegen=*/
            [this, current_hash_table_idx, level_idx, current_level_hash_table, &co](
                const std::vector<llvm::Value*>& prev_iters) {
              addJoinLoopIterator(prev_iters, level_idx);
              JoinLoopDomain domain{{0}};
              domain.slot_lookup_result =
                  current_level_hash_table->codegenSlot(co, current_hash_table_idx);
              return domain;
            },
            /*outer_condition_match=*/
            current_level_join_conditions.type == JoinType::LEFT &&
                    has_remaining_left_join_quals
                ? std::function<llvm::Value*(const std::vector<llvm::Value*>&)>(
                      outer_join_condition_remaining_quals_cb)
                : nullptr,
            /*found_outer_matches=*/current_level_join_conditions.type == JoinType::LEFT
                ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
                : nullptr,
            /*hoisted_filters=*/hoisted_filters_cb,
            /*is_deleted=*/is_deleted_cb,
            /*nested_loop_join=*/false);
      } else if (auto range_join_table =
                     dynamic_cast<RangeJoinHashTable*>(current_level_hash_table.get())) {
        join_loops.emplace_back(
            /* kind= */ JoinLoopKind::MultiSet,
            /* type= */ current_level_join_conditions.type,
            /* iteration_domain_codegen= */
            [this,
             range_join_table,
             current_hash_table_idx,
             level_idx,
             current_level_hash_table,
             &co](const std::vector<llvm::Value*>& prev_iters) {
              addJoinLoopIterator(prev_iters, level_idx);
              JoinLoopDomain domain{{0}};
              CHECK(!prev_iters.empty());
              const auto matching_set = range_join_table->codegenMatchingSetWithOffset(
                  co, current_hash_table_idx, prev_iters.back());
              domain.values_buffer = matching_set.elements;
              domain.element_count = matching_set.count;
              return domain;
            },
            /* outer_condition_match= */
            current_level_join_conditions.type == JoinType::LEFT
                ? std::function<llvm::Value*(const std::vector<llvm::Value*>&)>(
                      outer_join_condition_remaining_quals_cb)
                : nullptr,
            /* found_outer_matches= */
            current_level_join_conditions.type == JoinType::LEFT
                ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
                : nullptr,
            /* hoisted_filters= */ nullptr,  // <<! TODO
            /* is_deleted= */ is_deleted_cb,
            /*nested_loop_join=*/false);
      } else {
        join_loops.emplace_back(
            /*kind=*/JoinLoopKind::Set,
            /*type=*/current_level_join_conditions.type,
            /*iteration_domain_codegen=*/
            [this, current_hash_table_idx, level_idx, current_level_hash_table, &co](
                const std::vector<llvm::Value*>& prev_iters) {
              addJoinLoopIterator(prev_iters, level_idx);
              JoinLoopDomain domain{{0}};
              const auto matching_set = current_level_hash_table->codegenMatchingSet(
                  co, current_hash_table_idx);
              domain.values_buffer = matching_set.elements;
              domain.element_count = matching_set.count;
              return domain;
            },
            /*outer_condition_match=*/
            current_level_join_conditions.type == JoinType::LEFT
                ? std::function<llvm::Value*(const std::vector<llvm::Value*>&)>(
                      outer_join_condition_remaining_quals_cb)
                : nullptr,
            /*found_outer_matches=*/current_level_join_conditions.type == JoinType::LEFT
                ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
                : nullptr,
            /*hoisted_filters=*/hoisted_filters_cb,
            /*is_deleted=*/is_deleted_cb,
            /*nested_loop_join=*/false);
      }
      ++current_hash_table_idx;
    } else {
      const auto fail_reasons_str = current_level_join_conditions.quals.empty()
                                        ? "No equijoin expression found"
                                        : boost::algorithm::join(fail_reasons, " | ");
      check_if_loop_join_is_allowed(
          ra_exe_unit, eo, query_infos, level_idx, fail_reasons_str);
      // Callback provided to the `JoinLoop` framework to evaluate the (outer) join
      // condition.
      VLOG(1) << "Unable to build hash table, falling back to loop join: "
              << fail_reasons_str;
      const auto outer_join_condition_cb =
          [this, level_idx, &co, &current_level_join_conditions](
              const std::vector<llvm::Value*>& prev_iters) {
            // The values generated for the match path don't dominate all uses
            // since on the non-match path nulls are generated. Reset the cache
            // once the condition is generated to avoid incorrect reuse.
            FetchCacheAnchor anchor(cgen_state_.get());
            addJoinLoopIterator(prev_iters, level_idx + 1);
            llvm::Value* left_join_cond = cgen_state_->llBool(true);
            CodeGenerator code_generator(this);
            for (auto expr : current_level_join_conditions.quals) {
              left_join_cond = cgen_state_->ir_builder_.CreateAnd(
                  left_join_cond,
                  code_generator.toBool(
                      code_generator.codegen(expr.get(), true, co).front()));
            }
            return left_join_cond;
          };
      join_loops.emplace_back(
          /*kind=*/JoinLoopKind::UpperBound,
          /*type=*/current_level_join_conditions.type,
          /*iteration_domain_codegen=*/
          [this, level_idx](const std::vector<llvm::Value*>& prev_iters) {
            addJoinLoopIterator(prev_iters, level_idx);
            JoinLoopDomain domain{{0}};
            auto* arg = get_arg_by_name(cgen_state_->row_func_, "num_rows_per_scan");
            const auto rows_per_scan_ptr = cgen_state_->ir_builder_.CreateGEP(
                arg->getType()->getScalarType()->getPointerElementType(),
                arg,
                cgen_state_->llInt(int32_t(level_idx + 1)));
            domain.upper_bound = cgen_state_->ir_builder_.CreateLoad(
                rows_per_scan_ptr->getType()->getPointerElementType(),
                rows_per_scan_ptr,
                "num_rows_per_scan");
            return domain;
          },
          /*outer_condition_match=*/
          current_level_join_conditions.type == JoinType::LEFT
              ? std::function<llvm::Value*(const std::vector<llvm::Value*>&)>(
                    outer_join_condition_cb)
              : nullptr,
          /*found_outer_matches=*/
          current_level_join_conditions.type == JoinType::LEFT
              ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
              : nullptr,
          /*hoisted_filters=*/nullptr,
          /*is_deleted=*/is_deleted_cb,
          /*nested_loop_join=*/true);
    }
  }
  return join_loops;
}

namespace {

class ExprTableIdVisitor : public ScalarExprVisitor<std::set<shared::TableKey>> {
 protected:
  std::set<shared::TableKey> visitColumnVar(
      const Analyzer::ColumnVar* col_expr) const final {
    return {col_expr->getTableKey()};
  }

  std::set<shared::TableKey> visitFunctionOper(
      const Analyzer::FunctionOper* func_expr) const final {
    std::set<shared::TableKey> ret;
    for (size_t i = 0; i < func_expr->getArity(); i++) {
      ret = aggregateResult(ret, visit(func_expr->getArg(i)));
    }
    return ret;
  }

  std::set<shared::TableKey> visitBinOper(const Analyzer::BinOper* bin_oper) const final {
    std::set<shared::TableKey> ret;
    ret = aggregateResult(ret, visit(bin_oper->get_left_operand()));
    return aggregateResult(ret, visit(bin_oper->get_right_operand()));
  }

  std::set<shared::TableKey> visitUOper(const Analyzer::UOper* u_oper) const final {
    return visit(u_oper->get_operand());
  }

  std::set<shared::TableKey> aggregateResult(
      const std::set<shared::TableKey>& aggregate,
      const std::set<shared::TableKey>& next_result) const final {
    auto ret = aggregate;  // copy
    for (const auto& el : next_result) {
      ret.insert(el);
    }
    return ret;
  }
};

}  // namespace

JoinLoop::HoistedFiltersCallback Executor::buildHoistLeftHandSideFiltersCb(
    const RelAlgExecutionUnit& ra_exe_unit,
    const size_t level_idx,
    const shared::TableKey& inner_table_id,
    const CompilationOptions& co) {
  if (!g_enable_left_join_filter_hoisting) {
    return nullptr;
  }

  const auto& current_level_join_conditions = ra_exe_unit.join_quals[level_idx];
  if (level_idx == 0 && current_level_join_conditions.type == JoinType::LEFT) {
    const auto& condition = current_level_join_conditions.quals.front();
    const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(condition.get());
    CHECK(bin_oper) << condition->toString();
    const auto rhs =
        dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_right_operand());
    const auto lhs =
        dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_left_operand());
    if (lhs && rhs && lhs->getTableKey() != rhs->getTableKey()) {
      const Analyzer::ColumnVar* selected_lhs{nullptr};
      // grab the left hand side column -- this is somewhat similar to normalize column
      // pair, and a better solution may be to hoist that function out of the join
      // framework and normalize columns at the top of build join loops
      if (lhs->getTableKey() == inner_table_id) {
        selected_lhs = rhs;
      } else if (rhs->getTableKey() == inner_table_id) {
        selected_lhs = lhs;
      }
      if (selected_lhs) {
        std::list<std::shared_ptr<Analyzer::Expr>> hoisted_quals;
        // get all LHS-only filters
        auto should_hoist_qual = [&hoisted_quals](const auto& qual,
                                                  const shared::TableKey& table_key) {
          CHECK(qual);

          ExprTableIdVisitor visitor;
          const auto table_keys = visitor.visit(qual.get());
          if (table_keys.size() == 1 && table_keys.find(table_key) != table_keys.end()) {
            hoisted_quals.push_back(qual);
          }
        };
        for (const auto& qual : ra_exe_unit.simple_quals) {
          should_hoist_qual(qual, selected_lhs->getTableKey());
        }
        for (const auto& qual : ra_exe_unit.quals) {
          should_hoist_qual(qual, selected_lhs->getTableKey());
        }

        // build the filters callback and return it
        if (!hoisted_quals.empty()) {
          return [this, hoisted_quals, co](llvm::BasicBlock* true_bb,
                                           llvm::BasicBlock* exit_bb,
                                           const std::string& loop_name,
                                           llvm::Function* parent_func,
                                           CgenState* cgen_state) -> llvm::BasicBlock* {
            // make sure we have quals to hoist
            bool has_quals_to_hoist = false;
            for (const auto& qual : hoisted_quals) {
              // check to see if the filter was previously hoisted. if all filters were
              // previously hoisted, this callback becomes a noop
              if (plan_state_->hoisted_filters_.count(qual) == 0) {
                has_quals_to_hoist = true;
                break;
              }
            }

            if (!has_quals_to_hoist) {
              return nullptr;
            }

            AUTOMATIC_IR_METADATA(cgen_state);

            llvm::IRBuilder<>& builder = cgen_state->ir_builder_;
            auto& context = builder.getContext();

            const auto filter_bb =
                llvm::BasicBlock::Create(context,
                                         "hoisted_left_join_filters_" + loop_name,
                                         parent_func,
                                         /*insert_before=*/true_bb);
            builder.SetInsertPoint(filter_bb);

            llvm::Value* filter_lv = cgen_state_->llBool(true);
            CodeGenerator code_generator(this);
            CHECK(plan_state_);
            for (const auto& qual : hoisted_quals) {
              if (plan_state_->hoisted_filters_.insert(qual).second) {
                // qual was inserted into the hoisted filters map, which means we have not
                // seen this qual before. Generate filter.
                VLOG(1) << "Generating code for hoisted left hand side qualifier "
                        << qual->toString();
                auto cond = code_generator.toBool(
                    code_generator.codegen(qual.get(), true, co).front());
                filter_lv = builder.CreateAnd(filter_lv, cond);
              }
            }
            CHECK(filter_lv->getType()->isIntegerTy(1));

            builder.CreateCondBr(filter_lv, true_bb, exit_bb);
            return filter_bb;
          };
        }
      }
    }
  }
  return nullptr;
}

std::function<llvm::Value*(const std::vector<llvm::Value*>&, llvm::Value*)>
Executor::buildIsDeletedCb(const RelAlgExecutionUnit& ra_exe_unit,
                           const size_t level_idx,
                           const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  if (!co.filter_on_deleted_column) {
    return nullptr;
  }
  CHECK_LT(level_idx + 1, ra_exe_unit.input_descs.size());
  const auto input_desc = ra_exe_unit.input_descs[level_idx + 1];
  if (input_desc.getSourceType() != InputSourceType::TABLE) {
    return nullptr;
  }

  const auto deleted_cd = plan_state_->getDeletedColForTable(input_desc.getTableKey());
  if (!deleted_cd) {
    return nullptr;
  }
  CHECK(deleted_cd->columnType.is_boolean());
  const auto deleted_expr = makeExpr<Analyzer::ColumnVar>(
      deleted_cd->columnType,
      shared::ColumnKey{input_desc.getTableKey(), deleted_cd->columnId},
      input_desc.getNestLevel());
  return [this, deleted_expr, level_idx, &co](const std::vector<llvm::Value*>& prev_iters,
                                              llvm::Value* have_more_inner_rows) {
    const auto matching_row_index = addJoinLoopIterator(prev_iters, level_idx + 1);
    // Avoid fetching the deleted column from a position which is not valid.
    // An invalid position can be returned by a one to one hash lookup (negative)
    // or at the end of iteration over a set of matching values.
    llvm::Value* is_valid_it{nullptr};
    if (have_more_inner_rows) {
      is_valid_it = have_more_inner_rows;
    } else {
      is_valid_it = cgen_state_->ir_builder_.CreateICmp(
          llvm::ICmpInst::ICMP_SGE, matching_row_index, cgen_state_->llInt<int64_t>(0));
    }
    const auto it_valid_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "it_valid", cgen_state_->current_func_);
    const auto it_not_valid_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "it_not_valid", cgen_state_->current_func_);
    cgen_state_->ir_builder_.CreateCondBr(is_valid_it, it_valid_bb, it_not_valid_bb);
    const auto row_is_deleted_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "row_is_deleted", cgen_state_->current_func_);
    cgen_state_->ir_builder_.SetInsertPoint(it_valid_bb);
    CodeGenerator code_generator(this);
    const auto row_is_deleted = code_generator.toBool(
        code_generator.codegen(deleted_expr.get(), true, co).front());
    cgen_state_->ir_builder_.CreateBr(row_is_deleted_bb);
    cgen_state_->ir_builder_.SetInsertPoint(it_not_valid_bb);
    const auto row_is_deleted_default = cgen_state_->llBool(false);
    cgen_state_->ir_builder_.CreateBr(row_is_deleted_bb);
    cgen_state_->ir_builder_.SetInsertPoint(row_is_deleted_bb);
    auto row_is_deleted_or_default =
        cgen_state_->ir_builder_.CreatePHI(row_is_deleted->getType(), 2);
    row_is_deleted_or_default->addIncoming(row_is_deleted, it_valid_bb);
    row_is_deleted_or_default->addIncoming(row_is_deleted_default, it_not_valid_bb);
    return row_is_deleted_or_default;
  };
}

std::shared_ptr<HashJoin> Executor::buildCurrentLevelHashTable(
    const JoinCondition& current_level_join_conditions,
    size_t level_idx,
    RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co,
    const std::vector<InputTableInfo>& query_infos,
    ColumnCacheMap& column_cache,
    std::vector<std::string>& fail_reasons) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  std::shared_ptr<HashJoin> current_level_hash_table;
  auto handleNonHashtableQual = [&ra_exe_unit, &level_idx, this](
                                    JoinType join_type,
                                    std::shared_ptr<Analyzer::Expr> qual) {
    if (join_type == JoinType::LEFT) {
      plan_state_->addNonHashtableQualForLeftJoin(level_idx, qual);
    } else {
      add_qualifier_to_execution_unit(ra_exe_unit, qual);
    }
  };
  for (const auto& join_qual : current_level_join_conditions.quals) {
    auto qual_bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(join_qual);
    if (current_level_hash_table || !qual_bin_oper ||
        !IS_EQUIVALENCE(qual_bin_oper->get_optype())) {
      handleNonHashtableQual(current_level_join_conditions.type, join_qual);
      if (!current_level_hash_table) {
        fail_reasons.emplace_back("No equijoin expression found");
      }
      continue;
    }
    check_valid_join_qual(qual_bin_oper);
    JoinHashTableOrError hash_table_or_error;
    if (!current_level_hash_table) {
      hash_table_or_error = buildHashTableForQualifier(
          qual_bin_oper,
          query_infos,
          co.device_type == ExecutorDeviceType::GPU ? MemoryLevel::GPU_LEVEL
                                                    : MemoryLevel::CPU_LEVEL,
          current_level_join_conditions.type,
          HashType::OneToOne,
          column_cache,
          ra_exe_unit.hash_table_build_plan_dag,
          ra_exe_unit.query_hint,
          ra_exe_unit.table_id_to_node_map);
      current_level_hash_table = hash_table_or_error.hash_table;
    }
    if (hash_table_or_error.hash_table) {
      plan_state_->join_info_.join_hash_tables_.push_back(hash_table_or_error.hash_table);
      plan_state_->join_info_.equi_join_tautologies_.push_back(qual_bin_oper);
    } else {
      fail_reasons.push_back(hash_table_or_error.fail_reason);
      if (!current_level_hash_table) {
        VLOG(2) << "Building a hashtable based on a qual " << qual_bin_oper->toString()
                << " fails: " << hash_table_or_error.fail_reason;
      }
      handleNonHashtableQual(current_level_join_conditions.type, qual_bin_oper);
    }
  }
  return current_level_hash_table;
}

void Executor::redeclareFilterFunction() {
  if (!cgen_state_->filter_func_) {
    return;
  }

  // Loop over all the instructions used in the filter func.
  // The filter func instructions were generated as if for row func.
  // Remap any values used by those instructions to filter func args
  // and remember to forward them through the call in the row func.
  for (auto bb_it = cgen_state_->filter_func_->begin();
       bb_it != cgen_state_->filter_func_->end();
       ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      size_t i = 0;
      for (auto op_it = instr_it->value_op_begin(); op_it != instr_it->value_op_end();
           ++op_it, ++i) {
        llvm::Value* v = *op_it;

        // The last LLVM operand on a call instruction is the function to be called. Never
        // remap it.
        if (llvm::dyn_cast<const llvm::CallInst>(instr_it) &&
            op_it == instr_it->value_op_end() - 1) {
          continue;
        }

        CHECK(v);
        if (auto* instr = llvm::dyn_cast<llvm::Instruction>(v);
            instr && instr->getParent() &&
            instr->getParent()->getParent() == cgen_state_->row_func_) {
          // Remember that this filter func arg is needed.
          cgen_state_->filter_func_args_[v] = nullptr;
        } else if (auto* argum = llvm::dyn_cast<llvm::Argument>(v);
                   argum && argum->getParent() == cgen_state_->row_func_) {
          // Remember that this filter func arg is needed.
          cgen_state_->filter_func_args_[v] = nullptr;
        }
      }
    }
  }

  // Create filter_func2 with parameters only for those row func values that are known to
  // be used in the filter func code.
  std::vector<llvm::Type*> filter_func_arg_types;
  filter_func_arg_types.reserve(cgen_state_->filter_func_args_.v_.size());
  for (auto& arg : cgen_state_->filter_func_args_.v_) {
    filter_func_arg_types.push_back(arg->getType());
  }
  auto ft = llvm::FunctionType::get(
      get_int_type(32, cgen_state_->context_), filter_func_arg_types, false);
  cgen_state_->filter_func_->setName("old_filter_func");
  auto filter_func2 = llvm::Function::Create(ft,
                                             llvm::Function::ExternalLinkage,
                                             "filter_func",
                                             cgen_state_->filter_func_->getParent());
  CHECK_EQ(filter_func2->arg_size(), cgen_state_->filter_func_args_.v_.size());
  auto arg_it = cgen_state_->filter_func_args_.begin();
  size_t i = 0;
  for (llvm::Function::arg_iterator I = filter_func2->arg_begin(),
                                    E = filter_func2->arg_end();
       I != E;
       ++I, ++arg_it) {
    arg_it->second = &*I;
    if (arg_it->first->hasName()) {
      I->setName(arg_it->first->getName());
    } else {
      I->setName("extra" + std::to_string(i++));
    }
  }

  // copy the filter_func function body over
  // see
  // https://stackoverflow.com/questions/12864106/move-function-body-avoiding-full-cloning/18751365
  filter_func2->getBasicBlockList().splice(
      filter_func2->begin(), cgen_state_->filter_func_->getBasicBlockList());

  if (cgen_state_->current_func_ == cgen_state_->filter_func_) {
    cgen_state_->current_func_ = filter_func2;
  }
  cgen_state_->filter_func_ = filter_func2;

  // loop over all the operands in the filter func
  for (auto bb_it = cgen_state_->filter_func_->begin();
       bb_it != cgen_state_->filter_func_->end();
       ++bb_it) {
    for (auto instr_it = bb_it->begin(); instr_it != bb_it->end(); ++instr_it) {
      size_t i = 0;
      for (auto op_it = instr_it->op_begin(); op_it != instr_it->op_end(); ++op_it, ++i) {
        llvm::Value* v = op_it->get();
        if (auto arg_it = cgen_state_->filter_func_args_.find(v);
            arg_it != cgen_state_->filter_func_args_.end()) {
          // replace row func value with a filter func arg
          llvm::Use* use = &*op_it;
          use->set(arg_it->second);
        }
      }
    }
  }
}

llvm::Value* Executor::addJoinLoopIterator(const std::vector<llvm::Value*>& prev_iters,
                                           const size_t level_idx) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  // Iterators are added for loop-outer joins when the head of the loop is generated,
  // then once again when the body if generated. Allow this instead of special handling
  // of call sites.
  const auto it = cgen_state_->scan_idx_to_hash_pos_.find(level_idx);
  if (it != cgen_state_->scan_idx_to_hash_pos_.end()) {
    return it->second;
  }
  CHECK(!prev_iters.empty());
  llvm::Value* matching_row_index = prev_iters.back();
  const auto it_ok =
      cgen_state_->scan_idx_to_hash_pos_.emplace(level_idx, matching_row_index);
  CHECK(it_ok.second);
  return matching_row_index;
}

void Executor::codegenJoinLoops(const std::vector<JoinLoop>& join_loops,
                                const RelAlgExecutionUnit& ra_exe_unit,
                                GroupByAndAggregate& group_by_and_aggregate,
                                llvm::Function* query_func,
                                llvm::BasicBlock* entry_bb,
                                QueryMemoryDescriptor& query_mem_desc,
                                const CompilationOptions& co,
                                const ExecutionOptions& eo) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto exit_bb =
      llvm::BasicBlock::Create(cgen_state_->context_, "exit", cgen_state_->current_func_);
  cgen_state_->ir_builder_.SetInsertPoint(exit_bb);
  cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt<int32_t>(0));
  cgen_state_->ir_builder_.SetInsertPoint(entry_bb);
  CodeGenerator code_generator(this);

  llvm::BasicBlock* loops_entry_bb{nullptr};
  auto has_range_join =
      std::any_of(join_loops.begin(), join_loops.end(), [](const auto& join_loop) {
        return join_loop.kind() == JoinLoopKind::MultiSet;
      });
  if (has_range_join) {
    CHECK_EQ(join_loops.size(), size_t(1));
    const auto element_count =
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), 9);

    auto compute_packed_offset = [](const int32_t x, const int32_t y) -> uint64_t {
      const uint64_t y_shifted = static_cast<uint64_t>(y) << 32;
      return y_shifted | static_cast<uint32_t>(x);
    };

    const auto values_arr = std::vector<llvm::Constant*>{
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), 0),
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                               compute_packed_offset(0, 1)),
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                               compute_packed_offset(0, -1)),
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                               compute_packed_offset(1, 0)),
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                               compute_packed_offset(1, 1)),
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                               compute_packed_offset(1, -1)),
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                               compute_packed_offset(-1, 0)),
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                               compute_packed_offset(-1, 1)),
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_),
                               compute_packed_offset(-1, -1))};

    const auto constant_values_array = llvm::ConstantArray::get(
        get_int_array_type(64, 9, cgen_state_->context_), values_arr);
    CHECK(cgen_state_->module_);
    const auto values =
        new llvm::GlobalVariable(*cgen_state_->module_,
                                 get_int_array_type(64, 9, cgen_state_->context_),
                                 true,
                                 llvm::GlobalValue::LinkageTypes::InternalLinkage,
                                 constant_values_array);
    JoinLoop join_loop(
        JoinLoopKind::Set,
        JoinType::INNER,
        [element_count, values](const std::vector<llvm::Value*>& v) {
          JoinLoopDomain domain{{0}};
          domain.element_count = element_count;
          domain.values_buffer = values;
          return domain;
        },
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        "range_key_loop");

    loops_entry_bb = JoinLoop::codegen(
        {join_loop},
        [this,
         query_func,
         &query_mem_desc,
         &co,
         &eo,
         &group_by_and_aggregate,
         &join_loops,
         &ra_exe_unit](const std::vector<llvm::Value*>& prev_iters) {
          auto& builder = cgen_state_->ir_builder_;

          auto body_exit_bb =
              llvm::BasicBlock::Create(cgen_state_->context_,
                                       "range_key_inner_body_exit",
                                       builder.GetInsertBlock()->getParent());

          auto range_key_body_bb =
              llvm::BasicBlock::Create(cgen_state_->context_,
                                       "range_key_loop_body",
                                       builder.GetInsertBlock()->getParent());
          builder.SetInsertPoint(range_key_body_bb);

          const auto body_loops_entry_bb = JoinLoop::codegen(
              join_loops,
              [this,
               query_func,
               &query_mem_desc,
               &co,
               &eo,
               &group_by_and_aggregate,
               &join_loops,
               &ra_exe_unit](const std::vector<llvm::Value*>& prev_iters) {
                addJoinLoopIterator(prev_iters, join_loops.size());
                auto& builder = cgen_state_->ir_builder_;
                const auto loop_body_bb =
                    llvm::BasicBlock::Create(builder.getContext(),
                                             "loop_body",
                                             builder.GetInsertBlock()->getParent());
                builder.SetInsertPoint(loop_body_bb);
                const bool can_return_error =
                    compileBody(ra_exe_unit, group_by_and_aggregate, query_mem_desc, co);
                if (can_return_error || cgen_state_->needs_error_check_ ||
                    eo.with_dynamic_watchdog || eo.allow_runtime_query_interrupt) {
                  createErrorCheckControlFlow(query_func,
                                              eo.with_dynamic_watchdog,
                                              eo.allow_runtime_query_interrupt,
                                              join_loops,
                                              co.device_type,
                                              group_by_and_aggregate.query_infos_);
                }
                return loop_body_bb;
              },
              prev_iters.back(),
              body_exit_bb,
              cgen_state_.get());

          builder.SetInsertPoint(range_key_body_bb);
          cgen_state_->ir_builder_.CreateBr(body_loops_entry_bb);

          builder.SetInsertPoint(body_exit_bb);
          return range_key_body_bb;
        },
        code_generator.posArg(nullptr),
        exit_bb,
        cgen_state_.get());
  } else {
    loops_entry_bb = JoinLoop::codegen(
        join_loops,
        /*body_codegen=*/
        [this,
         query_func,
         &query_mem_desc,
         &co,
         &eo,
         &group_by_and_aggregate,
         &join_loops,
         &ra_exe_unit](const std::vector<llvm::Value*>& prev_iters) {
          AUTOMATIC_IR_METADATA(cgen_state_.get());
          addJoinLoopIterator(prev_iters, join_loops.size());
          auto& builder = cgen_state_->ir_builder_;
          const auto loop_body_bb = llvm::BasicBlock::Create(
              builder.getContext(), "loop_body", builder.GetInsertBlock()->getParent());
          builder.SetInsertPoint(loop_body_bb);
          const bool can_return_error =
              compileBody(ra_exe_unit, group_by_and_aggregate, query_mem_desc, co);
          if (can_return_error || cgen_state_->needs_error_check_ ||
              eo.with_dynamic_watchdog || eo.allow_runtime_query_interrupt) {
            createErrorCheckControlFlow(query_func,
                                        eo.with_dynamic_watchdog,
                                        eo.allow_runtime_query_interrupt,
                                        join_loops,
                                        co.device_type,
                                        group_by_and_aggregate.query_infos_);
          }
          return loop_body_bb;
        },
        /*outer_iter=*/code_generator.posArg(nullptr),
        exit_bb,
        cgen_state_.get());
  }
  CHECK(loops_entry_bb);
  cgen_state_->ir_builder_.SetInsertPoint(entry_bb);
  cgen_state_->ir_builder_.CreateBr(loops_entry_bb);
}

Executor::GroupColLLVMValue Executor::groupByColumnCodegen(
    Analyzer::Expr* group_by_col,
    const size_t col_width,
    const CompilationOptions& co,
    const bool translate_null_val,
    const int64_t translated_null_val,
    DiamondCodegen& diamond_codegen,
    std::stack<llvm::BasicBlock*>& array_loops,
    const bool thread_mem_shared) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  CHECK_GE(col_width, sizeof(int32_t));
  CodeGenerator code_generator(this);
  auto group_key = code_generator.codegen(group_by_col, true, co).front();
  auto key_to_cache = group_key;
  if (dynamic_cast<Analyzer::UOper*>(group_by_col) &&
      static_cast<Analyzer::UOper*>(group_by_col)->get_optype() == kUNNEST) {
    auto preheader = cgen_state_->ir_builder_.GetInsertBlock();
    auto array_loop_head = llvm::BasicBlock::Create(cgen_state_->context_,
                                                    "array_loop_head",
                                                    cgen_state_->current_func_,
                                                    preheader->getNextNode());
    diamond_codegen.setFalseTarget(array_loop_head);
    const auto ret_ty = get_int_type(32, cgen_state_->context_);
    auto array_idx_ptr = cgen_state_->ir_builder_.CreateAlloca(ret_ty);
    CHECK(array_idx_ptr);
    cgen_state_->ir_builder_.CreateStore(cgen_state_->llInt(int32_t(0)), array_idx_ptr);
    const auto arr_expr = static_cast<Analyzer::UOper*>(group_by_col)->get_operand();
    const auto& array_ti = arr_expr->get_type_info();
    CHECK(array_ti.is_array());
    const auto& elem_ti = array_ti.get_elem_type();
    auto array_len =
        (array_ti.get_size() > 0)
            ? cgen_state_->llInt(array_ti.get_size() / elem_ti.get_size())
            : cgen_state_->emitExternalCall(
                  "array_size",
                  ret_ty,
                  {group_key,
                   code_generator.posArg(arr_expr),
                   cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))});
    cgen_state_->ir_builder_.CreateBr(array_loop_head);
    cgen_state_->ir_builder_.SetInsertPoint(array_loop_head);
    CHECK(array_len);
    auto array_idx = cgen_state_->ir_builder_.CreateLoad(
        array_idx_ptr->getType()->getPointerElementType(), array_idx_ptr);
    auto bound_check = cgen_state_->ir_builder_.CreateICmp(
        llvm::ICmpInst::ICMP_SLT, array_idx, array_len);
    auto array_loop_body = llvm::BasicBlock::Create(
        cgen_state_->context_, "array_loop_body", cgen_state_->current_func_);
    cgen_state_->ir_builder_.CreateCondBr(
        bound_check,
        array_loop_body,
        array_loops.empty() ? diamond_codegen.orig_cond_false_ : array_loops.top());
    cgen_state_->ir_builder_.SetInsertPoint(array_loop_body);
    cgen_state_->ir_builder_.CreateStore(
        cgen_state_->ir_builder_.CreateAdd(array_idx, cgen_state_->llInt(int32_t(1))),
        array_idx_ptr);
    auto array_at_fname = "array_at_" + numeric_type_name(elem_ti);
    if (array_ti.get_size() < 0) {
      if (array_ti.get_notnull()) {
        array_at_fname = "notnull_" + array_at_fname;
      }
      array_at_fname = "varlen_" + array_at_fname;
    }
    const auto ar_ret_ty =
        elem_ti.is_fp()
            ? (elem_ti.get_type() == kDOUBLE
                   ? llvm::Type::getDoubleTy(cgen_state_->context_)
                   : llvm::Type::getFloatTy(cgen_state_->context_))
            : get_int_type(elem_ti.get_logical_size() * 8, cgen_state_->context_);
    group_key = cgen_state_->emitExternalCall(
        array_at_fname,
        ar_ret_ty,
        {group_key, code_generator.posArg(arr_expr), array_idx});
    if (need_patch_unnest_double(
            elem_ti, isArchMaxwell(co.device_type), thread_mem_shared)) {
      key_to_cache = spillDoubleElement(group_key, ar_ret_ty);
    } else {
      key_to_cache = group_key;
    }
    CHECK(array_loop_head);
    array_loops.push(array_loop_head);
  }
  cgen_state_->group_by_expr_cache_.push_back(key_to_cache);
  llvm::Value* orig_group_key{nullptr};
  if (translate_null_val) {
    const std::string translator_func_name(
        col_width == sizeof(int32_t) ? "translate_null_key_i32_" : "translate_null_key_");
    const auto& ti = group_by_col->get_type_info();
    const auto key_type = get_int_type(ti.get_logical_size() * 8, cgen_state_->context_);
    orig_group_key = group_key;
    group_key = cgen_state_->emitCall(
        translator_func_name + numeric_type_name(ti),
        {group_key,
         static_cast<llvm::Value*>(
             llvm::ConstantInt::get(key_type, inline_int_null_val(ti))),
         static_cast<llvm::Value*>(llvm::ConstantInt::get(
             llvm::Type::getInt64Ty(cgen_state_->context_), translated_null_val))});
  }
  group_key = cgen_state_->ir_builder_.CreateBitCast(
      cgen_state_->castToTypeIn(group_key, col_width * 8),
      get_int_type(col_width * 8, cgen_state_->context_));
  if (orig_group_key) {
    orig_group_key = cgen_state_->ir_builder_.CreateBitCast(
        cgen_state_->castToTypeIn(orig_group_key, col_width * 8),
        get_int_type(col_width * 8, cgen_state_->context_));
  }
  return {group_key, orig_group_key};
}

CodeGenerator::NullCheckCodegen::NullCheckCodegen(CgenState* cgen_state,
                                                  Executor* executor,
                                                  llvm::Value* nullable_lv,
                                                  const SQLTypeInfo& nullable_ti,
                                                  const std::string& name)
    : cgen_state(cgen_state), name(name) {
  AUTOMATIC_IR_METADATA(cgen_state);
  CHECK(nullable_ti.is_number() || nullable_ti.is_time() || nullable_ti.is_boolean() ||
        nullable_ti.is_dict_encoded_string());

  llvm::Value* is_null_lv{nullptr};
  if (nullable_ti.is_fp()) {
    is_null_lv = cgen_state->ir_builder_.CreateFCmp(
        llvm::FCmpInst::FCMP_OEQ, nullable_lv, cgen_state->inlineFpNull(nullable_ti));
  } else if (nullable_ti.is_boolean() &&
             nullable_lv->getType()->getIntegerBitWidth() == 1) {
    is_null_lv = cgen_state->ir_builder_.CreateICmp(
        llvm::ICmpInst::ICMP_EQ, nullable_lv, cgen_state->llBool(true));
  } else {
    is_null_lv = cgen_state->ir_builder_.CreateICmp(
        llvm::ICmpInst::ICMP_EQ, nullable_lv, cgen_state->inlineIntNull(nullable_ti));
  }
  CHECK(is_null_lv);
  null_check =
      std::make_unique<DiamondCodegen>(is_null_lv, executor, false, name, nullptr, false);

  // generate a phi node depending on whether we got a null or not
  nullcheck_bb = llvm::BasicBlock::Create(
      cgen_state->context_, name + "_bb", cgen_state->current_func_);

  // update the blocks created by diamond codegen to point to the newly created phi
  // block
  cgen_state->ir_builder_.SetInsertPoint(null_check->cond_true_);
  cgen_state->ir_builder_.CreateBr(nullcheck_bb);
  cgen_state->ir_builder_.SetInsertPoint(null_check->cond_false_);
}

llvm::Value* CodeGenerator::NullCheckCodegen::finalize(llvm::Value* null_lv,
                                                       llvm::Value* notnull_lv) {
  AUTOMATIC_IR_METADATA(cgen_state);
  CHECK(null_check);
  cgen_state->ir_builder_.CreateBr(nullcheck_bb);

  CHECK_EQ(null_lv->getType(), notnull_lv->getType());

  cgen_state->ir_builder_.SetInsertPoint(nullcheck_bb);
  nullcheck_value =
      cgen_state->ir_builder_.CreatePHI(null_lv->getType(), 2, name + "_value");
  nullcheck_value->addIncoming(notnull_lv, null_check->cond_false_);
  nullcheck_value->addIncoming(null_lv, null_check->cond_true_);

  null_check.reset(nullptr);
  cgen_state->ir_builder_.SetInsertPoint(nullcheck_bb);
  return nullcheck_value;
}
