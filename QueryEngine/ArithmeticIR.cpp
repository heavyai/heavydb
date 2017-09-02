/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "Execute.h"

#include "Parser/ParserNode.h"

// Code generation routines and helpers for basic arithmetic and unary minus.

namespace {

std::string numeric_or_time_interval_type_name(const SQLTypeInfo& ti1, const SQLTypeInfo& ti2) {
  if (ti2.is_timeinterval()) {
    return numeric_type_name(ti2);
  }
  return numeric_type_name(ti1);
}

}  // namespace

llvm::Value* Executor::codegenArith(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_ARITHMETIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();

  if (lhs_type.is_decimal() && rhs_type.is_decimal() && optype == kDIVIDE) {
    const auto ret = codegenDeciDiv(bin_oper, co);
    if (ret) {
      return ret;
    }
  }

  auto lhs_lv = codegen(lhs, true, co).front();
  auto rhs_lv = codegen(rhs, true, co).front();
  // Handle operations when a time interval operand is involved, an operation
  // between an integer and a time interval isn't normalized by the analyzer.
  if (lhs_type.is_timeinterval()) {
    rhs_lv = codegenCastBetweenIntTypes(rhs_lv, rhs_type, lhs_type);
  } else if (rhs_type.is_timeinterval()) {
    lhs_lv = codegenCastBetweenIntTypes(lhs_lv, lhs_type, rhs_type);
  } else {
    CHECK_EQ(lhs_type.get_type(), rhs_type.get_type());
  }
  if (lhs_type.is_integer() || lhs_type.is_decimal() || lhs_type.is_timeinterval()) {
    return codegenIntArith(bin_oper, lhs_lv, rhs_lv);
  }
  if (lhs_type.is_fp()) {
    return codegenFpArith(bin_oper, lhs_lv, rhs_lv);
  }
  CHECK(false);
  return nullptr;
}

// Handle integer or integer-like (decimal, time, date) operand types.
llvm::Value* Executor::codegenIntArith(const Analyzer::BinOper* bin_oper, llvm::Value* lhs_lv, llvm::Value* rhs_lv) {
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  const auto int_typename = numeric_or_time_interval_type_name(lhs_type, rhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  const auto& oper_type = rhs_type.is_timeinterval() ? rhs_type : lhs_type;
  switch (bin_oper->get_optype()) {
    case kMINUS:
      return codegenSub(
          bin_oper, lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
    case kPLUS:
      return codegenAdd(
          bin_oper, lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
    case kMULTIPLY:
      return codegenMul(
          bin_oper, lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
    case kDIVIDE:
      return codegenDiv(lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
    case kMODULO:
      return codegenMod(lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : int_typename, null_check_suffix, oper_type);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

// Handle floating point operand types.
llvm::Value* Executor::codegenFpArith(const Analyzer::BinOper* bin_oper, llvm::Value* lhs_lv, llvm::Value* rhs_lv) {
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  const auto fp_typename = numeric_type_name(lhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  llvm::ConstantFP* fp_null{lhs_type.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE)};
  switch (bin_oper->get_optype()) {
    case kMINUS:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFSub(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("sub_" + fp_typename + null_check_suffix, {lhs_lv, rhs_lv, fp_null});
    case kPLUS:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFAdd(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("add_" + fp_typename + null_check_suffix, {lhs_lv, rhs_lv, fp_null});
    case kMULTIPLY:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFMul(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("mul_" + fp_typename + null_check_suffix, {lhs_lv, rhs_lv, fp_null});
    case kDIVIDE:
      return codegenDiv(lhs_lv, rhs_lv, null_check_suffix.empty() ? "" : fp_typename, null_check_suffix, lhs_type);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

namespace {

bool is_temporary_column(const Analyzer::Expr* expr) {
  const auto col_expr = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (!col_expr) {
    return false;
  }
  return col_expr->get_table_id() < 0;
}

}  // namespace

// Returns true iff runtime overflow checks aren't needed thanks to range information.
bool Executor::checkExpressionRanges(const Analyzer::BinOper* bin_oper, int64_t min, int64_t max) {
  if (is_temporary_column(bin_oper->get_left_operand()) || is_temporary_column(bin_oper->get_right_operand())) {
    // Computing the range for temporary columns is a lot more expensive than the overflow check.
    return false;
  }
  if (bin_oper->get_type_info().is_decimal())
    return false;

  auto expr_range_info = cgen_state_->query_infos_.size() > 0
                             ? getExpressionRange(bin_oper, cgen_state_->query_infos_, this)
                             : ExpressionRange::makeInvalidRange();
  if (expr_range_info.getType() != ExpressionRangeType::Integer)
    return false;
  if (expr_range_info.getIntMin() >= min && expr_range_info.getIntMax() <= max)
    return true;

  return false;
}

llvm::Value* Executor::codegenAdd(const Analyzer::BinOper* bin_oper,
                                  llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check = !checkExpressionRanges(bin_oper,
                                                    static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                                                    static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  llvm::BasicBlock* add_ok{nullptr};
  llvm::BasicBlock* add_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    add_ok = llvm::BasicBlock::Create(cgen_state_->context_, "add_ok", cgen_state_->row_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, add_ok, ti);
    }
    add_fail = llvm::BasicBlock::Create(cgen_state_->context_, "add_fail", cgen_state_->row_func_);
    llvm::Value* detected{nullptr};
    auto const_zero = llvm::ConstantInt::get(lhs_lv->getType(), 0, true);
    auto overflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSGT(rhs_lv, cgen_state_->ir_builder_.CreateSub(chosen_max, lhs_lv)));
    auto underflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSLT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSLT(rhs_lv, cgen_state_->ir_builder_.CreateSub(chosen_min, lhs_lv)));
    detected = cgen_state_->ir_builder_.CreateOr(overflow, underflow);
    cgen_state_->ir_builder_.CreateCondBr(detected, add_fail, add_ok);
    cgen_state_->ir_builder_.SetInsertPoint(add_ok);
  }
  auto ret = null_check_suffix.empty() ? cgen_state_->ir_builder_.CreateAdd(lhs_lv, rhs_lv)
                                       : cgen_state_->emitCall("add_" + null_typename + null_check_suffix,
                                                               {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(add_fail);
    cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(add_ok);
  }
  return ret;
}

llvm::Value* Executor::codegenSub(const Analyzer::BinOper* bin_oper,
                                  llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check = !checkExpressionRanges(bin_oper,
                                                    static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                                                    static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  llvm::BasicBlock* sub_ok{nullptr};
  llvm::BasicBlock* sub_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    sub_ok = llvm::BasicBlock::Create(cgen_state_->context_, "sub_ok", cgen_state_->row_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, sub_ok, ti);
    }
    sub_fail = llvm::BasicBlock::Create(cgen_state_->context_, "sub_fail", cgen_state_->row_func_);
    llvm::Value* detected{nullptr};
    auto const_zero = llvm::ConstantInt::get(lhs_lv->getType(), 0, true);
    auto overflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSLT(rhs_lv, const_zero),  // sub going up, check the max
        cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, cgen_state_->ir_builder_.CreateAdd(chosen_max, rhs_lv)));
    auto underflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSGT(rhs_lv, const_zero),  // sub going down, check the min
        cgen_state_->ir_builder_.CreateICmpSLT(lhs_lv, cgen_state_->ir_builder_.CreateAdd(chosen_min, rhs_lv)));
    detected = cgen_state_->ir_builder_.CreateOr(overflow, underflow);
    cgen_state_->ir_builder_.CreateCondBr(detected, sub_fail, sub_ok);
    cgen_state_->ir_builder_.SetInsertPoint(sub_ok);
  }
  auto ret = null_check_suffix.empty() ? cgen_state_->ir_builder_.CreateSub(lhs_lv, rhs_lv)
                                       : cgen_state_->emitCall("sub_" + null_typename + null_check_suffix,
                                                               {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(sub_fail);
    cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(sub_ok);
  }
  return ret;
}

void Executor::codegenSkipOverflowCheckForNull(llvm::Value* lhs_lv,
                                               llvm::Value* rhs_lv,
                                               llvm::BasicBlock* no_overflow_bb,
                                               const SQLTypeInfo& ti) {
  const auto lhs_is_null_lv = codegenIsNullNumber(lhs_lv, ti);
  const auto has_null_operand_lv =
      rhs_lv ? cgen_state_->ir_builder_.CreateOr(lhs_is_null_lv, codegenIsNullNumber(rhs_lv, ti)) : lhs_is_null_lv;
  auto operands_not_null = llvm::BasicBlock::Create(cgen_state_->context_, "operands_not_null", cgen_state_->row_func_);
  cgen_state_->ir_builder_.CreateCondBr(has_null_operand_lv, no_overflow_bb, operands_not_null);
  cgen_state_->ir_builder_.SetInsertPoint(operands_not_null);
}

llvm::Value* Executor::codegenMul(const Analyzer::BinOper* bin_oper,
                                  llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti,
                                  bool downscale) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check = !checkExpressionRanges(bin_oper,
                                                    static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                                                    static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  llvm::BasicBlock* mul_ok{nullptr};
  llvm::BasicBlock* mul_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    mul_ok = llvm::BasicBlock::Create(cgen_state_->context_, "mul_ok", cgen_state_->row_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, mul_ok, ti);
    }
    mul_fail = llvm::BasicBlock::Create(cgen_state_->context_, "mul_fail", cgen_state_->row_func_);
    auto mul_check = llvm::BasicBlock::Create(cgen_state_->context_, "mul_check", cgen_state_->row_func_);
    auto const_zero = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
    cgen_state_->ir_builder_.CreateCondBr(cgen_state_->ir_builder_.CreateICmpEQ(rhs_lv, const_zero), mul_ok, mul_check);
    cgen_state_->ir_builder_.SetInsertPoint(mul_check);
    auto rhs_is_negative_lv = cgen_state_->ir_builder_.CreateICmpSLT(rhs_lv, const_zero);
    auto positive_rhs_lv =
        cgen_state_->ir_builder_.CreateSelect(rhs_is_negative_lv, cgen_state_->ir_builder_.CreateNeg(rhs_lv), rhs_lv);
    auto adjusted_lhs_lv =
        cgen_state_->ir_builder_.CreateSelect(rhs_is_negative_lv, cgen_state_->ir_builder_.CreateNeg(lhs_lv), lhs_lv);
    auto detected = cgen_state_->ir_builder_.CreateOr(  // overflow
        cgen_state_->ir_builder_.CreateICmpSGT(adjusted_lhs_lv,
                                               cgen_state_->ir_builder_.CreateSDiv(chosen_max, positive_rhs_lv)),
        // underflow
        cgen_state_->ir_builder_.CreateICmpSLT(adjusted_lhs_lv,
                                               cgen_state_->ir_builder_.CreateSDiv(chosen_min, positive_rhs_lv)));
    cgen_state_->ir_builder_.CreateCondBr(detected, mul_fail, mul_ok);
    cgen_state_->ir_builder_.SetInsertPoint(mul_ok);
  }
  const auto ret = null_check_suffix.empty() ? cgen_state_->ir_builder_.CreateMul(lhs_lv, rhs_lv)
                                             : cgen_state_->emitCall("mul_" + null_typename + null_check_suffix,
                                                                     {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(mul_fail);
    cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(mul_ok);
  }
  return ret;
}

llvm::Value* Executor::codegenDiv(llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti,
                                  bool upscale) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  if (ti.is_decimal()) {
    if (upscale) {
      CHECK(lhs_lv->getType()->isIntegerTy());
      const auto scale_lv = llvm::ConstantInt::get(lhs_lv->getType(), exp_to_scale(ti.get_scale()));

      lhs_lv = cgen_state_->ir_builder_.CreateSExt(lhs_lv, get_int_type(64, cgen_state_->context_));
      llvm::Value* chosen_max{nullptr};
      llvm::Value* chosen_min{nullptr};
      std::tie(chosen_max, chosen_min) = inlineIntMaxMin(8, true);
      auto decimal_div_ok = llvm::BasicBlock::Create(cgen_state_->context_, "decimal_div_ok", cgen_state_->row_func_);
      if (!null_check_suffix.empty()) {
        codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, decimal_div_ok, ti);
      }
      auto decimal_div_fail =
          llvm::BasicBlock::Create(cgen_state_->context_, "decimal_div_fail", cgen_state_->row_func_);
      auto lhs_max = static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue() / exp_to_scale(ti.get_scale());
      auto lhs_max_lv = llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), lhs_max);
      llvm::Value* detected{nullptr};
      if (ti.get_notnull()) {
        detected = cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, lhs_max_lv);
      } else {
        detected = toBool(cgen_state_->emitCall(
            "gt_" + numeric_type_name(ti) + "_nullable",
            {lhs_lv, lhs_max_lv, ll_int(inline_int_null_val(ti)), inlineIntNull(SQLTypeInfo(kBOOLEAN, false))}));
      }
      cgen_state_->ir_builder_.CreateCondBr(detected, decimal_div_fail, decimal_div_ok);

      cgen_state_->ir_builder_.SetInsertPoint(decimal_div_fail);
      cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));

      cgen_state_->ir_builder_.SetInsertPoint(decimal_div_ok);

      lhs_lv = null_typename.empty() ? cgen_state_->ir_builder_.CreateMul(lhs_lv, scale_lv)
                                     : cgen_state_->emitCall("mul_" + numeric_type_name(ti) + null_check_suffix,
                                                             {lhs_lv, scale_lv, ll_int(inline_int_null_val(ti))});
    }
  }
  if (g_null_div_by_zero) {
    llvm::Value* null_lv{nullptr};
    if (ti.is_fp()) {
      null_lv = ti.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE);
    } else {
      null_lv = ll_int(inline_int_null_val(ti));
    }
    return cgen_state_->emitCall("safe_div_" + numeric_type_name(ti), {lhs_lv, rhs_lv, null_lv});
  }
  cgen_state_->needs_error_check_ = true;
  auto div_ok = llvm::BasicBlock::Create(cgen_state_->context_, "div_ok", cgen_state_->row_func_);
  if (!null_check_suffix.empty()) {
    codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, div_ok, ti);
  }
  auto div_zero = llvm::BasicBlock::Create(cgen_state_->context_, "div_zero", cgen_state_->row_func_);
  auto zero_const = rhs_lv->getType()->isIntegerTy() ? llvm::ConstantInt::get(rhs_lv->getType(), 0, true)
                                                     : llvm::ConstantFP::get(rhs_lv->getType(), 0.);
  cgen_state_->ir_builder_.CreateCondBr(
      zero_const->getType()->isFloatingPointTy()
          ? cgen_state_->ir_builder_.CreateFCmp(llvm::FCmpInst::FCMP_ONE, rhs_lv, zero_const)
          : cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_NE, rhs_lv, zero_const),
      div_ok,
      div_zero);
  cgen_state_->ir_builder_.SetInsertPoint(div_ok);
  auto ret = zero_const->getType()->isIntegerTy()
                 ? (null_typename.empty() ? cgen_state_->ir_builder_.CreateSDiv(lhs_lv, rhs_lv)
                                          : cgen_state_->emitCall("div_" + null_typename + null_check_suffix,
                                                                  {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))}))
                 : (null_typename.empty()
                        ? cgen_state_->ir_builder_.CreateFDiv(lhs_lv, rhs_lv)
                        : cgen_state_->emitCall(
                              "div_" + null_typename + null_check_suffix,
                              {lhs_lv, rhs_lv, ti.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE)}));
  cgen_state_->ir_builder_.SetInsertPoint(div_zero);
  cgen_state_->ir_builder_.CreateRet(ll_int(ERR_DIV_BY_ZERO));
  cgen_state_->ir_builder_.SetInsertPoint(div_ok);
  return ret;
}

// Handle decimal division by an integer (constant or cast), return null if
// the expression doesn't match this pattern and let the general method kick in.
// For said patterns, we can simply divide the decimal operand by the non-scaled
// integer value instead of using the scaled value preceded by a multiplication.
// It is both more efficient and avoids the overflow for a lot of practical cases.
llvm::Value* Executor::codegenDeciDiv(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  auto lhs = bin_oper->get_left_operand();
  auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  CHECK(lhs_type.is_decimal() && rhs_type.is_decimal() && lhs_type.get_scale() == rhs_type.get_scale());

  auto rhs_constant = dynamic_cast<const Analyzer::Constant*>(rhs);
  auto rhs_cast = dynamic_cast<const Analyzer::UOper*>(rhs);
  if (rhs_constant && !rhs_constant->get_is_null() && rhs_constant->get_constval().bigintval != 0LL &&
      (rhs_constant->get_constval().bigintval % exp_to_scale(rhs_type.get_scale())) == 0LL) {
    // can safely downscale a scaled constant
  } else if (rhs_cast && rhs_cast->get_optype() == kCAST && rhs_cast->get_operand()->get_type_info().is_integer()) {
    // can skip upscale in the int to dec cast
  } else {
    return nullptr;
  }

  auto lhs_lv = codegen(lhs, true, co).front();
  llvm::Value* rhs_lv{nullptr};
  if (rhs_constant) {
    const auto rhs_lit =
        Parser::IntLiteral::analyzeValue(rhs_constant->get_constval().bigintval / exp_to_scale(rhs_type.get_scale()));
    auto rhs_lit_lv = codegenIntConst(dynamic_cast<const Analyzer::Constant*>(rhs_lit.get()));
    rhs_lv = codegenCastBetweenIntTypes(rhs_lit_lv, rhs_lit->get_type_info(), lhs_type, /*upscale*/ false);
  } else if (rhs_cast) {
    auto rhs_cast_oper = rhs_cast->get_operand();
    const auto& rhs_cast_oper_ti = rhs_cast_oper->get_type_info();
    auto rhs_cast_oper_lv = codegen(rhs_cast_oper, true, co).front();
    rhs_lv = codegenCastBetweenIntTypes(rhs_cast_oper_lv, rhs_cast_oper_ti, lhs_type, /*upscale*/ false);
  } else {
    CHECK(false);
  }
  const auto int_typename = numeric_or_time_interval_type_name(lhs_type, rhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  return codegenDiv(lhs_lv,
                    rhs_lv,
                    null_check_suffix.empty() ? "" : int_typename,
                    null_check_suffix,
                    lhs_type,
                    /*upscale*/ false);
}

llvm::Value* Executor::codegenMod(llvm::Value* lhs_lv,
                                  llvm::Value* rhs_lv,
                                  const std::string& null_typename,
                                  const std::string& null_check_suffix,
                                  const SQLTypeInfo& ti) {
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer());
  cgen_state_->needs_error_check_ = true;
  // Generate control flow for division by zero error handling.
  auto mod_ok = llvm::BasicBlock::Create(cgen_state_->context_, "mod_ok", cgen_state_->row_func_);
  auto mod_zero = llvm::BasicBlock::Create(cgen_state_->context_, "mod_zero", cgen_state_->row_func_);
  auto zero_const = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
  cgen_state_->ir_builder_.CreateCondBr(
      cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_NE, rhs_lv, zero_const), mod_ok, mod_zero);
  cgen_state_->ir_builder_.SetInsertPoint(mod_ok);
  auto ret = null_typename.empty() ? cgen_state_->ir_builder_.CreateSRem(lhs_lv, rhs_lv)
                                   : cgen_state_->emitCall("mod_" + null_typename + null_check_suffix,
                                                           {lhs_lv, rhs_lv, ll_int(inline_int_null_val(ti))});
  cgen_state_->ir_builder_.SetInsertPoint(mod_zero);
  cgen_state_->ir_builder_.CreateRet(ll_int(ERR_DIV_BY_ZERO));
  cgen_state_->ir_builder_.SetInsertPoint(mod_ok);
  return ret;
}

// Returns true iff runtime overflow checks aren't needed thanks to range information.
bool Executor::checkExpressionRanges(const Analyzer::UOper* uoper, int64_t min, int64_t max) {
  if (uoper->get_type_info().is_decimal())
    return false;

  auto expr_range_info = cgen_state_->query_infos_.size() > 0
                             ? getExpressionRange(uoper, cgen_state_->query_infos_, this)
                             : ExpressionRange::makeInvalidRange();
  if (expr_range_info.getType() != ExpressionRangeType::Integer)
    return false;
  if (expr_range_info.getIntMin() >= min && expr_range_info.getIntMax() <= max)
    return true;

  return false;
}

llvm::Value* Executor::codegenUMinus(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  CHECK_EQ(uoper->get_optype(), kUMINUS);
  const auto operand_lv = codegen(uoper->get_operand(), true, co).front();
  const auto& ti = uoper->get_type_info();
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  bool need_overflow_check = false;
  if (ti.is_integer() || ti.is_decimal() || ti.is_timeinterval()) {
    std::tie(chosen_max, chosen_min) = inlineIntMaxMin(ti.get_size(), true);
    need_overflow_check = !checkExpressionRanges(uoper,
                                                 static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                                                 static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  }
  llvm::BasicBlock* uminus_ok{nullptr};
  llvm::BasicBlock* uminus_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    uminus_ok = llvm::BasicBlock::Create(cgen_state_->context_, "uminus_ok", cgen_state_->row_func_);
    if (!ti.get_notnull()) {
      codegenSkipOverflowCheckForNull(operand_lv, nullptr, uminus_ok, ti);
    }
    uminus_fail = llvm::BasicBlock::Create(cgen_state_->context_, "uminus_fail", cgen_state_->row_func_);
    auto const_min = llvm::ConstantInt::get(
        operand_lv->getType(), static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(), true);
    auto overflow = cgen_state_->ir_builder_.CreateICmpEQ(operand_lv, const_min);
    cgen_state_->ir_builder_.CreateCondBr(overflow, uminus_fail, uminus_ok);
    cgen_state_->ir_builder_.SetInsertPoint(uminus_ok);
  }
  auto ret = ti.get_notnull() ? (ti.is_fp() ? cgen_state_->ir_builder_.CreateFNeg(operand_lv)
                                            : cgen_state_->ir_builder_.CreateNeg(operand_lv))
                              : cgen_state_->emitCall("uminus_" + numeric_type_name(ti) + "_nullable",
                                                      {operand_lv,
                                                       ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(ti))
                                                                  : static_cast<llvm::Value*>(inlineIntNull(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(uminus_fail);
    cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));
    cgen_state_->ir_builder_.SetInsertPoint(uminus_ok);
  }
  return ret;
}
