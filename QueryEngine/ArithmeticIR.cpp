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

#include "CodeGenerator.h"
#include "Execute.h"
#include "Parser/ParserNode.h"

#include <cstdlib>

// Code generation routines and helpers for basic arithmetic and unary minus.

using heavyai::ErrorCode;

namespace {

std::string numeric_or_time_interval_type_name(const SQLTypeInfo& ti1,
                                               const SQLTypeInfo& ti2) {
  if (ti2.is_timeinterval()) {
    return numeric_type_name(ti2);
  }
  return numeric_type_name(ti1);
}

}  // namespace

llvm::Value* CodeGenerator::codegenArith(const Analyzer::BinOper* bin_oper,
                                         const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto optype = bin_oper->get_optype();
  CHECK(IS_ARITHMETIC(optype));
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();

  if (lhs_type.is_decimal() && rhs_type.is_decimal() && optype == kDIVIDE) {
    return codegenDeciDiv(bin_oper, co);
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
    return codegenIntArith(bin_oper, lhs_lv, rhs_lv, co);
  }
  if (lhs_type.is_fp()) {
    return codegenFpArith(bin_oper, lhs_lv, rhs_lv);
  }
  CHECK(false);
  return nullptr;
}

// Handle integer or integer-like (decimal, time, date) operand types.
llvm::Value* CodeGenerator::codegenIntArith(const Analyzer::BinOper* bin_oper,
                                            llvm::Value* lhs_lv,
                                            llvm::Value* rhs_lv,
                                            const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  const auto int_typename = numeric_or_time_interval_type_name(lhs_type, rhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  const auto& oper_type = rhs_type.is_timeinterval() ? rhs_type : lhs_type;
  switch (bin_oper->get_optype()) {
    case kMINUS:
      return codegenSub(bin_oper,
                        lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type,
                        co);
    case kPLUS:
      return codegenAdd(bin_oper,
                        lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type,
                        co);
    case kMULTIPLY:
      return codegenMul(bin_oper,
                        lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type,
                        co);
    case kDIVIDE:
      return codegenDiv(lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type);
    case kMODULO:
      return codegenMod(lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : int_typename,
                        null_check_suffix,
                        oper_type);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

// Handle floating point operand types.
llvm::Value* CodeGenerator::codegenFpArith(const Analyzer::BinOper* bin_oper,
                                           llvm::Value* lhs_lv,
                                           llvm::Value* rhs_lv) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  const auto& lhs_type = lhs->get_type_info();
  const auto& rhs_type = rhs->get_type_info();
  const auto fp_typename = numeric_type_name(lhs_type);
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  llvm::ConstantFP* fp_null{lhs_type.get_type() == kFLOAT
                                ? cgen_state_->llFp(NULL_FLOAT)
                                : cgen_state_->llFp(NULL_DOUBLE)};
  switch (bin_oper->get_optype()) {
    case kMINUS:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFSub(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("sub_" + fp_typename + null_check_suffix,
                                         {lhs_lv, rhs_lv, fp_null});
    case kPLUS:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFAdd(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("add_" + fp_typename + null_check_suffix,
                                         {lhs_lv, rhs_lv, fp_null});
    case kMULTIPLY:
      return null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateFMul(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall("mul_" + fp_typename + null_check_suffix,
                                         {lhs_lv, rhs_lv, fp_null});
    case kDIVIDE:
      return codegenDiv(lhs_lv,
                        rhs_lv,
                        null_check_suffix.empty() ? "" : fp_typename,
                        null_check_suffix,
                        lhs_type);
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
  return col_expr->getColumnKey().table_id < 0;
}

}  // namespace

// Returns true iff runtime overflow checks aren't needed thanks to range information.
bool CodeGenerator::checkExpressionRanges(const Analyzer::BinOper* bin_oper,
                                          int64_t min,
                                          int64_t max) {
  if (is_temporary_column(bin_oper->get_left_operand()) ||
      is_temporary_column(bin_oper->get_right_operand())) {
    // Computing the range for temporary columns is a lot more expensive than the overflow
    // check.
    return false;
  }
  if (bin_oper->get_type_info().is_decimal()) {
    return false;
  }

  CHECK(plan_state_);
  if (executor_) {
    auto expr_range_info =
        plan_state_->query_infos_.size() > 0
            ? getExpressionRange(bin_oper, plan_state_->query_infos_, executor())
            : ExpressionRange::makeInvalidRange();
    if (expr_range_info.getType() != ExpressionRangeType::Integer) {
      return false;
    }
    if (expr_range_info.getIntMin() >= min && expr_range_info.getIntMax() <= max) {
      return true;
    }
  }

  return false;
}

llvm::Value* CodeGenerator::codegenAdd(const Analyzer::BinOper* bin_oper,
                                       llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const SQLTypeInfo& ti,
                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check =
      !checkExpressionRanges(bin_oper,
                             static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                             static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());

  if (need_overflow_check && co.device_type == ExecutorDeviceType::CPU) {
    return codegenBinOpWithOverflowForCPU(
        bin_oper, lhs_lv, rhs_lv, null_check_suffix, ti);
  }

  llvm::BasicBlock* add_ok{nullptr};
  llvm::BasicBlock* add_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    add_ok = llvm::BasicBlock::Create(
        cgen_state_->context_, "add_ok", cgen_state_->current_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, add_ok, ti);
    }
    add_fail = llvm::BasicBlock::Create(
        cgen_state_->context_, "add_fail", cgen_state_->current_func_);
    llvm::Value* detected{nullptr};
    auto const_zero = llvm::ConstantInt::get(lhs_lv->getType(), 0, true);
    auto overflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSGT(
            rhs_lv, cgen_state_->ir_builder_.CreateSub(chosen_max, lhs_lv)));
    auto underflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSLT(lhs_lv, const_zero),
        cgen_state_->ir_builder_.CreateICmpSLT(
            rhs_lv, cgen_state_->ir_builder_.CreateSub(chosen_min, lhs_lv)));
    detected = cgen_state_->ir_builder_.CreateOr(overflow, underflow);
    cgen_state_->ir_builder_.CreateCondBr(detected, add_fail, add_ok);
    cgen_state_->ir_builder_.SetInsertPoint(add_ok);
  }
  auto ret = null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateAdd(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "add_" + null_typename + null_check_suffix,
                       {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_val(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(add_fail);
    cgen_state_->ir_builder_.CreateRet(
        cgen_state_->llInt(int32_t(ErrorCode::OVERFLOW_OR_UNDERFLOW)));
    cgen_state_->ir_builder_.SetInsertPoint(add_ok);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegenSub(const Analyzer::BinOper* bin_oper,
                                       llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const SQLTypeInfo& ti,
                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check =
      !checkExpressionRanges(bin_oper,
                             static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                             static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());

  if (need_overflow_check && co.device_type == ExecutorDeviceType::CPU) {
    return codegenBinOpWithOverflowForCPU(
        bin_oper, lhs_lv, rhs_lv, null_check_suffix, ti);
  }

  llvm::BasicBlock* sub_ok{nullptr};
  llvm::BasicBlock* sub_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    sub_ok = llvm::BasicBlock::Create(
        cgen_state_->context_, "sub_ok", cgen_state_->current_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, sub_ok, ti);
    }
    sub_fail = llvm::BasicBlock::Create(
        cgen_state_->context_, "sub_fail", cgen_state_->current_func_);
    llvm::Value* detected{nullptr};
    auto const_zero = llvm::ConstantInt::get(lhs_lv->getType(), 0, true);
    auto overflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSLT(
            rhs_lv, const_zero),  // sub going up, check the max
        cgen_state_->ir_builder_.CreateICmpSGT(
            lhs_lv, cgen_state_->ir_builder_.CreateAdd(chosen_max, rhs_lv)));
    auto underflow = cgen_state_->ir_builder_.CreateAnd(
        cgen_state_->ir_builder_.CreateICmpSGT(
            rhs_lv, const_zero),  // sub going down, check the min
        cgen_state_->ir_builder_.CreateICmpSLT(
            lhs_lv, cgen_state_->ir_builder_.CreateAdd(chosen_min, rhs_lv)));
    detected = cgen_state_->ir_builder_.CreateOr(overflow, underflow);
    cgen_state_->ir_builder_.CreateCondBr(detected, sub_fail, sub_ok);
    cgen_state_->ir_builder_.SetInsertPoint(sub_ok);
  }
  auto ret = null_check_suffix.empty()
                 ? cgen_state_->ir_builder_.CreateSub(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "sub_" + null_typename + null_check_suffix,
                       {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_val(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(sub_fail);
    cgen_state_->ir_builder_.CreateRet(
        cgen_state_->llInt(int32_t(ErrorCode::OVERFLOW_OR_UNDERFLOW)));
    cgen_state_->ir_builder_.SetInsertPoint(sub_ok);
  }
  return ret;
}

void CodeGenerator::codegenSkipOverflowCheckForNull(llvm::Value* lhs_lv,
                                                    llvm::Value* rhs_lv,
                                                    llvm::BasicBlock* no_overflow_bb,
                                                    const SQLTypeInfo& ti) {
  const auto lhs_is_null_lv = codegenIsNullNumber(lhs_lv, ti);
  const auto has_null_operand_lv =
      rhs_lv ? cgen_state_->ir_builder_.CreateOr(lhs_is_null_lv,
                                                 codegenIsNullNumber(rhs_lv, ti))
             : lhs_is_null_lv;
  auto operands_not_null = llvm::BasicBlock::Create(
      cgen_state_->context_, "operands_not_null", cgen_state_->current_func_);
  cgen_state_->ir_builder_.CreateCondBr(
      has_null_operand_lv, no_overflow_bb, operands_not_null);
  cgen_state_->ir_builder_.SetInsertPoint(operands_not_null);
}

llvm::Value* CodeGenerator::codegenMul(const Analyzer::BinOper* bin_oper,
                                       llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const SQLTypeInfo& ti,
                                       const CompilationOptions& co,
                                       bool downscale) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer() || ti.is_decimal() || ti.is_timeinterval());
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(ti.get_size(), true);
  auto need_overflow_check =
      !checkExpressionRanges(bin_oper,
                             static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
                             static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());

  if (need_overflow_check && co.device_type == ExecutorDeviceType::CPU) {
    return codegenBinOpWithOverflowForCPU(
        bin_oper, lhs_lv, rhs_lv, null_check_suffix, ti);
  }

  llvm::BasicBlock* mul_ok{nullptr};
  llvm::BasicBlock* mul_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    mul_ok = llvm::BasicBlock::Create(
        cgen_state_->context_, "mul_ok", cgen_state_->current_func_);
    if (!null_check_suffix.empty()) {
      codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, mul_ok, ti);
    }
    mul_fail = llvm::BasicBlock::Create(
        cgen_state_->context_, "mul_fail", cgen_state_->current_func_);
    auto mul_check = llvm::BasicBlock::Create(
        cgen_state_->context_, "mul_check", cgen_state_->current_func_);
    auto const_zero = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
    cgen_state_->ir_builder_.CreateCondBr(
        cgen_state_->ir_builder_.CreateICmpEQ(rhs_lv, const_zero), mul_ok, mul_check);
    cgen_state_->ir_builder_.SetInsertPoint(mul_check);
    auto rhs_is_negative_lv = cgen_state_->ir_builder_.CreateICmpSLT(rhs_lv, const_zero);
    auto positive_rhs_lv = cgen_state_->ir_builder_.CreateSelect(
        rhs_is_negative_lv, cgen_state_->ir_builder_.CreateNeg(rhs_lv), rhs_lv);
    auto adjusted_lhs_lv = cgen_state_->ir_builder_.CreateSelect(
        rhs_is_negative_lv, cgen_state_->ir_builder_.CreateNeg(lhs_lv), lhs_lv);
    auto detected = cgen_state_->ir_builder_.CreateOr(  // overflow
        cgen_state_->ir_builder_.CreateICmpSGT(
            adjusted_lhs_lv,
            cgen_state_->ir_builder_.CreateSDiv(chosen_max, positive_rhs_lv)),
        // underflow
        cgen_state_->ir_builder_.CreateICmpSLT(
            adjusted_lhs_lv,
            cgen_state_->ir_builder_.CreateSDiv(chosen_min, positive_rhs_lv)));
    cgen_state_->ir_builder_.CreateCondBr(detected, mul_fail, mul_ok);
    cgen_state_->ir_builder_.SetInsertPoint(mul_ok);
  }
  const auto ret =
      null_check_suffix.empty()
          ? cgen_state_->ir_builder_.CreateMul(lhs_lv, rhs_lv)
          : cgen_state_->emitCall(
                "mul_" + null_typename + null_check_suffix,
                {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_val(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(mul_fail);
    cgen_state_->ir_builder_.CreateRet(
        cgen_state_->llInt(int32_t(ErrorCode::OVERFLOW_OR_UNDERFLOW)));
    cgen_state_->ir_builder_.SetInsertPoint(mul_ok);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegenDiv(llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const SQLTypeInfo& ti,
                                       const bool upscale) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());

  if (ti.is_decimal()) {
    if (upscale) {
      // Ensure the left-hand side (lhs) value is an integer type
      CHECK(lhs_lv->getType()->isIntegerTy());
      // Create a constant integer value representing the scale factor
      const auto scale_lv =
          llvm::ConstantInt::get(lhs_lv->getType(), exp_to_scale(ti.get_scale()));
      // Extend the lhs value to 64-bit integer to prevent overflow during calculations
      lhs_lv = cgen_state_->ir_builder_.CreateSExt(
          lhs_lv, get_int_type(64, cgen_state_->context_));
      // Initialize variables to hold the maximum and minimum values for the data type
      llvm::Value* chosen_max{nullptr};
      llvm::Value* chosen_min{nullptr};
      // Retrieve the max and min values for an 8-byte integer, considering sign
      std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(8, true);
      // Create a basic block for handling successful decimal division
      auto decimal_div_ok = llvm::BasicBlock::Create(
          cgen_state_->context_, "decimal_div_ok", cgen_state_->current_func_);
      // If a null check suffix is provided, skip overflow check for null values
      if (!null_check_suffix.empty()) {
        codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, decimal_div_ok, ti);
      }
      // Create a basic block for handling decimal division failure (overflow)
      auto decimal_div_fail = llvm::BasicBlock::Create(
          cgen_state_->context_, "decimal_div_fail", cgen_state_->current_func_);
      // Calculate the maximum lhs value adjusted for the decimal scale
      auto lhs_max = static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue() /
                     exp_to_scale(ti.get_scale());
      // Create an LLVM value for the calculated maximum lhs value
      auto lhs_max_lv =
          llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), lhs_max);
      // Initialize a variable to hold the overflow detection result
      llvm::Value* detected{nullptr};
      // TODO: QE-1154 If this block of code is needed then add lower bound check as well.
      // Currently this only checks if positive values overflow, not negative.
      if (ti.get_notnull()) {
        detected = cgen_state_->ir_builder_.CreateICmpSGT(lhs_lv, lhs_max_lv);
      } else {
        // Otherwise, call a function to handle nullable types comparison
        detected = toBool(cgen_state_->emitCall(
            "gt_" + numeric_type_name(ti) + "_nullable",
            {lhs_lv,
             lhs_max_lv,
             cgen_state_->llInt(inline_int_null_val(ti)),
             cgen_state_->inlineIntNull(SQLTypeInfo(kBOOLEAN, false))}));
      }
      // Conditionally branch to the appropriate block based on overflow detection
      cgen_state_->ir_builder_.CreateCondBr(detected, decimal_div_fail, decimal_div_ok);
      // Set the insertion point to the failure block and return an error code
      cgen_state_->ir_builder_.SetInsertPoint(decimal_div_fail);
      cgen_state_->ir_builder_.CreateRet(
          cgen_state_->llInt(int32_t(ErrorCode::OVERFLOW_OR_UNDERFLOW)));
      // Set the insertion point to the successful division block for further instructions
      cgen_state_->ir_builder_.SetInsertPoint(decimal_div_ok);
      // Multiply lhs value by the scale factor, handling nulls if necessary
      lhs_lv = null_typename.empty()
                   ? cgen_state_->ir_builder_.CreateMul(lhs_lv, scale_lv)
                   : cgen_state_->emitCall(
                         "mul_" + numeric_type_name(ti) + null_check_suffix,
                         {lhs_lv, scale_lv, cgen_state_->llInt(inline_int_null_val(ti))});
    }
  }
  if (g_null_div_by_zero) {
    llvm::Value* null_lv{nullptr};
    if (ti.is_fp()) {
      null_lv = ti.get_type() == kFLOAT ? cgen_state_->llFp(NULL_FLOAT)
                                        : cgen_state_->llFp(NULL_DOUBLE);
    } else {
      null_lv = cgen_state_->llInt(inline_int_null_val(ti));
    }
    return cgen_state_->emitCall("safe_div_" + numeric_type_name(ti),
                                 {lhs_lv, rhs_lv, null_lv});
  }
  cgen_state_->needs_error_check_ = true;
  auto div_ok = llvm::BasicBlock::Create(
      cgen_state_->context_, "div_ok", cgen_state_->current_func_);
  if (!null_check_suffix.empty()) {
    codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, div_ok, ti);
  }
  auto div_zero = llvm::BasicBlock::Create(
      cgen_state_->context_, "div_zero", cgen_state_->current_func_);
  auto zero_const = rhs_lv->getType()->isIntegerTy()
                        ? llvm::ConstantInt::get(rhs_lv->getType(), 0, true)
                        : llvm::ConstantFP::get(rhs_lv->getType(), 0.);
  cgen_state_->ir_builder_.CreateCondBr(
      zero_const->getType()->isFloatingPointTy()
          ? cgen_state_->ir_builder_.CreateFCmp(
                llvm::FCmpInst::FCMP_ONE, rhs_lv, zero_const)
          : cgen_state_->ir_builder_.CreateICmp(
                llvm::ICmpInst::ICMP_NE, rhs_lv, zero_const),
      div_ok,
      div_zero);
  cgen_state_->ir_builder_.SetInsertPoint(div_ok);
  auto ret =
      zero_const->getType()->isIntegerTy()
          ? (null_typename.empty()
                 ? cgen_state_->ir_builder_.CreateSDiv(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "div_" + null_typename + null_check_suffix,
                       {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_val(ti))}))
          : (null_typename.empty()
                 ? cgen_state_->ir_builder_.CreateFDiv(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "div_" + null_typename + null_check_suffix,
                       {lhs_lv,
                        rhs_lv,
                        ti.get_type() == kFLOAT ? cgen_state_->llFp(NULL_FLOAT)
                                                : cgen_state_->llFp(NULL_DOUBLE)}));
  cgen_state_->ir_builder_.SetInsertPoint(div_zero);
  cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(int32_t(ErrorCode::DIV_BY_ZERO)));
  cgen_state_->ir_builder_.SetInsertPoint(div_ok);
  return ret;
}

namespace {
template <size_t N>
// Return nullptr iff all N types are not nullable.
// Return llvm::Value* for true iff any of the N values are null.
// Assumes Decimal/Integer types.  Please add additional types as needed.
llvm::Value* codegen_null_checks(CgenState* const cgen_state,
                                 std::array<SQLTypeInfo const*, N> const types,
                                 std::array<llvm::Value*, N> const values) {
  llvm::Value* any_null_lv{nullptr};
  for (size_t i = 0; i < N; ++i) {
    if (!types[i]->get_notnull()) {
      auto* null_lv = cgen_state->llInt(inline_int_null_val(*types[i]));
      auto* is_null_lv =
          cgen_state->ir_builder_.CreateICmpEQ(values[i], null_lv, "is_null");
      any_null_lv = any_null_lv ? cgen_state->ir_builder_.CreateOr(
                                      any_null_lv, is_null_lv, "any_null")
                                : is_null_lv;
    }
  }
  return any_null_lv;
}
}  // namespace

/**
 * Handle decimal / decimal division.
 *
 * Outline of steps:
 * * Check if the upscale multiplication can be elided. If so, then
 *   return codegenDiv() with upscale=false, skipping any 128-bit arithmetic.
 * * Define BasicBlocks.
 * * Check for NULL operands.
 * * Check if denominator is 0. If it is, then either:
 *   * Set result to NULL if g_null_div_by_zero, otherwise
 *   * Return ErrorCode::DIV_BY_ZERO.
 * * Handle typical case by calling decimal_division(). This calculates
 *   lhs * pow10 / rhs where pow10=10^9 in the typical DECIMAL(19,9) case.
 *   Use a custom Uint128 class to do the arithmetic.
 *   If result is greater than 64 bits then return ErrorCode::OVERFLOW_OR_UNDERFLOW.
 * * Final PHINode combines prior branches to the final value.
 */
llvm::Value* CodeGenerator::codegenDeciDiv(const Analyzer::BinOper* bin_oper,
                                           const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  Analyzer::Expr const* const lhs = bin_oper->get_left_operand();
  Analyzer::Expr const* const rhs = bin_oper->get_right_operand();
  SQLTypeInfo const& lhs_type = lhs->get_type_info();
  SQLTypeInfo const& rhs_type = rhs->get_type_info();
  CHECK(lhs_type.is_decimal() && rhs_type.is_decimal() &&
        lhs_type.get_scale() == rhs_type.get_scale());

  constexpr bool fetch_columns = true;
  auto* lhs_lv = codegen(lhs, fetch_columns, co).front();

  // Check if upscale multiplication can be elided. Skip 128-bit arithmetic if so.
  if (auto* rhs_lv = codegenDivisorWithoutUpscale(co, lhs_type, rhs)) {
    constexpr bool upscale = false;
    const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
    const auto int_typename =
        null_check_suffix.empty()
            ? ""
            : numeric_or_time_interval_type_name(lhs_type, rhs_type);
    return codegenDiv(lhs_lv, rhs_lv, int_typename, null_check_suffix, lhs_type, upscale);
  }
  auto* rhs_lv = codegen(rhs, fetch_columns, co).front();

  // Define BasicBlocks.
  auto* done_bb =
      llvm::BasicBlock::Create(cgen_state_->context_, "done", cgen_state_->current_func_);
  auto* check_rhs_zero_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "check_rhs_zero", cgen_state_->current_func_);
  auto* overflow_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "overflow", cgen_state_->current_func_);
  cgen_state_->needs_error_check_ = true;
  llvm::BasicBlock* phi_src0_bb{nullptr};  // First phi-incoming basic block below.

  // Codegen NULL checks. Any NULL ? ->done_bb : ->check_rhs_zero_bb.
  if (auto* is_null_lv = codegen_null_checks<2u>(
          cgen_state_, {&lhs_type, &rhs_type}, {lhs_lv, rhs_lv})) {
    cgen_state_->ir_builder_.CreateCondBr(is_null_lv, done_bb, check_rhs_zero_bb);
    phi_src0_bb = cgen_state_->ir_builder_.GetInsertBlock();
  } else {
    // No operands can be null, so just branch to check_rhs_zero_bb.
    cgen_state_->ir_builder_.CreateBr(check_rhs_zero_bb);
  }

  // BasicBlock check_rhs_zero:
  // Check if denominator (rhs) is 0 and handle based on g_null_div_by_zero.
  cgen_state_->ir_builder_.SetInsertPoint(check_rhs_zero_bb);
  auto* zero_lv = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
  auto* is_rhs_zero_lv =
      cgen_state_->ir_builder_.CreateICmpEQ(rhs_lv, zero_lv, "is_rhs_zero");
  auto* rhs_zero_bb =
      g_null_div_by_zero
          ? done_bb
          : llvm::BasicBlock::Create(
                cgen_state_->context_, "rhs_zero", cgen_state_->current_func_);
  auto* rhs_nonzero_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "rhs_nonzero", cgen_state_->current_func_);
  cgen_state_->ir_builder_.CreateCondBr(is_rhs_zero_lv, rhs_zero_bb, rhs_nonzero_bb);

  // If denominator is 0 and !g_null_div_by_zero then return ErrorCode::DIV_BY_ZERO.
  if (!g_null_div_by_zero) {
    cgen_state_->ir_builder_.SetInsertPoint(rhs_zero_bb);
    cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(int(ErrorCode::DIV_BY_ZERO)));
  }

  // BasicBlock rhs_nonzero: Typical case with nonnull values and nonzero denominator.
  cgen_state_->ir_builder_.SetInsertPoint(rhs_nonzero_bb);
  auto* pow10_lv = cgen_state_->llInt(int64_t(shared::power10(lhs_type.get_scale())));
  auto* null_lv = cgen_state_->llInt(inline_int_null_val(lhs_type));
  char const* const func_name = co.device_type == ExecutorDeviceType::GPU
                                    ? "decimal_division_gpu"
                                    : "decimal_division";
  auto* result_lv = cgen_state_->emitCall(func_name, {lhs_lv, pow10_lv, rhs_lv, null_lv});
  auto* is_null_lv = cgen_state_->ir_builder_.CreateICmpEQ(result_lv, null_lv, "is_null");
  cgen_state_->ir_builder_.CreateCondBr(is_null_lv, overflow_bb, done_bb);

  // BasicBlock overflow: decimal_division() result is greater than 64 bits.
  cgen_state_->ir_builder_.SetInsertPoint(overflow_bb);
  cgen_state_->ir_builder_.CreateRet(
      cgen_state_->llInt(int(ErrorCode::OVERFLOW_OR_UNDERFLOW)));

  // BasicBlock done: Set return value based on phi value
  cgen_state_->ir_builder_.SetInsertPoint(done_bb);
  unsigned const num_inputs = bool(phi_src0_bb) + !g_null_div_by_zero + 1u;
  llvm::PHINode* phi =
      cgen_state_->ir_builder_.CreatePHI(lhs_lv->getType(), num_inputs, "phi");
  if (phi_src0_bb) {
    phi->addIncoming(null_lv, phi_src0_bb);  // an operand is NULL
  }
  if (g_null_div_by_zero) {
    phi->addIncoming(null_lv, check_rhs_zero_bb);  // NULL due to g_null_div_by_zero
  }
  phi->addIncoming(result_lv, rhs_nonzero_bb);  // Non-NULL result of division
  return phi;
}

// When converting a value into a DECIMAL(precision,scale) type, the value is multiplied
// by 10^scale. In the general case when dividing two decimal int64_t values, the
// numerator is first multiplied by 10^scale. This is called "upscale". However if the
// denominator is also a multiple of 10^scale, (which occurs when converting from INT to
// DECIMAL, for instance) then the upscale multiplication can be elided and the 128-bit
// arithmetic can be skipped.
//
// If the upscale multiplication can be elided, then return the rhs_lv value to be used
// for the denominator of the division. Otherwise return nullptr.
llvm::Value* CodeGenerator::codegenDivisorWithoutUpscale(
    CompilationOptions const& co,
    SQLTypeInfo const& lhs_type,
    Analyzer::Expr const* const rhs) {
  constexpr bool upscale = false;
  if (auto* rhs_constant = dynamic_cast<Analyzer::Constant const*>(rhs)) {
    if (!rhs_constant->get_is_null() && rhs_constant->get_constval().bigintval) {
      int64_t const pow10 = int64_t(shared::power10(rhs->get_type_info().get_scale()));
      auto const div = std::div(rhs_constant->get_constval().bigintval, pow10);
      if (div.rem == 0) {
        auto rhs_lit = Parser::IntLiteral::analyzeValue(div.quot);
        auto* rhs_lit_lv = CodeGenerator::codegenIntConst(
            dynamic_cast<Analyzer::Constant const*>(rhs_lit.get()), cgen_state_);
        return codegenCastBetweenIntTypes(
            rhs_lit_lv, rhs_lit->get_type_info(), lhs_type, upscale);
      }
    }
  }
  if (auto* rhs_cast = dynamic_cast<Analyzer::UOper const*>(rhs)) {
    if (rhs_cast->get_optype() == kCAST &&
        rhs_cast->get_operand()->get_type_info().is_integer()) {
      Analyzer::Expr const* rhs_cast_oper = rhs_cast->get_operand();
      constexpr bool fetch_columns = true;
      auto* rhs_cast_oper_lv = codegen(rhs_cast_oper, fetch_columns, co).front();
      return codegenCastBetweenIntTypes(
          rhs_cast_oper_lv, rhs_cast_oper->get_type_info(), lhs_type, upscale);
    }
  }
  return nullptr;
}

llvm::Value* CodeGenerator::codegenMod(llvm::Value* lhs_lv,
                                       llvm::Value* rhs_lv,
                                       const std::string& null_typename,
                                       const std::string& null_check_suffix,
                                       const SQLTypeInfo& ti) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(lhs_lv->getType(), rhs_lv->getType());
  CHECK(ti.is_integer());
  cgen_state_->needs_error_check_ = true;
  // Generate control flow for division by zero error handling.
  auto mod_ok = llvm::BasicBlock::Create(
      cgen_state_->context_, "mod_ok", cgen_state_->current_func_);
  auto mod_zero = llvm::BasicBlock::Create(
      cgen_state_->context_, "mod_zero", cgen_state_->current_func_);
  auto zero_const = llvm::ConstantInt::get(rhs_lv->getType(), 0, true);
  cgen_state_->ir_builder_.CreateCondBr(
      cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_NE, rhs_lv, zero_const),
      mod_ok,
      mod_zero);
  cgen_state_->ir_builder_.SetInsertPoint(mod_ok);
  auto ret = null_typename.empty()
                 ? cgen_state_->ir_builder_.CreateSRem(lhs_lv, rhs_lv)
                 : cgen_state_->emitCall(
                       "mod_" + null_typename + null_check_suffix,
                       {lhs_lv, rhs_lv, cgen_state_->llInt(inline_int_null_val(ti))});
  cgen_state_->ir_builder_.SetInsertPoint(mod_zero);
  cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(int32_t(ErrorCode::DIV_BY_ZERO)));
  cgen_state_->ir_builder_.SetInsertPoint(mod_ok);
  return ret;
}

// Returns true iff runtime overflow checks aren't needed thanks to range information.
bool CodeGenerator::checkExpressionRanges(const Analyzer::UOper* uoper,
                                          int64_t min,
                                          int64_t max) {
  if (uoper->get_type_info().is_decimal()) {
    return false;
  }

  CHECK(plan_state_);
  if (executor_) {
    auto expr_range_info =
        plan_state_->query_infos_.size() > 0
            ? getExpressionRange(uoper, plan_state_->query_infos_, executor())
            : ExpressionRange::makeInvalidRange();
    if (expr_range_info.getType() != ExpressionRangeType::Integer) {
      return false;
    }
    if (expr_range_info.getIntMin() >= min && expr_range_info.getIntMax() <= max) {
      return true;
    }
  }

  return false;
}

llvm::Value* CodeGenerator::codegenUMinus(const Analyzer::UOper* uoper,
                                          const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_EQ(uoper->get_optype(), kUMINUS);
  const auto operand_lv = codegen(uoper->get_operand(), true, co).front();
  const auto& ti = uoper->get_type_info();
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  bool need_overflow_check = false;
  if (ti.is_integer() || ti.is_decimal() || ti.is_timeinterval()) {
    std::tie(chosen_max, chosen_min) = cgen_state_->inlineIntMaxMin(ti.get_size(), true);
    need_overflow_check = !checkExpressionRanges(
        uoper,
        static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
        static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue());
  }
  llvm::BasicBlock* uminus_ok{nullptr};
  llvm::BasicBlock* uminus_fail{nullptr};
  if (need_overflow_check) {
    cgen_state_->needs_error_check_ = true;
    uminus_ok = llvm::BasicBlock::Create(
        cgen_state_->context_, "uminus_ok", cgen_state_->current_func_);
    if (!ti.get_notnull()) {
      codegenSkipOverflowCheckForNull(operand_lv, nullptr, uminus_ok, ti);
    }
    uminus_fail = llvm::BasicBlock::Create(
        cgen_state_->context_, "uminus_fail", cgen_state_->current_func_);
    auto const_min = llvm::ConstantInt::get(
        operand_lv->getType(),
        static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue(),
        true);
    auto overflow = cgen_state_->ir_builder_.CreateICmpEQ(operand_lv, const_min);
    cgen_state_->ir_builder_.CreateCondBr(overflow, uminus_fail, uminus_ok);
    cgen_state_->ir_builder_.SetInsertPoint(uminus_ok);
  }
  auto ret =
      ti.get_notnull()
          ? (ti.is_fp() ? cgen_state_->ir_builder_.CreateFNeg(operand_lv)
                        : cgen_state_->ir_builder_.CreateNeg(operand_lv))
          : cgen_state_->emitCall(
                "uminus_" + numeric_type_name(ti) + "_nullable",
                {operand_lv,
                 ti.is_fp() ? static_cast<llvm::Value*>(cgen_state_->inlineFpNull(ti))
                            : static_cast<llvm::Value*>(cgen_state_->inlineIntNull(ti))});
  if (need_overflow_check) {
    cgen_state_->ir_builder_.SetInsertPoint(uminus_fail);
    cgen_state_->ir_builder_.CreateRet(
        cgen_state_->llInt(int32_t(ErrorCode::OVERFLOW_OR_UNDERFLOW)));
    cgen_state_->ir_builder_.SetInsertPoint(uminus_ok);
  }
  return ret;
}

llvm::Function* CodeGenerator::getArithWithOverflowIntrinsic(
    const Analyzer::BinOper* bin_oper,
    llvm::Type* type) {
  llvm::Intrinsic::ID fn_id{llvm::Intrinsic::not_intrinsic};
  switch (bin_oper->get_optype()) {
    case kMINUS:
      fn_id = llvm::Intrinsic::ssub_with_overflow;
      break;
    case kPLUS:
      fn_id = llvm::Intrinsic::sadd_with_overflow;
      break;
    case kMULTIPLY:
      fn_id = llvm::Intrinsic::smul_with_overflow;
      break;
    default:
      LOG(FATAL) << "unexpected arith with overflow optype: " << bin_oper->toString();
  }

  return llvm::Intrinsic::getDeclaration(cgen_state_->module_, fn_id, type);
}

llvm::Value* CodeGenerator::codegenBinOpWithOverflowForCPU(
    const Analyzer::BinOper* bin_oper,
    llvm::Value* lhs_lv,
    llvm::Value* rhs_lv,
    const std::string& null_check_suffix,
    const SQLTypeInfo& ti) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  cgen_state_->needs_error_check_ = true;

  llvm::BasicBlock* check_ok = llvm::BasicBlock::Create(
      cgen_state_->context_, "ovf_ok", cgen_state_->current_func_);
  llvm::BasicBlock* check_fail = llvm::BasicBlock::Create(
      cgen_state_->context_, "ovf_detected", cgen_state_->current_func_);
  llvm::BasicBlock* null_check{nullptr};

  if (!null_check_suffix.empty()) {
    null_check = cgen_state_->ir_builder_.GetInsertBlock();
    codegenSkipOverflowCheckForNull(lhs_lv, rhs_lv, check_ok, ti);
  }

  // Compute result and overflow flag
  auto func = getArithWithOverflowIntrinsic(bin_oper, lhs_lv->getType());
  auto ret_and_overflow = cgen_state_->ir_builder_.CreateCall(
      func, std::vector<llvm::Value*>{lhs_lv, rhs_lv});
  auto ret = cgen_state_->ir_builder_.CreateExtractValue(ret_and_overflow,
                                                         std::vector<unsigned>{0});
  auto overflow = cgen_state_->ir_builder_.CreateExtractValue(ret_and_overflow,
                                                              std::vector<unsigned>{1});
  auto val_bb = cgen_state_->ir_builder_.GetInsertBlock();

  // Return error on overflow
  cgen_state_->ir_builder_.CreateCondBr(overflow, check_fail, check_ok);
  cgen_state_->ir_builder_.SetInsertPoint(check_fail);
  cgen_state_->ir_builder_.CreateRet(
      cgen_state_->llInt(int32_t(ErrorCode::OVERFLOW_OR_UNDERFLOW)));

  cgen_state_->ir_builder_.SetInsertPoint(check_ok);

  // In case of null check we have to use NULL result on check fail
  if (null_check) {
    auto phi = cgen_state_->ir_builder_.CreatePHI(ret->getType(), 2);
    phi->addIncoming(llvm::ConstantInt::get(ret->getType(), inline_int_null_val(ti)),
                     null_check);
    phi->addIncoming(ret, val_bb);
    ret = phi;
  }

  return ret;
}
