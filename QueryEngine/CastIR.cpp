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

llvm::Value* Executor::codegenCast(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  CHECK_EQ(uoper->get_optype(), kCAST);
  const auto& ti = uoper->get_type_info();
  const auto operand = uoper->get_operand();
  const auto operand_as_const = dynamic_cast<const Analyzer::Constant*>(operand);
  // For dictionary encoded constants, the cast holds the dictionary id
  // information as the compression parameter; handle this case separately.
  llvm::Value* operand_lv{nullptr};
  if (operand_as_const) {
    const auto operand_lvs = codegen(operand_as_const, ti.get_compression(), ti.get_comp_param(), co);
    if (operand_lvs.size() == 3) {
      operand_lv = cgen_state_->emitCall("string_pack", {operand_lvs[1], operand_lvs[2]});
    } else {
      operand_lv = operand_lvs.front();
    }
  } else {
    operand_lv = codegen(operand, true, co).front();
  }
  const auto& operand_ti = operand->get_type_info();
  return codegenCast(operand_lv, operand_ti, ti, operand_as_const, co);
}

llvm::Value* Executor::codegenCast(llvm::Value* operand_lv,
                                   const SQLTypeInfo& operand_ti,
                                   const SQLTypeInfo& ti,
                                   const bool operand_is_const,
                                   const CompilationOptions& co) {
  if (operand_lv->getType()->isIntegerTy()) {
    if (operand_ti.is_string()) {
      return codegenCastFromString(operand_lv, operand_ti, ti, operand_is_const, co);
    }
    CHECK(operand_ti.is_integer() || operand_ti.is_decimal() || operand_ti.is_time() || operand_ti.is_boolean());
    if (operand_ti.is_boolean()) {
      CHECK(operand_lv->getType()->isIntegerTy(1) || operand_lv->getType()->isIntegerTy(8));
      if (operand_lv->getType()->isIntegerTy(1)) {
        operand_lv = castToTypeIn(operand_lv, 8);
      }
    }
    if (operand_ti.get_type() == kTIMESTAMP && ti.get_type() == kDATE) {
      // Maybe we should instead generate DatetruncExpr directly from RelAlgTranslator
      // for this pattern. However, DatetruncExpr is supposed to return a timestamp,
      // whereas this cast returns a date. The underlying type for both is still the same,
      // but it still doesn't look like a good idea to misuse DatetruncExpr.
      return codegenCastTimestampToDate(operand_lv, !ti.get_notnull());
    }
    if (ti.is_integer() || ti.is_decimal() || ti.is_time()) {
      return codegenCastBetweenIntTypes(operand_lv, operand_ti, ti);
    } else {
      return codegenCastToFp(operand_lv, operand_ti, ti);
    }
  } else {
    return codegenCastFromFp(operand_lv, operand_ti, ti);
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* Executor::codegenCastTimestampToDate(llvm::Value* ts_lv, const bool nullable) {
  static_assert(sizeof(time_t) == 4 || sizeof(time_t) == 8, "Unsupported time_t size");
  CHECK(ts_lv->getType()->isIntegerTy(32) || ts_lv->getType()->isIntegerTy(64));
  if (sizeof(time_t) == 4 && ts_lv->getType()->isIntegerTy(64)) {
    ts_lv = cgen_state_->ir_builder_.CreateCast(
        llvm::Instruction::CastOps::Trunc, ts_lv, get_int_type(32, cgen_state_->context_));
  }
  std::vector<llvm::Value*> datetrunc_args{ll_int(static_cast<int32_t>(dtDAY)), ts_lv};
  std::string datetrunc_fname{"DateTruncate"};
  if (nullable) {
    datetrunc_args.push_back(inlineIntNull(SQLTypeInfo(ts_lv->getType()->isIntegerTy(64) ? kBIGINT : kINT, false)));
    datetrunc_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(datetrunc_fname, get_int_type(64, cgen_state_->context_), datetrunc_args);
}

llvm::Value* Executor::codegenCastFromString(llvm::Value* operand_lv,
                                             const SQLTypeInfo& operand_ti,
                                             const SQLTypeInfo& ti,
                                             const bool operand_is_const,
                                             const CompilationOptions& co) {
  if (!ti.is_string()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " + ti.get_type_name() +
                             " not supported");
  }
  if (operand_ti.get_compression() == kENCODING_NONE && ti.get_compression() == kENCODING_NONE) {
    return operand_lv;
  }
  // dictionary encode non-constant
  if (operand_ti.get_compression() != kENCODING_DICT && !operand_is_const) {
    if (g_cluster) {
      throw std::runtime_error(
          "Cast from none-encoded string to dictionary-encoded not supported for distributed queries");
    }
    if (g_enable_watchdog) {
      throw WatchdogException("Cast from none-encoded string to dictionary-encoded would be slow");
    }
    CHECK_EQ(kENCODING_NONE, operand_ti.get_compression());
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    CHECK(operand_lv->getType()->isIntegerTy(64));
    if (co.device_type_ == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    return cgen_state_->emitExternalCall(
        "string_compress",
        get_int_type(32, cgen_state_->context_),
        {operand_lv, ll_int(int64_t(getStringDictionaryProxy(ti.get_comp_param(), row_set_mem_owner_, true)))});
  }
  CHECK(operand_lv->getType()->isIntegerTy(32));
  if (ti.get_compression() == kENCODING_NONE) {
    if (g_cluster) {
      throw std::runtime_error(
          "Cast from dictionary-encoded string to none-encoded not supported for distributed queries");
    }
    if (g_enable_watchdog) {
      throw WatchdogException("Cast from dictionary-encoded string to none-encoded would be slow");
    }
    CHECK_EQ(kENCODING_DICT, operand_ti.get_compression());
    if (co.device_type_ == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    return cgen_state_->emitExternalCall(
        "string_decompress",
        get_int_type(64, cgen_state_->context_),
        {operand_lv, ll_int(int64_t(getStringDictionaryProxy(operand_ti.get_comp_param(), row_set_mem_owner_, true)))});
  }
  CHECK(operand_is_const);
  CHECK_EQ(kENCODING_DICT, ti.get_compression());
  return operand_lv;
}

llvm::Value* Executor::codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                                  const SQLTypeInfo& operand_ti,
                                                  const SQLTypeInfo& ti,
                                                  bool upscale) {
  if (ti.is_decimal()) {
    if (upscale) {
      CHECK(!operand_ti.is_decimal() || operand_ti.get_scale() <= ti.get_scale());
      if (operand_ti.get_scale() < ti.get_scale()) {  // scale only if needed
        auto scale = exp_to_scale(ti.get_scale() - operand_ti.get_scale());
        const auto scale_lv = llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), scale);
        operand_lv = cgen_state_->ir_builder_.CreateSExt(operand_lv, get_int_type(64, cgen_state_->context_));

        llvm::Value* chosen_max{nullptr};
        llvm::Value* chosen_min{nullptr};
        std::tie(chosen_max, chosen_min) = inlineIntMaxMin(8, true);

        cgen_state_->needs_error_check_ = true;
        auto cast_to_decimal_ok =
            llvm::BasicBlock::Create(cgen_state_->context_, "cast_to_decimal_ok", cgen_state_->row_func_);
        auto cast_to_decimal_fail =
            llvm::BasicBlock::Create(cgen_state_->context_, "cast_to_decimal_fail", cgen_state_->row_func_);
        auto operand_max = static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue() / scale;
        auto operand_max_lv = llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), operand_max);
        llvm::Value* detected{nullptr};
        if (operand_ti.get_notnull()) {
          detected = cgen_state_->ir_builder_.CreateICmpSGT(operand_lv, operand_max_lv);
        } else {
          detected = toBool(cgen_state_->emitCall("gt_" + numeric_type_name(ti) + "_nullable_lhs",
                                                  {operand_lv,
                                                   operand_max_lv,
                                                   ll_int(inline_int_null_val(operand_ti)),
                                                   inlineIntNull(SQLTypeInfo(kBOOLEAN, false))}));
        }
        cgen_state_->ir_builder_.CreateCondBr(detected, cast_to_decimal_fail, cast_to_decimal_ok);

        cgen_state_->ir_builder_.SetInsertPoint(cast_to_decimal_fail);
        cgen_state_->ir_builder_.CreateRet(ll_int(ERR_OVERFLOW_OR_UNDERFLOW));

        cgen_state_->ir_builder_.SetInsertPoint(cast_to_decimal_ok);

        if (operand_ti.get_notnull()) {
          operand_lv = cgen_state_->ir_builder_.CreateMul(operand_lv, scale_lv);
        } else {
          operand_lv = cgen_state_->emitCall("scale_decimal",
                                             {operand_lv,
                                              scale_lv,
                                              ll_int(inline_int_null_val(operand_ti)),
                                              inlineIntNull(SQLTypeInfo(kBIGINT, false))});
        }
      }
    }
  } else if (operand_ti.is_decimal()) {
    const auto scale_lv = llvm::ConstantInt::get(static_cast<llvm::IntegerType*>(operand_lv->getType()),
                                                 exp_to_scale(operand_ti.get_scale()));
    operand_lv = cgen_state_->emitCall("div_" + numeric_type_name(operand_ti) + "_nullable_lhs",
                                       {operand_lv, scale_lv, ll_int(inline_int_null_val(operand_ti))});
  }
  const auto operand_width = static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();
  const auto target_width = get_bit_width(ti);
  if (target_width == operand_width) {
    return operand_lv;
  }
  if (operand_ti.get_notnull()) {
    return cgen_state_->ir_builder_.CreateCast(
        target_width > operand_width ? llvm::Instruction::CastOps::SExt : llvm::Instruction::CastOps::Trunc,
        operand_lv,
        get_int_type(target_width, cgen_state_->context_));
  }
  return cgen_state_->emitCall("cast_" + numeric_type_name(operand_ti) + "_to_" + numeric_type_name(ti) + "_nullable",
                               {operand_lv, inlineIntNull(operand_ti), inlineIntNull(ti)});
}

llvm::Value* Executor::codegenCastToFp(llvm::Value* operand_lv, const SQLTypeInfo& operand_ti, const SQLTypeInfo& ti) {
  if (!ti.is_fp()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " + ti.get_type_name() +
                             " not supported");
  }
  const auto to_tname = numeric_type_name(ti);
  llvm::Value* result_lv{nullptr};
  if (operand_ti.get_notnull()) {
    result_lv =
        cgen_state_->ir_builder_.CreateSIToFP(operand_lv,
                                              ti.get_type() == kFLOAT ? llvm::Type::getFloatTy(cgen_state_->context_)
                                                                      : llvm::Type::getDoubleTy(cgen_state_->context_));
  } else {
    result_lv = cgen_state_->emitCall("cast_" + numeric_type_name(operand_ti) + "_to_" + to_tname + "_nullable",
                                      {operand_lv, inlineIntNull(operand_ti), inlineFpNull(ti)});
  }
  CHECK(result_lv);
  if (operand_ti.get_scale()) {
    result_lv = cgen_state_->ir_builder_.CreateFDiv(
        result_lv, llvm::ConstantFP::get(result_lv->getType(), exp_to_scale(operand_ti.get_scale())));
  }
  return result_lv;
}

llvm::Value* Executor::codegenCastFromFp(llvm::Value* operand_lv,
                                         const SQLTypeInfo& operand_ti,
                                         const SQLTypeInfo& ti) {
  if (!operand_ti.is_fp() || !ti.is_number() || ti.is_decimal()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " + ti.get_type_name() +
                             " not supported");
  }
  if (operand_ti.get_type() == ti.get_type()) {
    return operand_lv;
  }
  CHECK(operand_lv->getType()->isFloatTy() || operand_lv->getType()->isDoubleTy());
  if (operand_ti.get_notnull()) {
    if (ti.get_type() == kDOUBLE) {
      return cgen_state_->ir_builder_.CreateFPExt(operand_lv, llvm::Type::getDoubleTy(cgen_state_->context_));
    } else if (ti.get_type() == kFLOAT) {
      return cgen_state_->ir_builder_.CreateFPTrunc(operand_lv, llvm::Type::getFloatTy(cgen_state_->context_));
    } else if (ti.is_integer()) {
      return cgen_state_->ir_builder_.CreateFPToSI(operand_lv, get_int_type(get_bit_width(ti), cgen_state_->context_));
    } else {
      CHECK(false);
    }
  } else {
    const auto from_tname = numeric_type_name(operand_ti);
    const auto to_tname = numeric_type_name(ti);
    if (ti.is_fp()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv, inlineFpNull(operand_ti), inlineFpNull(ti)});
    } else if (ti.is_integer()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv, inlineFpNull(operand_ti), inlineIntNull(ti)});
    } else {
      CHECK(false);
    }
  }
  CHECK(false);
  return nullptr;
}
