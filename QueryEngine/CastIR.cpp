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

#include "CodeGenerator.h"
#include "Execute.h"

llvm::Value* CodeGenerator::codegenCast(const Analyzer::UOper* uoper,
                                        const CompilationOptions& co) {
  CHECK_EQ(uoper->get_optype(), kCAST);
  const auto& ti = uoper->get_type_info();
  const auto operand = uoper->get_operand();
  const auto operand_as_const = dynamic_cast<const Analyzer::Constant*>(operand);
  // For dictionary encoded constants, the cast holds the dictionary id
  // information as the compression parameter; handle this case separately.
  llvm::Value* operand_lv{nullptr};
  if (operand_as_const) {
    const auto operand_lvs =
        codegen(operand_as_const, ti.get_compression(), ti.get_comp_param(), co);
    if (operand_lvs.size() == 3) {
      operand_lv = cgen_state_->emitCall("string_pack", {operand_lvs[1], operand_lvs[2]});
    } else {
      operand_lv = operand_lvs.front();
    }
  } else {
    operand_lv = executor_->codegen(operand, true, co).front();
  }
  const auto& operand_ti = operand->get_type_info();
  return codegenCast(operand_lv, operand_ti, ti, operand_as_const, co);
}

llvm::Value* CodeGenerator::codegenCast(llvm::Value* operand_lv,
                                        const SQLTypeInfo& operand_ti,
                                        const SQLTypeInfo& ti,
                                        const bool operand_is_const,
                                        const CompilationOptions& co) {
  if (operand_lv->getType()->isIntegerTy()) {
    if (operand_ti.is_string()) {
      return codegenCastFromString(operand_lv, operand_ti, ti, operand_is_const, co);
    }
    CHECK(operand_ti.is_integer() || operand_ti.is_decimal() || operand_ti.is_time() ||
          operand_ti.is_boolean());
    if (operand_ti.is_boolean()) {
      CHECK(operand_lv->getType()->isIntegerTy(1) ||
            operand_lv->getType()->isIntegerTy(8));
      if (operand_lv->getType()->isIntegerTy(1)) {
        operand_lv = executor_->castToTypeIn(operand_lv, 8);
      }
    }
    if (operand_ti.get_type() == kTIMESTAMP && ti.get_type() == kDATE) {
      // Maybe we should instead generate DatetruncExpr directly from RelAlgTranslator
      // for this pattern. However, DatetruncExpr is supposed to return a timestamp,
      // whereas this cast returns a date. The underlying type for both is still the same,
      // but it still doesn't look like a good idea to misuse DatetruncExpr.
      // Date will have default precision of day, but TIMESTAMP dimension would
      // matter but while converting date through seconds
      return codegenCastTimestampToDate(
          operand_lv, operand_ti.get_dimension(), !ti.get_notnull());
    }
    if ((operand_ti.get_type() == kTIMESTAMP || operand_ti.get_type() == kDATE) &&
        ti.get_type() == kTIMESTAMP) {
      const auto operand_dimen =
          (operand_ti.is_timestamp()) ? operand_ti.get_dimension() : 0;
      if (operand_dimen != ti.get_dimension()) {
        return codegenCastBetweenTimestamps(
            operand_lv, operand_dimen, ti.get_dimension(), !ti.get_notnull());
      }
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

llvm::Value* CodeGenerator::codegenCastTimestampToDate(llvm::Value* ts_lv,
                                                       const int dimen,
                                                       const bool nullable) {
  CHECK(ts_lv->getType()->isIntegerTy(64));
  std::vector<llvm::Value*> datetrunc_args{ts_lv};
  if (dimen > 0) {
    static const std::string hptodate_fname = "DateTruncateHighPrecisionToDate";
    static const std::string hptodate_null_fname =
        "DateTruncateHighPrecisionToDateNullable";
    datetrunc_args.push_back(
        executor_->ll_int(DateTimeUtils::get_timestamp_precision_scale(dimen)));
    if (nullable) {
      datetrunc_args.push_back(executor_->inlineIntNull(SQLTypeInfo(kBIGINT, false)));
    }
    return cgen_state_->emitExternalCall(nullable ? hptodate_null_fname : hptodate_fname,
                                         get_int_type(64, cgen_state_->context_),
                                         datetrunc_args);
  }
  std::string datetrunc_fname{"DateTruncate"};
  datetrunc_args.insert(datetrunc_args.begin(),
                        executor_->ll_int(static_cast<int32_t>(dtDAY)));
  if (nullable) {
    datetrunc_args.push_back(executor_->inlineIntNull(SQLTypeInfo(kBIGINT, false)));
    datetrunc_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      datetrunc_fname, get_int_type(64, cgen_state_->context_), datetrunc_args);
}

llvm::Value* CodeGenerator::codegenCastBetweenTimestamps(llvm::Value* ts_lv,
                                                         const int operand_dimen,
                                                         const int target_dimen,
                                                         const bool nullable) {
  if (operand_dimen == target_dimen) {
    return ts_lv;
  }
  CHECK(ts_lv->getType()->isIntegerTy(64));
  static const std::string sup_fname{"DateTruncateAlterPrecisionScaleUp"};
  static const std::string sdn_fname{"DateTruncateAlterPrecisionScaleDown"};
  static const std::string sup_null_fname{"DateTruncateAlterPrecisionScaleUpNullable"};
  static const std::string sdn_null_fname{"DateTruncateAlterPrecisionScaleDownNullable"};
  std::vector<llvm::Value*> f_args{
      ts_lv,
      executor_->ll_int(static_cast<int64_t>(DateTimeUtils::get_timestamp_precision_scale(
          abs(operand_dimen - target_dimen))))};
  if (nullable) {
    f_args.push_back(executor_->inlineIntNull(SQLTypeInfo(kBIGINT, false)));
  }
  return operand_dimen < target_dimen
             ? cgen_state_->emitExternalCall(nullable ? sup_null_fname : sup_fname,
                                             get_int_type(64, cgen_state_->context_),
                                             f_args)
             : cgen_state_->emitExternalCall(nullable ? sdn_null_fname : sdn_fname,
                                             get_int_type(64, cgen_state_->context_),
                                             f_args);
}

llvm::Value* CodeGenerator::codegenCastFromString(llvm::Value* operand_lv,
                                                  const SQLTypeInfo& operand_ti,
                                                  const SQLTypeInfo& ti,
                                                  const bool operand_is_const,
                                                  const CompilationOptions& co) {
  if (!ti.is_string()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " +
                             ti.get_type_name() + " not supported");
  }
  if (operand_ti.get_compression() == kENCODING_NONE &&
      ti.get_compression() == kENCODING_NONE) {
    return operand_lv;
  }
  // dictionary encode non-constant
  if (operand_ti.get_compression() != kENCODING_DICT && !operand_is_const) {
    if (g_cluster) {
      throw std::runtime_error(
          "Cast from none-encoded string to dictionary-encoded not supported for "
          "distributed queries");
    }
    if (g_enable_watchdog) {
      throw WatchdogException(
          "Cast from none-encoded string to dictionary-encoded would be slow");
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
        {operand_lv,
         executor_->ll_int(int64_t(executor_->getStringDictionaryProxy(
             ti.get_comp_param(), executor_->getRowSetMemoryOwner(), true)))});
  }
  CHECK(operand_lv->getType()->isIntegerTy(32));
  if (ti.get_compression() == kENCODING_NONE) {
    if (g_cluster) {
      throw std::runtime_error(
          "Cast from dictionary-encoded string to none-encoded not supported for "
          "distributed queries");
    }
    if (g_enable_watchdog) {
      throw WatchdogException(
          "Cast from dictionary-encoded string to none-encoded would be slow");
    }
    CHECK_EQ(kENCODING_DICT, operand_ti.get_compression());
    if (co.device_type_ == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    return cgen_state_->emitExternalCall(
        "string_decompress",
        get_int_type(64, cgen_state_->context_),
        {operand_lv,
         executor_->ll_int(int64_t(executor_->getStringDictionaryProxy(
             operand_ti.get_comp_param(), executor_->getRowSetMemoryOwner(), true)))});
  }
  CHECK(operand_is_const);
  CHECK_EQ(kENCODING_DICT, ti.get_compression());
  return operand_lv;
}

llvm::Value* CodeGenerator::codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                                       const SQLTypeInfo& operand_ti,
                                                       const SQLTypeInfo& ti,
                                                       bool upscale) {
  if (ti.is_decimal() &&
      (!operand_ti.is_decimal() || operand_ti.get_scale() <= ti.get_scale())) {
    if (upscale) {
      if (operand_ti.get_scale() < ti.get_scale()) {  // scale only if needed
        auto scale = exp_to_scale(ti.get_scale() - operand_ti.get_scale());
        const auto scale_lv =
            llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), scale);
        operand_lv = cgen_state_->ir_builder_.CreateSExt(
            operand_lv, get_int_type(64, cgen_state_->context_));

        llvm::Value* chosen_max{nullptr};
        llvm::Value* chosen_min{nullptr};
        std::tie(chosen_max, chosen_min) = executor_->inlineIntMaxMin(8, true);

        cgen_state_->needs_error_check_ = true;
        auto cast_to_decimal_ok = llvm::BasicBlock::Create(
            cgen_state_->context_, "cast_to_decimal_ok", cgen_state_->row_func_);
        auto cast_to_decimal_fail = llvm::BasicBlock::Create(
            cgen_state_->context_, "cast_to_decimal_fail", cgen_state_->row_func_);
        auto operand_max =
            static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue() / scale;
        auto operand_max_lv =
            llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), operand_max);
        llvm::Value* detected{nullptr};
        if (operand_ti.get_notnull()) {
          detected = cgen_state_->ir_builder_.CreateICmpSGT(operand_lv, operand_max_lv);
        } else {
          detected = toBool(cgen_state_->emitCall(
              "gt_" + numeric_type_name(ti) + "_nullable_lhs",
              {operand_lv,
               operand_max_lv,
               executor_->ll_int(inline_int_null_val(operand_ti)),
               executor_->inlineIntNull(SQLTypeInfo(kBOOLEAN, false))}));
        }
        cgen_state_->ir_builder_.CreateCondBr(
            detected, cast_to_decimal_fail, cast_to_decimal_ok);

        cgen_state_->ir_builder_.SetInsertPoint(cast_to_decimal_fail);
        cgen_state_->ir_builder_.CreateRet(
            executor_->ll_int(Executor::ERR_OVERFLOW_OR_UNDERFLOW));

        cgen_state_->ir_builder_.SetInsertPoint(cast_to_decimal_ok);

        if (operand_ti.get_notnull()) {
          operand_lv = cgen_state_->ir_builder_.CreateMul(operand_lv, scale_lv);
        } else {
          operand_lv = cgen_state_->emitCall(
              "scale_decimal_up",
              {operand_lv,
               scale_lv,
               executor_->ll_int(inline_int_null_val(operand_ti)),
               executor_->inlineIntNull(SQLTypeInfo(kBIGINT, false))});
        }
      }
    }
  } else if (operand_ti.is_decimal()) {
    // rounded scale down
    auto scale = (int64_t)exp_to_scale(operand_ti.get_scale() - ti.get_scale());
    const auto scale_lv =
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), scale);

    const auto operand_width =
        static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();

    std::string method_name = "scale_decimal_down_nullable";
    if (operand_ti.get_notnull()) {
      method_name = "scale_decimal_down_not_nullable";
    }

    CHECK(operand_width == 64);
    operand_lv = cgen_state_->emitCall(
        method_name,
        {operand_lv, scale_lv, executor_->ll_int(inline_int_null_val(operand_ti))});
  }

  const auto operand_width =
      static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();
  const auto target_width = get_bit_width(ti);
  if (target_width == operand_width) {
    return operand_lv;
  }
  if (operand_ti.get_notnull()) {
    return cgen_state_->ir_builder_.CreateCast(
        target_width > operand_width ? llvm::Instruction::CastOps::SExt
                                     : llvm::Instruction::CastOps::Trunc,
        operand_lv,
        get_int_type(target_width, cgen_state_->context_));
  }
  return cgen_state_->emitCall(
      "cast_" + numeric_type_name(operand_ti) + "_to_" + numeric_type_name(ti) +
          "_nullable",
      {operand_lv, executor_->inlineIntNull(operand_ti), executor_->inlineIntNull(ti)});
}

llvm::Value* CodeGenerator::codegenCastToFp(llvm::Value* operand_lv,
                                            const SQLTypeInfo& operand_ti,
                                            const SQLTypeInfo& ti) {
  if (!ti.is_fp()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " +
                             ti.get_type_name() + " not supported");
  }
  const auto to_tname = numeric_type_name(ti);
  llvm::Value* result_lv{nullptr};
  if (operand_ti.get_notnull()) {
    result_lv = cgen_state_->ir_builder_.CreateSIToFP(
        operand_lv,
        ti.get_type() == kFLOAT ? llvm::Type::getFloatTy(cgen_state_->context_)
                                : llvm::Type::getDoubleTy(cgen_state_->context_));
  } else {
    result_lv = cgen_state_->emitCall(
        "cast_" + numeric_type_name(operand_ti) + "_to_" + to_tname + "_nullable",
        {operand_lv, executor_->inlineIntNull(operand_ti), executor_->inlineFpNull(ti)});
  }
  CHECK(result_lv);
  if (operand_ti.get_scale()) {
    result_lv = cgen_state_->ir_builder_.CreateFDiv(
        result_lv,
        llvm::ConstantFP::get(result_lv->getType(),
                              exp_to_scale(operand_ti.get_scale())));
  }
  return result_lv;
}

llvm::Value* CodeGenerator::codegenCastFromFp(llvm::Value* operand_lv,
                                              const SQLTypeInfo& operand_ti,
                                              const SQLTypeInfo& ti) {
  if (!operand_ti.is_fp() || !ti.is_number() || ti.is_decimal()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " +
                             ti.get_type_name() + " not supported");
  }
  if (operand_ti.get_type() == ti.get_type()) {
    // Should not have been called when both dimensions are same.
    return operand_lv;
  }
  CHECK(operand_lv->getType()->isFloatTy() || operand_lv->getType()->isDoubleTy());
  if (operand_ti.get_notnull()) {
    if (ti.get_type() == kDOUBLE) {
      return cgen_state_->ir_builder_.CreateFPExt(
          operand_lv, llvm::Type::getDoubleTy(cgen_state_->context_));
    } else if (ti.get_type() == kFLOAT) {
      return cgen_state_->ir_builder_.CreateFPTrunc(
          operand_lv, llvm::Type::getFloatTy(cgen_state_->context_));
    } else if (ti.is_integer()) {
      return cgen_state_->ir_builder_.CreateFPToSI(
          operand_lv, get_int_type(get_bit_width(ti), cgen_state_->context_));
    } else {
      CHECK(false);
    }
  } else {
    const auto from_tname = numeric_type_name(operand_ti);
    const auto to_tname = numeric_type_name(ti);
    if (ti.is_fp()) {
      return cgen_state_->emitCall(
          "cast_" + from_tname + "_to_" + to_tname + "_nullable",
          {operand_lv, executor_->inlineFpNull(operand_ti), executor_->inlineFpNull(ti)});
    } else if (ti.is_integer()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv,
                                    executor_->inlineFpNull(operand_ti),
                                    executor_->inlineIntNull(ti)});
    } else {
      CHECK(false);
    }
  }
  CHECK(false);
  return nullptr;
}
