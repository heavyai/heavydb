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
#include "StringDictionaryTranslationMgr.h"

llvm::Value* CodeGenerator::codegenCast(const Analyzer::UOper* uoper,
                                        const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
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
    operand_lv = codegen(operand, true, co).front();
  }

  // If the operand is a TextEncodingNone struct ({i8*, i64, i8})
  // unpack it into an int64_t using "string_pack" so that codegenCast
  // can properly cast it to a TextEncodingDict
  if (operand_lv->getType()->isPointerTy() &&
      operand_lv->getType()->getPointerElementType()->isStructTy()) {
    auto _struct = cgen_state_->ir_builder_.CreateLoad(
        operand_lv->getType()->getPointerElementType(), operand_lv);
    auto ptr = cgen_state_->ir_builder_.CreateExtractValue(_struct, {0});
    auto len = cgen_state_->ir_builder_.CreateTrunc(
        cgen_state_->ir_builder_.CreateExtractValue(_struct, {1}),
        get_int_type(32, cgen_state_->context_));
    operand_lv = cgen_state_->emitCall("string_pack", {ptr, len});
  }
  const auto& operand_ti = operand->get_type_info();
  return codegenCast(operand_lv, operand_ti, ti, operand_as_const, co);
}

namespace {

bool byte_array_cast(const SQLTypeInfo& operand_ti, const SQLTypeInfo& ti) {
  return (operand_ti.is_array() && ti.is_array() && ti.get_subtype() == kTINYINT &&
          operand_ti.get_size() > 0 && operand_ti.get_size() == ti.get_size());
}

}  // namespace

llvm::Value* CodeGenerator::codegenCast(llvm::Value* operand_lv,
                                        const SQLTypeInfo& operand_ti,
                                        const SQLTypeInfo& ti,
                                        const bool operand_is_const,
                                        const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (byte_array_cast(operand_ti, ti)) {
    auto* byte_array_type = get_int_array_type(8, ti.get_size(), cgen_state_->context_);
    return cgen_state_->ir_builder_.CreatePointerCast(operand_lv,
                                                      byte_array_type->getPointerTo());
  }
  if (operand_lv->getType()->isIntegerTy()) {
    if (operand_ti.is_string()) {
      return codegenCastFromString(operand_lv, operand_ti, ti, operand_is_const, co);
    }
    CHECK(operand_ti.is_integer() || operand_ti.is_decimal() || operand_ti.is_time() ||
          operand_ti.is_boolean());
    if (operand_ti.is_boolean()) {
      // cast boolean to int8
      CHECK(operand_lv->getType()->isIntegerTy(1) ||
            operand_lv->getType()->isIntegerTy(8));
      if (operand_lv->getType()->isIntegerTy(1)) {
        operand_lv = cgen_state_->castToTypeIn(operand_lv, 8);
      }
      if (ti.is_boolean()) {
        return operand_lv;
      }
    }
    if (operand_ti.is_integer() && operand_lv->getType()->isIntegerTy(8) &&
        ti.is_boolean()) {
      // cast int8 to boolean
      return codegenCastBetweenIntTypes(operand_lv, operand_ti, ti);
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
    if (operand_ti.get_type() == kTIMESTAMP && ti.get_type() == kTIME) {
      return codegenCastTimestampToTime(
          operand_lv, operand_ti.get_dimension(), !ti.get_notnull());
    }
    if ((operand_ti.get_type() == kTIMESTAMP || operand_ti.get_type() == kDATE) &&
        ti.get_type() == kTIMESTAMP) {
      const auto operand_dimen =
          (operand_ti.is_timestamp()) ? operand_ti.get_dimension() : 0;
      if (operand_dimen != ti.get_dimension()) {
        return codegenCastBetweenTimestamps(
            operand_lv, operand_ti, ti, !ti.get_notnull());
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

llvm::Value* CodeGenerator::codegenCastTimestampToTime(llvm::Value* ts_lv,
                                                       const int dimen,
                                                       const bool nullable) {
  std::vector<llvm::Value*> datetrunc_args{ts_lv};
  std::string hptodate_fname;
  if (dimen > 0) {
    hptodate_fname = "ExtractTimeFromHPTimestamp";
    datetrunc_args.push_back(
        cgen_state_->llInt(DateTimeUtils::get_timestamp_precision_scale(dimen)));
  } else {
    hptodate_fname = "ExtractTimeFromLPTimestamp";
  }
  if (nullable) {
    datetrunc_args.push_back(cgen_state_->inlineIntNull(SQLTypeInfo(kBIGINT, false)));
    hptodate_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      hptodate_fname, get_int_type(64, cgen_state_->context_), datetrunc_args);
}

llvm::Value* CodeGenerator::codegenCastTimestampToDate(llvm::Value* ts_lv,
                                                       const int dimen,
                                                       const bool nullable) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(ts_lv->getType()->isIntegerTy(64));
  if (dimen > 0) {
    if (nullable) {
      return cgen_state_->emitExternalCall(
          "DateTruncateHighPrecisionToDateNullable",
          get_int_type(64, cgen_state_->context_),
          {{ts_lv,
            cgen_state_->llInt(DateTimeUtils::get_timestamp_precision_scale(dimen)),
            cgen_state_->inlineIntNull(SQLTypeInfo(kBIGINT, false))}});
    }
    return cgen_state_->emitExternalCall(
        "DateTruncateHighPrecisionToDate",
        get_int_type(64, cgen_state_->context_),
        {{ts_lv,
          cgen_state_->llInt(DateTimeUtils::get_timestamp_precision_scale(dimen))}});
  }
  std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
  if (nullable) {
    nullcheck_codegen =
        std::make_unique<NullCheckCodegen>(cgen_state_,
                                           executor(),
                                           ts_lv,
                                           SQLTypeInfo(kTIMESTAMP, dimen, 0, !nullable),
                                           "cast_timestamp_nullcheck");
  }
  auto ret = cgen_state_->emitExternalCall(
      "datetrunc_day", get_int_type(64, cgen_state_->context_), {ts_lv});
  if (nullcheck_codegen) {
    ret = nullcheck_codegen->finalize(ll_int(NULL_BIGINT, cgen_state_->context_), ret);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegenCastBetweenTimestamps(llvm::Value* ts_lv,
                                                         const SQLTypeInfo& operand_ti,
                                                         const SQLTypeInfo& target_ti,
                                                         const bool nullable) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto operand_dimen = operand_ti.get_dimension();
  const auto target_dimen = target_ti.get_dimension();
  if (operand_dimen == target_dimen) {
    return ts_lv;
  }
  CHECK(ts_lv->getType()->isIntegerTy(64));
  const auto scale =
      DateTimeUtils::get_timestamp_precision_scale(abs(operand_dimen - target_dimen));
  if (operand_dimen < target_dimen) {
    codegenCastBetweenIntTypesOverflowChecks(ts_lv, operand_ti, target_ti, scale);
    return nullable
               ? cgen_state_->emitCall("mul_int64_t_nullable_lhs",
                                       {ts_lv,
                                        cgen_state_->llInt(static_cast<int64_t>(scale)),
                                        cgen_state_->inlineIntNull(operand_ti)})
               : cgen_state_->ir_builder_.CreateMul(
                     ts_lv, cgen_state_->llInt(static_cast<int64_t>(scale)));
  }
  return nullable
             ? cgen_state_->emitCall("floor_div_nullable_lhs",
                                     {ts_lv,
                                      cgen_state_->llInt(static_cast<int64_t>(scale)),
                                      cgen_state_->inlineIntNull(operand_ti)})
             : cgen_state_->ir_builder_.CreateSDiv(
                   ts_lv, cgen_state_->llInt(static_cast<int64_t>(scale)));
}

llvm::Value* CodeGenerator::codegenCastFromString(llvm::Value* operand_lv,
                                                  const SQLTypeInfo& operand_ti,
                                                  const SQLTypeInfo& ti,
                                                  const bool operand_is_const,
                                                  const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (!ti.is_string()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " +
                             ti.get_type_name() + " not supported");
  }
  if (operand_ti.get_compression() == kENCODING_NONE &&
      ti.get_compression() == kENCODING_NONE) {
    return operand_lv;
  }
  if (ti.get_compression() == kENCODING_DICT &&
      operand_ti.get_compression() == kENCODING_DICT) {
    if (ti.get_comp_param() == operand_ti.get_comp_param()) {
      return operand_lv;
    }
    if (operand_ti.get_comp_param() == DictRef::literalsDictId) {
      // Anything being casted from a literal dictionary is not materialized at this point
      // Should already have been kicked to CPU if it was originally a GPU query

      CHECK(co.device_type == ExecutorDeviceType::CPU);
      const int64_t source_string_proxy_handle =
          reinterpret_cast<int64_t>(executor()->getStringDictionaryProxy(
              operand_ti.get_comp_param(), executor()->getRowSetMemoryOwner(), true));

      const int64_t dest_string_proxy_handle =
          reinterpret_cast<int64_t>(executor()->getStringDictionaryProxy(
              ti.get_comp_param(), executor()->getRowSetMemoryOwner(), true));

      auto source_string_proxy_handle_lv = cgen_state_->llInt(source_string_proxy_handle);
      auto dest_string_proxy_handle_lv = cgen_state_->llInt(dest_string_proxy_handle);

      std::vector<llvm::Value*> string_cast_lvs{
          operand_lv, source_string_proxy_handle_lv, dest_string_proxy_handle_lv};
      if (ti.is_dict_intersection()) {
        return cgen_state_->emitExternalCall(
            "intersect_translate_string_id_to_other_dict",
            get_int_type(32, cgen_state_->context_),
            string_cast_lvs);
      } else {
        return cgen_state_->emitExternalCall("union_translate_string_id_to_other_dict",
                                             get_int_type(32, cgen_state_->context_),
                                             string_cast_lvs);
      }
    }

    const std::vector<StringOps_Namespace::StringOpInfo> string_op_infos;
    auto string_dictionary_translation_mgr =
        std::make_unique<StringDictionaryTranslationMgr>(
            operand_ti.get_comp_param(),
            ti.get_comp_param(),
            ti.is_dict_intersection(),
            string_op_infos,
            co.device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL
                                                      : Data_Namespace::CPU_LEVEL,
            executor()->deviceCount(co.device_type),
            executor(),
            &executor()->getCatalog()->getDataMgr(),
            false /* delay_translation */);

    return cgen_state_
        ->moveStringDictionaryTranslationMgr(std::move(string_dictionary_translation_mgr))
        ->codegen(operand_lv, operand_ti, true /* add_nullcheck */, co);
  }
  // dictionary encode non-constant
  if (operand_ti.get_compression() != kENCODING_DICT && !operand_is_const) {
    if (g_cluster) {
      throw std::runtime_error(
          "Cast from none-encoded string to dictionary-encoded not supported for "
          "distributed queries");
    }
    CHECK_EQ(kENCODING_NONE, operand_ti.get_compression());
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    CHECK(operand_lv->getType()->isIntegerTy(64));
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    return cgen_state_->emitExternalCall(
        "string_compress",
        get_int_type(32, cgen_state_->context_),
        {operand_lv,
         cgen_state_->llInt(int64_t(executor()->getStringDictionaryProxy(
             ti.get_comp_param(), executor()->getRowSetMemoryOwner(), true)))});
  }
  CHECK(operand_lv->getType()->isIntegerTy(32));
  if (ti.get_compression() == kENCODING_NONE) {
    if (g_cluster) {
      throw std::runtime_error(
          "Cast from dictionary-encoded string to none-encoded not "
          "currently supported for distributed queries.");
    }
    // Removed watchdog check here in exchange for row cardinality based check in
    // RelAlgExecutor
    CHECK_EQ(kENCODING_DICT, operand_ti.get_compression());
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    const int64_t string_dictionary_ptr =
        operand_ti.get_comp_param() == 0
            ? reinterpret_cast<int64_t>(
                  executor()->getRowSetMemoryOwner()->getLiteralStringDictProxy())
            : reinterpret_cast<int64_t>(executor()->getStringDictionaryProxy(
                  operand_ti.get_comp_param(), executor()->getRowSetMemoryOwner(), true));
    return cgen_state_->emitExternalCall(
        "string_decompress",
        get_int_type(64, cgen_state_->context_),
        {operand_lv, cgen_state_->llInt(string_dictionary_ptr)});
  }
  CHECK(operand_is_const);
  CHECK_EQ(kENCODING_DICT, ti.get_compression());
  return operand_lv;
}

llvm::Value* CodeGenerator::codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                                       const SQLTypeInfo& operand_ti,
                                                       const SQLTypeInfo& ti,
                                                       bool upscale) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (ti.is_decimal() &&
      (!operand_ti.is_decimal() || operand_ti.get_scale() <= ti.get_scale())) {
    if (upscale) {
      if (operand_ti.get_scale() < ti.get_scale()) {  // scale only if needed
        auto scale = exp_to_scale(ti.get_scale() - operand_ti.get_scale());
        const auto scale_lv =
            llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), scale);
        operand_lv = cgen_state_->ir_builder_.CreateSExt(
            operand_lv, get_int_type(64, cgen_state_->context_));

        codegenCastBetweenIntTypesOverflowChecks(operand_lv, operand_ti, ti, scale);

        if (operand_ti.get_notnull()) {
          operand_lv = cgen_state_->ir_builder_.CreateMul(operand_lv, scale_lv);
        } else {
          operand_lv = cgen_state_->emitCall(
              "scale_decimal_up",
              {operand_lv,
               scale_lv,
               cgen_state_->llInt(inline_int_null_val(operand_ti)),
               cgen_state_->inlineIntNull(SQLTypeInfo(kBIGINT, false))});
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
        {operand_lv, scale_lv, cgen_state_->llInt(inline_int_null_val(operand_ti))});
  }
  if (ti.is_integer() && operand_ti.is_integer() &&
      operand_ti.get_logical_size() > ti.get_logical_size()) {
    codegenCastBetweenIntTypesOverflowChecks(operand_lv, operand_ti, ti, 1);
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
  return cgen_state_->emitCall("cast_" + numeric_type_name(operand_ti) + "_to_" +
                                   numeric_type_name(ti) + "_nullable",
                               {operand_lv,
                                cgen_state_->inlineIntNull(operand_ti),
                                cgen_state_->inlineIntNull(ti)});
}

void CodeGenerator::codegenCastBetweenIntTypesOverflowChecks(
    llvm::Value* operand_lv,
    const SQLTypeInfo& operand_ti,
    const SQLTypeInfo& ti,
    const int64_t scale) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) =
      cgen_state_->inlineIntMaxMin(ti.get_logical_size(), true);

  cgen_state_->needs_error_check_ = true;
  auto cast_ok = llvm::BasicBlock::Create(
      cgen_state_->context_, "cast_ok", cgen_state_->current_func_);
  auto cast_fail = llvm::BasicBlock::Create(
      cgen_state_->context_, "cast_fail", cgen_state_->current_func_);
  auto operand_max = static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue() / scale;
  auto operand_min = static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue() / scale;
  const auto ti_llvm_type =
      get_int_type(8 * ti.get_logical_size(), cgen_state_->context_);
  llvm::Value* operand_max_lv = llvm::ConstantInt::get(ti_llvm_type, operand_max);
  llvm::Value* operand_min_lv = llvm::ConstantInt::get(ti_llvm_type, operand_min);
  const bool is_narrowing = operand_ti.get_logical_size() > ti.get_logical_size();
  if (is_narrowing) {
    const auto operand_ti_llvm_type =
        get_int_type(8 * operand_ti.get_logical_size(), cgen_state_->context_);
    operand_max_lv =
        cgen_state_->ir_builder_.CreateSExt(operand_max_lv, operand_ti_llvm_type);
    operand_min_lv =
        cgen_state_->ir_builder_.CreateSExt(operand_min_lv, operand_ti_llvm_type);
  }
  llvm::Value* over{nullptr};
  llvm::Value* under{nullptr};
  if (operand_ti.get_notnull()) {
    over = cgen_state_->ir_builder_.CreateICmpSGT(operand_lv, operand_max_lv);
    under = cgen_state_->ir_builder_.CreateICmpSLE(operand_lv, operand_min_lv);
  } else {
    const auto type_name =
        is_narrowing ? numeric_type_name(operand_ti) : numeric_type_name(ti);
    const auto null_operand_val = cgen_state_->llInt(inline_int_null_val(operand_ti));
    const auto null_bool_val = cgen_state_->inlineIntNull(SQLTypeInfo(kBOOLEAN, false));
    over = toBool(cgen_state_->emitCall(
        "gt_" + type_name + "_nullable_lhs",
        {operand_lv, operand_max_lv, null_operand_val, null_bool_val}));
    under = toBool(cgen_state_->emitCall(
        "le_" + type_name + "_nullable_lhs",
        {operand_lv, operand_min_lv, null_operand_val, null_bool_val}));
  }
  const auto detected = cgen_state_->ir_builder_.CreateOr(over, under, "overflow");
  cgen_state_->ir_builder_.CreateCondBr(detected, cast_fail, cast_ok);

  cgen_state_->ir_builder_.SetInsertPoint(cast_fail);
  cgen_state_->ir_builder_.CreateRet(
      cgen_state_->llInt(Executor::ERR_OVERFLOW_OR_UNDERFLOW));

  cgen_state_->ir_builder_.SetInsertPoint(cast_ok);
}

llvm::Value* CodeGenerator::codegenCastToFp(llvm::Value* operand_lv,
                                            const SQLTypeInfo& operand_ti,
                                            const SQLTypeInfo& ti) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (!ti.is_fp()) {
    throw std::runtime_error("Cast from " + operand_ti.get_type_name() + " to " +
                             ti.get_type_name() + " not supported");
  }
  llvm::Value* result_lv;
  if (operand_ti.get_notnull()) {
    auto const fp_type = ti.get_type() == kFLOAT
                             ? llvm::Type::getFloatTy(cgen_state_->context_)
                             : llvm::Type::getDoubleTy(cgen_state_->context_);
    result_lv = cgen_state_->ir_builder_.CreateSIToFP(operand_lv, fp_type);
    if (auto const scale = static_cast<unsigned>(operand_ti.get_scale())) {
      double const multiplier = shared::power10inv(scale);
      result_lv = cgen_state_->ir_builder_.CreateFMul(
          result_lv, llvm::ConstantFP::get(result_lv->getType(), multiplier));
    }
  } else {
    if (auto const scale = static_cast<unsigned>(operand_ti.get_scale())) {
      double const multiplier = shared::power10inv(scale);
      auto const fp_type = ti.get_type() == kFLOAT
                               ? llvm::Type::getFloatTy(cgen_state_->context_)
                               : llvm::Type::getDoubleTy(cgen_state_->context_);
      result_lv = cgen_state_->emitCall("cast_" + numeric_type_name(operand_ti) + "_to_" +
                                            numeric_type_name(ti) + "_scaled_nullable",
                                        {operand_lv,
                                         cgen_state_->inlineIntNull(operand_ti),
                                         cgen_state_->inlineFpNull(ti),
                                         llvm::ConstantFP::get(fp_type, multiplier)});
    } else {
      result_lv = cgen_state_->emitCall("cast_" + numeric_type_name(operand_ti) + "_to_" +
                                            numeric_type_name(ti) + "_nullable",
                                        {operand_lv,
                                         cgen_state_->inlineIntNull(operand_ti),
                                         cgen_state_->inlineFpNull(ti)});
    }
  }
  CHECK(result_lv);
  return result_lv;
}

llvm::Value* CodeGenerator::codegenCastFromFp(llvm::Value* operand_lv,
                                              const SQLTypeInfo& operand_ti,
                                              const SQLTypeInfo& ti) {
  AUTOMATIC_IR_METADATA(cgen_state_);
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
      // Round by adding/subtracting 0.5 before fptosi.
      auto* fp_type = operand_lv->getType()->isFloatTy()
                          ? llvm::Type::getFloatTy(cgen_state_->context_)
                          : llvm::Type::getDoubleTy(cgen_state_->context_);
      auto* zero = llvm::ConstantFP::get(fp_type, 0.0);
      auto* mhalf = llvm::ConstantFP::get(fp_type, -0.5);
      auto* phalf = llvm::ConstantFP::get(fp_type, 0.5);
      auto* is_negative = cgen_state_->ir_builder_.CreateFCmpOLT(operand_lv, zero);
      auto* offset = cgen_state_->ir_builder_.CreateSelect(is_negative, mhalf, phalf);
      operand_lv = cgen_state_->ir_builder_.CreateFAdd(operand_lv, offset);
      return cgen_state_->ir_builder_.CreateFPToSI(
          operand_lv, get_int_type(get_bit_width(ti), cgen_state_->context_));
    } else {
      CHECK(false);
    }
  } else {
    const auto from_tname = numeric_type_name(operand_ti);
    const auto to_tname = numeric_type_name(ti);
    if (ti.is_fp()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv,
                                    cgen_state_->inlineFpNull(operand_ti),
                                    cgen_state_->inlineFpNull(ti)});
    } else if (ti.is_integer()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv,
                                    cgen_state_->inlineFpNull(operand_ti),
                                    cgen_state_->inlineIntNull(ti)});
    } else {
      CHECK(false);
    }
  }
  CHECK(false);
  return nullptr;
}
