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

std::vector<llvm::Value*> Executor::codegen(const Analyzer::Constant* constant,
                                            const EncodingType enc_type,
                                            const int dict_id,
                                            const CompilationOptions& co) {
  if (co.hoist_literals_) {
    std::vector<const Analyzer::Constant*> constants(deviceCount(co.device_type_), constant);
    return codegenHoistedConstants(constants, enc_type, dict_id);
  }
  const auto& type_info = constant->get_type_info();
  const auto type = type_info.is_decimal() ? decimal_to_int_type(type_info) : type_info.get_type();
  switch (type) {
    case kBOOLEAN:
      return type_info.get_notnull() ? std::vector<llvm::Value*>{llvm::ConstantInt::get(
                                           get_int_type(1, cgen_state_->context_), constant->get_constval().boolval)}
                                     : std::vector<llvm::Value*>{llvm::ConstantInt::get(
                                           get_int_type(8, cgen_state_->context_), constant->get_constval().boolval)};
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return {codegenIntConst(constant)};
    case kFLOAT:
      return {llvm::ConstantFP::get(llvm::Type::getFloatTy(cgen_state_->context_), constant->get_constval().floatval)};
    case kDOUBLE:
      return {
          llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state_->context_), constant->get_constval().doubleval)};
    case kVARCHAR:
    case kCHAR:
    case kTEXT: {
      CHECK(constant->get_constval().stringval || constant->get_is_null());
      if (constant->get_is_null()) {
        if (enc_type == kENCODING_DICT) {
          return {ll_int(static_cast<int32_t>(inline_int_null_val(type_info)))};
        }
        return {ll_int(int64_t(0)),
                llvm::Constant::getNullValue(llvm::PointerType::get(get_int_type(8, cgen_state_->context_), 0)),
                ll_int(int32_t(0))};
      }
      const auto& str_const = *constant->get_constval().stringval;
      if (enc_type == kENCODING_DICT) {
        return {ll_int(getStringDictionaryProxy(dict_id, row_set_mem_owner_, true)->getIdOfString(str_const))};
      }
      return {ll_int(int64_t(0)),
              cgen_state_->addStringConstant(str_const),
              ll_int(static_cast<int32_t>(str_const.size()))};
    }
    default:
      CHECK(false);
  }
  abort();
}

llvm::ConstantInt* Executor::codegenIntConst(const Analyzer::Constant* constant) {
  const auto& type_info = constant->get_type_info();
  if (constant->get_is_null()) {
    return inlineIntNull(type_info);
  }
  const auto type = type_info.is_decimal() ? decimal_to_int_type(type_info) : type_info.get_type();
  switch (type) {
    case kSMALLINT:
      return ll_int(constant->get_constval().smallintval);
    case kINT:
      return ll_int(constant->get_constval().intval);
    case kBIGINT:
      return ll_int(constant->get_constval().bigintval);
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return ll_int(constant->get_constval().timeval);
    default:
      abort();
  }
}

std::vector<llvm::Value*> Executor::codegenHoistedConstants(const std::vector<const Analyzer::Constant*>& constants,
                                                            const EncodingType enc_type,
                                                            const int dict_id) {
  CHECK(!constants.empty());
  const auto& type_info = constants.front()->get_type_info();
  auto lit_buff_lv = get_arg_by_name(cgen_state_->row_func_, "literals");
  int16_t lit_off{-1};
  for (size_t device_id = 0; device_id < constants.size(); ++device_id) {
    const auto constant = constants[device_id];
    const auto& crt_type_info = constant->get_type_info();
    CHECK(type_info == crt_type_info);
    const int16_t dev_lit_off = cgen_state_->getOrAddLiteral(constant, enc_type, dict_id, device_id);
    if (device_id) {
      CHECK_EQ(lit_off, dev_lit_off);
    } else {
      lit_off = dev_lit_off;
    }
  }
  CHECK_GE(lit_off, int16_t(0));
  const auto lit_buf_start = cgen_state_->ir_builder_.CreateGEP(lit_buff_lv, ll_int(lit_off));
  if (type_info.is_string() && enc_type != kENCODING_DICT) {
    CHECK_EQ(kENCODING_NONE, type_info.get_compression());
    CHECK_EQ(size_t(4), literalBytes(LiteralValue(std::string(""))));
    auto off_and_len_ptr = cgen_state_->ir_builder_.CreateBitCast(
        lit_buf_start, llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0));
    // packed offset + length, 16 bits each
    auto off_and_len = cgen_state_->ir_builder_.CreateLoad(off_and_len_ptr);
    auto off_lv = cgen_state_->ir_builder_.CreateLShr(
        cgen_state_->ir_builder_.CreateAnd(off_and_len, ll_int(int32_t(0xffff0000))), ll_int(int32_t(16)));
    auto len_lv = cgen_state_->ir_builder_.CreateAnd(off_and_len, ll_int(int32_t(0x0000ffff)));
    return {ll_int(int64_t(0)), cgen_state_->ir_builder_.CreateGEP(lit_buff_lv, off_lv), len_lv};
  }
  llvm::Type* val_ptr_type{nullptr};
  const auto val_bits = get_bit_width(type_info);
  CHECK_EQ(size_t(0), val_bits % 8);
  if (type_info.is_integer() || type_info.is_decimal() || type_info.is_time() || type_info.is_timeinterval() ||
      type_info.is_string() || type_info.is_boolean()) {
    val_ptr_type = llvm::PointerType::get(llvm::IntegerType::get(cgen_state_->context_, val_bits), 0);
  } else {
    CHECK(type_info.get_type() == kFLOAT || type_info.get_type() == kDOUBLE);
    val_ptr_type = (type_info.get_type() == kFLOAT) ? llvm::Type::getFloatPtrTy(cgen_state_->context_)
                                                    : llvm::Type::getDoublePtrTy(cgen_state_->context_);
  }
  auto lit_lv =
      cgen_state_->ir_builder_.CreateLoad(cgen_state_->ir_builder_.CreateBitCast(lit_buf_start, val_ptr_type));
  if (type_info.is_boolean() && type_info.get_notnull()) {
    return {toBool(lit_lv)};
  }
  return {lit_lv};
}
