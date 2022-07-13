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

std::vector<llvm::Value*> CodeGenerator::codegen(const Analyzer::Constant* constant,
                                                 const EncodingType enc_type,
                                                 const int dict_id,
                                                 const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (co.hoist_literals) {
    if (const auto geo_constant = dynamic_cast<const Analyzer::GeoConstant*>(constant)) {
      return codegenGeoConstant(geo_constant, co);
    } else {
      std::vector<const Analyzer::Constant*> constants(
          executor()->deviceCount(co.device_type), constant);
      return codegenHoistedConstants(constants, enc_type, dict_id);
    }
  }
  const auto& type_info = constant->get_type_info();
  const auto type =
      type_info.is_decimal() ? decimal_to_int_type(type_info) : type_info.get_type();
  switch (type) {
    case kBOOLEAN:
      return {llvm::ConstantInt::get(get_int_type(8, cgen_state_->context_),
                                     constant->get_constval().boolval)};
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return {CodeGenerator::codegenIntConst(constant, cgen_state_)};
    case kFLOAT:
      return {llvm::ConstantFP::get(llvm::Type::getFloatTy(cgen_state_->context_),
                                    constant->get_constval().floatval)};
    case kDOUBLE:
      return {llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state_->context_),
                                    constant->get_constval().doubleval)};
    case kVARCHAR:
    case kCHAR:
    case kTEXT: {
      CHECK(constant->get_constval().stringval || constant->get_is_null());
      if (constant->get_is_null()) {
        if (enc_type == kENCODING_DICT) {
          return {
              cgen_state_->llInt(static_cast<int32_t>(inline_int_null_val(type_info)))};
        }
        return {cgen_state_->llInt(int64_t(0)),
                llvm::Constant::getNullValue(
                    llvm::PointerType::get(get_int_type(8, cgen_state_->context_), 0)),
                cgen_state_->llInt(int32_t(0))};
      }
      const auto& str_const = *constant->get_constval().stringval;
      if (enc_type == kENCODING_DICT) {
        return {
            cgen_state_->llInt(executor()
                                   ->getStringDictionaryProxy(
                                       dict_id, executor()->getRowSetMemoryOwner(), true)
                                   ->getIdOfString(str_const))};
      }
      return {cgen_state_->llInt(int64_t(0)),
              cgen_state_->addStringConstant(str_const),
              cgen_state_->llInt(static_cast<int32_t>(str_const.size()))};
    }
    default:
      CHECK(false);
  }
  abort();
}

llvm::ConstantInt* CodeGenerator::codegenIntConst(const Analyzer::Constant* constant,
                                                  CgenState* cgen_state) {
  const auto& type_info = constant->get_type_info();
  if (constant->get_is_null()) {
    return cgen_state->inlineIntNull(type_info);
  }
  const auto type =
      type_info.is_decimal() ? decimal_to_int_type(type_info) : type_info.get_type();
  switch (type) {
    case kTINYINT:
      return cgen_state->llInt(constant->get_constval().tinyintval);
    case kSMALLINT:
      return cgen_state->llInt(constant->get_constval().smallintval);
    case kINT:
      return cgen_state->llInt(constant->get_constval().intval);
    case kBIGINT:
      return cgen_state->llInt(constant->get_constval().bigintval);
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return cgen_state->llInt(constant->get_constval().bigintval);
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return nullptr;
}

namespace {

SQLTypes get_phys_int_type(const size_t byte_sz) {
  switch (byte_sz) {
    case 1:
      return kTINYINT;
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    case 8:
      return kBIGINT;
    default:
      CHECK(false);
  }
  return kNULLT;
}

}  // namespace

// Below, the val_bits_in used to always equal the val_bits_out so we're being cautious.
bool g_allow_invalid_literal_buffer_reads{false};

std::vector<llvm::Value*> CodeGenerator::codegenHoistedConstantsLoads(
    const SQLTypeInfo& type_info,
    const EncodingType enc_type,
    const int dict_id,
    const int16_t lit_off,
    const size_t lit_bytes) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  std::string literal_name = "literal_" + std::to_string(lit_off);
  auto lit_buff_query_func_lv = get_arg_by_name(cgen_state_->query_func_, "literals");
  const auto lit_buf_start = cgen_state_->query_func_entry_ir_builder_.CreateGEP(
      lit_buff_query_func_lv->getType()->getScalarType()->getPointerElementType(),
      lit_buff_query_func_lv,
      cgen_state_->llInt(lit_off));
  CHECK(!type_info.is_geometry());
  if (type_info.is_string() && enc_type != kENCODING_DICT) {
    CHECK_EQ(kENCODING_NONE, type_info.get_compression());
    CHECK_EQ(size_t(4),
             CgenState::literalBytes(CgenState::LiteralValue(std::string(""))));
    auto off_and_len_ptr = cgen_state_->query_func_entry_ir_builder_.CreateBitCast(
        lit_buf_start,
        llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0));
    // packed offset + length, 16 bits each
    auto off_and_len = cgen_state_->query_func_entry_ir_builder_.CreateLoad(
        off_and_len_ptr->getType()->getPointerElementType(), off_and_len_ptr);
    auto off_lv = cgen_state_->query_func_entry_ir_builder_.CreateLShr(
        cgen_state_->query_func_entry_ir_builder_.CreateAnd(
            off_and_len, cgen_state_->llInt(int32_t(0xffff0000))),
        cgen_state_->llInt(int32_t(16)));
    auto len_lv = cgen_state_->query_func_entry_ir_builder_.CreateAnd(
        off_and_len, cgen_state_->llInt(int32_t(0x0000ffff)));

    auto var_start = cgen_state_->llInt(int64_t(0));
    auto var_start_address = cgen_state_->query_func_entry_ir_builder_.CreateGEP(
        lit_buff_query_func_lv->getType()->getScalarType()->getPointerElementType(),
        lit_buff_query_func_lv,
        off_lv);
    auto var_length = len_lv;

    var_start->setName(literal_name + "_start");
    var_start_address->setName(literal_name + "_start_address");
    var_length->setName(literal_name + "_length");

    return {var_start, var_start_address, var_length};
  } else if (type_info.is_array() &&
             (enc_type == kENCODING_NONE || enc_type == kENCODING_GEOINT)) {
    if (enc_type == kENCODING_NONE) {
      CHECK_EQ(kENCODING_NONE, type_info.get_compression());
    } else if (enc_type == kENCODING_GEOINT) {
      CHECK_EQ(kENCODING_GEOINT, type_info.get_compression());
      CHECK_EQ(kTINYINT, type_info.get_subtype());
    }

    auto off_and_len_ptr = cgen_state_->query_func_entry_ir_builder_.CreateBitCast(
        lit_buf_start,
        llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0));
    // packed offset + length, 16 bits each
    auto off_and_len = cgen_state_->query_func_entry_ir_builder_.CreateLoad(
        off_and_len_ptr->getType()->getPointerElementType(), off_and_len_ptr);
    auto off_lv = cgen_state_->query_func_entry_ir_builder_.CreateLShr(
        cgen_state_->query_func_entry_ir_builder_.CreateAnd(
            off_and_len, cgen_state_->llInt(int32_t(0xffff0000))),
        cgen_state_->llInt(int32_t(16)));
    auto len_lv = cgen_state_->query_func_entry_ir_builder_.CreateAnd(
        off_and_len, cgen_state_->llInt(int32_t(0x0000ffff)));

    auto var_start_address = cgen_state_->query_func_entry_ir_builder_.CreateGEP(
        lit_buff_query_func_lv->getType()->getScalarType()->getPointerElementType(),
        lit_buff_query_func_lv,
        off_lv);
    auto var_length = len_lv;

    var_start_address->setName(literal_name + "_start_address");
    var_length->setName(literal_name + "_length");

    return {var_start_address, var_length};
  }

  // Load a literal from the literal buffer. See also getOrAddLiteral().
  llvm::Type* val_ptr_type{nullptr};
  // NOTE(sy): If val_bits_in is ever different from val_bits_out, that means we need to
  // generate casting below for the type that has this happen. Currently only the decimal
  // type is known to ever have this happen.
  const size_t lit_bits = lit_bytes * 8;
  const size_t val_bits_out = get_bit_width(type_info);
  const size_t val_bits_in = type_info.is_decimal() ? lit_bits : val_bits_out;
  if (val_bits_in != lit_bits && !g_allow_invalid_literal_buffer_reads) {
    // Refuse to read the wrong number of bytes from the literal buffer.
    std::stringstream ss;
    ss << "Invalid literal buffer read size " << val_bits_in << " (expected " << lit_bits
       << ") for type " << toString(type_info.get_type())
       << ". See also: --allow-invalid-literal-buffer-reads";
    LOG(ERROR) << "ERROR: " << ss.str();
    LOG(ERROR) << type_info.to_string();
    throw std::runtime_error(ss.str());
  }
  CHECK_EQ(size_t(0), val_bits_in % 8);
  CHECK_EQ(size_t(0), val_bits_out % 8);
  if (type_info.is_integer() || type_info.is_decimal() || type_info.is_time() ||
      type_info.is_timeinterval() || type_info.is_string() || type_info.is_boolean()) {
    val_ptr_type = llvm::PointerType::get(
        llvm::IntegerType::get(cgen_state_->context_, val_bits_in), 0);
  } else {
    CHECK(type_info.get_type() == kFLOAT || type_info.get_type() == kDOUBLE);
    val_ptr_type = (type_info.get_type() == kFLOAT)
                       ? llvm::Type::getFloatPtrTy(cgen_state_->context_)
                       : llvm::Type::getDoublePtrTy(cgen_state_->context_);
  }
  auto* bit_cast = cgen_state_->query_func_entry_ir_builder_.CreateBitCast(lit_buf_start,
                                                                           val_ptr_type);
  llvm::Value* lit_lv = cgen_state_->query_func_entry_ir_builder_.CreateLoad(
      bit_cast->getType()->getPointerElementType(), bit_cast);
  if (type_info.is_decimal() && val_bits_in != val_bits_out) {
    // Generate casting.
    SQLTypeInfo type_info_in(get_phys_int_type(val_bits_in / 8),
                             type_info.get_dimension(),
                             type_info.get_scale(),
                             false,
                             kENCODING_NONE,
                             0,
                             type_info.get_subtype());
    SQLTypeInfo type_info_out(get_phys_int_type(val_bits_out / 8),
                              type_info.get_dimension(),
                              type_info.get_scale(),
                              false,
                              kENCODING_NONE,
                              0,
                              type_info.get_subtype());
    lit_lv = cgen_state_->emitEntryCall("cast_int" + std::to_string(val_bits_in) +
                                            "_t_to_int" + std::to_string(val_bits_out) +
                                            "_t_nullable",
                                        {lit_lv,
                                         cgen_state_->inlineIntNull(type_info_in),
                                         cgen_state_->inlineIntNull(type_info_out)});
  }
  lit_lv->setName(literal_name);
  return {lit_lv};
}

std::vector<llvm::Value*> CodeGenerator::codegenHoistedConstantsPlaceholders(
    const SQLTypeInfo& type_info,
    const EncodingType enc_type,
    const int16_t lit_off,
    const std::vector<llvm::Value*>& literal_loads) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  std::string literal_name = "literal_" + std::to_string(lit_off);
  CHECK(!type_info.is_geometry());

  if (type_info.is_string() && enc_type != kENCODING_DICT) {
    CHECK_EQ(literal_loads.size(), 3u);

    llvm::Value* var_start = literal_loads[0];
    llvm::Value* var_start_address = literal_loads[1];
    llvm::Value* var_length = literal_loads[2];

    llvm::PointerType* placeholder0_type =
        llvm::PointerType::get(var_start->getType(), 0);
    auto* int_to_ptr0 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder0_type);
    auto placeholder0 = cgen_state_->ir_builder_.CreateLoad(
        int_to_ptr0->getType()->getPointerElementType(),
        int_to_ptr0,
        "__placeholder__" + literal_name + "_start");
    llvm::PointerType* placeholder1_type =
        llvm::PointerType::get(var_start_address->getType(), 0);
    auto* int_to_ptr1 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder1_type);
    auto placeholder1 = cgen_state_->ir_builder_.CreateLoad(
        int_to_ptr1->getType()->getPointerElementType(),
        int_to_ptr1,
        "__placeholder__" + literal_name + "_start_address");
    llvm::PointerType* placeholder2_type =
        llvm::PointerType::get(var_length->getType(), 0);
    auto* int_to_ptr2 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder2_type);
    auto placeholder2 = cgen_state_->ir_builder_.CreateLoad(
        int_to_ptr2->getType()->getPointerElementType(),
        int_to_ptr2,
        "__placeholder__" + literal_name + "_length");

    cgen_state_->row_func_hoisted_literals_[placeholder0] = {lit_off, 0};
    cgen_state_->row_func_hoisted_literals_[placeholder1] = {lit_off, 1};
    cgen_state_->row_func_hoisted_literals_[placeholder2] = {lit_off, 2};

    return {placeholder0, placeholder1, placeholder2};
  }

  if (type_info.is_array() &&
      (enc_type == kENCODING_NONE || enc_type == kENCODING_GEOINT)) {
    CHECK_EQ(literal_loads.size(), 2u);

    llvm::Value* var_start_address = literal_loads[0];
    llvm::Value* var_length = literal_loads[1];

    llvm::PointerType* placeholder0_type =
        llvm::PointerType::get(var_start_address->getType(), 0);
    auto* int_to_ptr0 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder0_type);
    auto placeholder0 = cgen_state_->ir_builder_.CreateLoad(
        int_to_ptr0->getType()->getPointerElementType(),
        int_to_ptr0,
        "__placeholder__" + literal_name + "_start_address");
    llvm::PointerType* placeholder1_type =
        llvm::PointerType::get(var_length->getType(), 0);
    auto* int_to_ptr1 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder1_type);
    auto placeholder1 = cgen_state_->ir_builder_.CreateLoad(
        int_to_ptr1->getType()->getPointerElementType(),
        int_to_ptr1,
        "__placeholder__" + literal_name + "_length");

    cgen_state_->row_func_hoisted_literals_[placeholder0] = {lit_off, 0};
    cgen_state_->row_func_hoisted_literals_[placeholder1] = {lit_off, 1};

    return {placeholder0, placeholder1};
  }

  CHECK_EQ(literal_loads.size(), 1u);
  llvm::Value* to_return_lv = literal_loads[0];

  auto* int_to_ptr = cgen_state_->ir_builder_.CreateIntToPtr(
      cgen_state_->llInt(0), llvm::PointerType::get(to_return_lv->getType(), 0));
  auto placeholder0 =
      cgen_state_->ir_builder_.CreateLoad(int_to_ptr->getType()->getPointerElementType(),
                                          int_to_ptr,
                                          "__placeholder__" + literal_name);

  cgen_state_->row_func_hoisted_literals_[placeholder0] = {lit_off, 0};

  return {placeholder0};
}

std::vector<llvm::Value*> CodeGenerator::codegenHoistedConstants(
    const std::vector<const Analyzer::Constant*>& constants,
    const EncodingType enc_type,
    const int dict_id) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(!constants.empty());
  const auto& type_info = constants.front()->get_type_info();
  checked_int16_t checked_lit_off{0};
  size_t next_lit_bytes{0};
  int16_t lit_off{-1};
  size_t lit_bytes;
  try {
    for (size_t device_id = 0; device_id < constants.size(); ++device_id) {
      const auto constant = constants[device_id];
      const auto& crt_type_info = constant->get_type_info();
      CHECK(type_info == crt_type_info);
      std::tie(checked_lit_off, next_lit_bytes) =
          cgen_state_->getOrAddLiteral(constant, enc_type, dict_id, device_id);
      if (device_id) {
        CHECK_EQ(lit_off, checked_lit_off);
        CHECK_EQ(lit_bytes, next_lit_bytes);
      } else {
        lit_off = static_cast<int16_t>(checked_lit_off);
        lit_bytes = next_lit_bytes;
      }
    }
  } catch (const std::range_error& e) {
    // detect literal buffer overflow when trying to
    // assign literal buf offset which is not in a valid range
    // to checked_type variable
    throw TooManyLiterals();
  }
  CHECK_GE(lit_off, 0);
  std::vector<llvm::Value*> hoisted_literal_loads;
  auto entry = cgen_state_->query_func_literal_loads_.find(lit_off);

  if (entry == cgen_state_->query_func_literal_loads_.end()) {
    hoisted_literal_loads =
        codegenHoistedConstantsLoads(type_info, enc_type, dict_id, lit_off, lit_bytes);
    cgen_state_->query_func_literal_loads_[lit_off] = hoisted_literal_loads;
  } else {
    hoisted_literal_loads = entry->second;
  }

  std::vector<llvm::Value*> literal_placeholders = codegenHoistedConstantsPlaceholders(
      type_info, enc_type, lit_off, hoisted_literal_loads);
  return literal_placeholders;
}
