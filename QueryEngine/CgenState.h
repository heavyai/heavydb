/*
 * Copyright 2019 OmniSci, Inc.
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

#include "IRCodegenUtils.h"
#include "InValuesBitmap.h"
#include "InputMetadata.h"
#include "LLVMGlobalContext.h"

#include "../Analyzer/Analyzer.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

struct CgenState {
 public:
  CgenState(const std::vector<InputTableInfo>& query_infos,
            const bool contains_left_deep_outer_join)
      : module_(nullptr)
      , row_func_(nullptr)
      , context_(getGlobalLLVMContext())
      , ir_builder_(context_)
      , contains_left_deep_outer_join_(contains_left_deep_outer_join)
      , outer_join_match_found_per_level_(std::max(query_infos.size(), size_t(1)) - 1)
      , query_infos_(query_infos)
      , needs_error_check_(false)
      , needs_geos_(false)
      , query_func_(nullptr)
      , query_func_entry_ir_builder_(context_){};

  size_t getOrAddLiteral(const Analyzer::Constant* constant,
                         const EncodingType enc_type,
                         const int dict_id,
                         const int device_id) {
    const auto& ti = constant->get_type_info();
    const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
    switch (type) {
      case kBOOLEAN:
        return getOrAddLiteral(constant->get_is_null()
                                   ? int8_t(inline_int_null_val(ti))
                                   : int8_t(constant->get_constval().boolval ? 1 : 0),
                               device_id);
      case kTINYINT:
        return getOrAddLiteral(constant->get_is_null()
                                   ? int8_t(inline_int_null_val(ti))
                                   : constant->get_constval().tinyintval,
                               device_id);
      case kSMALLINT:
        return getOrAddLiteral(constant->get_is_null()
                                   ? int16_t(inline_int_null_val(ti))
                                   : constant->get_constval().smallintval,
                               device_id);
      case kINT:
        return getOrAddLiteral(constant->get_is_null() ? int32_t(inline_int_null_val(ti))
                                                       : constant->get_constval().intval,
                               device_id);
      case kBIGINT:
        return getOrAddLiteral(constant->get_is_null()
                                   ? int64_t(inline_int_null_val(ti))
                                   : constant->get_constval().bigintval,
                               device_id);
      case kFLOAT:
        return getOrAddLiteral(constant->get_is_null()
                                   ? float(inline_fp_null_val(ti))
                                   : constant->get_constval().floatval,
                               device_id);
      case kDOUBLE:
        return getOrAddLiteral(constant->get_is_null()
                                   ? inline_fp_null_val(ti)
                                   : constant->get_constval().doubleval,
                               device_id);
      case kCHAR:
      case kTEXT:
      case kVARCHAR:
        if (enc_type == kENCODING_DICT) {
          if (constant->get_is_null()) {
            return getOrAddLiteral(int32_t(inline_int_null_val(ti)), device_id);
          }
          return getOrAddLiteral(
              std::make_pair(*constant->get_constval().stringval, dict_id), device_id);
        }
        CHECK_EQ(kENCODING_NONE, enc_type);
        if (constant->get_is_null()) {
          throw std::runtime_error(
              "CHAR / VARCHAR NULL literal not supported in this context");  // TODO(alex):
                                                                             // support
                                                                             // null
        }
        return getOrAddLiteral(*constant->get_constval().stringval, device_id);
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
        // TODO(alex): support null
        return getOrAddLiteral(constant->get_constval().bigintval, device_id);
      case kARRAY: {
        if (enc_type == kENCODING_NONE) {
          if (ti.get_subtype() == kDOUBLE) {
            std::vector<double> double_array_literal;
            for (const auto& value : constant->get_value_list()) {
              const auto c = dynamic_cast<const Analyzer::Constant*>(value.get());
              CHECK(c);
              double d = c->get_constval().doubleval;
              double_array_literal.push_back(d);
            }
            return getOrAddLiteral(double_array_literal, device_id);
          }
          if (ti.get_subtype() == kINT) {
            std::vector<int32_t> int32_array_literal;
            for (const auto& value : constant->get_value_list()) {
              const auto c = dynamic_cast<const Analyzer::Constant*>(value.get());
              CHECK(c);
              int32_t i = c->get_constval().intval;
              int32_array_literal.push_back(i);
            }
            return getOrAddLiteral(int32_array_literal, device_id);
          }
          if (ti.get_subtype() == kTINYINT) {
            std::vector<int8_t> int8_array_literal;
            for (const auto& value : constant->get_value_list()) {
              const auto c = dynamic_cast<const Analyzer::Constant*>(value.get());
              CHECK(c);
              int8_t i = c->get_constval().tinyintval;
              int8_array_literal.push_back(i);
            }
            if (ti.get_comp_param() == 64) {
              return getOrAddLiteral(std::make_pair(int8_array_literal, 64), device_id);
            }
            return getOrAddLiteral(int8_array_literal, device_id);
          }
          throw std::runtime_error("Unsupported literal array");
        }
        if (enc_type == kENCODING_GEOINT) {
          if (ti.get_subtype() == kTINYINT) {
            std::vector<int8_t> int8_array_literal;
            for (const auto& value : constant->get_value_list()) {
              const auto c = dynamic_cast<const Analyzer::Constant*>(value.get());
              CHECK(c);
              int8_t i = c->get_constval().tinyintval;
              int8_array_literal.push_back(i);
            }
            if (ti.get_comp_param() == 32) {
              return getOrAddLiteral(std::make_pair(int8_array_literal, 32), device_id);
            }
            return getOrAddLiteral(int8_array_literal, device_id);
          }
        }
        throw std::runtime_error("Encoded literal arrays are not supported");
      }
      default:
        abort();
    }
  }

  using LiteralValue = boost::variant<int8_t,
                                      int16_t,
                                      int32_t,
                                      int64_t,
                                      float,
                                      double,
                                      std::pair<std::string, int>,
                                      std::string,
                                      std::vector<double>,
                                      std::vector<int32_t>,
                                      std::vector<int8_t>,
                                      std::pair<std::vector<int8_t>, int>>;
  using LiteralValues = std::vector<LiteralValue>;

  const std::unordered_map<int, LiteralValues>& getLiterals() const { return literals_; }

  llvm::Value* addStringConstant(const std::string& str) {
    llvm::Value* str_lv = ir_builder_.CreateGlobalString(
        str, "str_const_" + std::to_string(std::hash<std::string>()(str)));
    auto i8_ptr = llvm::PointerType::get(get_int_type(8, context_), 0);
    str_constants_.push_back(str_lv);
    str_lv = ir_builder_.CreateBitCast(str_lv, i8_ptr);
    return str_lv;
  }

  const InValuesBitmap* addInValuesBitmap(
      std::unique_ptr<InValuesBitmap>& in_values_bitmap) {
    if (in_values_bitmap->isEmpty()) {
      return in_values_bitmap.get();
    }
    in_values_bitmaps_.emplace_back(std::move(in_values_bitmap));
    return in_values_bitmaps_.back().get();
  }
  // look up a runtime function based on the name, return type and type of
  // the arguments and call it; x64 only, don't call from GPU codegen
  llvm::Value* emitExternalCall(
      const std::string& fname,
      llvm::Type* ret_type,
      const std::vector<llvm::Value*> args,
      const std::vector<llvm::Attribute::AttrKind>& fnattrs = {},
      const bool has_struct_return = false) {
    std::vector<llvm::Type*> arg_types;
    for (const auto arg : args) {
      arg_types.push_back(arg->getType());
    }
    auto func_ty = llvm::FunctionType::get(ret_type, arg_types, false);
    llvm::AttributeList attrs;
    if (!fnattrs.empty()) {
      std::vector<std::pair<unsigned, llvm::Attribute>> indexedAttrs;
      indexedAttrs.reserve(fnattrs.size());
      for (auto attr : fnattrs) {
        indexedAttrs.emplace_back(llvm::AttributeList::FunctionIndex,
                                  llvm::Attribute::get(context_, attr));
      }
      attrs = llvm::AttributeList::get(context_,
                                       {&indexedAttrs.front(), indexedAttrs.size()});
    }

    auto func_p = module_->getOrInsertFunction(fname, func_ty, attrs);
    CHECK(func_p);
#if LLVM_VERSION_MAJOR > 8
    auto callee = func_p.getCallee();
#else
    auto callee = func_p;
#endif
    llvm::Function* func{nullptr};
    if (auto callee_cast = llvm::dyn_cast<llvm::ConstantExpr>(callee)) {
      // Get or insert function automatically adds a ConstantExpr cast if the return type
      // of the existing function does not match the supplied return type.
      CHECK(callee_cast->isCast());
      CHECK_EQ(callee_cast->getNumOperands(), size_t(1));
      func = llvm::dyn_cast<llvm::Function>(callee_cast->getOperand(0));
    } else {
      func = llvm::dyn_cast<llvm::Function>(callee);
    }
    CHECK(func);
#if LLVM_VERSION_MAJOR > 8
    llvm::FunctionType* func_type = func_p.getFunctionType();
#else
    llvm::FunctionType* func_type = func->getFunctionType();
#endif
    CHECK(func_type);
    if (has_struct_return) {
      const auto arg_ti = func_type->getParamType(0);
      CHECK(arg_ti->isPointerTy() && arg_ti->getPointerElementType()->isStructTy());
      auto attr_list = func->getAttributes();
      llvm::AttrBuilder arr_arg_builder(attr_list.getParamAttributes(0));
      arr_arg_builder.addAttribute(llvm::Attribute::StructRet);
      func->addParamAttrs(0, arr_arg_builder);
    }
    const size_t arg_start = has_struct_return ? 1 : 0;
    for (size_t i = arg_start; i < func->arg_size(); i++) {
      const auto arg_ti = func_type->getParamType(i);
      if (arg_ti->isPointerTy() && arg_ti->getPointerElementType()->isStructTy()) {
        auto attr_list = func->getAttributes();
        llvm::AttrBuilder arr_arg_builder(attr_list.getParamAttributes(i));
#if LLVM_VERSION_MAJOR > 8
        arr_arg_builder.addByValAttr(arg_ti->getPointerElementType());
#else
        arr_arg_builder.addAttribute(llvm::Attribute::ByVal);
#endif
        func->addParamAttrs(i, arr_arg_builder);
      }
    }
    llvm::Value* result = ir_builder_.CreateCall(func_p, args);
    // check the assumed type
    CHECK_EQ(result->getType(), ret_type);
    return result;
  }

  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);

  size_t getLiteralBufferUsage(const int device_id) { return literal_bytes_[device_id]; }

  llvm::Value* castToTypeIn(llvm::Value* val, const size_t bit_width);

  std::pair<llvm::ConstantInt*, llvm::ConstantInt*> inlineIntMaxMin(
      const size_t byte_width,
      const bool is_signed);

  llvm::ConstantInt* inlineIntNull(const SQLTypeInfo&);

  llvm::ConstantFP* inlineFpNull(const SQLTypeInfo&);

  template <class T>
  llvm::ConstantInt* llInt(const T v) const {
    return ::ll_int(v, context_);
  }

  llvm::ConstantFP* llFp(const float v) const {
    return static_cast<llvm::ConstantFP*>(
        llvm::ConstantFP::get(llvm::Type::getFloatTy(context_), v));
  }

  llvm::ConstantFP* llFp(const double v) const {
    return static_cast<llvm::ConstantFP*>(
        llvm::ConstantFP::get(llvm::Type::getDoubleTy(context_), v));
  }

  llvm::ConstantInt* llBool(const bool v) const { return ::ll_bool(v, context_); }

  void emitErrorCheck(llvm::Value* condition, llvm::Value* errorCode, std::string label);

  llvm::Module* module_;
  llvm::Function* row_func_;
  std::vector<llvm::Function*> helper_functions_;
  llvm::LLVMContext& context_;
  llvm::ValueToValueMapTy vmap_;  // used for cloning the runtime module
  llvm::IRBuilder<> ir_builder_;
  std::unordered_map<int, std::vector<llvm::Value*>> fetch_cache_;
  struct FunctionOperValue {
    const Analyzer::FunctionOper* foper;
    llvm::Value* lv;
  };
  std::vector<FunctionOperValue> ext_call_cache_;
  std::vector<llvm::Value*> group_by_expr_cache_;
  std::vector<llvm::Value*> str_constants_;
  std::vector<llvm::Value*> frag_offsets_;
  const bool contains_left_deep_outer_join_;
  std::vector<llvm::Value*> outer_join_match_found_per_level_;
  std::unordered_map<int, llvm::Value*> scan_idx_to_hash_pos_;
  std::vector<std::unique_ptr<const InValuesBitmap>> in_values_bitmaps_;
  const std::vector<InputTableInfo>& query_infos_;
  bool needs_error_check_;
  bool needs_geos_;

  llvm::Function* query_func_;
  llvm::IRBuilder<> query_func_entry_ir_builder_;
  std::unordered_map<int, std::vector<llvm::Value*>> query_func_literal_loads_;

  struct HoistedLiteralLoadLocator {
    int offset_in_literal_buffer;
    int index_of_literal_load;
  };
  std::unordered_map<llvm::Value*, HoistedLiteralLoadLocator> row_func_hoisted_literals_;

  static size_t literalBytes(const CgenState::LiteralValue& lit) {
    switch (lit.which()) {
      case 0:
        return 1;  // int8_t
      case 1:
        return 2;  // int16_t
      case 2:
        return 4;  // int32_t
      case 3:
        return 8;  // int64_t
      case 4:
        return 4;  // float
      case 5:
        return 8;  // double
      case 6:
        return 4;  // std::pair<std::string, int>
      case 7:
        return 4;  // std::string
      case 8:
        return 4;  // std::vector<double>
      case 9:
        return 4;  // std::vector<int32_t>
      case 10:
        return 4;  // std::vector<int8_t>
      case 11:
        return 4;  // std::pair<std::vector<int8_t>, int>
      default:
        abort();
    }
  }

  static size_t addAligned(const size_t off_in, const size_t alignment) {
    size_t off = off_in;
    if (off % alignment != 0) {
      off += (alignment - off % alignment);
    }
    return off + alignment;
  }

 private:
  template <class T>
  size_t getOrAddLiteral(const T& val, const int device_id) {
    const LiteralValue var_val(val);
    size_t literal_found_off{0};
    auto& literals = literals_[device_id];
    for (const auto& literal : literals) {
      const auto lit_bytes = literalBytes(literal);
      literal_found_off = addAligned(literal_found_off, lit_bytes);
      if (literal == var_val) {
        return literal_found_off - lit_bytes;
      }
    }
    literals.emplace_back(val);
    const auto lit_bytes = literalBytes(var_val);
    literal_bytes_[device_id] = addAligned(literal_bytes_[device_id], lit_bytes);
    return literal_bytes_[device_id] - lit_bytes;
  }

  std::unordered_map<int, LiteralValues> literals_;
  std::unordered_map<int, size_t> literal_bytes_;
};
