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
#include "ExtensionFunctions.hpp"
#include "ExtensionFunctionsBinding.h"
#include "ExtensionFunctionsWhitelist.h"

namespace {

llvm::Type* ext_arg_type_to_llvm_type(const ExtArgumentType ext_arg_type,
                                      llvm::LLVMContext& ctx) {
  switch (ext_arg_type) {
    case ExtArgumentType::Bool:  // pass thru to Int8
    case ExtArgumentType::Int8:
      return get_int_type(8, ctx);
    case ExtArgumentType::Int16:
      return get_int_type(16, ctx);
    case ExtArgumentType::Int32:
      return get_int_type(32, ctx);
    case ExtArgumentType::Int64:
      return get_int_type(64, ctx);
    case ExtArgumentType::Float:
      return llvm::Type::getFloatTy(ctx);
    case ExtArgumentType::Double:
      return llvm::Type::getDoubleTy(ctx);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

inline SQLTypeInfo get_sql_type_from_llvm_type(const llvm::Type* ll_type) {
  CHECK(ll_type);
  const auto bits = ll_type->getPrimitiveSizeInBits();

  if (ll_type->isFloatingPointTy()) {
    switch (bits) {
      case 32:
        return SQLTypeInfo(kFLOAT, true);
      case 64:
        return SQLTypeInfo(kDOUBLE, true);
      default:
        LOG(FATAL) << "Unsupported llvm floating point type: " << bits
                   << ", only 32 and 64 bit floating point is supported.";
    }
  } else {
    switch (bits) {
      case 1:
        return SQLTypeInfo(kBOOLEAN, true);
      case 8:
        return SQLTypeInfo(kTINYINT, true);
      case 16:
        return SQLTypeInfo(kSMALLINT, true);
      case 32:
        return SQLTypeInfo(kINT, true);
      case 64:
        return SQLTypeInfo(kBIGINT, true);
      default:
        LOG(FATAL) << "Unrecognized llvm type for SQL type: "
                   << bits;  // TODO let's get the real name here
    }
  }
  UNREACHABLE();
  return SQLTypeInfo();
}

bool ext_func_call_requires_nullcheck(const Analyzer::FunctionOper* function_oper) {
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    if (!arg_ti.get_notnull() && !arg_ti.is_array() && !arg_ti.is_geometry()) {
      return true;
    }
  }
  return false;
}

}  // namespace

#include "../Shared/sql_type_to_string.h"

extern "C" void register_buffer_with_executor_rsm(int64_t exec, int8_t* buffer) {
  Executor* exec_ptr = reinterpret_cast<Executor*>(exec);
  if (buffer != nullptr) {
    exec_ptr->getRowSetMemoryOwner()->addVarlenBuffer(buffer);
  }
}

llvm::Value* CodeGenerator::codegenFunctionOper(
    const Analyzer::FunctionOper* function_oper,
    const CompilationOptions& co) {
  auto ext_func_sig = bind_function(function_oper);

  const auto& ret_ti = function_oper->get_type_info();
  CHECK(ret_ti.is_integer() || ret_ti.is_fp() || ret_ti.is_boolean());
  const auto ret_ty =
      ext_arg_type_to_llvm_type(ext_func_sig.getRet(), cgen_state_->context_);
  const auto current_bb = cgen_state_->ir_builder_.GetInsertBlock();
  for (auto it : cgen_state_->ext_call_cache_) {
    if (*it.foper == *function_oper) {
      auto inst = llvm::dyn_cast<llvm::Instruction>(it.lv);
      if (inst && inst->getParent() == current_bb) {
        return it.lv;
      }
    }
  }
  std::vector<llvm::Value*> orig_arg_lvs;
  std::unordered_map<llvm::Value*, llvm::Value*> const_arr_size;
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto arg_cast = dynamic_cast<const Analyzer::UOper*>(arg);
    const auto arg0 =
        (arg_cast && arg_cast->get_optype() == kCAST) ? arg_cast->get_operand() : arg;
    const auto array_expr_arg = dynamic_cast<const Analyzer::ArrayExpr*>(arg0);
    auto is_local_alloc = (array_expr_arg && array_expr_arg->isLocalAlloc());
    const auto& arg_ti = arg->get_type_info();
    const auto arg_lvs = codegen(arg, true, co);
    // TODO(adb / d): Assuming no const array cols for geo (for now)
    if (arg_ti.is_geometry()) {
      CHECK_EQ(static_cast<size_t>(arg_ti.get_physical_coord_cols()), arg_lvs.size());
      for (size_t i = 0; i < arg_lvs.size(); i++) {
        orig_arg_lvs.push_back(arg_lvs[i]);
      }
    } else {
      if (arg_lvs.size() > 1) {
        CHECK(arg_ti.is_array());
        CHECK_EQ(size_t(2), arg_lvs.size());
        const_arr_size[arg_lvs.front()] = arg_lvs.back();
      } else {
        CHECK_EQ(size_t(1), arg_lvs.size());
        if (is_local_alloc && arg_ti.get_size() > 0) {
          const_arr_size[arg_lvs.front()] = cgen_state_->llInt(arg_ti.get_size());
        }
      }
      orig_arg_lvs.push_back(arg_lvs.front());
    }
  }
  // The extension function implementations don't handle NULL, they work under
  // the assumption that the inputs are validated before calling them. Generate
  // code to do the check at the call site: if any argument is NULL, return NULL
  // without calling the function at all.
  const auto bbs = beginArgsNullcheck(function_oper, orig_arg_lvs);
  CHECK_GE(orig_arg_lvs.size(), function_oper->getArity());
  // Arguments must be converted to the types the extension function can handle.
  const auto args = codegenFunctionOperCastArgs(
      function_oper, &ext_func_sig, orig_arg_lvs, const_arr_size, co);
  auto ext_call = cgen_state_->emitExternalCall(ext_func_sig.getName(), ret_ty, args);
  // Cast the return of the extension function to match the FunctionOper
  const auto extension_ret_ti = get_sql_type_from_llvm_type(ret_ty);
  if (bbs.args_null_bb &&
      extension_ret_ti.get_type() != function_oper->get_type_info().get_type()) {
    ext_call = codegenCast(
        ext_call, extension_ret_ti, function_oper->get_type_info(), false, co);
  }

  auto ext_call_nullcheck = endArgsNullcheck(bbs, ext_call, function_oper);
  cgen_state_->ext_call_cache_.push_back({function_oper, ext_call_nullcheck});
  return ext_call_nullcheck;
}

// Start the control flow needed for a call site check of NULL arguments.
CodeGenerator::ArgNullcheckBBs CodeGenerator::beginArgsNullcheck(
    const Analyzer::FunctionOper* function_oper,
    const std::vector<llvm::Value*>& orig_arg_lvs) {
  llvm::BasicBlock* args_null_bb{nullptr};
  llvm::BasicBlock* args_notnull_bb{nullptr};
  llvm::BasicBlock* orig_bb = cgen_state_->ir_builder_.GetInsertBlock();
  // Only generate the check if required (at least one argument must be nullable).
  if (ext_func_call_requires_nullcheck(function_oper)) {
    const auto args_notnull_lv = cgen_state_->ir_builder_.CreateNot(
        codegenFunctionOperNullArg(function_oper, orig_arg_lvs));
    args_notnull_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "args_notnull", cgen_state_->row_func_);
    args_null_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "args_null", cgen_state_->row_func_);
    cgen_state_->ir_builder_.CreateCondBr(args_notnull_lv, args_notnull_bb, args_null_bb);
    cgen_state_->ir_builder_.SetInsertPoint(args_notnull_bb);
  }
  return {args_null_bb, args_notnull_bb, orig_bb};
}

// Wrap up the control flow needed for NULL argument handling.
llvm::Value* CodeGenerator::endArgsNullcheck(
    const ArgNullcheckBBs& bbs,
    llvm::Value* fn_ret_lv,
    const Analyzer::FunctionOper* function_oper) {
  if (bbs.args_null_bb) {
    CHECK(bbs.args_notnull_bb);
    cgen_state_->ir_builder_.CreateBr(bbs.args_null_bb);
    cgen_state_->ir_builder_.SetInsertPoint(bbs.args_null_bb);

    // The return type of the FunctionOper. The extension function call will be cast to
    // this type if required.
    const auto& func_oper_ret_ti = function_oper->get_type_info();

    auto ext_call_phi = cgen_state_->ir_builder_.CreatePHI(
        func_oper_ret_ti.is_fp()
            ? get_fp_type(func_oper_ret_ti.get_size() * 8, cgen_state_->context_)
            : get_int_type(func_oper_ret_ti.get_size() * 8, cgen_state_->context_),
        2);

    ext_call_phi->addIncoming(fn_ret_lv, bbs.args_notnull_bb);

    const auto null_lv =
        func_oper_ret_ti.is_fp()
            ? static_cast<llvm::Value*>(cgen_state_->inlineFpNull(func_oper_ret_ti))
            : static_cast<llvm::Value*>(cgen_state_->inlineIntNull(func_oper_ret_ti));
    ext_call_phi->addIncoming(null_lv, bbs.orig_bb);
    return ext_call_phi;
  }
  return fn_ret_lv;
}

namespace {

bool call_requires_custom_type_handling(const Analyzer::FunctionOper* function_oper) {
  const auto& ret_ti = function_oper->get_type_info();
  if (!ret_ti.is_integer() && !ret_ti.is_fp()) {
    return true;
  }
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    if (!arg_ti.is_integer() && !arg_ti.is_fp()) {
      return true;
    }
  }
  return false;
}

}  // namespace

llvm::Value* CodeGenerator::codegenFunctionOperWithCustomTypeHandling(
    const Analyzer::FunctionOperWithCustomTypeHandling* function_oper,
    const CompilationOptions& co) {
  if (call_requires_custom_type_handling(function_oper)) {
    // Some functions need the return type to be the same as the input type.
    if (function_oper->getName() == "FLOOR" || function_oper->getName() == "CEIL") {
      CHECK_EQ(size_t(1), function_oper->getArity());
      const auto arg = function_oper->getArg(0);
      const auto& arg_ti = arg->get_type_info();
      CHECK(arg_ti.is_decimal());
      const auto arg_lvs = codegen(arg, true, co);
      CHECK_EQ(size_t(1), arg_lvs.size());
      const auto arg_lv = arg_lvs.front();
      CHECK(arg_lv->getType()->isIntegerTy(64));
      const auto bbs = beginArgsNullcheck(function_oper, {arg_lvs});
      const std::string func_name =
          (function_oper->getName() == "FLOOR") ? "decimal_floor" : "decimal_ceil";
      const auto covar_result_lv = cgen_state_->emitCall(
          func_name, {arg_lv, cgen_state_->llInt(exp_to_scale(arg_ti.get_scale()))});
      const auto ret_ti = function_oper->get_type_info();
      CHECK(ret_ti.is_decimal());
      CHECK_EQ(0, ret_ti.get_scale());
      const auto result_lv = cgen_state_->ir_builder_.CreateSDiv(
          covar_result_lv, cgen_state_->llInt(exp_to_scale(arg_ti.get_scale())));
      return endArgsNullcheck(bbs, result_lv, function_oper);
    } else if (function_oper->getName() == "ROUND" &&
               function_oper->getArg(0)->get_type_info().is_decimal()) {
      CHECK_EQ(size_t(2), function_oper->getArity());

      const auto arg0 = function_oper->getArg(0);
      const auto& arg0_ti = arg0->get_type_info();
      const auto arg0_lvs = codegen(arg0, true, co);
      CHECK_EQ(size_t(1), arg0_lvs.size());
      const auto arg0_lv = arg0_lvs.front();
      CHECK(arg0_lv->getType()->isIntegerTy(64));

      const auto arg1 = function_oper->getArg(1);
      const auto& arg1_ti = arg1->get_type_info();
      CHECK(arg1_ti.is_integer());
      const auto arg1_lvs = codegen(arg1, true, co);
      auto arg1_lv = arg1_lvs.front();
      if (arg1_ti.get_type() != kINT) {
        arg1_lv = codegenCast(arg1_lv, arg1_ti, SQLTypeInfo(kINT, true), false, co);
      }

      const auto bbs0 = beginArgsNullcheck(function_oper, {arg0_lv, arg1_lvs.front()});

      const std::string func_name = "Round__4";
      const auto ret_ti = function_oper->get_type_info();
      CHECK(ret_ti.is_decimal());
      const auto result_lv = cgen_state_->emitExternalCall(
          func_name,
          get_int_type(64, cgen_state_->context_),
          {arg0_lv, arg1_lv, cgen_state_->llInt(arg0_ti.get_scale())});

      return endArgsNullcheck(bbs0, result_lv, function_oper);
    }
    throw std::runtime_error("Type combination not supported for function " +
                             function_oper->getName());
  }
  return codegenFunctionOper(function_oper, co);
}

// Generates code which returns true iff at least one of the arguments is NULL.
llvm::Value* CodeGenerator::codegenFunctionOperNullArg(
    const Analyzer::FunctionOper* function_oper,
    const std::vector<llvm::Value*>& orig_arg_lvs) {
  llvm::Value* one_arg_null =
      llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), false);
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    if (arg_ti.get_notnull() || arg_ti.is_array() || arg_ti.is_geometry()) {
      continue;
    }
    CHECK(arg_ti.is_number());
    one_arg_null = cgen_state_->ir_builder_.CreateOr(
        one_arg_null, codegenIsNullNumber(orig_arg_lvs[i], arg_ti));
  }
  return one_arg_null;
}

// Generate CAST operations for arguments in `orig_arg_lvs` to the types required by
// `ext_func_sig`.
std::vector<llvm::Value*> CodeGenerator::codegenFunctionOperCastArgs(
    const Analyzer::FunctionOper* function_oper,
    const ExtensionFunction* ext_func_sig,
    const std::vector<llvm::Value*>& orig_arg_lvs,
    const std::unordered_map<llvm::Value*, llvm::Value*>& const_arr_size,
    const CompilationOptions& co) {
  CHECK(ext_func_sig);
  const auto& ext_func_args = ext_func_sig->getArgs();
  CHECK_LE(function_oper->getArity(), ext_func_args.size());
  std::vector<llvm::Value*> args;
  // i: argument in RA for the function op
  // j: extra offset in orig_arg_lvs (to account for additional values required for a col,
  // e.g. array cols) k: origin_arg_lvs counter
  for (size_t i = 0, j = 0, k = 0; i < function_oper->getArity(); ++i, ++k) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    llvm::Value* arg_lv{nullptr};
    if (arg_ti.is_array()) {
      bool const_arr = (const_arr_size.count(orig_arg_lvs[k]) > 0);
      const auto elem_ti = arg_ti.get_elem_type();
      // TODO: switch to fast fixlen variants
      const auto ptr_lv = (const_arr)
                              ? orig_arg_lvs[k]
                              : cgen_state_->emitExternalCall(
                                    "array_buff",
                                    llvm::Type::getInt8PtrTy(cgen_state_->context_),
                                    {orig_arg_lvs[k], posArg(arg)});
      const auto len_lv =
          (const_arr) ? const_arr_size.at(orig_arg_lvs[k])
                      : cgen_state_->emitExternalCall(
                            "array_size",
                            get_int_type(32, cgen_state_->context_),
                            {orig_arg_lvs[k],
                             posArg(arg),
                             cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))});
      args.push_back(castArrayPointer(ptr_lv, elem_ti));
      args.push_back(cgen_state_->ir_builder_.CreateZExt(
          len_lv, get_int_type(64, cgen_state_->context_)));
      j++;
    } else if (arg_ti.is_geometry()) {
      // Coords
      bool const_arr = (const_arr_size.count(orig_arg_lvs[k]) > 0);
      // NOTE(adb): We're generating code to handle the TINYINT array only -- the actual
      // geo encoding (or lack thereof) does not matter here
      const auto elem_ti = SQLTypeInfo(SQLTypes::kARRAY,
                                       0,
                                       0,
                                       false,
                                       EncodingType::kENCODING_NONE,
                                       0,
                                       SQLTypes::kTINYINT)
                               .get_elem_type();
      llvm::Value* ptr_lv;
      llvm::Value* len_lv;
      int32_t fixlen = -1;
      if (arg_ti.get_type() == kPOINT) {
        const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(arg);
        if (col_var) {
          const auto coords_cd = executor()->getPhysicalColumnDescriptor(col_var, 1);
          if (coords_cd && coords_cd->columnType.get_type() == kARRAY) {
            fixlen = coords_cd->columnType.get_size();
          }
        }
      }
      if (fixlen > 0) {
        ptr_lv =
            cgen_state_->emitExternalCall("fast_fixlen_array_buff",
                                          llvm::Type::getInt8PtrTy(cgen_state_->context_),
                                          {orig_arg_lvs[k], posArg(arg)});
        len_lv = cgen_state_->llInt(int64_t(fixlen));
      } else {
        // TODO: remove const_arr  and related code if it's not needed
        ptr_lv = (const_arr) ? orig_arg_lvs[k]
                             : cgen_state_->emitExternalCall(
                                   "array_buff",
                                   llvm::Type::getInt8PtrTy(cgen_state_->context_),
                                   {orig_arg_lvs[k], posArg(arg)});
        len_lv = (const_arr)
                     ? const_arr_size.at(orig_arg_lvs[k])
                     : cgen_state_->emitExternalCall(
                           "array_size",
                           get_int_type(32, cgen_state_->context_),
                           {orig_arg_lvs[k],
                            posArg(arg),
                            cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))});
      }
      args.push_back(castArrayPointer(ptr_lv, elem_ti));
      args.push_back(cgen_state_->ir_builder_.CreateZExt(
          len_lv, get_int_type(64, cgen_state_->context_)));
      j++;
      switch (arg_ti.get_type()) {
        case kPOINT:
        case kLINESTRING:
          break;
        case kPOLYGON: {
          k++;
          // Ring Sizes
          const auto elem_ti = SQLTypeInfo(SQLTypes::kARRAY,
                                           0,
                                           0,
                                           false,
                                           EncodingType::kENCODING_NONE,
                                           0,
                                           SQLTypes::kINT)
                                   .get_elem_type();
          const auto ptr_lv = cgen_state_->emitExternalCall(
              "array_buff",
              llvm::Type::getInt32PtrTy(cgen_state_->context_),
              {orig_arg_lvs[k], posArg(arg)});
          const auto len_lv = cgen_state_->emitExternalCall(
              "array_size",
              get_int_type(32, cgen_state_->context_),
              {orig_arg_lvs[k],
               posArg(arg),
               cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))});
          args.push_back(castArrayPointer(ptr_lv, elem_ti));
          args.push_back(cgen_state_->ir_builder_.CreateZExt(
              len_lv, get_int_type(64, cgen_state_->context_)));
          j++;
        } break;
        case kMULTIPOLYGON: {
          k++;
          // Ring Sizes
          {
            const auto elem_ti = SQLTypeInfo(SQLTypes::kARRAY,
                                             0,
                                             0,
                                             false,
                                             EncodingType::kENCODING_NONE,
                                             0,
                                             SQLTypes::kINT)
                                     .get_elem_type();
            const auto ptr_lv = cgen_state_->emitExternalCall(
                "array_buff",
                llvm::Type::getInt32PtrTy(cgen_state_->context_),
                {orig_arg_lvs[k], posArg(arg)});
            const auto len_lv = cgen_state_->emitExternalCall(
                "array_size",
                get_int_type(32, cgen_state_->context_),
                {orig_arg_lvs[k],
                 posArg(arg),
                 cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))});
            args.push_back(castArrayPointer(ptr_lv, elem_ti));
            args.push_back(cgen_state_->ir_builder_.CreateZExt(
                len_lv, get_int_type(64, cgen_state_->context_)));
          }
          j++, k++;

          // Poly Rings
          {
            const auto elem_ti = SQLTypeInfo(SQLTypes::kARRAY,
                                             0,
                                             0,
                                             false,
                                             EncodingType::kENCODING_NONE,
                                             0,
                                             SQLTypes::kINT)
                                     .get_elem_type();
            const auto ptr_lv = cgen_state_->emitExternalCall(
                "array_buff",
                llvm::Type::getInt32PtrTy(cgen_state_->context_),
                {orig_arg_lvs[k], posArg(arg)});
            const auto len_lv = cgen_state_->emitExternalCall(
                "array_size",
                get_int_type(32, cgen_state_->context_),
                {orig_arg_lvs[k],
                 posArg(arg),
                 cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))});
            args.push_back(castArrayPointer(ptr_lv, elem_ti));
            args.push_back(cgen_state_->ir_builder_.CreateZExt(
                len_lv, get_int_type(64, cgen_state_->context_)));
          }
          j++;
        } break;
        default:
          CHECK(false);
      }
    } else {
      const auto arg_target_ti = ext_arg_type_to_type_info(ext_func_args[k + j]);
      if (arg_ti.get_type() != arg_target_ti.get_type()) {
        arg_lv = codegenCast(orig_arg_lvs[k], arg_ti, arg_target_ti, false, co);
      } else {
        arg_lv = orig_arg_lvs[k];
      }
      CHECK_EQ(arg_lv->getType(),
               ext_arg_type_to_llvm_type(ext_func_args[k + j], cgen_state_->context_));
      args.push_back(arg_lv);
    }
  }
  return args;
}

llvm::Value* CodeGenerator::castArrayPointer(llvm::Value* ptr,
                                             const SQLTypeInfo& elem_ti) {
  if (elem_ti.get_type() == kFLOAT) {
    return cgen_state_->ir_builder_.CreatePointerCast(
        ptr, llvm::Type::getFloatPtrTy(cgen_state_->context_));
  }
  if (elem_ti.get_type() == kDOUBLE) {
    return cgen_state_->ir_builder_.CreatePointerCast(
        ptr, llvm::Type::getDoublePtrTy(cgen_state_->context_));
  }
  CHECK(elem_ti.is_integer() ||
        (elem_ti.is_string() && elem_ti.get_compression() == kENCODING_DICT));
  switch (elem_ti.get_size()) {
    case 1:
      return cgen_state_->ir_builder_.CreatePointerCast(
          ptr, llvm::Type::getInt8PtrTy(cgen_state_->context_));
    case 2:
      return cgen_state_->ir_builder_.CreatePointerCast(
          ptr, llvm::Type::getInt16PtrTy(cgen_state_->context_));
    case 4:
      return cgen_state_->ir_builder_.CreatePointerCast(
          ptr, llvm::Type::getInt32PtrTy(cgen_state_->context_));
    case 8:
      return cgen_state_->ir_builder_.CreatePointerCast(
          ptr, llvm::Type::getInt64PtrTy(cgen_state_->context_));
    default:
      CHECK(false);
  }
  return nullptr;
}
