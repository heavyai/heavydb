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
#include "ExtensionFunctionsBinding.h"
#include "ExtensionFunctionsWhitelist.h"
#include "ExtensionFunctions.hpp"

namespace {

llvm::Type* ext_arg_type_to_llvm_type(const ExtArgumentType ext_arg_type, llvm::LLVMContext& ctx) {
  switch (ext_arg_type) {
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

bool ext_func_call_requires_nullcheck(const Analyzer::FunctionOper* function_oper) {
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    if (!arg_ti.get_notnull() && !arg_ti.is_array()) {
      return true;
    }
  }
  return false;
}

}  // namespace

llvm::Value* Executor::codegenFunctionOper(const Analyzer::FunctionOper* function_oper, const CompilationOptions& co) {
  const auto ext_func_sigs = ExtensionFunctionsWhitelist::get(function_oper->getName());
  if (!ext_func_sigs) {
    throw std::runtime_error("Runtime function " + function_oper->getName() + " not supported");
  }
  CHECK(!ext_func_sigs->empty());
  const auto& ext_func_sig = bind_function(function_oper, *ext_func_sigs);
  const auto& ret_ti = function_oper->get_type_info();
  CHECK(ret_ti.is_integer() || ret_ti.is_fp());
  const auto ret_ty = ret_ti.is_fp() ? (ret_ti.get_type() == kDOUBLE ? llvm::Type::getDoubleTy(cgen_state_->context_)
                                                                     : llvm::Type::getFloatTy(cgen_state_->context_))
                                     : get_int_type(ret_ti.get_logical_size() * 8, cgen_state_->context_);
  if (ret_ty != ext_arg_type_to_llvm_type(ext_func_sig.getRet(), cgen_state_->context_)) {
    throw std::runtime_error("Inconsistent return type for " + function_oper->getName());
  }
  std::vector<llvm::Value*> orig_arg_lvs;
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg_lvs = codegen(function_oper->getArg(i), true, co);
    CHECK_EQ(size_t(1), arg_lvs.size());
    orig_arg_lvs.push_back(arg_lvs.front());
  }
  // The extension function implementations don't handle NULL, they work under
  // the assumption that the inputs are validated before calling them. Generate
  // code to do the check at the call site: if any argument is NULL, return NULL
  // without calling the function at all.
  const auto bbs = beginArgsNullcheck(function_oper, orig_arg_lvs);
  CHECK_EQ(orig_arg_lvs.size(), function_oper->getArity());
  // Arguments must be converted to the types the extension function can handle.
  const auto args = codegenFunctionOperCastArgs(function_oper, &ext_func_sig, orig_arg_lvs, co);
  auto ext_call = cgen_state_->emitExternalCall(ext_func_sig.getName(), ret_ty, args);
  return endArgsNullcheck(bbs, ext_call, function_oper);
}

// Start the control flow needed for a call site check of NULL arguments.
Executor::ArgNullcheckBBs Executor::beginArgsNullcheck(const Analyzer::FunctionOper* function_oper,
                                                       const std::vector<llvm::Value*>& orig_arg_lvs) {
  llvm::BasicBlock* args_null_bb{nullptr};
  llvm::BasicBlock* args_notnull_bb{nullptr};
  llvm::BasicBlock* orig_bb = cgen_state_->ir_builder_.GetInsertBlock();
  // Only generate the check if required (at least one argument must be nullable).
  if (ext_func_call_requires_nullcheck(function_oper)) {
    const auto args_notnull_lv =
        cgen_state_->ir_builder_.CreateNot(codegenFunctionOperNullArg(function_oper, orig_arg_lvs));
    args_notnull_bb = llvm::BasicBlock::Create(cgen_state_->context_, "args_notnull", cgen_state_->row_func_);
    args_null_bb = llvm::BasicBlock::Create(cgen_state_->context_, "args_null", cgen_state_->row_func_);
    cgen_state_->ir_builder_.CreateCondBr(args_notnull_lv, args_notnull_bb, args_null_bb);
    cgen_state_->ir_builder_.SetInsertPoint(args_notnull_bb);
  }
  return {args_null_bb, args_notnull_bb, orig_bb};
}

// Wrap up the control flow needed for NULL argument handling.
llvm::Value* Executor::endArgsNullcheck(const ArgNullcheckBBs& bbs,
                                        llvm::Value* fn_ret_lv,
                                        const Analyzer::FunctionOper* function_oper) {
  if (bbs.args_null_bb) {
    CHECK(bbs.args_notnull_bb);
    cgen_state_->ir_builder_.CreateBr(bbs.args_null_bb);
    cgen_state_->ir_builder_.SetInsertPoint(bbs.args_null_bb);
    auto ext_call_phi = cgen_state_->ir_builder_.CreatePHI(fn_ret_lv->getType(), 2);
    ext_call_phi->addIncoming(fn_ret_lv, bbs.args_notnull_bb);
    const auto& ret_ti = function_oper->get_type_info();
    const auto null_lv = ret_ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(ret_ti))
                                        : static_cast<llvm::Value*>(inlineIntNull(ret_ti));
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

llvm::Value* Executor::codegenFunctionOperWithCustomTypeHandling(
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
      const std::string func_name = (function_oper->getName() == "FLOOR") ? "decimal_floor" : "decimal_ceil";
      const auto covar_result_lv = cgen_state_->emitCall(func_name, {arg_lv, ll_int(exp_to_scale(arg_ti.get_scale()))});
      const auto ret_ti = function_oper->get_type_info();
      CHECK(ret_ti.is_decimal());
      CHECK_EQ(0, ret_ti.get_scale());
      const auto result_lv =
          cgen_state_->ir_builder_.CreateSDiv(covar_result_lv, ll_int(exp_to_scale(arg_ti.get_scale())));
      return endArgsNullcheck(bbs, result_lv, function_oper);
    }
    throw std::runtime_error("Type combination not supported for function " + function_oper->getName());
  }
  return codegenFunctionOper(function_oper, co);
}

// Generates code which returns true iff at least one of the arguments is NULL.
llvm::Value* Executor::codegenFunctionOperNullArg(const Analyzer::FunctionOper* function_oper,
                                                  const std::vector<llvm::Value*>& orig_arg_lvs) {
  llvm::Value* one_arg_null = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), false);
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    if (arg_ti.get_notnull() || arg_ti.is_array()) {
      continue;
    }
    CHECK(arg_ti.is_number());
    one_arg_null = cgen_state_->ir_builder_.CreateOr(one_arg_null, codegenIsNullNumber(orig_arg_lvs[i], arg_ti));
  }
  return one_arg_null;
}

// Generate CAST operations for arguments in `orig_arg_lvs` to the types required by `ext_func_sig`.
std::vector<llvm::Value*> Executor::codegenFunctionOperCastArgs(const Analyzer::FunctionOper* function_oper,
                                                                const ExtensionFunction* ext_func_sig,
                                                                const std::vector<llvm::Value*>& orig_arg_lvs,
                                                                const CompilationOptions& co) {
  CHECK(ext_func_sig);
  const auto& ext_func_args = ext_func_sig->getArgs();
  CHECK_LE(function_oper->getArity(), ext_func_args.size());
  std::vector<llvm::Value*> args;
  for (size_t i = 0; i < function_oper->getArity(); ++i) {
    const auto arg = function_oper->getArg(i);
    const auto& arg_ti = arg->get_type_info();
    llvm::Value* arg_lv{nullptr};
    if (arg_ti.is_array()) {
      const auto elem_ti = arg_ti.get_elem_type();
      const auto ptr_lv = cgen_state_->emitExternalCall(
          "array_buff", llvm::Type::getInt8PtrTy(cgen_state_->context_), {orig_arg_lvs[i], posArg(arg)});
      const auto len_lv =
          cgen_state_->emitExternalCall("array_size",
                                        get_int_type(32, cgen_state_->context_),
                                        {orig_arg_lvs[i], posArg(arg), ll_int(log2_bytes(elem_ti.get_logical_size()))});
      args.push_back(castArrayPointer(ptr_lv, elem_ti));
      args.push_back(cgen_state_->ir_builder_.CreateZExt(len_lv, get_int_type(64, cgen_state_->context_)));
    } else {
      const auto arg_target_ti = ext_arg_type_to_type_info(ext_func_args[i]);
      if (arg_ti.get_type() != arg_target_ti.get_type()) {
        arg_lv = codegenCast(orig_arg_lvs[i], arg_ti, arg_target_ti, false, co);
      } else {
        arg_lv = orig_arg_lvs[i];
      }
      CHECK_EQ(arg_lv->getType(), ext_arg_type_to_llvm_type(ext_func_args[i], cgen_state_->context_));
      args.push_back(arg_lv);
    }
  }
  return args;
}

llvm::Value* Executor::castArrayPointer(llvm::Value* ptr, const SQLTypeInfo& elem_ti) {
  switch (elem_ti.get_size()) {
    case 1:
      return ptr;
    case 2:
      return cgen_state_->ir_builder_.CreatePointerCast(ptr, llvm::Type::getInt16PtrTy(cgen_state_->context_));
    case 4:
      return cgen_state_->ir_builder_.CreatePointerCast(ptr, llvm::Type::getInt32PtrTy(cgen_state_->context_));
    case 8:
      return cgen_state_->ir_builder_.CreatePointerCast(ptr, llvm::Type::getInt64PtrTy(cgen_state_->context_));
    default:
      CHECK(false);
  }
  return nullptr;
}
