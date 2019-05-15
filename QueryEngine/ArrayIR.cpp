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

llvm::Value* CodeGenerator::codegenUnnest(const Analyzer::UOper* uoper,
                                          const CompilationOptions& co) {
  return executor_->codegen(uoper->get_operand(), true, co).front();
}

llvm::Value* CodeGenerator::codegenArrayAt(const Analyzer::BinOper* array_at,
                                           const CompilationOptions& co) {
  const auto arr_expr = array_at->get_left_operand();
  const auto idx_expr = array_at->get_right_operand();
  const auto& idx_ti = idx_expr->get_type_info();
  CHECK(idx_ti.is_integer());
  auto idx_lvs = executor_->codegen(idx_expr, true, co);
  CHECK_EQ(size_t(1), idx_lvs.size());
  auto idx_lv = idx_lvs.front();
  if (idx_ti.get_logical_size() < 8) {
    idx_lv = cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
                                                 idx_lv,
                                                 get_int_type(64, cgen_state_->context_));
  }
  const auto& array_ti = arr_expr->get_type_info();
  CHECK(array_ti.is_array());
  const auto& elem_ti = array_ti.get_elem_type();
  const std::string array_at_fname{
      elem_ti.is_fp()
          ? "array_at_" + std::string(elem_ti.get_type() == kDOUBLE ? "double_checked"
                                                                    : "float_checked")
          : "array_at_int" + std::to_string(elem_ti.get_logical_size() * 8) +
                "_t_checked"};
  const auto ret_ty =
      elem_ti.is_fp()
          ? (elem_ti.get_type() == kDOUBLE
                 ? llvm::Type::getDoubleTy(cgen_state_->context_)
                 : llvm::Type::getFloatTy(cgen_state_->context_))
          : get_int_type(elem_ti.get_logical_size() * 8, cgen_state_->context_);
  const auto arr_lvs = executor_->codegen(arr_expr, true, co);
  CHECK_EQ(size_t(1), arr_lvs.size());
  return cgen_state_->emitExternalCall(
      array_at_fname,
      ret_ty,
      {arr_lvs.front(),
       posArg(arr_expr),
       idx_lv,
       elem_ti.is_fp() ? static_cast<llvm::Value*>(executor_->inlineFpNull(elem_ti))
                       : static_cast<llvm::Value*>(executor_->inlineIntNull(elem_ti))});
}

llvm::Value* CodeGenerator::codegen(const Analyzer::CardinalityExpr* expr,
                                    const CompilationOptions& co) {
  const auto arr_expr = expr->get_arg();
  const auto& array_ti = arr_expr->get_type_info();
  CHECK(array_ti.is_array());
  const auto& elem_ti = array_ti.get_elem_type();
  auto arr_lv = executor_->codegen(arr_expr, true, co);
  std::string fn_name("array_size");

  std::vector<llvm::Value*> array_size_args{
      arr_lv.front(),
      posArg(arr_expr),
      executor_->ll_int(log2_bytes(elem_ti.get_logical_size()))};
  const bool is_nullable{!arr_expr->get_type_info().get_notnull()};
  if (is_nullable) {
    fn_name += "_nullable";
    array_size_args.push_back(executor_->inlineIntNull(expr->get_type_info()));
  }
  return cgen_state_->emitExternalCall(
      fn_name, get_int_type(32, cgen_state_->context_), array_size_args);
}

std::vector<llvm::Value*> CodeGenerator::codegenArrayExpr(
    Analyzer::ArrayExpr const* array_expr,
    CompilationOptions const& co) {
  using ValueVector = std::vector<llvm::Value*>;
  ValueVector argument_list;
  auto& ir_builder(cgen_state_->ir_builder_);

  const auto& return_type = array_expr->get_type_info();
  for (size_t i = 0; i < array_expr->getElementCount(); i++) {
    const auto arg = array_expr->getElement(i);
    const auto arg_lvs = executor_->codegen(arg, true, co);
    if (arg_lvs.size() == 1) {
      argument_list.push_back(arg_lvs.front());
    } else {
      throw std::runtime_error(
          "Unexpected argument count during array[] code generation.");
    }
  }

  auto array_element_size_bytes =
      return_type.get_elem_type().get_array_context_logical_size();
  auto* array_index_type =
      get_int_type(array_element_size_bytes * 8, cgen_state_->context_);
  auto* array_type = get_int_array_type(
      array_element_size_bytes * 8, array_expr->getElementCount(), cgen_state_->context_);

  llvm::Value* allocated_target_buffer =
      cgen_state_->emitExternalCall("allocate_varlen_buffer",
                                    llvm::Type::getInt8PtrTy(cgen_state_->context_),
                                    {executor_->ll_int(array_expr->getElementCount()),
                                     executor_->ll_int(array_element_size_bytes)});
  cgen_state_->emitExternalCall(
      "register_buffer_with_executor_rsm",
      llvm::Type::getVoidTy(cgen_state_->context_),
      {executor_->ll_int(reinterpret_cast<int64_t>(executor_)), allocated_target_buffer});
  llvm::Value* casted_allocated_target_buffer =
      ir_builder.CreatePointerCast(allocated_target_buffer, array_type->getPointerTo());

  for (size_t i = 0; i < array_expr->getElementCount(); i++) {
    auto* element = argument_list[i];
    auto* element_ptr =
        ir_builder.CreateGEP(array_type,
                             casted_allocated_target_buffer,
                             {executor_->ll_int(0), executor_->ll_int(i)});

    if (is_member_of_typeset<kTINYINT,
                             kSMALLINT,
                             kINT,
                             kBIGINT,
                             kTIMESTAMP,
                             kDATE,
                             kTIME,
                             kNUMERIC,
                             kDECIMAL,
                             kINTERVAL_DAY_TIME,
                             kINTERVAL_YEAR_MONTH,
                             kVARCHAR,
                             kTEXT,
                             kCHAR>(return_type.get_elem_type())) {
      auto sign_extended_element = ir_builder.CreateSExt(element, array_index_type);
      ir_builder.CreateStore(sign_extended_element, element_ptr);
    } else if (is_member_of_typeset<kBOOLEAN>(return_type.get_elem_type())) {
      auto byte_casted_bit = ir_builder.CreateIntCast(element, array_index_type, true);
      ir_builder.CreateStore(byte_casted_bit, element_ptr);
    } else if (is_member_of_typeset<kFLOAT>(return_type.get_elem_type())) {
      auto float_element_ptr = ir_builder.CreatePointerCast(
          element_ptr, llvm::Type::getFloatPtrTy(cgen_state_->context_));
      ir_builder.CreateStore(element, float_element_ptr);
    } else if (is_member_of_typeset<kDOUBLE>(return_type.get_elem_type())) {
      auto double_element_ptr = ir_builder.CreatePointerCast(
          element_ptr, llvm::Type::getDoublePtrTy(cgen_state_->context_));
      ir_builder.CreateStore(element, double_element_ptr);
    } else {
      throw std::runtime_error("Unsupported type used in ARRAY construction.");
    }
  }

  return {ir_builder.CreateGEP(
              array_type, casted_allocated_target_buffer, executor_->ll_int(0)),
          executor_->ll_int(array_expr->getElementCount())};
}
