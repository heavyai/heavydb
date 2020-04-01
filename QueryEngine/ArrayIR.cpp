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
  return codegen(uoper->get_operand(), true, co).front();
}

llvm::Value* CodeGenerator::codegenArrayAt(const Analyzer::BinOper* array_at,
                                           const CompilationOptions& co) {
  const auto arr_expr = array_at->get_left_operand();
  const auto idx_expr = array_at->get_right_operand();
  const auto& idx_ti = idx_expr->get_type_info();
  CHECK(idx_ti.is_integer());
  auto idx_lvs = codegen(idx_expr, true, co);
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
  const auto arr_lvs = codegen(arr_expr, true, co);
  CHECK_EQ(size_t(1), arr_lvs.size());
  return cgen_state_->emitExternalCall(
      array_at_fname,
      ret_ty,
      {arr_lvs.front(),
       posArg(arr_expr),
       idx_lv,
       elem_ti.is_fp() ? static_cast<llvm::Value*>(cgen_state_->inlineFpNull(elem_ti))
                       : static_cast<llvm::Value*>(cgen_state_->inlineIntNull(elem_ti))});
}

llvm::Value* CodeGenerator::codegen(const Analyzer::CardinalityExpr* expr,
                                    const CompilationOptions& co) {
  const auto arr_expr = expr->get_arg();
  const auto& array_ti = arr_expr->get_type_info();
  CHECK(array_ti.is_array());
  const auto& elem_ti = array_ti.get_elem_type();
  auto arr_lv = codegen(arr_expr, true, co);
  std::string fn_name("array_size");

  std::vector<llvm::Value*> array_size_args{
      arr_lv.front(),
      posArg(arr_expr),
      cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))};
  const bool is_nullable{!arr_expr->get_type_info().get_notnull()};
  if (is_nullable) {
    fn_name += "_nullable";
    array_size_args.push_back(cgen_state_->inlineIntNull(expr->get_type_info()));
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
    const auto arg_lvs = codegen(arg, true, co);
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

  if (array_expr->isNull()) {
    return {llvm::ConstantPointerNull::get(
                llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0)),
            cgen_state_->llInt(0)};
  }

  if (0 == array_expr->getElementCount()) {
    llvm::Constant* dead_const = cgen_state_->llInt(0xdead);
    llvm::Value* dead_pointer = llvm::ConstantExpr::getIntToPtr(
        dead_const, llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0));
    return {dead_pointer, cgen_state_->llInt(0)};
  }

  llvm::Value* allocated_target_buffer;
  if (array_expr->isLocalAlloc()) {
    allocated_target_buffer = ir_builder.CreateAlloca(array_type);
  } else {
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }

    allocated_target_buffer =
        cgen_state_->emitExternalCall("allocate_varlen_buffer",
                                      llvm::Type::getInt8PtrTy(cgen_state_->context_),
                                      {cgen_state_->llInt(array_expr->getElementCount()),
                                       cgen_state_->llInt(array_element_size_bytes)});
    cgen_state_->emitExternalCall(
        "register_buffer_with_executor_rsm",
        llvm::Type::getVoidTy(cgen_state_->context_),
        {cgen_state_->llInt(reinterpret_cast<int64_t>(executor())),
         allocated_target_buffer});
  }
  llvm::Value* casted_allocated_target_buffer =
      ir_builder.CreatePointerCast(allocated_target_buffer, array_type->getPointerTo());

  for (size_t i = 0; i < array_expr->getElementCount(); i++) {
    auto* element = argument_list[i];
    auto* element_ptr = ir_builder.CreateGEP(
        array_type,
        casted_allocated_target_buffer,
        std::vector<llvm::Value*>{cgen_state_->llInt(0), cgen_state_->llInt(i)});

    const auto& elem_ti = return_type.get_elem_type();
    if (elem_ti.is_boolean()) {
      const auto byte_casted_bit =
          ir_builder.CreateIntCast(element, array_index_type, true);
      ir_builder.CreateStore(byte_casted_bit, element_ptr);
    } else if (elem_ti.is_fp()) {
      switch (elem_ti.get_size()) {
        case sizeof(double): {
          const auto double_element_ptr = ir_builder.CreatePointerCast(
              element_ptr, llvm::Type::getDoublePtrTy(cgen_state_->context_));
          ir_builder.CreateStore(element, double_element_ptr);
          break;
        }
        case sizeof(float): {
          const auto float_element_ptr = ir_builder.CreatePointerCast(
              element_ptr, llvm::Type::getFloatPtrTy(cgen_state_->context_));
          ir_builder.CreateStore(element, float_element_ptr);
          break;
        }
        default:
          UNREACHABLE();
      }
    } else if (elem_ti.is_integer() || elem_ti.is_decimal() || elem_ti.is_date() ||
               elem_ti.is_timestamp() || elem_ti.is_time() || elem_ti.is_timeinterval() ||
               elem_ti.is_dict_encoded_string()) {
      // TODO(adb): this validation and handling should be done elsewhere
      const auto sign_extended_element = ir_builder.CreateSExt(element, array_index_type);
      ir_builder.CreateStore(sign_extended_element, element_ptr);
    } else {
      throw std::runtime_error("Unsupported type used in ARRAY construction.");
    }
  }

  return {ir_builder.CreateGEP(
              array_type, casted_allocated_target_buffer, cgen_state_->llInt(0)),
          cgen_state_->llInt(array_expr->getElementCount())};
}
