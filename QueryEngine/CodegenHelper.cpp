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

#include "CodegenHelper.h"

namespace CodegenUtil {

llvm::Function* findCalledFunction(llvm::CallInst& call_inst) {
  if (llvm::Function* cf = call_inst.getCalledFunction()) {
    return cf;
  } else if (llvm::Value* cv = call_inst.getCalledOperand()) {
    if (llvm::Function* cvf = llvm::dyn_cast<llvm::Function>(cv->stripPointerCasts())) {
      // this happens when bitcast function is called first before calling the actual
      // function i.e., %17 = call i64 bitcast (i64 (i8*)* @actual_func to i64 (i32*)*)
      return cvf;
    }
  }
  return nullptr;
}

std::optional<std::string_view> getCalledFunctionName(llvm::CallInst& call_inst) {
  if (llvm::Function* cf = findCalledFunction(call_inst)) {
    return std::make_optional<std::string_view>(cf->getName().data(),
                                                cf->getName().size());
  }
  return std::nullopt;
}

std::unordered_map<int, llvm::Value*> createPtrWithHoistedMemoryAddr(
    CgenState* cgen_state,
    CodeGenerator* code_generator,
    CompilationOptions const& co,
    llvm::ConstantInt* ptr_int_val,
    llvm::Type* type,
    std::set<int> const& target_device_ids) {
  if (!co.hoist_literals) {
    std::unordered_map<int, llvm::Value*> literal_ptr;
    for (auto const device_id : target_device_ids) {
      literal_ptr.emplace(device_id,
                          cgen_state->ir_builder_.CreateIntToPtr(ptr_int_val, type));
    }
    return literal_ptr;
  }
  Datum d;
  d.bigintval = ptr_int_val->getSExtValue();
  auto ptr = makeExpr<Analyzer::Constant>(kBIGINT, false, d);
  std::unordered_map<int, llvm::Value*> hoisted_literal_ptr;
  std::unordered_map<int, const Analyzer::Constant*> constant_per_device;
  for (auto const device_id : target_device_ids) {
    constant_per_device.emplace(device_id, ptr.get());
  }
  auto hoisted_literal_lvs =
      code_generator->codegenHoistedConstants(constant_per_device, kENCODING_NONE, {});
  for (auto const device_id : target_device_ids) {
    hoisted_literal_ptr.emplace(
        device_id,
        cgen_state->ir_builder_.CreateIntToPtr(hoisted_literal_lvs.front(), type));
  }
  return hoisted_literal_ptr;
}

// todo (yoonmin): support String literal
std::unordered_map<int, llvm::Value*> hoistLiteral(
    CodeGenerator* code_generator,
    CompilationOptions const& co,
    Datum d,
    SQLTypeInfo type,
    std::set<int> const& target_device_ids) {
  CHECK(co.hoist_literals);
  CHECK(type.is_integer() || type.is_decimal() || type.is_fp() || type.is_boolean());
  auto literal_expr = makeExpr<Analyzer::Constant>(type, false, d);
  std::unordered_map<int, llvm::Value*> hoisted_literal_ptr;
  std::unordered_map<int, const Analyzer::Constant*> constant_per_device;
  for (auto const device_id : target_device_ids) {
    constant_per_device.emplace(device_id, literal_expr.get());
  }
  auto hoisted_literal_lvs =
      code_generator->codegenHoistedConstants(constant_per_device, kENCODING_NONE, {});
  for (auto const device_id : target_device_ids) {
    hoisted_literal_ptr.emplace(device_id, hoisted_literal_lvs.front());
  }
  return hoisted_literal_ptr;
}

}  // namespace CodegenUtil
