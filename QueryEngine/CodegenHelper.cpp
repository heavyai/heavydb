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

// currently, we assume that when this is called for GPUs,
// the argument `num_devices_to_hoist_literal` indicates the number of devices
// that the system has equipped with which is reasonable under the current
// query execution logic, but when supporting a query execution with a specific
// set of devices, we must need to classify the exact devices to hoist the literal
// up for correctness
// todo (yoonmin) : support this to specific set of devices
std::vector<llvm::Value*> createPtrWithHoistedMemoryAddr(
    CgenState* cgen_state,
    CodeGenerator* code_generator,
    CompilationOptions const& co,
    llvm::ConstantInt* ptr_int_val,
    llvm::Type* type,
    size_t num_devices_to_hoist_literal) {
  if (!co.hoist_literals) {
    return {cgen_state->ir_builder_.CreateIntToPtr(ptr_int_val, type)};
  }
  Datum d;
  d.bigintval = ptr_int_val->getSExtValue();
  auto ptr = makeExpr<Analyzer::Constant>(kBIGINT, false, d);
  std::vector<Analyzer::Constant const*> literals(num_devices_to_hoist_literal,
                                                  ptr.get());
  auto hoisted_literal_lvs =
      code_generator->codegenHoistedConstants(literals, kENCODING_NONE, {});
  std::vector<llvm::Value*> hoisted_ptrs;
  hoisted_ptrs.reserve(num_devices_to_hoist_literal);
  for (size_t device_id = 0; device_id < num_devices_to_hoist_literal; device_id++) {
    hoisted_ptrs[device_id] =
        cgen_state->ir_builder_.CreateIntToPtr(hoisted_literal_lvs[device_id], type);
  }
  return hoisted_ptrs;
}

}  // namespace CodegenUtil
