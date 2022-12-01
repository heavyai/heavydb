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

}  // namespace CodegenUtil
