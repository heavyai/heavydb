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

#include "MaxwellCodegenPatch.h"

llvm::Value* Executor::spillDoubleElement(llvm::Value* elem_val, llvm::Type* elem_ty) {
  auto var_ptr = cgen_state_->ir_builder_.CreateAlloca(elem_ty);
  cgen_state_->ir_builder_.CreateStore(elem_val, var_ptr);
  return var_ptr;
}

bool Executor::isArchMaxwell(const ExecutorDeviceType dt) const {
  return dt == ExecutorDeviceType::GPU &&
         catalog_->getDataMgr().getCudaMgr()->isArchMaxwell();
}

bool GroupByAndAggregate::needsUnnestDoublePatch(llvm::Value* val_ptr,
                                                 const std::string& agg_base_name,
                                                 const CompilationOptions& co) const {
  return (executor_->isArchMaxwell(co.device_type_) &&
          query_mem_desc_.threadsShareMemory() && llvm::isa<llvm::AllocaInst>(val_ptr) &&
          val_ptr->getType() ==
              llvm::Type::getDoublePtrTy(executor_->cgen_state_->context_) &&
          "agg_id" == agg_base_name);
}

void GroupByAndAggregate::prependForceSync() {
  executor_->cgen_state_->ir_builder_.CreateCall(
      executor_->cgen_state_->module_->getFunction("force_sync"));
}
