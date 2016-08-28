#include "MaxwellCodegenPatch.h"

llvm::Value* Executor::spillDoubleElement(llvm::Value* elem_val, llvm::Type* elem_ty) {
  auto var_ptr = cgen_state_->ir_builder_.CreateAlloca(elem_ty);
  cgen_state_->ir_builder_.CreateStore(elem_val, var_ptr);
  return var_ptr;
}

bool Executor::isArchMaxwell(const ExecutorDeviceType dt) const {
  return dt == ExecutorDeviceType::GPU && catalog_->get_dataMgr().cudaMgr_->isArchMaxwell();
}

bool GroupByAndAggregate::needsUnnestDoublePatch(llvm::Value* val_ptr,
                                                 const std::string& agg_base_name,
                                                 const CompilationOptions& co) const {
  return (executor_->isArchMaxwell(co.device_type_) && query_mem_desc_.threadsShareMemory() &&
          llvm::isa<llvm::AllocaInst>(val_ptr) &&
          val_ptr->getType() == llvm::Type::getDoublePtrTy(executor_->cgen_state_->context_) &&
          "agg_id" == agg_base_name);
}

void GroupByAndAggregate::prependForceSync() {
  executor_->cgen_state_->ir_builder_.CreateCall(executor_->cgen_state_->module_->getFunction("force_sync"));
}
