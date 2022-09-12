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

#include "CodeGenerator.h"
#include "QueryEngine/Compiler/Backend.h"
#include "ScalarExprVisitor.h"

namespace {

class UsedColumnExpressions : public ScalarExprVisitor<ScalarCodeGenerator::ColumnMap> {
 protected:
  ScalarCodeGenerator::ColumnMap visitColumnVar(
      const hdk::ir::ColumnVar* column) const override {
    ScalarCodeGenerator::ColumnMap m;
    InputColDescriptor input_desc(column->get_column_info(), column->get_rte_idx());
    m.emplace(input_desc,
              std::static_pointer_cast<hdk::ir::ColumnVar>(column->deep_copy()));
    return m;
  }

  ScalarCodeGenerator::ColumnMap aggregateResult(
      const ScalarCodeGenerator::ColumnMap& aggregate,
      const ScalarCodeGenerator::ColumnMap& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

std::vector<InputTableInfo> g_table_infos;

llvm::Type* llvm_type_from_sql(const SQLTypeInfo& ti, llvm::LLVMContext& ctx) {
  switch (ti.get_type()) {
    case kINT: {
      return get_int_type(32, ctx);
    }
    default: {
      LOG(FATAL) << "Unsupported type";
      return nullptr;  // satisfy -Wreturn-type
    }
  }
}

}  // namespace

ScalarCodeGenerator::ColumnMap ScalarCodeGenerator::prepare(const hdk::ir::Expr* expr) {
  UsedColumnExpressions visitor;
  const auto used_columns = visitor.visit(expr);
  std::list<std::shared_ptr<const InputColDescriptor>> global_col_ids;
  for (const auto& used_column : used_columns) {
    global_col_ids.push_back(std::make_shared<InputColDescriptor>(used_column.first));
  }
  plan_state_->allocateLocalColumnIds(global_col_ids);
  return used_columns;
}

ScalarCodeGenerator::CompiledExpression ScalarCodeGenerator::compile(
    const hdk::ir::Expr* expr,
    const bool fetch_columns,
    const CompilationOptions& co,
    const compiler::CodegenTraits& traits) {
  own_plan_state_ =
      std::make_unique<PlanState>(false, std::vector<InputTableInfo>{}, nullptr);
  plan_state_ = own_plan_state_.get();
  const auto used_columns = prepare(expr);
  std::vector<llvm::Type*> arg_types(plan_state_->global_to_local_col_ids_.size() + 1);
  std::vector<std::shared_ptr<hdk::ir::ColumnVar>> inputs(arg_types.size() - 1);
  auto& ctx = module_->getContext();
  for (const auto& kv : plan_state_->global_to_local_col_ids_) {
    size_t arg_idx = kv.second;
    CHECK_LT(arg_idx, arg_types.size());
    const auto it = used_columns.find(kv.first);
    const auto col_expr = it->second;
    inputs[arg_idx] = col_expr;
    const auto& ti = col_expr->get_type_info();
    arg_types[arg_idx + 1] = llvm_type_from_sql(ti, ctx);
  }
  arg_types[0] = traits.globalPointerType(llvm_type_from_sql(expr->get_type_info(), ctx));
  auto ft = llvm::FunctionType::get(get_int_type(32, ctx), arg_types, false);
  auto scalar_expr_func = llvm::Function::Create(
      ft, llvm::Function::ExternalLinkage, "scalar_expr", module_.get());
  auto bb_entry = llvm::BasicBlock::Create(ctx, ".entry", scalar_expr_func, 0);
  // Scalar Code Generator uses the provided module, pass nullptr for extension module
  own_cgen_state_ = std::make_unique<CgenState>(
      g_table_infos.size(), false, /*extension_module_context=*/nullptr);
  own_cgen_state_->module_ = module_.get();
  own_cgen_state_->row_func_ = own_cgen_state_->current_func_ = scalar_expr_func;
  own_cgen_state_->ir_builder_.SetInsertPoint(bb_entry);
  cgen_state_ = own_cgen_state_.get();
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto expr_lvs = codegen(expr, fetch_columns, co);
  CHECK_EQ(expr_lvs.size(), size_t(1));
  cgen_state_->ir_builder_.CreateStore(expr_lvs.front(),
                                       cgen_state_->row_func_->arg_begin());
  cgen_state_->ir_builder_.CreateRet(ll_int<int32_t>(0, ctx));
  if (co.device_type == ExecutorDeviceType::GPU) {
    std::vector<llvm::Type*> wrapper_arg_types(arg_types.size() + 1);
    wrapper_arg_types[0] = traits.globalPointerType(get_int_type(32, ctx));
    wrapper_arg_types[1] = arg_types[0];
    for (size_t i = 1; i < arg_types.size(); ++i) {
      wrapper_arg_types[i + 1] = traits.globalPointerType(arg_types[i]);
    }
    auto wrapper_ft =
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapper_arg_types, false);
    auto wrapper_scalar_expr_func =
        llvm::Function::Create(wrapper_ft,
                               llvm::Function::ExternalLinkage,
                               "wrapper_scalar_expr",
                               module_.get());
    auto wrapper_bb_entry =
        llvm::BasicBlock::Create(ctx, ".entry", wrapper_scalar_expr_func, 0);
    llvm::IRBuilder<> b(ctx);
    b.SetInsertPoint(wrapper_bb_entry);
    std::vector<llvm::Value*> loaded_args = {wrapper_scalar_expr_func->arg_begin() + 1};
    for (size_t i = 2; i < wrapper_arg_types.size(); ++i) {
      auto* value = wrapper_scalar_expr_func->arg_begin() + i;
      loaded_args.push_back(
          b.CreateLoad(value->getType()->getPointerElementType(), value));
    }
    auto error_lv = b.CreateCall(scalar_expr_func, loaded_args);
    b.CreateStore(error_lv, wrapper_scalar_expr_func->arg_begin());
    b.CreateRetVoid();
    return {scalar_expr_func, wrapper_scalar_expr_func, inputs};
  }
  return {scalar_expr_func, nullptr, inputs};
}

std::vector<void*> ScalarCodeGenerator::generateNativeCode(
    Executor* executor,
    const CompiledExpression& compiled_expression,
    const CompilationOptions& co) {
  CHECK(module_) << "Invalid code generator state";
  module_.release();
  switch (co.device_type) {
    case ExecutorDeviceType::CPU: {
      cpu_compilation_context_ = compiler::CPUBackend::generateNativeCPUCode(
          compiled_expression.func, {compiled_expression.func}, co);
      return {cpu_compilation_context_->getPointerToFunction(compiled_expression.func)};
    }
    case ExecutorDeviceType::GPU: {
      return generateNativeGPUCode(
          executor, compiled_expression.func, compiled_expression.wrapper_func, co);
    }
    default: {
      LOG(FATAL) << "Invalid device type";
      return {};  // satisfy -Wreturn-type
    }
  }
}

std::vector<llvm::Value*> ScalarCodeGenerator::codegenColumn(
    const hdk::ir::ColumnVar* column,
    const bool fetch_column,
    const CompilationOptions& co) {
  int arg_idx = plan_state_->getLocalColumnId(column, fetch_column);
  CHECK_LT(static_cast<size_t>(arg_idx), cgen_state_->row_func_->arg_size());
  llvm::Value* arg = cgen_state_->row_func_->arg_begin() + arg_idx + 1;
  return {arg};
}

std::vector<void*> ScalarCodeGenerator::generateNativeGPUCode(
    Executor* executor,
    llvm::Function* func,
    llvm::Function* wrapper_func,
    const CompilationOptions& co) {
#ifdef HAVE_CUDA
  if (!nvptx_target_machine_) {
    nvptx_target_machine_ = compiler::CUDABackend::initializeNVPTXBackend(
        CudaMgr_Namespace::NvidiaDeviceArch::Kepler);
  }
#endif
  if (!gpu_mgr_) {
#ifdef HAVE_CUDA
    gpu_mgr_ = std::make_unique<CudaMgr_Namespace::CudaMgr>(0);
#elif HAVE_L0
    gpu_mgr_ = std::make_unique<l0::L0Manager>();
#endif
  }

  GPUTarget gpu_target = {/*.gpu_mgr=*/gpu_mgr_.get(),
                          /*.block_size=*/gpu_mgr_->getMaxBlockSize(),
                          /*.cgen_state=*/cgen_state_,
                          /*.row_func_not_inlined=*/false};
  switch (gpu_mgr_->getPlatform()) {
    case GpuMgrPlatform::CUDA: {
      auto cuda_context = compiler::CUDABackend::generateNativeGPUCode(
          executor->getExtensionModuleContext()->getExtensionModules(),
          func,
          wrapper_func,
          {func, wrapper_func},
          /*is_gpu_smem_used=*/false,
          co,
          gpu_target,
          nvptx_target_machine_.get());
      gpu_compilation_context_ = cuda_context;
      return cuda_context->getNativeFunctionPointers();
    }
    case GpuMgrPlatform::L0: {
      auto l0_context = compiler::L0Backend::generateNativeGPUCode(
          func, wrapper_func, {func, wrapper_func}, co, gpu_target);
      gpu_compilation_context_ = l0_context;
      return l0_context->getNativeFunctionPointers();
    }
    default:
      CHECK(false) << "Unsupported gpu platform.";
      return {};
  }
}
