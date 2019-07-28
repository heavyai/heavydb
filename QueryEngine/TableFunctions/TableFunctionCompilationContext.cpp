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

#include "QueryEngine/TableFunctions/TableFunctionCompilationContext.h"

#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <algorithm>

extern std::unique_ptr<llvm::Module> g_rt_module;

namespace {

// TODO(adb): merge with result set JIT once that lands
std::unique_ptr<llvm::Module> runtime_module_shallow_copy(llvm::LLVMContext& context,
                                                          CgenState* cgen_state) {
  return llvm::CloneModule(
#if LLVM_VERSION_MAJOR >= 7
      *g_rt_module.get(),
#else
      g_rt_module.get(),
#endif
      cgen_state->vmap_,
      [](const llvm::GlobalValue* gv) {
        auto func = llvm::dyn_cast<llvm::Function>(gv);
        if (!func) {
          return true;
        }
        return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
                func->getLinkage() == llvm::GlobalValue::LinkageTypes::InternalLinkage);
      });
}

llvm::Function* generate_entry_point(const CgenState* cgen_state) {
  auto& ctx = cgen_state->context_;
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto ppi8_type = llvm::PointerType::get(pi8_type, 0);
  const auto pi64_type = llvm::PointerType::get(get_int_type(64, ctx), 0);
  const auto ppi64_type = llvm::PointerType::get(pi64_type, 0);
  const auto i32_type = get_int_type(32, ctx);
  const auto i64_type = get_int_type(64, ctx);

  const auto pvoid_type = llvm::PointerType::get(llvm::Type::getVoidTy(ctx), 0);
  const auto func_type = llvm::FunctionType::get(
      i32_type, {ppi8_type, pi64_type, ppi64_type, pi64_type}, false);

  auto func = llvm::Function::Create(func_type,
                                     llvm::Function::ExternalLinkage,
                                     "call_table_function",
                                     cgen_state->module_);
  auto arg_it = func->arg_begin();
  const auto input_cols_arg = &*arg_it;
  input_cols_arg->setName("input_col_buffers");
  const auto input_row_count = &*(++arg_it);
  input_row_count->setName("input_row_count");
  const auto output_buffers = &*(++arg_it);
  output_buffers->setName("output_buffers");
  const auto output_row_count = &*(++arg_it);
  output_row_count->setName("output_row_count");
  return func;
}

}  // namespace

TableFunctionCompilationContext::TableFunctionCompilationContext()
    : cgen_state_(std::make_unique<CgenState>(std::vector<InputTableInfo>{}, false)) {
  auto cgen_state = cgen_state_.get();
  CHECK(cgen_state);
  auto& ctx = cgen_state->context_;

  std::unique_ptr<llvm::Module> module(runtime_module_shallow_copy(ctx, cgen_state));
  cgen_state->module_ = module.get();

  entry_point_func_ = generate_entry_point(cgen_state);
  module_ = std::move(module);
}

void TableFunctionCompilationContext::compile(const TableFunctionExecutionUnit& exe_unit,
                                              const CompilationOptions& co,
                                              Executor* executor) {
  generateEntryPoint(exe_unit.input_exprs.size());
  if (co.device_type_ == ExecutorDeviceType::GPU) {
    generateGpuKernel();
  }
  finalize(co, executor);
}

namespace {

std::vector<llvm::Value*> generate_column_heads_load(const int num_columns,
                                                     llvm::Value* byte_stream_arg,
                                                     CgenState* cgen_state,
                                                     llvm::LLVMContext& ctx) {
  CHECK(cgen_state);
  CHECK(byte_stream_arg);
  auto max_col_local_id = num_columns - 1;

  std::vector<llvm::Value*> col_heads;
  for (int col_id = 0; col_id <= max_col_local_id; ++col_id) {
    col_heads.emplace_back(
        cgen_state->ir_builder_.CreateLoad(cgen_state->ir_builder_.CreateGEP(
            byte_stream_arg,
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), col_id))));
  }
  return col_heads;
}

}  // namespace

void TableFunctionCompilationContext::generateEntryPoint(const size_t in_col_count) {
  CHECK(entry_point_func_);
  auto arg_it = entry_point_func_->arg_begin();
  const auto input_cols_arg = &*arg_it;
  const auto input_row_count = &*(++arg_it);
  const auto output_buffers_arg = &*(++arg_it);
  const auto output_row_count_ptr = &*(++arg_it);

  auto cgen_state = cgen_state_.get();
  CHECK(cgen_state);
  auto& ctx = cgen_state->context_;

  const auto bb_entry = llvm::BasicBlock::Create(ctx, ".entry", entry_point_func_, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);

  const auto bb_exit = llvm::BasicBlock::Create(ctx, ".exit", entry_point_func_);

  const auto func_body_bb = llvm::BasicBlock::Create(
      ctx, ".func_body", cgen_state->ir_builder_.GetInsertBlock()->getParent());
  cgen_state->ir_builder_.SetInsertPoint(func_body_bb);

  auto col_heads =
      generate_column_heads_load(in_col_count, input_cols_arg, cgen_state, ctx);
  CHECK_EQ(in_col_count, col_heads.size());

  // convert the first column from int8_t* to double*
  CHECK(!col_heads.empty());
  auto first_double_col_lv = cgen_state->ir_builder_.CreateBitCast(
      col_heads.front(), llvm::PointerType::get(get_fp_type(64, ctx), 0));
  auto first_int_col_lv = cgen_state->ir_builder_.CreateBitCast(
      col_heads[1], llvm::PointerType::get(get_int_type(32, ctx), 0));
  // prep the output buffer as well
  auto output_buffer_col = cgen_state->ir_builder_.CreateLoad(
      cgen_state->ir_builder_.CreateGEP(output_buffers_arg, cgen_state_->llInt(0)),
      "first_output_column");
  const auto table_func_return = cgen_state->emitExternalCall("my_table_func",
                                                              get_int_type(32, ctx),
                                                              {first_double_col_lv,
                                                               first_int_col_lv,
                                                               input_row_count,
                                                               output_buffer_col,
                                                               output_row_count_ptr});
  table_func_return->setName("table_func_ret");
  cgen_state->ir_builder_.SetInsertPoint(bb_exit);
  cgen_state->ir_builder_.CreateRet(table_func_return);

  cgen_state->ir_builder_.SetInsertPoint(func_body_bb);
  cgen_state->ir_builder_.CreateBr(bb_exit);

  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  cgen_state->ir_builder_.CreateBr(func_body_bb);

  verify_function_ir(entry_point_func_);
}

void TableFunctionCompilationContext::generateGpuKernel() {
  CHECK(entry_point_func_);
  std::vector<llvm::Type*> arg_types;
  arg_types.reserve(entry_point_func_->arg_size());
  std::for_each(entry_point_func_->arg_begin(),
                entry_point_func_->arg_end(),
                [&arg_types](const auto& arg) { arg_types.push_back(arg.getType()); });
  CHECK_EQ(arg_types.size(), entry_point_func_->arg_size());

  auto cgen_state = cgen_state_.get();
  CHECK(cgen_state);
  auto& ctx = cgen_state->context_;

  std::vector<llvm::Type*> wrapper_arg_types(arg_types.size() + 1);
  wrapper_arg_types[0] = llvm::PointerType::get(get_int_type(32, ctx), 0);
  wrapper_arg_types[1] = arg_types[0];

  for (size_t i = 1; i < arg_types.size(); ++i) {
    wrapper_arg_types[i + 1] = arg_types[i];
  }

  auto wrapper_ft =
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), wrapper_arg_types, false);
  kernel_func_ = llvm::Function::Create(wrapper_ft,
                                        llvm::Function::ExternalLinkage,
                                        "table_func_kernel",
                                        cgen_state->module_);

  auto wrapper_bb_entry = llvm::BasicBlock::Create(ctx, ".entry", kernel_func_, 0);
  llvm::IRBuilder<> b(ctx);
  b.SetInsertPoint(wrapper_bb_entry);
  std::vector<llvm::Value*> loaded_args = {kernel_func_->arg_begin() + 1};
  for (size_t i = 2; i < wrapper_arg_types.size(); ++i) {
    loaded_args.push_back(kernel_func_->arg_begin() + i);
  }
  auto error_lv = b.CreateCall(entry_point_func_, loaded_args);
  b.CreateStore(error_lv, kernel_func_->arg_begin());
  b.CreateRetVoid();
}

void TableFunctionCompilationContext::finalize(const CompilationOptions& co,
                                               Executor* executor) {
  module_.release();
  // Add code to cache?

  if (co.device_type_ == ExecutorDeviceType::GPU) {
    CHECK(executor);
    executor->initializeNVPTXBackend();
    const auto cuda_mgr = executor->catalog_->getDataMgr().getCudaMgr();
    CHECK(cuda_mgr);

    CodeGenerator::GPUTarget gpu_target{executor->nvptx_target_machine_.get(),
                                        cuda_mgr,
                                        executor->blockSize(),
                                        cgen_state_.get(),
                                        false};
    gpu_code_ = std::make_unique<CodeGenerator::GPUCode>(
        CodeGenerator::generateNativeGPUCode(entry_point_func_,
                                             kernel_func_,
                                             {entry_point_func_, kernel_func_},
                                             co,
                                             gpu_target));
  } else {
    auto ee =
        CodeGenerator::generateNativeCPUCode(entry_point_func_, {entry_point_func_}, co);
    func_ptr = reinterpret_cast<FuncPtr>(ee->getPointerToFunction(entry_point_func_));
    own_execution_engine_ = std::move(ee);
  }
}
