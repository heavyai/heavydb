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
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "QueryEngine/CodeGenerator.h"

extern std::unique_ptr<llvm::Module> g_rt_module;
extern std::unique_ptr<llvm::Module> rt_udf_cpu_module;

namespace {

llvm::Function* generate_entry_point(const CgenState* cgen_state) {
  auto& ctx = cgen_state->context_;
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto ppi8_type = llvm::PointerType::get(pi8_type, 0);
  const auto pi64_type = llvm::PointerType::get(get_int_type(64, ctx), 0);
  const auto ppi64_type = llvm::PointerType::get(pi64_type, 0);
  const auto i32_type = get_int_type(32, ctx);

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

  std::unique_ptr<llvm::Module> module(runtime_module_shallow_copy(cgen_state));
  cgen_state->module_ = module.get();

  entry_point_func_ = generate_entry_point(cgen_state);
  module_ = std::move(module);
}

void TableFunctionCompilationContext::compile(const TableFunctionExecutionUnit& exe_unit,
                                              const CompilationOptions& co,
                                              Executor* executor) {
  generateEntryPoint(exe_unit);
  if (co.device_type == ExecutorDeviceType::GPU) {
    generateGpuKernel();
  }
  finalize(co, executor);
}

void TableFunctionCompilationContext::generateEntryPoint(
    const TableFunctionExecutionUnit& exe_unit) {
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

  auto col_heads = generate_column_heads_load(
      exe_unit.input_exprs.size(), input_cols_arg, cgen_state->ir_builder_, ctx);
  CHECK_EQ(exe_unit.input_exprs.size(), col_heads.size());

  std::vector<llvm::Value*> func_args;
  for (size_t i = 0; i < exe_unit.input_exprs.size(); i++) {
    const auto& expr = exe_unit.input_exprs[i];
    const auto& ti = expr->get_type_info();
    if (ti.is_fp()) {
      func_args.push_back(cgen_state->ir_builder_.CreateBitCast(
          col_heads[i], llvm::PointerType::get(get_fp_type(get_bit_width(ti), ctx), 0)));
    } else if (ti.is_integer()) {
      func_args.push_back(cgen_state->ir_builder_.CreateBitCast(
          col_heads[i], llvm::PointerType::get(get_int_type(get_bit_width(ti), ctx), 0)));
    } else {
      throw std::runtime_error(
          "Only integer and floating point columns are supported as inputs to table "
          "functions.");
    }
  }

  func_args.push_back(input_row_count);
  func_args.push_back(output_row_count_ptr);

  for (size_t i = 0; i < exe_unit.target_exprs.size(); i++) {
    auto output_load = cgen_state->ir_builder_.CreateLoad(
        cgen_state->ir_builder_.CreateGEP(output_buffers_arg, cgen_state_->llInt(i)));
    const auto& ti = exe_unit.target_exprs[i]->get_type_info();
    if (ti.is_fp()) {
      func_args.push_back(cgen_state->ir_builder_.CreateBitCast(
          output_load, llvm::PointerType::get(get_fp_type(get_bit_width(ti), ctx), 0)));
    } else if (ti.is_integer()) {
      func_args.push_back(cgen_state->ir_builder_.CreateBitCast(
          output_load, llvm::PointerType::get(get_int_type(get_bit_width(ti), ctx), 0)));
    } else {
      throw std::runtime_error(
          "Only integer and floating point columns are supported as outputs to table "
          "functions.");
    }
  }

  auto func_name = exe_unit.table_func_name;
  boost::algorithm::to_lower(func_name);
  const auto table_func_return =
      cgen_state->emitExternalCall(func_name, get_int_type(32, ctx), func_args);
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
  if (rt_udf_cpu_module != nullptr) {
    /*
      TODO 1: eliminate need for OverrideFromSrc
      TODO 2: detect and link only the udf's that are needed
     */
    CodeGenerator::link_udf_module(rt_udf_cpu_module,
                                   *module_,
                                   cgen_state_.get(),
                                   llvm::Linker::Flags::OverrideFromSrc);
  }

  module_.release();
  // Add code to cache?

  LOG(IR) << "Table Function Entry Point IR\n"
          << serialize_llvm_object(entry_point_func_);

  if (co.device_type == ExecutorDeviceType::GPU) {
    LOG(IR) << "Table Function Kernel IR\n" << serialize_llvm_object(kernel_func_);

    CHECK(executor);
    executor->initializeNVPTXBackend();
    const auto cuda_mgr = executor->catalog_->getDataMgr().getCudaMgr();
    CHECK(cuda_mgr);

    CodeGenerator::GPUTarget gpu_target{executor->nvptx_target_machine_.get(),
                                        cuda_mgr,
                                        executor->blockSize(),
                                        cgen_state_.get(),
                                        false};
    gpu_code_ = CodeGenerator::generateNativeGPUCode(entry_point_func_,
                                                     kernel_func_,
                                                     {entry_point_func_, kernel_func_},
                                                     co,
                                                     gpu_target);
  } else {
    auto ee =
        CodeGenerator::generateNativeCPUCode(entry_point_func_, {entry_point_func_}, co);
    func_ptr = reinterpret_cast<FuncPtr>(ee->getPointerToFunction(entry_point_func_));
    own_execution_engine_ = std::move(ee);
  }

  LOG(IR) << "End of IR";
}
