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
extern std::unique_ptr<llvm::Module> rt_udf_gpu_module;

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
  const auto input_row_counts = &*(++arg_it);
  input_row_counts->setName("input_row_counts");
  const auto output_buffers = &*(++arg_it);
  output_buffers->setName("output_buffers");
  const auto output_row_count = &*(++arg_it);
  output_row_count->setName("output_row_count");
  return func;
}

inline llvm::Type* get_llvm_type_from_sql_column_type(const SQLTypeInfo elem_ti,
                                                      llvm::LLVMContext& ctx) {
  if (elem_ti.is_fp()) {
    switch (elem_ti.get_size()) {
      case 4:
        return llvm::Type::getFloatPtrTy(ctx);
      case 8:
        return llvm::Type::getDoublePtrTy(ctx);
    }
  }
  if (elem_ti.is_boolean()) {
    return llvm::Type::getInt8PtrTy(ctx);
  }
  CHECK(elem_ti.is_integer());
  switch (elem_ti.get_size()) {
    case 1:
      return llvm::Type::getInt8PtrTy(ctx);
    case 2:
      return llvm::Type::getInt16PtrTy(ctx);
    case 4:
      return llvm::Type::getInt32PtrTy(ctx);
    case 8:
      return llvm::Type::getInt64PtrTy(ctx);
  }
  LOG(FATAL) << "get_llvm_type_from_sql_column_type: not implemented for "
             << ::toString(elem_ti);
  return nullptr;
}

llvm::Value* alloc_column(std::string col_name,
                          const SQLTypeInfo& data_target_info,
                          llvm::Value* data_ptr,
                          llvm::Value* data_size,
                          llvm::LLVMContext& ctx,
                          llvm::IRBuilder<>& ir_builder,
                          bool byval) {
  /*
    Creates a new Column instance of given element type and initialize
    its data ptr and sz members. If data ptr or sz are unspecified
    (have nullptr values) then the corresponding members are
    initialized with NULL and -1, respectively.
   */
  llvm::Type* data_ptr_llvm_type =
      get_llvm_type_from_sql_column_type(data_target_info, ctx);
  llvm::StructType* col_struct_type =
      llvm::StructType::get(ctx,
                            {
                                data_ptr_llvm_type,         /* T* ptr */
                                llvm::Type::getInt64Ty(ctx) /* int64_t sz */
                            });
  auto col = ir_builder.CreateAlloca(col_struct_type);
  col->setName(col_name);
  auto col_ptr_ptr = ir_builder.CreateStructGEP(col_struct_type, col, 0);
  auto col_sz_ptr = ir_builder.CreateStructGEP(col_struct_type, col, 1);
  col_ptr_ptr->setName(col_name + ".ptr");
  col_sz_ptr->setName(col_name + ".sz");

  if (data_ptr != nullptr) {
    if (data_ptr->getType() == data_ptr_llvm_type->getPointerElementType()) {
      ir_builder.CreateStore(data_ptr, col_ptr_ptr);
    } else {
      auto tmp = ir_builder.CreateBitCast(data_ptr, data_ptr_llvm_type);
      ir_builder.CreateStore(tmp, col_ptr_ptr);
    }
  } else {
    ir_builder.CreateStore(llvm::Constant::getNullValue(data_ptr_llvm_type), col_ptr_ptr);
  }
  if (data_size != nullptr) {
    auto data_size_type = data_size->getType();
    if (data_size_type->isPointerTy()) {
      CHECK(data_size_type->getPointerElementType()->isIntegerTy(64));
      auto val = ir_builder.CreateLoad(data_size);
      ir_builder.CreateStore(val, col_sz_ptr);
    } else {
      CHECK(data_size_type->isIntegerTy(64));
      ir_builder.CreateStore(data_size, col_sz_ptr);
    }
  } else {
    auto const_minus1 = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), -1, true);
    ir_builder.CreateStore(const_minus1, col_sz_ptr);
  }

  if (byval) {
    return ir_builder.CreateLoad(col);
  } else {
    auto col_ptr = ir_builder.CreatePointerCast(
        col_ptr_ptr, llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0));
    col_ptr->setName(col_name + "_ptr");
    return col_ptr;
  }
}

}  // namespace

TableFunctionCompilationContext::TableFunctionCompilationContext()
    : cgen_state_(std::make_unique<CgenState>(/*num_query_infos=*/0,
                                              /*contains_left_deep_outer_join=*/false)) {
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
  const auto input_row_counts_arg = &*(++arg_it);
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

  auto row_count_heads = generate_column_heads_load(
      exe_unit.input_exprs.size(), input_row_counts_arg, cgen_state->ir_builder_, ctx);

  // The column arguments of C++ UDTFs processed by clang must be
  // passed by reference, see rbc issue 200.
  auto pass_column_by_value = exe_unit.table_func.isRuntime();
  std::vector<llvm::Value*> func_args;
  for (size_t i = 0; i < exe_unit.input_exprs.size(); i++) {
    const auto& expr = exe_unit.input_exprs[i];
    const auto& ti = expr->get_type_info();
    if (ti.is_fp()) {
      auto r = cgen_state->ir_builder_.CreateBitCast(
          col_heads[i], llvm::PointerType::get(get_fp_type(get_bit_width(ti), ctx), 0));
      func_args.push_back(cgen_state->ir_builder_.CreateLoad(r));
    } else if (ti.is_integer()) {
      auto r = cgen_state->ir_builder_.CreateBitCast(
          col_heads[i], llvm::PointerType::get(get_int_type(get_bit_width(ti), ctx), 0));
      func_args.push_back(cgen_state->ir_builder_.CreateLoad(r));
    } else if (ti.is_column()) {
      auto col = alloc_column(std::string("input_col.") + std::to_string(i),
                              ti.get_elem_type(),
                              col_heads[i],
                              row_count_heads[i],
                              ctx,
                              cgen_state_->ir_builder_,
                              pass_column_by_value);
      func_args.push_back(col);
    } else {
      throw std::runtime_error(
          "Only integer and floating point columns or scalars are supported as inputs to "
          "table "
          "functions, got " +
          ti.get_type_name());
    }
  }
  std::vector<llvm::Value*> output_col_args;
  for (size_t i = 0; i < exe_unit.target_exprs.size(); i++) {
    auto output_load = cgen_state->ir_builder_.CreateLoad(
        cgen_state->ir_builder_.CreateGEP(output_buffers_arg, cgen_state_->llInt(i)));
    const auto& expr = exe_unit.target_exprs[i];
    const auto& ti = expr->get_type_info();
    CHECK(!ti.is_column());  // UDTF output column type is its data type
    auto col = alloc_column(std::string("output_col.") + std::to_string(i),
                            ti,
                            output_load,
                            output_row_count_ptr,
                            ctx,
                            cgen_state_->ir_builder_,
                            pass_column_by_value);
    func_args.push_back(col);
  }
  auto func_name = exe_unit.table_func.getName();
  boost::algorithm::to_lower(func_name);
  const auto table_func_return =
      cgen_state->emitExternalCall(func_name, get_int_type(32, ctx), func_args);
  table_func_return->setName("table_func_ret");

  // If table_func_return is non-negative then store the value in
  // output_row_count and return zero. Otherwise, return
  // table_func_return that negative value contains the error code.
  const auto bb_exit_0 = llvm::BasicBlock::Create(ctx, ".exit0", entry_point_func_);

  auto const_zero = llvm::ConstantInt::get(table_func_return->getType(), 0, true);
  auto is_ok = cgen_state_->ir_builder_.CreateICmpSGE(table_func_return, const_zero);
  cgen_state_->ir_builder_.CreateCondBr(is_ok, bb_exit_0, bb_exit);

  cgen_state_->ir_builder_.SetInsertPoint(bb_exit_0);
  auto r = cgen_state->ir_builder_.CreateIntCast(
      table_func_return, get_int_type(64, ctx), true);
  cgen_state->ir_builder_.CreateStore(r, output_row_count_ptr);
  cgen_state->ir_builder_.CreateRet(const_zero);

  cgen_state->ir_builder_.SetInsertPoint(bb_exit);
  cgen_state->ir_builder_.CreateRet(table_func_return);

  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  cgen_state->ir_builder_.CreateBr(func_body_bb);

  /*
  std::cout << "=================================" << std::endl;
  entry_point_func_->print(llvm::outs());
  std::cout << "=================================" << std::endl;
  */

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
  /*
    TODO 1: eliminate need for OverrideFromSrc
    TODO 2: detect and link only the udf's that are needed
  */
  if (co.device_type == ExecutorDeviceType::GPU && rt_udf_gpu_module != nullptr) {
    CodeGenerator::link_udf_module(rt_udf_gpu_module,
                                   *module_,
                                   cgen_state_.get(),
                                   llvm::Linker::Flags::OverrideFromSrc);
  }
  if (co.device_type == ExecutorDeviceType::CPU && rt_udf_cpu_module != nullptr) {
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
