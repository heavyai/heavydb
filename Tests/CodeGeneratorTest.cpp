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

#include <gtest/gtest.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "IR/Expr.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Compiler/HelperFunctions.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/IRCodegenUtils.h"
#include "QueryEngine/LLVMGlobalContext.h"
#include "TestHelpers.h"

namespace {
#ifdef HAVE_CUDA
using DevicePtr = CUdeviceptr;
using DeviceFuncPtr = CUfunction;
void run_test_kernel(DeviceFuncPtr func, std::vector<void*>& params, GpuMgr*) {
  cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, nullptr, &params[0], nullptr);
}
#elif HAVE_L0
using DevicePtr = int8_t*;
using DeviceFuncPtr = l0::L0Kernel*;
void run_test_kernel(DeviceFuncPtr func, std::vector<void*>& params, GpuMgr* mgr) {
  l0::L0Manager* mgr_ = dynamic_cast<l0::L0Manager*>(mgr);
  auto device = mgr_->drivers()[0]->devices()[0];
  auto q = device->command_queue();
  auto q_list = device->create_command_list();
  func->group_size() = {1, 1, 1};
  ze_group_count_t dispatchTraits;
  dispatchTraits.groupCountX = 1u;
  dispatchTraits.groupCountY = 1u;
  dispatchTraits.groupCountZ = 1u;
  for (unsigned i = 0; i < params.size(); ++i) {
    L0_SAFE_CALL(zeKernelSetArgumentValue(func->handle(), i, sizeof(void*), params[i]));
  }
  L0_SAFE_CALL(zeCommandListAppendLaunchKernel(
      q_list->handle(), func->handle(), &dispatchTraits, nullptr, 0, nullptr));
  q_list->submit(*q.get());
}
#endif

compiler::CodegenTraits get_traits() {
#ifdef HAVE_L0
  return compiler::CodegenTraits::get(4, 1);
#else
  return compiler::CodegenTraits::get(0, 0);
#endif
}
}  // namespace

TEST(CodeGeneratorTest, IntegerConstant) {
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get();
  auto llvm_module = llvm::CloneModule(*executor->get_rt_module(/*is_l0=*/false));
  ScalarCodeGenerator code_generator(executor->getConfig(), std::move(llvm_module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;
  auto constant = hdk::ir::makeExpr<hdk::ir::Constant>(kINT, false, d);
  const auto compiled_expr =
      code_generator.compile(constant.get(), true, co, get_traits());

  compiler::verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  using FuncPtr = int (*)(int*);
  auto func_ptr = reinterpret_cast<FuncPtr>(
      code_generator.generateNativeCode(executor, compiled_expr, co).front());
  CHECK(func_ptr);
  int out;
  int err = func_ptr(&out);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(out, d.intval);
}

TEST(CodeGeneratorTest, IntegerAdd) {
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get();
  auto llvm_module = llvm::CloneModule(*executor->get_rt_module(/*is_l0=*/false));
  ScalarCodeGenerator code_generator(executor->getConfig(), std::move(llvm_module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;
  auto lhs = hdk::ir::makeExpr<hdk::ir::Constant>(kINT, false, d);
  auto rhs = hdk::ir::makeExpr<hdk::ir::Constant>(kINT, false, d);
  auto plus = hdk::ir::makeExpr<hdk::ir::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co, get_traits());

  compiler::verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  using FuncPtr = int (*)(int*);
  auto func_ptr = reinterpret_cast<FuncPtr>(
      code_generator.generateNativeCode(executor, compiled_expr, co).front());
  CHECK(func_ptr);
  int out;
  int err = func_ptr(&out);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(out, d.intval + d.intval);
}

TEST(CodeGeneratorTest, IntegerColumn) {
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get();
  auto llvm_module = llvm::CloneModule(*executor->get_rt_module(/*is_l0=*/false));
  ScalarCodeGenerator code_generator(executor->getConfig(), std::move(llvm_module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;

  auto col = hdk::ir::makeExpr<hdk::ir::ColumnVar>(ti, table_id, column_id, rte_idx);
  const auto compiled_expr = code_generator.compile(col.get(), true, co, get_traits());

  compiler::verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *col);

  using FuncPtr = int (*)(int*, int);
  auto func_ptr = reinterpret_cast<FuncPtr>(
      code_generator.generateNativeCode(executor, compiled_expr, co).front());
  CHECK(func_ptr);
  int out;
  int err = func_ptr(&out, 17);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(out, 17);
}

TEST(CodeGeneratorTest, IntegerExpr) {
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get();
  auto llvm_module = llvm::CloneModule(*executor->get_rt_module(/*is_l0=*/false));
  ScalarCodeGenerator code_generator(executor->getConfig(), std::move(llvm_module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;
  auto lhs = hdk::ir::makeExpr<hdk::ir::ColumnVar>(ti, table_id, column_id, rte_idx);
  Datum d;
  d.intval = 42;

  auto rhs = hdk::ir::makeExpr<hdk::ir::Constant>(kINT, false, d);
  auto plus = hdk::ir::makeExpr<hdk::ir::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co, get_traits());

  compiler::verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *lhs);

  using FuncPtr = int (*)(int*, int);
  auto func_ptr = reinterpret_cast<FuncPtr>(
      code_generator.generateNativeCode(executor, compiled_expr, co).front());
  CHECK(func_ptr);
  int out;
  int err = func_ptr(&out, 58);
  ASSERT_EQ(err, 0);
  ASSERT_EQ(out, 100);
}

#ifdef HAVE_L0
#define IS_L0 true
#else
#define IS_L0 false
#endif

#if defined(HAVE_CUDA) || defined(HAVE_L0)
void free_param_pointers(const std::vector<void*>& param_ptrs, GpuMgr* gpu_mgr) {
  for (const auto param_ptr : param_ptrs) {
    const auto device_ptr =
        reinterpret_cast<int8_t*>(*reinterpret_cast<DevicePtr*>(param_ptr));
    gpu_mgr->freeDeviceMem(device_ptr);
  }
}

TEST(CodeGeneratorTest, IntegerConstantGPU) {
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get();
  auto llvm_module = llvm::CloneModule(*executor->get_rt_module(/*is_l0=*/IS_L0));
  ScalarCodeGenerator code_generator(executor->getConfig(), std::move(llvm_module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::GPU);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;

  auto constant = hdk::ir::makeExpr<hdk::ir::Constant>(kINT, false, d);
  const auto compiled_expr =
      code_generator.compile(constant.get(), true, co, get_traits());

  compiler::verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  const auto native_function_pointers =
      code_generator.generateNativeCode(executor, compiled_expr, co);

  for (size_t gpu_idx = 0; gpu_idx < native_function_pointers.size(); ++gpu_idx) {
    const auto native_function_pointer = native_function_pointers[gpu_idx];
    auto func_ptr = reinterpret_cast<DeviceFuncPtr>(native_function_pointer);

    std::vector<void*> param_ptrs;
    DevicePtr err = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    DevicePtr out = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    param_ptrs.push_back(&err);
    param_ptrs.push_back(&out);
    run_test_kernel(func_ptr, param_ptrs, code_generator.getGpuMgr());
    int32_t host_err;
    int32_t host_out;
    code_generator.getGpuMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_err),
                                                 reinterpret_cast<const int8_t*>(err),
                                                 4,
                                                 gpu_idx);
    code_generator.getGpuMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_out),
                                                 reinterpret_cast<const int8_t*>(out),
                                                 4,
                                                 gpu_idx);

    ASSERT_EQ(host_err, 0);
    ASSERT_EQ(host_out, d.intval);
    free_param_pointers(param_ptrs, code_generator.getGpuMgr());
  }
}

TEST(CodeGeneratorTest, IntegerAddGPU) {
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get();
  auto llvm_module = llvm::CloneModule(*executor->get_rt_module(/*is_l0=*/IS_L0));
  ScalarCodeGenerator code_generator(executor->getConfig(), std::move(llvm_module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::GPU);
  co.hoist_literals = false;

  Datum d;
  d.intval = 42;

  auto lhs = hdk::ir::makeExpr<hdk::ir::Constant>(kINT, false, d);
  auto rhs = hdk::ir::makeExpr<hdk::ir::Constant>(kINT, false, d);
  auto plus = hdk::ir::makeExpr<hdk::ir::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co, get_traits());

  compiler::verify_function_ir(compiled_expr.func);
  ASSERT_TRUE(compiled_expr.inputs.empty());

  const auto native_function_pointers =
      code_generator.generateNativeCode(executor, compiled_expr, co);

  for (size_t gpu_idx = 0; gpu_idx < native_function_pointers.size(); ++gpu_idx) {
    const auto native_function_pointer = native_function_pointers[gpu_idx];
    auto func_ptr = reinterpret_cast<DeviceFuncPtr>(native_function_pointer);

    std::vector<void*> param_ptrs;
    DevicePtr err = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    DevicePtr out = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    param_ptrs.push_back(&err);
    param_ptrs.push_back(&out);
    run_test_kernel(func_ptr, param_ptrs, code_generator.getGpuMgr());
    int32_t host_err;
    int32_t host_out;
    code_generator.getGpuMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_err),
                                                 reinterpret_cast<const int8_t*>(err),
                                                 4,
                                                 gpu_idx);
    code_generator.getGpuMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_out),
                                                 reinterpret_cast<const int8_t*>(out),
                                                 4,
                                                 gpu_idx);

    ASSERT_EQ(host_err, 0);
    ASSERT_EQ(host_out, d.intval + d.intval);
    free_param_pointers(param_ptrs, code_generator.getGpuMgr());
  }
}

TEST(CodeGeneratorTest, IntegerColumnGPU) {
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get();
  auto llvm_module = llvm::CloneModule(*executor->get_rt_module(/*is_l0=*/IS_L0));
  ScalarCodeGenerator code_generator(executor->getConfig(), std::move(llvm_module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::GPU);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;

  auto col = hdk::ir::makeExpr<hdk::ir::ColumnVar>(ti, table_id, column_id, rte_idx);
  const auto compiled_expr = code_generator.compile(col.get(), true, co, get_traits());

  compiler::verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *col);

  const auto native_function_pointers =
      code_generator.generateNativeCode(executor, compiled_expr, co);

  for (size_t gpu_idx = 0; gpu_idx < native_function_pointers.size(); ++gpu_idx) {
    const auto native_function_pointer = native_function_pointers[gpu_idx];
    auto func_ptr = reinterpret_cast<DeviceFuncPtr>(native_function_pointer);

    std::vector<void*> param_ptrs;
    DevicePtr err = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    DevicePtr out = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    DevicePtr in = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    int host_in = 17;
    code_generator.getGpuMgr()->copyHostToDevice(
        reinterpret_cast<int8_t*>(in),
        reinterpret_cast<const int8_t*>(&host_in),
        4,
        gpu_idx);
    param_ptrs.push_back(&err);
    param_ptrs.push_back(&out);
    param_ptrs.push_back(&in);
    run_test_kernel(func_ptr, param_ptrs, code_generator.getGpuMgr());
    int32_t host_err;
    int32_t host_out;
    code_generator.getGpuMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_err),
                                                 reinterpret_cast<const int8_t*>(err),
                                                 4,
                                                 gpu_idx);
    code_generator.getGpuMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_out),
                                                 reinterpret_cast<const int8_t*>(out),
                                                 4,
                                                 gpu_idx);

    ASSERT_EQ(host_err, 0);
    ASSERT_EQ(host_out, 17);
    free_param_pointers(param_ptrs, code_generator.getGpuMgr());
  }
}

TEST(CodeGeneratorTest, IntegerExprGPU) {
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, nullptr, nullptr).get();
  auto llvm_module = llvm::CloneModule(*executor->get_rt_module(/*is_l0=*/IS_L0));
  ScalarCodeGenerator code_generator(executor->getConfig(), std::move(llvm_module));
  CompilationOptions co = CompilationOptions::defaults(ExecutorDeviceType::GPU);
  co.hoist_literals = false;

  SQLTypeInfo ti(kINT, false);
  int table_id = 1;
  int column_id = 5;
  int rte_idx = 0;
  auto lhs = hdk::ir::makeExpr<hdk::ir::ColumnVar>(ti, table_id, column_id, rte_idx);
  Datum d;
  d.intval = 42;

  auto rhs = hdk::ir::makeExpr<hdk::ir::Constant>(kINT, false, d);
  auto plus = hdk::ir::makeExpr<hdk::ir::BinOper>(kINT, kPLUS, kONE, lhs, rhs);
  const auto compiled_expr = code_generator.compile(plus.get(), true, co, get_traits());

  compiler::verify_function_ir(compiled_expr.func);
  ASSERT_EQ(compiled_expr.inputs.size(), size_t(1));
  ASSERT_TRUE(*compiled_expr.inputs.front() == *lhs);

  const auto native_function_pointers =
      code_generator.generateNativeCode(executor, compiled_expr, co);

  for (size_t gpu_idx = 0; gpu_idx < native_function_pointers.size(); ++gpu_idx) {
    const auto native_function_pointer = native_function_pointers[gpu_idx];
    auto func_ptr = reinterpret_cast<DeviceFuncPtr>(native_function_pointer);

    std::vector<void*> param_ptrs;
    DevicePtr err = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    DevicePtr out = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    DevicePtr in = reinterpret_cast<DevicePtr>(
        code_generator.getGpuMgr()->allocateDeviceMem(4, gpu_idx));
    int host_in = 58;
    code_generator.getGpuMgr()->copyHostToDevice(
        reinterpret_cast<int8_t*>(in),
        reinterpret_cast<const int8_t*>(&host_in),
        4,
        gpu_idx);
    param_ptrs.push_back(&err);
    param_ptrs.push_back(&out);
    param_ptrs.push_back(&in);
    run_test_kernel(func_ptr, param_ptrs, code_generator.getGpuMgr());
    int32_t host_err;
    int32_t host_out;
    code_generator.getGpuMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_err),
                                                 reinterpret_cast<const int8_t*>(err),
                                                 4,
                                                 gpu_idx);
    code_generator.getGpuMgr()->copyDeviceToHost(reinterpret_cast<int8_t*>(&host_out),
                                                 reinterpret_cast<const int8_t*>(out),
                                                 4,
                                                 gpu_idx);

    ASSERT_EQ(host_err, 0);
    ASSERT_EQ(host_out, 100);
    free_param_pointers(param_ptrs, code_generator.getGpuMgr());
  }
}
#endif  // HAVE_CUDA || HAVE_L0

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();
  return err;
}
