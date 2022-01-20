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

#pragma once

#include "CudaMgr/CudaMgr.h"
#include "Logger/Logger.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/GpuSharedMemoryUtils.h"
#include "QueryEngine/LLVMFunctionAttributesUtil.h"
#include "QueryEngine/NvidiaKernel.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "ResultSetTestUtils.h"
#include "Shared/TargetInfo.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_os_ostream.h>
#include <iostream>
#include <memory>

class StrideNumberGenerator : public NumberGenerator {
 public:
  StrideNumberGenerator(const int64_t start, const int64_t stride)
      : crt_(start), stride_(stride), start_(start) {}

  int64_t getNextValue() override {
    const auto crt = crt_;
    crt_ += stride_;
    return crt;
  }

  void reset() override { crt_ = start_; }

 private:
  int64_t crt_;
  int64_t stride_;
  int64_t start_;
};

class GpuReductionTester : public GpuSharedMemCodeBuilder {
 public:
  GpuReductionTester(llvm::Module* module,
                     llvm::LLVMContext& context,
                     const QueryMemoryDescriptor& qmd,
                     const std::vector<TargetInfo>& targets,
                     const std::vector<int64_t>& init_agg_values,
                     CudaMgr_Namespace::CudaMgr* cuda_mgr)
      : GpuSharedMemCodeBuilder(module,
                                context,
                                qmd,
                                targets,
                                init_agg_values,
                                Executor::UNITARY_EXECUTOR_ID)
      , cuda_mgr_(cuda_mgr) {
    // CHECK(getReductionFunction());
  }
  void codegenWrapperKernel();
  llvm::Function* getWrapperKernel() const { return wrapper_kernel_; }
  void performReductionTest(const std::vector<std::unique_ptr<ResultSet>>& result_sets,
                            const ResultSetStorage* gpu_result_storage,
                            const size_t device_id);

 private:
  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
  llvm::Function* wrapper_kernel_;
};
