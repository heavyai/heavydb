/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Descriptors/QueryMemoryDescriptor.h"
#include "IRCodegenUtils.h"
#include "Logger/Logger.h"
#include "ResultSet.h"
#include "Shared/TargetInfo.h"

/**
 * This is a builder class for extra functions that are required to
 * support GPU shared memory usage for GroupByPerfectHash query types.
 *
 * This class does not own its own LLVM module and uses a pointer to the
 * global module provided to it as an argument during construction
 */
class GpuSharedMemCodeBuilder {
 public:
  GpuSharedMemCodeBuilder(llvm::Module* module,
                          llvm::LLVMContext& context,
                          const QueryMemoryDescriptor& qmd,
                          const std::vector<TargetInfo>& targets,
                          const std::vector<int64_t>& init_agg_values,
                          const size_t executor_id);
  /**
   * generates code for both the reduction and initialization steps required for shared
   * memory usage
   */
  void codegen();

  /**
   * Once the reduction and init functions are generated, this function takes the main
   * query function and replaces the previous placeholders, which were inserted in the
   * query template, with these new functions.
   */
  void injectFunctionsInto(llvm::Function* query_func);

  llvm::Function* getReductionFunction() const { return reduction_func_; }
  llvm::Function* getInitFunction() const { return init_func_; }
  std::string toString() const;

 protected:
  /**
   * Generates code for the reduction functionality (from shared memory into global
   * memory)
   */
  void codegenReduction();
  /**
   * Generates code for the shared memory buffer initialization
   */
  void codegenInitialization();
  /**
   * Create the reduction function in the LLVM module, with predefined arguments and
   * return type
   */
  llvm::Function* createReductionFunction() const;
  /**
   * Creates the initialization function in the LLVM module, with predefined arguments and
   * return type
   */
  llvm::Function* createInitFunction() const;
  /**
   * Search for a particular funciton name in the module, and returns it if found
   */
  llvm::Function* getFunction(const std::string& func_name) const;

  size_t executor_id_;
  llvm::Module* module_;
  llvm::LLVMContext& context_;
  llvm::Function* reduction_func_;
  llvm::Function* init_func_;
  const QueryMemoryDescriptor query_mem_desc_;
  const std::vector<TargetInfo> targets_;
  const std::vector<int64_t> init_agg_values_;
};
