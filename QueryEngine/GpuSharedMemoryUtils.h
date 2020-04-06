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
#include "ResultSet.h"
#include "Shared/Logger.h"
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
                          const std::vector<int64_t>& init_agg_values);
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

  llvm::Module* module_;
  llvm::LLVMContext& context_;
  llvm::Function* reduction_func_;
  llvm::Function* init_func_;
  const QueryMemoryDescriptor query_mem_desc_;
  const std::vector<TargetInfo> targets_;
  const std::vector<int64_t> init_agg_values_;
};
