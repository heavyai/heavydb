/*
    Copyright 2021 OmniSci, Inc.
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "HelperFunctions.h"

#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>

#include "QueryEngine/Compiler/Exceptions.h"
#include "QueryEngine/Optimization/AnnotateInternalFunctionsPass.h"

namespace compiler {
void throw_parseIR_error(const llvm::SMDiagnostic& parse_error,
                         std::string src,
                         const bool is_gpu) {
  std::string excname = (is_gpu ? "NVVM IR ParseError: " : "LLVM IR ParseError: ");
  llvm::raw_string_ostream ss(excname);
  parse_error.print(src.c_str(), ss, false, false);
  throw ParseIRError(ss.str());
}

llvm::StringRef get_gpu_target_triple_string() {
  return llvm::StringRef("nvptx64-nvidia-cuda");
}

llvm::StringRef get_gpu_data_layout() {
  return llvm::StringRef(
      "e-p:64:64:64-i1:8:8-i8:8:8-"
      "i16:16:16-i32:32:32-i64:64:64-"
      "f32:32:32-f64:64:64-v16:16:16-"
      "v32:32:32-v64:64:64-v128:128:128-n16:32:64");
}

void verify_function_ir(const llvm::Function* func) {
  std::stringstream err_ss;
  llvm::raw_os_ostream err_os(err_ss);
  err_os << "\n-----\n";
  if (llvm::verifyFunction(*func, &err_os)) {
    err_os << "\n-----\n";
    func->print(err_os, nullptr);
    err_os << "\n-----\n";
    LOG(FATAL) << err_ss.str();
  }
}

#if defined(HAVE_CUDA) || !defined(WITH_JIT_DEBUG)
void eliminate_dead_self_recursive_funcs(
    llvm::Module& M,
    const std::unordered_set<llvm::Function*>& live_funcs) {
  std::vector<llvm::Function*> dead_funcs;
  for (auto& F : M) {
    bool bAlive = false;
    if (live_funcs.count(&F)) {
      continue;
    }
    for (auto U : F.users()) {
      auto* C = llvm::dyn_cast<const llvm::CallInst>(U);
      if (!C || C->getParent()->getParent() != &F) {
        bAlive = true;
        break;
      }
    }
    if (!bAlive) {
      dead_funcs.push_back(&F);
    }
  }
  for (auto pFn : dead_funcs) {
    pFn->eraseFromParent();
  }
}
#endif

void optimize_ir(llvm::Function* query_func,
                 llvm::Module* llvm_module,
                 llvm::legacy::PassManager& pass_manager,
                 const std::unordered_set<llvm::Function*>& live_funcs,
                 const bool is_gpu_smem_used,
                 const CompilationOptions& co) {
  auto timer = DEBUG_TIMER(__func__);
  // the always inliner legacy pass must always run first
  pass_manager.add(llvm::createVerifierPass());
  pass_manager.add(llvm::createAlwaysInlinerLegacyPass());

  pass_manager.add(new AnnotateInternalFunctionsPass());

  pass_manager.add(llvm::createSROAPass());
  // mem ssa drops unused load and store instructions, e.g. passing variables directly
  // where possible
  pass_manager.add(
      llvm::createEarlyCSEPass(/*enable_mem_ssa=*/true));  // Catch trivial redundancies

  if (!is_gpu_smem_used) {
    // thread jumps can change the execution order around SMEM sections guarded by
    // `__syncthreads()`, which results in race conditions. For now, disable jump
    // threading for shared memory queries. In the future, consider handling shared memory
    // aggregations with a separate kernel launch
    pass_manager.add(llvm::createJumpThreadingPass());  // Thread jumps.
  }
  pass_manager.add(llvm::createCFGSimplificationPass());

  // remove load/stores in PHIs if instructions can be accessed directly post thread jumps
  pass_manager.add(llvm::createNewGVNPass());

  pass_manager.add(llvm::createDeadStoreEliminationPass());
  pass_manager.add(llvm::createLICMPass());

  pass_manager.add(llvm::createInstructionCombiningPass());

  // module passes
  pass_manager.add(llvm::createPromoteMemoryToRegisterPass());
  pass_manager.add(llvm::createGlobalOptimizerPass());

  pass_manager.add(llvm::createCFGSimplificationPass());  // cleanup after everything

  pass_manager.run(*llvm_module);

  eliminate_dead_self_recursive_funcs(*llvm_module, live_funcs);
}
}  // namespace compiler
