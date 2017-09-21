/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include <glog/logging.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#include "../IRCodegenUtils.h"

#include <functional>
#include <vector>

enum class JoinLoopKind {
  UpperBound,  // loop join
  Set,         // one to many hash join
  Singleton    // one to one hash join
};

struct JoinLoopDomain {
  union {
    llvm::Value* upper_bound;         // for UpperBound
    llvm::Value* element_count;       // for Set
    llvm::Value* slot_lookup_result;  // for Singleton
  };
  llvm::Value* values_buffer;  // used for Set
};

// Any join is logically a loop. Hash joins just limit the domain of iteration,
// which can be as little as one element for one to one hash join, in which case
// we'll not generate IR for an actual loop.
class JoinLoop {
 public:
  JoinLoop(const JoinLoopKind,
           const std::function<JoinLoopDomain(const std::vector<llvm::Value*>&)>&,
           const std::string& name = "");

  static llvm::BasicBlock* codegen(
      const std::vector<JoinLoop>& join_loops,
      const std::function<llvm::BasicBlock*(const std::vector<llvm::Value*>&)>& body_codegen,
      llvm::Value* outer_iter,
      llvm::BasicBlock* exit_bb,
      llvm::IRBuilder<>& builder);

 private:
  const JoinLoopKind kind_;
  const std::function<JoinLoopDomain(const std::vector<llvm::Value*>&)> iteration_domain_codegen_;
  const std::string name_;
};
