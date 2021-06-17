/*
 * Copyright 2021 OmniSci, Inc.
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

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Value.h>

#include <string>

class Executor;

/**
 * Helper struct for generating a branching instruction in LLVM IR. The diamond refers to
 * both branches, the starting basic block, and the ending block. The true and false basic
 * blocks are created on initialization, and the branches are created at destruction.
 */
struct DiamondCodegen {
  DiamondCodegen(llvm::Value* cond,
                 Executor* executor,
                 const bool chain_to_next,
                 const std::string& label_prefix,
                 DiamondCodegen* parent,
                 const bool share_false_edge_with_parent);
  void setChainToNext();
  void setFalseTarget(llvm::BasicBlock* cond_false);
  ~DiamondCodegen();

  Executor* executor_;
  llvm::BasicBlock* cond_true_;
  llvm::BasicBlock* cond_false_;
  llvm::BasicBlock* orig_cond_false_;
  bool chain_to_next_;
  DiamondCodegen* parent_;
};
