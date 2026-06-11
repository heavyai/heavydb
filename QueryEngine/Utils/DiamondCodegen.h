/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
