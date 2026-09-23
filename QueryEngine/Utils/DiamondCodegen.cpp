/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/Utils/DiamondCodegen.h"

#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"

DiamondCodegen::DiamondCodegen(llvm::Value* cond,
                               Executor* executor,
                               const bool chain_to_next,
                               const std::string& label_prefix,
                               DiamondCodegen* parent,
                               const bool share_false_edge_with_parent)
    : executor_(executor), chain_to_next_(chain_to_next), parent_(parent) {
  auto* cgen_state = executor_->cgen_state_.get();
  CHECK(cgen_state);
  AUTOMATIC_IR_METADATA(cgen_state);
  if (parent_) {
    CHECK(!chain_to_next_);
  }
  cond_true_ = llvm::BasicBlock::Create(
      cgen_state->context_, label_prefix + "_true", cgen_state->current_func_);
  if (share_false_edge_with_parent) {
    CHECK(parent);
    orig_cond_false_ = cond_false_ = parent_->cond_false_;
  } else {
    orig_cond_false_ = cond_false_ = llvm::BasicBlock::Create(
        cgen_state->context_, label_prefix + "_false", cgen_state->current_func_);
  }

  cgen_state->ir_builder_.CreateCondBr(cond, cond_true_, cond_false_);
  cgen_state->ir_builder_.SetInsertPoint(cond_true_);
}

void DiamondCodegen::setChainToNext() {
  CHECK(!parent_);
  chain_to_next_ = true;
}

void DiamondCodegen::setFalseTarget(llvm::BasicBlock* cond_false) {
  CHECK(!parent_ || orig_cond_false_ != parent_->cond_false_);
  cond_false_ = cond_false;
}

DiamondCodegen::~DiamondCodegen() {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto& builder = executor_->cgen_state_->ir_builder_;
  if (parent_ && orig_cond_false_ != parent_->cond_false_) {
    builder.CreateBr(parent_->cond_false_);
  } else if (chain_to_next_) {
    builder.CreateBr(cond_false_);
  }
  if (!parent_ || (!chain_to_next_ && cond_false_ != parent_->cond_false_)) {
    builder.SetInsertPoint(orig_cond_false_);
  }
}
