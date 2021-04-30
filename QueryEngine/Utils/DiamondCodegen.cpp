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
