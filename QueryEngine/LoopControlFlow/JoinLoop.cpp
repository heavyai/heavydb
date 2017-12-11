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

#include "JoinLoop.h"

#include <llvm/IR/Type.h>

#include <stack>

JoinLoop::JoinLoop(const JoinLoopKind kind,
                   const JoinType type,
                   const std::function<JoinLoopDomain(const std::vector<llvm::Value*>&)>& iteration_domain_codegen,
                   const std::function<llvm::Value*(const std::vector<llvm::Value*>&)>& outer_condition_match,
                   const std::string& name)
    : kind_(kind),
      type_(type),
      iteration_domain_codegen_(iteration_domain_codegen),
      outer_condition_match_(outer_condition_match),
      name_(name) {
  CHECK(outer_condition_match == nullptr || type == JoinType::LEFT);
}

llvm::BasicBlock* JoinLoop::codegen(
    const std::vector<JoinLoop>& join_loops,
    const std::function<llvm::BasicBlock*(const std::vector<llvm::Value*>&)>& body_codegen,
    llvm::Value* outer_iter,
    llvm::BasicBlock* exit_bb,
    llvm::IRBuilder<>& builder) {
  llvm::BasicBlock* prev_exit_bb{exit_bb};
  llvm::BasicBlock* prev_iter_advance_bb{nullptr};
  llvm::BasicBlock* last_head_bb{nullptr};
  auto& context = builder.getContext();
  const auto parent_func = builder.GetInsertBlock()->getParent();
  llvm::Value* prev_comparison_result{nullptr};
  llvm::BasicBlock* entry{nullptr};
  std::vector<llvm::Value*> iterators;
  iterators.push_back(outer_iter);
  for (const auto& join_loop : join_loops) {
    switch (join_loop.kind_) {
      case JoinLoopKind::UpperBound:
      case JoinLoopKind::Set: {
        const auto preheader_bb =
            llvm::BasicBlock::Create(context, "ub_iter_preheader_" + join_loop.name_, parent_func);
        if (!entry) {
          entry = preheader_bb;
        }
        if (prev_comparison_result) {
          builder.CreateCondBr(prev_comparison_result, preheader_bb, prev_exit_bb);
        }
        prev_exit_bb = prev_iter_advance_bb ? prev_iter_advance_bb : exit_bb;
        builder.SetInsertPoint(preheader_bb);
        const auto iteration_counter_ptr =
            builder.CreateAlloca(get_int_type(64, context), nullptr, "ub_iter_counter_ptr_" + join_loop.name_);
        builder.CreateStore(ll_int(int64_t(0), context), iteration_counter_ptr);
        const auto iteration_domain = join_loop.iteration_domain_codegen_(iterators);
        const auto head_bb = llvm::BasicBlock::Create(context, "ub_iter_head_" + join_loop.name_, parent_func);
        builder.CreateBr(head_bb);
        builder.SetInsertPoint(head_bb);
        llvm::Value* iteration_counter =
            builder.CreateLoad(iteration_counter_ptr, "ub_iter_counter_val_" + join_loop.name_);
        auto iteration_val = iteration_counter;
        CHECK(join_loop.kind_ == JoinLoopKind::Set || !iteration_domain.values_buffer);
        if (join_loop.kind_ == JoinLoopKind::Set) {
          iteration_val = builder.CreateGEP(iteration_domain.values_buffer, iteration_counter);
        }
        iterators.push_back(iteration_val);
        prev_comparison_result =
            builder.CreateICmpSLT(iteration_counter,
                                  join_loop.kind_ == JoinLoopKind::UpperBound ? iteration_domain.upper_bound
                                                                              : iteration_domain.element_count);
        const auto iter_advance_bb =
            llvm::BasicBlock::Create(context, "ub_iter_advance_" + join_loop.name_, parent_func);
        builder.SetInsertPoint(iter_advance_bb);
        builder.CreateStore(builder.CreateAdd(iteration_counter, ll_int(int64_t(1), context)), iteration_counter_ptr);
        builder.CreateBr(head_bb);
        builder.SetInsertPoint(head_bb);
        prev_iter_advance_bb = iter_advance_bb;
        last_head_bb = head_bb;
        break;
      }
      case JoinLoopKind::Singleton: {
        const auto true_bb = llvm::BasicBlock::Create(context, "singleton_true_" + join_loop.name_, parent_func);
        if (!entry) {
          entry = true_bb;
        }
        if (prev_comparison_result) {
          builder.CreateCondBr(prev_comparison_result, true_bb, prev_exit_bb);
        }
        prev_exit_bb = prev_iter_advance_bb ? prev_iter_advance_bb : exit_bb;
        builder.SetInsertPoint(true_bb);
        const auto iteration_domain = join_loop.iteration_domain_codegen_(iterators);
        CHECK(!iteration_domain.values_buffer);
        iterators.push_back(iteration_domain.slot_lookup_result);
        switch (join_loop.type_) {
          case JoinType::INNER: {
            prev_comparison_result =
                builder.CreateICmpSGE(iteration_domain.slot_lookup_result, ll_int<int64_t>(0, context));
            break;
          }
          case JoinType::LEFT: {
            // For outer joins, do the iteration regardless of the result of the match.
            prev_comparison_result = llvm::ConstantInt::get(get_int_type(1, context), true);
            break;
          }
          default:
            CHECK(false);
        }
        if (!prev_iter_advance_bb) {
          prev_iter_advance_bb = prev_exit_bb;
        }
        last_head_bb = true_bb;
        break;
      }
      default:
        CHECK(false);
    }
  }
  const auto body_bb = body_codegen(iterators);
  builder.CreateBr(prev_iter_advance_bb);
  builder.SetInsertPoint(last_head_bb);
  builder.CreateCondBr(prev_comparison_result, body_bb, prev_exit_bb);
  return entry;
}
