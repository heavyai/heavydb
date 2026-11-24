/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "JoinLoop.h"
#include "../CgenState.h"
#include "Logger/Logger.h"

#include <llvm/IR/Type.h>

#include <stack>

JoinLoop::JoinLoop(const JoinLoopKind kind,
                   const JoinType type,
                   const std::function<JoinLoopDomain(const std::vector<llvm::Value*>&)>&
                       iteration_domain_codegen,
                   const std::function<llvm::Value*(const std::vector<llvm::Value*>&)>&
                       outer_condition_match,
                   const std::function<void(llvm::Value*)>& found_outer_matches,
                   const HoistedFiltersCallback& hoisted_filters,
                   const std::function<llvm::Value*(const std::vector<llvm::Value*>&,
                                                    llvm::Value*)>& is_deleted,
                   const bool nested_loop_join,
                   const std::string& name)
    : kind_(kind)
    , type_(type)
    , iteration_domain_codegen_(iteration_domain_codegen)
    , outer_condition_match_(outer_condition_match)
    , found_outer_matches_(found_outer_matches)
    , hoisted_filters_(hoisted_filters)
    , is_deleted_(is_deleted)
    , nested_loop_join_(nested_loop_join)
    , name_(name) {
  CHECK(outer_condition_match == nullptr || type == JoinType::LEFT ||
        type == JoinType::INNER);
  CHECK_EQ(static_cast<bool>(found_outer_matches), (type == JoinType::LEFT));
}

llvm::BasicBlock* JoinLoop::codegen(
    const std::vector<JoinLoop>& join_loops,
    const std::function<llvm::BasicBlock*(const std::vector<llvm::Value*>&)>&
        body_codegen,
    llvm::Value* outer_iter,
    llvm::BasicBlock* exit_bb,
    CgenState* cgen_state) {
  AUTOMATIC_IR_METADATA(cgen_state);
  llvm::IRBuilder<>& builder = cgen_state->ir_builder_;
  llvm::BasicBlock* prev_exit_bb{exit_bb};
  llvm::BasicBlock* prev_iter_advance_bb{nullptr};
  llvm::BasicBlock* last_head_bb{nullptr};
  auto& context = builder.getContext();
  const auto parent_func = builder.GetInsertBlock()->getParent();
  llvm::Value* prev_comparison_result{nullptr};
  llvm::BasicBlock* entry{nullptr};
  std::vector<llvm::Value*> iterators;
  iterators.push_back(outer_iter);
  JoinType prev_join_type{JoinType::INVALID};
  // For inner nested-loop joins, a false join predicate advances the current
  // inner loop. Exhausting the inner loop is wired directly to prev_exit_bb.
  bool prev_condition_false_advances_iter{false};
  for (const auto& join_loop : join_loops) {
    bool current_condition_false_advances_iter{false};
    switch (join_loop.kind_) {
      case JoinLoopKind::UpperBound:
      case JoinLoopKind::Set:
      case JoinLoopKind::MultiSet: {
        const auto preheader_bb = llvm::BasicBlock::Create(
            context, "ub_iter_preheader_" + join_loop.name_, parent_func);

        llvm::BasicBlock* filter_bb{nullptr};
        if (join_loop.hoisted_filters_) {
          filter_bb = join_loop.hoisted_filters_(
              preheader_bb, prev_exit_bb, join_loop.name_, parent_func, cgen_state);
        }

        if (!entry) {
          entry = filter_bb ? filter_bb : preheader_bb;
        }

        if (prev_comparison_result) {
          builder.CreateCondBr(
              prev_comparison_result,
              filter_bb ? filter_bb : preheader_bb,
              (prev_join_type == JoinType::LEFT || prev_condition_false_advances_iter)
                  ? prev_iter_advance_bb
                  : prev_exit_bb);
        }
        prev_exit_bb = prev_iter_advance_bb ? prev_iter_advance_bb : exit_bb;
        builder.SetInsertPoint(preheader_bb);

        const auto iteration_counter_ptr = builder.CreateAlloca(
            get_int_type(64, context), nullptr, "ub_iter_counter_ptr_" + join_loop.name_);
        llvm::Value* found_an_outer_match_ptr{nullptr};
        llvm::Value* current_condition_match_ptr{nullptr};
        if (join_loop.type_ == JoinType::LEFT) {
          found_an_outer_match_ptr = builder.CreateAlloca(
              get_int_type(1, context), nullptr, "found_an_outer_match");
          builder.CreateStore(ll_bool(false, context), found_an_outer_match_ptr);
          current_condition_match_ptr = builder.CreateAlloca(
              get_int_type(1, context), nullptr, "outer_condition_current_match");
        }
        builder.CreateStore(ll_int(int64_t(0), context), iteration_counter_ptr);
        const auto iteration_domain = join_loop.iteration_domain_codegen_(iterators);

        const auto head_bb = llvm::BasicBlock::Create(
            context, "ub_iter_head_" + join_loop.name_, parent_func);

        if (iteration_domain.error_code) {
          cgen_state->needs_error_check_ = true;
          auto ub_iter_success_code = ll_int(int32_t(0), context);
          const auto ub_iter_error_condition =
              builder.CreateICmpEQ(iteration_domain.error_code, ub_iter_success_code);
          auto error_bb =
              llvm::BasicBlock::Create(context, "ub_iter_error_exit", parent_func);
          builder.CreateCondBr(ub_iter_error_condition, head_bb, error_bb);

          builder.SetInsertPoint(error_bb);
          builder.CreateRet(iteration_domain.error_code);
        } else {
          builder.CreateBr(head_bb);
        }

        builder.SetInsertPoint(head_bb);
        llvm::Value* iteration_counter =
            builder.CreateLoad(iteration_counter_ptr->getType()->getPointerElementType(),
                               iteration_counter_ptr,
                               "ub_iter_counter_val_" + join_loop.name_);
        auto iteration_val = iteration_counter;
        CHECK(join_loop.kind_ == JoinLoopKind::Set ||
              join_loop.kind_ == JoinLoopKind::MultiSet ||
              !iteration_domain.values_buffer);
        if (join_loop.kind_ == JoinLoopKind::Set ||
            join_loop.kind_ == JoinLoopKind::MultiSet) {
          CHECK(iteration_domain.values_buffer->getType()->isPointerTy());
          const auto ptr_type =
              static_cast<llvm::PointerType*>(iteration_domain.values_buffer->getType());
          if (ptr_type->getPointerElementType()->isArrayTy()) {
            iteration_val = builder.CreateGEP(
                iteration_domain.values_buffer->getType()
                    ->getScalarType()
                    ->getPointerElementType(),
                iteration_domain.values_buffer,
                std::vector<llvm::Value*>{
                    llvm::ConstantInt::get(get_int_type(64, context), 0),
                    iteration_counter},
                "ub_iter_counter_" + join_loop.name_);
          } else {
            iteration_val = builder.CreateGEP(iteration_domain.values_buffer->getType()
                                                  ->getScalarType()
                                                  ->getPointerElementType(),
                                              iteration_domain.values_buffer,
                                              iteration_counter,
                                              "ub_iter_counter_" + join_loop.name_);
          }
        }
        iterators.push_back(iteration_val);
        const auto have_more_inner_rows = builder.CreateICmpSLT(
            iteration_counter,
            join_loop.kind_ == JoinLoopKind::UpperBound ? iteration_domain.upper_bound
                                                        : iteration_domain.element_count,
            "have_more_inner_rows");
        const auto iter_advance_bb = llvm::BasicBlock::Create(
            context, "ub_iter_advance_" + join_loop.name_, parent_func);
        llvm::BasicBlock* row_not_deleted_bb{nullptr};
        if (join_loop.is_deleted_) {
          row_not_deleted_bb = llvm::BasicBlock::Create(
              context, "row_not_deleted_" + join_loop.name_, parent_func);
          const auto row_is_deleted =
              join_loop.is_deleted_(iterators, have_more_inner_rows);
          builder.CreateCondBr(row_is_deleted, iter_advance_bb, row_not_deleted_bb);
          builder.SetInsertPoint(row_not_deleted_bb);
        }
        if (join_loop.type_ == JoinType::LEFT) {
          std::tie(last_head_bb, prev_comparison_result) =
              evaluateOuterJoinCondition(join_loop,
                                         iteration_domain,
                                         iterators,
                                         iteration_counter,
                                         have_more_inner_rows,
                                         found_an_outer_match_ptr,
                                         current_condition_match_ptr,
                                         cgen_state);
        } else if (join_loop.outer_condition_match_) {
          std::tie(last_head_bb, prev_comparison_result) = evaluateInnerJoinCondition(
              join_loop, iterators, have_more_inner_rows, prev_exit_bb, cgen_state);
          // A predicate miss should keep scanning the current join domain.
          current_condition_false_advances_iter = true;
        } else {
          prev_comparison_result = have_more_inner_rows;
          last_head_bb = row_not_deleted_bb ? row_not_deleted_bb : head_bb;
        }
        builder.SetInsertPoint(iter_advance_bb);
        const auto iteration_counter_next_val =
            builder.CreateAdd(iteration_counter, ll_int(int64_t(1), context));
        builder.CreateStore(iteration_counter_next_val, iteration_counter_ptr);
        if (join_loop.type_ == JoinType::LEFT) {
          const auto no_more_inner_rows =
              builder.CreateICmpSGT(iteration_counter_next_val,
                                    join_loop.kind_ == JoinLoopKind::UpperBound
                                        ? iteration_domain.upper_bound
                                        : iteration_domain.element_count,
                                    "no_more_inner_rows");
          builder.CreateCondBr(no_more_inner_rows, prev_exit_bb, head_bb);
        } else {
          builder.CreateBr(head_bb);
        }
        builder.SetInsertPoint(last_head_bb);
        prev_iter_advance_bb = iter_advance_bb;
        break;
      }
      case JoinLoopKind::Singleton: {
        const auto true_bb = llvm::BasicBlock::Create(
            context, "singleton_true_" + join_loop.name_, parent_func);

        llvm::BasicBlock* filter_bb{nullptr};
        if (join_loop.hoisted_filters_) {
          filter_bb = join_loop.hoisted_filters_(
              true_bb, prev_exit_bb, join_loop.name_, parent_func, cgen_state);
        }

        if (!entry) {
          entry = filter_bb ? filter_bb : true_bb;
        }

        if (prev_comparison_result) {
          builder.CreateCondBr(
              prev_comparison_result,
              filter_bb ? filter_bb : true_bb,
              (prev_join_type == JoinType::LEFT || prev_condition_false_advances_iter)
                  ? prev_iter_advance_bb
                  : prev_exit_bb);
        }
        prev_exit_bb = prev_iter_advance_bb ? prev_iter_advance_bb : exit_bb;

        builder.SetInsertPoint(true_bb);
        const auto iteration_domain = join_loop.iteration_domain_codegen_(iterators);
        CHECK(!iteration_domain.values_buffer);
        iterators.push_back(iteration_domain.slot_lookup_result);
        auto join_cond_match = builder.CreateICmpSGE(iteration_domain.slot_lookup_result,
                                                     ll_int<int64_t>(0, context));
        llvm::Value* remaining_cond_match = builder.CreateAlloca(
            get_int_type(1, context), nullptr, "remaining_outer_cond_match");
        builder.CreateStore(ll_bool(true, context), remaining_cond_match);

        if (join_loop.type_ == JoinType::LEFT && join_loop.outer_condition_match_) {
          const auto parent_func = builder.GetInsertBlock()->getParent();
          const auto evaluate_remaining_outer_cond_bb = llvm::BasicBlock::Create(
              context, "eval_remaining_outer_cond_" + join_loop.name_, parent_func);
          const auto after_evaluate_outer_cond_bb = llvm::BasicBlock::Create(
              context, "after_eval_outer_cond_" + join_loop.name_, parent_func);
          builder.CreateCondBr(join_cond_match,
                               evaluate_remaining_outer_cond_bb,
                               after_evaluate_outer_cond_bb);
          builder.SetInsertPoint(evaluate_remaining_outer_cond_bb);
          const auto outer_cond_match = join_loop.outer_condition_match_(iterators);
          const auto true_left_cond_match =
              builder.CreateAnd(outer_cond_match, join_cond_match);
          builder.CreateStore(true_left_cond_match, remaining_cond_match);
          builder.CreateBr(after_evaluate_outer_cond_bb);
          builder.SetInsertPoint(after_evaluate_outer_cond_bb);
        }
        auto match_found = builder.CreateAnd(
            join_cond_match,
            builder.CreateLoad(remaining_cond_match->getType()->getPointerElementType(),
                               remaining_cond_match));
        CHECK(match_found);
        if (join_loop.is_deleted_) {
          match_found = builder.CreateAnd(
              match_found, builder.CreateNot(join_loop.is_deleted_(iterators, nullptr)));
        }
        auto match_found_bb = builder.GetInsertBlock();
        switch (join_loop.type_) {
          case JoinType::INNER:
          case JoinType::SEMI: {
            prev_comparison_result = match_found;
            break;
          }
          case JoinType::ANTI: {
            auto match_found_for_anti_join = builder.CreateICmpSLT(
                iteration_domain.slot_lookup_result, ll_int<int64_t>(0, context));
            prev_comparison_result = match_found_for_anti_join;
            break;
          }
          case JoinType::LEFT: {
            join_loop.found_outer_matches_(match_found);
            // For outer joins, do the iteration regardless of the result of the match.
            prev_comparison_result = ll_bool(true, context);
            break;
          }
          default:
            CHECK(false);
        }
        if (!prev_iter_advance_bb) {
          prev_iter_advance_bb = prev_exit_bb;
        }
        last_head_bb = match_found_bb;
        break;
      }
      default:
        CHECK(false);
    }
    prev_join_type = join_loop.type_;
    prev_condition_false_advances_iter = current_condition_false_advances_iter;
  }

  const auto body_bb = body_codegen(iterators);
  builder.CreateBr(prev_iter_advance_bb);
  builder.SetInsertPoint(last_head_bb);
  builder.CreateCondBr(
      prev_comparison_result,
      body_bb,
      (prev_join_type == JoinType::LEFT || prev_condition_false_advances_iter)
          ? prev_iter_advance_bb
          : prev_exit_bb);
  return entry;
}

std::pair<llvm::BasicBlock*, llvm::Value*> JoinLoop::evaluateOuterJoinCondition(
    const JoinLoop& join_loop,
    const JoinLoopDomain& iteration_domain,
    const std::vector<llvm::Value*>& iterators,
    llvm::Value* iteration_counter,
    llvm::Value* have_more_inner_rows,
    llvm::Value* found_an_outer_match_ptr,
    llvm::Value* current_condition_match_ptr,
    CgenState* cgen_state) {
  AUTOMATIC_IR_METADATA(cgen_state);
  llvm::IRBuilder<>& builder = cgen_state->ir_builder_;
  auto& context = builder.getContext();
  const auto parent_func = builder.GetInsertBlock()->getParent();
  builder.CreateStore(ll_bool(false, context), current_condition_match_ptr);
  const auto evaluate_outer_condition_bb = llvm::BasicBlock::Create(
      context, "eval_outer_cond_" + join_loop.name_, parent_func);
  const auto after_evaluate_outer_condition_bb = llvm::BasicBlock::Create(
      context, "after_eval_outer_cond_" + join_loop.name_, parent_func);
  builder.CreateCondBr(have_more_inner_rows,
                       evaluate_outer_condition_bb,
                       after_evaluate_outer_condition_bb);
  builder.SetInsertPoint(evaluate_outer_condition_bb);
  const auto current_condition_match = join_loop.outer_condition_match_
                                           ? join_loop.outer_condition_match_(iterators)
                                           : ll_bool(true, context);
  builder.CreateStore(current_condition_match, current_condition_match_ptr);
  const auto updated_condition_match = builder.CreateOr(
      current_condition_match,
      builder.CreateLoad(found_an_outer_match_ptr->getType()->getPointerElementType(),
                         found_an_outer_match_ptr));
  builder.CreateStore(updated_condition_match, found_an_outer_match_ptr);
  builder.CreateBr(after_evaluate_outer_condition_bb);
  builder.SetInsertPoint(after_evaluate_outer_condition_bb);
  const auto no_matches_found = builder.CreateNot(
      builder.CreateLoad(found_an_outer_match_ptr->getType()->getPointerElementType(),
                         found_an_outer_match_ptr));
  const auto no_more_inner_rows = builder.CreateICmpEQ(
      iteration_counter,
      join_loop.kind_ == JoinLoopKind::UpperBound ? iteration_domain.upper_bound
                                                  : iteration_domain.element_count);
  // Do the iteration if the outer condition is true or it's the last iteration and no
  // matches have been found.
  const auto do_iteration = builder.CreateOr(
      builder.CreateLoad(current_condition_match_ptr->getType()->getPointerElementType(),
                         current_condition_match_ptr),
      builder.CreateAnd(no_matches_found, no_more_inner_rows));
  join_loop.found_outer_matches_(
      builder.CreateLoad(current_condition_match_ptr->getType()->getPointerElementType(),
                         current_condition_match_ptr));
  return {after_evaluate_outer_condition_bb, do_iteration};
}

std::pair<llvm::BasicBlock*, llvm::Value*> JoinLoop::evaluateInnerJoinCondition(
    const JoinLoop& join_loop,
    const std::vector<llvm::Value*>& iterators,
    llvm::Value* have_more_inner_rows,
    llvm::BasicBlock* no_more_inner_rows_bb,
    CgenState* cgen_state) {
  AUTOMATIC_IR_METADATA(cgen_state);
  CHECK(join_loop.outer_condition_match_);
  llvm::IRBuilder<>& builder = cgen_state->ir_builder_;
  auto& context = builder.getContext();
  const auto parent_func = builder.GetInsertBlock()->getParent();
  const auto evaluate_condition_bb = llvm::BasicBlock::Create(
      context, "eval_inner_loop_join_cond_" + join_loop.name_, parent_func);

  // No more rows is not a predicate miss; it exits to the enclosing loop or row.
  builder.CreateCondBr(
      have_more_inner_rows, evaluate_condition_bb, no_more_inner_rows_bb);
  builder.SetInsertPoint(evaluate_condition_bb);
  const auto current_condition_match = join_loop.outer_condition_match_(iterators);
  return {builder.GetInsertBlock(), current_condition_match};
}
