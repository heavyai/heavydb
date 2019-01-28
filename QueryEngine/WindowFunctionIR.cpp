/*
 * Copyright 2018 OmniSci, Inc.
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

#include "Execute.h"
#include "WindowContext.h"

llvm::Value* Executor::codegenWindowFunction(const Analyzer::WindowFunction* window_func,
                                             const size_t target_index,
                                             const CompilationOptions& co) {
  const auto window_func_context =
      WindowProjectNodeContext::get()->activateWindowFunctionContext(target_index);
  switch (window_func->getKind()) {
    case SqlWindowFunctionKind::ROW_NUMBER:
    case SqlWindowFunctionKind::RANK:
    case SqlWindowFunctionKind::DENSE_RANK:
    case SqlWindowFunctionKind::NTILE: {
      return cgen_state_->emitCall(
          "row_number_window_func",
          {ll_int(reinterpret_cast<const int64_t>(window_func_context->output())),
           posArg(nullptr)});
    }
    case SqlWindowFunctionKind::PERCENT_RANK:
    case SqlWindowFunctionKind::CUME_DIST: {
      return cgen_state_->emitCall(
          "percent_window_func",
          {ll_int(reinterpret_cast<const int64_t>(window_func_context->output())),
           posArg(nullptr)});
    }
    case SqlWindowFunctionKind::LAG:
    case SqlWindowFunctionKind::LEAD:
    case SqlWindowFunctionKind::FIRST_VALUE:
    case SqlWindowFunctionKind::LAST_VALUE: {
      CHECK(WindowProjectNodeContext::get());
      const auto& args = window_func->getArgs();
      CHECK(!args.empty());
      const auto lag_lvs = codegen(args.front().get(), true, co);
      CHECK_EQ(lag_lvs.size(), size_t(1));
      return lag_lvs.front();
    }
    case SqlWindowFunctionKind::MIN:
    case SqlWindowFunctionKind::MAX: {
      return codegenWindowAggregate(window_func, window_func_context, co);
    }
    default: { LOG(FATAL) << "Invalid window function kind"; }
  }
  return nullptr;
}

llvm::Value* Executor::codegenWindowAggregate(
    const Analyzer::WindowFunction* window_func,
    const WindowFunctionContext* window_func_context,
    const CompilationOptions& co) {
  const auto bitset =
      ll_int(reinterpret_cast<const int64_t>(window_func_context->partitionStart()));
  const auto min_val = ll_int(int64_t(0));
  const auto max_val = ll_int(window_func_context->elementCount() - 1);
  const auto null_val = ll_int(inline_int_null_value<int64_t>());
  const auto null_bool_val = ll_int<int8_t>(inline_int_null_value<int8_t>());
  const auto pi64_type =
      llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
  const auto aggregate_state_i64 =
      ll_int(reinterpret_cast<const int64_t>(window_func_context->aggregateState()));
  const auto aggregate_state =
      cgen_state_->ir_builder_.CreateIntToPtr(aggregate_state_i64, pi64_type);
  const auto window_func_null_val =
      castToTypeIn(inlineIntNull(window_func->get_type_info()), 64);
  const auto reset_state = toBool(cgen_state_->emitCall(
      "bit_is_set",
      {bitset, posArg(nullptr), min_val, max_val, null_val, null_bool_val}));
  const auto reset_state_true_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "reset_state.true", cgen_state_->row_func_);
  const auto reset_state_false_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "reset_state.false", cgen_state_->row_func_);
  cgen_state_->ir_builder_.CreateCondBr(
      reset_state, reset_state_true_bb, reset_state_false_bb);
  cgen_state_->ir_builder_.SetInsertPoint(reset_state_true_bb);
  cgen_state_->emitCall("agg_id", {aggregate_state, window_func_null_val});
  cgen_state_->ir_builder_.CreateBr(reset_state_false_bb);
  cgen_state_->ir_builder_.SetInsertPoint(reset_state_false_bb);
  CHECK(WindowProjectNodeContext::get());
  const auto& args = window_func->getArgs();
  CHECK(!args.empty());
  const auto lag_lvs = codegen(args.front().get(), true, co);
  CHECK_EQ(lag_lvs.size(), size_t(1));
  const auto crt_val = castToTypeIn(lag_lvs.front(), 64);
  switch (window_func->getKind()) {
    case SqlWindowFunctionKind::MIN: {
      cgen_state_->emitCall("agg_min_skip_val",
                            {aggregate_state, crt_val, window_func_null_val});
      break;
    }
    case SqlWindowFunctionKind::MAX: {
      cgen_state_->emitCall("agg_max_skip_val",
                            {aggregate_state, crt_val, window_func_null_val});
      break;
    }
    default: { LOG(FATAL) << "Invalid window function kind"; }
  }
  return cgen_state_->ir_builder_.CreateLoad(aggregate_state);
}
