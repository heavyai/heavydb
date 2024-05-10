/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "CodeGenerator.h"
#include "CodegenHelper.h"
#include "Execute.h"
#include "WindowContext.h"

llvm::Value* Executor::codegenWindowFunction(const size_t target_index,
                                             const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  CodeGenerator code_generator(this);

  const auto window_func_context =
      WindowProjectNodeContext::get(this)->activateWindowFunctionContext(this,
                                                                         target_index);
  const auto window_func = window_func_context->getWindowFunction();
  switch (window_func->getKind()) {
    case SqlWindowFunctionKind::ROW_NUMBER:
    case SqlWindowFunctionKind::RANK:
    case SqlWindowFunctionKind::DENSE_RANK:
    case SqlWindowFunctionKind::NTILE:
      return code_generator.codegenWindowPosition(window_func_context,
                                                  code_generator.posArg(nullptr));
    case SqlWindowFunctionKind::PERCENT_RANK:
    case SqlWindowFunctionKind::CUME_DIST:
      return cgen_state_->emitCall("percent_window_func",
                                   {cgen_state_->llInt(reinterpret_cast<const int64_t>(
                                        window_func_context->output())),
                                    code_generator.posArg(nullptr)});
    case SqlWindowFunctionKind::LAG:
    case SqlWindowFunctionKind::LEAD:
    case SqlWindowFunctionKind::FIRST_VALUE:
    case SqlWindowFunctionKind::LAST_VALUE:
    case SqlWindowFunctionKind::NTH_VALUE: {
      // they are always evaluated on the current frame
      CHECK(WindowProjectNodeContext::get(this));
      const auto& args = window_func->getArgs();
      CHECK(!args.empty());
      const auto arg_lvs = code_generator.codegen(args.front().get(), true, co);
      CHECK_EQ(arg_lvs.size(), size_t(1));
      return arg_lvs.front();
    }
    case SqlWindowFunctionKind::AVG:
    case SqlWindowFunctionKind::MIN:
    case SqlWindowFunctionKind::MAX:
    case SqlWindowFunctionKind::SUM:
    case SqlWindowFunctionKind::SUM_IF:
    case SqlWindowFunctionKind::COUNT:
    case SqlWindowFunctionKind::COUNT_IF:
    case SqlWindowFunctionKind::CONDITIONAL_CHANGE_EVENT:
      return codegenWindowFunctionAggregate(&code_generator, co);
    case SqlWindowFunctionKind::LEAD_IN_FRAME:
    case SqlWindowFunctionKind::LAG_IN_FRAME:
    case SqlWindowFunctionKind::NTH_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::FIRST_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::LAST_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::FORWARD_FILL:
    case SqlWindowFunctionKind::BACKWARD_FILL:
      return codegenWindowNavigationFunctionOnFrame(co);
    default:
      LOG(FATAL) << "Invalid window function kind";
  }
  return nullptr;
}

namespace {

std::string get_window_agg_name(const SqlWindowFunctionKind kind,
                                const SQLTypeInfo& window_func_ti) {
  std::string agg_name;
  switch (kind) {
    case SqlWindowFunctionKind::MIN: {
      agg_name = "agg_min";
      break;
    }
    case SqlWindowFunctionKind::MAX: {
      agg_name = "agg_max";
      break;
    }
    case SqlWindowFunctionKind::AVG:
    case SqlWindowFunctionKind::SUM:
    case SqlWindowFunctionKind::CONDITIONAL_CHANGE_EVENT: {
      agg_name = "agg_sum";
      break;
    }
    case SqlWindowFunctionKind::COUNT: {
      agg_name = "agg_count";
      break;
    }
    case SqlWindowFunctionKind::COUNT_IF: {
      agg_name = "agg_count_if";
      break;
    }
    case SqlWindowFunctionKind::SUM_IF: {
      agg_name = "agg_sum_if";
      break;
    }
    default: {
      LOG(FATAL) << "Invalid window function kind";
    }
  }
  switch (window_func_ti.get_type()) {
    case kFLOAT: {
      agg_name += "_float";
      break;
    }
    case kDOUBLE: {
      agg_name += "_double";
      break;
    }
    default: {
      break;
    }
  }
  return agg_name;
}

SQLTypeInfo get_adjusted_window_type_info(const Analyzer::WindowFunction* window_func) {
  const auto& args = window_func->getArgs();
  return ((window_func->getKind() == SqlWindowFunctionKind::COUNT && !args.empty()) ||
          window_func->getKind() == SqlWindowFunctionKind::AVG)
             ? args.front()->get_type_info()
             : window_func->get_type_info();
}

std::string get_col_type_name_by_size(const size_t size, const bool is_fp) {
  if (is_fp) {
    if (size == 4) {
      return "float";
    }
    return "double";
  }
  if (size == 1) {
    return "int8_t";
  } else if (size == 2) {
    return "int16_t";
  } else if (size == 4) {
    return "int32_t";
  }
  return "int64_t";
}

llvm::Value* get_null_value_by_size(CgenState* cgen_state, SQLTypeInfo col_ti) {
  if (col_ti.is_fp()) {
    if (col_ti.get_type() == kFLOAT) {
      return cgen_state->llFp(inline_fp_null_value<float>());
    } else {
      return cgen_state->llFp(inline_fp_null_value<double>());
    }
  } else if (col_ti.is_dict_encoded_string()) {
    if (col_ti.get_size() == 2) {
      return cgen_state->llInt((int16_t)inline_int_null_value<int16_t>());
    } else {
      CHECK_EQ(col_ti.get_size(), 4);
      return cgen_state->llInt((int32_t)inline_int_null_value<int32_t>());
    }
  } else if (col_ti.is_date()) {
    if (col_ti.get_comp_param() == 16) {
      return cgen_state->llInt((int16_t)inline_int_null_value<int16_t>());
    } else if (col_ti.get_comp_param() == 32) {
      return cgen_state->llInt((int32_t)inline_int_null_value<int32_t>());
    } else {
      return cgen_state->llInt((int64_t)inline_int_null_value<int64_t>());
    }
  } else {
    switch (col_ti.get_type()) {
      case kBOOLEAN:
      case kTINYINT:
        return cgen_state->llInt((int8_t)inline_int_null_value<int8_t>());
      case kSMALLINT:
        if (col_ti.get_compression() == kENCODING_FIXED) {
          return cgen_state->llInt((int16_t)(inline_fixed_encoding_null_val(col_ti)));
        }
        return cgen_state->llInt((int16_t)inline_int_null_value<int16_t>());
      case kINT:
        if (col_ti.get_compression() == kENCODING_FIXED) {
          return cgen_state->llInt((int32_t)(inline_fixed_encoding_null_val(col_ti)));
        }
        return cgen_state->llInt((int32_t)inline_int_null_value<int32_t>());
      case kTIME:
      case kTIMESTAMP:
        if (col_ti.get_compression() == kENCODING_FIXED) {
          return cgen_state->llInt((int64_t)(inline_fixed_encoding_null_val(col_ti)));
        }
      case kBIGINT:
        if (col_ti.get_compression() == kENCODING_FIXED) {
          return cgen_state->llInt((int64_t)(inline_fixed_encoding_null_val(col_ti)));
        }
        return cgen_state->llInt((int64_t)inline_int_null_value<int64_t>());
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
      case kDECIMAL:
      case kNUMERIC:
        return cgen_state->llInt((int64_t)inline_int_null_value<int64_t>());
      default:
        abort();
    }
    return cgen_state->llInt(inline_int_null_val(col_ti));
  }
}

llvm::Value* get_null_value_by_size_with_encoding(CgenState* cgen_state,
                                                  SQLTypeInfo col_ti) {
  if (col_ti.is_fp()) {
    if (col_ti.get_type() == kFLOAT) {
      return cgen_state->llFp(inline_fp_null_value<float>());
    } else {
      return cgen_state->llFp(inline_fp_null_value<double>());
    }
  } else {
    llvm::Value* ret_val{nullptr};
    if (col_ti.get_compression() == kENCODING_FIXED ||
        col_ti.get_compression() == kENCODING_DATE_IN_DAYS) {
      ret_val = cgen_state->llInt(inline_fixed_encoding_null_val(col_ti));
    } else {
      ret_val = cgen_state->llInt(inline_int_null_val(col_ti));
    }
    size_t ret_val_col_size_in_bytes = col_ti.get_logical_size() * 8;
    if (ret_val->getType()->getIntegerBitWidth() > ret_val_col_size_in_bytes) {
      return cgen_state->castToTypeIn(ret_val, ret_val_col_size_in_bytes);
    }
    return ret_val;
  }
}

}  // namespace

llvm::Value* Executor::aggregateWindowStatePtr(CodeGenerator* code_generator,
                                               const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto arg_ti = get_adjusted_window_type_info(window_func);
  llvm::Type* aggregate_state_type =
      arg_ti.get_type() == kFLOAT
          ? llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0)
          : llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
  const auto aggregate_state_i64 = cgen_state_->llInt(
      reinterpret_cast<const int64_t>(window_func_context->aggregateState()));
  return CodegenUtil::createPtrWithHoistedMemoryAddr(
             cgen_state_.get(),
             code_generator,
             co,
             aggregate_state_i64,
             aggregate_state_type,
             WindowFunctionContext::NUM_EXECUTION_DEVICES)
      .front();
}

llvm::Value* Executor::codegenWindowFunctionAggregate(CodeGenerator* code_generator,
                                                      const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  auto [reset_state_false_bb, aggregate_state] =
      codegenWindowResetStateControlFlow(code_generator, co);
  llvm::Value* aggregate_state_count = nullptr;
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
    const auto aggregate_state_count_i64 = cgen_state_->llInt(
        reinterpret_cast<const int64_t>(window_func_context->aggregateStateCount()));
    const auto pi64_type =
        llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
    aggregate_state_count = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                cgen_state_.get(),
                                code_generator,
                                co,
                                aggregate_state_count_i64,
                                pi64_type,
                                WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                .front();
  }
  codegenWindowFunctionStateInit(code_generator, co, aggregate_state);
  if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
    const auto count_zero = cgen_state_->llInt(int64_t(0));
    cgen_state_->emitCall("agg_id", {aggregate_state_count, count_zero});
  }
  cgen_state_->ir_builder_.CreateBr(reset_state_false_bb);
  cgen_state_->ir_builder_.SetInsertPoint(reset_state_false_bb);
  CHECK(WindowProjectNodeContext::get(this));
  return codegenWindowFunctionAggregateCalls(aggregate_state, co);
}

std::pair<llvm::BasicBlock*, llvm::Value*> Executor::codegenWindowResetStateControlFlow(
    CodeGenerator* code_generator,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  auto aggregate_state = aggregateWindowStatePtr(code_generator, co);
  const auto bitset = cgen_state_->llInt(
      reinterpret_cast<const int64_t>(window_func_context->partitionStart()));
  const auto bitset_lv =
      CodegenUtil::createPtrWithHoistedMemoryAddr(
          cgen_state_.get(),
          code_generator,
          co,
          bitset,
          llvm::PointerType::get(get_int_type(8, cgen_state_->context_), 0),
          WindowFunctionContext::NUM_EXECUTION_DEVICES)
          .front();
  const auto min_val = cgen_state_->llInt(int64_t(0));
  const auto max_val = cgen_state_->llInt(window_func_context->elementCount() - 1);
  const auto null_val = cgen_state_->llInt(inline_int_null_value<int64_t>());
  const auto null_bool_val = cgen_state_->llInt<int8_t>(inline_int_null_value<int8_t>());
  const auto reset_state =
      code_generator->toBool(cgen_state_->emitCall("bit_is_set",
                                                   {bitset_lv,
                                                    code_generator->posArg(nullptr),
                                                    min_val,
                                                    max_val,
                                                    null_val,
                                                    null_bool_val}));
  const auto reset_state_true_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "reset_state.true", cgen_state_->current_func_);
  const auto reset_state_false_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "reset_state.false", cgen_state_->current_func_);
  cgen_state_->ir_builder_.CreateCondBr(
      reset_state, reset_state_true_bb, reset_state_false_bb);
  cgen_state_->ir_builder_.SetInsertPoint(reset_state_true_bb);
  return std::make_pair(reset_state_false_bb, aggregate_state);
}

void Executor::codegenWindowFunctionStateInit(CodeGenerator* code_generator,
                                              const CompilationOptions& co,
                                              llvm::Value* aggregate_state) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto window_func_ti = get_adjusted_window_type_info(window_func);
  const auto window_func_null_val =
      window_func_ti.is_fp()
          ? cgen_state_->inlineFpNull(window_func_ti)
          : cgen_state_->castToTypeIn(cgen_state_->inlineIntNull(window_func_ti), 64);
  llvm::Value* window_func_init_val;
  const auto window_func_kind = window_func_context->getWindowFunction()->getKind();
  if (window_func_kind == SqlWindowFunctionKind::COUNT ||
      window_func_kind == SqlWindowFunctionKind::COUNT_IF) {
    switch (window_func_ti.get_type()) {
      case kFLOAT: {
        window_func_init_val = cgen_state_->llFp(float(0));
        break;
      }
      case kDOUBLE: {
        window_func_init_val = cgen_state_->llFp(double(0));
        break;
      }
      default: {
        window_func_init_val = cgen_state_->llInt(int64_t(0));
        break;
      }
    }
  } else {
    window_func_init_val = window_func_null_val;
  }
  const auto pi32_type =
      llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0);
  switch (window_func_ti.get_type()) {
    case kDOUBLE: {
      cgen_state_->emitCall("agg_id_double", {aggregate_state, window_func_init_val});
      break;
    }
    case kFLOAT: {
      aggregate_state =
          cgen_state_->ir_builder_.CreateBitCast(aggregate_state, pi32_type);
      cgen_state_->emitCall("agg_id_float", {aggregate_state, window_func_init_val});
      break;
    }
    default: {
      cgen_state_->emitCall("agg_id", {aggregate_state, window_func_init_val});
      break;
    }
  }
}

llvm::Value* Executor::codegenWindowNavigationFunctionOnFrame(
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto window_func_kind = window_func->getKind();
  const auto& args = window_func->getArgs();
  CHECK(args.size() >= 1 && args.size() <= 3);
  CodeGenerator code_generator(this);

  const auto target_col_ti = args.front()->get_type_info();
  const auto target_col_size = target_col_ti.get_size();
  const auto target_col_type_name =
      get_col_type_name_by_size(target_col_size, target_col_ti.is_fp());
  const auto target_col_logical_type_name = get_col_type_name_by_size(
      window_func->get_type_info().get_size(), window_func->get_type_info().is_fp());

  // when target_column is fixed encoded, we store the actual column value by
  // considering it, but our resultset analyzer only considers the type without encoding
  // scheme so we handle them separately
  auto logical_null_val_lv =
      get_null_value_by_size(cgen_state_.get(), window_func->get_type_info());
  auto target_col_null_val_lv =
      get_null_value_by_size_with_encoding(cgen_state_.get(), target_col_ti);
  if (window_func_context->elementCount() == 0) {
    // we do not need to generate a code for an empty input table
    return target_col_null_val_lv;
  }

  auto current_row_pos_lv = code_generator.posArg(nullptr);
  auto partition_index_lv = codegenCurrentPartitionIndex(
      window_func_context, &code_generator, co, current_row_pos_lv);

  // load window function input expression; target_column
  size_t target_col_size_in_byte = target_col_size * 8;
  llvm::Type* col_buf_ptr_type =
      target_col_ti.is_fp()
          ? get_fp_type(target_col_size_in_byte, cgen_state_->context_)
          : get_int_type(target_col_size_in_byte, cgen_state_->context_);
  auto col_buf_type = llvm::PointerType::get(col_buf_ptr_type, 0);
  auto target_col_buf_ptr_lv = cgen_state_->llInt(reinterpret_cast<int64_t>(
      window_func_context->getColumnBufferForWindowFunctionExpressions().front()));
  auto target_col_buf_lv = CodegenUtil::createPtrWithHoistedMemoryAddr(
                               cgen_state_.get(),
                               &code_generator,
                               co,
                               target_col_buf_ptr_lv,
                               col_buf_type,
                               WindowFunctionContext::NUM_EXECUTION_DEVICES)
                               .front();

  // prepare various buffer ptrs related to the window partition
  auto partition_buf_ptrs = codegenLoadPartitionBuffers(
      window_func_context, &code_generator, co, partition_index_lv);

  // null value of the ordering column
  const auto order_key_buf_ti =
      window_func_context->getOrderKeyColumnBufferTypes().front();
  auto const ordering_spec = window_func->getCollation().front();
  auto order_key_col_null_val_lv =
      get_null_value_by_size_with_encoding(cgen_state_.get(), order_key_buf_ti);

  // load ordering column
  auto [order_col_type_name, order_key_buf_ptr_lv] =
      codegenLoadOrderKeyBufPtr(window_func_context, &code_generator, co);

  // null range
  auto [null_start_pos_lv, null_end_pos_lv] =
      codegenFrameNullRange(window_func_context, &code_generator, co, partition_index_lv);

  // compute a row index of the current row w.r.t the window frame it belongs to
  std::string row_idx_on_frame_func = "compute_";
  row_idx_on_frame_func += order_col_type_name;
  row_idx_on_frame_func += ordering_spec.is_desc ? "_greater_equal" : "_less_equal";
  row_idx_on_frame_func += "_current_row_idx_in_frame";
  auto int64_t_one_val_lv = cgen_state_->llInt((int64_t)1);
  auto nulls_first_lv = cgen_state_->llBool(ordering_spec.nulls_first);
  auto cur_row_idx_in_frame_lv =
      cgen_state_->emitCall(row_idx_on_frame_func,
                            {partition_buf_ptrs.num_elem_current_partition_lv,
                             current_row_pos_lv,
                             order_key_buf_ptr_lv,
                             partition_buf_ptrs.target_partition_rowid_ptr_lv,
                             partition_buf_ptrs.target_partition_sorted_rowid_ptr_lv,
                             order_key_col_null_val_lv,
                             nulls_first_lv,
                             null_start_pos_lv,
                             null_end_pos_lv});

  if (window_func->isMissingValueFillingFunction()) {
    // We classify both FORWARD_FILL and BACKWARD_FILL as window frame navigate function
    // b/c they need to determine the current row index within a sorted partition
    // (as we did for window frame navigation functions) to compute the correct and
    // consistent resultset Otherwise, the query result may differ per execution due to
    // missing table ordering Now we know the current row's index in the sorted
    // partition (cur_row_idx_in_frame_lv), so we can return by calling the runtime
    // function with the index we computed
    std::string func_name = "fill_" + target_col_type_name + "_missing_value";

    llvm::Value* forward_fill_lv =
        cgen_state_->llBool(window_func_kind == SqlWindowFunctionKind::FORWARD_FILL);
    return cgen_state_->emitCall(func_name,
                                 {cur_row_idx_in_frame_lv,
                                  target_col_null_val_lv,
                                  target_col_buf_lv,
                                  partition_buf_ptrs.num_elem_current_partition_lv,
                                  partition_buf_ptrs.target_partition_rowid_ptr_lv,
                                  partition_buf_ptrs.target_partition_sorted_rowid_ptr_lv,
                                  forward_fill_lv});
  }

  // compute frame bound for the current row
  auto [frame_start_bound_expr_lv, frame_end_bound_expr_lv] =
      codegenFrameBoundRange(window_func, code_generator, co);

  // compute frame bound for the current row
  auto const int64_t_zero_val_lv = cgen_state_->llInt((int64_t)0);
  WindowFrameBoundFuncArgs WindowFrameBoundFuncArgs{
      frame_start_bound_expr_lv,
      frame_end_bound_expr_lv,
      window_func->hasRangeModeFraming() ? current_row_pos_lv : cur_row_idx_in_frame_lv,
      nullptr,
      window_func->hasRangeModeFraming()
          ? int64_t_zero_val_lv
          : partition_buf_ptrs.current_partition_start_offset_lv,
      int64_t_zero_val_lv,
      int64_t_one_val_lv,
      partition_buf_ptrs.num_elem_current_partition_lv,
      order_key_buf_ptr_lv,
      "",
      partition_buf_ptrs.target_partition_rowid_ptr_lv,
      partition_buf_ptrs.target_partition_sorted_rowid_ptr_lv,
      nulls_first_lv,
      null_start_pos_lv,
      null_end_pos_lv};
  auto [frame_start_bound_lv, frame_end_bound_lv] =
      codegenWindowFrameBounds(window_func_context,
                               window_func->getFrameStartBound(),
                               window_func->getFrameEndBound(),
                               order_key_col_null_val_lv,
                               WindowFrameBoundFuncArgs,
                               code_generator);

  // compute the index of the current row in frame it belongs to
  llvm::Value* modified_cur_row_idx_in_frame_lv{nullptr};
  llvm::Value* offset_lv{nullptr};
  switch (window_func_kind) {
    case SqlWindowFunctionKind::LAG_IN_FRAME:
      offset_lv = cgen_state_->castToTypeIn(
          code_generator.codegen(args[1].get(), true, co)[0], 64);
      modified_cur_row_idx_in_frame_lv =
          cgen_state_->ir_builder_.CreateSub(cur_row_idx_in_frame_lv, offset_lv);
      break;
    case SqlWindowFunctionKind::LEAD_IN_FRAME:
      offset_lv = cgen_state_->castToTypeIn(
          code_generator.codegen(args[1].get(), true, co)[0], 64);
      modified_cur_row_idx_in_frame_lv =
          cgen_state_->ir_builder_.CreateAdd(cur_row_idx_in_frame_lv, offset_lv);
      break;
    case SqlWindowFunctionKind::FIRST_VALUE_IN_FRAME:
      modified_cur_row_idx_in_frame_lv = frame_start_bound_lv;
      break;
    case SqlWindowFunctionKind::LAST_VALUE_IN_FRAME:
      modified_cur_row_idx_in_frame_lv = frame_end_bound_lv;
      break;
    case SqlWindowFunctionKind::NTH_VALUE_IN_FRAME: {
      offset_lv = cgen_state_->castToTypeIn(
          code_generator.codegen(args[1].get(), true, co)[0], 64);
      auto candidate_offset_lv =
          cgen_state_->ir_builder_.CreateAdd(frame_start_bound_lv, offset_lv);
      auto out_of_frame_bound_lv =
          cgen_state_->ir_builder_.CreateICmpSGT(candidate_offset_lv, frame_end_bound_lv);
      modified_cur_row_idx_in_frame_lv = cgen_state_->ir_builder_.CreateSelect(
          out_of_frame_bound_lv, cgen_state_->llInt((int64_t)-1), candidate_offset_lv);
      break;
    }
    default:
      UNREACHABLE() << "Unsupported window function to navigate a window frame.";
  }
  CHECK(modified_cur_row_idx_in_frame_lv);

  // get the target column value in the frame w.r.t the offset
  std::string target_func_name = "get_";
  target_func_name += target_col_type_name + "_value_";
  target_func_name += target_col_logical_type_name + "_type_";
  target_func_name += "in_frame";
  auto res_lv =
      cgen_state_->emitCall(target_func_name,
                            {modified_cur_row_idx_in_frame_lv,
                             frame_start_bound_lv,
                             frame_end_bound_lv,
                             target_col_buf_lv,
                             partition_buf_ptrs.target_partition_rowid_ptr_lv,
                             partition_buf_ptrs.target_partition_sorted_rowid_ptr_lv,
                             logical_null_val_lv,
                             target_col_null_val_lv});
  if (target_col_ti.get_compression() == kENCODING_DATE_IN_DAYS) {
    res_lv = cgen_state_->emitCall(
        "encode_date",
        {res_lv, logical_null_val_lv, cgen_state_->llInt((int64_t)kSecsPerDay)});
  }
  CHECK(res_lv);
  return res_lv;
}

llvm::Value* Executor::codegenFrameBoundExpr(const Analyzer::WindowFunction* window_func,
                                             const Analyzer::WindowFrame* frame_bound,
                                             CodeGenerator& code_generator,
                                             const CompilationOptions& co) {
  auto needs_bound_expr_codegen = [](const Analyzer::WindowFrame* window_frame) {
    return window_frame->getBoundType() == SqlWindowFrameBoundType::EXPR_FOLLOWING ||
           window_frame->getBoundType() == SqlWindowFrameBoundType::EXPR_PRECEDING;
  };
  const auto order_col_ti = window_func->getOrderKeys().front()->get_type_info();
  auto encode_date_col_val = [&order_col_ti, this](llvm::Value* bound_expr_lv) {
    if (order_col_ti.get_comp_param() == 16) {
      return cgen_state_->emitCall(
          "fixed_width_date_encode_noinline",
          {bound_expr_lv,
           cgen_state_->castToTypeIn(cgen_state_->inlineIntNull(SQLTypeInfo(kSMALLINT)),
                                     32),
           cgen_state_->inlineIntNull(SQLTypeInfo(kBIGINT))});
    } else {
      return cgen_state_->emitCall("fixed_width_date_encode_noinline",
                                   {bound_expr_lv,
                                    cgen_state_->inlineIntNull(SQLTypeInfo(kINT)),
                                    cgen_state_->inlineIntNull(SQLTypeInfo(kBIGINT))});
    }
  };
  llvm::Value* bound_expr_lv{nullptr};
  if (needs_bound_expr_codegen(frame_bound)) {
    auto bound_expr = frame_bound->getBoundExpr();
    if (auto dateadd_expr = dynamic_cast<const Analyzer::DateaddExpr*>(bound_expr)) {
      if (dateadd_expr->get_datetime_expr()->get_type_info().is_encoded_timestamp()) {
        dateadd_expr->set_fixed_encoding_null_val();
      }
    }
    auto bound_expr_lvs = code_generator.codegen(bound_expr, true, co);
    bound_expr_lv = bound_expr_lvs.front();
    if (order_col_ti.is_date() && window_func->hasRangeModeFraming()) {
      if (g_cluster) {
        throw std::runtime_error(
            "Range mode with date type ordering column is not supported yet.");
      }
      bound_expr_lv = encode_date_col_val(bound_expr_lv);
    }
    if (frame_bound->getBoundExpr()->get_type_info().get_size() != 8) {
      bound_expr_lv = cgen_state_->castToTypeIn(bound_expr_lv, 64);
    }
  } else {
    bound_expr_lv = cgen_state_->llInt((int64_t)-1);
  }
  CHECK(bound_expr_lv);
  return bound_expr_lv;
}

llvm::Value* Executor::codegenFrameBound(bool for_start_bound,
                                         bool for_range_mode,
                                         bool for_window_frame_naviation,
                                         const Analyzer::WindowFrame* frame_bound,
                                         bool is_timestamp_type_frame,
                                         llvm::Value* order_key_null_val,
                                         const WindowFrameBoundFuncArgs& args) {
  const auto bound_type = frame_bound->getBoundType();
  auto adjust_frame_end_bound = [&](llvm::Value* target_bound_lv) {
    return cgen_state_->ir_builder_.CreateSub(target_bound_lv, args.int64_t_one_val_lv);
  };
  if (bound_type == SqlWindowFrameBoundType::UNBOUNDED_PRECEDING) {
    CHECK(for_start_bound) << "frame end cannot be UNBOUNDED PRECEDING";
    return args.int64_t_zero_val_lv;
  } else if (bound_type == SqlWindowFrameBoundType::UNBOUNDED_FOLLOWING) {
    CHECK(!for_start_bound) << "frame start cannot be UNBOUNDED FOLLOWING";
    // adjust frame bound w.r.t the open frame interval if necessary
    return for_window_frame_naviation
               ? adjust_frame_end_bound(args.num_elem_current_partition_lv)
               : args.num_elem_current_partition_lv;
  }
  std::vector<llvm::Value*> func_args;
  std::string op_name =
      bound_type == SqlWindowFrameBoundType::EXPR_FOLLOWING ? "add" : "sub";
  if (!for_range_mode) {
    llvm::Value* current_row_bound_expr_lv{nullptr};
    if (for_window_frame_naviation) {
      // we already know a current row's index in (ordered) window frame in this case
      auto bound_expr =
          for_start_bound ? args.frame_start_bound_expr_lv : args.frame_end_bound_expr_lv;
      if (bound_type == SqlWindowFrameBoundType::EXPR_FOLLOWING) {
        current_row_bound_expr_lv =
            cgen_state_->ir_builder_.CreateAdd(args.current_row_pos_lv, bound_expr);
      } else if (bound_type == SqlWindowFrameBoundType::EXPR_PRECEDING) {
        current_row_bound_expr_lv =
            cgen_state_->ir_builder_.CreateSub(args.current_row_pos_lv, bound_expr);
      } else {
        CHECK(bound_type == SqlWindowFrameBoundType::CURRENT_ROW);
        current_row_bound_expr_lv = args.current_row_pos_lv;
      }
      // adjust frame bound w.r.t the open frame interval
      if (for_start_bound) {
        return cgen_state_->ir_builder_.CreateSelect(
            cgen_state_->ir_builder_.CreateICmpSLT(current_row_bound_expr_lv,
                                                   args.int64_t_zero_val_lv),
            args.int64_t_zero_val_lv,
            current_row_bound_expr_lv);
      } else {
        return cgen_state_->ir_builder_.CreateSelect(
            cgen_state_->ir_builder_.CreateICmpSGE(current_row_bound_expr_lv,
                                                   args.num_elem_current_partition_lv),
            adjust_frame_end_bound(args.num_elem_current_partition_lv),
            current_row_bound_expr_lv);
      }
    } else {
      std::string func_class = for_start_bound ? "start" : "end";
      auto const func_name = "compute_row_mode_" + func_class + "_index_" + op_name;
      func_args = prepareRowModeFuncArgs(for_start_bound, bound_type, args);
      current_row_bound_expr_lv = cgen_state_->emitCall(func_name, func_args);
    }
    return current_row_bound_expr_lv;
  } else {
    std::string func_class = for_start_bound ? "lower" : "upper";
    auto const func_name = getFramingFuncName(
        func_class,
        args.order_type_col_name,
        op_name,
        bound_type != SqlWindowFrameBoundType::CURRENT_ROW && is_timestamp_type_frame);
    func_args = prepareRangeModeFuncArgs(
        for_start_bound, frame_bound, is_timestamp_type_frame, order_key_null_val, args);
    auto frame_bound_lv = cgen_state_->emitCall(func_name, func_args);
    if (!for_start_bound && for_window_frame_naviation) {
      // adjust frame end bound w.r.t the open frame interval
      frame_bound_lv = cgen_state_->ir_builder_.CreateSelect(
          cgen_state_->ir_builder_.CreateICmpSGE(frame_bound_lv,
                                                 args.num_elem_current_partition_lv),
          adjust_frame_end_bound(args.num_elem_current_partition_lv),
          frame_bound_lv);
    }
    return frame_bound_lv;
  }
}

const SQLTypeInfo Executor::getFirstOrderColTypeInfo(
    WindowFunctionContext* window_func_context) const {
  const auto window_func = window_func_context->getWindowFunction();
  return window_func->getOrderKeys().front()->get_type_info();
}

size_t Executor::getOrderKeySize(WindowFunctionContext* window_func_context) const {
  const auto order_key_size = getFirstOrderColTypeInfo(window_func_context).get_size();
  return order_key_size;
}

const std::string Executor::getOrderKeyTypeName(
    WindowFunctionContext* window_func_context) const {
  auto const order_key_size = getOrderKeySize(window_func_context);
  auto const order_key_ptr =
      window_func_context->getWindowFunction()->getOrderKeys().front();
  CHECK(order_key_ptr);
  return get_col_type_name_by_size(order_key_size,
                                   order_key_ptr->get_type_info().is_fp());
}

llvm::Value* Executor::codegenLoadCurrentValueFromColBuf(
    WindowFunctionContext* window_func_context,
    CodeGenerator& code_generator,
    WindowFrameBoundFuncArgs& args) const {
  llvm::Value* current_col_value_ptr_lv{nullptr};
  const auto order_key_size_in_byte = getOrderKeySize(window_func_context) * 8;
  auto const order_key_ptr =
      window_func_context->getWindowFunction()->getOrderKeys().front();
  CHECK(order_key_ptr);
  auto const order_col_ti = order_key_ptr->get_type_info();
  auto const order_col_llvm_type =
      order_col_ti.is_fp() ? get_fp_type(order_key_size_in_byte, cgen_state_->context_)
                           : get_int_type(order_key_size_in_byte, cgen_state_->context_);
  if (!window_func_context->getWindowFunction()->isFrameNavigateWindowFunction()) {
    auto rowid_in_partition_lv = code_generator.codegenWindowPosition(
        window_func_context, args.current_row_pos_lv);
    current_col_value_ptr_lv = cgen_state_->ir_builder_.CreateGEP(
        order_col_llvm_type, args.order_key_buf_ptr_lv, rowid_in_partition_lv);
  } else {
    current_col_value_ptr_lv = cgen_state_->ir_builder_.CreateGEP(
        order_col_llvm_type, args.order_key_buf_ptr_lv, args.current_row_pos_lv);
  }
  return cgen_state_->ir_builder_.CreateLoad(
      current_col_value_ptr_lv->getType()->getPointerElementType(),
      current_col_value_ptr_lv,
      "current_col_value");
}

llvm::Value* Executor::codegenCurrentPartitionIndex(
    const WindowFunctionContext* window_func_context,
    CodeGenerator* code_generator,
    const CompilationOptions& co,
    llvm::Value* current_row_pos_lv) {
  const auto pi64_type =
      llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
  const auto pi32_type =
      llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0);
  auto row_pos_lv = current_row_pos_lv;
  if (window_func_context->getWindowFunction()->isFrameNavigateWindowFunction()) {
    // `current_row_pos_lv` indicates the index of the current row, but to figure out
    // it's index of window partition it belongs to, we need a special approach
    // especially for window framing navigation function  for instance, when we have
    // five rows having two columns pc and val such as (2,1), (2,2), (2,3), (1,1),
    // (1,2), we build a OneToMany Perfect Hash Table as: offset: 0 2 / count: 2 3 /
    // payload: i1, i2, i3, i4, i5 where i1 ~ i3 and i4 ~ i5 are rows for partition 1
    // (i.e., pc = 1) and 2 (i.e., prc = 2), respectively. But when processing the first
    // row (2, 1), the original `current_row_pos_lv` stands for zero so computing which
    // partitions it belongs to is hard unless hashing the value at runtime. Even if we
    // do hash, we cannot know the exact hash slot unless we do binary + linear searches
    // multiple times (via payload buffer and the ordered payload buffer) i.e., when the
    // row (1,2) is assigned to the partition[4], we cannot find the hash slot index '4'
    // by using `current_row_pos_lv` unless doing a costly operation like a linear
    // search over the entire window partition Instead, we collect a hash slot that each
    // row is assigned to and keep this info at the payload buffer
    // `hash_slot_idx_ptr_lv` and use it for computing window frame navigation functions
    auto* const hash_slot_idx_ptr =
        window_func_context->payload() + window_func_context->elementCount();
    auto hash_slot_idx_buf_lv =
        cgen_state_->llInt(reinterpret_cast<int64_t>(hash_slot_idx_ptr));
    auto hash_slot_idx_ptr_lv = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                    cgen_state_.get(),
                                    code_generator,
                                    co,
                                    hash_slot_idx_buf_lv,
                                    pi32_type,
                                    WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                    .front();
    auto hash_slot_idx_load_lv = cgen_state_->ir_builder_.CreateGEP(
        hash_slot_idx_ptr_lv->getType()->getPointerElementType(),
        hash_slot_idx_ptr_lv,
        current_row_pos_lv);
    row_pos_lv = cgen_state_->castToTypeIn(
        cgen_state_->ir_builder_.CreateLoad(
            hash_slot_idx_load_lv->getType()->getPointerElementType(),
            hash_slot_idx_load_lv,
            "cur_row_hash_slot_idx"),
        64);
  }
  auto partition_count_lv = cgen_state_->llInt(window_func_context->partitionCount());
  auto partition_num_count_buf_lv = cgen_state_->llInt(
      reinterpret_cast<int64_t>(window_func_context->partitionNumCountBuf()));
  auto partition_num_count_ptr_lv = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                        cgen_state_.get(),
                                        code_generator,
                                        co,
                                        partition_num_count_buf_lv,
                                        pi64_type,
                                        WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                        .front();
  return cgen_state_->emitCall(
      "compute_int64_t_lower_bound",
      {partition_count_lv, row_pos_lv, partition_num_count_ptr_lv});
}

std::string Executor::getFramingFuncName(const std::string& bound_type,
                                         const std::string& order_col_type,
                                         const std::string& op_type,
                                         bool for_timestamp_type) const {
  auto target_val_type = for_timestamp_type ? "int64_t" : order_col_type;
  auto null_type = for_timestamp_type ? "int64_t" : order_col_type;
  return "range_mode_" + target_val_type + "_" + order_col_type + "_" + null_type + "_" +
         op_type + "_frame_" + bound_type + "_bound";
}

std::vector<llvm::Value*> Executor::prepareRowModeFuncArgs(
    bool for_start_bound,
    SqlWindowFrameBoundType bound_type,
    const WindowFrameBoundFuncArgs& args) const {
  std::vector<llvm::Value*> frame_args{args.current_row_pos_lv,
                                       args.current_partition_start_offset_lv};
  if (bound_type == SqlWindowFrameBoundType::CURRENT_ROW) {
    frame_args.push_back(args.int64_t_zero_val_lv);
  } else {
    frame_args.push_back(for_start_bound ? args.frame_start_bound_expr_lv
                                         : args.frame_end_bound_expr_lv);
    if (bound_type == SqlWindowFrameBoundType::EXPR_FOLLOWING) {
      frame_args.push_back(args.num_elem_current_partition_lv);
    }
  }
  return frame_args;
}

std::vector<llvm::Value*> Executor::prepareRangeModeFuncArgs(
    bool for_start_bound,
    const Analyzer::WindowFrame* frame_bound,
    bool is_timestamp_type_frame,
    llvm::Value* order_key_null_val,
    const WindowFrameBoundFuncArgs& args) const {
  llvm::Value* bound_expr_lv =
      for_start_bound ? args.frame_start_bound_expr_lv : args.frame_end_bound_expr_lv;
  llvm::Value* target_val_lv =
      frame_bound->isCurrentRowBound() || !is_timestamp_type_frame
          ? args.current_col_value_lv
          : bound_expr_lv;
  llvm::Value* frame_bound_val_lv =
      frame_bound->isCurrentRowBound() || is_timestamp_type_frame
          ? args.int64_t_zero_val_lv
          : bound_expr_lv;
  std::vector<llvm::Value*> frame_args{args.num_elem_current_partition_lv,
                                       target_val_lv,
                                       args.order_key_buf_ptr_lv,
                                       args.target_partition_rowid_ptr_lv,
                                       args.target_partition_sorted_rowid_ptr_lv,
                                       frame_bound_val_lv,
                                       order_key_null_val,
                                       args.nulls_first_lv,
                                       args.null_start_pos_lv,
                                       args.null_end_pos_lv};
  return frame_args;
}

std::pair<llvm::Value*, llvm::Value*> Executor::codegenFrameNullRange(
    WindowFunctionContext* window_func_context,
    CodeGenerator* code_generator,
    const CompilationOptions& co,
    llvm::Value* partition_index_lv) const {
  const auto pi64_type =
      llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
  const auto null_start_pos_buf = cgen_state_->llInt(
      reinterpret_cast<int64_t>(window_func_context->getNullValueStartPos()));
  const auto null_start_pos_buf_ptr = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                          cgen_state_.get(),
                                          code_generator,
                                          co,
                                          null_start_pos_buf,
                                          pi64_type,
                                          WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                          .front();
  const auto null_start_pos_ptr =
      cgen_state_->ir_builder_.CreateGEP(get_int_type(64, cgen_state_->context_),
                                         null_start_pos_buf_ptr,
                                         partition_index_lv);
  auto null_start_pos_lv = cgen_state_->ir_builder_.CreateLoad(
      null_start_pos_ptr->getType()->getPointerElementType(),
      null_start_pos_ptr,
      "null_start_pos");
  const auto null_end_pos_buf = cgen_state_->llInt(
      reinterpret_cast<int64_t>(window_func_context->getNullValueEndPos()));
  const auto null_end_pos_buf_ptr = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                        cgen_state_.get(),
                                        code_generator,
                                        co,
                                        null_end_pos_buf,
                                        pi64_type,
                                        WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                        .front();
  const auto null_end_pos_ptr = cgen_state_->ir_builder_.CreateGEP(
      get_int_type(64, cgen_state_->context_), null_end_pos_buf_ptr, partition_index_lv);
  auto null_end_pos_lv = cgen_state_->ir_builder_.CreateLoad(
      null_end_pos_ptr->getType()->getPointerElementType(),
      null_end_pos_ptr,
      "null_end_pos");
  return std::make_pair(null_start_pos_lv, null_end_pos_lv);
}

std::pair<std::string, llvm::Value*> Executor::codegenLoadOrderKeyBufPtr(
    WindowFunctionContext* window_func_context,
    CodeGenerator* code_generator,
    const CompilationOptions& co) const {
  auto const order_key_ti =
      window_func_context->getWindowFunction()->getOrderKeys().front()->get_type_info();
  auto const order_key_size = order_key_ti.get_size();
  auto const order_col_type_name = get_col_type_name_by_size(
      order_key_size,
      window_func_context->getOrderKeyColumnBufferTypes().front().is_fp());
  size_t order_key_size_in_byte = order_key_size * 8;
  auto const order_key_type =
      order_key_ti.is_fp() ? get_fp_type(order_key_size_in_byte, cgen_state_->context_)
                           : get_int_type(order_key_size_in_byte, cgen_state_->context_);
  auto const order_key_buf_type = llvm::PointerType::get(order_key_type, 0);
  auto const order_key_buf = cgen_state_->llInt(
      reinterpret_cast<int64_t>(window_func_context->getOrderKeyColumnBuffers().front()));
  auto const order_key_buf_ptr_lv = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                        cgen_state_.get(),
                                        code_generator,
                                        co,
                                        order_key_buf,
                                        order_key_buf_type,
                                        WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                        .front();
  return std::make_pair(order_col_type_name, order_key_buf_ptr_lv);
}

WindowPartitionBufferPtrs Executor::codegenLoadPartitionBuffers(
    WindowFunctionContext* window_func_context,
    CodeGenerator* code_generator,
    const CompilationOptions& co,
    llvm::Value* partition_index_lv) const {
  WindowPartitionBufferPtrs bufferPtrs;
  const auto pi64_type =
      llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
  const auto pi32_type =
      llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0);

  // partial sum of # elems of partitions
  auto partition_start_offset_buf_lv = cgen_state_->llInt(
      reinterpret_cast<int64_t>(window_func_context->partitionStartOffset()));
  auto partition_start_offset_ptr_lv = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                           cgen_state_.get(),
                                           code_generator,
                                           co,
                                           partition_start_offset_buf_lv,
                                           pi64_type,
                                           WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                           .front();

  // get start offset of the current partition
  auto current_partition_start_offset_ptr_lv =
      cgen_state_->ir_builder_.CreateGEP(get_int_type(64, cgen_state_->context_),
                                         partition_start_offset_ptr_lv,
                                         partition_index_lv);
  bufferPtrs.current_partition_start_offset_lv = cgen_state_->ir_builder_.CreateLoad(
      current_partition_start_offset_ptr_lv->getType()->getPointerElementType(),
      current_partition_start_offset_ptr_lv);

  // row_id buf of the current partition
  const auto partition_rowid_buf_lv =
      cgen_state_->llInt(reinterpret_cast<int64_t>(window_func_context->payload()));
  const auto partition_rowid_ptr_lv = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                          cgen_state_.get(),
                                          code_generator,
                                          co,
                                          partition_rowid_buf_lv,
                                          pi32_type,
                                          WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                          .front();
  bufferPtrs.target_partition_rowid_ptr_lv =
      cgen_state_->ir_builder_.CreateGEP(get_int_type(32, cgen_state_->context_),
                                         partition_rowid_ptr_lv,
                                         bufferPtrs.current_partition_start_offset_lv);

  // row_id buf of ordered current partition
  const auto sorted_rowid_lv = cgen_state_->llInt(
      reinterpret_cast<int64_t>(window_func_context->sortedPartition()));
  const auto sorted_rowid_ptr_lv = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                       cgen_state_.get(),
                                       code_generator,
                                       co,
                                       sorted_rowid_lv,
                                       pi64_type,
                                       WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                       .front();
  bufferPtrs.target_partition_sorted_rowid_ptr_lv =
      cgen_state_->ir_builder_.CreateGEP(get_int_type(64, cgen_state_->context_),
                                         sorted_rowid_ptr_lv,
                                         bufferPtrs.current_partition_start_offset_lv);

  // # elems per partition
  const auto partition_count_buf =
      cgen_state_->llInt(reinterpret_cast<int64_t>(window_func_context->counts()));
  auto partition_count_buf_ptr_lv = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                        cgen_state_.get(),
                                        code_generator,
                                        co,
                                        partition_count_buf,
                                        pi32_type,
                                        WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                        .front();

  // # elems of the given partition
  const auto num_elem_current_partition_ptr =
      cgen_state_->ir_builder_.CreateGEP(get_int_type(32, cgen_state_->context_),
                                         partition_count_buf_ptr_lv,
                                         partition_index_lv);
  bufferPtrs.num_elem_current_partition_lv = cgen_state_->castToTypeIn(
      cgen_state_->ir_builder_.CreateLoad(
          num_elem_current_partition_ptr->getType()->getPointerElementType(),
          num_elem_current_partition_ptr),
      64);
  return bufferPtrs;
}

std::pair<llvm::Value*, llvm::Value*> Executor::codegenFrameBoundRange(
    const Analyzer::WindowFunction* window_func,
    CodeGenerator& code_generator,
    const CompilationOptions& co) {
  const auto frame_start_bound = window_func->getFrameStartBound();
  const auto frame_end_bound = window_func->getFrameEndBound();
  auto frame_start_bound_expr_lv =
      codegenFrameBoundExpr(window_func, frame_start_bound, code_generator, co);
  auto frame_end_bound_expr_lv =
      codegenFrameBoundExpr(window_func, frame_end_bound, code_generator, co);
  CHECK(frame_start_bound_expr_lv);
  CHECK(frame_end_bound_expr_lv);
  return std::make_pair(frame_start_bound_expr_lv, frame_end_bound_expr_lv);
}

std::pair<llvm::Value*, llvm::Value*> Executor::codegenWindowFrameBounds(
    WindowFunctionContext* window_func_context,
    const Analyzer::WindowFrame* frame_start_bound,
    const Analyzer::WindowFrame* frame_end_bound,
    llvm::Value* order_key_col_null_val_lv,
    WindowFrameBoundFuncArgs& args,
    CodeGenerator& code_generator) {
  const auto window_func = window_func_context->getWindowFunction();
  CHECK(window_func);
  const auto is_timestamp_type_frame = frame_start_bound->hasTimestampTypeFrameBound() ||
                                       frame_end_bound->hasTimestampTypeFrameBound();

  if (window_func->hasRangeModeFraming()) {
    CHECK(window_func_context->getOrderKeyColumnBuffers().size() == 1);
    CHECK(window_func->getOrderKeys().size() == 1UL);
    CHECK(window_func_context->getOrderKeyColumnBuffers().size() == 1UL);
    args.order_type_col_name = getOrderKeyTypeName(window_func_context);
    args.current_col_value_lv =
        codegenLoadCurrentValueFromColBuf(window_func_context, code_generator, args);
  }

  auto get_order_key_null_val = [is_timestamp_type_frame,
                                 &order_key_col_null_val_lv,
                                 this](const Analyzer::WindowFrame* frame_bound) {
    return is_timestamp_type_frame && !frame_bound->isCurrentRowBound()
               ? cgen_state_->castToTypeIn(order_key_col_null_val_lv, 64)
               : order_key_col_null_val_lv;
  };
  auto frame_start_bound_lv =
      codegenFrameBound(true,
                        window_func->hasRangeModeFraming(),
                        window_func->isFrameNavigateWindowFunction(),
                        frame_start_bound,
                        is_timestamp_type_frame,
                        get_order_key_null_val(frame_start_bound),
                        args);
  auto frame_end_bound_lv =
      codegenFrameBound(false,
                        window_func->hasRangeModeFraming(),
                        window_func->isFrameNavigateWindowFunction(),
                        frame_end_bound,
                        is_timestamp_type_frame,
                        get_order_key_null_val(frame_end_bound),
                        args);
  CHECK(frame_start_bound_lv);
  CHECK(frame_end_bound_lv);
  return std::make_pair(frame_start_bound_lv, frame_end_bound_lv);
}

llvm::Value* Executor::codegenWindowFunctionAggregateCalls(llvm::Value* aggregate_state,
                                                           const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto window_func_ti = get_adjusted_window_type_info(window_func);
  const auto window_func_null_val =
      window_func_ti.is_fp()
          ? cgen_state_->inlineFpNull(window_func_ti)
          : cgen_state_->castToTypeIn(cgen_state_->inlineIntNull(window_func_ti), 64);
  if (window_func_context->elementCount() == 0) {
    // we do not need to generate a code for an empty input table
    return window_func->getKind() == SqlWindowFunctionKind::AVG
               ? cgen_state_->inlineFpNull(SQLTypeInfo(SQLTypes::kDOUBLE))
               : window_func_null_val;
  }
  const auto& args = window_func->getArgs();
  CodeGenerator code_generator(this);
  if (window_func_context->needsToBuildAggregateTree()) {
    // compute an aggregated value for each row of the window frame by using segment
    // tree when constructing a window context, we build a necessary segment tree (so
    // called `aggregate tree`) to query the aggregated value of the specific window
    // frame
    const auto pi64_type =
        llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
    const auto ppi64_type = llvm::PointerType::get(
        llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0), 0);

    auto [frame_start_bound_expr_lv, frame_end_bound_expr_lv] =
        codegenFrameBoundRange(window_func, code_generator, co);

    // compute aggregated value over the computed frame range
    auto current_row_pos_lv = code_generator.posArg(nullptr);
    auto partition_index_lv = codegenCurrentPartitionIndex(
        window_func_context, &code_generator, co, current_row_pos_lv);

    // ordering column buffer
    const auto target_col_ti = args.front()->get_type_info();
    const auto target_col_size = target_col_ti.get_size();
    const auto col_type_name =
        get_col_type_name_by_size(target_col_size, target_col_ti.is_fp());

    const auto partition_buf_ptrs = codegenLoadPartitionBuffers(
        window_func_context, &code_generator, co, partition_index_lv);

    auto [order_col_type_name, order_key_buf_ptr_lv] =
        codegenLoadOrderKeyBufPtr(window_func_context, &code_generator, co);

    // null value of the ordering column
    const auto order_key_buf_ti =
        window_func_context->getOrderKeyColumnBufferTypes().front();
    auto const ordering_spec = window_func->getCollation().front();
    llvm::Value* order_key_col_null_val_lv{nullptr};
    switch (order_key_buf_ti.get_type()) {
      case kDATE:
      case kTIMESTAMP:
      case kTIME: {
        if (order_key_buf_ti.get_compression() == kENCODING_FIXED ||
            order_key_buf_ti.get_compression() == kENCODING_DATE_IN_DAYS) {
          auto null_val = inline_fixed_encoding_null_val(order_key_buf_ti);
          order_key_col_null_val_lv = cgen_state_->llInt((int32_t)null_val);
          break;
        }
      }
      default: {
        order_key_col_null_val_lv = cgen_state_->inlineNull(order_key_buf_ti);
        break;
      }
    }

    auto [null_start_pos_lv, null_end_pos_lv] = codegenFrameNullRange(
        window_func_context, &code_generator, co, partition_index_lv);
    auto nulls_first_lv = cgen_state_->llBool(ordering_spec.nulls_first);

    WindowFrameBoundFuncArgs WindowFrameBoundFuncArgs{
        frame_start_bound_expr_lv,
        frame_end_bound_expr_lv,
        current_row_pos_lv,
        nullptr,
        partition_buf_ptrs.current_partition_start_offset_lv,
        cgen_state_->llInt((int64_t)0),
        cgen_state_->llInt((int64_t)1),
        partition_buf_ptrs.num_elem_current_partition_lv,
        order_key_buf_ptr_lv,
        "",
        partition_buf_ptrs.target_partition_rowid_ptr_lv,
        partition_buf_ptrs.target_partition_sorted_rowid_ptr_lv,
        nulls_first_lv,
        null_start_pos_lv,
        null_end_pos_lv};
    auto [frame_start_bound_lv, frame_end_bound_lv] =
        codegenWindowFrameBounds(window_func_context,
                                 window_func->getFrameStartBound(),
                                 window_func->getFrameEndBound(),
                                 order_key_col_null_val_lv,
                                 WindowFrameBoundFuncArgs,
                                 code_generator);

    // codegen to send a query with frame bound to aggregate tree searcher
    llvm::ConstantInt* aggregation_trees_lv{nullptr};
    llvm::Value* invalid_val_lv{nullptr};
    llvm::Value* null_val_lv{nullptr};
    std::string aggregation_tree_search_func_name{"search_"};
    std::string aggregation_tree_getter_func_name{"get_"};

    // prepare null values and aggregate_tree getter and searcher depending on
    // a type of the ordering column
    auto agg_expr_ti = args.front()->get_type_info();
    if (agg_expr_ti.is_fp()) {
      if (window_func->getKind() == SqlWindowFunctionKind::MIN) {
        invalid_val_lv = cgen_state_->llFp(std::numeric_limits<double>::max());
      } else if (window_func->getKind() == SqlWindowFunctionKind::MAX) {
        invalid_val_lv = cgen_state_->llFp(std::numeric_limits<double>::lowest());
      } else {
        invalid_val_lv = cgen_state_->llFp((double)0);
      }
      null_val_lv = cgen_state_->inlineFpNull(SQLTypeInfo(kDOUBLE));
      aggregation_tree_search_func_name += "double";
      aggregation_tree_getter_func_name += "double";
    } else {
      if (window_func->getKind() == SqlWindowFunctionKind::MIN) {
        invalid_val_lv = cgen_state_->llInt(std::numeric_limits<int64_t>::max());
      } else if (window_func->getKind() == SqlWindowFunctionKind::MAX) {
        invalid_val_lv = cgen_state_->llInt(std::numeric_limits<int64_t>::lowest());
      } else {
        invalid_val_lv = cgen_state_->llInt((int64_t)0);
      }
      null_val_lv = cgen_state_->llInt(inline_int_null_value<int64_t>());
      aggregation_tree_search_func_name += "int64_t";
      aggregation_tree_getter_func_name += "integer";
    }

    // derived aggregation has a different code path
    if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
      aggregation_tree_search_func_name += "_derived";
      aggregation_tree_getter_func_name += "_derived";
    }

    // get a buffer holding aggregate trees for each partition
    if (agg_expr_ti.is_fp()) {
      if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
        aggregation_trees_lv = cgen_state_->llInt(reinterpret_cast<int64_t>(
            window_func_context->getDerivedAggregationTreesForDoubleTypeWindowExpr()));
      } else {
        aggregation_trees_lv = cgen_state_->llInt(reinterpret_cast<int64_t>(
            window_func_context->getAggregationTreesForDoubleTypeWindowExpr()));
      }
    } else {
      if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
        aggregation_trees_lv = cgen_state_->llInt(reinterpret_cast<int64_t>(
            window_func_context->getDerivedAggregationTreesForIntegerTypeWindowExpr()));
      } else {
        aggregation_trees_lv = cgen_state_->llInt(reinterpret_cast<int64_t>(
            window_func_context->getAggregationTreesForIntegerTypeWindowExpr()));
      }
    }

    CHECK(aggregation_trees_lv);
    CHECK(invalid_val_lv);
    aggregation_tree_search_func_name += "_aggregation_tree";
    aggregation_tree_getter_func_name += "_aggregation_tree";

    // get the aggregate tree of the current partition from a window context
    auto aggregation_trees_ptr = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                     cgen_state_.get(),
                                     &code_generator,
                                     co,
                                     aggregation_trees_lv,
                                     ppi64_type,
                                     WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                     .front();
    auto target_aggregation_tree_lv = cgen_state_->emitCall(
        aggregation_tree_getter_func_name, {aggregation_trees_ptr, partition_index_lv});

    // a depth of segment tree
    const auto tree_depth_buf = cgen_state_->llInt(
        reinterpret_cast<int64_t>(window_func_context->getAggregateTreeDepth()));
    const auto tree_depth_buf_ptr = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                        cgen_state_.get(),
                                        &code_generator,
                                        co,
                                        tree_depth_buf,
                                        pi64_type,
                                        WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                        .front();
    const auto current_partition_tree_depth_buf_ptr = cgen_state_->ir_builder_.CreateGEP(
        get_int_type(64, cgen_state_->context_), tree_depth_buf_ptr, partition_index_lv);
    const auto current_partition_tree_depth_lv = cgen_state_->ir_builder_.CreateLoad(
        current_partition_tree_depth_buf_ptr->getType()->getPointerElementType(),
        current_partition_tree_depth_buf_ptr);

    // a fanout of the current partition's segment tree
    const auto aggregation_tree_fanout_lv = cgen_state_->llInt(
        static_cast<int64_t>(window_func_context->getAggregateTreeFanout()));

    // agg_type
    const auto agg_type_lv =
        cgen_state_->llInt(static_cast<int32_t>(window_func->getKind()));

    // send a query to the aggregate tree with the frame range:
    // `frame_start_bound_lv` ~ `frame_end_bound_lv`
    auto res_lv =
        cgen_state_->emitCall(aggregation_tree_search_func_name,
                              {target_aggregation_tree_lv,
                               frame_start_bound_lv,
                               frame_end_bound_lv,
                               current_partition_tree_depth_lv,
                               aggregation_tree_fanout_lv,
                               cgen_state_->llBool(agg_expr_ti.is_decimal()),
                               cgen_state_->llInt((int64_t)agg_expr_ti.get_scale()),
                               invalid_val_lv,
                               null_val_lv,
                               agg_type_lv});

    // handling returned null value if exists
    std::string null_handler_func_name{"handle_null_val_"};
    std::vector<llvm::Value*> null_handler_args{res_lv, null_val_lv};

    // determine null_handling function's name
    if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
      // average aggregate function returns a value as a double
      // (and our search* function also returns a double)
      if (agg_expr_ti.is_fp()) {
        // fp type: double null value
        null_handler_func_name += "double_double";
      } else {
        // non-fp type: int64_t null type
        null_handler_func_name += "double_int64_t";
      }
    } else if (agg_expr_ti.is_fp()) {
      // fp type: double null value
      null_handler_func_name += "double_double";
    } else {
      // non-fp type: int64_t null type
      null_handler_func_name += "int64_t_int64_t";
    }
    null_handler_func_name += "_window_framing_agg";

    // prepare null_val
    if (window_func->getKind() == SqlWindowFunctionKind::COUNT) {
      if (agg_expr_ti.is_fp()) {
        null_handler_args.push_back(cgen_state_->llFp((double)0));
      } else {
        null_handler_args.push_back(cgen_state_->llInt((int64_t)0));
      }
    } else if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
      null_handler_args.push_back(cgen_state_->inlineFpNull(SQLTypeInfo(kDOUBLE)));
    } else {
      null_handler_args.push_back(cgen_state_->castToTypeIn(window_func_null_val, 64));
    }
    res_lv = cgen_state_->emitCall(null_handler_func_name, null_handler_args);

    // when AGG_TYPE is double, we get a double type return value we expect an integer
    // type value for the count aggregation
    if (window_func->getKind() == SqlWindowFunctionKind::COUNT && agg_expr_ti.is_fp()) {
      return cgen_state_->ir_builder_.CreateFPToSI(
          res_lv, get_int_type(64, cgen_state_->context_));
    } else if (window_func->getKind() != SqlWindowFunctionKind::COUNT &&
               agg_expr_ti.is_date_in_days()) {
      // we need to decode the "encoded" date column value
      auto date_null_val = get_null_value_by_size(cgen_state_.get(), agg_expr_ti);
      if (date_null_val->getType()->getScalarSizeInBits() != 32) {
        date_null_val = cgen_state_->castToTypeIn(date_null_val, 32);
      }
      return cgen_state_->emitCall("fixed_width_date_decode",
                                   {res_lv, date_null_val, null_val_lv});
    }
    return res_lv;
  } else {
    auto agg_name = get_window_agg_name(window_func->getKind(), window_func_ti);
    Analyzer::Expr* arg_target_expr;
    std::vector<llvm::Value*> agg_func_args{aggregate_state};
    auto modified_window_func_null_val = window_func_null_val;
    if (args.empty() ||
        (window_func->getKind() == SqlWindowFunctionKind::COUNT &&
         dynamic_cast<Analyzer::Constant*>(args.front().get()) != nullptr)) {
      // a count aggregation without an expression: COUNT(1) or COUNT(*)
      agg_func_args.push_back(cgen_state_->llInt(int64_t(1)));
    } else {
      // we use #base_agg_func_name##_skip_val agg function
      // i.e.,int64_t agg_sum_skip_val(int64_t* agg, int64_t val, int64_t skip_val)
      arg_target_expr = args.front().get();
      const auto arg_lvs = code_generator.codegen(arg_target_expr, true, co);
      CHECK_EQ(arg_lvs.size(), size_t(1));
      // handling current row's value
      auto crt_val = arg_lvs.front();
      if ((window_func->getKind() == SqlWindowFunctionKind::SUM ||
           window_func->getKind() == SqlWindowFunctionKind::SUM_IF) &&
          !window_func_ti.is_fp()) {
        crt_val = code_generator.codegenCastBetweenIntTypes(
            arg_lvs.front(), args.front()->get_type_info(), window_func_ti, false);
      }
      agg_func_args.push_back(window_func_ti.get_type() == kFLOAT
                                  ? crt_val
                                  : cgen_state_->castToTypeIn(crt_val, 64));
      // handle null value and conditional value for conditional aggregates if necessary
      llvm::Value* cond_lv{nullptr};
      if (window_function_conditional_aggregate(window_func->getKind())) {
        switch (window_func->getKind()) {
          case SqlWindowFunctionKind::COUNT_IF:
            // COUNT_IF has a single condition expr which is always bool type
            modified_window_func_null_val = cgen_state_->castToTypeIn(
                cgen_state_->inlineNull(SQLTypeInfo(kTINYINT)), 64);
            break;
          case SqlWindowFunctionKind::SUM_IF: {
            // FP type input col uses its own null value depending on the type
            // otherwise (integer type input col), we use 8-byte type
            if (args.front()->get_type_info().is_integer()) {
              agg_func_args[1] = cgen_state_->castToTypeIn(agg_func_args[1], 64);
              // keep the null value but casting its type to 8-byte
              modified_window_func_null_val =
                  cgen_state_->castToTypeIn(window_func_null_val, 64);
            }
            auto cond_expr_lv = code_generator.codegen(args[1].get(), true, co).front();
            cond_lv =
                codegenConditionalAggregateCondValSelector(cond_expr_lv, kSUM_IF, co);
          }
          default:
            break;
        }
      }
      agg_name += "_skip_val";
      agg_func_args.push_back(modified_window_func_null_val);
      if (cond_lv) {
        agg_func_args.push_back(cond_lv);
      }
    }
    cgen_state_->emitCall(agg_name, agg_func_args);
    if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
      codegenWindowAvgEpilogue(
          &code_generator, co, agg_func_args[1], window_func_null_val);
    }
    return codegenAggregateWindowState(&code_generator, co, aggregate_state);
  }
}

void Executor::codegenWindowAvgEpilogue(CodeGenerator* code_generator,
                                        const CompilationOptions& co,
                                        llvm::Value* crt_val,
                                        llvm::Value* window_func_null_val) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const auto window_func = window_func_context->getWindowFunction();
  const auto window_func_ti = get_adjusted_window_type_info(window_func);
  const auto pi32_type =
      llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0);
  const auto pi64_type =
      llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
  const auto aggregate_state_type =
      window_func_ti.get_type() == kFLOAT ? pi32_type : pi64_type;
  const auto aggregate_state_count_i64 = cgen_state_->llInt(
      reinterpret_cast<const int64_t>(window_func_context->aggregateStateCount()));
  auto aggregate_state_count = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                   cgen_state_.get(),
                                   code_generator,
                                   co,
                                   aggregate_state_count_i64,
                                   aggregate_state_type,
                                   WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                   .front();
  std::string agg_count_func_name = "agg_count";
  switch (window_func_ti.get_type()) {
    case kFLOAT: {
      agg_count_func_name += "_float";
      break;
    }
    case kDOUBLE: {
      agg_count_func_name += "_double";
      break;
    }
    default: {
      break;
    }
  }
  agg_count_func_name += "_skip_val";
  cgen_state_->emitCall(agg_count_func_name,
                        {aggregate_state_count, crt_val, window_func_null_val});
}

llvm::Value* Executor::codegenAggregateWindowState(CodeGenerator* code_generator,
                                                   const CompilationOptions& co,
                                                   llvm::Value* aggregate_state) {
  AUTOMATIC_IR_METADATA(cgen_state_.get());
  const auto pi32_type =
      llvm::PointerType::get(get_int_type(32, cgen_state_->context_), 0);
  const auto pi64_type =
      llvm::PointerType::get(get_int_type(64, cgen_state_->context_), 0);
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(this);
  const Analyzer::WindowFunction* window_func = window_func_context->getWindowFunction();
  const auto window_func_ti = get_adjusted_window_type_info(window_func);
  const auto aggregate_state_type =
      window_func_ti.get_type() == kFLOAT ? pi32_type : pi64_type;
  if (window_func->getKind() == SqlWindowFunctionKind::AVG) {
    const auto aggregate_state_count_i64 = cgen_state_->llInt(
        reinterpret_cast<const int64_t>(window_func_context->aggregateStateCount()));
    auto aggregate_state_count = CodegenUtil::createPtrWithHoistedMemoryAddr(
                                     cgen_state_.get(),
                                     code_generator,
                                     co,
                                     aggregate_state_count_i64,
                                     aggregate_state_type,
                                     WindowFunctionContext::NUM_EXECUTION_DEVICES)
                                     .front();
    const auto double_null_lv = cgen_state_->inlineFpNull(SQLTypeInfo(kDOUBLE));
    switch (window_func_ti.get_type()) {
      case kFLOAT: {
        return cgen_state_->emitCall(
            "load_avg_float", {aggregate_state, aggregate_state_count, double_null_lv});
      }
      case kDOUBLE: {
        return cgen_state_->emitCall(
            "load_avg_double", {aggregate_state, aggregate_state_count, double_null_lv});
      }
      case kDECIMAL: {
        return cgen_state_->emitCall(
            "load_avg_decimal",
            {aggregate_state,
             aggregate_state_count,
             double_null_lv,
             cgen_state_->llInt<int32_t>(window_func_ti.get_scale())});
      }
      default: {
        return cgen_state_->emitCall(
            "load_avg_int", {aggregate_state, aggregate_state_count, double_null_lv});
      }
    }
  }
  if (window_func->getKind() == SqlWindowFunctionKind::COUNT) {
    return cgen_state_->ir_builder_.CreateLoad(
        aggregate_state->getType()->getPointerElementType(), aggregate_state);
  }
  switch (window_func_ti.get_type()) {
    case kFLOAT: {
      return cgen_state_->emitCall("load_float", {aggregate_state});
    }
    case kDOUBLE: {
      return cgen_state_->emitCall("load_double", {aggregate_state});
    }
    default: {
      return cgen_state_->ir_builder_.CreateLoad(
          aggregate_state->getType()->getPointerElementType(), aggregate_state);
    }
  }
}

llvm::Value* Executor::codegenConditionalAggregateCondValSelector(
    llvm::Value* cond_lv,
    SQLAgg const aggKind,
    CompilationOptions const& co) const {
  llvm::Value* res_cond_lv{nullptr};
  switch (aggKind) {
    case kSUM_IF:
      if (cond_lv->getType()->isIntegerTy(1)) {
        // cond_expr returns i1 type val, just need to cast to i8 type
        // i.e., cond_expr IS NULL
        res_cond_lv = cgen_state_->castToTypeIn(cond_lv, 8);
      } else {
        CHECK(cond_lv->getType()->isIntegerTy(8));
        // cond_expr may have null value instead of upcasted bool (i1-type) value
        // so we have to correctly set true condition
        // i.e., i8 @gt_int32_t_nullable_lhs(..., i64 -2147483648, i8 -128)
        // has one of the following i8-type values: 1, 0, -128
        auto true_cond_lv =
            cgen_state_->ir_builder_.CreateICmpEQ(cond_lv, cgen_state_->llInt((int8_t)1));
        res_cond_lv = cgen_state_->ir_builder_.CreateSelect(
            true_cond_lv, cgen_state_->llInt((int8_t)1), cgen_state_->llInt((int8_t)0));
      }
      break;
    default:
      break;
  }
  return res_cond_lv;
}
