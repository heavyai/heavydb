/*
 * Copyright 2019 OmniSci, Inc.
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
#include "ResultSet.h"

#include "CodeGenerator.h"
#include "IRCodegenUtils.h"
#include "LLVMGlobalContext.h"

#include "Shared/mapdpath.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

extern std::unique_ptr<llvm::Module> g_rt_module;
bool g_reduction_jit_interp{false};

namespace {

thread_local CodeCache g_code_cache(10000);

std::unique_ptr<llvm::Module> runtime_module_shallow_copy(llvm::LLVMContext& context,
                                                          CgenState* cgen_state) {
  return llvm::CloneModule(
#if LLVM_VERSION_MAJOR >= 7
      *g_rt_module.get(),
#else
      g_rt_module.get(),
#endif
      cgen_state->vmap_,
      [](const llvm::GlobalValue* gv) {
        auto func = llvm::dyn_cast<llvm::Function>(gv);
        if (!func) {
          return true;
        }
        return (func->getLinkage() == llvm::GlobalValue::LinkageTypes::PrivateLinkage ||
                func->getLinkage() == llvm::GlobalValue::LinkageTypes::InternalLinkage);
      });
}

llvm::Value* emit_load(llvm::Value* ptr, llvm::Type* loaded_type, CgenState* cgen_state) {
  return cgen_state->ir_builder_.CreateLoad(
      cgen_state->ir_builder_.CreateBitCast(ptr, loaded_type));
}

llvm::Value* emit_load_i32(llvm::Value* ptr, CgenState* cgen_state) {
  const auto pi32_type = llvm::Type::getInt32PtrTy(cgen_state->context_);
  return emit_load(ptr, pi32_type, cgen_state);
}

llvm::Value* emit_load_i64(llvm::Value* ptr, CgenState* cgen_state) {
  const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
  return emit_load(ptr, pi64_type, cgen_state);
}

void emit_aggregate_one_value(const std::string& agg_kind,
                              llvm::Value* val_ptr,
                              llvm::Value* other_ptr,
                              const size_t chosen_bytes,
                              const TargetInfo& agg_info,
                              CgenState* cgen_state) {
  const auto sql_type = get_compact_type(agg_info);
  const auto pi32_type = llvm::Type::getInt32PtrTy(cgen_state->context_);
  const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
  const auto pf32_type = llvm::Type::getFloatPtrTy(cgen_state->context_);
  const auto pf64_type = llvm::Type::getDoublePtrTy(cgen_state->context_);
  if (sql_type.is_fp()) {
    if (chosen_bytes == sizeof(float)) {
      const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type);
      const auto val = emit_load(other_ptr, pf32_type, cgen_state);
      cgen_state->emitCall("agg_" + agg_kind + "_float", {agg, val});
    } else {
      CHECK_EQ(chosen_bytes, sizeof(double));
      const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
      const auto val = emit_load(other_ptr, pf64_type, cgen_state);
      cgen_state->emitCall("agg_" + agg_kind + "_double", {agg, val});
    }
  } else {
    if (chosen_bytes == sizeof(int32_t)) {
      const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type);
      const auto val = emit_load(other_ptr, pi32_type, cgen_state);
      cgen_state->emitCall("agg_" + agg_kind + "_int32", {agg, val});
    } else {
      CHECK_EQ(chosen_bytes, sizeof(int64_t));
      const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
      const auto val = emit_load(other_ptr, pi64_type, cgen_state);
      cgen_state->emitCall("agg_" + agg_kind, {agg, val});
    }
  }
}

void emit_aggregate_one_nullable_value(const std::string& agg_kind,
                                       llvm::Value* val_ptr,
                                       llvm::Value* other_ptr,
                                       const int64_t init_val,
                                       const size_t chosen_bytes,
                                       const TargetInfo& agg_info,
                                       CgenState* cgen_state) {
  if (agg_info.skip_null_val) {
    const auto pi32_type = llvm::Type::getInt32PtrTy(cgen_state->context_);
    const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
    const auto pf32_type = llvm::Type::getFloatPtrTy(cgen_state->context_);
    const auto pf64_type = llvm::Type::getDoublePtrTy(cgen_state->context_);
    const auto sql_type = get_compact_type(agg_info);
    if (sql_type.is_fp()) {
      if (chosen_bytes == sizeof(float)) {
        const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type);
        const auto val = emit_load(other_ptr, pf32_type, cgen_state);
        const auto init_val_lv =
            cgen_state->llFp(*reinterpret_cast<const float*>(may_alias_ptr(&init_val)));
        cgen_state->emitCall("agg_" + agg_kind + "_float_skip_val",
                             {agg, val, init_val_lv});
      } else {
        CHECK_EQ(chosen_bytes, sizeof(double));
        const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
        const auto val = emit_load(other_ptr, pf64_type, cgen_state);
        const auto init_val_lv =
            cgen_state->llFp(*reinterpret_cast<const double*>(may_alias_ptr(&init_val)));
        cgen_state->emitCall("agg_" + agg_kind + "_double_skip_val",
                             {agg, val, init_val_lv});
      }
    } else {
      if (chosen_bytes == sizeof(int32_t)) {
        const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type);
        const auto val = emit_load(other_ptr, pi32_type, cgen_state);
        const auto init_val_lv = cgen_state->llInt<int32_t>(init_val);
        cgen_state->emitCall("agg_" + agg_kind + "_int32_skip_val",
                             {agg, val, init_val_lv});
      } else {
        CHECK_EQ(chosen_bytes, sizeof(int64_t));
        const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
        const auto val = emit_load(other_ptr, pi64_type, cgen_state);
        const auto init_val_lv = cgen_state->llInt(init_val);
        cgen_state->emitCall("agg_" + agg_kind + "_skip_val", {agg, val, init_val_lv});
      }
    }
  } else {
    emit_aggregate_one_value(
        agg_kind, val_ptr, other_ptr, chosen_bytes, agg_info, cgen_state);
  }
}

void emit_aggregate_one_count(llvm::Value* val_ptr,
                              llvm::Value* other_ptr,
                              const size_t chosen_bytes,
                              CgenState* cgen_state) {
  if (chosen_bytes == sizeof(int32_t)) {
    const auto pi32_type = llvm::Type::getInt32PtrTy(cgen_state->context_);
    const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type);
    const auto val = emit_load(other_ptr, pi32_type, cgen_state);
    cgen_state->emitCall("agg_sum_int32", {agg, val});
  } else {
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
    const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
    const auto val = emit_load(other_ptr, pi64_type, cgen_state);
    cgen_state->emitCall("agg_sum", {agg, val});
  }
}

void emit_write_projection(llvm::Value* slot_pi8,
                           llvm::Value* other_pi8,
                           const int64_t init_val,
                           const size_t chosen_bytes,
                           CgenState* cgen_state) {
  const auto func_name = "write_projection_int" + std::to_string(chosen_bytes * 8);
  if (chosen_bytes == sizeof(int32_t)) {
    const auto proj_val = emit_load_i32(other_pi8, cgen_state);
    cgen_state->emitCall(func_name, {slot_pi8, proj_val, cgen_state->llInt(init_val)});
  } else {
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto proj_val = emit_load_i64(other_pi8, cgen_state);
    cgen_state->emitCall(func_name, {slot_pi8, proj_val, cgen_state->llInt(init_val)});
  }
}

ReductionCode setup_reduction_function(const QueryDescriptionType hash_type) {
  ReductionCode reduction_code;
  reduction_code.cgen_state.reset(new CgenState({}, false));
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  std::unique_ptr<llvm::Module> module(runtime_module_shallow_copy(ctx, cgen_state));
  cgen_state->module_ = module.get();
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto pvoid_type = llvm::PointerType::get(llvm::Type::getVoidTy(ctx), 0);
  const auto func_type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                              {pi8_type, pi8_type, pvoid_type, pvoid_type, pvoid_type},
                              false);
  const auto func = llvm::Function::Create(
      func_type, llvm::Function::PrivateLinkage, "reduce_one_entry", module.get());
  reduction_code.ir_reduce_func = func;
  {
    const auto i32_type = get_int_type(32, ctx);
    const auto pvoid_type = llvm::PointerType::get(llvm::Type::getVoidTy(ctx), 0);
    const auto reduction_idx_func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(ctx),
        {pi8_type, pi8_type, i32_type, i32_type, pvoid_type, pvoid_type, pvoid_type},
        false);
    reduction_code.ir_reduce_func_idx =
        llvm::Function::Create(reduction_idx_func_type,
                               llvm::Function::ExternalLinkage,
                               "reduce_one_entry_idx",
                               module.get());
    const auto arg_it = reduction_code.ir_reduce_func_idx->arg_begin();
    const auto this_buff_arg = &*arg_it;
    const auto that_buff_arg = &*(arg_it + 1);
    const auto that_entry_idx_arg = &*(arg_it + 2);
    const auto that_entry_count_arg = &*(arg_it + 3);
    const auto this_qmd_handle_arg = &*(arg_it + 4);
    const auto that_qmd_handle_arg = &*(arg_it + 5);
    const auto serialized_varlen_buffer_arg = &*(arg_it + 6);
    this_buff_arg->setName("this_buff");
    that_buff_arg->setName("that_buff");
    that_entry_idx_arg->setName("that_entry_idx");
    that_entry_count_arg->setName("that_entry_count");
    this_qmd_handle_arg->setName("this_qmd_handle");
    that_qmd_handle_arg->setName("that_qmd_handle");
    serialized_varlen_buffer_arg->setName("serialized_varlen_buffer");
  }
  {
    const auto is_empty_func_type =
        llvm::FunctionType::get(get_int_type(1, ctx), {pi8_type}, false);
    reduction_code.ir_is_empty_func =
        llvm::Function::Create(is_empty_func_type,
                               llvm::Function::InternalLinkage,
                               "is_empty_entry",
                               module.get());
  }
  const auto arg_it = func->arg_begin();
  switch (hash_type) {
    case QueryDescriptionType::GroupByPerfectHash: {
      const auto this_targets_ptr_arg = &*arg_it;
      const auto that_targets_ptr_arg = &*(arg_it + 1);
      this_targets_ptr_arg->setName("this_targets_ptr");
      that_targets_ptr_arg->setName("that_targets_ptr");
      break;
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      const auto this_row_ptr_arg = &*arg_it;
      const auto that_row_ptr_arg = &*(arg_it + 1);
      this_row_ptr_arg->setName("this_row_ptr");
      that_row_ptr_arg->setName("that_row_ptr");
      break;
    }
    default: {
      LOG(FATAL) << "Unexpected query description type";
    }
  }
  const auto this_qmd_arg = &*(arg_it + 2);
  const auto that_qmd_arg = &*(arg_it + 3);
  const auto serialized_varlen_buffer_arg = &*(arg_it + 4);
  this_qmd_arg->setName("this_qmd");
  that_qmd_arg->setName("that_qmd");
  serialized_varlen_buffer_arg->setName("serialized_varlen_buffer_arg");
  const auto bb_entry = llvm::BasicBlock::Create(ctx, ".entry", func, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  reduction_code.module = std::move(module);
  return std::move(reduction_code);
}

ExecutionEngineWrapper generate_native_reduction_code(llvm::Function* func) {
  auto module = func->getParent();

  llvm::ExecutionEngine* execution_engine{nullptr};

  std::string err_str;
  std::unique_ptr<llvm::Module> owner(module);
  llvm::EngineBuilder eb(std::move(owner));
  eb.setErrorStr(&err_str);
  eb.setEngineKind(llvm::EngineKind::Interpreter);
  execution_engine = eb.create();
  CHECK(execution_engine);

  return ExecutionEngineWrapper(execution_engine);
}

}  // namespace

ReductionCode ResultSetStorage::reduceOneEntryJIT(const ResultSetStorage& that) const {
  if (query_mem_desc_.didOutputColumnar()) {
    return {};
  }
  switch (query_mem_desc_.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByPerfectHash: {
      return reduceOneEntryNoCollisionsRowWiseJIT(that);
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      return reduceOneEntrySlotsBaselineJIT(that);
    }
    default: {
      return {};
    }
  }
}

namespace {

void return_early(llvm::Value* cond,
                  const ReductionCode& reduction_code,
                  llvm::Function* func) {
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto early_return = llvm::BasicBlock::Create(ctx, ".early_return", func, 0);
  const auto do_reduction = llvm::BasicBlock::Create(ctx, ".do_reduction", func, 0);
  cgen_state->ir_builder_.CreateCondBr(cond, early_return, do_reduction);
  cgen_state->ir_builder_.SetInsertPoint(early_return);
  cgen_state->ir_builder_.CreateRetVoid();
  cgen_state->ir_builder_.SetInsertPoint(do_reduction);
}

void return_on_that_empty(const ReductionCode& reduction_code) {
  auto cgen_state = reduction_code.cgen_state.get();
  const auto arg_it = reduction_code.ir_reduce_func->arg_begin();
  const auto that_row_ptr = &*(arg_it + 1);
  const auto that_is_empty =
      cgen_state->ir_builder_.CreateCall(reduction_code.ir_is_empty_func, that_row_ptr);
  return_early(that_is_empty, reduction_code, reduction_code.ir_reduce_func);
}

}  // namespace

ReductionCode ResultSetStorage::reduceOneEntryNoCollisionsRowWiseJIT(
    const ResultSetStorage& that) const {
  ReductionCode reduction_code =
      setup_reduction_function(query_mem_desc_.getQueryDescriptionType());
  return_on_that_empty(reduction_code);

  const auto& col_slot_context = query_mem_desc_.getColSlotContext();

  const auto arg_it = reduction_code.ir_reduce_func->arg_begin();
  const auto this_row_ptr = &*arg_it;
  const auto that_row_ptr = &*(arg_it + 1);

  const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
  auto cgen_state = reduction_code.cgen_state.get();
  if (key_bytes) {  // copy the key from right hand side
    cgen_state->ir_builder_.CreateMemCpy(
        this_row_ptr, 0, that_row_ptr, 0, cgen_state->llInt<int32_t>(key_bytes));
  }

  const auto key_bytes_with_padding = align_to_int64(key_bytes);
  const auto key_bytes_lv = cgen_state->llInt<int32_t>(key_bytes_with_padding);
  const auto this_targets_start_ptr =
      cgen_state->ir_builder_.CreateGEP(this_row_ptr, key_bytes_lv);
  const auto that_targets_start_ptr =
      cgen_state->ir_builder_.CreateGEP(that_row_ptr, key_bytes_lv);

  llvm::Value* this_targets_ptr = this_targets_start_ptr;
  llvm::Value* that_targets_ptr = that_targets_start_ptr;
  size_t init_agg_val_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
       ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    const auto& slots_for_col = col_slot_context.getSlotsForCol(target_logical_idx);
    llvm::Value* this_ptr2{nullptr};
    llvm::Value* that_ptr2{nullptr};

    bool two_slot_target{false};
    if (target_info.is_agg &&
        (target_info.agg_kind == kAVG ||
         (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()))) {
      // Note that this assumes if one of the slot pairs in a given target is an array,
      // all slot pairs are arrays. Currently this is true for all geo targets, but we
      // should better codify and store this information in the future
      two_slot_target = true;
    }

    for (size_t target_slot_idx = slots_for_col.front();
         target_slot_idx < slots_for_col.back() + 1;
         target_slot_idx += 2) {
      const auto slot_off_val = query_mem_desc_.getPaddedSlotWidthBytes(target_slot_idx);
      const auto slot_off = cgen_state->llInt<int32_t>(slot_off_val);
      if (UNLIKELY(two_slot_target)) {
        this_ptr2 = cgen_state->ir_builder_.CreateGEP(this_targets_ptr, slot_off);
        that_ptr2 = cgen_state->ir_builder_.CreateGEP(that_targets_ptr, slot_off);
      }
      reduceOneSlotJIT(this_targets_ptr,
                       this_ptr2,
                       that_targets_ptr,
                       that_ptr2,
                       target_info,
                       target_logical_idx,
                       target_slot_idx,
                       init_agg_val_idx,
                       that,
                       slots_for_col.front(),
                       reduction_code);
      auto increment_agg_val_idx_maybe =
          [&init_agg_val_idx, &target_logical_idx, this](const int slot_count) {
            if (query_mem_desc_.targetGroupbyIndicesSize() == 0 ||
                query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) < 0) {
              init_agg_val_idx += slot_count;
            }
          };
      if (target_logical_idx + 1 == targets_.size() &&
          target_slot_idx + 1 >= slots_for_col.back()) {
        break;
      }
      if (UNLIKELY(two_slot_target)) {
        increment_agg_val_idx_maybe(2);
        const auto two_slot_off = cgen_state->llInt<int32_t>(
            slot_off_val + query_mem_desc_.getPaddedSlotWidthBytes(target_slot_idx + 1));
        this_targets_ptr =
            cgen_state->ir_builder_.CreateGEP(this_targets_ptr, two_slot_off);
        that_targets_ptr =
            cgen_state->ir_builder_.CreateGEP(that_targets_ptr, two_slot_off);
      } else {
        increment_agg_val_idx_maybe(1);
        this_targets_ptr = cgen_state->ir_builder_.CreateGEP(this_targets_ptr, slot_off);
        that_targets_ptr = cgen_state->ir_builder_.CreateGEP(that_targets_ptr, slot_off);
      }
    }
  }
  return finalizeReductionCode(std::move(reduction_code), cgen_state);
}

ReductionCode ResultSetStorage::reduceOneEntrySlotsBaselineJIT(
    const ResultSetStorage& that) const {
  ReductionCode reduction_code =
      setup_reduction_function(query_mem_desc_.getQueryDescriptionType());
  auto cgen_state = reduction_code.cgen_state.get();
  const auto arg_it = reduction_code.ir_reduce_func->arg_begin();
  const auto this_targets_ptr_arg = &*arg_it;
  const auto that_targets_ptr_arg = &*(arg_it + 1);
  llvm::Value* this_ptr1 = this_targets_ptr_arg;
  llvm::Value* that_ptr1 = that_targets_ptr_arg;
  size_t j = 0;
  size_t init_agg_val_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
       ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    llvm::Value* this_ptr2{nullptr};
    llvm::Value* that_ptr2{nullptr};
    if (target_info.is_agg &&
        (target_info.agg_kind == kAVG ||
         (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()))) {
      const auto second_slot_rel_off = cgen_state->llInt<int32_t>(sizeof(int64_t));
      this_ptr2 = cgen_state->ir_builder_.CreateGEP(this_ptr1, second_slot_rel_off);
      that_ptr2 = cgen_state->ir_builder_.CreateGEP(that_ptr1, second_slot_rel_off);
    }
    reduceOneSlotJIT(this_ptr1,
                     this_ptr2,
                     that_ptr1,
                     that_ptr2,
                     target_info,
                     target_logical_idx,
                     j,
                     init_agg_val_idx,
                     that,
                     j,
                     reduction_code);
    if (target_logical_idx + 1 == targets_.size()) {
      break;
    }
    if (query_mem_desc_.targetGroupbyIndicesSize() == 0) {
      init_agg_val_idx = advance_slot(init_agg_val_idx, target_info, false);
    } else {
      if (query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) < 0) {
        init_agg_val_idx = advance_slot(init_agg_val_idx, target_info, false);
      }
    }
    j = advance_slot(j, target_info, false);
    auto next_slot_rel_off =
        cgen_state->llInt<int32_t>(init_agg_val_idx * sizeof(int64_t));
    this_ptr1 =
        cgen_state->ir_builder_.CreateGEP(this_targets_ptr_arg, next_slot_rel_off);
    that_ptr1 =
        cgen_state->ir_builder_.CreateGEP(that_targets_ptr_arg, next_slot_rel_off);
  }
  return finalizeReductionCode(std::move(reduction_code), cgen_state);
}

extern "C" void get_group_value_reduction_rt(int8_t* groups_buffer,
                                             const int8_t* key,
                                             const uint32_t key_count,
                                             const void* this_qmd_handle,
                                             const int8_t* that_buff,
                                             const uint32_t that_entry_idx,
                                             const uint32_t that_entry_count,
                                             const uint32_t row_size_bytes,
                                             int64_t** buff_out,
                                             uint8_t* empty) {
  const auto& this_qmd = *reinterpret_cast<const QueryMemoryDescriptor*>(this_qmd_handle);
  const auto gvi = get_group_value_reduction(reinterpret_cast<int64_t*>(groups_buffer),
                                             this_qmd.getEntryCount(),
                                             reinterpret_cast<const int64_t*>(key),
                                             key_count,
                                             this_qmd.getEffectiveKeyWidth(),
                                             this_qmd,
                                             reinterpret_cast<const int64_t*>(that_buff),
                                             that_entry_idx,
                                             that_entry_count,
                                             row_size_bytes >> 3);
  *buff_out = gvi.first;
  *empty = gvi.second;
}

void ResultSetStorage::reduceOneEntryNoCollisionsRowWiseIdxJIT(
    const ReductionCode& reduction_code) const {
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash);
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", reduction_code.ir_reduce_func_idx, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  const auto arg_it = reduction_code.ir_reduce_func_idx->arg_begin();
  const auto this_buff = &*arg_it;
  const auto that_buff = &*(arg_it + 1);
  const auto entry_idx = &*(arg_it + 2);
  const auto this_qmd_handle = &*(arg_it + 4);
  const auto that_qmd_handle = &*(arg_it + 5);
  const auto serialized_varlen_buffer_arg = &*(arg_it + 6);
  const auto row_bytes = cgen_state->llInt<int32_t>(get_row_bytes(query_mem_desc_));
  const auto row_off_in_bytes = cgen_state->ir_builder_.CreateMul(entry_idx, row_bytes);
  const auto this_row_ptr =
      cgen_state->ir_builder_.CreateGEP(this_buff, row_off_in_bytes);
  const auto that_row_ptr =
      cgen_state->ir_builder_.CreateGEP(that_buff, row_off_in_bytes);
  cgen_state->ir_builder_.CreateCall(reduction_code.ir_reduce_func,
                                     {this_row_ptr,
                                      that_row_ptr,
                                      this_qmd_handle,
                                      that_qmd_handle,
                                      serialized_varlen_buffer_arg});
  cgen_state->ir_builder_.CreateRetVoid();
  verify_function_ir(reduction_code.ir_reduce_func_idx);
}

void ResultSetStorage::reduceOneEntryBaselineJIT(
    const ReductionCode& reduction_code) const {
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByBaselineHash);
  CHECK(!query_mem_desc_.hasKeylessHash());
  CHECK(!query_mem_desc_.didOutputColumnar());
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", reduction_code.ir_reduce_func_idx, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  const auto arg_it = reduction_code.ir_reduce_func_idx->arg_begin();
  const auto this_buff = &*arg_it;
  const auto that_buff = &*(arg_it + 1);
  const auto that_entry_idx = &*(arg_it + 2);
  const auto that_entry_count = &*(arg_it + 3);
  const auto this_qmd_handle = &*(arg_it + 4);
  const auto that_qmd_handle = &*(arg_it + 5);
  const auto serialized_varlen_buffer_arg = &*(arg_it + 6);
  const auto row_bytes = cgen_state->llInt<int32_t>(get_row_bytes(query_mem_desc_));
  const auto that_row_off = cgen_state->ir_builder_.CreateMul(that_entry_idx, row_bytes);
  const auto that_row_ptr = cgen_state->ir_builder_.CreateGEP(that_buff, that_row_off);
  const auto that_is_empty =
      cgen_state->ir_builder_.CreateCall(reduction_code.ir_is_empty_func, that_row_ptr);
  return_early(that_is_empty, reduction_code, reduction_code.ir_reduce_func_idx);
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
  const auto bool_type = llvm::Type::getInt8Ty(cgen_state->context_);
  const auto this_targets_ptr_i64_ptr =
      cgen_state->ir_builder_.CreateAlloca(pi64_type, cgen_state->llInt<int32_t>(1));
  const auto this_is_empty_ptr =
      cgen_state->ir_builder_.CreateAlloca(bool_type, cgen_state->llInt<int32_t>(1));
  cgen_state->emitExternalCall("get_group_value_reduction_rt",
                               llvm::Type::getVoidTy(ctx),
                               {this_buff,
                                that_row_ptr,
                                cgen_state->llInt<int32_t>(key_count),
                                this_qmd_handle,
                                that_buff,
                                that_entry_idx,
                                that_entry_count,
                                row_bytes,
                                this_targets_ptr_i64_ptr,
                                this_is_empty_ptr});
  const auto this_targets_ptr_i64 =
      cgen_state->ir_builder_.CreateLoad(this_targets_ptr_i64_ptr);
  llvm::Value* this_is_empty = cgen_state->ir_builder_.CreateLoad(this_is_empty_ptr);
  this_is_empty =
      cgen_state->ir_builder_.CreateTrunc(this_is_empty, get_int_type(1, ctx), "tobool");
  return_early(this_is_empty, reduction_code, reduction_code.ir_reduce_func_idx);
  const auto pi8_type = llvm::Type::getInt8PtrTy(cgen_state->context_);
  const auto key_qw_count = get_slot_off_quad(query_mem_desc_);
  const auto this_targets_ptr =
      cgen_state->ir_builder_.CreateBitCast(this_targets_ptr_i64, pi8_type);
  const auto key_byte_count = key_qw_count * sizeof(int64_t);
  const auto key_byte_count_lv = cgen_state->llInt<int32_t>(key_byte_count);
  const auto that_targets_ptr =
      cgen_state->ir_builder_.CreateGEP(that_row_ptr, key_byte_count_lv);
  cgen_state->ir_builder_.CreateCall(reduction_code.ir_reduce_func,
                                     {this_targets_ptr,
                                      that_targets_ptr,
                                      this_qmd_handle,
                                      that_qmd_handle,
                                      serialized_varlen_buffer_arg});
  cgen_state->ir_builder_.CreateRetVoid();
  verify_function_ir(reduction_code.ir_reduce_func_idx);
}

ReductionCode ResultSetStorage::finalizeReductionCode(ReductionCode reduction_code,
                                                      CgenState* cgen_state) const {
  cgen_state->ir_builder_.CreateRetVoid();
  verify_function_ir(reduction_code.ir_reduce_func);
  isEmptyJit(reduction_code);
  switch (query_mem_desc_.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByPerfectHash: {
      reduceOneEntryNoCollisionsRowWiseIdxJIT(reduction_code);
      break;
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      reduceOneEntryBaselineJIT(reduction_code);
      break;
    }
    default: {
      LOG(FATAL) << "Unexpected query description type";
    }
  }
  const auto key0 = serialize_llvm_object(reduction_code.ir_is_empty_func);
  const auto key1 = serialize_llvm_object(reduction_code.ir_reduce_func);
  const auto key2 = serialize_llvm_object(reduction_code.ir_reduce_func_idx);
  CodeCacheKey key{key0, key1, key2};
  const auto val_ptr = g_code_cache.get(key);
  if (val_ptr) {
    return {
        nullptr,
        std::get<1>(val_ptr->first.front()).get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<ReductionCode::FuncPtr>(std::get<0>(val_ptr->first.front()))};
  }
  CompilationOptions co{ExecutorDeviceType::CPU, false, ExecutorOptLevel::Default, false};
  reduction_code.module.release();
  auto ee =
      g_reduction_jit_interp
          ? generate_native_reduction_code(reduction_code.ir_reduce_func_idx)
          : CodeGenerator::generateNativeCPUCode(reduction_code.ir_reduce_func_idx,
                                                 {reduction_code.ir_reduce_func_idx},
                                                 co);
  reduction_code.func_ptr =
      g_reduction_jit_interp
          ? nullptr
          : reinterpret_cast<ReductionCode::FuncPtr>(
                ee->getPointerToFunction(reduction_code.ir_reduce_func_idx));
  reduction_code.execution_engine = ee.get();
  if (!g_reduction_jit_interp) {
    std::tuple<void*, ExecutionEngineWrapper> cache_val =
        std::make_tuple(reinterpret_cast<void*>(reduction_code.func_ptr), std::move(ee));
    std::vector<std::tuple<void*, ExecutionEngineWrapper>> cache_vals;
    cache_vals.emplace_back(std::move(cache_val));
    Executor::addCodeToCache({key},
                             std::move(cache_vals),
                             reduction_code.ir_reduce_func_idx->getParent(),
                             g_code_cache);
  }
  return std::move(reduction_code);
}

extern "C" int64_t read_int_from_buff_rt(const int8_t* ptr, const int8_t compact_sz) {
  return read_int_from_buff(ptr, compact_sz);
}

void ResultSetStorage::isEmptyJit(const ReductionCode& reduction_code) const {
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByPerfectHash ||
        query_mem_desc_.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByBaselineHash);
  CHECK(!query_mem_desc_.didOutputColumnar());
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", reduction_code.ir_is_empty_func, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  llvm::Value* key{nullptr};
  llvm::Value* empty_key_val{nullptr};
  const auto arg_it = reduction_code.ir_is_empty_func->arg_begin();
  const auto keys_ptr = &*arg_it;
  if (query_mem_desc_.hasKeylessHash()) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash);
    CHECK_GE(query_mem_desc_.getTargetIdxForKey(), 0);
    CHECK_LT(static_cast<size_t>(query_mem_desc_.getTargetIdxForKey()),
             target_init_vals_.size());
    const auto target_slot_off =
        get_byteoff_of_slot(query_mem_desc_.getTargetIdxForKey(), query_mem_desc_);
    const auto slot_ptr = cgen_state->ir_builder_.CreateGEP(
        keys_ptr, cgen_state->llInt<int32_t>(target_slot_off));
    const auto compact_sz = cgen_state->llInt<int32_t>(
        query_mem_desc_.getPaddedSlotWidthBytes(query_mem_desc_.getTargetIdxForKey()));
    key = cgen_state->emitExternalCall(
        "read_int_from_buff_rt", get_int_type(64, ctx), {slot_ptr, compact_sz});
    empty_key_val =
        cgen_state->llInt(target_init_vals_[query_mem_desc_.getTargetIdxForKey()]);
  } else {
    switch (query_mem_desc_.getEffectiveKeyWidth()) {
      case 4: {
        CHECK(QueryDescriptionType::GroupByPerfectHash !=
              query_mem_desc_.getQueryDescriptionType());
        key = emit_load_i32(keys_ptr, cgen_state);
        empty_key_val = cgen_state->llInt<int32_t>(EMPTY_KEY_32);
        break;
      }
      case 8: {
        key = emit_load_i64(keys_ptr, cgen_state);
        empty_key_val = cgen_state->llInt<int64_t>(EMPTY_KEY_64);
        break;
      }
      default:
        LOG(FATAL) << "Invalid key width";
    }
  }
  const auto ret = cgen_state->ir_builder_.CreateICmpEQ(key, empty_key_val);
  cgen_state->ir_builder_.CreateRet(ret);
  verify_function_ir(reduction_code.ir_is_empty_func);
}

namespace {

void varlen_buffer_sample(int8_t* this_ptr1,
                          int8_t* this_ptr2,
                          const int8_t* that_ptr1,
                          const int8_t* that_ptr2,
                          const int64_t init_val) {
  const auto rhs_proj_col = *reinterpret_cast<const int64_t*>(that_ptr1);
  if (rhs_proj_col != init_val) {
    *reinterpret_cast<int64_t*>(this_ptr1) = rhs_proj_col;
  }
  CHECK(this_ptr2 && that_ptr2);
  *reinterpret_cast<int64_t*>(this_ptr2) = *reinterpret_cast<const int64_t*>(that_ptr2);
}

}  // namespace

extern "C" void serialized_varlen_buffer_sample(
    const void* serialized_varlen_buffer_handle,
    int8_t* this_ptr1,
    int8_t* this_ptr2,
    const int8_t* that_ptr1,
    const int8_t* that_ptr2,
    const int64_t init_val,
    const int64_t length_to_elems) {
  if (!serialized_varlen_buffer_handle) {
    varlen_buffer_sample(this_ptr1, this_ptr2, that_ptr1, that_ptr2, init_val);
    return;
  }
  const auto& serialized_varlen_buffer =
      *reinterpret_cast<const std::vector<std::string>*>(serialized_varlen_buffer_handle);
  if (!serialized_varlen_buffer.empty()) {
    const auto rhs_proj_col = *reinterpret_cast<const int64_t*>(that_ptr1);
    CHECK_LT(static_cast<size_t>(rhs_proj_col), serialized_varlen_buffer.size());
    const auto& varlen_bytes_str = serialized_varlen_buffer[rhs_proj_col];
    const auto str_ptr = reinterpret_cast<const int8_t*>(varlen_bytes_str.c_str());
    *reinterpret_cast<int64_t*>(this_ptr1) = reinterpret_cast<const int64_t>(str_ptr);
    *reinterpret_cast<int64_t*>(this_ptr2) =
        static_cast<int64_t>(varlen_bytes_str.size() / length_to_elems);
  } else {
    varlen_buffer_sample(this_ptr1, this_ptr2, that_ptr1, that_ptr2, init_val);
  }
}

void ResultSetStorage::reduceOneSlotJIT(llvm::Value* this_ptr1,
                                        llvm::Value* this_ptr2,
                                        llvm::Value* that_ptr1,
                                        llvm::Value* that_ptr2,
                                        const TargetInfo& target_info,
                                        const size_t target_logical_idx,
                                        const size_t target_slot_idx,
                                        const size_t init_agg_val_idx,
                                        const ResultSetStorage& that,
                                        const size_t first_slot_idx_for_target,
                                        const ReductionCode& reduction_code) const {
  if (query_mem_desc_.targetGroupbyIndicesSize() > 0) {
    if (query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) >= 0) {
      return;
    }
  }
  CHECK_LT(init_agg_val_idx, target_init_vals_.size());
  const bool float_argument_input = takes_float_argument(target_info);
  const auto chosen_bytes =
      get_width_for_slot(target_slot_idx, float_argument_input, query_mem_desc_);
  auto init_val = target_init_vals_[init_agg_val_idx];
  const auto cgen_state = reduction_code.cgen_state.get();
  if (target_info.is_agg && target_info.agg_kind != kSAMPLE) {
    switch (target_info.agg_kind) {
      case kCOUNT:
      case kAPPROX_COUNT_DISTINCT: {
        if (is_distinct_target(target_info)) {
          CHECK_EQ(static_cast<size_t>(chosen_bytes), sizeof(int64_t));
          reduceOneCountDistinctSlotJIT(
              this_ptr1, that_ptr1, target_logical_idx, that, reduction_code);
          break;
        }
        CHECK_EQ(int64_t(0), init_val);
        emit_aggregate_one_count(this_ptr1, that_ptr1, chosen_bytes, cgen_state);
        break;
      }
      case kAVG: {
        // Ignore float argument compaction for count component for fear of its overflow
        emit_aggregate_one_count(this_ptr2,
                                 that_ptr2,
                                 query_mem_desc_.getPaddedSlotWidthBytes(target_slot_idx),
                                 cgen_state);
      }
      // fall thru
      case kSUM: {
        emit_aggregate_one_nullable_value(
            "sum", this_ptr1, that_ptr1, init_val, chosen_bytes, target_info, cgen_state);
        break;
      }
      case kMIN: {
        emit_aggregate_one_nullable_value(
            "min", this_ptr1, that_ptr1, init_val, chosen_bytes, target_info, cgen_state);
        break;
      }
      case kMAX: {
        emit_aggregate_one_nullable_value(
            "max", this_ptr1, that_ptr1, init_val, chosen_bytes, target_info, cgen_state);
        break;
      }
      default:
        LOG(FATAL) << "Invalid aggregate type";
    }
  } else {
    emit_write_projection(this_ptr1, that_ptr1, init_val, chosen_bytes, cgen_state);
    if (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()) {
      CHECK(this_ptr2 && that_ptr2);
      size_t length_to_elems{0};
      if (target_info.sql_type.is_geometry()) {
        // TODO: Assumes hard-coded sizes for geometry targets
        length_to_elems = target_slot_idx == first_slot_idx_for_target ? 1 : 4;
      } else {
        const auto& elem_ti = target_info.sql_type.get_elem_type();
        length_to_elems = target_info.sql_type.is_string() ? 1 : elem_ti.get_size();
      }
      const auto arg_it = reduction_code.ir_reduce_func->arg_begin();
      const auto serialized_varlen_buffer_arg = &*(arg_it + 4);
      cgen_state->emitExternalCall("serialized_varlen_buffer_sample",
                                   llvm::Type::getVoidTy(cgen_state->context_),
                                   {serialized_varlen_buffer_arg,
                                    this_ptr1,
                                    this_ptr2,
                                    that_ptr1,
                                    that_ptr2,
                                    cgen_state->llInt<int64_t>(init_val),
                                    cgen_state->llInt<int64_t>(length_to_elems)});
    }
  }
}

extern "C" void count_distinct_set_union_jit_rt(const int64_t new_set_handle,
                                                const int64_t old_set_handle,
                                                const void* that_qmd_handle,
                                                const void* this_qmd_handle,
                                                const int64_t target_logical_idx) {
  const auto that_qmd = reinterpret_cast<const QueryMemoryDescriptor*>(that_qmd_handle);
  const auto this_qmd = reinterpret_cast<const QueryMemoryDescriptor*>(this_qmd_handle);
  const auto& new_count_distinct_desc =
      that_qmd->getCountDistinctDescriptor(target_logical_idx);
  const auto& old_count_distinct_desc =
      this_qmd->getCountDistinctDescriptor(target_logical_idx);
  CHECK(old_count_distinct_desc.impl_type_ != CountDistinctImplType::Invalid);
  CHECK(old_count_distinct_desc.impl_type_ == new_count_distinct_desc.impl_type_);
  count_distinct_set_union(
      new_set_handle, old_set_handle, new_count_distinct_desc, old_count_distinct_desc);
}

void ResultSetStorage::reduceOneCountDistinctSlotJIT(
    llvm::Value* this_ptr1,
    llvm::Value* that_ptr1,
    const size_t target_logical_idx,
    const ResultSetStorage& that,
    const ReductionCode& reduction_code) const {
  CHECK_LT(target_logical_idx, query_mem_desc_.getCountDistinctDescriptorsSize());
  const auto cgen_state = reduction_code.cgen_state.get();
  const auto old_set_handle = emit_load_i64(this_ptr1, cgen_state);
  const auto new_set_handle = emit_load_i64(that_ptr1, cgen_state);
  const auto arg_it = reduction_code.ir_reduce_func->arg_begin();
  const auto this_qmd_arg = &*(arg_it + 2);
  const auto that_qmd_arg = &*(arg_it + 3);
  cgen_state->emitExternalCall("count_distinct_set_union_jit_rt",
                               llvm::Type::getVoidTy(cgen_state->context_),
                               {new_set_handle,
                                old_set_handle,
                                that_qmd_arg,
                                this_qmd_arg,
                                cgen_state->llInt<int64_t>(target_logical_idx)});
}
