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

#include "ResultSetReductionJIT.h"

#include "CodeGenerator.h"
#include "DynamicWatchdog.h"
#include "Execute.h"
#include "IRCodegenUtils.h"
#include "LLVMFunctionAttributesUtil.h"

#include "Shared/likely.h"
#include "Shared/mapdpath.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

extern std::unique_ptr<llvm::Module> g_rt_module;

CodeCache ResultSetReductionJIT::s_code_cache(10000);

namespace {

// Error code to be returned when the watchdog timer triggers during the reduction.
const int32_t WATCHDOG_ERROR{-1};
// Use the LLVM interpreter, not the JIT, for a number of entries lower than the
// threshold.
const size_t INTERP_THRESHOLD{0};

// Make a shallow copy (just declarations) of the runtime module. Function definitions are
// cloned only if they're used from the generated code.
std::unique_ptr<llvm::Module> runtime_module_shallow_copy(CgenState* cgen_state) {
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

// Load the value stored at 'ptr' as the given 'loaded_type'.
llvm::Value* emit_load(llvm::Value* ptr, llvm::Type* loaded_type, CgenState* cgen_state) {
  return cgen_state->ir_builder_.CreateLoad(
      cgen_state->ir_builder_.CreateBitCast(ptr, loaded_type),
      ptr->getName() + "_loaded");
}

// Load the value stored at 'ptr' as a 32-bit signed integer.
llvm::Value* emit_load_i32(llvm::Value* ptr, CgenState* cgen_state) {
  const auto pi32_type = llvm::Type::getInt32PtrTy(cgen_state->context_);
  return emit_load(ptr, pi32_type, cgen_state);
}

// Load the value stored at 'ptr' as a 64-bit signed integer.
llvm::Value* emit_load_i64(llvm::Value* ptr, CgenState* cgen_state) {
  const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
  return emit_load(ptr, pi64_type, cgen_state);
}

// Read a 32- or 64-bit integer stored at 'ptr' and sign extend to 64-bit.
llvm::Value* emit_read_int_from_buff(llvm::Value* ptr,
                                     const int8_t compact_sz,
                                     CgenState* cgen_state) {
  switch (compact_sz) {
    case 8: {
      return emit_load_i64(ptr, cgen_state);
    }
    case 4: {
      const auto loaded_val = emit_load_i32(ptr, cgen_state);
      auto& ctx = cgen_state->context_;
      const auto i64_type = get_int_type(64, ctx);
      return cgen_state->ir_builder_.CreateSExt(loaded_val, i64_type);
    }
    default: {
      LOG(FATAL) << "Invalid byte width: " << compact_sz;
      return nullptr;
    }
  }
}

// Emit a runtime call to accumulate into the 'val_ptr' byte address the 'other_ptr'
// value when the type is specified as not null.
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
  const auto dest_name = agg_kind + "_dest";
  if (sql_type.is_fp()) {
    if (chosen_bytes == sizeof(float)) {
      const auto agg =
          cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type, dest_name);
      const auto val = emit_load(other_ptr, pf32_type, cgen_state);
      cgen_state->emitCall("agg_" + agg_kind + "_float", {agg, val});
    } else {
      CHECK_EQ(chosen_bytes, sizeof(double));
      const auto agg =
          cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type, dest_name);
      const auto val = emit_load(other_ptr, pf64_type, cgen_state);
      cgen_state->emitCall("agg_" + agg_kind + "_double", {agg, val});
    }
  } else {
    if (chosen_bytes == sizeof(int32_t)) {
      const auto agg =
          cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type, dest_name);
      const auto val = emit_load(other_ptr, pi32_type, cgen_state);
      cgen_state->emitCall("agg_" + agg_kind + "_int32", {agg, val});
    } else {
      CHECK_EQ(chosen_bytes, sizeof(int64_t));
      const auto agg =
          cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type, dest_name);
      const auto val = emit_load(other_ptr, pi64_type, cgen_state);
      cgen_state->emitCall("agg_" + agg_kind, {agg, val});
    }
  }
}

// Same as above, but support nullable types as well.
void emit_aggregate_one_nullable_value(const std::string& agg_kind,
                                       llvm::Value* val_ptr,
                                       llvm::Value* other_ptr,
                                       const int64_t init_val,
                                       const size_t chosen_bytes,
                                       const TargetInfo& agg_info,
                                       CgenState* cgen_state) {
  const auto dest_name = agg_kind + "_dest";
  if (agg_info.skip_null_val) {
    const auto pi32_type = llvm::Type::getInt32PtrTy(cgen_state->context_);
    const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
    const auto pf32_type = llvm::Type::getFloatPtrTy(cgen_state->context_);
    const auto pf64_type = llvm::Type::getDoublePtrTy(cgen_state->context_);
    const auto sql_type = get_compact_type(agg_info);
    if (sql_type.is_fp()) {
      if (chosen_bytes == sizeof(float)) {
        const auto agg =
            cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type, dest_name);
        const auto val = emit_load(other_ptr, pf32_type, cgen_state);
        const auto init_val_lv =
            cgen_state->llFp(*reinterpret_cast<const float*>(may_alias_ptr(&init_val)));
        cgen_state->emitCall("agg_" + agg_kind + "_float_skip_val",
                             {agg, val, init_val_lv});
      } else {
        CHECK_EQ(chosen_bytes, sizeof(double));
        const auto agg =
            cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type, dest_name);
        const auto val = emit_load(other_ptr, pf64_type, cgen_state);
        const auto init_val_lv =
            cgen_state->llFp(*reinterpret_cast<const double*>(may_alias_ptr(&init_val)));
        cgen_state->emitCall("agg_" + agg_kind + "_double_skip_val",
                             {agg, val, init_val_lv});
      }
    } else {
      if (chosen_bytes == sizeof(int32_t)) {
        const auto agg =
            cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type, dest_name);
        const auto val = emit_load(other_ptr, pi32_type, cgen_state);
        const auto init_val_lv = cgen_state->llInt<int32_t>(init_val);
        cgen_state->emitCall("agg_" + agg_kind + "_int32_skip_val",
                             {agg, val, init_val_lv});
      } else {
        CHECK_EQ(chosen_bytes, sizeof(int64_t));
        const auto agg =
            cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type, dest_name);
        const auto val = emit_load(other_ptr, pi64_type, cgen_state);
        const auto init_val_lv = cgen_state->llInt<int64_t>(init_val);
        cgen_state->emitCall("agg_" + agg_kind + "_skip_val", {agg, val, init_val_lv});
      }
    }
  } else {
    emit_aggregate_one_value(
        agg_kind, val_ptr, other_ptr, chosen_bytes, agg_info, cgen_state);
  }
}

// Emit code to accumulate the 'other_ptr' count into the 'val_ptr' destination.
void emit_aggregate_one_count(llvm::Value* val_ptr,
                              llvm::Value* other_ptr,
                              const size_t chosen_bytes,
                              CgenState* cgen_state) {
  const auto dest_name = "count_dest";
  if (chosen_bytes == sizeof(int32_t)) {
    const auto pi32_type = llvm::Type::getInt32PtrTy(cgen_state->context_);
    const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type, dest_name);
    const auto val = emit_load(other_ptr, pi32_type, cgen_state);
    cgen_state->emitCall("agg_sum_int32", {agg, val});
  } else {
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
    const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type, dest_name);
    const auto val = emit_load(other_ptr, pi64_type, cgen_state);
    cgen_state->emitCall("agg_sum", {agg, val});
  }
}

// Emit code to load the value stored at the 'other_pi8' as an integer of the given width
// 'chosen_bytes' and write it to the 'slot_pi8' destination only if necessary (the
// existing value at destination is the initialization value).
void emit_write_projection(llvm::Value* slot_pi8,
                           llvm::Value* other_pi8,
                           const int64_t init_val,
                           const size_t chosen_bytes,
                           CgenState* cgen_state) {
  const auto func_name = "write_projection_int" + std::to_string(chosen_bytes * 8);
  if (chosen_bytes == sizeof(int32_t)) {
    const auto proj_val = emit_load_i32(other_pi8, cgen_state);
    cgen_state->emitCall(func_name,
                         {slot_pi8, proj_val, cgen_state->llInt<int64_t>(init_val)});
  } else {
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto proj_val = emit_load_i64(other_pi8, cgen_state);
    cgen_state->emitCall(func_name,
                         {slot_pi8, proj_val, cgen_state->llInt<int64_t>(init_val)});
  }
}

// Create the declaration for the 'is_empty_entry' function. Use private linkage since
// it's a helper only called from the generated code and mark it as always inline.
llvm::Function* setup_is_empty_entry(const CgenState* cgen_state) {
  auto& ctx = cgen_state->context_;
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto func_type = llvm::FunctionType::get(get_int_type(1, ctx), {pi8_type}, false);
  auto func = llvm::Function::Create(
      func_type, llvm::Function::PrivateLinkage, "is_empty_entry", cgen_state->module_);
  const auto arg_it = func->arg_begin();
  const auto row_ptr_arg = &*arg_it;
  row_ptr_arg->setName("row_ptr");
  mark_function_always_inline(func);
  return func;
}

// Create the declaration for the 'reduce_one_entry' helper.
llvm::Function* setup_reduce_one_entry(const CgenState* cgen_state,
                                       const QueryDescriptionType hash_type) {
  auto& ctx = cgen_state->context_;
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto pvoid_type = llvm::PointerType::get(llvm::Type::getVoidTy(ctx), 0);
  const auto func_type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                              {pi8_type, pi8_type, pvoid_type, pvoid_type, pvoid_type},
                              false);
  const auto func = llvm::Function::Create(
      func_type, llvm::Function::PrivateLinkage, "reduce_one_entry", cgen_state->module_);
  const auto arg_it = func->arg_begin();
  switch (hash_type) {
    case QueryDescriptionType::GroupByBaselineHash: {
      const auto this_targets_ptr_arg = &*arg_it;
      const auto that_targets_ptr_arg = &*(arg_it + 1);
      this_targets_ptr_arg->setName("this_targets_ptr");
      that_targets_ptr_arg->setName("that_targets_ptr");
      break;
    }
    case QueryDescriptionType::GroupByPerfectHash: {
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
  mark_function_always_inline(func);
  return func;
}

// Create the declaration for the 'reduce_one_entry_idx' helper.
llvm::Function* setup_reduce_one_entry_idx(const CgenState* cgen_state) {
  auto& ctx = cgen_state->context_;
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto i32_type = get_int_type(32, ctx);
  const auto pvoid_type = llvm::PointerType::get(llvm::Type::getVoidTy(ctx), 0);
  const auto func_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx),
      {pi8_type, pi8_type, i32_type, i32_type, pvoid_type, pvoid_type, pvoid_type},
      false);
  auto func = llvm::Function::Create(func_type,
                                     llvm::Function::PrivateLinkage,
                                     "reduce_one_entry_idx",
                                     cgen_state->module_);
  const auto arg_it = func->arg_begin();
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
  mark_function_always_inline(func);
  return func;
}

// Create the declaration for the 'reduce_loop' entry point. Use external linkage, this is
// the public API of the generated code directly used from result set reduction.
llvm::Function* setup_reduce_loop(const CgenState* cgen_state) {
  auto& ctx = cgen_state->context_;
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto i32_type = get_int_type(32, ctx);
  const auto pvoid_type = llvm::PointerType::get(llvm::Type::getVoidTy(ctx), 0);
  const auto func_type = llvm::FunctionType::get(i32_type,
                                                 {pi8_type,
                                                  pi8_type,
                                                  i32_type,
                                                  i32_type,
                                                  i32_type,
                                                  pvoid_type,
                                                  pvoid_type,
                                                  pvoid_type},
                                                 false);
  auto func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "reduce_loop", cgen_state->module_);
  const auto arg_it = func->arg_begin();
  const auto this_buff_arg = &*arg_it;
  const auto that_buff_arg = &*(arg_it + 1);
  const auto start_index_arg = &*(arg_it + 2);
  const auto end_index_arg = &*(arg_it + 3);
  const auto that_entry_count_arg = &*(arg_it + 4);
  const auto this_qmd_handle_arg = &*(arg_it + 5);
  const auto that_qmd_handle_arg = &*(arg_it + 6);
  const auto serialized_varlen_buffer_arg = &*(arg_it + 7);
  this_buff_arg->setName("this_buff");
  that_buff_arg->setName("that_buff");
  start_index_arg->setName("start_index");
  end_index_arg->setName("end_index");
  that_entry_count_arg->setName("that_entry_count");
  this_qmd_handle_arg->setName("this_qmd_handle");
  that_qmd_handle_arg->setName("that_qmd_handle");
  serialized_varlen_buffer_arg->setName("serialized_varlen_buffer");
  return func;
}

// Setup the reduction function and helpers declarations, create a module and a code
// generation state object.
ReductionCode setup_functions_ir(const QueryDescriptionType hash_type) {
  ReductionCode reduction_code{};
  reduction_code.cgen_state.reset(new CgenState({}, false));
  auto cgen_state = reduction_code.cgen_state.get();
  std::unique_ptr<llvm::Module> module(runtime_module_shallow_copy(cgen_state));
  cgen_state->module_ = module.get();
  reduction_code.ir_is_empty = setup_is_empty_entry(cgen_state);
  reduction_code.ir_reduce_one_entry = setup_reduce_one_entry(cgen_state, hash_type);
  reduction_code.ir_reduce_one_entry_idx = setup_reduce_one_entry_idx(cgen_state);
  reduction_code.ir_reduce_loop = setup_reduce_loop(cgen_state);
  reduction_code.module = std::move(module);
  return reduction_code;
}

// When the number of entries is below 'INTERP_THRESHOLD', run the generated function in
// its IR form, without compiling to native code.
ExecutionEngineWrapper create_interpreter_engine(llvm::Function* func) {
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

bool is_group_query(const QueryDescriptionType hash_type) {
  return hash_type == QueryDescriptionType::GroupByBaselineHash ||
         hash_type == QueryDescriptionType::GroupByPerfectHash;
}

// Emit an early return from a function when the provided 'cond' is true, which the caller
// code can use when entries are empty or the watchdog is triggered. For functions which
// return void, the specified error code is ignored. For functions which return an
// integer, the error code is returned.
void return_early(llvm::Value* cond,
                  const ReductionCode& reduction_code,
                  llvm::Function* func,
                  int error_code) {
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto early_return = llvm::BasicBlock::Create(ctx, ".early_return", func, 0);
  const auto do_reduction = llvm::BasicBlock::Create(ctx, ".do_reduction", func, 0);
  cgen_state->ir_builder_.CreateCondBr(cond, early_return, do_reduction);
  cgen_state->ir_builder_.SetInsertPoint(early_return);
  if (func->getReturnType()->isVoidTy()) {
    cgen_state->ir_builder_.CreateRetVoid();
  } else {
    cgen_state->ir_builder_.CreateRet(cgen_state->llInt<int32_t>(error_code));
  }
  cgen_state->ir_builder_.SetInsertPoint(do_reduction);
}

// Variable length sample fast path (no serialized variable length buffer).
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

// Wrappers to be called from the generated code, sharing implementation with the rest of
// the system.

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

extern "C" uint8_t check_watchdog_rt(const size_t sample_seed) {
  if (UNLIKELY(g_enable_dynamic_watchdog && (sample_seed & 0x3F) == 0 &&
               dynamic_watchdog())) {
    return true;
  }
  return false;
}

ResultSetReductionJIT::ResultSetReductionJIT(const QueryMemoryDescriptor& query_mem_desc,
                                             const std::vector<TargetInfo>& targets,
                                             const std::vector<int64_t>& target_init_vals)
    : query_mem_desc_(query_mem_desc)
    , targets_(targets)
    , target_init_vals_(target_init_vals) {}

// The code generated for a reduction between two result set buffers is structured in
// several functions and their IR is stored in the 'ReductionCode' structure. At a high
// level, the pseudocode is:
//
// func is_empty_func(row_ptr):
//   ...
//
// func reduce_func_baseline(this_ptr, that_ptr):
//   if is_empty_func(that_ptr):
//     return
//   for each target in the row:
//     reduce target from that_ptr into this_ptr
//
// func reduce_func_perfect_hash(this_ptr, that_ptr):
//   if is_empty_func(that_ptr):
//     return
//   for each target in the row:
//     reduce target from that_ptr into this_ptr
//
// func reduce_func_idx(this_buff, that_buff, that_entry_index):
//   that_ptr = that_result_set[that_entry_index]
//   # Retrieval of 'this_ptr' is different between perfect hash and baseline.
//   this_ptr = this_result_set[that_entry_index]
//                or
//              get_row(key(that_row_ptr), this_result_set_buffer)
//   reduce_func_[baseline|perfect_hash](this_ptr, that_ptr)
//
// func reduce_loop(this_buff, that_buff, start_entry_index, end_entry_index):
//   for that_entry_index in [start_entry_index, end_entry_index):
//     reduce_func_idx(this_buff, that_buff, that_entry_index)

ReductionCode ResultSetReductionJIT::codegen() const {
  static std::mutex s_codegen_mutex;
  std::lock_guard<std::mutex> s_codegen_guard(s_codegen_mutex);
  const auto hash_type = query_mem_desc_.getQueryDescriptionType();
  if (query_mem_desc_.didOutputColumnar() || !is_group_query(hash_type)) {
    return {};
  }
  ReductionCode reduction_code = setup_functions_ir(hash_type);
  isEmpty(reduction_code);
  switch (query_mem_desc_.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByPerfectHash: {
      reduceOneEntryNoCollisions(reduction_code);
      reduceOneEntryNoCollisionsIdx(reduction_code);
      break;
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      reduceOneEntryBaseline(reduction_code);
      reduceOneEntryBaselineIdx(reduction_code);
      break;
    }
    default: {
      LOG(FATAL) << "Unexpected query description type";
    }
  }
  reduceLoop(reduction_code);
  return finalizeReductionCode(std::move(reduction_code));
}

void ResultSetReductionJIT::isEmpty(const ReductionCode& reduction_code) const {
  CHECK(is_group_query(query_mem_desc_.getQueryDescriptionType()));
  CHECK(!query_mem_desc_.didOutputColumnar());
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", reduction_code.ir_is_empty, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  llvm::Value* key{nullptr};
  llvm::Value* empty_key_val{nullptr};
  const auto arg_it = reduction_code.ir_is_empty->arg_begin();
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
        keys_ptr, cgen_state->llInt<int32_t>(target_slot_off), "is_empty_slot_ptr");
    const auto compact_sz =
        query_mem_desc_.getPaddedSlotWidthBytes(query_mem_desc_.getTargetIdxForKey());
    key = emit_read_int_from_buff(slot_ptr, compact_sz, cgen_state);
    empty_key_val = cgen_state->llInt<int64_t>(
        target_init_vals_[query_mem_desc_.getTargetIdxForKey()]);
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
  const auto ret =
      cgen_state->ir_builder_.CreateICmpEQ(key, empty_key_val, "is_key_empty");
  cgen_state->ir_builder_.CreateRet(ret);
  verify_function_ir(reduction_code.ir_is_empty);
}

void ResultSetReductionJIT::reduceOneEntryNoCollisions(
    const ReductionCode& reduction_code) const {
  auto cgen_state = reduction_code.cgen_state.get();
  const auto bb_entry = llvm::BasicBlock::Create(
      cgen_state->context_, ".entry", reduction_code.ir_reduce_one_entry, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  const auto arg_it = reduction_code.ir_reduce_one_entry->arg_begin();
  const auto this_row_ptr = &*arg_it;
  const auto that_row_ptr = &*(arg_it + 1);
  const auto that_is_empty = cgen_state->ir_builder_.CreateCall(
      reduction_code.ir_is_empty, that_row_ptr, "that_is_empty");
  return_early(that_is_empty, reduction_code, reduction_code.ir_reduce_one_entry, 0);

  const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
  if (key_bytes) {  // copy the key from right hand side
    cgen_state->ir_builder_.CreateMemCpy(
        this_row_ptr, 0, that_row_ptr, 0, cgen_state->llInt<int32_t>(key_bytes));
  }

  const auto key_bytes_with_padding = align_to_int64(key_bytes);
  const auto key_bytes_lv = cgen_state->llInt<int32_t>(key_bytes_with_padding);
  const auto this_targets_start_ptr =
      cgen_state->ir_builder_.CreateGEP(this_row_ptr, key_bytes_lv, "this_targets_start");
  const auto that_targets_start_ptr =
      cgen_state->ir_builder_.CreateGEP(that_row_ptr, key_bytes_lv, "that_targets_start");

  reduceOneEntryTargetsNoCollisions(
      reduction_code, this_targets_start_ptr, that_targets_start_ptr);
  verify_function_ir(reduction_code.ir_reduce_one_entry);
}

void ResultSetReductionJIT::reduceOneEntryTargetsNoCollisions(
    const ReductionCode& reduction_code,
    llvm::Value* this_targets_start_ptr,
    llvm::Value* that_targets_start_ptr) const {
  auto cgen_state = reduction_code.cgen_state.get();
  const auto& col_slot_context = query_mem_desc_.getColSlotContext();
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
        const auto desc = "target_" + std::to_string(target_logical_idx) + "_second_slot";
        this_ptr2 =
            cgen_state->ir_builder_.CreateGEP(this_targets_ptr, slot_off, "this_" + desc);
        that_ptr2 =
            cgen_state->ir_builder_.CreateGEP(that_targets_ptr, slot_off, "that_" + desc);
      }
      reduceOneSlot(this_targets_ptr,
                    this_ptr2,
                    that_targets_ptr,
                    that_ptr2,
                    target_info,
                    target_logical_idx,
                    target_slot_idx,
                    init_agg_val_idx,
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
      const auto next_desc =
          "target_" + std::to_string(target_logical_idx + 1) + "_first_slot";
      if (UNLIKELY(two_slot_target)) {
        increment_agg_val_idx_maybe(2);
        const auto two_slot_off = cgen_state->llInt<int32_t>(
            slot_off_val + query_mem_desc_.getPaddedSlotWidthBytes(target_slot_idx + 1));
        this_targets_ptr = cgen_state->ir_builder_.CreateGEP(
            this_targets_ptr, two_slot_off, "this_" + next_desc);
        that_targets_ptr = cgen_state->ir_builder_.CreateGEP(
            that_targets_ptr, two_slot_off, "that_" + next_desc);
      } else {
        increment_agg_val_idx_maybe(1);
        this_targets_ptr = cgen_state->ir_builder_.CreateGEP(
            this_targets_ptr, slot_off, "this_" + next_desc);
        that_targets_ptr = cgen_state->ir_builder_.CreateGEP(
            that_targets_ptr, slot_off, "that_" + next_desc);
      }
    }
  }
  reduction_code.cgen_state->ir_builder_.CreateRetVoid();
  verify_function_ir(reduction_code.ir_reduce_one_entry);
}

void ResultSetReductionJIT::reduceOneEntryBaseline(
    const ReductionCode& reduction_code) const {
  auto cgen_state = reduction_code.cgen_state.get();
  const auto bb_entry = llvm::BasicBlock::Create(
      cgen_state->context_, ".entry", reduction_code.ir_reduce_one_entry, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  const auto arg_it = reduction_code.ir_reduce_one_entry->arg_begin();
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
      const auto desc = "target_" + std::to_string(target_logical_idx) + "_second_slot";
      const auto second_slot_rel_off = cgen_state->llInt<int32_t>(sizeof(int64_t));
      this_ptr2 = cgen_state->ir_builder_.CreateGEP(
          this_ptr1, second_slot_rel_off, "this_" + desc);
      that_ptr2 = cgen_state->ir_builder_.CreateGEP(
          that_ptr1, second_slot_rel_off, "that_" + desc);
    }
    reduceOneSlot(this_ptr1,
                  this_ptr2,
                  that_ptr1,
                  that_ptr2,
                  target_info,
                  target_logical_idx,
                  j,
                  init_agg_val_idx,
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
    const auto next_desc =
        "target_" + std::to_string(target_logical_idx + 1) + "_first_slot";
    auto next_slot_rel_off =
        cgen_state->llInt<int32_t>(init_agg_val_idx * sizeof(int64_t));
    this_ptr1 = cgen_state->ir_builder_.CreateGEP(
        this_targets_ptr_arg, next_slot_rel_off, next_desc);
    that_ptr1 = cgen_state->ir_builder_.CreateGEP(
        that_targets_ptr_arg, next_slot_rel_off, next_desc);
  }
  reduction_code.cgen_state->ir_builder_.CreateRetVoid();
  verify_function_ir(reduction_code.ir_reduce_one_entry);
}

void ResultSetReductionJIT::reduceOneEntryNoCollisionsIdx(
    const ReductionCode& reduction_code) const {
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash);
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", reduction_code.ir_reduce_one_entry_idx, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  const auto arg_it = reduction_code.ir_reduce_one_entry_idx->arg_begin();
  const auto this_buff = &*arg_it;
  const auto that_buff = &*(arg_it + 1);
  const auto entry_idx = &*(arg_it + 2);
  const auto this_qmd_handle = &*(arg_it + 4);
  const auto that_qmd_handle = &*(arg_it + 5);
  const auto serialized_varlen_buffer_arg = &*(arg_it + 6);
  const auto row_bytes = cgen_state->llInt<int32_t>(get_row_bytes(query_mem_desc_));
  const auto row_off_in_bytes =
      cgen_state->ir_builder_.CreateMul(entry_idx, row_bytes, "row_off_in_bytes");
  const auto this_row_ptr =
      cgen_state->ir_builder_.CreateGEP(this_buff, row_off_in_bytes, "this_row_ptr");
  const auto that_row_ptr =
      cgen_state->ir_builder_.CreateGEP(that_buff, row_off_in_bytes, "that_row_ptr");
  cgen_state->ir_builder_.CreateCall(reduction_code.ir_reduce_one_entry,
                                     {this_row_ptr,
                                      that_row_ptr,
                                      this_qmd_handle,
                                      that_qmd_handle,
                                      serialized_varlen_buffer_arg});
  cgen_state->ir_builder_.CreateRetVoid();
  verify_function_ir(reduction_code.ir_reduce_one_entry_idx);
}

void ResultSetReductionJIT::reduceOneEntryBaselineIdx(
    const ReductionCode& reduction_code) const {
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByBaselineHash);
  CHECK(!query_mem_desc_.hasKeylessHash());
  CHECK(!query_mem_desc_.didOutputColumnar());
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", reduction_code.ir_reduce_one_entry_idx, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  const auto arg_it = reduction_code.ir_reduce_one_entry_idx->arg_begin();
  const auto this_buff = &*arg_it;
  const auto that_buff = &*(arg_it + 1);
  const auto that_entry_idx = &*(arg_it + 2);
  const auto that_entry_count = &*(arg_it + 3);
  const auto this_qmd_handle = &*(arg_it + 4);
  const auto that_qmd_handle = &*(arg_it + 5);
  const auto serialized_varlen_buffer_arg = &*(arg_it + 6);
  const auto row_bytes = cgen_state->llInt<int32_t>(get_row_bytes(query_mem_desc_));
  const auto that_row_off_in_bytes = cgen_state->ir_builder_.CreateMul(
      that_entry_idx, row_bytes, "that_row_off_in_bytes");
  const auto that_row_ptr =
      cgen_state->ir_builder_.CreateGEP(that_buff, that_row_off_in_bytes, "that_row_ptr");
  const auto that_is_empty = cgen_state->ir_builder_.CreateCall(
      reduction_code.ir_is_empty, that_row_ptr, "that_is_empty");
  return_early(that_is_empty, reduction_code, reduction_code.ir_reduce_one_entry_idx, 0);
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
  const auto bool_type = llvm::Type::getInt8Ty(cgen_state->context_);
  const auto this_targets_ptr_i64_ptr = cgen_state->ir_builder_.CreateAlloca(
      pi64_type, cgen_state->llInt<int32_t>(1), "this_targets_ptr_out");
  const auto this_is_empty_ptr = cgen_state->ir_builder_.CreateAlloca(
      bool_type, cgen_state->llInt<int32_t>(1), "this_is_empty_out");
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
  const auto this_targets_ptr_i64 = cgen_state->ir_builder_.CreateLoad(
      this_targets_ptr_i64_ptr, "this_targets_ptr_i64");
  llvm::Value* this_is_empty =
      cgen_state->ir_builder_.CreateLoad(this_is_empty_ptr, "this_is_empty");
  this_is_empty = cgen_state->ir_builder_.CreateTrunc(
      this_is_empty, get_int_type(1, ctx), "this_is_empty_bool");
  return_early(this_is_empty, reduction_code, reduction_code.ir_reduce_one_entry_idx, 0);
  const auto pi8_type = llvm::Type::getInt8PtrTy(cgen_state->context_);
  const auto key_qw_count = get_slot_off_quad(query_mem_desc_);
  const auto this_targets_ptr = cgen_state->ir_builder_.CreateBitCast(
      this_targets_ptr_i64, pi8_type, "this_targets_ptr");
  const auto key_byte_count = key_qw_count * sizeof(int64_t);
  const auto key_byte_count_lv = cgen_state->llInt<int32_t>(key_byte_count);
  const auto that_targets_ptr = cgen_state->ir_builder_.CreateGEP(
      that_row_ptr, key_byte_count_lv, "that_targets_ptr");
  cgen_state->ir_builder_.CreateCall(reduction_code.ir_reduce_one_entry,
                                     {this_targets_ptr,
                                      that_targets_ptr,
                                      this_qmd_handle,
                                      that_qmd_handle,
                                      serialized_varlen_buffer_arg});
  cgen_state->ir_builder_.CreateRetVoid();
  verify_function_ir(reduction_code.ir_reduce_one_entry_idx);
}

namespace {

llvm::BasicBlock* generate_loop_body(const ReductionCode& reduction_code,
                                     llvm::Value* this_buff,
                                     llvm::Value* that_buff,
                                     llvm::Value* iterator,
                                     llvm::Value* start_index,
                                     llvm::Value* that_entry_count,
                                     llvm::Value* this_qmd_handle,
                                     llvm::Value* that_qmd_handle,
                                     llvm::Value* serialized_varlen_buffer) {
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ir_builder = cgen_state->ir_builder_;
  auto& ctx = cgen_state->context_;
  const auto loop_body_bb = llvm::BasicBlock::Create(
      ctx, ".loop_body", ir_builder.GetInsertBlock()->getParent());
  ir_builder.SetInsertPoint(loop_body_bb);
  const auto loop_iter =
      ir_builder.CreateTrunc(iterator, get_int_type(32, ctx), "relative_entry_idx");
  const auto that_entry_idx =
      ir_builder.CreateAdd(loop_iter, start_index, "that_entry_idx");
  const auto watchdog_sample_seed =
      ir_builder.CreateSExt(that_entry_idx, get_int_type(64, ctx));
  const auto watchdog_triggered = cgen_state->emitExternalCall(
      "check_watchdog_rt", get_int_type(8, ctx), {watchdog_sample_seed});
  const auto watchdog_triggered_bool = cgen_state->ir_builder_.CreateICmpNE(
      watchdog_triggered, cgen_state->llInt<int8_t>(0));
  return_early(watchdog_triggered_bool,
               reduction_code,
               reduction_code.ir_reduce_loop,
               WATCHDOG_ERROR);
  ir_builder.CreateCall(reduction_code.ir_reduce_one_entry_idx,
                        {this_buff,
                         that_buff,
                         that_entry_idx,
                         that_entry_count,
                         this_qmd_handle,
                         that_qmd_handle,
                         serialized_varlen_buffer});
  return loop_body_bb;
}

}  // namespace

void ResultSetReductionJIT::reduceLoop(const ReductionCode& reduction_code) const {
  const auto arg_it = reduction_code.ir_reduce_loop->arg_begin();
  const auto this_buff_arg = &*arg_it;
  const auto that_buff_arg = &*(arg_it + 1);
  const auto start_index_arg = &*(arg_it + 2);
  const auto end_index_arg = &*(arg_it + 3);
  const auto that_entry_count_arg = &*(arg_it + 4);
  const auto this_qmd_handle_arg = &*(arg_it + 5);
  const auto that_qmd_handle_arg = &*(arg_it + 6);
  const auto serialized_varlen_buffer_arg = &*(arg_it + 7);
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  const auto bb_entry =
      llvm::BasicBlock::Create(ctx, ".entry", reduction_code.ir_reduce_loop, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  const auto i64_type = get_int_type(64, cgen_state->context_);
  const auto iteration_count = cgen_state->ir_builder_.CreateSub(
      end_index_arg, start_index_arg, "iteration_count");
  const auto upper_bound = cgen_state->ir_builder_.CreateSExt(iteration_count, i64_type);
  const auto bb_exit =
      llvm::BasicBlock::Create(ctx, ".exit", reduction_code.ir_reduce_loop);
  cgen_state->ir_builder_.SetInsertPoint(bb_exit);
  cgen_state->ir_builder_.CreateRet(cgen_state->llInt<int32_t>(0));
  JoinLoop join_loop(
      JoinLoopKind::UpperBound,
      JoinType::INNER,
      [upper_bound](const std::vector<llvm::Value*>& v) {
        JoinLoopDomain domain{0};
        domain.upper_bound = upper_bound;
        return domain;
      },
      nullptr,
      nullptr,
      nullptr,
      "reduction_loop");
  const auto bb_loop_body = JoinLoop::codegen(
      {join_loop},
      [&reduction_code,
       this_buff_arg,
       that_buff_arg,
       start_index_arg,
       that_entry_count_arg,
       this_qmd_handle_arg,
       that_qmd_handle_arg,
       serialized_varlen_buffer_arg](const std::vector<llvm::Value*>& iterators) {
        return generate_loop_body(reduction_code,
                                  this_buff_arg,
                                  that_buff_arg,
                                  iterators.back(),
                                  start_index_arg,
                                  that_entry_count_arg,
                                  this_qmd_handle_arg,
                                  that_qmd_handle_arg,
                                  serialized_varlen_buffer_arg);
      },
      nullptr,
      bb_exit,
      cgen_state->ir_builder_);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  cgen_state->ir_builder_.CreateBr(bb_loop_body);
  verify_function_ir(reduction_code.ir_reduce_loop);
}

void ResultSetReductionJIT::reduceOneSlot(llvm::Value* this_ptr1,
                                          llvm::Value* this_ptr2,
                                          llvm::Value* that_ptr1,
                                          llvm::Value* that_ptr2,
                                          const TargetInfo& target_info,
                                          const size_t target_logical_idx,
                                          const size_t target_slot_idx,
                                          const size_t init_agg_val_idx,
                                          const size_t first_slot_idx_for_target,
                                          const ReductionCode& reduction_code) const {
  if (query_mem_desc_.targetGroupbyIndicesSize() > 0) {
    if (query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) >= 0) {
      return;
    }
  }
  const bool float_argument_input = takes_float_argument(target_info);
  const auto chosen_bytes =
      get_width_for_slot(target_slot_idx, float_argument_input, query_mem_desc_);
  CHECK_LT(init_agg_val_idx, target_init_vals_.size());
  auto init_val = target_init_vals_[init_agg_val_idx];
  if (target_info.is_agg && target_info.agg_kind != kSAMPLE) {
    reduceOneAggregateSlot(this_ptr1,
                           this_ptr2,
                           that_ptr1,
                           that_ptr2,
                           target_info,
                           target_logical_idx,
                           target_slot_idx,
                           init_val,
                           chosen_bytes,
                           reduction_code);
  } else {
    const auto cgen_state = reduction_code.cgen_state.get();
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
      const auto arg_it = reduction_code.ir_reduce_one_entry->arg_begin();
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

void ResultSetReductionJIT::reduceOneAggregateSlot(
    llvm::Value* this_ptr1,
    llvm::Value* this_ptr2,
    llvm::Value* that_ptr1,
    llvm::Value* that_ptr2,
    const TargetInfo& target_info,
    const size_t target_logical_idx,
    const size_t target_slot_idx,
    const int64_t init_val,
    const int8_t chosen_bytes,
    const ReductionCode& reduction_code) const {
  const auto cgen_state = reduction_code.cgen_state.get();
  switch (target_info.agg_kind) {
    case kCOUNT:
    case kAPPROX_COUNT_DISTINCT: {
      if (is_distinct_target(target_info)) {
        CHECK_EQ(static_cast<size_t>(chosen_bytes), sizeof(int64_t));
        reduceOneCountDistinctSlot(
            this_ptr1, that_ptr1, target_logical_idx, reduction_code);
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
}

void ResultSetReductionJIT::reduceOneCountDistinctSlot(
    llvm::Value* this_ptr1,
    llvm::Value* that_ptr1,
    const size_t target_logical_idx,
    const ReductionCode& reduction_code) const {
  CHECK_LT(target_logical_idx, query_mem_desc_.getCountDistinctDescriptorsSize());
  const auto cgen_state = reduction_code.cgen_state.get();
  const auto old_set_handle = emit_load_i64(this_ptr1, cgen_state);
  const auto new_set_handle = emit_load_i64(that_ptr1, cgen_state);
  const auto arg_it = reduction_code.ir_reduce_one_entry->arg_begin();
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

ReductionCode ResultSetReductionJIT::finalizeReductionCode(
    ReductionCode reduction_code) const {
  const auto key0 = serialize_llvm_object(reduction_code.ir_is_empty);
  const auto key1 = serialize_llvm_object(reduction_code.ir_reduce_one_entry);
  const auto key2 = serialize_llvm_object(reduction_code.ir_reduce_one_entry_idx);
  CodeCacheKey key{key0, key1, key2};
  const auto val_ptr = s_code_cache.get(key);
  if (val_ptr) {
    return {
        nullptr,
        std::get<1>(val_ptr->first.front()).get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<ReductionCode::FuncPtr>(std::get<0>(val_ptr->first.front()))};
  }
  CompilationOptions co{
      ExecutorDeviceType::CPU, false, ExecutorOptLevel::ReductionJIT, false};
  reduction_code.module.release();
  const bool use_interp = query_mem_desc_.getEntryCount() < INTERP_THRESHOLD;
  auto ee = use_interp
                ? create_interpreter_engine(reduction_code.ir_reduce_loop)
                : CodeGenerator::generateNativeCPUCode(
                      reduction_code.ir_reduce_loop, {reduction_code.ir_reduce_loop}, co);
  reduction_code.func_ptr =
      use_interp ? nullptr
                 : reinterpret_cast<ReductionCode::FuncPtr>(
                       ee->getPointerToFunction(reduction_code.ir_reduce_loop));
  reduction_code.execution_engine = ee.get();
  if (use_interp) {
    reduction_code.own_execution_engine = std::move(ee);
  } else {
    std::tuple<void*, ExecutionEngineWrapper> cache_val =
        std::make_tuple(reinterpret_cast<void*>(reduction_code.func_ptr), std::move(ee));
    std::vector<std::tuple<void*, ExecutionEngineWrapper>> cache_vals;
    cache_vals.emplace_back(std::move(cache_val));
    Executor::addCodeToCache({key},
                             std::move(cache_vals),
                             reduction_code.ir_reduce_loop->getParent(),
                             s_code_cache);
  }
  return reduction_code;
}
