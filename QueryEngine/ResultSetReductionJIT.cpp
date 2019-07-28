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
      const auto val = cgen_state->ir_builder_.CreateLoad(
          cgen_state->ir_builder_.CreateBitCast(other_ptr, pf32_type));
      cgen_state->emitCall("agg_" + agg_kind + "_float", {agg, val});
    } else {
      CHECK_EQ(chosen_bytes, sizeof(double));
      const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
      const auto val = cgen_state->ir_builder_.CreateLoad(
          cgen_state->ir_builder_.CreateBitCast(other_ptr, pf64_type));
      cgen_state->emitCall("agg_" + agg_kind + "_double", {agg, val});
    }
  } else {
    if (chosen_bytes == sizeof(int32_t)) {
      const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type);
      const auto val = cgen_state->ir_builder_.CreateLoad(
          cgen_state->ir_builder_.CreateBitCast(other_ptr, pi32_type));
      cgen_state->emitCall("agg_" + agg_kind + "_int32", {agg, val});
    } else {
      CHECK_EQ(chosen_bytes, sizeof(int64_t));
      const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
      const auto val = cgen_state->ir_builder_.CreateLoad(
          cgen_state->ir_builder_.CreateBitCast(other_ptr, pi64_type));
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
        const auto val = cgen_state->ir_builder_.CreateLoad(
            cgen_state->ir_builder_.CreateBitCast(other_ptr, pf32_type));
        const auto init_val_lv =
            cgen_state->llFp(*reinterpret_cast<const float*>(may_alias_ptr(&init_val)));
        cgen_state->emitCall("agg_" + agg_kind + "_float_skip_val",
                             {agg, val, init_val_lv});
      } else {
        CHECK_EQ(chosen_bytes, sizeof(double));
        const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
        const auto val = cgen_state->ir_builder_.CreateLoad(
            cgen_state->ir_builder_.CreateBitCast(other_ptr, pf64_type));
        const auto init_val_lv =
            cgen_state->llFp(*reinterpret_cast<const double*>(may_alias_ptr(&init_val)));
        cgen_state->emitCall("agg_" + agg_kind + "_double_skip_val",
                             {agg, val, init_val_lv});
      }
    } else {
      if (chosen_bytes == sizeof(int32_t)) {
        const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi32_type);
        const auto val = cgen_state->ir_builder_.CreateLoad(
            cgen_state->ir_builder_.CreateBitCast(other_ptr, pi32_type));
        const auto init_val_lv = cgen_state->llInt<int32_t>(init_val);
        cgen_state->emitCall("agg_" + agg_kind + "_int32_skip_val",
                             {agg, val, init_val_lv});
      } else {
        CHECK_EQ(chosen_bytes, sizeof(int64_t));
        const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
        const auto val = cgen_state->ir_builder_.CreateLoad(
            cgen_state->ir_builder_.CreateBitCast(other_ptr, pi64_type));
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
    const auto val = cgen_state->ir_builder_.CreateLoad(
        cgen_state->ir_builder_.CreateBitCast(other_ptr, pi32_type));
    cgen_state->emitCall("agg_sum_int32", {agg, val});
  } else {
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
    const auto agg = cgen_state->ir_builder_.CreateBitCast(val_ptr, pi64_type);
    const auto val = cgen_state->ir_builder_.CreateLoad(
        cgen_state->ir_builder_.CreateBitCast(other_ptr, pi64_type));
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
    const auto pi32_type = llvm::Type::getInt32PtrTy(cgen_state->context_);
    const auto proj_val = cgen_state->ir_builder_.CreateLoad(
        cgen_state->ir_builder_.CreateBitCast(other_pi8, pi32_type));
    cgen_state->emitCall(func_name, {slot_pi8, proj_val, cgen_state->llInt(init_val)});
  } else {
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
    const auto proj_val = cgen_state->ir_builder_.CreateLoad(
        cgen_state->ir_builder_.CreateBitCast(other_pi8, pi64_type));
    cgen_state->emitCall(func_name, {slot_pi8, proj_val, cgen_state->llInt(init_val)});
  }
}

ReductionCode setup_reduction_function() {
  ReductionCode reduction_code;
  reduction_code.cgen_state.reset(new CgenState({}, false));
  auto cgen_state = reduction_code.cgen_state.get();
  auto& ctx = cgen_state->context_;
  std::unique_ptr<llvm::Module> module(runtime_module_shallow_copy(ctx, cgen_state));
  cgen_state->module_ = module.get();
  const auto pi8_type = llvm::PointerType::get(get_int_type(8, ctx), 0);
  const auto pvoid_type = llvm::PointerType::get(llvm::Type::getVoidTy(ctx), 0);
  const auto func_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx), {pi8_type, pi8_type, pvoid_type, pvoid_type}, false);
  const auto func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "reduce_one_entry", module.get());
  reduction_code.ir_func = func;
  const auto arg_it = func->arg_begin();
  const auto this_targets_ptr_arg = &*arg_it;
  const auto that_targets_ptr_arg = &*(arg_it + 1);
  const auto this_qmd_arg = &*(arg_it + 2);
  const auto that_qmd_arg = &*(arg_it + 3);
  this_targets_ptr_arg->setName("this_targets_ptr");
  that_targets_ptr_arg->setName("that_targets_ptr");
  this_qmd_arg->setName("this_qmd");
  that_qmd_arg->setName("that_qmd");
  const auto bb_entry = llvm::BasicBlock::Create(ctx, ".entry", func, 0);
  cgen_state->ir_builder_.SetInsertPoint(bb_entry);
  reduction_code.module = std::move(module);
  return std::move(reduction_code);
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

ReductionCode ResultSetStorage::reduceOneEntryNoCollisionsRowWiseJIT(
    const ResultSetStorage& that) const {
  const auto sample_targets =
      std::count_if(targets_.begin(), targets_.end(), [](const TargetInfo& target_info) {
        return target_info.agg_kind == kSAMPLE;
      });
  if (sample_targets) {
    return {};
  }
  ReductionCode reduction_code = setup_reduction_function();
  auto cgen_state = reduction_code.cgen_state.get();
  const auto& col_slot_context = query_mem_desc_.getColSlotContext();

  const auto arg_it = reduction_code.ir_func->arg_begin();
  const auto this_targets_ptr_arg = &*arg_it;
  const auto that_targets_ptr_arg = &*(arg_it + 1);
  llvm::Value* this_targets_ptr = this_targets_ptr_arg;
  llvm::Value* that_targets_ptr = that_targets_ptr_arg;
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
      if (target_logical_idx + 1 == targets_.size()) {
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
  if (query_mem_desc_.didOutputColumnar()) {
    return {};
  }
  const auto sample_targets =
      std::count_if(targets_.begin(), targets_.end(), [](const TargetInfo& target_info) {
        return target_info.agg_kind == kSAMPLE;
      });
  if (sample_targets) {
    return {};
  }
  ReductionCode reduction_code = setup_reduction_function();
  auto cgen_state = reduction_code.cgen_state.get();
  const auto arg_it = reduction_code.ir_func->arg_begin();
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

ReductionCode ResultSetStorage::finalizeReductionCode(ReductionCode reduction_code,
                                                      CgenState* cgen_state) const {
  cgen_state->ir_builder_.CreateRetVoid();
  verify_function_ir(reduction_code.ir_func);
  const auto key = serialize_llvm_object(reduction_code.ir_func);
  auto val_ptr = g_code_cache.get({key});
  if (val_ptr) {
    return {
        nullptr,
        std::get<1>(val_ptr->first.front()).get(),
        nullptr,
        nullptr,
        reinterpret_cast<ReductionCode::FuncPtr>(std::get<0>(val_ptr->first.front()))};
  }
  CompilationOptions co{ExecutorDeviceType::CPU, false, ExecutorOptLevel::Default, false};
  reduction_code.module.release();
  auto ee = g_reduction_jit_interp
                ? generate_native_reduction_code(reduction_code.ir_func)
                : CodeGenerator::generateNativeCPUCode(
                      reduction_code.ir_func, {reduction_code.ir_func}, co);
  reduction_code.func_ptr = g_reduction_jit_interp
                                ? nullptr
                                : reinterpret_cast<ReductionCode::FuncPtr>(
                                      ee->getPointerToFunction(reduction_code.ir_func));
  reduction_code.execution_engine = ee.get();
  if (!g_reduction_jit_interp) {
    std::tuple<void*, ExecutionEngineWrapper> cache_val =
        std::make_tuple(reinterpret_cast<void*>(reduction_code.func_ptr), std::move(ee));
    std::vector<std::tuple<void*, ExecutionEngineWrapper>> cache_vals;
    cache_vals.emplace_back(std::move(cache_val));
    Executor::addCodeToCache(
        {key}, std::move(cache_vals), reduction_code.ir_func->getParent(), g_code_cache);
  }
  return std::move(reduction_code);
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
    CHECK(target_info.agg_kind != kSAMPLE) << "Not supported yet";
    emit_write_projection(this_ptr1, that_ptr1, init_val, chosen_bytes, cgen_state);
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
  const auto pi64_type = llvm::Type::getInt64PtrTy(cgen_state->context_);
  const auto old_set_handle = cgen_state->ir_builder_.CreateLoad(
      cgen_state->ir_builder_.CreateBitCast(this_ptr1, pi64_type));
  const auto new_set_handle = cgen_state->ir_builder_.CreateLoad(
      cgen_state->ir_builder_.CreateBitCast(that_ptr1, pi64_type));
  const auto arg_it = reduction_code.ir_func->arg_begin();
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
