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
#include "ResultSetReductionCodegen.h"
#include "ResultSetReductionInterpreterStubs.h"

#include "CodeGenerator.h"
#include "DynamicWatchdog.h"
#include "Execute.h"
#include "IRCodegenUtils.h"
#include "LLVMFunctionAttributesUtil.h"
#include "Shared/likely.h"
#include "Shared/quantile.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_os_ostream.h>

extern std::unique_ptr<llvm::Module> g_rt_module;

CodeCache ResultSetReductionJIT::s_code_cache(10000);

std::mutex ReductionCode::s_reduction_mutex;

namespace {

// Error code to be returned when the watchdog timer triggers during the reduction.
const int32_t WATCHDOG_ERROR{-1};
// Use the interpreter, not the JIT, for a number of entries lower than the threshold.
const size_t INTERP_THRESHOLD{25};

// Load the value stored at 'ptr' interpreted as 'ptr_type'.
Value* emit_load(Value* ptr, Type ptr_type, Function* function) {
  return function->add<Load>(
      function->add<Cast>(Cast::CastOp::BitCast, ptr, ptr_type, ""),
      ptr->label() + "_loaded");
}

// Load the value stored at 'ptr' as a 32-bit signed integer.
Value* emit_load_i32(Value* ptr, Function* function) {
  return emit_load(ptr, Type::Int32Ptr, function);
}

// Load the value stored at 'ptr' as a 64-bit signed integer.
Value* emit_load_i64(Value* ptr, Function* function) {
  return emit_load(ptr, Type::Int64Ptr, function);
}

// Read a 32- or 64-bit integer stored at 'ptr' and sign extend to 64-bit.
Value* emit_read_int_from_buff(Value* ptr, const int8_t compact_sz, Function* function) {
  switch (compact_sz) {
    case 8: {
      return emit_load_i64(ptr, function);
    }
    case 4: {
      const auto loaded_val = emit_load_i32(ptr, function);
      return function->add<Cast>(Cast::CastOp::SExt, loaded_val, Type::Int64, "");
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
                              Value* val_ptr,
                              Value* other_ptr,
                              const size_t chosen_bytes,
                              const TargetInfo& agg_info,
                              Function* ir_reduce_one_entry) {
  const auto sql_type = get_compact_type(agg_info);
  const auto dest_name = agg_kind + "_dest";
  if (sql_type.is_fp()) {
    if (chosen_bytes == sizeof(float)) {
      const auto agg = ir_reduce_one_entry->add<Cast>(
          Cast::CastOp::BitCast, val_ptr, Type::Int32Ptr, dest_name);
      const auto val = emit_load(other_ptr, Type::FloatPtr, ir_reduce_one_entry);
      ir_reduce_one_entry->add<Call>(
          "agg_" + agg_kind + "_float", std::vector<const Value*>{agg, val}, "");
    } else {
      CHECK_EQ(chosen_bytes, sizeof(double));
      const auto agg = ir_reduce_one_entry->add<Cast>(
          Cast::CastOp::BitCast, val_ptr, Type::Int64Ptr, dest_name);
      const auto val = emit_load(other_ptr, Type::DoublePtr, ir_reduce_one_entry);
      ir_reduce_one_entry->add<Call>(
          "agg_" + agg_kind + "_double", std::vector<const Value*>{agg, val}, "");
    }
  } else {
    if (chosen_bytes == sizeof(int32_t)) {
      const auto agg = ir_reduce_one_entry->add<Cast>(
          Cast::CastOp::BitCast, val_ptr, Type::Int32Ptr, dest_name);
      const auto val = emit_load(other_ptr, Type::Int32Ptr, ir_reduce_one_entry);
      ir_reduce_one_entry->add<Call>(
          "agg_" + agg_kind + "_int32", std::vector<const Value*>{agg, val}, "");
    } else {
      CHECK_EQ(chosen_bytes, sizeof(int64_t));
      const auto agg = ir_reduce_one_entry->add<Cast>(
          Cast::CastOp::BitCast, val_ptr, Type::Int64Ptr, dest_name);
      const auto val = emit_load(other_ptr, Type::Int64Ptr, ir_reduce_one_entry);
      ir_reduce_one_entry->add<Call>(
          "agg_" + agg_kind, std::vector<const Value*>{agg, val}, "");
    }
  }
}

// Same as above, but support nullable types as well.
void emit_aggregate_one_nullable_value(const std::string& agg_kind,
                                       Value* val_ptr,
                                       Value* other_ptr,
                                       const int64_t init_val,
                                       const size_t chosen_bytes,
                                       const TargetInfo& agg_info,
                                       Function* ir_reduce_one_entry) {
  const auto dest_name = agg_kind + "_dest";
  if (agg_info.skip_null_val) {
    const auto sql_type = get_compact_type(agg_info);
    if (sql_type.is_fp()) {
      if (chosen_bytes == sizeof(float)) {
        const auto agg = ir_reduce_one_entry->add<Cast>(
            Cast::CastOp::BitCast, val_ptr, Type::Int32Ptr, dest_name);
        const auto val = emit_load(other_ptr, Type::FloatPtr, ir_reduce_one_entry);
        const auto init_val_lv = ir_reduce_one_entry->addConstant<ConstantFP>(
            *reinterpret_cast<const float*>(may_alias_ptr(&init_val)), Type::Float);
        ir_reduce_one_entry->add<Call>("agg_" + agg_kind + "_float_skip_val",
                                       std::vector<const Value*>{agg, val, init_val_lv},
                                       "");
      } else {
        CHECK_EQ(chosen_bytes, sizeof(double));
        const auto agg = ir_reduce_one_entry->add<Cast>(
            Cast::CastOp::BitCast, val_ptr, Type::Int64Ptr, dest_name);
        const auto val = emit_load(other_ptr, Type::DoublePtr, ir_reduce_one_entry);
        const auto init_val_lv = ir_reduce_one_entry->addConstant<ConstantFP>(
            *reinterpret_cast<const double*>(may_alias_ptr(&init_val)), Type::Double);
        ir_reduce_one_entry->add<Call>("agg_" + agg_kind + "_double_skip_val",
                                       std::vector<const Value*>{agg, val, init_val_lv},
                                       "");
      }
    } else {
      if (chosen_bytes == sizeof(int32_t)) {
        const auto agg = ir_reduce_one_entry->add<Cast>(
            Cast::CastOp::BitCast, val_ptr, Type::Int32Ptr, dest_name);
        const auto val = emit_load(other_ptr, Type::Int32Ptr, ir_reduce_one_entry);
        const auto init_val_lv =
            ir_reduce_one_entry->addConstant<ConstantInt>(init_val, Type::Int32);
        ir_reduce_one_entry->add<Call>("agg_" + agg_kind + "_int32_skip_val",
                                       std::vector<const Value*>{agg, val, init_val_lv},
                                       "");
      } else {
        CHECK_EQ(chosen_bytes, sizeof(int64_t));
        const auto agg = ir_reduce_one_entry->add<Cast>(
            Cast::CastOp::BitCast, val_ptr, Type::Int64Ptr, dest_name);
        const auto val = emit_load(other_ptr, Type::Int64Ptr, ir_reduce_one_entry);
        const auto init_val_lv =
            ir_reduce_one_entry->addConstant<ConstantInt>(init_val, Type::Int64);
        ir_reduce_one_entry->add<Call>("agg_" + agg_kind + "_skip_val",
                                       std::vector<const Value*>{agg, val, init_val_lv},
                                       "");
      }
    }
  } else {
    emit_aggregate_one_value(
        agg_kind, val_ptr, other_ptr, chosen_bytes, agg_info, ir_reduce_one_entry);
  }
}

// Emit code to accumulate the 'other_ptr' count into the 'val_ptr' destination.
void emit_aggregate_one_count(Value* val_ptr,
                              Value* other_ptr,
                              const size_t chosen_bytes,
                              Function* ir_reduce_one_entry) {
  const auto dest_name = "count_dest";
  if (chosen_bytes == sizeof(int32_t)) {
    const auto agg = ir_reduce_one_entry->add<Cast>(
        Cast::CastOp::BitCast, val_ptr, Type::Int32Ptr, dest_name);
    const auto val = emit_load(other_ptr, Type::Int32Ptr, ir_reduce_one_entry);
    ir_reduce_one_entry->add<Call>(
        "agg_sum_int32", std::vector<const Value*>{agg, val}, "");
  } else {
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto agg = ir_reduce_one_entry->add<Cast>(
        Cast::CastOp::BitCast, val_ptr, Type::Int64Ptr, dest_name);
    const auto val = emit_load(other_ptr, Type::Int64Ptr, ir_reduce_one_entry);
    ir_reduce_one_entry->add<Call>("agg_sum", std::vector<const Value*>{agg, val}, "");
  }
}

// Emit code to load the value stored at the 'other_pi8' as an integer of the given width
// 'chosen_bytes' and write it to the 'slot_pi8' destination only if necessary (the
// existing value at destination is the initialization value).
void emit_write_projection(Value* slot_pi8,
                           Value* other_pi8,
                           const int64_t init_val,
                           const size_t chosen_bytes,
                           Function* ir_reduce_one_entry) {
  const auto func_name = "write_projection_int" + std::to_string(chosen_bytes * 8);
  if (chosen_bytes == sizeof(int32_t)) {
    const auto proj_val = emit_load_i32(other_pi8, ir_reduce_one_entry);
    ir_reduce_one_entry->add<Call>(
        func_name,
        std::vector<const Value*>{
            slot_pi8,
            proj_val,
            ir_reduce_one_entry->addConstant<ConstantInt>(init_val, Type::Int64)},
        "");
  } else {
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto proj_val = emit_load_i64(other_pi8, ir_reduce_one_entry);
    ir_reduce_one_entry->add<Call>(
        func_name,
        std::vector<const Value*>{
            slot_pi8,
            proj_val,
            ir_reduce_one_entry->addConstant<ConstantInt>(init_val, Type::Int64)},
        "");
  }
}

// Emit code to load the value stored at the 'other_pi8' as an integer of the given width
// 'chosen_bytes' and write it to the 'slot_pi8' destination only if necessary (the
// existing value at destination is the initialization value).
const Value* emit_checked_write_projection(Value* slot_pi8,
                                           Value* other_pi8,
                                           const int64_t init_val,
                                           const size_t chosen_bytes,
                                           Function* ir_reduce_one_entry) {
  if (chosen_bytes == sizeof(int32_t)) {
    const auto func_name = "checked_single_agg_id_int32";
    const auto proj_val = emit_load_i32(other_pi8, ir_reduce_one_entry);
    const auto slot_pi32 = ir_reduce_one_entry->add<Cast>(
        Cast::CastOp::BitCast, slot_pi8, Type::Int32Ptr, "");
    return ir_reduce_one_entry->add<Call>(
        func_name,
        Type::Int32,
        std::vector<const Value*>{
            slot_pi32,
            proj_val,
            ir_reduce_one_entry->addConstant<ConstantInt>(init_val, Type::Int32)},
        "");
  } else {
    const auto func_name = "checked_single_agg_id";
    CHECK_EQ(chosen_bytes, sizeof(int64_t));
    const auto proj_val = emit_load_i64(other_pi8, ir_reduce_one_entry);
    const auto slot_pi64 = ir_reduce_one_entry->add<Cast>(
        Cast::CastOp::BitCast, slot_pi8, Type::Int64Ptr, "");

    return ir_reduce_one_entry->add<Call>(
        func_name,
        Type::Int32,
        std::vector<const Value*>{
            slot_pi64,
            proj_val,
            ir_reduce_one_entry->addConstant<ConstantInt>(init_val, Type::Int64)},
        "");
  }
}

std::unique_ptr<Function> create_function(
    const std::string name,
    const std::vector<Function::NamedArg>& arg_types,
    const Type ret_type,
    const bool always_inline) {
  return std::make_unique<Function>(name, arg_types, ret_type, always_inline);
}

// Create the declaration for the 'is_empty_entry' function. Use private linkage since
// it's a helper only called from the generated code and mark it as always inline.
std::unique_ptr<Function> setup_is_empty_entry(ReductionCode* reduction_code) {
  return create_function(
      "is_empty_entry", {{"row_ptr", Type::Int8Ptr}}, Type::Int1, /*always_inline=*/true);
}

// Create the declaration for the 'reduce_one_entry' helper.
std::unique_ptr<Function> setup_reduce_one_entry(ReductionCode* reduction_code,
                                                 const QueryDescriptionType hash_type) {
  std::string this_ptr_name;
  std::string that_ptr_name;
  switch (hash_type) {
    case QueryDescriptionType::GroupByBaselineHash: {
      this_ptr_name = "this_targets_ptr";
      that_ptr_name = "that_targets_ptr";
      break;
    }
    case QueryDescriptionType::GroupByPerfectHash:
    case QueryDescriptionType::NonGroupedAggregate: {
      this_ptr_name = "this_row_ptr";
      that_ptr_name = "that_row_ptr";
      break;
    }
    default: {
      LOG(FATAL) << "Unexpected query description type";
    }
  }
  return create_function("reduce_one_entry",
                         {{this_ptr_name, Type::Int8Ptr},
                          {that_ptr_name, Type::Int8Ptr},
                          {"this_qmd", Type::VoidPtr},
                          {"that_qmd", Type::VoidPtr},
                          {"serialized_varlen_buffer_arg", Type::VoidPtr}},
                         Type::Int32,
                         /*always_inline=*/true);
}

// Create the declaration for the 'reduce_one_entry_idx' helper.
std::unique_ptr<Function> setup_reduce_one_entry_idx(ReductionCode* reduction_code) {
  return create_function("reduce_one_entry_idx",
                         {{"this_buff", Type::Int8Ptr},
                          {"that_buff", Type::Int8Ptr},
                          {"that_entry_idx", Type::Int32},
                          {"that_entry_count", Type::Int32},
                          {"this_qmd_handle", Type::VoidPtr},
                          {"that_qmd_handle", Type::VoidPtr},
                          {"serialized_varlen_buffer", Type::VoidPtr}},
                         Type::Int32,
                         /*always_inline=*/true);
}

// Create the declaration for the 'reduce_loop' entry point. Use external linkage, this is
// the public API of the generated code directly used from result set reduction.
std::unique_ptr<Function> setup_reduce_loop(ReductionCode* reduction_code) {
  return create_function("reduce_loop",
                         {{"this_buff", Type::Int8Ptr},
                          {"that_buff", Type::Int8Ptr},
                          {"start_index", Type::Int32},
                          {"end_index", Type::Int32},
                          {"that_entry_count", Type::Int32},
                          {"this_qmd_handle", Type::VoidPtr},
                          {"that_qmd_handle", Type::VoidPtr},
                          {"serialized_varlen_buffer", Type::VoidPtr}},
                         Type::Int32,
                         /*always_inline=*/false);
}

llvm::Function* create_llvm_function(const Function* function, CgenState* cgen_state) {
  AUTOMATIC_IR_METADATA(cgen_state);
  auto& ctx = cgen_state->context_;
  std::vector<llvm::Type*> parameter_types;
  const auto& arg_types = function->arg_types();
  for (const auto& named_arg : arg_types) {
    CHECK(named_arg.type != Type::Void);
    parameter_types.push_back(llvm_type(named_arg.type, ctx));
  }
  const auto func_type = llvm::FunctionType::get(
      llvm_type(function->ret_type(), ctx), parameter_types, false);
  const auto linkage = function->always_inline() ? llvm::Function::PrivateLinkage
                                                 : llvm::Function::ExternalLinkage;
  auto func =
      llvm::Function::Create(func_type, linkage, function->name(), cgen_state->module_);
  const auto arg_it = func->arg_begin();
  for (size_t i = 0; i < arg_types.size(); ++i) {
    const auto arg = &*(arg_it + i);
    arg->setName(arg_types[i].name);
  }
  if (function->always_inline()) {
    mark_function_always_inline(func);
  }
  return func;
}

// Setup the reduction function and helpers declarations, create a module and a code
// generation state object.
ReductionCode setup_functions_ir(const QueryDescriptionType hash_type) {
  ReductionCode reduction_code{};
  reduction_code.ir_is_empty = setup_is_empty_entry(&reduction_code);
  reduction_code.ir_reduce_one_entry = setup_reduce_one_entry(&reduction_code, hash_type);
  reduction_code.ir_reduce_one_entry_idx = setup_reduce_one_entry_idx(&reduction_code);
  reduction_code.ir_reduce_loop = setup_reduce_loop(&reduction_code);
  return reduction_code;
}

bool is_aggregate_query(const QueryDescriptionType hash_type) {
  return hash_type == QueryDescriptionType::GroupByBaselineHash ||
         hash_type == QueryDescriptionType::GroupByPerfectHash ||
         hash_type == QueryDescriptionType::NonGroupedAggregate;
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

extern "C" void approx_median_jit_rt(const int64_t new_set_handle,
                                     const int64_t old_set_handle,
                                     const void* that_qmd_handle,
                                     const void* this_qmd_handle,
                                     const int64_t target_logical_idx) {
  auto* accumulator = reinterpret_cast<quantile::TDigest*>(old_set_handle);
  auto* incoming = reinterpret_cast<quantile::TDigest*>(new_set_handle);
  accumulator->mergeTDigest(*incoming);
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
  const auto gvi =
      result_set::get_group_value_reduction(reinterpret_cast<int64_t*>(groups_buffer),
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
//              get_row(key(that_row_ptr), this_result_setBuffer)
//   reduce_func_[baseline|perfect_hash](this_ptr, that_ptr)
//
// func reduce_loop(this_buff, that_buff, start_entry_index, end_entry_index):
//   for that_entry_index in [start_entry_index, end_entry_index):
//     reduce_func_idx(this_buff, that_buff, that_entry_index)

ReductionCode ResultSetReductionJIT::codegen() const {
  const auto hash_type = query_mem_desc_.getQueryDescriptionType();
  if (query_mem_desc_.didOutputColumnar() || !is_aggregate_query(hash_type)) {
    return {};
  }
  auto reduction_code = setup_functions_ir(hash_type);
  isEmpty(reduction_code);
  switch (query_mem_desc_.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByPerfectHash:
    case QueryDescriptionType::NonGroupedAggregate: {
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
  // For small result sets, avoid native code generation and use the interpreter instead.
  if (query_mem_desc_.getEntryCount() < INTERP_THRESHOLD &&
      (!query_mem_desc_.getExecutor() || query_mem_desc_.blocksShareMemory())) {
    return reduction_code;
  }
  std::lock_guard<std::mutex> reduction_guard(ReductionCode::s_reduction_mutex);
  CodeCacheKey key{cacheKey()};
  const auto compilation_context = s_code_cache.get(key);
  if (compilation_context) {
    auto cpu_context =
        std::dynamic_pointer_cast<CpuCompilationContext>(compilation_context->first);
    CHECK(cpu_context);
    return {reinterpret_cast<ReductionCode::FuncPtr>(cpu_context->func()),
            nullptr,
            nullptr,
            nullptr,
            std::move(reduction_code.ir_is_empty),
            std::move(reduction_code.ir_reduce_one_entry),
            std::move(reduction_code.ir_reduce_one_entry_idx),
            std::move(reduction_code.ir_reduce_loop)};
  }
  reduction_code.cgen_state.reset(new CgenState({}, false));
  auto cgen_state = reduction_code.cgen_state.get();
  std::unique_ptr<llvm::Module> module = runtime_module_shallow_copy(cgen_state);
  cgen_state->module_ = module.get();
  AUTOMATIC_IR_METADATA(cgen_state);
  auto ir_is_empty = create_llvm_function(reduction_code.ir_is_empty.get(), cgen_state);
  auto ir_reduce_one_entry =
      create_llvm_function(reduction_code.ir_reduce_one_entry.get(), cgen_state);
  auto ir_reduce_one_entry_idx =
      create_llvm_function(reduction_code.ir_reduce_one_entry_idx.get(), cgen_state);
  auto ir_reduce_loop =
      create_llvm_function(reduction_code.ir_reduce_loop.get(), cgen_state);
  std::unordered_map<const Function*, llvm::Function*> f;
  f.emplace(reduction_code.ir_is_empty.get(), ir_is_empty);
  f.emplace(reduction_code.ir_reduce_one_entry.get(), ir_reduce_one_entry);
  f.emplace(reduction_code.ir_reduce_one_entry_idx.get(), ir_reduce_one_entry_idx);
  f.emplace(reduction_code.ir_reduce_loop.get(), ir_reduce_loop);
  translate_function(reduction_code.ir_is_empty.get(), ir_is_empty, reduction_code, f);
  translate_function(
      reduction_code.ir_reduce_one_entry.get(), ir_reduce_one_entry, reduction_code, f);
  translate_function(reduction_code.ir_reduce_one_entry_idx.get(),
                     ir_reduce_one_entry_idx,
                     reduction_code,
                     f);
  translate_function(
      reduction_code.ir_reduce_loop.get(), ir_reduce_loop, reduction_code, f);
  reduction_code.llvm_reduce_loop = ir_reduce_loop;
  reduction_code.module = std::move(module);
  AUTOMATIC_IR_METADATA_DONE();
  return finalizeReductionCode(std::move(reduction_code),
                               ir_is_empty,
                               ir_reduce_one_entry,
                               ir_reduce_one_entry_idx,
                               key);
}

void ResultSetReductionJIT::clearCache() {
  // Clear stub cache to avoid crash caused by non-deterministic static destructor order
  // of LLVM context and the cache.
  StubGenerator::clearCache();
  s_code_cache.clear();
  g_rt_module = nullptr;
}

void ResultSetReductionJIT::isEmpty(const ReductionCode& reduction_code) const {
  auto ir_is_empty = reduction_code.ir_is_empty.get();
  CHECK(is_aggregate_query(query_mem_desc_.getQueryDescriptionType()));
  CHECK(!query_mem_desc_.didOutputColumnar());
  Value* key{nullptr};
  Value* empty_key_val{nullptr};
  const auto keys_ptr = ir_is_empty->arg(0);
  if (query_mem_desc_.hasKeylessHash()) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash);
    CHECK_GE(query_mem_desc_.getTargetIdxForKey(), 0);
    CHECK_LT(static_cast<size_t>(query_mem_desc_.getTargetIdxForKey()),
             target_init_vals_.size());
    const int64_t target_slot_off = result_set::get_byteoff_of_slot(
        query_mem_desc_.getTargetIdxForKey(), query_mem_desc_);
    const auto slot_ptr = ir_is_empty->add<GetElementPtr>(
        keys_ptr,
        ir_is_empty->addConstant<ConstantInt>(target_slot_off, Type::Int32),
        "is_empty_slot_ptr");
    const auto compact_sz =
        query_mem_desc_.getPaddedSlotWidthBytes(query_mem_desc_.getTargetIdxForKey());
    key = emit_read_int_from_buff(slot_ptr, compact_sz, ir_is_empty);
    empty_key_val = ir_is_empty->addConstant<ConstantInt>(
        target_init_vals_[query_mem_desc_.getTargetIdxForKey()], Type::Int64);
  } else {
    switch (query_mem_desc_.getEffectiveKeyWidth()) {
      case 4: {
        CHECK(QueryDescriptionType::GroupByPerfectHash !=
              query_mem_desc_.getQueryDescriptionType());
        key = emit_load_i32(keys_ptr, ir_is_empty);
        empty_key_val = ir_is_empty->addConstant<ConstantInt>(EMPTY_KEY_32, Type::Int32);
        break;
      }
      case 8: {
        key = emit_load_i64(keys_ptr, ir_is_empty);
        empty_key_val = ir_is_empty->addConstant<ConstantInt>(EMPTY_KEY_64, Type::Int64);
        break;
      }
      default:
        LOG(FATAL) << "Invalid key width";
    }
  }
  const auto ret =
      ir_is_empty->add<ICmp>(ICmp::Predicate::EQ, key, empty_key_val, "is_key_empty");
  ir_is_empty->add<Ret>(ret);
}

void ResultSetReductionJIT::reduceOneEntryNoCollisions(
    const ReductionCode& reduction_code) const {
  auto ir_reduce_one_entry = reduction_code.ir_reduce_one_entry.get();
  const auto this_row_ptr = ir_reduce_one_entry->arg(0);
  const auto that_row_ptr = ir_reduce_one_entry->arg(1);
  const auto that_is_empty =
      ir_reduce_one_entry->add<Call>(reduction_code.ir_is_empty.get(),
                                     std::vector<const Value*>{that_row_ptr},
                                     "that_is_empty");
  ir_reduce_one_entry->add<ReturnEarly>(
      that_is_empty, ir_reduce_one_entry->addConstant<ConstantInt>(0, Type::Int32), "");

  const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
  if (key_bytes) {  // copy the key from right hand side
    ir_reduce_one_entry->add<MemCpy>(
        this_row_ptr,
        that_row_ptr,
        ir_reduce_one_entry->addConstant<ConstantInt>(key_bytes, Type::Int32));
  }

  const auto key_bytes_with_padding = align_to_int64(key_bytes);
  const auto key_bytes_lv =
      ir_reduce_one_entry->addConstant<ConstantInt>(key_bytes_with_padding, Type::Int32);
  const auto this_targets_start_ptr = ir_reduce_one_entry->add<GetElementPtr>(
      this_row_ptr, key_bytes_lv, "this_targets_start");
  const auto that_targets_start_ptr = ir_reduce_one_entry->add<GetElementPtr>(
      that_row_ptr, key_bytes_lv, "that_targets_start");

  reduceOneEntryTargetsNoCollisions(
      ir_reduce_one_entry, this_targets_start_ptr, that_targets_start_ptr);
}

void ResultSetReductionJIT::reduceOneEntryTargetsNoCollisions(
    Function* ir_reduce_one_entry,
    Value* this_targets_start_ptr,
    Value* that_targets_start_ptr) const {
  const auto& col_slot_context = query_mem_desc_.getColSlotContext();
  Value* this_targets_ptr = this_targets_start_ptr;
  Value* that_targets_ptr = that_targets_start_ptr;
  size_t init_agg_val_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
       ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    const auto& slots_for_col = col_slot_context.getSlotsForCol(target_logical_idx);
    Value* this_ptr2{nullptr};
    Value* that_ptr2{nullptr};

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
      const auto slot_off =
          ir_reduce_one_entry->addConstant<ConstantInt>(slot_off_val, Type::Int32);
      if (UNLIKELY(two_slot_target)) {
        const auto desc = "target_" + std::to_string(target_logical_idx) + "_second_slot";
        this_ptr2 = ir_reduce_one_entry->add<GetElementPtr>(
            this_targets_ptr, slot_off, "this_" + desc);
        that_ptr2 = ir_reduce_one_entry->add<GetElementPtr>(
            that_targets_ptr, slot_off, "that_" + desc);
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
                    ir_reduce_one_entry);
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
        const auto two_slot_off = ir_reduce_one_entry->addConstant<ConstantInt>(
            slot_off_val + query_mem_desc_.getPaddedSlotWidthBytes(target_slot_idx + 1),
            Type::Int32);
        this_targets_ptr = ir_reduce_one_entry->add<GetElementPtr>(
            this_targets_ptr, two_slot_off, "this_" + next_desc);
        that_targets_ptr = ir_reduce_one_entry->add<GetElementPtr>(
            that_targets_ptr, two_slot_off, "that_" + next_desc);
      } else {
        increment_agg_val_idx_maybe(1);
        this_targets_ptr = ir_reduce_one_entry->add<GetElementPtr>(
            this_targets_ptr, slot_off, "this_" + next_desc);
        that_targets_ptr = ir_reduce_one_entry->add<GetElementPtr>(
            that_targets_ptr, slot_off, "that_" + next_desc);
      }
    }
  }
  ir_reduce_one_entry->add<Ret>(
      ir_reduce_one_entry->addConstant<ConstantInt>(0, Type::Int32));
}

void ResultSetReductionJIT::reduceOneEntryBaseline(
    const ReductionCode& reduction_code) const {
  auto ir_reduce_one_entry = reduction_code.ir_reduce_one_entry.get();
  const auto this_targets_ptr_arg = ir_reduce_one_entry->arg(0);
  const auto that_targets_ptr_arg = ir_reduce_one_entry->arg(1);
  Value* this_ptr1 = this_targets_ptr_arg;
  Value* that_ptr1 = that_targets_ptr_arg;
  size_t j = 0;
  size_t init_agg_val_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
       ++target_logical_idx) {
    const auto& target_info = targets_[target_logical_idx];
    Value* this_ptr2{nullptr};
    Value* that_ptr2{nullptr};
    if (target_info.is_agg &&
        (target_info.agg_kind == kAVG ||
         (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()))) {
      const auto desc = "target_" + std::to_string(target_logical_idx) + "_second_slot";
      const auto second_slot_rel_off =
          ir_reduce_one_entry->addConstant<ConstantInt>(sizeof(int64_t), Type::Int32);
      this_ptr2 = ir_reduce_one_entry->add<GetElementPtr>(
          this_ptr1, second_slot_rel_off, "this_" + desc);
      that_ptr2 = ir_reduce_one_entry->add<GetElementPtr>(
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
                  ir_reduce_one_entry);
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
    auto next_slot_rel_off = ir_reduce_one_entry->addConstant<ConstantInt>(
        init_agg_val_idx * sizeof(int64_t), Type::Int32);
    this_ptr1 = ir_reduce_one_entry->add<GetElementPtr>(
        this_targets_ptr_arg, next_slot_rel_off, next_desc);
    that_ptr1 = ir_reduce_one_entry->add<GetElementPtr>(
        that_targets_ptr_arg, next_slot_rel_off, next_desc);
  }
  ir_reduce_one_entry->add<Ret>(
      ir_reduce_one_entry->addConstant<ConstantInt>(0, Type::Int32));
}

void ResultSetReductionJIT::reduceOneEntryNoCollisionsIdx(
    const ReductionCode& reduction_code) const {
  auto ir_reduce_one_entry_idx = reduction_code.ir_reduce_one_entry_idx.get();
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByPerfectHash ||
        query_mem_desc_.getQueryDescriptionType() ==
            QueryDescriptionType::NonGroupedAggregate);
  const auto this_buff = ir_reduce_one_entry_idx->arg(0);
  const auto that_buff = ir_reduce_one_entry_idx->arg(1);
  const auto entry_idx = ir_reduce_one_entry_idx->arg(2);
  const auto this_qmd_handle = ir_reduce_one_entry_idx->arg(4);
  const auto that_qmd_handle = ir_reduce_one_entry_idx->arg(5);
  const auto serialized_varlen_buffer_arg = ir_reduce_one_entry_idx->arg(6);
  const auto row_bytes = ir_reduce_one_entry_idx->addConstant<ConstantInt>(
      get_row_bytes(query_mem_desc_), Type::Int64);
  const auto entry_idx_64 = ir_reduce_one_entry_idx->add<Cast>(
      Cast::CastOp::SExt, entry_idx, Type::Int64, "entry_idx_64");
  const auto row_off_in_bytes = ir_reduce_one_entry_idx->add<BinaryOperator>(
      BinaryOperator::BinaryOp::Mul, entry_idx_64, row_bytes, "row_off_in_bytes");
  const auto this_row_ptr = ir_reduce_one_entry_idx->add<GetElementPtr>(
      this_buff, row_off_in_bytes, "this_row_ptr");
  const auto that_row_ptr = ir_reduce_one_entry_idx->add<GetElementPtr>(
      that_buff, row_off_in_bytes, "that_row_ptr");
  const auto reduce_rc = ir_reduce_one_entry_idx->add<Call>(
      reduction_code.ir_reduce_one_entry.get(),
      std::vector<const Value*>{this_row_ptr,
                                that_row_ptr,
                                this_qmd_handle,
                                that_qmd_handle,
                                serialized_varlen_buffer_arg},
      "");
  ir_reduce_one_entry_idx->add<Ret>(reduce_rc);
}

void ResultSetReductionJIT::reduceOneEntryBaselineIdx(
    const ReductionCode& reduction_code) const {
  auto ir_reduce_one_entry_idx = reduction_code.ir_reduce_one_entry_idx.get();
  CHECK(query_mem_desc_.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByBaselineHash);
  CHECK(!query_mem_desc_.hasKeylessHash());
  CHECK(!query_mem_desc_.didOutputColumnar());
  const auto this_buff = ir_reduce_one_entry_idx->arg(0);
  const auto that_buff = ir_reduce_one_entry_idx->arg(1);
  const auto that_entry_idx = ir_reduce_one_entry_idx->arg(2);
  const auto that_entry_count = ir_reduce_one_entry_idx->arg(3);
  const auto this_qmd_handle = ir_reduce_one_entry_idx->arg(4);
  const auto that_qmd_handle = ir_reduce_one_entry_idx->arg(5);
  const auto serialized_varlen_buffer_arg = ir_reduce_one_entry_idx->arg(6);
  const auto row_bytes = ir_reduce_one_entry_idx->addConstant<ConstantInt>(
      get_row_bytes(query_mem_desc_), Type::Int64);
  const auto that_entry_idx_64 = ir_reduce_one_entry_idx->add<Cast>(
      Cast::CastOp::SExt, that_entry_idx, Type::Int64, "that_entry_idx_64");
  const auto that_row_off_in_bytes =
      ir_reduce_one_entry_idx->add<BinaryOperator>(BinaryOperator::BinaryOp::Mul,
                                                   that_entry_idx_64,
                                                   row_bytes,
                                                   "that_row_off_in_bytes");
  const auto that_row_ptr = ir_reduce_one_entry_idx->add<GetElementPtr>(
      that_buff, that_row_off_in_bytes, "that_row_ptr");
  const auto that_is_empty =
      ir_reduce_one_entry_idx->add<Call>(reduction_code.ir_is_empty.get(),
                                         std::vector<const Value*>{that_row_ptr},
                                         "that_is_empty");
  ir_reduce_one_entry_idx->add<ReturnEarly>(
      that_is_empty,
      ir_reduce_one_entry_idx->addConstant<ConstantInt>(0, Type::Int32),
      "");
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  const auto one_element =
      ir_reduce_one_entry_idx->addConstant<ConstantInt>(1, Type::Int32);
  const auto this_targets_ptr_i64_ptr = ir_reduce_one_entry_idx->add<Alloca>(
      Type::Int64Ptr, one_element, "this_targets_ptr_out");
  const auto this_is_empty_ptr =
      ir_reduce_one_entry_idx->add<Alloca>(Type::Int8, one_element, "this_is_empty_out");
  ir_reduce_one_entry_idx->add<ExternalCall>(
      "get_group_value_reduction_rt",
      Type::Void,
      std::vector<const Value*>{
          this_buff,
          that_row_ptr,
          ir_reduce_one_entry_idx->addConstant<ConstantInt>(key_count, Type::Int32),
          this_qmd_handle,
          that_buff,
          that_entry_idx,
          that_entry_count,
          row_bytes,
          this_targets_ptr_i64_ptr,
          this_is_empty_ptr},
      "");
  const auto this_targets_ptr_i64 = ir_reduce_one_entry_idx->add<Load>(
      this_targets_ptr_i64_ptr, "this_targets_ptr_i64");
  auto this_is_empty =
      ir_reduce_one_entry_idx->add<Load>(this_is_empty_ptr, "this_is_empty");
  this_is_empty = ir_reduce_one_entry_idx->add<Cast>(
      Cast::CastOp::Trunc, this_is_empty, Type::Int1, "this_is_empty_bool");
  ir_reduce_one_entry_idx->add<ReturnEarly>(
      this_is_empty,
      ir_reduce_one_entry_idx->addConstant<ConstantInt>(0, Type::Int32),
      "");
  const auto key_qw_count = get_slot_off_quad(query_mem_desc_);
  const auto this_targets_ptr = ir_reduce_one_entry_idx->add<Cast>(
      Cast::CastOp::BitCast, this_targets_ptr_i64, Type::Int8Ptr, "this_targets_ptr");
  const auto key_byte_count = key_qw_count * sizeof(int64_t);
  const auto key_byte_count_lv =
      ir_reduce_one_entry_idx->addConstant<ConstantInt>(key_byte_count, Type::Int32);
  const auto that_targets_ptr = ir_reduce_one_entry_idx->add<GetElementPtr>(
      that_row_ptr, key_byte_count_lv, "that_targets_ptr");
  const auto reduce_rc = ir_reduce_one_entry_idx->add<Call>(
      reduction_code.ir_reduce_one_entry.get(),
      std::vector<const Value*>{this_targets_ptr,
                                that_targets_ptr,
                                this_qmd_handle,
                                that_qmd_handle,
                                serialized_varlen_buffer_arg},
      "");
  ir_reduce_one_entry_idx->add<Ret>(reduce_rc);
}

namespace {

void generate_loop_body(For* for_loop,
                        Function* ir_reduce_loop,
                        Function* ir_reduce_one_entry_idx,
                        Value* this_buff,
                        Value* that_buff,
                        Value* start_index,
                        Value* that_entry_count,
                        Value* this_qmd_handle,
                        Value* that_qmd_handle,
                        Value* serialized_varlen_buffer) {
  const auto that_entry_idx = for_loop->add<BinaryOperator>(
      BinaryOperator::BinaryOp::Add, for_loop->iter(), start_index, "that_entry_idx");
  const auto watchdog_sample_seed =
      for_loop->add<Cast>(Cast::CastOp::SExt, that_entry_idx, Type::Int64, "");
  const auto watchdog_triggered =
      for_loop->add<ExternalCall>("check_watchdog_rt",
                                  Type::Int8,
                                  std::vector<const Value*>{watchdog_sample_seed},
                                  "");
  const auto watchdog_triggered_bool =
      for_loop->add<ICmp>(ICmp::Predicate::NE,
                          watchdog_triggered,
                          ir_reduce_loop->addConstant<ConstantInt>(0, Type::Int8),
                          "");
  for_loop->add<ReturnEarly>(
      watchdog_triggered_bool,
      ir_reduce_loop->addConstant<ConstantInt>(WATCHDOG_ERROR, Type::Int32),
      "");
  const auto reduce_rc =
      for_loop->add<Call>(ir_reduce_one_entry_idx,
                          std::vector<const Value*>{this_buff,
                                                    that_buff,
                                                    that_entry_idx,
                                                    that_entry_count,
                                                    this_qmd_handle,
                                                    that_qmd_handle,
                                                    serialized_varlen_buffer},
                          "");

  auto reduce_rc_bool =
      for_loop->add<ICmp>(ICmp::Predicate::NE,
                          reduce_rc,
                          ir_reduce_loop->addConstant<ConstantInt>(0, Type::Int32),
                          "");
  for_loop->add<ReturnEarly>(reduce_rc_bool, reduce_rc, "");
}

}  // namespace

void ResultSetReductionJIT::reduceLoop(const ReductionCode& reduction_code) const {
  auto ir_reduce_loop = reduction_code.ir_reduce_loop.get();
  const auto this_buff_arg = ir_reduce_loop->arg(0);
  const auto that_buff_arg = ir_reduce_loop->arg(1);
  const auto start_index_arg = ir_reduce_loop->arg(2);
  const auto end_index_arg = ir_reduce_loop->arg(3);
  const auto that_entry_count_arg = ir_reduce_loop->arg(4);
  const auto this_qmd_handle_arg = ir_reduce_loop->arg(5);
  const auto that_qmd_handle_arg = ir_reduce_loop->arg(6);
  const auto serialized_varlen_buffer_arg = ir_reduce_loop->arg(7);
  For* for_loop =
      static_cast<For*>(ir_reduce_loop->add<For>(start_index_arg, end_index_arg, ""));
  generate_loop_body(for_loop,
                     ir_reduce_loop,
                     reduction_code.ir_reduce_one_entry_idx.get(),
                     this_buff_arg,
                     that_buff_arg,
                     start_index_arg,
                     that_entry_count_arg,
                     this_qmd_handle_arg,
                     that_qmd_handle_arg,
                     serialized_varlen_buffer_arg);
  ir_reduce_loop->add<Ret>(ir_reduce_loop->addConstant<ConstantInt>(0, Type::Int32));
}

void ResultSetReductionJIT::reduceOneSlot(Value* this_ptr1,
                                          Value* this_ptr2,
                                          Value* that_ptr1,
                                          Value* that_ptr2,
                                          const TargetInfo& target_info,
                                          const size_t target_logical_idx,
                                          const size_t target_slot_idx,
                                          const size_t init_agg_val_idx,
                                          const size_t first_slot_idx_for_target,
                                          Function* ir_reduce_one_entry) const {
  if (query_mem_desc_.targetGroupbyIndicesSize() > 0) {
    if (query_mem_desc_.getTargetGroupbyIndex(target_logical_idx) >= 0) {
      return;
    }
  }
  const bool float_argument_input = takes_float_argument(target_info);
  const auto chosen_bytes = result_set::get_width_for_slot(
      target_slot_idx, float_argument_input, query_mem_desc_);
  CHECK_LT(init_agg_val_idx, target_init_vals_.size());
  auto init_val = target_init_vals_[init_agg_val_idx];
  if (target_info.is_agg &&
      (target_info.agg_kind != kSINGLE_VALUE && target_info.agg_kind != kSAMPLE)) {
    reduceOneAggregateSlot(this_ptr1,
                           this_ptr2,
                           that_ptr1,
                           that_ptr2,
                           target_info,
                           target_logical_idx,
                           target_slot_idx,
                           init_val,
                           chosen_bytes,
                           ir_reduce_one_entry);
  } else if (target_info.agg_kind == kSINGLE_VALUE) {
    const auto checked_rc = emit_checked_write_projection(
        this_ptr1, that_ptr1, init_val, chosen_bytes, ir_reduce_one_entry);

    auto checked_rc_bool = ir_reduce_one_entry->add<ICmp>(
        ICmp::Predicate::NE,
        checked_rc,
        ir_reduce_one_entry->addConstant<ConstantInt>(0, Type::Int32),
        "");

    ir_reduce_one_entry->add<ReturnEarly>(checked_rc_bool, checked_rc, "");

  } else {
    emit_write_projection(
        this_ptr1, that_ptr1, init_val, chosen_bytes, ir_reduce_one_entry);
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
      const auto serialized_varlen_buffer_arg = ir_reduce_one_entry->arg(4);
      ir_reduce_one_entry->add<ExternalCall>(
          "serialized_varlen_buffer_sample",
          Type::Void,
          std::vector<const Value*>{
              serialized_varlen_buffer_arg,
              this_ptr1,
              this_ptr2,
              that_ptr1,
              that_ptr2,
              ir_reduce_one_entry->addConstant<ConstantInt>(init_val, Type::Int64),
              ir_reduce_one_entry->addConstant<ConstantInt>(length_to_elems,
                                                            Type::Int64)},
          "");
    }
  }
}

void ResultSetReductionJIT::reduceOneAggregateSlot(Value* this_ptr1,
                                                   Value* this_ptr2,
                                                   Value* that_ptr1,
                                                   Value* that_ptr2,
                                                   const TargetInfo& target_info,
                                                   const size_t target_logical_idx,
                                                   const size_t target_slot_idx,
                                                   const int64_t init_val,
                                                   const int8_t chosen_bytes,
                                                   Function* ir_reduce_one_entry) const {
  switch (target_info.agg_kind) {
    case kCOUNT:
    case kAPPROX_COUNT_DISTINCT: {
      if (is_distinct_target(target_info)) {
        CHECK_EQ(static_cast<size_t>(chosen_bytes), sizeof(int64_t));
        reduceOneCountDistinctSlot(
            this_ptr1, that_ptr1, target_logical_idx, ir_reduce_one_entry);
        break;
      }
      CHECK_EQ(int64_t(0), init_val);
      emit_aggregate_one_count(this_ptr1, that_ptr1, chosen_bytes, ir_reduce_one_entry);
      break;
    }
    case kAPPROX_MEDIAN:
      CHECK_EQ(chosen_bytes, static_cast<int8_t>(sizeof(int64_t)));
      reduceOneApproxMedianSlot(
          this_ptr1, that_ptr1, target_logical_idx, ir_reduce_one_entry);
      break;
    case kAVG: {
      // Ignore float argument compaction for count component for fear of its overflow
      emit_aggregate_one_count(this_ptr2,
                               that_ptr2,
                               query_mem_desc_.getPaddedSlotWidthBytes(target_slot_idx),
                               ir_reduce_one_entry);
    }
    // fall thru
    case kSUM: {
      emit_aggregate_one_nullable_value("sum",
                                        this_ptr1,
                                        that_ptr1,
                                        init_val,
                                        chosen_bytes,
                                        target_info,
                                        ir_reduce_one_entry);
      break;
    }
    case kMIN: {
      emit_aggregate_one_nullable_value("min",
                                        this_ptr1,
                                        that_ptr1,
                                        init_val,
                                        chosen_bytes,
                                        target_info,
                                        ir_reduce_one_entry);
      break;
    }
    case kMAX: {
      emit_aggregate_one_nullable_value("max",
                                        this_ptr1,
                                        that_ptr1,
                                        init_val,
                                        chosen_bytes,
                                        target_info,
                                        ir_reduce_one_entry);
      break;
    }
    default:
      LOG(FATAL) << "Invalid aggregate type";
  }
}

void ResultSetReductionJIT::reduceOneCountDistinctSlot(
    Value* this_ptr1,
    Value* that_ptr1,
    const size_t target_logical_idx,
    Function* ir_reduce_one_entry) const {
  CHECK_LT(target_logical_idx, query_mem_desc_.getCountDistinctDescriptorsSize());
  const auto old_set_handle = emit_load_i64(this_ptr1, ir_reduce_one_entry);
  const auto new_set_handle = emit_load_i64(that_ptr1, ir_reduce_one_entry);
  const auto this_qmd_arg = ir_reduce_one_entry->arg(2);
  const auto that_qmd_arg = ir_reduce_one_entry->arg(3);
  ir_reduce_one_entry->add<ExternalCall>(
      "count_distinct_set_union_jit_rt",
      Type::Void,
      std::vector<const Value*>{
          new_set_handle,
          old_set_handle,
          that_qmd_arg,
          this_qmd_arg,
          ir_reduce_one_entry->addConstant<ConstantInt>(target_logical_idx, Type::Int64)},
      "");
}

void ResultSetReductionJIT::reduceOneApproxMedianSlot(
    Value* this_ptr1,
    Value* that_ptr1,
    const size_t target_logical_idx,
    Function* ir_reduce_one_entry) const {
  CHECK_LT(target_logical_idx, query_mem_desc_.getCountDistinctDescriptorsSize());
  const auto old_set_handle = emit_load_i64(this_ptr1, ir_reduce_one_entry);
  const auto new_set_handle = emit_load_i64(that_ptr1, ir_reduce_one_entry);
  const auto this_qmd_arg = ir_reduce_one_entry->arg(2);
  const auto that_qmd_arg = ir_reduce_one_entry->arg(3);
  ir_reduce_one_entry->add<ExternalCall>(
      "approx_median_jit_rt",
      Type::Void,
      std::vector<const Value*>{
          new_set_handle,
          old_set_handle,
          that_qmd_arg,
          this_qmd_arg,
          ir_reduce_one_entry->addConstant<ConstantInt>(target_logical_idx, Type::Int64)},
      "");
}

ReductionCode ResultSetReductionJIT::finalizeReductionCode(
    ReductionCode reduction_code,
    const llvm::Function* ir_is_empty,
    const llvm::Function* ir_reduce_one_entry,
    const llvm::Function* ir_reduce_one_entry_idx,
    const CodeCacheKey& key) const {
  CompilationOptions co{
      ExecutorDeviceType::CPU, false, ExecutorOptLevel::ReductionJIT, false};

#ifdef NDEBUG
  LOG(IR) << "Reduction Loop:\n"
          << serialize_llvm_object(reduction_code.llvm_reduce_loop);
  LOG(IR) << "Reduction Is Empty Func:\n" << serialize_llvm_object(ir_is_empty);
  LOG(IR) << "Reduction One Entry Func:\n" << serialize_llvm_object(ir_reduce_one_entry);
  LOG(IR) << "Reduction One Entry Idx Func:\n"
          << serialize_llvm_object(ir_reduce_one_entry_idx);
#else
  LOG(IR) << serialize_llvm_object(reduction_code.cgen_state->module_);
#endif

  reduction_code.module.release();
  auto ee = CodeGenerator::generateNativeCPUCode(
      reduction_code.llvm_reduce_loop, {reduction_code.llvm_reduce_loop}, co);
  reduction_code.func_ptr = reinterpret_cast<ReductionCode::FuncPtr>(
      ee->getPointerToFunction(reduction_code.llvm_reduce_loop));

  auto cpu_compilation_context = std::make_shared<CpuCompilationContext>(std::move(ee));
  cpu_compilation_context->setFunctionPointer(reduction_code.llvm_reduce_loop);
  reduction_code.compilation_context = cpu_compilation_context;
  Executor::addCodeToCache(key,
                           reduction_code.compilation_context,
                           reduction_code.llvm_reduce_loop->getParent(),
                           s_code_cache);
  return reduction_code;
}

namespace {

std::string target_info_key(const TargetInfo& target_info) {
  return std::to_string(target_info.is_agg) + "\n" +
         std::to_string(target_info.agg_kind) + "\n" +
         target_info.sql_type.get_type_name() + "\n" +
         std::to_string(target_info.sql_type.get_notnull()) + "\n" +
         target_info.agg_arg_type.get_type_name() + "\n" +
         std::to_string(target_info.agg_arg_type.get_notnull()) + "\n" +
         std::to_string(target_info.skip_null_val) + "\n" +
         std::to_string(target_info.is_distinct);
}

}  // namespace

std::string ResultSetReductionJIT::cacheKey() const {
  std::vector<std::string> target_init_vals_strings;
  std::transform(target_init_vals_.begin(),
                 target_init_vals_.end(),
                 std::back_inserter(target_init_vals_strings),
                 [](const int64_t v) { return std::to_string(v); });
  const auto target_init_vals_key =
      boost::algorithm::join(target_init_vals_strings, ", ");
  std::vector<std::string> targets_strings;
  std::transform(
      targets_.begin(),
      targets_.end(),
      std::back_inserter(targets_strings),
      [](const TargetInfo& target_info) { return target_info_key(target_info); });
  const auto targets_key = boost::algorithm::join(targets_strings, ", ");
  return query_mem_desc_.reductionKey() + "\n" + target_init_vals_key + "\n" +
         targets_key;
}

ReductionCode GpuReductionHelperJIT::codegen() const {
  const auto hash_type = query_mem_desc_.getQueryDescriptionType();
  auto reduction_code = setup_functions_ir(hash_type);
  CHECK(hash_type == QueryDescriptionType::GroupByPerfectHash);
  isEmpty(reduction_code);
  reduceOneEntryNoCollisions(reduction_code);
  reduceOneEntryNoCollisionsIdx(reduction_code);
  reduceLoop(reduction_code);
  reduction_code.cgen_state.reset(new CgenState({}, false));
  auto cgen_state = reduction_code.cgen_state.get();
  std::unique_ptr<llvm::Module> module(runtime_module_shallow_copy(cgen_state));

  cgen_state->module_ = module.get();
  AUTOMATIC_IR_METADATA(cgen_state);
  auto ir_is_empty = create_llvm_function(reduction_code.ir_is_empty.get(), cgen_state);
  auto ir_reduce_one_entry =
      create_llvm_function(reduction_code.ir_reduce_one_entry.get(), cgen_state);
  auto ir_reduce_one_entry_idx =
      create_llvm_function(reduction_code.ir_reduce_one_entry_idx.get(), cgen_state);
  auto ir_reduce_loop =
      create_llvm_function(reduction_code.ir_reduce_loop.get(), cgen_state);
  std::unordered_map<const Function*, llvm::Function*> f;
  f.emplace(reduction_code.ir_is_empty.get(), ir_is_empty);
  f.emplace(reduction_code.ir_reduce_one_entry.get(), ir_reduce_one_entry);
  f.emplace(reduction_code.ir_reduce_one_entry_idx.get(), ir_reduce_one_entry_idx);
  f.emplace(reduction_code.ir_reduce_loop.get(), ir_reduce_loop);
  translate_function(reduction_code.ir_is_empty.get(), ir_is_empty, reduction_code, f);
  translate_function(
      reduction_code.ir_reduce_one_entry.get(), ir_reduce_one_entry, reduction_code, f);
  translate_function(reduction_code.ir_reduce_one_entry_idx.get(),
                     ir_reduce_one_entry_idx,
                     reduction_code,
                     f);
  translate_function(
      reduction_code.ir_reduce_loop.get(), ir_reduce_loop, reduction_code, f);
  reduction_code.llvm_reduce_loop = ir_reduce_loop;
  reduction_code.module = std::move(module);
  return reduction_code;
}
