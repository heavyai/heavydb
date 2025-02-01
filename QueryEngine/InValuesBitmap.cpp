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

#include "InValuesBitmap.h"
#include "CodeGenerator.h"
#include "Execute.h"
#ifdef HAVE_CUDA
#include "GpuMemUtils.h"
#endif  // HAVE_CUDA
#include "../Parser/ParserNode.h"
#include "../Shared/checked_alloc.h"
#include "GroupByAndAggregate.h"
#include "Logger/Logger.h"
#include "QueryEngine/CodegenHelper.h"
#include "QueryEngine/QueryEngine.h"
#include "RuntimeFunctions.h"

#include <limits>

extern int64_t g_bitmap_memory_limit;

InValuesBitmap::InValuesBitmap(const std::vector<int64_t>& values,
                               const int64_t null_val,
                               const Data_Namespace::MemoryLevel memory_level,
                               Executor* executor,
                               CompilationOptions const& co)
    : rhs_has_null_(false), null_val_(null_val), memory_level_(memory_level), co_(co) {
#ifdef HAVE_CUDA
  CHECK(memory_level_ == Data_Namespace::CPU_LEVEL ||
        memory_level == Data_Namespace::GPU_LEVEL);
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
#endif  // HAVE_CUDA
  if (values.empty()) {
    return;
  }
  min_val_ = std::numeric_limits<int64_t>::max();
  max_val_ = std::numeric_limits<int64_t>::min();
  for (const auto value : values) {
    if (value == null_val) {
      rhs_has_null_ = true;
      continue;
    }
    if (value < min_val_) {
      min_val_ = value;
    }
    if (value > max_val_) {
      max_val_ = value;
    }
  }
  if (max_val_ < min_val_) {
    CHECK_EQ(std::numeric_limits<int64_t>::max(), min_val_);
    CHECK_EQ(std::numeric_limits<int64_t>::min(), max_val_);
    CHECK(rhs_has_null_);
    return;
  }
  uint64_t const bitmap_sz_bits_minus_one = max_val_ - min_val_;
  if (static_cast<uint64_t>(g_bitmap_memory_limit) <= bitmap_sz_bits_minus_one) {
    throw FailedToCreateBitmap();
  }
  // bitmap_sz_bytes = ceil(bitmap_sz_bits / 8.0) = (bitmap_sz_bits-1) / 8 + 1
  uint64_t const bitmap_sz_bytes = bitmap_sz_bits_minus_one / 8 + 1;
  auto cpu_bitset = static_cast<int8_t*>(checked_calloc(bitmap_sz_bytes, 1));
  for (const auto value : values) {
    if (value == null_val) {
      continue;
    }
    agg_count_distinct_bitmap(
        reinterpret_cast<int64_t*>(&cpu_bitset), value, min_val_, 0);
  }
  constexpr int cpu_device_id = 0;
#ifdef HAVE_CUDA
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    for (auto const device_id : executor->getAvailableDevicesToProcessQuery()) {
      auto device_allocator = executor->getCudaAllocator(device_id);
      CHECK(device_allocator);
      auto gpu_bitset = device_allocator->alloc(bitmap_sz_bytes);
      device_allocator->copyToDevice(
          gpu_bitset, cpu_bitset, bitmap_sz_bytes, "In-value bitset");
      bitsets_per_devices_.emplace(device_id, gpu_bitset);
    }
    free(cpu_bitset);
  } else {
    bitsets_per_devices_.emplace(cpu_device_id, cpu_bitset);
  }
#else
  CHECK_EQ(1u, executor->getAvailableDevicesToProcessQuery().size());
  bitsets_per_devices_.emplace(cpu_device_id, cpu_bitset);
#endif  // HAVE_CUDA
}

InValuesBitmap::~InValuesBitmap() {
  if (bitsets_per_devices_.empty()) {
    return;
  }
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    CHECK_EQ(size_t(1), bitsets_per_devices_.size());
    free(bitsets_per_devices_.begin()->second);
  }
}

InValuesBitmap::BitIsSetParams InValuesBitmap::prepareBitIsSetParams(
    Executor* executor,
    std::unordered_map<int, std::shared_ptr<const Analyzer::Constant>> const&
        constant_owned) const {
  BitIsSetParams params;
  auto pi8_ty =
      llvm::PointerType::get(get_int_type(8, executor->cgen_state_->context_), 0);
  CodeGenerator code_generator(executor);
  params.null_val_lv =
      CodegenUtil::hoistLiteral(&code_generator,
                                co_,
                                make_datum<int64_t>(null_val_),
                                kBIGINT,
                                executor->getAvailableDevicesToProcessQuery())
          .begin()
          ->second;
  if (bitsets_per_devices_.empty()) {
    auto const zero_lvs =
        CodegenUtil::hoistLiteral(&code_generator,
                                  co_,
                                  make_datum<int64_t>(0),
                                  kBIGINT,
                                  executor->getAvailableDevicesToProcessQuery());
    params.min_val_lv = zero_lvs.begin()->second;
    params.max_val_lv = zero_lvs.begin()->second;
    params.bitmap_ptr_lv = executor->cgen_state_->ir_builder_.CreateIntToPtr(
        zero_lvs.begin()->second, pi8_ty);
  } else {
    params.min_val_lv =
        CodegenUtil::hoistLiteral(&code_generator,
                                  co_,
                                  make_datum<int64_t>(min_val_),
                                  kBIGINT,
                                  executor->getAvailableDevicesToProcessQuery())
            .begin()
            ->second;
    params.max_val_lv =
        CodegenUtil::hoistLiteral(&code_generator,
                                  co_,
                                  make_datum<int64_t>(max_val_),
                                  kBIGINT,
                                  executor->getAvailableDevicesToProcessQuery())
            .begin()
            ->second;
    std::unordered_map<int, const Analyzer::Constant*> bitmap_constants;
    for (auto& kv : constant_owned) {
      bitmap_constants.emplace(kv.first, kv.second.get());
    }
    const auto bitset_handle_lvs =
        code_generator.codegenHoistedConstants(bitmap_constants, kENCODING_NONE, {});
    CHECK_EQ(size_t(1), bitset_handle_lvs.size());
    params.bitmap_ptr_lv = executor->cgen_state_->ir_builder_.CreateIntToPtr(
        bitset_handle_lvs.front(), pi8_ty);
  }
  return params;
}

llvm::Value* InValuesBitmap::codegen(llvm::Value* needle, Executor* executor) const {
  auto cgen_state = executor->getCgenStatePtr();
  AUTOMATIC_IR_METADATA(cgen_state);
  std::unordered_map<int, std::shared_ptr<const Analyzer::Constant>> constants_owned;
  for (const auto kv : bitsets_per_devices_) {
    const int64_t bitset_handle = reinterpret_cast<int64_t>(kv.second);
    const auto bitset_handle_literal = std::dynamic_pointer_cast<Analyzer::Constant>(
        Parser::IntLiteral::analyzeValue(bitset_handle));
    CHECK(bitset_handle_literal);
    CHECK_EQ(kENCODING_NONE, bitset_handle_literal->get_type_info().get_compression());
    constants_owned.emplace(kv.first, bitset_handle_literal);
  }
  const auto needle_i64 = cgen_state->castToTypeIn(needle, 64);
  const auto null_bool_val =
      static_cast<int8_t>(inline_int_null_val(SQLTypeInfo(kBOOLEAN, false)));
  auto const func_params = prepareBitIsSetParams(executor, constants_owned);
  return cgen_state->emitCall("bit_is_set",
                              {func_params.bitmap_ptr_lv,
                               needle_i64,
                               func_params.min_val_lv,
                               func_params.max_val_lv,
                               func_params.null_val_lv,
                               cgen_state->llInt(null_bool_val)});
}

bool InValuesBitmap::isEmpty() const {
  return bitsets_per_devices_.empty();
}

bool InValuesBitmap::hasNull() const {
  return rhs_has_null_;
}
