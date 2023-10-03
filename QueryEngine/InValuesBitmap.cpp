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
                               const int device_count,
                               Data_Namespace::DataMgr* data_mgr,
                               CompilationOptions const& co)
    : rhs_has_null_(false)
    , null_val_(null_val)
    , memory_level_(memory_level)
    , device_count_(device_count)
    , data_mgr_(data_mgr)
    , co_(co) {
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
#ifdef HAVE_CUDA
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      auto device_allocator = std::make_unique<CudaAllocator>(
          data_mgr_, device_id, getQueryEngineCudaStreamForDevice(device_id));
      gpu_buffers_.emplace_back(
          data_mgr->alloc(Data_Namespace::GPU_LEVEL, device_id, bitmap_sz_bytes));
      auto gpu_bitset = gpu_buffers_.back()->getMemoryPtr();
      device_allocator->copyToDevice(gpu_bitset, cpu_bitset, bitmap_sz_bytes);
      bitsets_.push_back(gpu_bitset);
    }
    free(cpu_bitset);
  } else {
    bitsets_.push_back(cpu_bitset);
  }
#else
  CHECK_EQ(1, device_count_);
  bitsets_.push_back(cpu_bitset);
#endif  // HAVE_CUDA
}

InValuesBitmap::~InValuesBitmap() {
  if (bitsets_.empty()) {
    return;
  }
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    CHECK_EQ(size_t(1), bitsets_.size());
    free(bitsets_.front());
  } else {
    CHECK(data_mgr_);
    for (auto& gpu_buffer : gpu_buffers_) {
      data_mgr_->free(gpu_buffer);
    }
  }
}

InValuesBitmap::BitIsSetParams InValuesBitmap::prepareBitIsSetParams(
    Executor* executor,
    std::vector<std::shared_ptr<const Analyzer::Constant>> const& constant_owned) const {
  BitIsSetParams params;
  auto pi8_ty =
      llvm::PointerType::get(get_int_type(8, executor->cgen_state_->context_), 0);
  CodeGenerator code_generator(executor);
  params.null_val_lv =
      CodegenUtil::hoistLiteral(
          &code_generator, co_, make_datum<int64_t>(null_val_), kBIGINT, device_count_)
          .front();
  if (bitsets_.empty()) {
    auto const zero_lvs = CodegenUtil::hoistLiteral(
        &code_generator, co_, make_datum<int64_t>(0), kBIGINT, device_count_);
    params.min_val_lv = zero_lvs.front();
    params.max_val_lv = zero_lvs.front();
    params.bitmap_ptr_lv =
        executor->cgen_state_->ir_builder_.CreateIntToPtr(zero_lvs.front(), pi8_ty);
  } else {
    params.min_val_lv =
        CodegenUtil::hoistLiteral(
            &code_generator, co_, make_datum<int64_t>(min_val_), kBIGINT, device_count_)
            .front();
    params.max_val_lv =
        CodegenUtil::hoistLiteral(
            &code_generator, co_, make_datum<int64_t>(max_val_), kBIGINT, device_count_)
            .front();
    auto to_raw_ptr = [](const auto& ptr) { return ptr.get(); };
    auto begin = boost::make_transform_iterator(constant_owned.begin(), to_raw_ptr);
    auto end = boost::make_transform_iterator(constant_owned.end(), to_raw_ptr);
    std::vector<const Analyzer::Constant*> bitmap_constants(begin, end);
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
  std::vector<std::shared_ptr<const Analyzer::Constant>> constants_owned;
  for (const auto bitset : bitsets_) {
    const int64_t bitset_handle = reinterpret_cast<int64_t>(bitset);
    const auto bitset_handle_literal = std::dynamic_pointer_cast<Analyzer::Constant>(
        Parser::IntLiteral::analyzeValue(bitset_handle));
    CHECK(bitset_handle_literal);
    CHECK_EQ(kENCODING_NONE, bitset_handle_literal->get_type_info().get_compression());
    constants_owned.push_back(bitset_handle_literal);
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
  return bitsets_.empty();
}

bool InValuesBitmap::hasNull() const {
  return rhs_has_null_;
}
