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

#include "InValuesBitmap.h"
#include "Execute.h"
#ifdef HAVE_CUDA
#include "GpuMemUtils.h"
#endif  // HAVE_CUDA
#include "../Parser/ParserNode.h"
#include "../Shared/checked_alloc.h"
#include "GroupByAndAggregate.h"
#include "RuntimeFunctions.h"

#include <glog/logging.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <limits>

using checked_int64_t = boost::multiprecision::number<
    boost::multiprecision::cpp_int_backend<64,
                                           64,
                                           boost::multiprecision::signed_magnitude,
                                           boost::multiprecision::checked,
                                           void>>;

InValuesBitmap::InValuesBitmap(const std::vector<int64_t>& values,
                               const int64_t null_val,
                               const Data_Namespace::MemoryLevel memory_level,
                               const int device_count,
                               Data_Namespace::DataMgr* data_mgr)
    : rhs_has_null_(false)
    , null_val_(null_val)
    , memory_level_(memory_level)
    , device_count_(device_count) {
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
  const int64_t MAX_BITMAP_BITS{8 * 1000 * 1000 * 1000L};
  const auto bitmap_sz_bits =
      static_cast<int64_t>(checked_int64_t(max_val_) - min_val_ + 1);
  if (bitmap_sz_bits > MAX_BITMAP_BITS) {
    throw FailedToCreateBitmap();
  }
  const auto bitmap_sz_bytes = bitmap_bits_to_bytes(bitmap_sz_bits);
  auto cpu_bitset = static_cast<int8_t*>(checked_calloc(bitmap_sz_bytes, 1));
  for (const auto value : values) {
    if (value == null_val) {
      continue;
    }
    agg_count_distinct_bitmap(reinterpret_cast<int64_t*>(&cpu_bitset), value, min_val_);
  }
#ifdef HAVE_CUDA
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      auto gpu_bitset = alloc_gpu_mem(data_mgr, bitmap_sz_bytes, device_id, nullptr);
      copy_to_gpu(data_mgr, gpu_bitset, cpu_bitset, bitmap_sz_bytes, device_id);
      bitsets_.push_back(reinterpret_cast<int8_t*>(gpu_bitset));
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
  }
}

llvm::Value* InValuesBitmap::codegen(llvm::Value* needle, Executor* executor) const {
  CHECK(!bitsets_.empty());
  std::vector<std::shared_ptr<const Analyzer::Constant>> constants_owned;
  std::vector<const Analyzer::Constant*> constants;
  for (const auto bitset : bitsets_) {
    const int64_t bitset_handle = reinterpret_cast<int64_t>(bitset);
    const auto bitset_handle_literal = std::dynamic_pointer_cast<Analyzer::Constant>(
        Parser::IntLiteral::analyzeValue(bitset_handle));
    CHECK(bitset_handle_literal);
    CHECK_EQ(kENCODING_NONE, bitset_handle_literal->get_type_info().get_compression());
    constants_owned.push_back(bitset_handle_literal);
    constants.push_back(bitset_handle_literal.get());
  }
  const auto bitset_handle_lvs =
      executor->codegenHoistedConstants(constants, kENCODING_NONE, 0);
  CHECK_EQ(size_t(1), bitset_handle_lvs.size());
  const auto needle_i64 = executor->castToTypeIn(needle, 64);
  const auto null_bool_val =
      static_cast<int8_t>(inline_int_null_val(SQLTypeInfo(kBOOLEAN, false)));
  return executor->cgen_state_->emitCall(
      "bit_is_set",
      {executor->castToTypeIn(bitset_handle_lvs.front(), 64),
       needle_i64,
       executor->ll_int(min_val_),
       executor->ll_int(max_val_),
       executor->ll_int(null_val_),
       executor->ll_int(null_bool_val)});
}

bool InValuesBitmap::isEmpty() const {
  return bitsets_.empty();
}

bool InValuesBitmap::hasNull() const {
  return rhs_has_null_;
}
