/*
 * Copyright 2020 OmniSci, Inc.
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

/**
 * @file    ResultSetStorage.cpp
 * @author
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2020 OmniSci, Inc.,  All rights reserved.
 */

#include "ResultSetStorage.h"

#include "DataMgr/Allocators/CudaAllocator.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Execute.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "OutputBufferInitialization.h"
#include "RuntimeFunctions.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/checked_alloc.h"
#include "Shared/likely.h"

#include <algorithm>
#include <bitset>
#include <future>
#include <numeric>

int8_t* VarlenOutputInfo::computeCpuOffset(const int64_t gpu_offset_address) const {
  const auto gpu_start_address_ptr = reinterpret_cast<int8_t*>(gpu_start_address);
  const auto gpu_offset_address_ptr = reinterpret_cast<int8_t*>(gpu_offset_address);
  if (gpu_offset_address_ptr == 0) {
    return 0;
  }
  const auto offset_bytes =
      static_cast<int64_t>(gpu_offset_address_ptr - gpu_start_address_ptr);
  CHECK_GE(offset_bytes, int64_t(0));
  return cpu_buffer_ptr + offset_bytes;
}

ResultSetStorage::ResultSetStorage(const std::vector<TargetInfo>& targets,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   int8_t* buff,
                                   const bool buff_is_provided)
    : targets_(targets)
    , query_mem_desc_(query_mem_desc)
    , buff_(buff)
    , buff_is_provided_(buff_is_provided)
    , target_init_vals_(result_set::initialize_target_values_for_storage(targets)) {}

int8_t* ResultSetStorage::getUnderlyingBuffer() const {
  return buff_;
}

void ResultSetStorage::addCountDistinctSetPointerMapping(const int64_t remote_ptr,
                                                         const int64_t ptr) {
  const auto it_ok = count_distinct_sets_mapping_.emplace(remote_ptr, ptr);
  CHECK(it_ok.second);
}

int64_t ResultSetStorage::mappedPtr(const int64_t remote_ptr) const {
  const auto it = count_distinct_sets_mapping_.find(remote_ptr);
  // Due to the removal of completely zero bitmaps in a distributed transfer there will be
  // remote ptr that do not not exists. Return 0 if no pointer found
  if (it == count_distinct_sets_mapping_.end()) {
    return int64_t(0);
  }
  return it->second;
}

std::vector<int64_t> result_set::initialize_target_values_for_storage(
    const std::vector<TargetInfo>& targets) {
  std::vector<int64_t> target_init_vals;
  for (const auto& target_info : targets) {
    if (target_info.agg_kind == kCOUNT ||
        target_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
      target_init_vals.push_back(0);
      continue;
    }
    if (!target_info.sql_type.get_notnull()) {
      int64_t init_val =
          null_val_bit_pattern(target_info.sql_type, takes_float_argument(target_info));
      target_init_vals.push_back(target_info.is_agg ? init_val : 0);
    } else {
      target_init_vals.push_back(target_info.is_agg ? 0xdeadbeef : 0);
    }
    if (target_info.agg_kind == kAVG) {
      target_init_vals.push_back(0);
    } else if (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_geometry()) {
      for (int i = 1; i < 2 * target_info.sql_type.get_physical_coord_cols(); i++) {
        target_init_vals.push_back(0);
      }
    } else if (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()) {
      target_init_vals.push_back(0);
    }
  }
  return target_init_vals;
}

int64_t result_set::lazy_decode(const ColumnLazyFetchInfo& col_lazy_fetch,
                                const int8_t* byte_stream,
                                const int64_t pos) {
  CHECK(col_lazy_fetch.is_lazily_fetched);
  const auto& type_info = col_lazy_fetch.type;
  if (type_info.is_fp()) {
    if (type_info.get_type() == kFLOAT) {
      double fval = fixed_width_float_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<const int64_t*>(may_alias_ptr(&fval));
    } else {
      double fval = fixed_width_double_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<const int64_t*>(may_alias_ptr(&fval));
    }
  }
  CHECK(type_info.is_integer() || type_info.is_decimal() || type_info.is_time() ||
        type_info.is_timeinterval() || type_info.is_boolean() || type_info.is_string() ||
        type_info.is_array());
  size_t type_bitwidth = get_bit_width(type_info);
  if (type_info.get_compression() == kENCODING_FIXED) {
    type_bitwidth = type_info.get_comp_param();
  } else if (type_info.get_compression() == kENCODING_DICT) {
    type_bitwidth = 8 * type_info.get_size();
  }
  CHECK_EQ(size_t(0), type_bitwidth % 8);
  int64_t val;
  if (type_info.is_date_in_days()) {
    val = type_info.get_comp_param() == 16
              ? fixed_width_small_date_decode_noinline(
                    byte_stream, 2, NULL_SMALLINT, NULL_BIGINT, pos)
              : fixed_width_small_date_decode_noinline(
                    byte_stream, 4, NULL_INT, NULL_BIGINT, pos);
  } else {
    val = (type_info.get_compression() == kENCODING_DICT &&
           type_info.get_size() < type_info.get_logical_size() &&
           type_info.get_comp_param())
              ? fixed_width_unsigned_decode_noinline(byte_stream, type_bitwidth / 8, pos)
              : fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
  }
  if (type_info.get_compression() != kENCODING_NONE &&
      type_info.get_compression() != kENCODING_DATE_IN_DAYS) {
    CHECK(type_info.get_compression() == kENCODING_FIXED ||
          type_info.get_compression() == kENCODING_DICT);
    auto encoding = type_info.get_compression();
    if (encoding == kENCODING_FIXED) {
      encoding = kENCODING_NONE;
    }
    SQLTypeInfo col_logical_ti(type_info.get_type(),
                               type_info.get_dimension(),
                               type_info.get_scale(),
                               false,
                               encoding,
                               0,
                               type_info.get_subtype());
    if (val == inline_fixed_encoding_null_val(type_info)) {
      return inline_int_null_val(col_logical_ti);
    }
  }
  return val;
}

size_t ResultSetStorage::getColOffInBytes(size_t column_idx) const {
  return query_mem_desc_.getColOffInBytes(column_idx);
}
