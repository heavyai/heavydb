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

#include "ResultRows.h"

#include "AggregateUtils.h"
#include "Execute.h"
#include "InPlaceSort.h"
#include "OutputBufferInitialization.h"

#include "../DataMgr/BufferMgr/BufferMgr.h"
#include "../Shared/likely.h"

#include <future>

// The legacy way of representing result sets. Don't change it, it's going away.

ResultRows::ResultRows(std::shared_ptr<ResultSet> result_set)
    : result_set_(result_set), group_by_buffer_(nullptr), in_place_(false) {}

ResultRows::ResultRows(const QueryMemoryDescriptor& query_mem_desc,
                       const std::vector<Analyzer::Expr*>& targets,
                       const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                       const std::vector<int64_t>& init_vals,
                       int64_t* group_by_buffer,
                       const size_t groups_buffer_entry_count,
                       const bool output_columnar,
                       const std::vector<std::vector<const int8_t*>>& col_buffers,
                       const ExecutorDeviceType device_type,
                       const int device_id)
    : result_set_(nullptr),
      executor_(query_mem_desc.executor_),
      query_mem_desc_(query_mem_desc),
      row_set_mem_owner_(row_set_mem_owner),
      agg_init_vals_(init_vals),
      group_by_buffer_(nullptr),
      groups_buffer_entry_count_(groups_buffer_entry_count),
      group_by_buffer_idx_(0),
      min_val_(0),
      warp_count_(0),
      output_columnar_(output_columnar),
      in_place_(!query_mem_desc.is_sort_plan && query_mem_desc.usesCachedContext()),
      device_type_(device_type),
      device_id_(device_id),
      crt_row_idx_(0),
      crt_row_buff_idx_(0),
      drop_first_(0),
      keep_first_(0),
      fetch_started_(false),
      in_place_buff_idx_(0),
      just_explain_(false) {
  if (group_by_buffer) {
    in_place_group_by_buffers_.push_back(group_by_buffer);
    in_place_groups_by_buffers_entry_count_.push_back(groups_buffer_entry_count);
  }
  bool has_lazy_columns = false;
  for (const auto target_expr : targets) {
    const auto agg_info = target_info(target_expr);
    bool is_real_string = agg_info.sql_type.is_string() && agg_info.sql_type.get_compression() == kENCODING_NONE;
    bool is_array = agg_info.sql_type.is_array();
    CHECK(!is_real_string || !is_array);
    if (executor_->plan_state_->isLazyFetchColumn(target_expr) || is_real_string || is_array) {
      has_lazy_columns = true;
    }
    targets_.push_back(agg_info);
  }
  std::vector<TargetValue> row;
  if (in_place_ && has_lazy_columns) {
    while (fetchLazyOrBuildRow(row, col_buffers, targets, false, false, true)) {
    };
  }
  moveToBegin();
}

namespace {

const bool min_check_flag = false;
const bool max_check_flag = false;
const bool sum_check_flag = true;

}  // namespace

#define AGGREGATE_ONE_VALUE(agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__)                            \
  {                                                                                                                    \
    const auto sql_type = get_compact_type(agg_info__);                                                                \
    if (sql_type.is_fp()) {                                                                                            \
      if (chosen_bytes__ == sizeof(float)) {                                                                           \
        agg_##agg_kind__##_float(reinterpret_cast<int32_t*>(val_ptr__), *reinterpret_cast<const float*>(other_ptr__)); \
      } else {                                                                                                         \
        agg_##agg_kind__##_double(reinterpret_cast<int64_t*>(val_ptr__),                                               \
                                  *reinterpret_cast<const double*>(other_ptr__));                                      \
      }                                                                                                                \
    } else {                                                                                                           \
      if (chosen_bytes__ == sizeof(int32_t)) {                                                                         \
        auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                                                          \
        auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);                                                \
        if (agg_kind__##_check_flag &&                                                                                 \
            detect_overflow_and_underflow(*val_ptr, *other_ptr, false, int32_t(0), sql_type)) {                        \
          throw OverflowOrUnderflow();                                                                                 \
        }                                                                                                              \
        agg_##agg_kind__##_int32(val_ptr, *other_ptr);                                                                 \
      } else {                                                                                                         \
        auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                                                          \
        auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);                                                \
        if (agg_kind__##_check_flag &&                                                                                 \
            detect_overflow_and_underflow(*val_ptr, *other_ptr, false, int64_t(0), sql_type)) {                        \
          throw OverflowOrUnderflow();                                                                                 \
        }                                                                                                              \
        agg_##agg_kind__(val_ptr, *other_ptr);                                                                         \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define AGGREGATE_ONE_NULLABLE_VALUE(agg_kind__, val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__) \
  {                                                                                                              \
    if (agg_info__.skip_null_val) {                                                                              \
      const auto sql_type = get_compact_type(agg_info__);                                                        \
      if (sql_type.is_fp()) {                                                                                    \
        if (chosen_bytes__ == sizeof(float)) {                                                                   \
          agg_##agg_kind__##_float_skip_val(reinterpret_cast<int32_t*>(val_ptr__),                               \
                                            *reinterpret_cast<const float*>(other_ptr__),                        \
                                            *reinterpret_cast<const float*>(may_alias_ptr(&init_val__)));        \
        } else {                                                                                                 \
          agg_##agg_kind__##_double_skip_val(reinterpret_cast<int64_t*>(val_ptr__),                              \
                                             *reinterpret_cast<const double*>(other_ptr__),                      \
                                             *reinterpret_cast<const double*>(may_alias_ptr(&init_val__)));      \
        }                                                                                                        \
      } else {                                                                                                   \
        if (chosen_bytes__ == sizeof(int32_t)) {                                                                 \
          auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                                                  \
          auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);                                        \
          const auto null_val = static_cast<int32_t>(init_val__);                                                \
          if (agg_kind__##_check_flag &&                                                                         \
              detect_overflow_and_underflow(*val_ptr, *other_ptr, true, null_val, sql_type)) {                   \
            throw OverflowOrUnderflow();                                                                         \
          }                                                                                                      \
          agg_##agg_kind__##_int32_skip_val(val_ptr, *other_ptr, null_val);                                      \
        } else {                                                                                                 \
          auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                                                  \
          auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);                                        \
          const auto null_val = static_cast<int64_t>(init_val__);                                                \
          if (agg_kind__##_check_flag &&                                                                         \
              detect_overflow_and_underflow(*val_ptr, *other_ptr, true, null_val, sql_type)) {                   \
            throw OverflowOrUnderflow();                                                                         \
          }                                                                                                      \
          agg_##agg_kind__##_skip_val(val_ptr, *other_ptr, null_val);                                            \
        }                                                                                                        \
      }                                                                                                          \
    } else {                                                                                                     \
      AGGREGATE_ONE_VALUE(agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__);                       \
    }                                                                                                            \
  }

#define AGGREGATE_ONE_COUNT(val_ptr__, other_ptr__, chosen_bytes__, agg_info__)                                      \
  {                                                                                                                  \
    const auto sql_type = get_compact_type(agg_info__);                                                              \
    if (sql_type.is_fp()) {                                                                                          \
      if (chosen_bytes__ == sizeof(float)) {                                                                         \
        agg_sum_float(reinterpret_cast<int32_t*>(val_ptr__), *reinterpret_cast<const float*>(other_ptr__));          \
      } else {                                                                                                       \
        agg_sum_double(reinterpret_cast<int64_t*>(val_ptr__), *reinterpret_cast<const double*>(other_ptr__));        \
      }                                                                                                              \
    } else {                                                                                                         \
      if (chosen_bytes__ == sizeof(int32_t)) {                                                                       \
        auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                                                        \
        auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);                                              \
        if (detect_overflow_and_underflow(                                                                           \
                static_cast<uint32_t>(*val_ptr), static_cast<uint32_t>(*other_ptr), false, uint32_t(0), sql_type)) { \
          throw OverflowOrUnderflow();                                                                               \
        }                                                                                                            \
        agg_sum_int32(val_ptr, *other_ptr);                                                                          \
      } else {                                                                                                       \
        auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                                                        \
        auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);                                              \
        if (detect_overflow_and_underflow(                                                                           \
                static_cast<uint64_t>(*val_ptr), static_cast<uint64_t>(*other_ptr), false, uint64_t(0), sql_type)) { \
          throw OverflowOrUnderflow();                                                                               \
        }                                                                                                            \
        agg_sum(val_ptr, *other_ptr);                                                                                \
      }                                                                                                              \
    }                                                                                                                \
  }

#define AGGREGATE_ONE_NULLABLE_COUNT(val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__) \
  {                                                                                                  \
    if (agg_info__.skip_null_val) {                                                                  \
      const auto sql_type = get_compact_type(agg_info__);                                            \
      if (sql_type.is_fp()) {                                                                        \
        if (chosen_bytes__ == sizeof(float)) {                                                       \
          agg_sum_float_skip_val(reinterpret_cast<int32_t*>(val_ptr__),                              \
                                 *reinterpret_cast<const float*>(other_ptr__),                       \
                                 *reinterpret_cast<const float*>(may_alias_ptr(&init_val__)));       \
        } else {                                                                                     \
          agg_sum_double_skip_val(reinterpret_cast<int64_t*>(val_ptr__),                             \
                                  *reinterpret_cast<const double*>(other_ptr__),                     \
                                  *reinterpret_cast<const double*>(may_alias_ptr(&init_val__)));     \
        }                                                                                            \
      } else {                                                                                       \
        if (chosen_bytes__ == sizeof(int32_t)) {                                                     \
          auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                                      \
          auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);                            \
          const auto null_val = static_cast<int32_t>(init_val__);                                    \
          if (detect_overflow_and_underflow(static_cast<uint32_t>(*val_ptr),                         \
                                            static_cast<uint32_t>(*other_ptr),                       \
                                            true,                                                    \
                                            static_cast<uint32_t>(null_val),                         \
                                            sql_type)) {                                             \
            throw OverflowOrUnderflow();                                                             \
          }                                                                                          \
          agg_sum_int32_skip_val(val_ptr, *other_ptr, null_val);                                     \
        } else {                                                                                     \
          auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                                      \
          auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);                            \
          const auto null_val = static_cast<int64_t>(init_val__);                                    \
          if (detect_overflow_and_underflow(static_cast<uint64_t>(*val_ptr),                         \
                                            static_cast<uint64_t>(*other_ptr),                       \
                                            true,                                                    \
                                            static_cast<uint64_t>(null_val),                         \
                                            sql_type)) {                                             \
            throw OverflowOrUnderflow();                                                             \
          }                                                                                          \
          agg_sum_skip_val(val_ptr, *other_ptr, null_val);                                           \
        }                                                                                            \
      }                                                                                              \
    } else {                                                                                         \
      AGGREGATE_ONE_COUNT(val_ptr__, other_ptr__, chosen_bytes__, agg_info__);                       \
    }                                                                                                \
  }

bool ResultRows::reduceSingleRow(const int8_t* row_ptr,
                                 const int8_t warp_count,
                                 const bool is_columnar,
                                 const bool replace_bitmap_ptr_with_bitmap_sz,
                                 std::vector<int64_t>& agg_vals) const {
  CHECK(!result_set_);
  return reduceSingleRow(row_ptr,
                         warp_count,
                         is_columnar,
                         replace_bitmap_ptr_with_bitmap_sz,
                         agg_vals,
                         query_mem_desc_,
                         targets_,
                         agg_init_vals_);
}

bool ResultRows::reduceSingleRow(const int8_t* row_ptr,
                                 const int8_t warp_count,
                                 const bool is_columnar,
                                 const bool replace_bitmap_ptr_with_bitmap_sz,
                                 std::vector<int64_t>& agg_vals,
                                 const QueryMemoryDescriptor& query_mem_desc,
                                 const std::vector<TargetInfo>& targets,
                                 const std::vector<int64_t>& agg_init_vals) {
  const size_t agg_col_count{agg_vals.size()};
  const auto row_size = query_mem_desc.getRowSize();
  CHECK_EQ(agg_col_count, query_mem_desc.agg_col_widths.size());
  CHECK_GE(agg_col_count, targets.size());
  CHECK_EQ(is_columnar, query_mem_desc.output_columnar);
  CHECK(query_mem_desc.keyless_hash);
  std::vector<int64_t> partial_agg_vals(agg_col_count, 0);
  bool discard_row = true;
  for (int8_t warp_idx = 0; warp_idx < warp_count; ++warp_idx) {
    bool discard_partial_result = true;
    for (size_t target_idx = 0, agg_col_idx = 0; target_idx < targets.size() && agg_col_idx < agg_col_count;
         ++target_idx, ++agg_col_idx) {
      const auto& agg_info = targets[target_idx];
      const bool float_argument_input = takes_float_argument(agg_info);
      const auto chosen_bytes =
          float_argument_input ? sizeof(float) : query_mem_desc.agg_col_widths[agg_col_idx].compact;
      auto partial_bin_val = get_component(row_ptr + query_mem_desc.getColOnlyOffInBytes(agg_col_idx), chosen_bytes);
      partial_agg_vals[agg_col_idx] = partial_bin_val;
      if (is_distinct_target(agg_info)) {
        CHECK_EQ(int8_t(1), warp_count);
        CHECK(agg_info.is_agg && (agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT));
        partial_bin_val =
            count_distinct_set_size(partial_bin_val, target_idx, query_mem_desc.count_distinct_descriptors_);
        if (replace_bitmap_ptr_with_bitmap_sz) {
          partial_agg_vals[agg_col_idx] = partial_bin_val;
        }
      }
      if (kAVG == agg_info.agg_kind) {
        CHECK(agg_info.is_agg && !agg_info.is_distinct);
        ++agg_col_idx;
        partial_bin_val = partial_agg_vals[agg_col_idx] =
            get_component(row_ptr + query_mem_desc.getColOnlyOffInBytes(agg_col_idx),
                          query_mem_desc.agg_col_widths[agg_col_idx].compact);
      }
      if (agg_col_idx == static_cast<size_t>(query_mem_desc.idx_target_as_key) &&
          partial_bin_val != agg_init_vals[query_mem_desc.idx_target_as_key]) {
        CHECK(agg_info.is_agg);
        discard_partial_result = false;
      }
    }
    row_ptr += row_size;
    if (discard_partial_result) {
      continue;
    }
    discard_row = false;
    for (size_t target_idx = 0, agg_col_idx = 0; target_idx < targets.size() && agg_col_idx < agg_col_count;
         ++target_idx, ++agg_col_idx) {
      auto partial_bin_val = partial_agg_vals[agg_col_idx];
      const auto& agg_info = targets[target_idx];
      const bool float_argument_input = takes_float_argument(agg_info);
      const auto chosen_bytes =
          float_argument_input ? sizeof(float) : query_mem_desc.agg_col_widths[agg_col_idx].compact;
      const auto& chosen_type = get_compact_type(agg_info);
      if (agg_info.is_agg) {
        try {
          switch (agg_info.agg_kind) {
            case kCOUNT:
            case kAPPROX_COUNT_DISTINCT:
              AGGREGATE_ONE_NULLABLE_COUNT(reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                                           reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                                           agg_init_vals[agg_col_idx],
                                           chosen_bytes,
                                           agg_info);
              break;
            case kAVG:
              // Ignore float argument compaction for count component for fear of its overflow
              AGGREGATE_ONE_COUNT(reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx + 1]),
                                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx + 1]),
                                  query_mem_desc.agg_col_widths[agg_col_idx].compact,
                                  agg_info);
            // fall thru
            case kSUM:
              AGGREGATE_ONE_NULLABLE_VALUE(sum,
                                           reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                                           reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                                           agg_init_vals[agg_col_idx],
                                           chosen_bytes,
                                           agg_info);
              break;
            case kMIN:
              AGGREGATE_ONE_NULLABLE_VALUE(min,
                                           reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                                           reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                                           agg_init_vals[agg_col_idx],
                                           chosen_bytes,
                                           agg_info);
              break;
            case kMAX:
              AGGREGATE_ONE_NULLABLE_VALUE(max,
                                           reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                                           reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                                           agg_init_vals[agg_col_idx],
                                           chosen_bytes,
                                           agg_info);
              break;
            default:
              CHECK(false);
              break;
          }
        } catch (std::runtime_error& e) {
          // TODO(miyu): handle the case where chosen_bytes < 8
          LOG(ERROR) << e.what();
        }
        if (chosen_type.is_integer() || chosen_type.is_decimal()) {
          switch (chosen_bytes) {
            case 8:
              break;
            case 4: {
              int32_t ret = *reinterpret_cast<const int32_t*>(&agg_vals[agg_col_idx]);
              if (!(agg_info.agg_kind == kCOUNT && ret != agg_init_vals[agg_col_idx])) {
                agg_vals[agg_col_idx] = static_cast<int64_t>(ret);
              }
              break;
            }
            default:
              CHECK(false);
          }
        }
        if (kAVG == agg_info.agg_kind) {
          ++agg_col_idx;
        }
      } else {
        if (agg_vals[agg_col_idx]) {
          CHECK_EQ(agg_vals[agg_col_idx], partial_bin_val);
        } else {
          agg_vals[agg_col_idx] = partial_bin_val;
        }
      }
    }
  }
  return discard_row;
}

void ResultRows::addKeylessGroupByBuffer(const int64_t* group_by_buffer,
                                         const int32_t groups_buffer_entry_count,
                                         const int64_t min_val,
                                         const int8_t warp_count,
                                         const bool is_columnar) {
  CHECK(!result_set_);
  CHECK(!is_columnar || warp_count == 1);
  const size_t agg_col_count{query_mem_desc_.agg_col_widths.size()};
  std::vector<int64_t> agg_vals(agg_col_count, 0);
  simple_keys_.reserve(groups_buffer_entry_count);
  target_values_.reserve(groups_buffer_entry_count);
  for (int32_t bin = 0; bin < groups_buffer_entry_count; ++bin) {
    memcpy(&agg_vals[0], &agg_init_vals_[0], agg_col_count * sizeof(agg_vals[0]));
    beginRow(bin + min_val);
    if (reduceSingleRow(reinterpret_cast<const int8_t*>(group_by_buffer) + query_mem_desc_.getColOffInBytes(bin, 0),
                        warp_count,
                        is_columnar,
                        false,
                        agg_vals)) {
      discardRow();
      continue;
    }
    addValues(agg_vals);
  }
}

void ResultRows::reduceSingleColumn(int8_t* crt_val_i1,
                                    int8_t* crt_val_i2,
                                    const int8_t* new_val_i1,
                                    const int8_t* new_val_i2,
                                    const int64_t agg_skip_val,
                                    const size_t target_idx,
                                    size_t crt_byte_width,
                                    size_t next_byte_width) {
  const auto agg_info = targets_[target_idx];
  const auto& chosen_type = get_compact_type(agg_info);
  CHECK(chosen_type.is_integer() || chosen_type.is_decimal() || chosen_type.is_time() || chosen_type.is_boolean() ||
        chosen_type.is_string() || chosen_type.is_fp() || chosen_type.is_timeinterval());
  switch (agg_info.agg_kind) {
    case kCOUNT:
    case kAPPROX_COUNT_DISTINCT:
      if (is_distinct_target(agg_info)) {
        CHECK(agg_info.is_agg);
        CHECK(agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT);
        CHECK_EQ(crt_byte_width, sizeof(int64_t));
        auto crt_val_i1_ptr = reinterpret_cast<int64_t*>(crt_val_i1);
        CHECK_LT(target_idx, query_mem_desc_.count_distinct_descriptors_.size());
        const auto& count_distinct_desc = query_mem_desc_.count_distinct_descriptors_[target_idx];
        CHECK(count_distinct_desc.impl_type_ != CountDistinctImplType::Invalid);
        auto old_set_ptr = reinterpret_cast<const int64_t*>(crt_val_i1_ptr);
        auto new_set_ptr = reinterpret_cast<const int64_t*>(new_val_i1);
        CHECK(old_set_ptr && new_set_ptr);
        count_distinct_set_union(*new_set_ptr, *old_set_ptr, count_distinct_desc);
        break;
      }
      AGGREGATE_ONE_NULLABLE_COUNT(crt_val_i1, new_val_i1, agg_skip_val, crt_byte_width, agg_info);
      break;
    case kAVG:
      CHECK(crt_val_i2 && new_val_i2);
      AGGREGATE_ONE_COUNT(crt_val_i2, new_val_i2, next_byte_width, agg_info);
    // fall thru
    case kSUM:
      AGGREGATE_ONE_NULLABLE_VALUE(sum, crt_val_i1, new_val_i1, agg_skip_val, crt_byte_width, agg_info);
      break;
    case kMIN:
      if (!agg_info.is_agg) {  // projected groupby key
        if (crt_byte_width == sizeof(int32_t)) {
          auto new_key = *reinterpret_cast<const int32_t*>(new_val_i1);
          if (new_key) {
            *reinterpret_cast<int32_t*>(crt_val_i1) = new_key;
          }
        } else {
          auto new_key = *reinterpret_cast<const int64_t*>(new_val_i1);
          if (new_key) {
            *reinterpret_cast<int64_t*>(crt_val_i1) = new_key;
          }
        }
        break;
      }
      AGGREGATE_ONE_NULLABLE_VALUE(min, crt_val_i1, new_val_i1, agg_skip_val, crt_byte_width, agg_info);
      break;
    case kMAX:
      AGGREGATE_ONE_NULLABLE_VALUE(max, crt_val_i1, new_val_i1, agg_skip_val, crt_byte_width, agg_info);
      break;
    default:
      CHECK(false);
  }
}

void ResultRows::reduceInPlaceDispatch(int64_t** group_by_buffer_ptr,
                                       const int64_t* other_group_by_buffer,
                                       const int32_t groups_buffer_entry_count,
                                       const GroupByColRangeType hash_type,
                                       const QueryMemoryDescriptor& query_mem_desc_in,
                                       const size_t start,
                                       const size_t end) {
  const bool output_columnar{query_mem_desc_in.output_columnar};
  const bool isometric_layout{query_mem_desc_in.isCompactLayoutIsometric()};
  const size_t consist_col_width{query_mem_desc_in.getCompactByteWidth()};
  const size_t off_stride{isometric_layout && output_columnar ? consist_col_width
                                                              : query_mem_desc_in.getColOffInBytesInNextBin(0)};
  const size_t group_by_col_count{query_mem_desc_in.group_col_widths.size()};
  const size_t row_size_quad{output_columnar ? 0 : query_mem_desc_in.getRowSize() / sizeof(int64_t)};
  const size_t key_base_stride_quad{output_columnar ? 1 : row_size_quad};
  const int64_t min_val{query_mem_desc_in.min_val};

  for (size_t bin = start, bin_base_off = query_mem_desc_in.getColOffInBytes(start, 0); bin < end;
       ++bin, bin_base_off += off_stride) {
    const size_t other_key_off_quad = key_base_stride_quad * bin;
    const auto other_key_buff = &other_group_by_buffer[other_key_off_quad];
    const auto consist_col_offset =
        output_columnar ? consist_col_width * query_mem_desc_in.entry_count : consist_col_width;
    if (other_key_buff[0] == EMPTY_KEY_64) {
      continue;
    }
    int64_t* group_val_buff{nullptr};
    size_t target_bin{0};
    switch (hash_type) {
      case GroupByColRangeType::OneColKnownRange:
        if (output_columnar) {
          target_bin = static_cast<size_t>(get_columnar_group_bin_offset(
              *group_by_buffer_ptr, other_key_buff[0], min_val, query_mem_desc_in.bucket));
        } else {
          group_val_buff = get_group_value_fast(
              *group_by_buffer_ptr, other_key_buff[0], min_val, query_mem_desc_in.bucket, row_size_quad);
        }
        break;
      case GroupByColRangeType::MultiColPerfectHash:
        group_val_buff = get_matching_group_value_perfect_hash(
            *group_by_buffer_ptr, bin, other_key_buff, group_by_col_count, row_size_quad);
        break;
      case GroupByColRangeType::MultiCol:
        group_val_buff = get_group_value(*group_by_buffer_ptr,
                                         groups_buffer_entry_count,
                                         other_key_buff,
                                         group_by_col_count,
                                         query_mem_desc_.getEffectiveKeyWidth(),
                                         row_size_quad);
        break;
      default:
        CHECK(false);
    }
    if (!output_columnar && !group_val_buff) {
      throw ReductionRanOutOfSlots();
    }
    if (hash_type == GroupByColRangeType::OneColKnownRange &&
        other_group_by_buffer[other_key_off_quad] != (*group_by_buffer_ptr)[other_key_off_quad]) {
      CHECK(EMPTY_KEY_64 == other_group_by_buffer[other_key_off_quad] ||
            EMPTY_KEY_64 == (*group_by_buffer_ptr)[other_key_off_quad]);
    }
    size_t target_idx{0};
    size_t col_idx{0};
    size_t other_off{bin_base_off};
    for (const auto& agg_info : targets_) {
      const auto chosen_bytes = query_mem_desc_in.agg_col_widths[col_idx].compact;
      auto next_chosen_bytes = chosen_bytes;
      const auto other_ptr = reinterpret_cast<const int8_t*>(other_group_by_buffer) + other_off;
      const auto other_val = get_component(other_ptr, chosen_bytes);
      int8_t* col_ptr{nullptr};
      if (output_columnar) {
        col_ptr = reinterpret_cast<int8_t*>(*group_by_buffer_ptr) +
                  (isometric_layout ? query_mem_desc_in.getConsistColOffInBytes(target_bin, col_idx)
                                    : query_mem_desc_in.getColOffInBytes(target_bin, col_idx));
      } else {
        col_ptr = reinterpret_cast<int8_t*>(group_val_buff) + query_mem_desc_in.getColOnlyOffInBytes(col_idx);
      }

      int8_t* next_col_ptr{nullptr};
      const int8_t* other_next_ptr{nullptr};
      if (agg_info.is_agg && agg_info.agg_kind == kAVG) {
        if (output_columnar) {
          next_col_ptr = reinterpret_cast<int8_t*>(*group_by_buffer_ptr) +
                         (isometric_layout ? query_mem_desc_in.getConsistColOffInBytes(target_bin, col_idx + 1)
                                           : query_mem_desc_in.getColOffInBytes(target_bin, col_idx + 1));
        } else {
          next_col_ptr =
              reinterpret_cast<int8_t*>(group_val_buff) + query_mem_desc_in.getColOnlyOffInBytes(col_idx + 1);
        }
        next_chosen_bytes = query_mem_desc_in.agg_col_widths[col_idx + 1].compact;
        other_off +=
            isometric_layout ? consist_col_offset : query_mem_desc_in.getNextColOffInBytes(other_ptr, bin, col_idx + 1);
        other_next_ptr = reinterpret_cast<const int8_t*>(other_group_by_buffer) + other_off;
      }

      switch (chosen_bytes) {
        case 4: {
          if (agg_info.is_agg) {
            reduceSingleColumn(col_ptr,
                               next_col_ptr,
                               other_ptr,
                               other_next_ptr,
                               agg_init_vals_[col_idx],
                               target_idx,
                               chosen_bytes,
                               next_chosen_bytes);
          } else {
            *reinterpret_cast<int32_t*>(col_ptr) = static_cast<int32_t>(other_val);
          }
          break;
        }
        case 8: {
          if (agg_info.is_agg) {
            reduceSingleColumn(col_ptr,
                               next_col_ptr,
                               other_ptr,
                               other_next_ptr,
                               agg_init_vals_[col_idx],
                               target_idx,
                               chosen_bytes,
                               next_chosen_bytes);
          } else {
            *reinterpret_cast<int64_t*>(col_ptr) = other_val;
          }
          break;
        }
        default:
          CHECK(false);
      }
      other_off +=
          isometric_layout ? consist_col_offset : query_mem_desc_in.getNextColOffInBytes(other_ptr, bin, col_idx);
      col_idx += (agg_info.is_agg && agg_info.agg_kind == kAVG) ? 2 : 1;
      ++target_idx;
    }
  }
}

void ResultRows::reduceInPlace(int64_t** group_by_buffer_ptr,
                               const int64_t* other_group_by_buffer,
                               const int32_t groups_buffer_entry_count,
                               const int32_t other_groups_buffer_entry_count,
                               const GroupByColRangeType hash_type,
                               const QueryMemoryDescriptor& query_mem_desc_in) {
  const auto available_cpus = cpu_threads();
  CHECK_LT(0, available_cpus);
  const size_t stride = (other_groups_buffer_entry_count + (available_cpus - 1)) / available_cpus;
  const bool multithreaded = query_mem_desc_in.isCompactLayoutIsometric() &&
                             hash_type == GroupByColRangeType::OneColKnownRange && stride > 1000;
  CHECK_LE(other_groups_buffer_entry_count, groups_buffer_entry_count);
  if (multithreaded) {
    std::vector<std::future<void>> reducer_threads;
    for (size_t tidx = 0,
                start = 0,
                end = std::min(start + stride, static_cast<size_t>(other_groups_buffer_entry_count));
         tidx < static_cast<size_t>(available_cpus);
         ++tidx,
                start = std::min(start + stride, static_cast<size_t>(other_groups_buffer_entry_count)),
                end = std::min(end + stride, static_cast<size_t>(other_groups_buffer_entry_count))) {
      reducer_threads.push_back(std::async(std::launch::async,
                                           &ResultRows::reduceInPlaceDispatch,
                                           this,
                                           group_by_buffer_ptr,
                                           other_group_by_buffer,
                                           groups_buffer_entry_count,
                                           hash_type,
                                           query_mem_desc_in,
                                           start,
                                           end));
    }
    for (auto& child : reducer_threads) {
      child.wait();
    }
    for (auto& child : reducer_threads) {
      child.get();
    }
  } else {
    reduceInPlaceDispatch(group_by_buffer_ptr,
                          other_group_by_buffer,
                          groups_buffer_entry_count,
                          hash_type,
                          query_mem_desc_in,
                          0,
                          other_groups_buffer_entry_count);
  }
}

void ResultRows::reduceDispatch(int64_t* group_by_buffer,
                                const int64_t* other_group_by_buffer,
                                const QueryMemoryDescriptor& query_mem_desc_in,
                                const size_t start,
                                const size_t end) {
  if (start >= end) {
    return;
  }
  const bool output_columnar{query_mem_desc_in.output_columnar};
  const bool isometric_layout{query_mem_desc_in.isCompactLayoutIsometric()};
  const size_t consist_col_width{query_mem_desc_in.getCompactByteWidth()};
  const size_t consist_col_offset{output_columnar ? consist_col_width * query_mem_desc_in.entry_count
                                                  : consist_col_width};
  const size_t agg_col_count{query_mem_desc_in.agg_col_widths.size()};
  auto crt_results = reinterpret_cast<int8_t*>(group_by_buffer);
  auto new_results = reinterpret_cast<const int8_t*>(other_group_by_buffer);
  size_t row_size = output_columnar ? 1 : query_mem_desc_in.getRowSize();
  std::vector<size_t> col_offsets(agg_col_count);
  for (size_t agg_col_idx = 0; agg_col_idx < agg_col_count; ++agg_col_idx) {
    col_offsets[agg_col_idx] = query_mem_desc_in.getColOffInBytesInNextBin(agg_col_idx);
  }
  for (size_t target_index = 0, agg_col_idx = 0, col_base_off = query_mem_desc_in.getColOffInBytes(start, 0);
       target_index < targets_.size() && agg_col_idx < agg_col_count;
       ++target_index,
              col_base_off +=
              (isometric_layout ? consist_col_offset : query_mem_desc_in.getNextColOffInBytes(
                                                           &crt_results[col_base_off], 0, agg_col_idx - 1))) {
    const auto agg_info = targets_[target_index];
    auto chosen_bytes = query_mem_desc_in.agg_col_widths[agg_col_idx].compact;
    auto next_chosen_bytes = chosen_bytes;
    if (kAVG == agg_info.agg_kind) {
      next_chosen_bytes = query_mem_desc_in.agg_col_widths[agg_col_idx + 1].compact;
    }

    for (size_t bin = start, bin_base_off = col_base_off; bin < end; ++bin, bin_base_off += col_offsets[agg_col_idx]) {
      int8_t* crt_next_result_ptr{nullptr};
      const int8_t* new_next_result_ptr{nullptr};
      if (kAVG == agg_info.agg_kind) {
        auto additional_off =
            bin_base_off + (isometric_layout ? consist_col_offset : query_mem_desc_in.getNextColOffInBytes(
                                                                        &crt_results[bin_base_off], bin, agg_col_idx));
        crt_next_result_ptr = &crt_results[additional_off];
        new_next_result_ptr = &new_results[additional_off];
      }

      for (size_t warp_idx = 0, row_base_off = bin_base_off; warp_idx < static_cast<size_t>(warp_count_);
           ++warp_idx, row_base_off += row_size) {
        reduceSingleColumn(&crt_results[row_base_off],
                           crt_next_result_ptr,
                           &new_results[row_base_off],
                           new_next_result_ptr,
                           agg_init_vals_[agg_col_idx],
                           target_index,
                           chosen_bytes,
                           next_chosen_bytes);
        if (kAVG == agg_info.agg_kind) {
          crt_next_result_ptr += row_size;
          new_next_result_ptr += row_size;
        }
      }
    }
    if (kAVG == agg_info.agg_kind) {
      col_base_off += isometric_layout ? consist_col_offset : query_mem_desc_in.getNextColOffInBytes(
                                                                  &crt_results[col_base_off], 0, agg_col_idx);
      ++agg_col_idx;
    }
    ++agg_col_idx;
  }
}

void ResultRows::reduce(const ResultRows& other_results,
                        const QueryMemoryDescriptor& query_mem_desc,
                        const bool output_columnar) {
  CHECK(!result_set_);
  if (other_results.definitelyHasNoRows()) {
    return;
  }
  if (definitelyHasNoRows()) {
    *this = other_results;
    return;
  }

  const size_t consist_col_width{query_mem_desc.getCompactByteWidth()};
  CHECK_EQ(output_columnar_, query_mem_desc.output_columnar);

  if (group_by_buffer_ && !in_place_) {
    CHECK(!query_mem_desc.sortOnGpu());
    CHECK(other_results.group_by_buffer_);
    CHECK(query_mem_desc.keyless_hash);
    CHECK(warp_count_ == 1 || !output_columnar);
    CHECK_EQ(static_cast<size_t>(warp_count_), query_mem_desc.getWarpCount());
    const auto available_cpus = cpu_threads();
    CHECK_LT(0, available_cpus);
    const size_t stride = (groups_buffer_entry_count_ + (available_cpus - 1)) / available_cpus;
    const bool multithreaded = query_mem_desc.isCompactLayoutIsometric() && stride > 1000;
    if (multithreaded) {
      std::vector<std::future<void>> reducer_threads;
      for (size_t tidx = 0, start = 0, end = std::min(start + stride, groups_buffer_entry_count_);
           tidx < static_cast<size_t>(available_cpus);
           ++tidx,
                  start = std::min(start + stride, groups_buffer_entry_count_),
                  end = std::min(end + stride, groups_buffer_entry_count_)) {
        reducer_threads.push_back(std::async(std::launch::async,
                                             &ResultRows::reduceDispatch,
                                             this,
                                             group_by_buffer_,
                                             other_results.group_by_buffer_,
                                             query_mem_desc,
                                             start,
                                             end));
      }
      for (auto& child : reducer_threads) {
        child.wait();
      }
      for (auto& child : reducer_threads) {
        child.get();
      }
    } else {
      reduceDispatch(
          group_by_buffer_, other_results.group_by_buffer_, query_mem_desc, size_t(0), groups_buffer_entry_count_);
    }
    return;
  }

  if (simple_keys_.empty() && multi_keys_.empty() && !in_place_) {
    CHECK(!query_mem_desc.sortOnGpu());
    CHECK_EQ(size_t(1), rowCount());
    CHECK_EQ(size_t(1), other_results.rowCount());
    auto& crt_results = target_values_.front();
    const auto& new_results = other_results.target_values_.front();
    for (size_t agg_col_idx = 0; agg_col_idx < colCount(); ++agg_col_idx) {
      reduceSingleColumn(reinterpret_cast<int8_t*>(&crt_results[agg_col_idx].i1),
                         reinterpret_cast<int8_t*>(&crt_results[agg_col_idx].i2),
                         reinterpret_cast<const int8_t*>(&new_results[agg_col_idx].i1),
                         reinterpret_cast<const int8_t*>(&new_results[agg_col_idx].i2),
                         get_initial_val(targets_[agg_col_idx], consist_col_width),
                         agg_col_idx);
    }
    return;
  }

  if (!in_place_) {
    CHECK_NE(simple_keys_.empty(), multi_keys_.empty());
  }
  CHECK(!query_mem_desc.sortOnGpu() || multi_keys_.empty());

  if (in_place_) {
    CHECK(other_results.in_place_);
    CHECK(in_place_group_by_buffers_.size() == other_results.in_place_group_by_buffers_.size());
    CHECK(in_place_group_by_buffers_.size() == 1 || in_place_group_by_buffers_.size() == 2);

    CHECK(other_results.in_place_groups_by_buffers_entry_count_[0] <= in_place_groups_by_buffers_entry_count_[0]);
    auto group_by_buffer_ptr = &in_place_group_by_buffers_[0];
    auto other_group_by_buffer = other_results.in_place_group_by_buffers_[0];

    if (query_mem_desc.getSmallBufferSizeBytes()) {
      CHECK_EQ(size_t(2), in_place_groups_by_buffers_entry_count_.size());
      CHECK_EQ(size_t(2), in_place_group_by_buffers_.size());
      CHECK(!output_columnar_);
      reduceInPlace(group_by_buffer_ptr,
                    other_group_by_buffer,
                    in_place_groups_by_buffers_entry_count_[0],
                    other_results.in_place_groups_by_buffers_entry_count_[0],
                    GroupByColRangeType::OneColKnownRange,
                    query_mem_desc);
      group_by_buffer_ptr = &in_place_group_by_buffers_[1];
      other_group_by_buffer = other_results.in_place_group_by_buffers_[1];
      reduceInPlace(group_by_buffer_ptr,
                    other_group_by_buffer,
                    in_place_groups_by_buffers_entry_count_[1],
                    other_results.in_place_groups_by_buffers_entry_count_[1],
                    GroupByColRangeType::MultiCol,
                    query_mem_desc);
    } else {
      reduceInPlace(group_by_buffer_ptr,
                    other_group_by_buffer,
                    in_place_groups_by_buffers_entry_count_[0],
                    other_results.in_place_groups_by_buffers_entry_count_[0],
                    query_mem_desc.hash_type,
                    query_mem_desc);
    }
    return;
  }

  createReductionMap();
  other_results.createReductionMap();
  for (const auto& kv : other_results.as_map_) {
    auto it = as_map_.find(kv.first);
    if (it == as_map_.end()) {
      as_map_.insert(std::make_pair(kv.first, kv.second));
      continue;
    }
    auto& old_agg_results = it->second;
    CHECK_EQ(old_agg_results.size(), kv.second.size());
    const size_t agg_col_count = old_agg_results.size();
    for (size_t agg_col_idx = 0; agg_col_idx < agg_col_count; ++agg_col_idx) {
      reduceSingleColumn(reinterpret_cast<int8_t*>(&old_agg_results[agg_col_idx].i1),
                         reinterpret_cast<int8_t*>(&old_agg_results[agg_col_idx].i2),
                         reinterpret_cast<const int8_t*>(&kv.second[agg_col_idx].i1),
                         reinterpret_cast<const int8_t*>(&kv.second[agg_col_idx].i2),
                         get_initial_val(targets_[agg_col_idx], consist_col_width),
                         agg_col_idx);
    }
  }
  for (const auto& kv : other_results.as_unordered_map_) {
    auto it = as_unordered_map_.find(kv.first);
    if (it == as_unordered_map_.end()) {
      auto it_ok = as_unordered_map_.insert(std::make_pair(kv.first, kv.second));
      CHECK(it_ok.second);
      continue;
    }
    auto& old_agg_results = it->second;
    CHECK_EQ(old_agg_results.size(), kv.second.size());
    const size_t agg_col_count = old_agg_results.size();
    for (size_t agg_col_idx = 0; agg_col_idx < agg_col_count; ++agg_col_idx) {
      const auto agg_info = targets_[agg_col_idx];
      if (agg_info.is_agg) {
        reduceSingleColumn(reinterpret_cast<int8_t*>(&old_agg_results[agg_col_idx].i1),
                           reinterpret_cast<int8_t*>(&old_agg_results[agg_col_idx].i2),
                           reinterpret_cast<const int8_t*>(&kv.second[agg_col_idx].i1),
                           reinterpret_cast<const int8_t*>(&kv.second[agg_col_idx].i2),
                           get_initial_val(agg_info, consist_col_width),
                           agg_col_idx);
      } else {
        old_agg_results[agg_col_idx] = kv.second[agg_col_idx];
      }
    }
  }

  CHECK(simple_keys_.empty() != multi_keys_.empty());
  target_values_.clear();
  target_values_.reserve(std::max(as_map_.size(), as_unordered_map_.size()));
  if (simple_keys_.empty()) {
    multi_keys_.clear();
    multi_keys_.reserve(as_map_.size());
    for (const auto& kv : as_map_) {
      multi_keys_.push_back(kv.first);
      target_values_.push_back(kv.second);
    }
  } else {
    simple_keys_.clear();
    simple_keys_.reserve(as_unordered_map_.size());
    for (const auto& kv : as_unordered_map_) {
      simple_keys_.push_back(kv.first);
      target_values_.push_back(kv.second);
    }
  }
}

void ResultRows::sort(const std::list<Analyzer::OrderEntry>& order_entries,
                      const bool remove_duplicates,
                      const int64_t top_n) {
  if (result_set_) {
    result_set_->sort(order_entries, top_n);
    return;
  }
  if (definitelyHasNoRows()) {
    return;
  }
  if (query_mem_desc_.sortOnGpu()) {
    try {
      inplaceSortGpu(order_entries);
    } catch (const OutOfMemory&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      inplaceSortCpu(order_entries);
    } catch (const std::bad_alloc&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      inplaceSortCpu(order_entries);
    }
    return;
  }
  if (query_mem_desc_.keyless_hash) {
    addKeylessGroupByBuffer(group_by_buffer_, groups_buffer_entry_count_, min_val_, warp_count_, output_columnar_);
    in_place_ = false;
    group_by_buffer_ = nullptr;
  }
  CHECK(!in_place_);
  const bool use_heap{order_entries.size() == 1 && !remove_duplicates && top_n};
  auto compare = [this, &order_entries, use_heap](const InternalRow& lhs, const InternalRow& rhs) {
    // NB: The compare function must define a strict weak ordering, otherwise
    // std::sort will trigger a segmentation fault (or corrupt memory).
    for (const auto order_entry : order_entries) {
      CHECK_GE(order_entry.tle_no, 1);
      CHECK_LE(static_cast<size_t>(order_entry.tle_no), targets_.size());
      const auto& entry_ti = get_compact_type(targets_[order_entry.tle_no - 1]);
      const auto is_dict = entry_ti.is_string() && entry_ti.get_compression() == kENCODING_DICT;
      const auto& lhs_v = lhs[order_entry.tle_no - 1];
      const auto& rhs_v = rhs[order_entry.tle_no - 1];
      if (UNLIKELY(isNull(entry_ti, lhs_v) && isNull(entry_ti, rhs_v))) {
        return false;
      }
      if (UNLIKELY(isNull(entry_ti, lhs_v) && !isNull(entry_ti, rhs_v))) {
        return use_heap ? !order_entry.nulls_first : order_entry.nulls_first;
      }
      if (UNLIKELY(isNull(entry_ti, rhs_v) && !isNull(entry_ti, lhs_v))) {
        return use_heap ? order_entry.nulls_first : !order_entry.nulls_first;
      }
      const bool use_desc_cmp = use_heap ? !order_entry.is_desc : order_entry.is_desc;
      if (LIKELY(lhs_v.isInt())) {
        CHECK(rhs_v.isInt());
        if (UNLIKELY(is_dict)) {
          CHECK_EQ(4, entry_ti.get_logical_size());
          auto string_dict_proxy =
              executor_->getStringDictionaryProxy(entry_ti.get_comp_param(), row_set_mem_owner_, false);
          auto lhs_str = string_dict_proxy->getString(lhs_v.i1);
          auto rhs_str = string_dict_proxy->getString(rhs_v.i1);
          if (lhs_str == rhs_str) {
            continue;
          }
          return use_desc_cmp ? lhs_str > rhs_str : lhs_str < rhs_str;
        }
        if (UNLIKELY(is_distinct_target(targets_[order_entry.tle_no - 1]))) {
          const auto lhs_sz =
              count_distinct_set_size(lhs_v.i1, order_entry.tle_no - 1, query_mem_desc_.count_distinct_descriptors_);
          const auto rhs_sz =
              count_distinct_set_size(rhs_v.i1, order_entry.tle_no - 1, query_mem_desc_.count_distinct_descriptors_);
          if (lhs_sz == rhs_sz) {
            continue;
          }
          return use_desc_cmp ? lhs_sz > rhs_sz : lhs_sz < rhs_sz;
        }
        if (lhs_v.i1 == rhs_v.i1) {
          continue;
        }
        if (entry_ti.is_fp()) {
          const auto lhs_dval = *reinterpret_cast<const double*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval = *reinterpret_cast<const double*>(may_alias_ptr(&rhs_v.i1));
          return use_desc_cmp ? lhs_dval > rhs_dval : lhs_dval < rhs_dval;
        }
        return use_desc_cmp ? lhs_v.i1 > rhs_v.i1 : lhs_v.i1 < rhs_v.i1;
      } else {
        if (lhs_v.isPair()) {
          CHECK(rhs_v.isPair());
          const auto lhs = pair_to_double({lhs_v.i1, lhs_v.i2}, entry_ti, false);
          const auto rhs = pair_to_double({rhs_v.i1, rhs_v.i2}, entry_ti, false);
          if (lhs == rhs) {
            continue;
          }
          return use_desc_cmp ? lhs > rhs : lhs < rhs;
        } else {
          CHECK(lhs_v.isStr() && rhs_v.isStr());
          const auto lhs = lhs_v.strVal();
          const auto rhs = rhs_v.strVal();
          if (lhs == rhs) {
            continue;
          }
          return use_desc_cmp ? lhs > rhs : lhs < rhs;
        }
      }
    }
    return false;
  };
  if (g_enable_watchdog && target_values_.size() > 100000) {
    throw WatchdogException("Sorting the result would be too slow");
  }
  if (use_heap) {
    target_values_.top(top_n, compare);
    return;
  }
  target_values_.sort(compare);
  if (remove_duplicates) {
    target_values_.removeDuplicates();
  }
}

void ResultRows::inplaceSortGpu(const std::list<Analyzer::OrderEntry>& order_entries) {
  auto data_mgr = &executor_->catalog_->get_dataMgr();
  const int device_id{0};
  CHECK(in_place_);
  CHECK_EQ(size_t(1), in_place_group_by_buffers_.size());
  std::vector<int64_t*> group_by_buffers(executor_->blockSize());
  group_by_buffers[0] = in_place_group_by_buffers_.front();
  auto gpu_query_mem = create_dev_group_by_buffers(data_mgr,
                                                   group_by_buffers,
                                                   {},
                                                   query_mem_desc_,
                                                   executor_->blockSize(),
                                                   executor_->gridSize(),
                                                   device_id,
                                                   true,
                                                   true,
                                                   nullptr);
  inplaceSortGpuImpl(order_entries, query_mem_desc_, gpu_query_mem, data_mgr, device_id);
  copy_group_by_buffers_from_gpu(data_mgr,
                                 group_by_buffers,
                                 query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU),
                                 gpu_query_mem.group_by_buffers.second,
                                 query_mem_desc_,
                                 executor_->blockSize(),
                                 executor_->gridSize(),
                                 device_id,
                                 false);
}

void ResultRows::inplaceSortGpuImpl(const std::list<Analyzer::OrderEntry>& order_entries,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const GpuQueryMemory& gpu_query_mem,
                                    Data_Namespace::DataMgr* data_mgr,
                                    const int device_id) {
  ThrustAllocator alloc(data_mgr, device_id);
  CHECK_EQ(size_t(1), order_entries.size());
  const auto idx_buff =
      gpu_query_mem.group_by_buffers.second - align_to_int64(query_mem_desc.entry_count * sizeof(int32_t));
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc.getColOffInBytes(0, target_idx);
    const auto chosen_bytes = query_mem_desc.agg_col_widths[target_idx].compact;
    sort_groups_gpu(reinterpret_cast<int64_t*>(val_buff),
                    reinterpret_cast<int32_t*>(idx_buff),
                    query_mem_desc.entry_count,
                    order_entry.is_desc,
                    chosen_bytes,
                    alloc);
    if (!query_mem_desc.keyless_hash) {
      apply_permutation_gpu(reinterpret_cast<int64_t*>(gpu_query_mem.group_by_buffers.second),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc.entry_count,
                            sizeof(int64_t),
                            alloc);
    }
    for (size_t target_idx = 0; target_idx < query_mem_desc.agg_col_widths.size(); ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc.agg_col_widths[target_idx].compact;
      const auto val_buff = gpu_query_mem.group_by_buffers.second + query_mem_desc.getColOffInBytes(0, target_idx);
      apply_permutation_gpu(reinterpret_cast<int64_t*>(val_buff),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc.entry_count,
                            chosen_bytes,
                            alloc);
    }
  }
}

void ResultRows::inplaceSortCpu(const std::list<Analyzer::OrderEntry>& order_entries) {
  CHECK(in_place_);
  CHECK(!query_mem_desc_.keyless_hash);
  CHECK_EQ(size_t(1), in_place_group_by_buffers_.size());
  std::vector<int64_t> tmp_buff(query_mem_desc_.entry_count);
  std::vector<int32_t> idx_buff(query_mem_desc_.entry_count);
  CHECK_EQ(size_t(1), order_entries.size());
  auto buffer_ptr = reinterpret_cast<int8_t*>(in_place_group_by_buffers_.front());
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto sortkey_val_buff =
        reinterpret_cast<int64_t*>(buffer_ptr + query_mem_desc_.getColOffInBytes(0, target_idx));
    const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
    sort_groups_cpu(sortkey_val_buff, &idx_buff[0], query_mem_desc_.entry_count, order_entry.is_desc, chosen_bytes);
    apply_permutation_cpu(
        in_place_group_by_buffers_.front(), &idx_buff[0], query_mem_desc_.entry_count, &tmp_buff[0], sizeof(int64_t));
    for (size_t target_idx = 0; target_idx < query_mem_desc_.agg_col_widths.size(); ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
      const auto satellite_val_buff =
          reinterpret_cast<int64_t*>(buffer_ptr + query_mem_desc_.getColOffInBytes(0, target_idx));
      apply_permutation_cpu(satellite_val_buff, &idx_buff[0], query_mem_desc_.entry_count, &tmp_buff[0], chosen_bytes);
    }
  }
}

namespace {

TargetValue result_rows_get_impl(const InternalTargetValue& col_val,
                                 const size_t col_idx,
                                 const TargetInfo& agg_info,
                                 const SQLTypeInfo& target_type,
                                 const bool decimal_to_double,
                                 const bool translate_strings,
                                 const Executor* executor,
                                 const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                 const QueryMemoryDescriptor& query_mem_desc) {
  const auto& chosen_type = get_compact_type(agg_info);
  if (agg_info.agg_kind == kAVG) {
    CHECK(!chosen_type.is_string());
    CHECK(col_val.isPair());
    return pair_to_double({col_val.i1, col_val.i2}, chosen_type, false);
  }
  if (chosen_type.is_integer() || chosen_type.is_decimal() || chosen_type.is_boolean() || chosen_type.is_time() ||
      chosen_type.is_timeinterval()) {
    if (is_distinct_target(agg_info)) {
      return TargetValue(count_distinct_set_size(col_val.i1, col_idx, query_mem_desc.count_distinct_descriptors_));
    }
    CHECK(col_val.isInt());
    if (chosen_type.is_decimal() && decimal_to_double) {
      if (col_val.i1 == inline_int_null_val(SQLTypeInfo(decimal_to_int_type(chosen_type), false))) {
        return NULL_DOUBLE;
      }
      return static_cast<double>(col_val.i1) / exp_to_scale(chosen_type.get_scale());
    }
    if (inline_int_null_val(chosen_type) == col_val.i1) {
      return inline_int_null_val(target_type);
    }
    return col_val.i1;
  } else if (chosen_type.is_string()) {
    if (chosen_type.get_compression() == kENCODING_DICT) {
      const int dict_id = chosen_type.get_comp_param();
      const auto string_id = col_val.i1;
      if (!translate_strings) {
        return TargetValue(string_id);
      }
      return string_id == NULL_INT
                 ? TargetValue(nullptr)
                 : TargetValue(
                       executor->getStringDictionaryProxy(dict_id, row_set_mem_owner, false)->getString(string_id));
    } else {
      CHECK_EQ(kENCODING_NONE, chosen_type.get_compression());
      return col_val.isNull() ? TargetValue(nullptr) : TargetValue(col_val.strVal());
    }
  } else if (chosen_type.is_array()) {
    const auto& elem_type = chosen_type.get_elem_type();
    CHECK(col_val.ty == InternalTargetValue::ITVType::Arr || col_val.ty == InternalTargetValue::ITVType::Null);
    if (col_val.ty == InternalTargetValue::ITVType::Null) {
      return std::vector<ScalarTargetValue>{};
    }
    CHECK(col_val.i1);
    std::vector<ScalarTargetValue> tv_arr;
    if (elem_type.is_integer() || elem_type.is_boolean() || elem_type.is_fp()) {
      const auto& int_arr = *reinterpret_cast<std::vector<int64_t>*>(col_val.i1);
      tv_arr.reserve(int_arr.size());
      if (elem_type.is_integer() || elem_type.is_boolean()) {
        for (const auto x : int_arr) {
          tv_arr.emplace_back(x);
        }
      } else {
        for (const auto x : int_arr) {
          tv_arr.emplace_back(*reinterpret_cast<const double*>(may_alias_ptr(&x)));
        }
      }
    } else if (elem_type.is_string()) {
      CHECK_EQ(kENCODING_DICT, chosen_type.get_compression());
      const auto& string_ids = *reinterpret_cast<std::vector<int64_t>*>(col_val.i1);
      const int dict_id = chosen_type.get_comp_param();
      for (const auto string_id : string_ids) {
        tv_arr.emplace_back(
            string_id == NULL_INT
                ? NullableString(nullptr)
                : NullableString(
                      executor->getStringDictionaryProxy(dict_id, row_set_mem_owner, false)->getString(string_id)));
      }
    } else {
      CHECK(false);
    }
    return tv_arr;
  } else {
    CHECK(chosen_type.is_fp());
    if (chosen_type.get_type() == kFLOAT) {
      return ScalarTargetValue(static_cast<float>(*reinterpret_cast<const double*>(may_alias_ptr(&col_val.i1))));
    }
    return ScalarTargetValue(*reinterpret_cast<const double*>(may_alias_ptr(&col_val.i1)));
  }
  CHECK(false);
}

int64_t lazy_decode(const Analyzer::ColumnVar* col_var, const int8_t* byte_stream, const int64_t pos) {
  const auto enc_type = col_var->get_compression();
  const auto& type_info = col_var->get_type_info();
  if (type_info.is_fp()) {
    if (type_info.get_type() == kFLOAT) {
      float fval = fixed_width_float_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<const int32_t*>(may_alias_ptr(&fval));
    } else {
      double fval = fixed_width_double_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<const int64_t*>(may_alias_ptr(&fval));
    }
  }
  CHECK(type_info.is_integer() || type_info.is_decimal() || type_info.is_time() || type_info.is_boolean() ||
        (type_info.is_string() && enc_type == kENCODING_DICT));
  size_t type_bitwidth = get_bit_width(type_info);
  if (type_info.get_compression() == kENCODING_FIXED) {
    type_bitwidth = type_info.get_comp_param();
  } else if (type_info.get_compression() == kENCODING_DICT) {
    type_bitwidth = 8 * type_info.get_size();
  }
  CHECK_EQ(size_t(0), type_bitwidth % 8);
  auto val = type_info.get_compression() == kENCODING_DICT && type_info.get_size() < type_info.get_logical_size()
                 ? fixed_width_unsigned_decode_noinline(byte_stream, type_bitwidth / 8, pos)
                 : fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
  if (type_info.get_compression() != kENCODING_NONE) {
    CHECK(type_info.get_compression() == kENCODING_FIXED || type_info.get_compression() == kENCODING_DICT);
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

}  // namespace

TargetValue ResultRows::getRowAt(const size_t row_idx,
                                 const size_t col_idx,
                                 const bool translate_strings,
                                 const bool decimal_to_double /* = true */) const {
  if (!result_set_ && just_explain_) {
    return explanation_;
  }

  if (in_place_ || group_by_buffer_ || result_set_) {
    moveToBegin();
    for (size_t i = 0; i < row_idx; ++i) {
      auto crt_row = getNextRow(translate_strings, decimal_to_double);
      CHECK(!crt_row.empty());
    }
    auto crt_row = getNextRow(translate_strings, decimal_to_double);
    CHECK(!crt_row.empty());
    return crt_row[col_idx];
  }
  CHECK_LT(row_idx, target_values_.size());
  CHECK_LT(col_idx, targets_.size());
  CHECK_LT(col_idx, target_values_[row_idx].size());

  size_t agg_col_idx{0};
  for (size_t target_idx = 0; target_idx < col_idx && agg_col_idx < agg_init_vals_.size();
       ++target_idx, ++agg_col_idx) {
    if (kAVG == targets_[target_idx].agg_kind) {
      ++agg_col_idx;
    }
  }

  return result_rows_get_impl(target_values_[row_idx][col_idx],
                              col_idx,
                              targets_[col_idx],
                              targets_[col_idx].sql_type,
                              decimal_to_double,
                              translate_strings,
                              executor_,
                              row_set_mem_owner_,
                              query_mem_desc_);
}

std::vector<TargetValue> ResultRows::getNextRow(const bool translate_strings, const bool decimal_to_double) const {
  if (result_set_) {
    return result_set_->getNextRow(translate_strings, decimal_to_double);
  }
  if (just_explain_) {
    if (crt_row_buff_idx_) {
      return {};
    }
    crt_row_buff_idx_ = 1;
    return {explanation_};
  }
  if (in_place_ || group_by_buffer_) {
    if (!fetch_started_) {
      for (size_t i = 0; i < drop_first_; ++i) {
        std::vector<TargetValue> row;
        if (!fetchLazyOrBuildRow(row, {}, {}, translate_strings, decimal_to_double, false)) {
          return {};
        }
      }
      fetch_started_ = true;
    }
    if (keep_first_ && crt_row_idx_ >= drop_first_ + keep_first_) {
      return {};
    }
    std::vector<TargetValue> row;
    fetchLazyOrBuildRow(row, {}, {}, translate_strings, decimal_to_double, false);
    return row;
  } else {
    std::vector<TargetValue> result;
    if (crt_row_idx_ >= target_values_.size()) {
      return result;
    }
    const auto internal_row = target_values_[crt_row_idx_];
    for (size_t col_idx = 0; col_idx < internal_row.size(); ++col_idx) {
      result.push_back(getRowAt(crt_row_idx_, col_idx, translate_strings, decimal_to_double));
    }
    ++crt_row_idx_;
    return result;
  }
}

namespace {

template <class T>
int64_t arr_elem_bitcast(const T val) {
  return val;
}

template <>
int64_t arr_elem_bitcast(const float val) {
  const double dval{val};
  return *reinterpret_cast<const int64_t*>(may_alias_ptr(&dval));
}

template <>
int64_t arr_elem_bitcast(const double val) {
  return *reinterpret_cast<const int64_t*>(may_alias_ptr(&val));
}

template <class T>
std::vector<int64_t> arr_from_buffer(const int8_t* buff, const size_t buff_sz) {
  std::vector<int64_t> result;
  auto buff_elems = reinterpret_cast<const T*>(buff);
  CHECK_EQ(size_t(0), buff_sz % sizeof(T));
  const size_t num_elems = buff_sz / sizeof(T);
  for (size_t i = 0; i < num_elems; ++i) {
    result.push_back(arr_elem_bitcast(buff_elems[i]));
  }
  return result;
}

}  // namespace

bool ResultRows::fetchLazyOrBuildRow(std::vector<TargetValue>& row,
                                     const std::vector<std::vector<const int8_t*>>& col_buffers,
                                     const std::vector<Analyzer::Expr*>& targets,
                                     const bool translate_strings,
                                     const bool decimal_to_double,
                                     const bool fetch_lazy) const {
  if (group_by_buffer_) {
    for (size_t bin_base_off = group_by_buffer_idx_ < static_cast<size_t>(groups_buffer_entry_count_)
                                   ? query_mem_desc_.getColOffInBytes(group_by_buffer_idx_, 0)
                                   : -1;
         group_by_buffer_idx_ < static_cast<size_t>(groups_buffer_entry_count_);
         ++group_by_buffer_idx_, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
      const int8_t warp_count = query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1;
      CHECK(!output_columnar_ || warp_count == 1);
      const size_t agg_col_count{query_mem_desc_.agg_col_widths.size()};
      std::vector<int64_t> agg_vals(agg_col_count, 0);
      CHECK_EQ(agg_col_count, agg_init_vals_.size());
      memcpy(&agg_vals[0], &agg_init_vals_[0], agg_col_count * sizeof(agg_vals[0]));
      if (reduceSingleRow(reinterpret_cast<int8_t*>(group_by_buffer_) + bin_base_off,
                          warp_count,
                          output_columnar_,
                          true,
                          agg_vals)) {
        continue;
      }
      for (size_t target_idx = 0, agg_col_idx = 0; target_idx < targets_.size() && agg_col_idx < agg_col_count;
           ++target_idx, ++agg_col_idx) {
        const auto& agg_info = targets_[target_idx];
        if (is_distinct_target(agg_info)) {
          row.emplace_back(agg_vals[agg_col_idx]);
        } else {
          const auto chosen_bytes = query_mem_desc_.agg_col_widths[agg_col_idx].compact;
          const auto& chosen_type = get_compact_type(agg_info);
          if (chosen_type.is_fp() && chosen_bytes == sizeof(float)) {
            agg_vals[agg_col_idx] = float_to_double_bin(agg_vals[agg_col_idx], !chosen_type.get_notnull());
          }
          auto target_val =
              (kAVG == agg_info.agg_kind ? InternalTargetValue(agg_vals[agg_col_idx], agg_vals[agg_col_idx + 1])
                                         : InternalTargetValue(agg_vals[agg_col_idx]));
          row.push_back(result_rows_get_impl(target_val,
                                             target_idx,
                                             agg_info,
                                             agg_info.sql_type,
                                             decimal_to_double,
                                             translate_strings,
                                             executor_,
                                             row_set_mem_owner_,
                                             query_mem_desc_));
          if (kAVG == agg_info.agg_kind) {
            ++agg_col_idx;
          }
        }
      }
      ++crt_row_idx_;
      ++group_by_buffer_idx_;
      return true;
    }
    return false;
  }

  std::vector<bool> is_lazy_fetched(targets.size(), false);
  for (size_t col_idx = 0, col_count = targets.size(); col_idx < col_count; ++col_idx) {
    is_lazy_fetched[col_idx] = executor_->plan_state_->isLazyFetchColumn(targets[col_idx]);
  }

  while (in_place_buff_idx_ < in_place_group_by_buffers_.size()) {
    auto group_by_buffer = in_place_group_by_buffers_[in_place_buff_idx_];
    const auto group_entry_count = in_place_groups_by_buffers_entry_count_[in_place_buff_idx_];
    for (size_t bin_base_off = crt_row_buff_idx_ < static_cast<size_t>(group_entry_count)
                                   ? query_mem_desc_.getColOffInBytes(crt_row_buff_idx_, 0)
                                   : -1;
         crt_row_buff_idx_ < static_cast<size_t>(group_entry_count);
         ++crt_row_buff_idx_, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
      auto key_off = query_mem_desc_.getKeyOffInBytes(crt_row_buff_idx_) / sizeof(int64_t);
      if (group_by_buffer[key_off] == EMPTY_KEY_64) {
        continue;
      }
      size_t out_vec_idx = 0;
      auto col_ptr = reinterpret_cast<int8_t*>(group_by_buffer) + bin_base_off;
      for (size_t col_idx = 0; col_idx < colCount();
           ++col_idx, col_ptr += query_mem_desc_.getNextColOffInBytes(col_ptr, crt_row_buff_idx_, out_vec_idx++)) {
        const auto& agg_info = targets_[col_idx];
        const auto& chosen_type = get_compact_type(agg_info);
        auto chosen_bytes = query_mem_desc_.agg_col_widths[out_vec_idx].compact;
        size_t next_chosen_bytes = chosen_bytes;
        auto val1 = get_component(col_ptr, chosen_bytes);
        if (chosen_type.is_fp() && chosen_bytes == sizeof(float)) {
          val1 = float_to_double_bin(val1, !chosen_type.get_notnull());
        }
        bool is_real_string = chosen_type.is_string() && chosen_type.get_compression() == kENCODING_NONE;
        bool is_array = chosen_type.is_array();
        CHECK(!is_real_string || !is_array);
        int64_t val2{0};
        int8_t* next_col_ptr{nullptr};
        if (agg_info.agg_kind == kAVG || is_real_string || is_array) {
          next_col_ptr = col_ptr + query_mem_desc_.getNextColOffInBytes(col_ptr, crt_row_buff_idx_, out_vec_idx++);
          next_chosen_bytes = query_mem_desc_.agg_col_widths[out_vec_idx].compact;
          val2 = get_component(next_col_ptr, next_chosen_bytes);
        }
        if (fetch_lazy) {
          const auto target_expr = targets[col_idx];
          if (is_lazy_fetched[col_idx]) {
            const auto col_var = dynamic_cast<Analyzer::ColumnVar*>(target_expr);
            CHECK(col_var);
            auto col_id = executor_->getLocalColumnId(col_var, false);
            CHECK_EQ(size_t(1), col_buffers.size());
            auto& frag_col_buffers = col_buffers.front();
            bool is_end{false};
            if (is_real_string) {
              VarlenDatum vd;
              ChunkIter_get_nth(reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffers[col_id])),
                                val1,
                                false,
                                &vd,
                                &is_end);
              CHECK(!is_end);
              if (vd.is_null) {
                set_component(col_ptr, chosen_bytes, 0);
                set_component(next_col_ptr, next_chosen_bytes, 0);
              } else {
                CHECK(vd.pointer);
                CHECK_GT(vd.length, 0);
                std::string fetched_str(reinterpret_cast<char*>(vd.pointer), vd.length);
                set_component(
                    col_ptr, chosen_bytes, reinterpret_cast<int64_t>(row_set_mem_owner_->addString(fetched_str)));
                set_component(next_col_ptr, next_chosen_bytes, vd.length);
              }
            } else if (is_array) {
              ArrayDatum ad;
              ChunkIter_get_nth(
                  reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffers[col_id])), val1, &ad, &is_end);
              CHECK(!is_end);
              if (ad.is_null) {
                set_component(col_ptr, chosen_bytes, 0);
                set_component(next_col_ptr, next_chosen_bytes, 0);
              } else {
                CHECK(target_expr);
                const auto& target_ti = target_expr->get_type_info();
                CHECK(target_ti.is_array());
                const auto& elem_ti = target_ti.get_elem_type();
                std::vector<int64_t>* arr_owned{nullptr};
                switch (elem_ti.get_logical_size()) {
                  case 1:
                    arr_owned = row_set_mem_owner_->addArray(arr_from_buffer<int8_t>(ad.pointer, ad.length));
                    break;
                  case 2:
                    arr_owned = row_set_mem_owner_->addArray(arr_from_buffer<int16_t>(ad.pointer, ad.length));
                    break;
                  case 4: {
                    if (elem_ti.is_fp()) {
                      arr_owned = row_set_mem_owner_->addArray(arr_from_buffer<float>(ad.pointer, ad.length));
                    } else {
                      arr_owned = row_set_mem_owner_->addArray(arr_from_buffer<int32_t>(ad.pointer, ad.length));
                    }
                    break;
                  }
                  case 8:
                    if (elem_ti.is_fp()) {
                      arr_owned = row_set_mem_owner_->addArray(arr_from_buffer<double>(ad.pointer, ad.length));
                    } else {
                      arr_owned = row_set_mem_owner_->addArray(arr_from_buffer<int64_t>(ad.pointer, ad.length));
                    }
                    break;
                  default:
                    CHECK(false);
                }
                CHECK(arr_owned);
                set_component(col_ptr, chosen_bytes, reinterpret_cast<int64_t>(arr_owned));
              }
            } else {
              set_component(
                  col_ptr,
                  chosen_bytes,
                  lazy_decode(static_cast<Analyzer::ColumnVar*>(target_expr), frag_col_buffers[col_id], val1));
            }
          } else if (is_real_string || is_array) {
            CHECK(device_type_ == ExecutorDeviceType::CPU || device_type_ == ExecutorDeviceType::GPU);
            int elem_sz{1};
            bool is_fp{false};
            if (is_array) {
              CHECK(!is_real_string);
              const auto& target_ti = target_expr->get_type_info();
              CHECK(target_ti.is_array());
              const auto& elem_ti = target_ti.get_elem_type();
              elem_sz = elem_ti.get_logical_size();
              is_fp = elem_ti.is_fp();
            }
            val2 *= elem_sz;
            std::vector<int8_t> cpu_buffer(val2);
            if (val1) {
              CHECK_GT(val2, 0);
              if (device_type_ == ExecutorDeviceType::GPU) {
                auto& data_mgr = executor_->catalog_->get_dataMgr();
                copy_from_gpu(&data_mgr, &cpu_buffer[0], static_cast<CUdeviceptr>(val1), val2, device_id_);
                val1 = reinterpret_cast<int64_t>(&cpu_buffer[0]);
              }
              if (is_real_string) {
                set_component(col_ptr,
                              chosen_bytes,
                              reinterpret_cast<int64_t>(
                                  row_set_mem_owner_->addString(std::string(reinterpret_cast<char*>(val1), val2))));
              } else {
                CHECK(is_array);
                std::vector<int64_t>* arr_owned{nullptr};
                switch (elem_sz) {
                  case 1:
                    arr_owned =
                        row_set_mem_owner_->addArray(arr_from_buffer<int8_t>(reinterpret_cast<int8_t*>(val1), val2));
                    break;
                  case 2:
                    arr_owned =
                        row_set_mem_owner_->addArray(arr_from_buffer<int16_t>(reinterpret_cast<int8_t*>(val1), val2));
                    break;
                  case 4:
                    if (is_fp) {
                      arr_owned =
                          row_set_mem_owner_->addArray(arr_from_buffer<float>(reinterpret_cast<int8_t*>(val1), val2));
                    } else {
                      arr_owned =
                          row_set_mem_owner_->addArray(arr_from_buffer<int32_t>(reinterpret_cast<int8_t*>(val1), val2));
                    }
                    break;
                  case 8:
                    if (is_fp) {
                      arr_owned =
                          row_set_mem_owner_->addArray(arr_from_buffer<double>(reinterpret_cast<int8_t*>(val1), val2));
                    } else {
                      arr_owned =
                          row_set_mem_owner_->addArray(arr_from_buffer<int64_t>(reinterpret_cast<int8_t*>(val1), val2));
                    }
                    break;
                  default:
                    CHECK(false);
                }
                CHECK(arr_owned);
                set_component(col_ptr, chosen_bytes, reinterpret_cast<int64_t>(arr_owned));
              }
            }
          }
        } else {
          auto build_itv = [is_real_string, is_array, &agg_info](const int64_t val1, const int64_t val2) {
            if (agg_info.agg_kind == kAVG) {
              return InternalTargetValue(val1, val2);
            }
            if (is_real_string) {
              return val1 ? InternalTargetValue(reinterpret_cast<std::string*>(val1)) : InternalTargetValue();
            }
            if (is_array) {
              CHECK(val1);
              return InternalTargetValue(reinterpret_cast<std::vector<int64_t>*>(val1));
            }
            return InternalTargetValue(val1);
          };
          row.push_back(result_rows_get_impl(build_itv(val1, val2),
                                             col_idx,
                                             agg_info,
                                             agg_info.sql_type,
                                             decimal_to_double,
                                             translate_strings,
                                             executor_,
                                             row_set_mem_owner_,
                                             query_mem_desc_));
        }
        if (next_col_ptr) {
          col_ptr = next_col_ptr;
        }
      }
      ++crt_row_buff_idx_;
      ++crt_row_idx_;
      return true;
    }
    ++in_place_buff_idx_;
    crt_row_buff_idx_ = 0;
  }
  return false;
}

bool ResultRows::isNull(const SQLTypeInfo& ti, const InternalTargetValue& val) {
  if (val.isInt()) {
    if (!ti.is_fp()) {
      return val.i1 == inline_int_null_val(ti);
    }
    const auto null_val = inline_fp_null_val(ti);
    return val.i1 == *reinterpret_cast<const int64_t*>(may_alias_ptr(&null_val));
  }
  if (val.isPair()) {
    // TODO(alex): Review this logic, we're not supposed to hit val.i2 == 0 anymore.
    //             Also, if AVG gets compacted to float, this will become incorrect.
    if (!val.i2) {
      LOG(ERROR) << "The AVG pair looks incorrect";
    }
    return (val.i2 == 0) || pair_to_double({val.i1, val.i2}, ti, false) == NULL_DOUBLE;
  }
  if (val.isStr()) {
    return false;
  }
  CHECK(val.isNull());
  return true;
}

void ResultRows::fillOneRow(const std::vector<int64_t>& row) {
  if (result_set_) {
    result_set_->fillOneEntry(row);
    return;
  }
  beginRow();
  size_t slot_idx = 0;
  for (const auto target : targets_) {
    CHECK(target.is_agg);
    if (target.agg_kind == kAVG) {
      addValue(row[slot_idx], row[slot_idx + 1]);
      ++slot_idx;
    } else {
      addValue(row[slot_idx]);
    }
    ++slot_idx;
  }
}

const std::vector<const int8_t*>& QueryExecutionContext::getColumnFrag(const size_t table_idx,
                                                                       int64_t& global_idx) const {
#ifdef ENABLE_MULTIFRAG_JOIN
  if (col_buffers_.size() > 1) {
    int64_t frag_id = 0;
    int64_t local_idx = global_idx;
    if (consistent_frag_sizes_[table_idx] != -1) {
      frag_id = global_idx / consistent_frag_sizes_[table_idx];
      local_idx = global_idx % consistent_frag_sizes_[table_idx];
    } else {
      std::tie(frag_id, local_idx) = get_frag_id_and_local_idx(frag_offsets_, table_idx, global_idx);
    }
    CHECK_GE(frag_id, int64_t(0));
    CHECK_LT(frag_id, col_buffers_.size());
    global_idx = local_idx;
    return col_buffers_[frag_id];
  } else
#endif
  {
    CHECK_EQ(size_t(1), col_buffers_.size());
    return col_buffers_.front();
  }
}

void QueryExecutionContext::outputBin(ResultRows& results,
                                      const std::vector<Analyzer::Expr*>& targets,
                                      int64_t* group_by_buffer,
                                      const size_t bin) const {
  if (isEmptyBin(group_by_buffer, bin, 0)) {
    return;
  }

  const size_t key_width{query_mem_desc_.getEffectiveKeyWidth()};
  const size_t group_by_col_count{query_mem_desc_.group_col_widths.size()};
  size_t out_vec_idx = 0;
  int8_t* buffer_ptr = reinterpret_cast<int8_t*>(group_by_buffer) + query_mem_desc_.getKeyOffInBytes(bin);

  if (group_by_col_count > 1) {
    std::vector<int64_t> multi_key;
    CHECK(!output_columnar_);
    for (size_t key_idx = 0; key_idx < group_by_col_count; ++key_idx) {
      const auto key_comp = get_component(buffer_ptr, key_width);
      multi_key.push_back(key_comp);
      buffer_ptr += query_mem_desc_.getNextKeyOffInBytes(key_idx);
    }
    results.beginRow(multi_key);
  } else {
    const auto key_comp = get_component(buffer_ptr, key_width);
    results.beginRow(key_comp);
    buffer_ptr += query_mem_desc_.getNextKeyOffInBytes(0);
  }
  buffer_ptr = align_to_int64(buffer_ptr);
  for (const auto target_expr : targets) {
    bool is_real_string = (target_expr && target_expr->get_type_info().is_string() &&
                           target_expr->get_type_info().get_compression() == kENCODING_NONE);
    bool is_array = target_expr && target_expr->get_type_info().is_array();
    const bool is_lazy_fetched = query_mem_desc_.executor_->plan_state_->isLazyFetchColumn(target_expr);
    CHECK(!is_real_string || !is_array);
    const auto col_var = dynamic_cast<Analyzer::ColumnVar*>(target_expr);
    const int global_col_id{col_var ? col_var->get_column_id() : -1};
    if (is_real_string || is_array) {
      CHECK(!output_columnar_);
      int64_t str_ptr = get_component(buffer_ptr, query_mem_desc_.agg_col_widths[out_vec_idx].compact);
      buffer_ptr += query_mem_desc_.getNextColOffInBytes(buffer_ptr, bin, out_vec_idx);
      int64_t str_len = get_component(buffer_ptr, query_mem_desc_.agg_col_widths[out_vec_idx + 1].compact);
      buffer_ptr += query_mem_desc_.getNextColOffInBytes(buffer_ptr, bin, out_vec_idx + 1);
      if (is_lazy_fetched) {  // TODO(alex): expensive!!!, remove
        CHECK_GE(str_len, 0);
        CHECK_EQ(str_ptr, str_len);  // both are the row index in this cases
        bool is_end;
        CHECK_GE(global_col_id, 0);
        CHECK(col_var);
        auto col_id = query_mem_desc_.executor_->getLocalColumnId(col_var, false);
        auto& frag_col_buffers = getColumnFrag(col_var->get_rte_idx(), str_ptr);
        str_len = str_ptr;
        if (is_real_string) {
          VarlenDatum vd;
          ChunkIter_get_nth(reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffers[col_id])),
                            str_ptr,
                            false,
                            &vd,
                            &is_end);
          CHECK(!is_end);
          if (!vd.is_null) {
            results.addValue(std::string(reinterpret_cast<char*>(vd.pointer), vd.length));
          } else {
            results.addValue();
          }
        } else {
          ArrayDatum ad;
          ChunkIter_get_nth(
              reinterpret_cast<ChunkIter*>(const_cast<int8_t*>(frag_col_buffers[col_id])), str_ptr, &ad, &is_end);
          CHECK(!is_end);
          if (!ad.is_null) {
            CHECK(target_expr);
            const auto& target_ti = target_expr->get_type_info();
            CHECK(target_ti.is_array());
            const auto& elem_ti = target_ti.get_elem_type();
            switch (elem_ti.get_logical_size()) {
              case 1:
                results.addValue(arr_from_buffer<int8_t>(ad.pointer, ad.length));
                break;
              case 2:
                results.addValue(arr_from_buffer<int16_t>(ad.pointer, ad.length));
                break;
              case 4: {
                if (elem_ti.is_fp()) {
                  results.addValue(arr_from_buffer<float>(ad.pointer, ad.length));
                } else {
                  results.addValue(arr_from_buffer<int32_t>(ad.pointer, ad.length));
                }
                break;
              }
              case 8:
                if (elem_ti.is_fp()) {
                  results.addValue(arr_from_buffer<double>(ad.pointer, ad.length));
                } else {
                  results.addValue(arr_from_buffer<int64_t>(ad.pointer, ad.length));
                }
                break;
              default:
                CHECK(false);
            }
          } else {
            results.addValue();
          }
        }
      } else {
        CHECK_GE(str_len, 0);
        int elem_sz{1};
        bool is_fp{false};
        if (is_array) {
          CHECK(!is_real_string);
          const auto& target_ti = target_expr->get_type_info();
          CHECK(target_ti.is_array());
          const auto& elem_ti = target_ti.get_elem_type();
          elem_sz = elem_ti.get_logical_size();
          is_fp = elem_ti.is_fp();
        }
        str_len *= elem_sz;
        CHECK(device_type_ == ExecutorDeviceType::CPU || device_type_ == ExecutorDeviceType::GPU);
        std::vector<int8_t> cpu_buffer;
        if (str_ptr && device_type_ == ExecutorDeviceType::GPU) {
          cpu_buffer.resize(str_len);
          auto& data_mgr = query_mem_desc_.executor_->catalog_->get_dataMgr();
          copy_from_gpu(&data_mgr, &cpu_buffer[0], static_cast<CUdeviceptr>(str_ptr), str_len, device_id_);
          str_ptr = reinterpret_cast<int64_t>(&cpu_buffer[0]);
        }
        if (str_ptr) {
          if (is_real_string) {
            results.addValue(std::string(reinterpret_cast<char*>(str_ptr), str_len));
          } else {
            CHECK(is_array);
            switch (elem_sz) {
              case 1:
                results.addValue(arr_from_buffer<int8_t>(reinterpret_cast<int8_t*>(str_ptr), str_len));
                break;
              case 2:
                results.addValue(arr_from_buffer<int16_t>(reinterpret_cast<int8_t*>(str_ptr), str_len));
                break;
              case 4:
                if (is_fp) {
                  results.addValue(arr_from_buffer<float>(reinterpret_cast<int8_t*>(str_ptr), str_len));
                } else {
                  results.addValue(arr_from_buffer<int32_t>(reinterpret_cast<int8_t*>(str_ptr), str_len));
                }
                break;
              case 8:
                if (is_fp) {
                  results.addValue(arr_from_buffer<double>(reinterpret_cast<int8_t*>(str_ptr), str_len));
                } else {
                  results.addValue(arr_from_buffer<int64_t>(reinterpret_cast<int8_t*>(str_ptr), str_len));
                }
                break;
              default:
                CHECK(false);
            }
          }
        } else {
          results.addValue();
        }
      }
      out_vec_idx += 2;
    } else {
      const auto chosen_byte_width = query_mem_desc_.agg_col_widths[out_vec_idx].compact;
      auto val1 = get_component(buffer_ptr, chosen_byte_width);

      if (is_lazy_fetched) {
        CHECK_GE(global_col_id, 0);
        CHECK(col_var);
        auto col_id = query_mem_desc_.executor_->getLocalColumnId(col_var, false);
        auto& frag_col_buffers = getColumnFrag(col_var->get_rte_idx(), val1);
        val1 = lazy_decode(static_cast<Analyzer::ColumnVar*>(target_expr), frag_col_buffers[col_id], val1);
      }
      const auto agg_info = target_info(target_expr);
      if (get_compact_type(agg_info).get_type() == kFLOAT && (is_lazy_fetched || chosen_byte_width == sizeof(float))) {
        val1 = float_to_double_bin(val1);
      }
      if (agg_info.agg_kind == kAVG) {
        CHECK(!is_lazy_fetched);
        buffer_ptr += query_mem_desc_.getNextColOffInBytes(buffer_ptr, bin, out_vec_idx++);
        auto val2 = get_component(buffer_ptr, query_mem_desc_.agg_col_widths[out_vec_idx].compact);
        results.addValue(val1, val2);
      } else {
        results.addValue(val1);
      }
      buffer_ptr += query_mem_desc_.getNextColOffInBytes(buffer_ptr, bin, out_vec_idx++);
    }
  }
}

RowSetPtr QueryExecutionContext::groupBufferToResults(const size_t i,
                                                      const std::vector<Analyzer::Expr*>& targets,
                                                      const bool was_auto_device) const {
  if (can_use_result_set(query_mem_desc_, device_type_)) {
    if (query_mem_desc_.interleavedBins(device_type_)) {
      return groupBufferToDeinterleavedResults(i);
    }
    CHECK_LT(i, result_sets_.size());
    return boost::make_unique<ResultRows>(std::shared_ptr<ResultSet>(result_sets_[i].release()));
  }
  const size_t group_by_col_count{query_mem_desc_.group_col_widths.size()};
  CHECK(!output_columnar_ || group_by_col_count == 1);
  auto impl = [group_by_col_count, was_auto_device, this, &targets](const size_t groups_buffer_entry_count,
                                                                    int64_t* group_by_buffer) {
    if (query_mem_desc_.keyless_hash) {
      CHECK(!sort_on_gpu_);
      CHECK_EQ(size_t(1), group_by_col_count);
      const int8_t warp_count = query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1;
      if (!query_mem_desc_.interleavedBins(ExecutorDeviceType::GPU) || !was_auto_device) {
        return boost::make_unique<ResultRows>(query_mem_desc_,
                                              targets,
                                              executor_,
                                              row_set_mem_owner_,
                                              init_agg_vals_,
                                              device_type_,
                                              group_by_buffer,
                                              groups_buffer_entry_count,
                                              query_mem_desc_.min_val,
                                              warp_count);
      }
      // Can't do the fast reduction in auto mode for interleaved bins, warp count isn't the same
      RowSetPtr results = boost::make_unique<ResultRows>(
          QueryMemoryDescriptor{}, targets, executor_, row_set_mem_owner_, init_agg_vals_, ExecutorDeviceType::CPU);
      CHECK(results);
      results->addKeylessGroupByBuffer(
          group_by_buffer, groups_buffer_entry_count, query_mem_desc_.min_val, warp_count, output_columnar_);
      return results;
    }
    RowSetPtr results = boost::make_unique<ResultRows>(query_mem_desc_,
                                                       targets,
                                                       row_set_mem_owner_,
                                                       init_agg_vals_,
                                                       group_by_buffer,
                                                       groups_buffer_entry_count,
                                                       output_columnar_,
                                                       col_buffers_,
                                                       device_type_,
                                                       device_id_);
    CHECK(results);
    if (results->isInPlace()) {
      return results;
    }
    for (size_t bin = 0; bin < groups_buffer_entry_count; ++bin) {
      outputBin(*results, targets, group_by_buffer, bin);
    }
    return results;
  };
  RowSetPtr results{nullptr};
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
    CHECK(!sort_on_gpu_);
    results = impl(query_mem_desc_.entry_count_small, small_group_by_buffers_[i]);
    CHECK(results);
  }
  CHECK_LT(i, group_by_buffers_.size());
  auto more_results = impl(query_mem_desc_.entry_count, group_by_buffers_[i]);
  if (query_mem_desc_.keyless_hash) {
    CHECK(!sort_on_gpu_);
    return more_results;
  }
  if (results) {
    results->append(*more_results);
    return results;
  } else {
    return more_results;
  }
}

RowSetPtr QueryExecutionContext::groupBufferToDeinterleavedResults(const size_t i) const {
  CHECK(!output_columnar_);
  const auto& result_set = result_sets_[i];
  auto deinterleaved_query_mem_desc = ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_);
  deinterleaved_query_mem_desc.interleaved_bins_on_gpu = false;
  for (auto& col_widths : deinterleaved_query_mem_desc.agg_col_widths) {
    col_widths.actual = col_widths.compact = 8;
  }
  auto deinterleaved_result_set = std::make_shared<ResultSet>(result_set->getTargetInfos(),
                                                              std::vector<ColumnLazyFetchInfo>{},
                                                              std::vector<std::vector<const int8_t*>>{},
#ifdef ENABLE_MULTIFRAG_JOIN
                                                              std::vector<std::vector<int64_t>>{},
                                                              std::vector<int64_t>{},
#endif
                                                              ExecutorDeviceType::CPU,
                                                              -1,
                                                              deinterleaved_query_mem_desc,
                                                              row_set_mem_owner_,
                                                              executor_);
  auto deinterleaved_storage = deinterleaved_result_set->allocateStorage(executor_->plan_state_->init_agg_vals_);
  auto deinterleaved_buffer = reinterpret_cast<int64_t*>(deinterleaved_storage->getUnderlyingBuffer());
  const auto rows_ptr = result_set->getStorage()->getUnderlyingBuffer();
  size_t deinterleaved_buffer_idx = 0;
  const size_t agg_col_count{query_mem_desc_.agg_col_widths.size()};
  for (size_t bin_base_off = query_mem_desc_.getColOffInBytes(0, 0), bin_idx = 0; bin_idx < result_set->entryCount();
       ++bin_idx, bin_base_off += query_mem_desc_.getColOffInBytesInNextBin(0)) {
    std::vector<int64_t> agg_vals(agg_col_count, 0);
    memcpy(&agg_vals[0], &executor_->plan_state_->init_agg_vals_[0], agg_col_count * sizeof(agg_vals[0]));
    ResultRows::reduceSingleRow(rows_ptr + bin_base_off,
                                executor_->warpSize(),
                                false,
                                true,
                                agg_vals,
                                query_mem_desc_,
                                result_set->getTargetInfos(),
                                executor_->plan_state_->init_agg_vals_);
    for (size_t agg_idx = 0; agg_idx < agg_col_count; ++agg_idx, ++deinterleaved_buffer_idx) {
      deinterleaved_buffer[deinterleaved_buffer_idx] = agg_vals[agg_idx];
    }
  }
  result_sets_[i].reset();
  return boost::make_unique<ResultRows>(deinterleaved_result_set);
}
